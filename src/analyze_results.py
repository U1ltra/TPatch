"""
Visualize patch attack evaluation results from demo.py eval().

Usage:
    # single entry (interactive)
    python analyze_results.py demo/20260220-131307_results.pkl --idx 0

    # single entry saved to file
    python analyze_results.py demo/20260220-131307_results.pkl --idx 5 --out entry5.png

    # all entries saved to demo/20260220-131307_viz/
    python analyze_results.py demo/20260220-131307_results.pkl --all

    # print summary statistics only
    python analyze_results.py demo/20260220-131307_results.pkl --summary
"""

import _init_path
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# 80-class COCO names in YOLO order (matches train_eval.py LabelConverter.category80)
COCO80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

NMS_CONF_THRESH = 0.25  # threshold used during NMS (for reference line in histogram)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cls_name(cls_id):
    idx = int(cls_id)
    return COCO80[idx] if idx < len(COCO80) else f"cls{idx}"


def xywh_to_xyxy(boxes):
    """[M,4] center xywh (pixel coords) → corner xyxy."""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def centers_inside(cx, cy, xyxy):
    """Boolean mask: which (cx,cy) centers fall inside xyxy box."""
    x1, y1, x2, y2 = xyxy
    return (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)


def iou_with_gt(raw_boxes_xywh, gt_xyxy):
    """
    Compute IoU between each anchor [M,4 xywh] and a single GT box [4 xyxy].
    Returns (mask, iou_vals) where mask is a boolean array for IoU > 0.
    """
    anchors = xywh_to_xyxy(raw_boxes_xywh)           # [M,4] xyxy
    ix1 = np.maximum(anchors[:, 0], gt_xyxy[0])
    iy1 = np.maximum(anchors[:, 1], gt_xyxy[1])
    ix2 = np.minimum(anchors[:, 2], gt_xyxy[2])
    iy2 = np.minimum(anchors[:, 3], gt_xyxy[3])
    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter   = inter_w * inter_h
    area_a  = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_g  = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])
    union   = area_a + area_g - inter
    iou     = np.where(union > 0, inter / union, 0.0)
    return iou > 0, iou


def draw_box(ax, xyxy, label=None, color='red', linestyle='-', lw=2, fontsize=9):
    x1, y1, x2, y2 = xyxy
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor='none', linestyle=linestyle,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1 + 2, y1 - 4, label, color='white', fontsize=fontsize,
                bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor='none'))


# ---------------------------------------------------------------------------
# Core visualization
# ---------------------------------------------------------------------------

def visualize_entry(entry, out_path=None):
    """
    3-panel figure for one result entry:
      Panel 1 — Post-NMS final detections on the patched image.
      Panel 2 — Pre-NMS hypothesis map: anchor centers colored by
                 obj × target-class probability.
      Panel 3 — Score distribution: in-box vs out-of-box anchors,
                 with NMS threshold shown for reference.
    """
    # --- unpack -----------------------------------------------------------
    target_cls  = int(entry['gt_box'][0, 4])
    gt_xyxy     = entry['gt_box'][0, :4]          # [x1,y1,x2,y2]
    nms_dets    = entry['nms_detections']          # [K,6]
    raw_boxes   = entry['raw_boxes']               # [M,4] xywh
    raw_obj     = entry['raw_obj']                 # [M]
    raw_cls     = entry['raw_cls']                 # [M,80]

    raw_xyxy    = xywh_to_xyxy(raw_boxes)
    raw_cx      = raw_boxes[:, 0]
    raw_cy      = raw_boxes[:, 1]
    raw_tcls    = raw_cls[:, target_cls]           # [M] target class prob
    raw_score   = raw_obj * raw_tcls               # combined score

    in_gt       = centers_inside(raw_cx, raw_cy, gt_xyxy)
    n_in        = int(in_gt.sum())
    n_out       = int((~in_gt).sum())

    # background image: use saved img_np if present, else gray canvas
    if 'img_np' in entry:
        bg = entry['img_np']                       # HWC uint8
    else:
        bg = np.full((640, 640, 3), 180, dtype=np.uint8)
    img_h, img_w = bg.shape[:2]

    t_name      = cls_name(target_cls)
    s_str       = "HIDDEN ✓" if entry['attack_success'] else "DETECTED ✗"
    s_color     = "green" if entry['attack_success'] else "red"

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    fig.suptitle(
        f"Image #{entry['image_idx']}  |  "
        f"resize={entry['set_resize']:.2f}  rotate={entry['set_rotate']:.1f}°  |  "
        f"[{s_str}]",
        fontsize=12, fontweight='bold', color=s_color,
    )

    # ------------------------------------------------------------------
    # Panel 1 — Post-NMS final detections
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.imshow(bg)

    # Ground-truth location (where the patch was composited)
    draw_box(ax, gt_xyxy, label=f'GT: {t_name}',
             color='lime', linestyle='--', lw=2, fontsize=8)

    # NMS detections
    if nms_dets.shape[0] > 0:
        for det in nms_dets:
            x1, y1, x2, y2, conf, cid = det
            is_target = int(cid) == target_cls
            color = 'red' if is_target else 'orange'
            draw_box(ax, [x1, y1, x2, y2],
                     label=f"{cls_name(cid)} {conf:.2f}",
                     color=color, lw=1.5, fontsize=8)
    else:
        ax.text(0.5, 0.03, 'No detections', transform=ax.transAxes,
                ha='center', fontsize=10, color='white',
                bbox=dict(facecolor='firebrick', alpha=0.7, edgecolor='none', pad=3))

    ax.set_title(f'Post-NMS  ({nms_dets.shape[0]} detection{"s" if nms_dets.shape[0] != 1 else ""})',
                 fontsize=11)
    ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.axis('off')

    # ------------------------------------------------------------------
    # Panel 2 — Pre-NMS hypothesis map
    # ------------------------------------------------------------------
    ax = axes[1]
    ax.imshow(bg)
    draw_box(ax, gt_xyxy, color='lime', linestyle='--', lw=2)

    score_max = raw_score.max() if raw_score.size > 0 else 1.0
    norm = Normalize(vmin=0, vmax=max(score_max, NMS_CONF_THRESH))

    if raw_score.size > 0:
        # draw lowest-scoring anchors first so high-scoring ones render on top
        order = np.argsort(raw_score)
        sc = ax.scatter(
            raw_cx[order], raw_cy[order],
            c=raw_score[order], cmap='plasma', norm=norm,
            s=6, alpha=0.55, linewidths=0, rasterized=True,
        )
        # highlight anchors whose center is inside the GT box with diamonds
        if n_in > 0:
            order_in = np.argsort(raw_score[in_gt])
            ax.scatter(
                raw_cx[in_gt][order_in], raw_cy[in_gt][order_in],
                c=raw_score[in_gt][order_in], cmap='plasma', norm=norm,
                s=50, alpha=1.0, marker='D', linewidths=0.6,
                edgecolors='white', rasterized=True,
            )
        cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('obj × cls_prob', fontsize=9)
        cb.ax.axhline(NMS_CONF_THRESH, color='white', linestyle='--', linewidth=1)

    ax.set_title(
        f'Pre-NMS Hypotheses  (obj > 0.01,  M={raw_boxes.shape[0]})\n'
        f'◆ inside GT box (n={n_in}) · · outside (n={n_out})',
        fontsize=11,
    )
    ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.axis('off')

    # ------------------------------------------------------------------
    # Panel 3 — obj vs cls-prob scatter with NMS filter boundaries
    #
    # NMS applies two independent filters:
    #   1. obj > 0.25          (vertical line)  — objectness pre-filter
    #   2. obj × cls > 0.25   (hyperbola)       — combined confidence filter
    # Anchors must clear BOTH to become detection candidates (green zone).
    # ------------------------------------------------------------------
    ax = axes[2]

    # draw background zones
    obj_range = np.linspace(NMS_CONF_THRESH, 1.0, 300)
    cls_boundary = np.minimum(NMS_CONF_THRESH / obj_range, 1.0)
    ax.fill_between(obj_range, cls_boundary, 1.0, alpha=0.07, color='green')  # survivor zone

    # out-of-box anchors: small, blue, semi-transparent
    if n_out > 0:
        ax.scatter(raw_obj[~in_gt], raw_tcls[~in_gt],
                   s=20, alpha=0.7, color='steelblue', linewidths=0, rasterized=True,
                   label=f'outside GT box (n={n_out})')

    # in-box anchors: larger, red diamonds — always render on top
    if n_in > 0:
        ax.scatter(raw_obj[in_gt], raw_tcls[in_gt],
                   s=35, alpha=0.9, color='crimson', marker='D', linewidths=0.4,
                   edgecolors='white', rasterized=True,
                   label=f'inside GT box (n={n_in})')

        # label top-3 in-box anchors with their argmax predicted class
        top3_idx = np.argsort(raw_score[in_gt])[-3:][::-1]
        in_obj  = raw_obj[in_gt]
        in_tcls = raw_tcls[in_gt]
        in_cls  = raw_cls[in_gt]
        for k in top3_idx:
            argmax_cls = int(in_cls[k].argmax())
            ax.annotate(
                cls_name(argmax_cls),
                (in_obj[k], in_tcls[k]),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, color='crimson',
            )

    # NMS boundary lines
    ax.axvline(NMS_CONF_THRESH, color='dimgray', linestyle='--', linewidth=1.2,
               label=f'① obj pre-filter ({NMS_CONF_THRESH})')
    ax.plot(obj_range, cls_boundary, color='black', linestyle=':', linewidth=1.2,
            label=f'② combined filter (obj×cls={NMS_CONF_THRESH})')
    ax.text(0.97, 0.97, 'detection\nzone', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='green', alpha=0.7)

    ax.set_xlabel('Objectness (raw)', fontsize=10)
    ax.set_ylabel(f'{t_name} class prob (raw)', fontsize=10)
    ax.set_title(
        'Pre-NMS: obj vs class prob\n(labels = argmax class of top-3 in-box anchors)',
        fontsize=11,
    )
    # ax.legend(fontsize=8, loc='lower right')
    # legend outside the plot area to avoid overlapping with data points
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# IoU-based scatter figure
# ---------------------------------------------------------------------------

def visualize_iou_scatter(entry, out_path=None):
    """
    Scatter: objectness (x) vs max class probability (y) for every pre-NMS anchor
    that has non-zero IoU with the GT bounding box.

      Red diamond  — argmax class IS the target (stop sign)
      Blue circle  — argmax class is something else
    """
    target_cls = int(entry['gt_box'][0, 4])
    gt_xyxy    = entry['gt_box'][0, :4]
    raw_boxes  = entry['raw_boxes']    # [M,4] xywh
    raw_obj    = entry['raw_obj']      # [M]
    raw_cls    = entry['raw_cls']      # [M,80]

    has_iou, iou_vals = iou_with_gt(raw_boxes, gt_xyxy)
    n_total = int(has_iou.sum())

    t_name  = cls_name(target_cls)
    s_str   = "HIDDEN ✓" if entry['attack_success'] else "DETECTED ✗"
    s_color = "green" if entry['attack_success'] else "red"

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"Image #{entry['image_idx']}  |  "
        f"resize={entry['set_resize']:.2f}  rotate={entry['set_rotate']:.1f}°  |  "
        f"[{s_str}]",
        fontsize=12, fontweight='bold', color=s_color,
    )

    if n_total == 0:
        ax.text(0.5, 0.5, 'No anchors overlap GT box',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
    else:
        obj_f  = raw_obj[has_iou]          # [N]
        cls_f  = raw_cls[has_iou]          # [N,80]
        max_p  = cls_f.max(axis=1)         # [N] highest class prob
        argmax = cls_f.argmax(axis=1)      # [N] which class
        is_tgt = argmax == target_cls

        n_red  = int(is_tgt.sum())
        n_blue = n_total - n_red

        # plot non-target first so target points render on top
        if n_blue > 0:
            ax.scatter(obj_f[~is_tgt], max_p[~is_tgt],
                       s=25, alpha=0.6, color='steelblue', linewidths=0,
                       rasterized=True,
                       label=f'other class (n={n_blue})')
        if n_red > 0:
            ax.scatter(obj_f[is_tgt], max_p[is_tgt],
                       s=40, alpha=0.9, color='crimson', marker='D',
                       linewidths=0.4, edgecolors='white', rasterized=True,
                       label=f'{t_name} (n={n_red})')

        # annotate top-3 points by objectness across ALL points
        top_k   = min(3, n_total)
        top_idx = np.argsort(obj_f)[-top_k:][::-1]
        for k in top_idx:
            ac    = int(argmax[k])
            color = 'crimson' if is_tgt[k] else 'steelblue'
            ax.annotate(
                cls_name(ac),
                (obj_f[k], max_p[k]),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, color=color,
            )

        # NMS boundary lines (same as Panel 3)
        obj_range    = np.linspace(NMS_CONF_THRESH, 1.0, 300)
        cls_boundary = np.minimum(NMS_CONF_THRESH / obj_range, 1.0)
        ax.fill_between(obj_range, cls_boundary, 1.0, alpha=0.07, color='green')
        ax.axvline(NMS_CONF_THRESH, color='dimgray', linestyle='--', linewidth=1.2,
                   label=f'① obj pre-filter ({NMS_CONF_THRESH})')
        ax.plot(obj_range, cls_boundary, color='black', linestyle=':', linewidth=1.2,
                label=f'② combined filter (obj×cls={NMS_CONF_THRESH})')
        ax.text(0.97, 0.97, 'detection\nzone', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='green', alpha=0.7)

        ax.legend(fontsize=9, loc='lower right')

    ax.set_xlabel('Objectness (raw)', fontsize=11)
    ax.set_ylabel('Max class probability (over all classes)', fontsize=11)
    ax.set_title(
        f'Anchors with IoU > 0 vs GT box  (N={n_total})\n'
        f'Red ◆ = argmax is {t_name}  ·  Blue = other class',
        fontsize=11,
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results):
    n = len(results)
    n_success = sum(r['attack_success'] for r in results)
    print(f"\n{'='*50}")
    print(f"Total entries : {n}")
    print(f"Attack success: {n_success} / {n}  ({100*n_success/n:.1f}%)")

    # max in-box pre-NMS score per image
    target_cls = int(results[0]['gt_box'][0, 4])
    in_box_scores = []
    for r in results:
        raw_score = r['raw_obj'] * r['raw_cls'][:, target_cls]
        gt_xyxy   = r['gt_box'][0, :4]
        cx, cy    = r['raw_boxes'][:, 0], r['raw_boxes'][:, 1]
        in_gt     = centers_inside(cx, cy, gt_xyxy)
        if in_gt.sum() > 0:
            in_box_scores.append(raw_score[in_gt].max())
        else:
            in_box_scores.append(0.0)

    scores = np.array(in_box_scores)
    print(f"\nMax pre-NMS in-box score (obj × stop-sign cls) over {n} images:")
    print(f"  mean : {scores.mean():.4f}")
    print(f"  std  : {scores.std():.4f}")
    print(f"  min  : {scores.min():.4f}")
    print(f"  max  : {scores.max():.4f}")
    print(f"  > NMS combined threshold ({NMS_CONF_THRESH}): "
          f"{(scores > NMS_CONF_THRESH).sum()} / {n} images")

    # Class confusion: for in-box anchors with obj > NMS_CONF_THRESH,
    # what class does the model actually predict?
    from collections import Counter
    argmax_counts = Counter()
    for r in results:
        gt_xyxy = r['gt_box'][0, :4]
        cx, cy  = r['raw_boxes'][:, 0], r['raw_boxes'][:, 1]
        in_gt   = centers_inside(cx, cy, gt_xyxy)
        high_obj = r['raw_obj'] > NMS_CONF_THRESH
        mask = in_gt & high_obj
        if mask.sum() > 0:
            top_cls = r['raw_cls'][mask].argmax(axis=1)
            argmax_counts.update(top_cls.tolist())
    if argmax_counts:
        print(f"\nArgmax class of in-box anchors with obj > {NMS_CONF_THRESH}:")
        for cls_id, count in argmax_counts.most_common(5):
            print(f"  {cls_name(cls_id):20s}: {count}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize patch attack results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('pkl', help='Path to *_results.pkl file from demo.py eval()')
    parser.add_argument('--idx',     type=int, default=0,
                        help='Index into results list to visualize (default: 0)')
    parser.add_argument('--all',     action='store_true',
                        help='Save visualizations for all entries')
    parser.add_argument('--out',     default=None,
                        help='Output PNG path (default: show interactively)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics and exit')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    print_summary(results)

    if args.summary:
        return

    if args.all:
        base = args.pkl.replace('_results.pkl', '')
        viz_dir = f"{base}_viz"
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Saving {len(results)} visualizations to {viz_dir}/")
        for entry in results:
            idx_str = f"{entry['image_idx']:04d}"
            visualize_entry(entry, out_path=os.path.join(viz_dir, f"{idx_str}.png"))
            visualize_iou_scatter(entry, out_path=os.path.join(viz_dir, f"{idx_str}_iou.png"))
    else:
        if args.idx >= len(results):
            print(f"ERROR: --idx {args.idx} out of range (max {len(results)-1})")
            sys.exit(1)
        entry = results[args.idx]
        print(f"Visualizing entry idx={args.idx}  image_idx={entry['image_idx']}  "
              f"success={entry['attack_success']}")
        visualize_entry(entry, out_path=args.out)
        # IoU scatter: derive output path from --out if provided
        if args.out:
            iou_out = args.out.replace('.png', '_iou.png')
            if iou_out == args.out:
                iou_out = args.out + '_iou.png'
        else:
            iou_out = None
        visualize_iou_scatter(entry, out_path=iou_out)


if __name__ == '__main__':
    main()
