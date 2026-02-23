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
    # Panel 3 — Score distribution: in-box vs out-of-box
    # ------------------------------------------------------------------
    ax = axes[2]
    bins = np.linspace(0, 1, 51)

    groups = [
        (in_gt,  f'inside GT box  (n={n_in})',  'crimson',   0.85),
        (~in_gt, f'outside GT box (n={n_out})', 'steelblue', 0.5),
    ]
    for mask, label, color, alpha in groups:
        if mask.sum() == 0:
            continue
        ax.hist(raw_score[mask], bins=bins, label=label, color=color,
                alpha=alpha, density=True)

    ax.axvline(NMS_CONF_THRESH, color='gray', linestyle='--', linewidth=1.2,
               label=f'NMS threshold ({NMS_CONF_THRESH})')

    # annotate max in-box score
    if n_in > 0:
        max_in = raw_score[in_gt].max()
        ax.axvline(max_in, color='crimson', linestyle=':', linewidth=1.2,
                   label=f'max in-box score ({max_in:.3f})')

    ax.set_xlabel(f'obj × {t_name} prob', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(
        'Pre-NMS Score Distribution\n(in-box vs out-of-box anchors)',
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

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
    print(f"\nMax pre-NMS in-box score (obj × cls) over {n} images:")
    print(f"  mean : {scores.mean():.4f}")
    print(f"  std  : {scores.std():.4f}")
    print(f"  min  : {scores.min():.4f}")
    print(f"  max  : {scores.max():.4f}")
    print(f"  > NMS threshold ({NMS_CONF_THRESH}): "
          f"{(scores > NMS_CONF_THRESH).sum()} / {n} images")
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
            out = os.path.join(viz_dir, f"{entry['image_idx']:04d}.png")
            visualize_entry(entry, out_path=out)
    else:
        if args.idx >= len(results):
            print(f"ERROR: --idx {args.idx} out of range (max {len(results)-1})")
            sys.exit(1)
        entry = results[args.idx]
        print(f"Visualizing entry idx={args.idx}  image_idx={entry['image_idx']}  "
              f"success={entry['attack_success']}")
        visualize_entry(entry, out_path=args.out)


if __name__ == '__main__':
    main()
