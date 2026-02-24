"""
Aggregate scatter plot across all successful hiding-attack entries.

For each entry where attack_success=True, find the pre-NMS anchor with the
highest IoU against the GT bounding box, then plot:

  x-axis : objectness of that anchor
  y-axis : highest class probability (max over all 80 classes) of that anchor
  color  : red  if argmax class == stop sign
           blue if argmax class is anything else
  alpha  : IoU value, lower-bounded to 0.1

Usage:
    python scatter_success.py demo/20260220-131307_results.pkl
    python scatter_success.py demo/20260220-131307_results.pkl --out success_scatter.png
"""

# import _init_path
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---- shared helpers from analyze_results (inlined to keep this self-contained) ----

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

STOP_SIGN_IDX = COCO80.index('stop sign')   # == 11
ALPHA_MIN     = 0.1
NMS_CONF_THRESH = 0.25


def cls_name(cls_id):
    idx = int(cls_id)
    return COCO80[idx] if idx < len(COCO80) else f"cls{idx}"


def xywh_to_xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def iou_all(raw_boxes_xywh, gt_xyxy):
    """IoU between every anchor [M,4 xywh] and one GT box [4 xyxy]. Returns [M]."""
    anchors = xywh_to_xyxy(raw_boxes_xywh)
    ix1 = np.maximum(anchors[:, 0], gt_xyxy[0])
    iy1 = np.maximum(anchors[:, 1], gt_xyxy[1])
    ix2 = np.minimum(anchors[:, 2], gt_xyxy[2])
    iy2 = np.minimum(anchors[:, 3], gt_xyxy[3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_g = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])
    union  = area_a + area_g - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Aggregate collector
# ---------------------------------------------------------------------------

def collect_best_iou_anchors(results):
    """
    For every successful entry return a dict with per-anchor data for the
    anchor that has the highest IoU with the GT box.

    Returns lists: obj, max_cls_prob, is_stop_sign, iou, image_idx
    """
    obj_vals, maxp_vals, is_ss_vals, iou_vals, idx_vals = [], [], [], [], []
    cls_vals_best = []

    skipped = 0
    for entry in results:
        if not entry['attack_success']:
            continue

        gt_xyxy   = entry['gt_box'][0, :4]
        raw_boxes = entry['raw_boxes']   # [M,4] xywh
        raw_obj   = entry['raw_obj']     # [M]
        raw_cls   = entry['raw_cls']     # [M,80]

        iou = iou_all(raw_boxes, gt_xyxy)   # [M]
        best = int(np.argmax(iou))
        best_iou = float(iou[best])

        if best_iou == 0.0:
            skipped += 1
            continue

        best_obj  = float(raw_obj[best])
        best_cls  = raw_cls[best]          # [80]
        best_maxp = float(best_cls.max())
        best_argmax = int(best_cls.argmax())

        obj_vals.append(best_obj)
        maxp_vals.append(best_maxp)
        is_ss_vals.append(best_argmax == STOP_SIGN_IDX)
        iou_vals.append(best_iou)
        idx_vals.append(entry['image_idx'])
        cls_vals_best.append(best_argmax)

    if skipped:
        print(f"  [warn] {skipped} successful entries had no anchor overlapping the GT box — skipped.")

    return (
        np.array(obj_vals),
        np.array(maxp_vals),
        np.array(is_ss_vals, dtype=bool),
        np.array(iou_vals),
        idx_vals,
        cls_vals_best,
    )


# ---------------------------------------------------------------------------
# Aggregate collector — best class-probability anchor within IoU threshold
# ---------------------------------------------------------------------------

def collect_best_maxp_anchors(results, min_iou=0.0):
    """
    For every successful entry, keep only anchors with IoU >= min_iou, then
    select the one with the highest max-class-probability among those anchors.

    Returns: obj, maxp, is_stop_sign, iou, image_idx_list, cls_best_list
    """
    obj_vals, maxp_vals, is_ss_vals, iou_vals, idx_vals, cls_vals_best = \
        [], [], [], [], [], []

    skipped = 0
    for entry in results:
        if not entry['attack_success']:
            continue

        gt_xyxy   = entry['gt_box'][0, :4]
        raw_boxes = entry['raw_boxes']   # [M,4] xywh
        raw_obj   = entry['raw_obj']     # [M]
        raw_cls   = entry['raw_cls']     # [M,80]

        iou = iou_all(raw_boxes, gt_xyxy)   # [M]
        mask = iou >= min_iou
        if not mask.any():
            skipped += 1
            continue

        # among qualifying anchors pick the one with highest max-class-prob
        maxp_cands  = raw_cls[mask].max(axis=1)          # [K]
        best_local  = int(np.argmax(maxp_cands))
        orig_indices = np.where(mask)[0]
        best         = orig_indices[best_local]

        obj_vals.append(float(raw_obj[best]))
        maxp_vals.append(float(raw_cls[best].max()))
        argmax_cls = int(raw_cls[best].argmax())
        is_ss_vals.append(argmax_cls == STOP_SIGN_IDX)
        iou_vals.append(float(iou[best]))
        idx_vals.append(entry['image_idx'])
        cls_vals_best.append(argmax_cls)

    if skipped:
        print(f"  [warn] {skipped} successful entries had no anchor with "
              f"IoU >= {min_iou:.3f} — skipped.")

    return (
        np.array(obj_vals),
        np.array(maxp_vals),
        np.array(is_ss_vals, dtype=bool),
        np.array(iou_vals),
        idx_vals,
        cls_vals_best,
    )


# ---------------------------------------------------------------------------
# Aggregate collector — highest-objectiveness anchor within IoU threshold
# ---------------------------------------------------------------------------

def collect_best_obj_anchors(results, min_iou=0.0):
    """
    For every successful entry, keep only anchors with IoU >= min_iou, then
    select the one with the highest objectiveness score among those anchors.

    Returns: obj, maxp, is_stop_sign, iou, image_idx_list, cls_best_list
    """
    obj_vals, maxp_vals, is_ss_vals, iou_vals, idx_vals, cls_vals_best = \
        [], [], [], [], [], []

    skipped = 0
    for entry in results:
        if not entry['attack_success']:
            continue

        gt_xyxy   = entry['gt_box'][0, :4]
        raw_boxes = entry['raw_boxes']   # [M,4] xywh
        raw_obj   = entry['raw_obj']     # [M]
        raw_cls   = entry['raw_cls']     # [M,80]

        iou = iou_all(raw_boxes, gt_xyxy)   # [M]
        mask = iou >= min_iou
        if not mask.any():
            skipped += 1
            continue

        # among qualifying anchors pick the one with highest objectiveness
        obj_cands    = raw_obj[mask]                     # [K]
        best_local   = int(np.argmax(obj_cands))
        orig_indices = np.where(mask)[0]
        best         = orig_indices[best_local]

        argmax_cls = int(raw_cls[best].argmax())
        obj_vals.append(float(raw_obj[best]))
        maxp_vals.append(float(raw_cls[best].max()))
        is_ss_vals.append(argmax_cls == STOP_SIGN_IDX)
        iou_vals.append(float(iou[best]))
        idx_vals.append(entry['image_idx'])
        cls_vals_best.append(argmax_cls)

    if skipped:
        print(f"  [warn] {skipped} successful entries had no anchor with "
              f"IoU >= {min_iou:.3f} — skipped.")

    return (
        np.array(obj_vals),
        np.array(maxp_vals),
        np.array(is_ss_vals, dtype=bool),
        np.array(iou_vals),
        idx_vals,
        cls_vals_best,
    )


# ---------------------------------------------------------------------------
# Plot (shared by both modes)
# ---------------------------------------------------------------------------

def _draw_scatter(ax, obj, maxp, is_ss, iou):
    """Render dots and NMS guides onto ax. Returns (blue_n, red_n)."""
    alphas    = np.maximum(ALPHA_MIN, iou)
    blue_mask = ~is_ss
    red_mask  =  is_ss

    if blue_mask.any():
        for xi, yi, ai in zip(obj[blue_mask], maxp[blue_mask], alphas[blue_mask]):
            ax.scatter(xi, yi, s=30, color='steelblue', alpha=float(ai),
                       linewidths=0, rasterized=True)
        ax.scatter([], [], s=30, color='steelblue', alpha=0.7, linewidths=0,
                   label=f'other class  (n={int(blue_mask.sum())})')

    if red_mask.any():
        for xi, yi, ai in zip(obj[red_mask], maxp[red_mask], alphas[red_mask]):
            ax.scatter(xi, yi, s=40, color='crimson', alpha=float(ai),
                       linewidths=0, rasterized=True)
        ax.scatter([], [], s=40, color='crimson', alpha=0.7, linewidths=0,
                   label=f'stop sign  (n={int(red_mask.sum())})')

    obj_range    = np.linspace(NMS_CONF_THRESH, 1.0, 300)
    cls_boundary = np.minimum(NMS_CONF_THRESH / obj_range, 1.0)
    ax.fill_between(obj_range, cls_boundary, 1.0, alpha=0.06, color='green')
    ax.axvline(NMS_CONF_THRESH, color='dimgray', linestyle='--', linewidth=1,
               label=f'obj pre-filter ({NMS_CONF_THRESH})')
    ax.plot(obj_range, cls_boundary, color='black', linestyle=':', linewidth=1,
            label=f'combined filter (obj×cls={NMS_CONF_THRESH})')
    ax.text(0.97, 0.97, 'detection\nzone', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='green', alpha=0.7)

    ax.set_xlabel('Objectness (raw)', fontsize=12)
    ax.set_ylabel('Max class probability (over all classes)', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return int(blue_mask.sum()), int(red_mask.sum())


def _add_iou_colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('IoU with GT box  (→ alpha)', fontsize=9)


def plot_scatter(obj, maxp, is_ss, iou, out_path=None):
    """Best-IoU mode: one point per successful entry (anchor with max IoU)."""
    n   = len(obj)
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_scatter(ax, obj, maxp, is_ss, iou)
    ax.set_title(
        f'Successful hiding attacks  —  best-IoU anchor per entry  (N={n})\n'
        f'Color: red = stop sign predicted  ·  blue = other\n'
        f'Transparency: IoU with GT box  (min={ALPHA_MIN})',
        fontsize=11,
    )
    _add_iou_colorbar(fig, ax)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_scatter_best_maxp(obj, maxp, is_ss, iou, min_iou, out_path=None):
    """Best-class-prob mode: one point per entry (highest-confidence anchor
    among those with IoU >= min_iou)."""
    n   = len(obj)
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_scatter(ax, obj, maxp, is_ss, iou)
    ax.set_title(
        f'Successful hiding attacks  —  highest-confidence anchor  (N={n})\n'
        f'IoU filter: ≥ {min_iou:.2f}  ·  '
        f'Color: red = stop sign  ·  blue = other\n'
        f'Transparency: IoU with GT box  (min={ALPHA_MIN})',
        fontsize=11,
    )
    _add_iou_colorbar(fig, ax)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_scatter_best_obj(obj, maxp, is_ss, iou, min_iou, out_path=None):
    """Best-objectiveness mode: one point per entry (highest-obj anchor
    among those with IoU >= min_iou)."""
    n   = len(obj)
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_scatter(ax, obj, maxp, is_ss, iou)
    ax.set_title(
        f'Successful hiding attacks  —  highest-objectiveness anchor  (N={n})\n'
        f'IoU filter: ≥ {min_iou:.2f}  ·  '
        f'Color: red = stop sign  ·  blue = other\n'
        f'Transparency: IoU with GT box  (min={ALPHA_MIN})',
        fontsize=11,
    )
    _add_iou_colorbar(fig, ax)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_cls_dist(is_ss, cls_best):
    other_cls_ids = [c for c, s in zip(cls_best, is_ss) if not s]
    if not other_cls_ids:
        return
    counts = np.bincount(other_cls_ids)
    print("  Other class distribution:")
    for cls_id, count in enumerate(counts):
        if count > 0:
            print(f"    {cls_name(cls_id):<15} : {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate scatter plot for successful hiding attacks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('pkl', help='Path to *_results.pkl from demo.py eval()')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: show interactively)')
    parser.add_argument(
        '--mode', choices=['best-iou', 'best-cls', 'best-obj'], default='best-iou',
        help=(
            'best-iou : one point per entry — anchor with highest IoU (default)\n'
            'best-cls : one point per entry — anchor with highest max-class-prob '
            'among those with IoU >= --min-iou\n'
            'best-obj : one point per entry — anchor with highest objectiveness '
            'among those with IoU >= --min-iou'
        ),
    )
    parser.add_argument(
        '--min-iou', type=float, default=0.0, metavar='THRESH',
        help='Minimum IoU for anchors considered in best-cls mode (default: 0.0)',
    )
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    n_total   = len(results)
    n_success = sum(r['attack_success'] for r in results)
    print(f"Loaded {n_total} entries  |  {n_success} successful ({100*n_success/n_total:.1f}%)")

    if args.mode == 'best-iou':
        obj, maxp, is_ss, iou, _, cls_best = collect_best_iou_anchors(results)
        print(f"Mode: best-IoU  |  points: {len(obj)}  |  "
              f"stop-sign: {is_ss.sum()}  other: {(~is_ss).sum()}")
        _print_cls_dist(is_ss, cls_best)
        if len(obj) == 0:
            print("Nothing to plot.")
            return
        plot_scatter(obj, maxp, is_ss, iou, out_path=args.out)

    elif args.mode == 'best-cls':
        obj, maxp, is_ss, iou, _, cls_best = collect_best_maxp_anchors(
            results, min_iou=args.min_iou)
        print(f"Mode: best-cls  |  min_iou={args.min_iou:.3f}  |  "
              f"points: {len(obj)}  |  "
              f"stop-sign: {is_ss.sum()}  other: {(~is_ss).sum()}")
        _print_cls_dist(is_ss, cls_best)
        if len(obj) == 0:
            print("Nothing to plot.")
            return
        plot_scatter_best_maxp(obj, maxp, is_ss, iou,
                               min_iou=args.min_iou, out_path=args.out)

    else:  # best-obj
        obj, maxp, is_ss, iou, _, cls_best = collect_best_obj_anchors(
            results, min_iou=args.min_iou)
        print(f"Mode: best-obj  |  min_iou={args.min_iou:.3f}  |  "
              f"points: {len(obj)}  |  "
              f"stop-sign: {is_ss.sum()}  other: {(~is_ss).sum()}")
        _print_cls_dist(is_ss, cls_best)
        if len(obj) == 0:
            print("Nothing to plot.")
            return
        plot_scatter_best_obj(obj, maxp, is_ss, iou,
                              min_iou=args.min_iou, out_path=args.out)


if __name__ == '__main__':
    main()
