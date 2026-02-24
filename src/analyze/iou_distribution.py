"""
IoU distribution across all pre-NMS predictions, averaged over entries.

For each entry, every pre-NMS anchor's IoU with the GT box is computed and
binned into [0.0-0.1, 0.1-0.2, ..., 0.9-1.0].  Each bin count is expressed
as a percentage of the total anchors in that entry.  These per-entry
percentages are then averaged separately for:

  • entries where the hiding attack succeeded  (green)
  • entries where the hiding attack failed     (red)

The result is a grouped bar chart showing the mean percentage ± std per bin.

If a companion *_results_orig.pkl is available (produced by demo.py when eval_only
is True), a second figure is generated comparing:

  • attack-success entries from the adversarial-patch results  (green)
  • all entries from the original (no-patch) results           (steelblue)

The same printed stats are also produced for the w/ patch vs w/o patch split.

Usage:
    python iou_distribution.py demo/20260220-131307_results.pkl
    python iou_distribution.py demo/20260220-131307_results.pkl --out iou_dist.png
    python iou_distribution.py demo/20260220-131307_results.pkl --no-std
    # explicit orig path (auto-detected by default):
    python iou_distribution.py demo/20260220-131307_results.pkl --orig demo/20260220-131307_results_orig.pkl
"""

# import _init_path
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

N_BINS   = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)   # [0.0, 0.1, ..., 1.0]
BIN_LABELS = [f"{BIN_EDGES[i]:.1f}–{BIN_EDGES[i+1]:.1f}" for i in range(N_BINS)]


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


def entry_iou_percentages(entry):
    """
    Returns a length-N_BINS array of percentages (sum = 100) for one entry.
    IoU == 1.0 is included in the last bin [0.9, 1.0].
    """
    gt_xyxy   = entry['gt_box'][0, :4]
    raw_boxes = entry['raw_boxes']
    iou       = iou_all(raw_boxes, gt_xyxy)                     # [M]

    # clip so IoU=1.0 lands in bin index 9 (not 10)
    bin_idx = np.clip(np.floor(iou * N_BINS).astype(int), 0, N_BINS - 1)
    counts  = np.bincount(bin_idx, minlength=N_BINS).astype(float)
    total   = counts.sum()
    return (counts / total * 100.0) if total > 0 else counts


def collect_distributions(results):
    """
    Returns two arrays:
      success_pcts  [S, N_BINS]  — per-entry percentages for successful entries
      failure_pcts  [F, N_BINS]  — per-entry percentages for failed entries
    """
    success_rows, failure_rows = [], []
    for entry in results:
        pct = entry_iou_percentages(entry)
        if entry['attack_success']:
            success_rows.append(pct)
        else:
            failure_rows.append(pct)

    success_pcts = np.array(success_rows) if success_rows else np.empty((0, N_BINS))
    failure_pcts = np.array(failure_rows) if failure_rows else np.empty((0, N_BINS))
    return success_pcts, failure_pcts


def plot_iou_distribution(success_pcts, failure_pcts, show_std=True, out_path=None):
    """
    Grouped bar chart: mean percentage per IoU bin for success vs failure.
    Error bars show ±1 std when show_std=True.
    """
    n_s = success_pcts.shape[0]
    n_f = failure_pcts.shape[0]

    mean_s = success_pcts.mean(axis=0) if n_s > 0 else np.zeros(N_BINS)
    mean_f = failure_pcts.mean(axis=0) if n_f > 0 else np.zeros(N_BINS)
    std_s  = success_pcts.std(axis=0)  if n_s > 0 else np.zeros(N_BINS)
    std_f  = failure_pcts.std(axis=0)  if n_f > 0 else np.zeros(N_BINS)

    x      = np.arange(N_BINS)
    width  = 0.38
    offset = width / 2

    fig, ax = plt.subplots(figsize=(11, 5))

    err_kw = dict(capsize=4, capthick=1.2, elinewidth=1.2)

    bars_s = ax.bar(
        x - offset, mean_s, width,
        color='seagreen', alpha=0.82, label=f'Attack success  (n={n_s})',
        yerr=(std_s if show_std else None), error_kw=err_kw,
    )
    bars_f = ax.bar(
        x + offset, mean_f, width,
        color='tomato', alpha=0.82, label=f'Attack failure  (n={n_f})',
        yerr=(std_f if show_std else None), error_kw=err_kw,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('IoU bin (pre-NMS anchor vs GT box)', fontsize=11)
    ax.set_ylabel('Mean % of anchors per entry', fontsize=11)
    ax.set_title(
        'Pre-NMS anchor IoU distribution  —  success vs failure\n'
        + (f'Error bars: ±1 std  (n_success={n_s}, n_failure={n_f})'
           if show_std else f'n_success={n_s},  n_failure={n_f}'),
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.6, N_BINS - 0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def collect_score_stats_by_bin(results):
    """
    For each entry and each IoU bin, record the MIN and MAX of each metric
    (objectiveness, max-class-prob, obj×max-class-prob) among the anchors
    that fall into that bin.  Bins with no anchors contribute NaN.

    Returns two dicts, one for success entries and one for failure entries.
    Each dict is keyed by metric name ('obj', 'maxp', 'score') and maps to
    a sub-dict {'min': [N_entries, N_BINS], 'max': [N_entries, N_BINS]}.
    """
    def empty_bucket():
        return {'obj':   {'min': [], 'max': []},
                'maxp':  {'min': [], 'max': []},
                'score': {'min': [], 'max': []}}

    success = empty_bucket()
    failure = empty_bucket()

    for entry in results:
        raw_obj   = entry['raw_obj']           # [M]
        raw_cls   = entry['raw_cls']           # [M, 80]
        gt_xyxy   = entry['gt_box'][0, :4]
        raw_boxes = entry['raw_boxes']

        iou   = iou_all(raw_boxes, gt_xyxy)    # [M]
        maxp  = raw_cls.max(axis=1)            # [M]
        score = raw_obj * maxp                 # [M]

        bin_idx = np.clip(np.floor(iou * N_BINS).astype(int), 0, N_BINS - 1)

        metrics_data = {'obj': raw_obj, 'maxp': maxp, 'score': score}
        row_min = {k: np.full(N_BINS, np.nan) for k in metrics_data}
        row_max = {k: np.full(N_BINS, np.nan) for k in metrics_data}

        for b in range(N_BINS):
            mask = bin_idx == b
            if mask.any():
                for k, vals in metrics_data.items():
                    row_min[k][b] = vals[mask].min()
                    row_max[k][b] = vals[mask].max()

        bucket = success if entry['attack_success'] else failure
        for k in metrics_data:
            bucket[k]['min'].append(row_min[k])
            bucket[k]['max'].append(row_max[k])

    def to_arr(lst):
        return np.array(lst) if lst else np.full((0, N_BINS), np.nan)

    for bucket in (success, failure):
        for k in bucket:
            bucket[k]['min'] = to_arr(bucket[k]['min'])
            bucket[k]['max'] = to_arr(bucket[k]['max'])

    return success, failure


def print_score_stats_by_bin(success_stats, failure_stats):
    """
    For each metric and each IoU bin print the average (across entries) of
    the per-entry minimum and maximum, separated by success / failure.
    """
    metrics = [
        ('Objectiveness',           'obj'),
        ('Max class prob',           'maxp'),
        ('Obj × max class prob',     'score'),
    ]

    n_s = next(iter(success_stats.values()))['min'].shape[0]
    n_f = next(iter(failure_stats.values()))['min'].shape[0]

    def fmt(arr, axis=0):
        """nanmean over entries; returns [N_BINS]."""
        return np.nanmean(arr, axis=axis) if arr.shape[0] > 0 else np.full(N_BINS, np.nan)

    for label, key in metrics:
        s_min = fmt(success_stats[key]['min'])
        s_max = fmt(success_stats[key]['max'])
        f_min = fmt(failure_stats[key]['min'])
        f_max = fmt(failure_stats[key]['max'])

        col = 18
        print(f"\n  {label}  (n_success={n_s}, n_failure={n_f})")
        print(f"  {'IoU bin':<12}  "
              f"{'success avg-min':>{col}}  {'success avg-max':>{col}}  "
              f"{'failure avg-min':>{col}}  {'failure avg-max':>{col}}")
        print("  " + "-" * (12 + 4 * (col + 2) + 2))
        for i, bin_label in enumerate(BIN_LABELS):
            def v(x): return f"{x:.4f}" if not np.isnan(x) else "  n/a  "
            print(f"  {bin_label:<12}  "
                  f"{v(s_min[i]):>{col}}  {v(s_max[i]):>{col}}  "
                  f"{v(f_min[i]):>{col}}  {v(f_max[i]):>{col}}")
    print()


def print_table(success_pcts, failure_pcts):
    n_s = success_pcts.shape[0]
    n_f = failure_pcts.shape[0]
    mean_s = success_pcts.mean(axis=0) if n_s > 0 else np.zeros(N_BINS)
    mean_f = failure_pcts.mean(axis=0) if n_f > 0 else np.zeros(N_BINS)
    std_s  = success_pcts.std(axis=0)  if n_s > 0 else np.zeros(N_BINS)
    std_f  = failure_pcts.std(axis=0)  if n_f > 0 else np.zeros(N_BINS)

    print(f"\n{'IoU bin':<12}  {'success mean±std':>18}  {'failure mean±std':>18}")
    print("-" * 54)
    for i, label in enumerate(BIN_LABELS):
        print(f"{label:<12}  {mean_s[i]:6.2f} ± {std_s[i]:5.2f}    "
              f"{mean_f[i]:6.2f} ± {std_f[i]:5.2f}")
    print()


# ---------------------------------------------------------------------------
# Patch-vs-original comparison helpers
# ---------------------------------------------------------------------------

def collect_all_pcts(results):
    """Returns [N, N_BINS] IoU-percentage array for every entry (no split)."""
    rows = [entry_iou_percentages(entry) for entry in results]
    return np.array(rows) if rows else np.empty((0, N_BINS))


def collect_all_score_stats(results):
    """
    Like collect_score_stats_by_bin but returns a single bucket covering all
    entries.  Structure: {'obj': {'min': [N,N_BINS], 'max': ...}, ...}
    """
    bucket = {'obj':   {'min': [], 'max': []},
              'maxp':  {'min': [], 'max': []},
              'score': {'min': [], 'max': []}}

    for entry in results:
        raw_obj   = entry['raw_obj']
        raw_cls   = entry['raw_cls']
        gt_xyxy   = entry['gt_box'][0, :4]
        raw_boxes = entry['raw_boxes']

        iou   = iou_all(raw_boxes, gt_xyxy)
        maxp  = raw_cls.max(axis=1)
        score = raw_obj * maxp

        bin_idx = np.clip(np.floor(iou * N_BINS).astype(int), 0, N_BINS - 1)
        metrics_data = {'obj': raw_obj, 'maxp': maxp, 'score': score}
        row_min = {k: np.full(N_BINS, np.nan) for k in metrics_data}
        row_max = {k: np.full(N_BINS, np.nan) for k in metrics_data}

        for b in range(N_BINS):
            mask = bin_idx == b
            if mask.any():
                for k, vals in metrics_data.items():
                    row_min[k][b] = vals[mask].min()
                    row_max[k][b] = vals[mask].max()

        for k in metrics_data:
            bucket[k]['min'].append(row_min[k])
            bucket[k]['max'].append(row_max[k])

    def to_arr(lst):
        return np.array(lst) if lst else np.full((0, N_BINS), np.nan)

    for k in bucket:
        bucket[k]['min'] = to_arr(bucket[k]['min'])
        bucket[k]['max'] = to_arr(bucket[k]['max'])

    return bucket


def plot_iou_patch_vs_orig(patch_success_pcts, orig_pcts, show_std=True, out_path=None):
    """
    Grouped bar chart: attack-success cases (w/ adversarial patch) vs all
    entries from the original (no-patch) evaluation.
    """
    n_p = patch_success_pcts.shape[0]
    n_o = orig_pcts.shape[0]

    mean_p = patch_success_pcts.mean(axis=0) if n_p > 0 else np.zeros(N_BINS)
    mean_o = orig_pcts.mean(axis=0)          if n_o > 0 else np.zeros(N_BINS)
    std_p  = patch_success_pcts.std(axis=0)  if n_p > 0 else np.zeros(N_BINS)
    std_o  = orig_pcts.std(axis=0)           if n_o > 0 else np.zeros(N_BINS)

    x      = np.arange(N_BINS)
    width  = 0.38
    offset = width / 2

    fig, ax = plt.subplots(figsize=(11, 5))
    err_kw = dict(capsize=4, capthick=1.2, elinewidth=1.2)

    ax.bar(x - offset, mean_p, width,
           color='seagreen', alpha=0.82,
           label=f'w/ patch — attack success  (n={n_p})',
           yerr=(std_p if show_std else None), error_kw=err_kw)
    ax.bar(x + offset, mean_o, width,
           color='steelblue', alpha=0.82,
           label=f'w/o patch — original  (n={n_o})',
           yerr=(std_o if show_std else None), error_kw=err_kw)

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('IoU bin (pre-NMS anchor vs GT box)', fontsize=11)
    ax.set_ylabel('Mean % of anchors per entry', fontsize=11)
    ax.set_title(
        'Pre-NMS anchor IoU distribution  —  w/ patch (attack success) vs w/o patch\n'
        + (f'Error bars: ±1 std  (n_patch={n_p}, n_orig={n_o})'
           if show_std else f'n_patch={n_p},  n_orig={n_o}'),
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.6, N_BINS - 0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def print_table_patch_vs_orig(patch_success_pcts, orig_pcts):
    n_p = patch_success_pcts.shape[0]
    n_o = orig_pcts.shape[0]
    mean_p = patch_success_pcts.mean(axis=0) if n_p > 0 else np.zeros(N_BINS)
    mean_o = orig_pcts.mean(axis=0)          if n_o > 0 else np.zeros(N_BINS)
    std_p  = patch_success_pcts.std(axis=0)  if n_p > 0 else np.zeros(N_BINS)
    std_o  = orig_pcts.std(axis=0)           if n_o > 0 else np.zeros(N_BINS)

    col = 26
    print(f"\n{'IoU bin':<12}  "
          f"{'w/ patch success mean±std':>{col}}  "
          f"{'w/o patch mean±std':>{col}}")
    print("-" * (14 + 2 * (col + 2)))
    for i, label in enumerate(BIN_LABELS):
        print(f"{label:<12}  "
              f"{mean_p[i]:6.2f} ± {std_p[i]:5.2f}{'':>{col - 16}}  "
              f"{mean_o[i]:6.2f} ± {std_o[i]:5.2f}")
    print()


def print_score_stats_patch_vs_orig(patch_success_stats, orig_all_stats):
    """
    Same layout as print_score_stats_by_bin but columns are
    w/ patch (attack-success) vs w/o patch (all original entries).
    """
    metrics = [
        ('Objectiveness',        'obj'),
        ('Max class prob',       'maxp'),
        ('Obj × max class prob', 'score'),
    ]

    n_p = next(iter(patch_success_stats.values()))['min'].shape[0]
    n_o = next(iter(orig_all_stats.values()))['min'].shape[0]

    def fmt(arr):
        return np.nanmean(arr, axis=0) if arr.shape[0] > 0 else np.full(N_BINS, np.nan)

    for label, key in metrics:
        p_min = fmt(patch_success_stats[key]['min'])
        p_max = fmt(patch_success_stats[key]['max'])
        o_min = fmt(orig_all_stats[key]['min'])
        o_max = fmt(orig_all_stats[key]['max'])

        col = 20
        print(f"\n  {label}  (n_patch_success={n_p}, n_orig={n_o})")
        print(f"  {'IoU bin':<12}  "
              f"{'w/patch avg-min':>{col}}  {'w/patch avg-max':>{col}}  "
              f"{'w/o patch avg-min':>{col}}  {'w/o patch avg-max':>{col}}")
        print("  " + "-" * (12 + 4 * (col + 2) + 2))
        for i, bin_label in enumerate(BIN_LABELS):
            def v(x): return f"{x:.4f}" if not np.isnan(x) else "  n/a  "
            print(f"  {bin_label:<12}  "
                  f"{v(p_min[i]):>{col}}  {v(p_max[i]):>{col}}  "
                  f"{v(o_min[i]):>{col}}  {v(o_max[i]):>{col}}")
    print()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='IoU distribution of pre-NMS anchors vs GT box',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('pkl', help='Path to *_results.pkl from demo.py eval()')
    parser.add_argument('--orig', default=None,
                        help='Path to *_results_orig.pkl (auto-detected if omitted)')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: show interactively)')
    parser.add_argument('--no-std', action='store_true',
                        help='Hide std error bars')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    n_total   = len(results)
    n_success = sum(r['attack_success'] for r in results)
    print(f"Loaded {n_total} entries  |  "
          f"success: {n_success}  failure: {n_total - n_success}")

    # --- existing: success vs failure within patched results ---
    success_stats, failure_stats = collect_score_stats_by_bin(results)
    print_score_stats_by_bin(success_stats, failure_stats)

    success_pcts, failure_pcts = collect_distributions(results)
    print_table(success_pcts, failure_pcts)
    plot_iou_distribution(success_pcts, failure_pcts,
                          show_std=not args.no_std,
                          out_path=args.out)

    # --- new: w/ patch vs w/o patch comparison ---
    orig_pkl = args.orig
    if orig_pkl is None:
        # auto-detect: foo_results.pkl -> foo_results_orig.pkl
        orig_pkl = args.pkl.replace('_results', '_results_orig')

    if not os.path.exists(orig_pkl):
        print(f"[patch-vs-orig] no orig pkl found at {orig_pkl!r} — skipping")
        return

    with open(orig_pkl, 'rb') as f:
        results_orig = pickle.load(f)

    n_orig = len(results_orig)
    print(f"\nLoaded {n_orig} original (no-patch) entries from {orig_pkl!r}")

    print("\n=== w/ patch (attack success) vs w/o patch ===")
    patch_success_stats, _ = collect_score_stats_by_bin(results)
    # only keep success bucket — already computed above, reuse
    orig_all_stats = collect_all_score_stats(results_orig)
    print_score_stats_patch_vs_orig(patch_success_stats, orig_all_stats)

    orig_pcts = collect_all_pcts(results_orig)
    print_table_patch_vs_orig(success_pcts, orig_pcts)

    orig_out = None
    if args.out:
        base, ext = os.path.splitext(args.out)
        orig_out = f"{base}_patch_vs_orig{ext}"
    plot_iou_patch_vs_orig(success_pcts, orig_pcts,
                           show_std=not args.no_std,
                           out_path=orig_out)


if __name__ == '__main__':
    main()
