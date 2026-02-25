"""
Bounding-box shape distribution across pre-NMS predictions.

Three geometric features are computed for every pre-NMS anchor:
  (1) dist  — Euclidean distance (px) from anchor centre to GT box centre
  (2) ar    — Aspect ratio  w / h
  (3) size  — Absolute size  (w + h) / 2  (px)

Bin edges are derived from the pooled data (quantile-based, so every bin
covers roughly equal fractions of the overall anchor population).  Bins
are shared between the patched and original results so distributions are
directly comparable.

For each feature, two sets of grouped bar charts are produced:
  • success vs failure     (within the adversarial-patch results)
  • w/ patch vs w/o patch  (if *_results_orig.pkl is available)

Per-bin avg-min / avg-max of objectness, max-class-prob, and
obj × max-class-prob are also printed for both comparisons.

Usage:
    python bbox_distribution.py demo/20260220-131307_results.pkl
    python bbox_distribution.py demo/20260220-131307_results.pkl --out demo/bbox
    python bbox_distribution.py demo/20260220-131307_results.pkl --no-std --bins 8
    # explicit orig path (auto-detected by default):
    python bbox_distribution.py demo/20260220-131307_results.pkl \\
        --orig demo/20260220-131307_results_orig.pkl
"""

# import _init_path
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# (key, x-axis label, figure title snippet)
FEATURES = [
    ('dist', 'Relative distance  (dist / patch diagonal)',         'GT-centre relative distance'),
    ('ar',   'Relative aspect ratio  (anchor w/h ÷ patch w/h)',   'Relative aspect ratio'),
    ('size', 'Relative size  (anchor (w+h)/2 ÷ patch (w+h)/2)',   'Relative size'),
]

METRICS = [
    ('Objectness',           'obj'),
    ('Max class prob',       'maxp'),
    ('Obj × max class prob', 'score'),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _iou_with_gt(raw_boxes_xywh, gt_xyxy):
    """IoU of every anchor [M,4 xywh] with one GT box [4 xyxy]. Returns [M]."""
    x1 = raw_boxes_xywh[:, 0] - raw_boxes_xywh[:, 2] / 2
    y1 = raw_boxes_xywh[:, 1] - raw_boxes_xywh[:, 3] / 2
    x2 = raw_boxes_xywh[:, 0] + raw_boxes_xywh[:, 2] / 2
    y2 = raw_boxes_xywh[:, 1] + raw_boxes_xywh[:, 3] / 2
    ix1 = np.maximum(x1, gt_xyxy[0]);  iy1 = np.maximum(y1, gt_xyxy[1])
    ix2 = np.minimum(x2, gt_xyxy[2]);  iy2 = np.minimum(y2, gt_xyxy[3])
    inter  = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (x2 - x1) * (y2 - y1)
    area_g = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])
    union  = area_a + area_g - inter
    return np.where(union > 0, inter / union, 0.0)


def extract_entry_features(entry):
    """
    Returns a flat dict of [M]-length arrays for each pre-NMS anchor,
    restricted to anchors with nonzero IoU with the GT box so that
    unrelated objects / background regions are excluded.

    All three geometric features are normalised by the corresponding patch
    dimension so results are entry-independent and directly comparable:

      dist  → divided by patch diagonal  sqrt(gt_w² + gt_h²)
               → 1.0 means the anchor centre is one diagonal-length away
      ar    → divided by patch aspect ratio  gt_w / gt_h
               → 1.0 means the anchor has the same w/h as the patch
      size  → divided by patch size  (gt_w + gt_h) / 2
               → 1.0 means the anchor is the same size as the patch

    raw_boxes are xywh centre-format pixel coords.
    gt_box is [x1, y1, x2, y2] pixel coords.
    """
    gt    = entry['gt_box'][0, :4]            # [x1, y1, x2, y2]
    gt_cx = (gt[0] + gt[2]) / 2
    gt_cy = (gt[1] + gt[3]) / 2
    gt_w  = gt[2] - gt[0]
    gt_h  = gt[3] - gt[1]
    gt_diag = np.sqrt(gt_w ** 2 + gt_h ** 2) + 1e-6
    gt_ar   = gt_w / (gt_h + 1e-6)
    gt_size = (gt_w + gt_h) / 2 + 1e-6

    raw = entry['raw_boxes']                  # [M, 4] xywh centre-format

    # Keep only anchors that overlap with the GT box
    iou  = _iou_with_gt(raw, gt)
    keep = iou >= 0
    raw  = raw[keep]
    cx, cy = raw[:, 0], raw[:, 1]
    w,  h  = raw[:, 2], raw[:, 3]

    obj   = entry['raw_obj'][keep]            # [M']
    maxp  = entry['raw_cls'][keep].max(axis=1)

    return {
        'dist':  np.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2) / gt_diag,
        'ar':    (w / np.maximum(h, 1e-6)) / gt_ar,
        'size':  ((w + h) / 2) / gt_size,
        'obj':   obj,
        'maxp':  maxp,
        'score': obj * maxp,
    }


# ---------------------------------------------------------------------------
# Quantile-based bin-edge computation
# ---------------------------------------------------------------------------

def compute_bin_edges(results_list, n_bins=10):
    """
    Pool anchor feature values from every entry in every results list, then
    compute quantile-based bin edges for each geometric feature.

    Using the same edges for patched and original results ensures the bars
    are directly comparable.

    Returns {feat_key: np.ndarray of shape [k+1]}  (k ≤ n_bins).
    """
    pooled = {fk: [] for fk, _, _ in FEATURES}
    for results in results_list:
        for entry in results:
            feats = extract_entry_features(entry)
            for fk in pooled:
                arr = feats[fk]
                if arr.size > 0:
                    pooled[fk].append(arr)

    edges = {}
    for fk, feat_label, _ in FEATURES:
        vals = np.concatenate(pooled[fk]) if pooled[fk] else np.array([0.0, 1.0])
        vals = vals[np.isfinite(vals)]
        qs   = np.linspace(0, 100, n_bins + 1)
        raw  = np.percentile(vals, qs)
        uniq = np.unique(raw)
        # Guard against degenerate data collapsing all quantiles to one value
        edges[fk] = uniq if len(uniq) >= 2 else np.array([vals.min(), vals.max() + 1e-6])
    return edges


def _fmt_edge(v):
    """Compact number format for bin-edge labels."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    elif abs(v) >= 1:
        return f"{v:.2f}"
    else:
        return f"{v:.3f}"


def make_bin_labels(edges):
    """List of 'lo–hi' strings, one per bin."""
    return [f"{_fmt_edge(edges[i])}–{_fmt_edge(edges[i+1])}"
            for i in range(len(edges) - 1)]


def _assign_bins(vals, edges):
    """
    Map each value to a bin index in [0, n_bins-1].
    Uses np.digitize so bins are left-inclusive, right-exclusive;
    the last bin includes its right endpoint.
    Values outside [edges[0], edges[-1]] are clamped to the boundary bins.
    """
    n_bins = len(edges) - 1
    return np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)


# ---------------------------------------------------------------------------
# Per-entry percentage arrays
# ---------------------------------------------------------------------------

def entry_feat_pcts(entry, feat_key, edges):
    """[n_bins] percentage array (sums to 100) for one entry and one feature."""
    vals   = extract_entry_features(entry)[feat_key]
    n_bins = len(edges) - 1
    idx    = _assign_bins(vals, edges)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    total  = counts.sum()
    return (counts / total * 100.0) if total > 0 else counts


def collect_feat_distributions(results, bin_edges):
    """
    Returns (succ_pcts, fail_pcts), each {feat_key: [N, n_bins] array}.
    Splits by entry['attack_success'].
    """
    succ = {fk: [] for fk, _, _ in FEATURES}
    fail = {fk: [] for fk, _, _ in FEATURES}
    for entry in results:
        dst = succ if entry['attack_success'] else fail
        for fk, _, _ in FEATURES:
            dst[fk].append(entry_feat_pcts(entry, fk, bin_edges[fk]))

    def to_arr(lst, nb):
        return np.array(lst) if lst else np.empty((0, nb))

    s, f = {}, {}
    for fk, _, _ in FEATURES:
        nb = len(bin_edges[fk]) - 1
        s[fk] = to_arr(succ[fk], nb)
        f[fk] = to_arr(fail[fk], nb)
    return s, f


def collect_all_feat_pcts(results, bin_edges):
    """Returns {feat_key: [N, n_bins]} for all entries (no split)."""
    rows = {fk: [] for fk, _, _ in FEATURES}
    for entry in results:
        for fk, _, _ in FEATURES:
            rows[fk].append(entry_feat_pcts(entry, fk, bin_edges[fk]))

    def to_arr(lst, nb):
        return np.array(lst) if lst else np.empty((0, nb))

    return {fk: to_arr(rows[fk], len(bin_edges[fk]) - 1) for fk, _, _ in FEATURES}


# ---------------------------------------------------------------------------
# Per-bin score statistics
# ---------------------------------------------------------------------------

def _empty_bucket():
    return {fk: {mk: {'min': [], 'max': []}
                 for _, mk in METRICS}
            for fk, _, _ in FEATURES}


def _stats_into_bucket(bucket, entry, bin_edges):
    """Accumulate per-bin min/max score stats for one entry into bucket."""
    feats = extract_entry_features(entry)
    for fk, _, _ in FEATURES:
        edges  = bin_edges[fk]
        n_bins = len(edges) - 1
        idx    = _assign_bins(feats[fk], edges)
        for _, mk in METRICS:
            mvals   = feats[mk]
            row_min = np.full(n_bins, np.nan)
            row_max = np.full(n_bins, np.nan)
            for b in range(n_bins):
                mask = idx == b
                if mask.any():
                    row_min[b] = mvals[mask].min()
                    row_max[b] = mvals[mask].max()
            bucket[fk][mk]['min'].append(row_min)
            bucket[fk][mk]['max'].append(row_max)


def _finalize_bucket(bucket, bin_edges):
    def to_arr(lst, nb):
        return np.array(lst) if lst else np.full((0, nb), np.nan)
    for fk, _, _ in FEATURES:
        nb = len(bin_edges[fk]) - 1
        for _, mk in METRICS:
            bucket[fk][mk]['min'] = to_arr(bucket[fk][mk]['min'], nb)
            bucket[fk][mk]['max'] = to_arr(bucket[fk][mk]['max'], nb)


def collect_feat_score_stats(results, bin_edges):
    """
    Returns (succ_stats, fail_stats).
    Each: {feat_key: {metric_key: {'min': [N,nb], 'max': [N,nb]}}}
    """
    succ, fail = _empty_bucket(), _empty_bucket()
    for entry in results:
        _stats_into_bucket(succ if entry['attack_success'] else fail, entry, bin_edges)
    _finalize_bucket(succ, bin_edges)
    _finalize_bucket(fail, bin_edges)
    return succ, fail


def collect_all_feat_score_stats(results, bin_edges):
    """Single-bucket (all entries) version of collect_feat_score_stats."""
    bucket = _empty_bucket()
    for entry in results:
        _stats_into_bucket(bucket, entry, bin_edges)
    _finalize_bucket(bucket, bin_edges)
    return bucket


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _nanmean(arr, nb):
    return np.nanmean(arr, axis=0) if arr.shape[0] > 0 else np.full(nb, np.nan)


def _fv(x):
    return f"{x:.4f}" if not np.isnan(x) else "  n/a  "


def print_feat_score_stats(succ_stats, fail_stats, bin_edges):
    """
    For each feature and each metric, print a table of avg-min / avg-max
    separated by attack success vs failure.
    """
    for fk, feat_label, _ in FEATURES:
        labels = make_bin_labels(bin_edges[fk])
        nb     = len(labels)
        ns     = succ_stats[fk]['obj']['min'].shape[0]
        nf     = fail_stats[fk]['obj']['min'].shape[0]
        col    = 18
        for mlabel, mk in METRICS:
            sm = _nanmean(succ_stats[fk][mk]['min'], nb)
            sx = _nanmean(succ_stats[fk][mk]['max'], nb)
            fm = _nanmean(fail_stats[fk][mk]['min'], nb)
            fx = _nanmean(fail_stats[fk][mk]['max'], nb)
            print(f"\n  {feat_label} | {mlabel}  (n_success={ns}, n_failure={nf})")
            print(f"  {'Bin':<22}  "
                  f"{'succ avg-min':>{col}}  {'succ avg-max':>{col}}  "
                  f"{'fail avg-min':>{col}}  {'fail avg-max':>{col}}")
            print("  " + "-" * (22 + 4 * (col + 2) + 4))
            for i, lbl in enumerate(labels):
                print(f"  {lbl:<22}  "
                      f"{_fv(sm[i]):>{col}}  {_fv(sx[i]):>{col}}  "
                      f"{_fv(fm[i]):>{col}}  {_fv(fx[i]):>{col}}")
    print()


def print_feat_table(succ_pcts, fail_pcts, bin_edges):
    for fk, feat_label, _ in FEATURES:
        labels = make_bin_labels(bin_edges[fk])
        sp, fp = succ_pcts[fk], fail_pcts[fk]
        ns, nf = sp.shape[0], fp.shape[0]
        ms = sp.mean(axis=0) if ns > 0 else np.zeros(len(labels))
        mf = fp.mean(axis=0) if nf > 0 else np.zeros(len(labels))
        ss = sp.std(axis=0)  if ns > 0 else np.zeros(len(labels))
        sf = fp.std(axis=0)  if nf > 0 else np.zeros(len(labels))
        print(f"\n  {feat_label}  (n_success={ns}, n_failure={nf})")
        print(f"  {'Bin':<22}  {'success mean±std':>22}  {'failure mean±std':>22}")
        print("  " + "-" * 72)
        for i, lbl in enumerate(labels):
            print(f"  {lbl:<22}  {ms[i]:6.2f} ± {ss[i]:5.2f}              "
                  f"{mf[i]:6.2f} ± {sf[i]:5.2f}")
    print()


def print_feat_score_stats_patch_vs_orig(patch_stats, orig_stats, bin_edges):
    """
    For each feature and each metric, print a table of avg-min / avg-max
    for w/ patch vs w/o patch.
    """
    for fk, feat_label, _ in FEATURES:
        labels = make_bin_labels(bin_edges[fk])
        nb     = len(labels)
        n_p    = patch_stats[fk]['obj']['min'].shape[0]
        n_o    = orig_stats[fk]['obj']['min'].shape[0]
        col    = 20
        for mlabel, mk in METRICS:
            pm = _nanmean(patch_stats[fk][mk]['min'], nb)
            px = _nanmean(patch_stats[fk][mk]['max'], nb)
            om = _nanmean(orig_stats[fk][mk]['min'],  nb)
            ox = _nanmean(orig_stats[fk][mk]['max'],  nb)
            print(f"\n  {feat_label} | {mlabel}  (n_patch={n_p}, n_orig={n_o})")
            print(f"  {'Bin':<22}  "
                  f"{'w/patch avg-min':>{col}}  {'w/patch avg-max':>{col}}  "
                  f"{'w/o patch avg-min':>{col}}  {'w/o patch avg-max':>{col}}")
            print("  " + "-" * (22 + 4 * (col + 2) + 4))
            for i, lbl in enumerate(labels):
                print(f"  {lbl:<22}  "
                      f"{_fv(pm[i]):>{col}}  {_fv(px[i]):>{col}}  "
                      f"{_fv(om[i]):>{col}}  {_fv(ox[i]):>{col}}")
    print()


def print_feat_table_patch_vs_orig(patch_pcts, orig_pcts, bin_edges):
    for fk, feat_label, _ in FEATURES:
        labels   = make_bin_labels(bin_edges[fk])
        pp, op   = patch_pcts[fk], orig_pcts[fk]
        n_p, n_o = pp.shape[0], op.shape[0]
        mp = pp.mean(axis=0) if n_p > 0 else np.zeros(len(labels))
        mo = op.mean(axis=0) if n_o > 0 else np.zeros(len(labels))
        sp = pp.std(axis=0)  if n_p > 0 else np.zeros(len(labels))
        so = op.std(axis=0)  if n_o > 0 else np.zeros(len(labels))
        print(f"\n  {feat_label}  (n_patch={n_p}, n_orig={n_o})")
        print(f"  {'Bin':<22}  {'w/ patch mean±std':>22}  {'w/o patch mean±std':>22}")
        print("  " + "-" * 72)
        for i, lbl in enumerate(labels):
            print(f"  {lbl:<22}  {mp[i]:6.2f} ± {sp[i]:5.2f}              "
                  f"{mo[i]:6.2f} ± {so[i]:5.2f}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_bars(ax, x, mean_a, mean_b, std_a, std_b,
               label_a, label_b, color_a, color_b, show_std):
    n_bins = len(x)
    width  = 0.38
    offset = width / 2
    err_kw = dict(capsize=4, capthick=1.2, elinewidth=1.2)
    ax.bar(x - offset, mean_a, width, color=color_a, alpha=0.82, label=label_a,
           yerr=(std_a if show_std else None), error_kw=err_kw)
    ax.bar(x + offset, mean_b, width, color=color_b, alpha=0.82, label=label_b,
           yerr=(std_b if show_std else None), error_kw=err_kw)
    ax.set_xlim(-0.6, n_bins - 0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)


def plot_feat_distribution(succ_pcts, fail_pcts, bin_edges,
                           show_std=True, out_prefix=None):
    """One figure per feature: attack success (green) vs failure (red)."""
    for fk, xlabel, title in FEATURES:
        labels = make_bin_labels(bin_edges[fk])
        nb     = len(labels)
        sp, fp = succ_pcts[fk], fail_pcts[fk]
        ns, nf = sp.shape[0], fp.shape[0]

        mean_s = sp.mean(axis=0) if ns > 0 else np.zeros(nb)
        mean_f = fp.mean(axis=0) if nf > 0 else np.zeros(nb)
        std_s  = sp.std(axis=0)  if ns > 0 else np.zeros(nb)
        std_f  = fp.std(axis=0)  if nf > 0 else np.zeros(nb)

        fig, ax = plt.subplots(figsize=(max(10, nb * 1.1), 5))
        _draw_bars(ax, np.arange(nb),
                   mean_s, mean_f, std_s, std_f,
                   f'Attack success  (n={ns})', f'Attack failure  (n={nf})',
                   'seagreen', 'tomato', show_std)
        ax.set_xticks(np.arange(nb))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Mean % of anchors per entry', fontsize=11)
        ax.set_title(
            f'{title}  —  success vs failure\n'
            + (f'Error bars: ±1 std  (n_success={ns}, n_failure={nf})'
               if show_std else f'n_success={ns},  n_failure={nf}'),
            fontsize=12,
        )
        plt.tight_layout()
        if out_prefix:
            p = f"{out_prefix}_{fk}_succ_vs_fail.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved → {p}")
        else:
            plt.show()
        plt.close(fig)


def plot_feat_patch_vs_orig(patch_pcts, orig_pcts, bin_edges,
                            show_std=True, out_prefix=None):
    """One figure per feature: w/ patch (green) vs w/o patch (steelblue)."""
    for fk, xlabel, title in FEATURES:
        labels   = make_bin_labels(bin_edges[fk])
        nb       = len(labels)
        pp, op   = patch_pcts[fk], orig_pcts[fk]
        n_p, n_o = pp.shape[0], op.shape[0]

        mean_p = pp.mean(axis=0) if n_p > 0 else np.zeros(nb)
        mean_o = op.mean(axis=0) if n_o > 0 else np.zeros(nb)
        std_p  = pp.std(axis=0)  if n_p > 0 else np.zeros(nb)
        std_o  = op.std(axis=0)  if n_o > 0 else np.zeros(nb)

        fig, ax = plt.subplots(figsize=(max(10, nb * 1.1), 5))
        _draw_bars(ax, np.arange(nb),
                   mean_p, mean_o, std_p, std_o,
                   f'w/ patch  (n={n_p})', f'w/o patch — original  (n={n_o})',
                   'seagreen', 'steelblue', show_std)
        ax.set_xticks(np.arange(nb))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Mean % of anchors per entry', fontsize=11)
        ax.set_title(
            f'{title}  —  w/ patch vs w/o patch\n'
            + (f'Error bars: ±1 std  (n_patch={n_p}, n_orig={n_o})'
               if show_std else f'n_patch={n_p},  n_orig={n_o}'),
            fontsize=12,
        )
        plt.tight_layout()
        if out_prefix:
            p = f"{out_prefix}_{fk}_patch_vs_orig.png"
            plt.savefig(p, dpi=150, bbox_inches='tight')
            print(f"Saved → {p}")
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Bounding-box shape distribution of pre-NMS anchors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('pkl', help='Path to *_results.pkl from demo.py eval()')
    parser.add_argument('--orig', default=None,
                        help='Path to *_results_orig.pkl (auto-detected if omitted)')
    parser.add_argument('--out', default=None,
                        help='Output filename prefix for PNGs (default: show interactively)')
    parser.add_argument('--no-std', action='store_true',
                        help='Hide std error bars')
    parser.add_argument('--bins', type=int, default=10,
                        help='Number of quantile bins (default: 10)')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    n_total   = len(results)
    n_success = sum(r['attack_success'] for r in results)
    print(f"Loaded {n_total} entries  |  "
          f"success: {n_success}  failure: {n_total - n_success}")

    # Auto-detect orig pkl
    orig_pkl = args.orig or args.pkl.replace('_results', '_results_orig')
    results_orig = None
    if os.path.exists(orig_pkl):
        with open(orig_pkl, 'rb') as f:
            results_orig = pickle.load(f)
        print(f"Loaded {len(results_orig)} original (no-patch) entries from {orig_pkl!r}")
    else:
        msg = f"--orig path not found: {orig_pkl!r}" if args.orig else \
              f"no orig pkl found at {orig_pkl!r} — patch-vs-orig comparison will be skipped"
        print(f"[patch-vs-orig] {msg}")

    # Compute shared quantile bin edges from ALL available data
    all_results_lists = [results] + ([results_orig] if results_orig else [])
    bin_edges = compute_bin_edges(all_results_lists, n_bins=args.bins)

    print("\nBin edges:")
    for fk, feat_label, _ in FEATURES:
        print(f"  {feat_label}: {make_bin_labels(bin_edges[fk])}")

    out_prefix = os.path.splitext(args.out)[0] if args.out else None

    # --- Success vs failure (patched results) ---
    print("\n=== Success vs Failure (patched results) ===")
    succ_stats, fail_stats = collect_feat_score_stats(results, bin_edges)
    print_feat_score_stats(succ_stats, fail_stats, bin_edges)

    succ_pcts, fail_pcts = collect_feat_distributions(results, bin_edges)
    print_feat_table(succ_pcts, fail_pcts, bin_edges)
    plot_feat_distribution(succ_pcts, fail_pcts, bin_edges,
                           show_std=not args.no_std,
                           out_prefix=out_prefix)

    if results_orig is None:
        return

    # --- Patch vs original ---
    print("\n=== w/ patch (all entries) vs w/o patch ===")
    patch_stats = collect_all_feat_score_stats(results,      bin_edges)
    orig_stats  = collect_all_feat_score_stats(results_orig, bin_edges)
    print_feat_score_stats_patch_vs_orig(patch_stats, orig_stats, bin_edges)

    patch_pcts = collect_all_feat_pcts(results,      bin_edges)
    orig_pcts  = collect_all_feat_pcts(results_orig, bin_edges)
    print_feat_table_patch_vs_orig(patch_pcts, orig_pcts, bin_edges)
    plot_feat_patch_vs_orig(patch_pcts, orig_pcts, bin_edges,
                            show_std=not args.no_std,
                            out_prefix=out_prefix)


if __name__ == '__main__':
    main()
