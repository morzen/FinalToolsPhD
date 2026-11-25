#!/usr/bin/env python3
"""
make_graphs.py -- the figures of the feature-combination sweep, drawn from tables/sweep_metrics.csv
(written by analyse_sweep.py).  Every number on a figure is read from that file; nothing is typed.

  python make_graphs.py --metrics ../tables/sweep_metrics.csv --out ../figures

Figures (PDF, vector, and PNG at 300 dpi; width = the thesis text width)
  sweep_precision_by_size          stress-test Top-1 of every feature combination, by number of feature types,
                                   one panel per approach (the eight of the thesis tables): the variation of the
                                   precision across the combinations
  sweep_precision_by_size_<metric> the same for sanity-check Top-1, stress-test Top-5 and the CTI score
  sweep_precision_by_size_hybrids  the same for the three hybrids (Chapter 5)
  sweep_family_effect(_hybrids)    change in stress-test Top-1 when one feature type is added to a combination
  sweep_overview(_sanity, _cti)    every combination x every approach and hybrid, values printed (the table view):
                                   stress-test Top-1, sanity-check Top-1, CTI score (27 reports, combinations with TTPs)
  sweep_all_in_one(_sanity, _cti)  everything in one graph: one row per approach and per weighting of the Hybrid
                                   Attribution Framework (Equal, Self-eval, CTI-based), one dot per combination coloured
                                   by its number of feature types, the middle half (box), the median, the TTP-only,
                                   three- and six-feature combinations, and the lowest-highest value of each row
  Approaches whose combinations are not all run yet show the ones done ("n/63 so far" in the panel title).
  ttp_vs_3f_6f                     TTP-only against the thesis' three- and six-feature tools, every method and hybrid
  ttp_vs_3f_6f_methods / _hybrids  the same, split: the eight methods (Chapter 4) and the three hybrids (Chapter 5)
                                   A light band behind a marker = the range over the trainings of that tool, when it
                                   was trained more than once (run_neural_sweep.sh with REPEATS, CONFIGS=6f).
"""
import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

# ---- style: reference palette of the dataviz method (first three categorical slots validated all-pairs on white)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, BAND = '#e1e0d9', '#c3c2b7', '#f5f5f3'
SURFACE = '#ffffff'
SEQ = ['#f3f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # one hue, light->dark
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'text.color': INK,
})
TEXT_W = 5.95          # inches: the thesis text width (428 pt)

FAMILIES = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
FAM_LABEL = {'techniques': 'TTPs', 'software': 'Software', 'capec': 'CAPEC', 'cwe': 'CWE', 'cve': 'CVE',
             'pulse': 'Pulse'}
LETTER = {'techniques': 'T', 'software': 'S', 'capec': 'A', 'cwe': 'W', 'cve': 'V', 'pulse': 'P'}
# the names of the thesis tables: the three WRRF weightings of the Hybrid Attribution Framework are Equal, Self-eval
# and CTI-based (hybrid_no_weights, hybrid_self_eval_weights and hybrid_CTI_boosting in the code)
METHOD_LABEL = {'xgboost': 'XGBoost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'Random Forest',
                'jaccard': 'Jaccard', 'tvd': 'TVD', 'featureMatching': 'Feature Matching',
                'hybrid_no_weights': 'Hybrid (equal)', 'hybrid_self_eval_weights': 'Hybrid (self-eval)',
                'hybrid_CTI_boosting': 'Hybrid (CTI-based)'}
ALL_METHODS = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'jaccard', 'tvd', 'featureMatching',
               'hybrid_no_weights', 'hybrid_self_eval_weights', 'hybrid_CTI_boosting']
EIGHT = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'featureMatching', 'jaccard', 'tvd']     # the thesis tables' order
HYBRIDS = ['hybrid_no_weights', 'hybrid_self_eval_weights', 'hybrid_CTI_boosting']
GROUPS = [('ML models', ['xgboost', 'gnn', 'siamese', 'mil', 'rf']),
          ('Mathematical algorithm', ['featureMatching', 'jaccard', 'tvd']), ('Hybrid Attribution Framework', HYBRIDS)]
SHORT = {'xgboost': 'XGBoost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'RF',
         'featureMatching': 'Feature\nMatching', 'jaccard': 'Jaccard', 'tvd': 'TVD', 'hybrid_no_weights': 'Equal',
         'hybrid_self_eval_weights': 'Self-eval', 'hybrid_CTI_boosting': 'CTI-based'}
# fig_all_in_one: one row per approach, named as in the columns of the thesis tables
ROW_LABEL = {'xgboost': 'XGBoost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'RF',
             'featureMatching': 'Feature Matching', 'jaccard': 'Jaccard', 'tvd': 'TVD', 'hybrid_no_weights': 'Equal',
             'hybrid_self_eval_weights': 'Self-eval', 'hybrid_CTI_boosting': 'CTI-based'}
# colour of a combination = its number of feature types: one blue, light -> dark, five steps (1, 2, 3, 4, 5 or 6; the
# only six-type combination, TSAWVP, has its own marker).  Ordinal ramp of the dataviz method, checked with its
# validator against white: lightness monotone, adjacent steps >= 0.06 apart, lightest step 2.1:1 on white.
LEVELS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
LEVEL_LABEL = ['1', '2', '3', '4', '5 or 6']
MARKED = {'T': ('o', 'TTP-only (T)'), 'TSV': ('s', 'three feature types (TSV)'), 'TSAWVP': ('D', 'six feature types (all)')}
METRIC_LABEL = {('stress', 'top1'): 'Stress test, Top-1 (%)', ('stress', 'top5'): 'Stress test, Top-5 (%)',
                ('sanity', 'top1'): 'Sanity check, Top-1 (%)', ('cti_ttp', 'cti_score'): 'CTI score (27 TTP reports)'}


def read(path):
    data = defaultdict(dict)           # (source, config, method) -> {evaluation: row}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ('top1', 'top5', 'top10', 'top20', 'mean_rank', 'cti_score'):
                r[k] = float(r[k]) if r.get(k) not in (None, '', 'None') else np.nan
            data[(r['source'], r['config'], r['method'])][r['evaluation']] = r
    return data


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + '.pdf'), metadata={'CreationDate': None})   # no date: same file every run
    fig.savefig(os.path.join(out, name + '.png'), dpi=300)
    plt.close(fig)
    print(f"[graphs] {name}.pdf / .png")


def swarm(ys, ax, diameter_pt, max_off):
    """Horizontal offsets (data units) so that markers of one column do not overlap (a deterministic beeswarm)."""
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xu = w_in / abs(x1 - x0) * 72.0                 # points per x unit
    yu = h_in / abs(y1 - y0) * 72.0                 # points per y unit
    d = diameter_pt * 1.08
    order = np.argsort(ys, kind='mergesort')
    placed, offs = [], np.zeros(len(ys))
    for i in order:
        y = ys[i] * yu
        for k in range(0, 200):
            cand = (k + 1) // 2 * (1 if k % 2 else -1) * d * 0.5
            if all((cand - px) ** 2 + (y - py) ** 2 >= d * d for px, py in placed):
                break
        cand = max(-max_off * xu, min(max_off * xu, cand))
        placed.append((cand, y))
        offs[i] = cand / xu
    return offs


def swarm_flat(xs, ax, diameters_pt, max_off, first=()):
    """Vertical offsets (data units) for the markers of one horizontal row (a deterministic beeswarm on its side).
    diameters_pt: the size of each marker (points); first: indices placed before the others, nearest the middle of
    the row.  Where more markers share a value than the row can hold, they overlap evenly across the row (the
    least-overlapping place) instead of piling up at its edges."""
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xu = w_in / abs(x1 - x0) * 72.0                 # points per x unit
    yu = h_in / abs(y1 - y0) * 72.0                 # points per y unit
    ds = np.asarray(diameters_pt, dtype=float) * 1.08
    step = ds.min() * 0.5
    cap = max_off * yu
    cands = [0.0] + [s * k * step for k in range(1, int(cap // step) + 1) for s in (1, -1)]
    order = sorted(first, key=lambda i: xs[i]) + [i for i in np.argsort(xs, kind='mergesort') if i not in first]
    placed, offs = [], np.zeros(len(xs))
    for i in order:
        x = xs[i] * xu
        near = [(px, py, pd) for px, py, pd in placed if abs(px - x) < (pd + ds[i]) / 2]
        best, best_gap = 0.0, -np.inf
        for c in cands:
            gap = min([np.hypot(x - px, c - py) - (pd + ds[i]) / 2 for px, py, pd in near], default=np.inf)
            if gap >= 0:
                best = c
                break
            if gap > best_gap + 1e-9:
                best, best_gap = c, gap
        placed.append((x, best, ds[i]))
        offs[i] = best / yu
    return offs


def clean(ax, keep_left=True):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)


# ------------------------------------------------------------------------------ panel grid (2 columns)
def panel_grid(n, panel_h, top_in=0.30, bottom_in=None, hspace_in=0.36, left=0.10):
    """n panels in two columns, sharing both axes; returns the figure, the n axes, the panels that carry the x and the
    y labels, and the box of the unused slot (odd n: the legend goes there) or None."""
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    if bottom_in is None:
        bottom_in = 0.95 if n % 2 == 0 else 0.55          # odd n: the legend sits in the free slot, not below
    fig_h = top_in + nrows * panel_h + (nrows - 1) * hspace_in + bottom_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXT_W, fig_h), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=left, right=0.99, top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
                        hspace=hspace_in / panel_h, wspace=0.08)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    bottoms = []
    for c in range(ncols):                      # the lowest visible panel of each column carries the x axis labels
        col = [flat[r * ncols + c] for r in range(nrows) if r * ncols + c < n]
        for ax in col[:-1]:
            ax.tick_params(labelbottom=False)
        if col:
            col[-1].tick_params(labelbottom=True)
            bottoms.append(col[-1])
    lefts = [flat[r * ncols] for r in range(nrows) if r * ncols < n]
    free = flat[n].get_position() if n < len(flat) else None
    return fig, flat[:n], bottoms, lefts, free


def place_legend(fig, handles, free, ncol, fontsize):
    """Below the panels, or in the free slot of an odd number of panels."""
    if free is None:
        fig.legend(handles=handles, loc='lower center', ncol=ncol, bbox_to_anchor=(0.54, 0.0), fontsize=fontsize,
                   handletextpad=0.3, columnspacing=1.2)
    else:
        fig.legend(handles=handles, loc='center', ncol=1, fontsize=fontsize, handletextpad=0.5, labelspacing=0.7,
                   bbox_to_anchor=((free.x0 + free.x1) / 2, (free.y0 + free.y1) / 2))


# ------------------------------------------------------------------------------ figure 1: variation by size
def fig_by_size(data, source, methods, metric, out, name):
    """One panel per approach: every feature combination as a dot, placed by its number of feature types."""
    ev, field = metric
    fig, axes, bottoms, lefts, free = panel_grid(len(methods), 1.40 if len(methods) > 4 else 1.60)
    seen = set()
    for ax in axes:
        ax.set_xlim(0.45, 6.55)
        ax.set_ylim(-3, 103)
    for ax, m in zip(axes, methods):
        cols = defaultdict(list)
        for (src, key, meth), evs in data.items():
            if src != source or meth != m or ev not in evs:
                continue
            if ev.startswith('cti') and 'T' not in key:
                continue                      # the 27 CTI reports are TTP reports: combinations with TTPs only
            v = evs[ev][field]
            if not np.isnan(v):
                cols[len(key)].append((key, v))
        for n, lst in cols.items():
            ys = np.array([v for _, v in lst])
            offs = swarm(ys, ax, 4.4, 0.44)
            for (key, v), dx in zip(lst, offs):
                col = BLUE if 'T' in key else ORANGE
                seen.add(col)
                if key in MARKED:
                    ax.scatter(n + dx, v, s=40, marker=MARKED[key][0], color=col, edgecolor=INK, linewidth=1.0, zorder=4)
                else:
                    ax.scatter(n + dx, v, s=17, color=col, edgecolor=SURFACE, linewidth=0.5, zorder=3)
            if len(ys) > 1:                   # the mean of the combinations with n feature types
                ax.plot([n - 0.46, n + 0.46], [ys.mean()] * 2, color=INK, lw=1.5, zorder=3.5, solid_capstyle='butt')
        n_done = sum(len(v) for v in cols.values())
        n_all = 32 if ev.startswith('cti') else 63
        ax.set_title(METHOD_LABEL[m] + ('' if n_done == n_all else f'  ({n_done}/{n_all} so far)'), loc='left')
        ax.set_xticks(range(1, 7))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='x', visible=False)
        clean(ax)
    for ax in bottoms:
        ax.set_xlabel('Number of feature types in the combination')
    for ax in lefts:
        ax.set_ylabel(METRIC_LABEL[metric])
    handles = [Line2D([], [], marker='o', ls='', color=col, markeredgecolor=SURFACE, markersize=5.5, label=lab)
               for col, lab in ((BLUE, 'combination with TTPs'), (ORANGE, 'combination without TTPs')) if col in seen]
    handles += [Line2D([], [], marker=mk, ls='', color=SURFACE, markeredgecolor=INK, markeredgewidth=1.0, markersize=6.5,
                       label=lab) for mk, lab in MARKED.values()]
    handles.append(Line2D([], [], color=INK, lw=1.5, label='mean for that number of types'))
    place_legend(fig, handles, free, 3, 7.5)
    save(fig, out, name)


# ------------------------------------------------------------------------------ figure 2: effect of adding a type
def fig_family_effect(data, source, methods, out, name, ev='stress', field='top1'):
    vals = {}
    for (src, key, meth), evs in data.items():
        if src == source and ev in evs:
            vals[(meth, key)] = evs[ev][field]
    deltas_all = {}
    lo, hi = 0.0, 0.0
    for m in methods:
        for f in FAMILIES:
            d = []
            for (meth, key), v in vals.items():
                if meth != m or LETTER[f] in key:
                    continue
                plus = ''.join(c for c in 'TSAWVP' if c in key + LETTER[f])
                if (m, plus) in vals and not np.isnan(v) and not np.isnan(vals[(m, plus)]):
                    d.append(vals[(m, plus)] - v)
            deltas_all[(m, f)] = np.array(d)
            if len(d):
                lo, hi = min(lo, min(d)), max(hi, max(d))
    fig, axes, bottoms, lefts, free = panel_grid(len(methods), 1.40 if len(methods) > 4 else 1.60, left=0.11)
    for ax in axes:
        ax.set_xlim(-0.6, len(FAMILIES) - 0.4)
        ax.set_ylim(5 * np.floor((lo - 3) / 5), 5 * np.ceil((hi + 3) / 5))
    for ax, m in zip(axes, methods):
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        n_pairs = 0
        for j, f in enumerate(FAMILIES):
            d = deltas_all[(m, f)]
            n_pairs += len(d)
            if len(d) == 0:
                continue
            offs = swarm(d, ax, 3.4, 0.42)
            ax.scatter(j + offs, d, s=11, color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=2)
            med = float(np.median(d))
            ax.plot([j - 0.40, j + 0.40], [med, med], color=INK, lw=1.8, zorder=3, solid_capstyle='round')
        ax.set_title(METHOD_LABEL[m] + ('' if n_pairs == 31 * len(FAMILIES) else f'  ({n_pairs}/186 pairs so far)'),
                     loc='left')
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=7.0)
        ax.grid(axis='x', visible=False)
        clean(ax)
    for ax in bottoms:
        ax.set_xlabel('Feature type added to the combination')
    for ax in lefts:
        ax.set_ylabel('Change in stress-test\nTop-1 (points)')
    pair = ('one pair: a combination without the type and the same combination with it (31 per type)' if free is None
            else 'one pair: a combination without\nthe type and the same combination\nwith it (31 per type)')
    handles = [Line2D([], [], marker='o', ls='', color=BLUE, markeredgecolor=SURFACE, markersize=4.5, label=pair),
               Line2D([], [], color=INK, lw=1.8, label='median')]
    place_legend(fig, handles, free, 2, 7.2)
    save(fig, out, name)
    return {k: (float(np.median(v)), float(v.min()), float(v.max())) for k, v in deltas_all.items() if len(v)}


# ------------------------------------------------------------------------------ figure 3: every combination x approach
def fig_heatmap(data, source, methods, metric, out, name, ttp_only=False):
    """Rows: the feature combinations (black dots: types included), grouped by number of types; columns: the
    approaches (ML models, mathematical algorithms, hybrids); cells: the metric, '-' where there is no value yet."""
    ev, field = metric
    keys = sorted({key for (src, key, meth) in data if src == source},
                  key=lambda k: (len(k), ['TSAWVP'.index(c) for c in k]))
    if ttp_only:
        keys = [k for k in keys if 'T' in k]
    groups = [(g, [m for m in ms if m in methods]) for g, ms in GROUPS]
    groups = [(g, ms) for g, ms in groups if ms]
    row_h, head_h, foot_h = 0.108, 0.95, 0.25
    gaps = sum(1 for i in range(1, len(keys)) if len(keys[i]) != len(keys[i - 1]))
    fig_h = head_h + (len(keys) + gaps * 0.5) * row_h + foot_h
    fig = plt.figure(figsize=(TEXT_W, fig_h))
    fam_x = [0.14 + 0.175 * j for j in range(6)]
    n_cols = sum(len(ms) for _, ms in groups)
    group_gap, x_start, x_end = 0.10, 1.24, TEXT_W - 0.08
    cell_w = (x_end - x_start - group_gap * (len(groups) - 1)) / n_cols
    cell_x, x = {}, x_start
    for gi, (g, ms) in enumerate(groups):
        for m in ms:
            cell_x[m] = x + cell_w / 2
            x += cell_w
        x += group_gap
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_W)
    ax.set_ylim(fig_h, 0)
    ax.axis('off')
    norm = matplotlib.colors.Normalize(0, 100)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', SEQ)
    y, prev = head_h, None
    for key in keys:
        if prev is not None and len(key) != len(prev):
            y += 0.5 * row_h
        yc = y + row_h / 2
        for j, f in enumerate(FAMILIES):
            on = LETTER[f] in key
            ax.add_patch(plt.Circle((fam_x[j], yc), 0.034 if on else 0.018, color=INK if on else AXIS, lw=0))
        for m, xc in cell_x.items():
            r = data.get((source, key, m), {}).get(ev)
            v = r[field] if r else np.nan
            if np.isnan(v):
                ax.text(xc, yc, '–', ha='center', va='center', fontsize=5.5, color=MUTED)
                continue
            ax.add_patch(plt.Rectangle((xc - cell_w / 2 + 0.012, y + 0.010), cell_w - 0.024, row_h - 0.020,
                                       color=cmap(norm(v)), lw=0))
            ax.text(xc, yc + 0.003, f'{v:.0f}', ha='center', va='center', fontsize=5.4,
                    color=SURFACE if v >= 60 else INK)
        y += row_h
        prev = key
    for j, f in enumerate(FAMILIES):
        ax.text(fam_x[j], head_h - 0.06, FAM_LABEL[f], rotation=90, ha='center', va='bottom', fontsize=6.3, color=INK2)
    wrap = {'Mathematical algorithm': 'Mathematical\nalgorithm', 'Hybrid Attribution Framework': 'Hybrid Attribution\nFramework'}
    for g, ms in groups:
        xs = [cell_x[m] for m in ms]
        ax.text((xs[0] + xs[-1]) / 2, head_h - 0.42, wrap.get(g, g) if len(ms) < 4 else g, ha='center', va='bottom',
                fontsize=7.4, fontweight='bold', linespacing=1.0)
        ax.plot([xs[0] - cell_w / 2 + 0.02, xs[-1] + cell_w / 2 - 0.02], [head_h - 0.39, head_h - 0.39], color=AXIS, lw=0.7)
        for m in ms:
            ax.text(cell_x[m], head_h - 0.06, SHORT[m], ha='center', va='bottom', fontsize=5.8, color=INK2, linespacing=1.0)
    what = {('stress', 'top1'): 'stress-test Top-1 (%)', ('sanity', 'top1'): 'sanity-check Top-1 (%)',
            ('stress', 'top5'): 'stress-test Top-5 (%)', ('cti_ttp', 'cti_score'): 'CTI score on the 27 reports with TTPs'}
    ax.text(0.05, fig_h - 0.12, f'Black dot: feature type in the combination. Values: {what[metric]}; "–": no result.',
            fontsize=5.8, color=INK2, va='center')
    save(fig, out, name)


# ------------------------------------------------------------------------------ figure 4: TTP-only vs 3f vs 6f
def tint(color, share=0.55):
    """The colour mixed with white (share of the colour, the rest white): a band drawn behind a marker of that colour."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(share * c + (1 - share) * 255):02x}' for c in rgb)


def fig_ttp_compare(data, sources, out, name, only=None, runs=None, xlim=(-3, 103), xticks=(0, 25, 50, 75, 100)):
    """sources: [(label, source, config, color, marker)], one marker each: the TTP-only tool (its first training) and
    the thesis' three- and six-feature tools (the thesis models).
    runs: {index in sources: [(source, config), ...]}: further trainings of that tool.  The range over all its
    trainings (the marker's included) is drawn as a light band of the marker's colour behind the marker; the band is
    hidden by the marker where the trainings differ by less than the width of the marker.
    only: restrict to these methods.  xlim / xticks: the x axis."""
    runs = runs or {}
    panels = [('sanity', 'top1', 'Sanity check, Top-1 (%)'), ('stress', 'top1', 'Stress test, Top-1 (%)'),
              ('cti_ttp', 'cti_score', 'CTI score (27 TTP reports)')]
    methods = [m for m in ALL_METHODS if (only is None or m in only)
               and any((s, c, m) in data for _, s, c, _, _ in sources)]
    banded = [si for si in range(len(sources)) if runs.get(si)]
    legend_rows = 3 if banded else 1
    fig_h = 0.30 * len(methods) + (1.0 if len(methods) > 4 else 1.25) + 0.16 * (legend_rows - 2)
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, fig_h), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.985, top=1 - 0.35 / fig_h, bottom=(0.30 + 0.16 * legend_rows) / fig_h,
                        wspace=0.10)
    dodge = {0: -0.22, 1: 0.0, 2: 0.22}

    def value(s, c, m, ev, field):
        r = data.get((s, c, m), {}).get(ev)
        return None if r is None or np.isnan(r[field]) else r[field]

    for ax, (ev, field, title) in zip(axes, panels):
        for i, m in enumerate(methods):
            xs = [x for x in (value(s, c, m, ev, field) for _, s, c, _, _ in sources) if x is not None]
            if len(xs) > 1:
                ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=1.2, zorder=1)
            for si in banded:
                _, s, c, col, _ = sources[si]
                vals = [x for x in (value(rs, rc, m, ev, field) for rs, rc in [(s, c)] + runs[si]) if x is not None]
                if len(vals) > 1 and max(vals) > min(vals):
                    ax.plot([min(vals), max(vals)], [i + dodge[si]] * 2, color=tint(col), lw=4.0, zorder=2,
                            solid_capstyle='butt')
        for si, (label, s, c, col, mk) in enumerate(sources):
            for i, m in enumerate(methods):
                x = value(s, c, m, ev, field)
                if x is not None:
                    ax.scatter(x, i + dodge[si], s=24, marker=mk, color=col, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.set_title(title, loc='left', fontsize=8)
        ax.set_xlim(*xlim)
        ax.set_xticks(list(xticks))
        ax.grid(axis='y', visible=False)
        clean(ax, keep_left=False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([METHOD_LABEL[m] for m in methods])
    axes[0].set_ylim(len(methods) - 0.5, -0.6)
    handles = [Line2D([], [], marker=mk, ls='', color=col, markeredgecolor=SURFACE, markersize=6, label=label)
               for label, _, _, col, mk in sources]
    if banded:                      # column 1: the markers; column 2: the range bands (legend columns fill top-down)
        bands = [Line2D([], [], color=tint(sources[si][3]), lw=4.0, solid_capstyle='butt',
                        label=f'{sources[si][0].replace(" (thesis tool)", "").replace(" (retrained)", "")}: '
                              f'range of {1 + len(runs[si])} trainings')
                 for si in banded]
        blank = [Line2D([], [], ls='', marker='', label=' ') for _ in range(3 - len(bands))]
        handles = handles + bands + blank
    fig.legend(handles=handles, loc='lower center', ncol=2 if banded else 3, bbox_to_anchor=(0.58, 0.0),
               fontsize=7.2)
    save(fig, out, name)


# ------------------------------------------------------------------------------ figure 5: everything in one panel
def fig_all_in_one(data, source, methods, metric, out, name, ttp_only=False):
    """One row per approach (the eight of the thesis tables, then the three weightings of the Hybrid Attribution
    Framework, grouped as in the tables), one dot per feature combination coloured by its number of feature types.
    Grey box: the middle half of the combinations (25th to 75th percentile); black line: their median; outlined
    markers: the TTP-only (T), three-feature (TSV) and six-feature (TSAWVP) combinations; right: lowest-highest value.
    ttp_only: only the combinations with TTPs (the CTI score: the 27 reports are TTP reports)."""
    ev, field = metric
    groups = [(g, [m for m in ms if m in methods]) for g, ms in GROUPS]
    groups = [(g, ms) for g, ms in groups if ms]
    head, gap = 0.95, 0.40                          # in rows: the line of a group's name, the space between groups
    rows, heads, spans, y = {}, [], [], 0.0
    for gi, (g, ms) in enumerate(groups):
        y += gap if gi else 0.0
        heads.append((g, y + head * 0.55))
        y += head
        spans.append((y, y + len(ms)))             # the rows of the group: the gridlines are drawn there only
        for m in ms:
            rows[m] = y + 0.5
            y += 1.0
    total = y
    row_in, legend_in, axis_in, top_in = 0.29, 0.50, 0.40, 0.06
    left_in, right_in = 1.02, 0.66
    plot_h = total * row_in
    fig_h = top_in + legend_in + plot_h + axis_in
    plot_w = TEXT_W - left_in - right_in
    fig = plt.figure(figsize=(TEXT_W, fig_h))
    ax = fig.add_axes([left_in / TEXT_W, axis_in / fig_h, plot_w / TEXT_W, plot_h / fig_h])
    ax.set_xlim(-1.5, 101.5)
    ax.set_ylim(total, 0)
    n_all = 32 if ttp_only else 63
    counts = {}
    for m, yr in rows.items():
        pts = []
        for (src, key, meth), evs in data.items():
            if src != source or meth != m or ev not in evs or (ttp_only and 'T' not in key):
                continue
            v = evs[ev][field]
            if not np.isnan(v):
                pts.append((key, v))
        counts[m] = len(pts)
        if not pts:
            continue
        pts.sort(key=lambda kv: (len(kv[0]), ['TSAWVP'.index(c) for c in kv[0]]))
        xs = np.array([v for _, v in pts])
        q1, med, q3 = np.percentile(xs, [25, 50, 75])
        ax.add_patch(plt.Rectangle((q1, yr - 0.40), max(q3 - q1, 0.25), 0.80, facecolor='#ecebe6', edgecolor='none',
                                   zorder=1))
        ax.plot([med, med], [yr - 0.44, yr + 0.44], color=INK, lw=1.4, zorder=3.5, solid_capstyle='butt')
        marked = [i for i, (key, _) in enumerate(pts) if key in MARKED]     # T, TSV, TSAWVP: in the middle of the row
        offs = swarm_flat(xs, ax, [5.8 if key in MARKED else 3.4 for key, _ in pts], 0.40, first=marked)
        for (key, v), dy in zip(pts, offs):
            col = LEVELS[min(len(key), 5) - 1]
            if key in MARKED:
                ax.scatter(v, yr + dy, s=30, marker=MARKED[key][0], color=col, edgecolor=INK, linewidth=0.9, zorder=4)
            else:
                ax.scatter(v, yr + dy, s=11.5, color=col, edgecolor=SURFACE, linewidth=0.35, zorder=3)
        lo, hi = xs.min(), xs.max()
        ax.text(1.0 + 0.08 / plot_w, yr, f'{lo:.1f}–{hi:.1f}', transform=ax.get_yaxis_transform(), ha='left',
                va='center', fontsize=7.2, color=INK2)
    ax.text(1.0 + 0.08 / plot_w, heads[0][1], 'lowest–\nhighest', transform=ax.get_yaxis_transform(), ha='left',
            va='center', fontsize=6.8, color=MUTED, linespacing=1.0)
    ax.set_yticks([rows[m] for m in rows])
    ax.set_yticklabels([ROW_LABEL[m] + ('' if counts[m] == n_all else f' ({counts[m]}/{n_all})') for m in rows],
                       fontsize=8, color=INK)
    for g, yh in heads:
        ax.text(-left_in / plot_w + 0.04 / plot_w, yh, g, transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=8.3, fontweight='bold', color=INK)
        ax.plot([-left_in / plot_w + 0.04 / plot_w, 1.0], [yh + head * 0.42] * 2, transform=ax.get_yaxis_transform(),
                color=AXIS, lw=0.6, clip_on=False, zorder=0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(METRIC_LABEL[metric] + (' — the 32 combinations with TTPs' if ttp_only else
                                          ' — the 63 feature combinations'))
    ax.grid(False)
    for a, b in spans:                              # vertical gridlines behind the rows, not across the group names
        for gx in (0, 25, 50, 75, 100):
            ax.plot([gx, gx], [a, b], color=GRID, lw=0.6, zorder=0.2)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=4)
    # legend, above the panel: what a dot is and its colour, then the outlined markers, the median and the box
    y1, y2 = 1 - (top_in + 0.10) / fig_h, 1 - (top_in + 0.33) / fig_h
    x_leg = 0.04 / TEXT_W
    t = fig.text(x_leg, y1, 'One dot per feature combination, coloured by its number of feature types:', ha='left',
                 va='center', fontsize=7.5, color=INK2)
    t_end = t.get_window_extent(fig.canvas.get_renderer()).x1 / (fig.dpi * TEXT_W)
    lv = [Line2D([], [], marker='o', ls='', color=c, markeredgecolor=SURFACE, markersize=5.5, label=lab)
          for c, lab in zip(LEVELS, LEVEL_LABEL)]
    fig.legend(handles=lv, loc='center left', ncol=5, bbox_to_anchor=(t_end + 0.04 / TEXT_W, y1), fontsize=7.5,
               handletextpad=0.05, columnspacing=0.7, borderaxespad=0)
    mk = [Line2D([], [], marker=m_, ls='', color=SURFACE, markeredgecolor=INK, markeredgewidth=0.9, markersize=5.8,
                 label=lab) for m_, lab in MARKED.values()]
    mk += [Line2D([], [], color=INK, lw=1.4, label='median'),
           matplotlib.patches.Patch(facecolor='#ecebe6', edgecolor='none', label='middle half of the combinations')]
    fig.legend(handles=mk, loc='center left', ncol=5, bbox_to_anchor=(x_leg - 0.02 / TEXT_W, y2), fontsize=7.5,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0, handlelength=1.2)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--metrics', default=os.path.join(here, '..', 'tables', 'sweep_metrics.csv'))
    ap.add_argument('--out', default=os.path.join(here, '..', 'figures'))
    ap.add_argument('--sweep-source', default='sweep',
                    help='source of the 63-combination sweep (make_outputs.sh: "sweep" = results_cloud + results_mac_rf '
                         '+ results_mac, one training per combination); "cloud" for older metrics files')
    ap.add_argument('--ttp-source', default='mac', help='source of the TTP-only results of all eight methods')
    args = ap.parse_args()
    data = read(args.metrics)
    sources_present = {s for s, _, _ in data}
    sw = args.sweep_source if args.sweep_source in sources_present else 'cloud'
    if sw in sources_present:
        have = {m for s, _, m in data if s == sw}
        eight = [m for m in EIGHT if m in have]
        hybrids = [m for m in HYBRIDS if m in have]
        fig_by_size(data, sw, eight, ('stress', 'top1'), args.out, 'sweep_precision_by_size')
        fig_by_size(data, sw, eight, ('sanity', 'top1'), args.out, 'sweep_precision_by_size_sanity_top1')
        fig_by_size(data, sw, eight, ('stress', 'top5'), args.out, 'sweep_precision_by_size_stress_top5')
        fig_by_size(data, sw, eight, ('cti_ttp', 'cti_score'), args.out, 'sweep_precision_by_size_cti')
        fig_family_effect(data, sw, eight, args.out, 'sweep_family_effect')
        fig_heatmap(data, sw, eight + hybrids, ('stress', 'top1'), args.out, 'sweep_overview')
        fig_heatmap(data, sw, eight + hybrids, ('sanity', 'top1'), args.out, 'sweep_overview_sanity')
        fig_heatmap(data, sw, eight + hybrids, ('cti_ttp', 'cti_score'), args.out, 'sweep_overview_cti', ttp_only=True)
        if hybrids:
            fig_by_size(data, sw, hybrids, ('stress', 'top1'), args.out, 'sweep_precision_by_size_hybrids')
            fig_family_effect(data, sw, hybrids, args.out, 'sweep_family_effect_hybrids')
        # everything in one figure: every approach and hybrid x every combination
        fig_all_in_one(data, sw, eight + hybrids, ('stress', 'top1'), args.out, 'sweep_all_in_one')
        fig_all_in_one(data, sw, eight + hybrids, ('sanity', 'top1'), args.out, 'sweep_all_in_one_sanity')
        fig_all_in_one(data, sw, eight + hybrids, ('cti_ttp', 'cti_score'), args.out, 'sweep_all_in_one_cti',
                       ttp_only=True)
    has_mac = any(s == args.ttp_source and c == 'T' for s, c, _ in data)
    ttp_src = args.ttp_source if has_mac else args.sweep_source
    # 'TTP-only tool (retrained)': combination T of the sweep, NOT the TTP-only evidence of run_ttp_evidence.sh
    sources = [('TTP-only tool (retrained)', ttp_src, 'T', BLUE, 'o'),
               ('three features (thesis tool)', 'thesis3f', 'TSV', ORANGE, 's'),
               ('six features (thesis tool)', 'thesis6f', 'TSAWVP', AQUA, 'D')]
    # further trainings, all from the Mac runs (run_neural_sweep.sh: results_mac/ = "mac", results_mac/repeatN/ = "mac_rN")
    mac = sorted({s for s, _, _ in data if s == args.ttp_source or s.startswith(args.ttp_source + '_r')})
    present = {(s, c) for s, c, _ in data}
    runs = {si: [(s, c) for s in mac if (s, c) in present and (s, c) != (src, c)]
            for si, (_, src, c, _, _) in enumerate(sources)}
    if any(s == 'thesis6f' for s, _, _ in data):
        fig_ttp_compare(data, sources, args.out, 'ttp_vs_3f_6f', runs=runs)
        fig_ttp_compare(data, sources, args.out, 'ttp_vs_3f_6f_methods', only=ALL_METHODS[:8], runs=runs)
        # the hybrids sit high: the axis starts at the 10 below their lowest value (at most at 70)
        hv = [r[f] for (s, c, m), evs in data.items() if m in ALL_METHODS[8:] and
              ((s, c) in {(src, cf) for _, src, cf, _, _ in sources} or any((s, c) in rr for rr in runs.values()))
              for ev, f in (('sanity', 'top1'), ('stress', 'top1'), ('cti_ttp', 'cti_score'))
              for r in [evs.get(ev)] if r is not None and not np.isnan(r[f])]
        lo = min(70, 10 * int(np.floor((min(hv) - 1) / 10))) if hv else 70
        fig_ttp_compare(data, sources, args.out, 'ttp_vs_3f_6f_hybrids', only=ALL_METHODS[8:], runs=runs,
                        xlim=(lo - 2, 101.5), xticks=tuple(range(lo, 101, 10)))


if __name__ == '__main__':
    main()
