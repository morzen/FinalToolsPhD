#!/usr/bin/env python3
"""
analyse_sweep.py -- metrics and the thesis' WRRF hybrids from the saved rank matrices of the sweep.

Inputs
  --results <label>=<folder>   results folder(s) holding ranks/<method>/<config>.npz (run_sweep.py output),
                               e.g. cloud=results_cloud  mac=results_mac
  --thesis <label>=<tool>      a thesis tool folder: its SAVED results are read as they are (attribution_ranks_*.npy of
                               every method, CTI_eval/CTI_method_ranks.npz), e.g. thesis6f=.../FinalToolsPhD_6feat
  --tool-root <6f tool>        common/protocol.py is imported from here (WRRF, mid-ranks, HYBRID_WEIGHTS: nothing
                               is re-implemented) and the CTI template is read from here
Output
  <out>/sweep_metrics.csv      one row per (source, configuration, method, evaluation)
  --quick                      a short table on the screen instead

Evaluations (thesis protocol, mid-ranks for ties, Top-k hit iff rank <= k, N = number of groups in the dataset)
  sanity   full profiles, one query per group
  stress   shuffled & divided profiles, one query per group
  cti_ttp  the 27 CTI reports that contain techniques (the same reports in every configuration with techniques)
  cti      the CTI reports that keep at least one item known to the dataset in the configuration
  cti30    all the reports; a report with no usable item counts at the middle rank (CTI score 50): this is what the tool
           returns for evidence it knows nothing of (every group tied)
Hybrids: WRRF (k = 60) of the eight methods with HYBRID_WEIGHTS, only where all eight methods are present for the same
configuration in the same source (exactly as 6.hybrid/hybrid_eval.py and CTI_eval.py fuse the saved ranks).
"""
import os
import sys
import csv
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np                                                       # noqa: E402
from sweep_core import (load_result, cfg_families, all_configs, LETTER, FAMILIES, METHODS, TOP_K,  # noqa: E402
                        summarize)

# the number of groups is read from the rank matrices (it was fixed at 181 before 23 November 2026)
EVALS = ['sanity', 'stress', 'cti_ttp', 'cti', 'cti30']


def import_protocol(tool_root):
    tool_root = os.path.abspath(tool_root)
    if tool_root not in sys.path:
        sys.path.insert(0, tool_root)
    from common import protocol as P
    return P


def ttp_mask(template_path):
    with open(template_path) as fh:
        t = json.load(fh)
    mask, keys = [], []
    for apt, reps in t.items():
        for rep in reps:
            if rep.get('report_link') == 'combination_of_evidence':
                continue
            mask.append(len(rep['Data'].get('techniques', [])) > 0)
            keys.append((apt, rep['report_link']))
    return np.array(mask), keys


# ------------------------------------------------------------------------------ loading
def load_folder(folder):
    """{config: {method: result dict}} from a run_sweep.py results folder."""
    out = {}
    base = os.path.join(folder, 'ranks')
    if not os.path.isdir(base):
        return out
    for method in sorted(os.listdir(base)):
        d = os.path.join(base, method)
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.npz') and not fn.endswith('.tmp.npz'):
                out.setdefault(fn[:-4], {})[method] = load_result(os.path.join(d, fn))
    return out


def load_thesis_tool(tool, P, report_keys):
    """The saved results of a thesis tool folder, in the same layout as run_sweep.py results."""
    import ast
    feats = None
    with open(os.path.join(tool, 'common', 'protocol.py')) as fh:
        for line in fh:
            if line.startswith('FEATURES = '):
                feats = ast.literal_eval(line.split('=', 1)[1].strip())
                break
    key = ''.join(LETTER[f] for f in FAMILIES if f in feats)
    z = np.load(os.path.join(tool, 'CTI_eval', 'CTI_method_ranks.npz'), allow_pickle=False)
    saved_methods = [str(x) for x in z['methods']]
    row_of = {(str(a), str(l)): i for i, (a, l) in enumerate(zip(z['apt'], z['report_link']))}
    names = [str(x) for x in z['apt_names']]
    res = {}
    for m in METHODS:
        d = os.path.join(tool, P.METHOD_DIRS[m], 'results')
        full = np.load(os.path.join(d, 'attribution_ranks_full.npy')).astype(np.float32)
        div = np.load(os.path.join(d, 'attribution_ranks_divided.npy')).astype(np.float32)
        j = saved_methods.index(m)
        cti = np.stack([z['ranks'][row_of[k], j] for k in report_keys]).astype(np.float32)
        true_idx = np.array([names.index(a) for a, _ in report_keys])
        res[m] = {'full': full, 'divided': div, 'cti_ranks': cti, 'cti_true_idx': true_idx,
                  'cti_items': np.ones(len(report_keys), dtype=np.int32), 'meta': {'source': tool}}
    return key, res


# ------------------------------------------------------------------------------ metrics
def hybrid(P, per_method, weights):
    """WRRF of the eight methods (same calls as hybrid_eval.py / CTI_eval.py)."""
    out = {}
    for part in ('full', 'divided'):
        fused = P.wrrf_scores({m: per_method[m][part].astype(np.float64) for m in METHODS}, weights)
        out[part] = np.vstack([P.midranks(row) for row in fused]).astype(np.float32)
    cti = per_method[METHODS[0]]['cti_ranks']
    rows = []
    for i in range(cti.shape[0]):
        if any(np.isnan(per_method[m]['cti_ranks'][i]).any() for m in METHODS):
            rows.append(np.full(cti.shape[1], np.nan, dtype=np.float32))
            continue
        fused = P.wrrf_scores({m: per_method[m]['cti_ranks'][i].astype(np.float64) for m in METHODS}, weights)
        rows.append(P.midranks(fused).astype(np.float32))
    out['cti_ranks'] = np.vstack(rows)
    out['cti_true_idx'] = per_method[METHODS[0]]['cti_true_idx']
    return out


def metrics(res, mask_ttp):
    full = np.diag(res['full'].astype(np.float64))
    div = np.diag(res['divided'].astype(np.float64))
    cr = res['cti_ranks'].astype(np.float64)
    tr = cr[np.arange(cr.shape[0]), res['cti_true_idx']]          # NaN where the report is unusable
    usable = ~np.isnan(tr)
    n_groups = res['full'].shape[1]                              # number of groups in the dataset
    chance = (n_groups + 1) / 2.0
    out = {'sanity': summarize(full, n_groups), 'stress': summarize(div, n_groups),
           'cti': summarize(tr[usable], n_groups),
           'cti_ttp': summarize(tr[mask_ttp], n_groups) if usable[mask_ttp].all() else {'n': int(usable[mask_ttp].sum())},
           'cti30': summarize(np.where(usable, tr, chance), n_groups)}      # 'cti30' = ALL the reports of the template
    out['cti30']['n_unusable'] = int((~usable).sum())
    return out


def collect(args):
    P = import_protocol(args.tool_root)
    template = os.path.join(args.tool_root, 'CTI_eval', 'CTI_Eval_empty_template.json')
    mask, report_keys = ttp_mask(template)
    sources = {}
    for spec in args.results or []:                 # the same label twice: the later folder overrides per (config, method)
        label, folder = spec.split('=', 1)
        for key, per_method in load_folder(folder).items():
            sources.setdefault(label, {}).setdefault(key, {}).update(per_method)
    for spec in args.thesis or []:
        label, tool = spec.split('=', 1)
        key, res = load_thesis_tool(tool, P, report_keys)
        sources[label] = {key: res}
    rows = []
    for label, by_cfg in sources.items():
        for key, per_method in sorted(by_cfg.items()):
            fams = cfg_families(key)
            stats = next((r['meta'].get('stats') for r in per_method.values() if r['meta'].get('stats')), {}) or {}
            entries = dict(per_method)
            if all(m in per_method for m in METHODS):
                for cfg, w in P.HYBRID_WEIGHTS.items():
                    entries[f'hybrid_{cfg}'] = hybrid(P, per_method, w)
            for method, res in entries.items():
                for ev, v in metrics(res, mask).items():
                    rows.append({'source': label, 'config': key, 'n_families': len(fams), 'families': '+'.join(fams),
                                 'method': method, 'evaluation': ev, 'n': v.get('n'),
                                 **{f'top{k}': v.get(f'top{k}') for k in TOP_K},
                                 'mean_rank': v.get('mean_rank'), 'cti_score': v.get('cti'),
                                 'n_unusable': v.get('n_unusable', ''),
                                 'dimension': stats.get('dimension', ''), 'n_groups_empty': stats.get('n_groups_empty', '')})
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ['source', 'config', 'n_families', 'families', 'method', 'evaluation', 'n'] + [f'top{k}' for k in TOP_K] + \
           ['mean_rank', 'cti_score', 'n_unusable', 'dimension', 'n_groups_empty']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.12g}' if isinstance(v, float) else v) for k, v in r.items()})


def quick(rows):
    order = METHODS + ['hybrid_no_weights', 'hybrid_self_eval_weights', 'hybrid_CTI_boosting']
    by = {}
    for r in rows:
        by.setdefault((r['source'], r['config']), {}).setdefault(r['method'], {})[r['evaluation']] = r
    for (src, key), meth in sorted(by.items()):
        print(f"\n[{src}] configuration {key} ({'+'.join(cfg_families(key))})")
        print(f"  {'method':26s} {'sanity@1':>9s} {'stress@1':>9s} {'stress@5':>9s} {'CTI(TTP rep)':>12s} {'CTI(all)':>8s}")
        for m in order:
            if m not in meth:
                continue
            e = meth[m]
            f = lambda x: '' if x is None or x == '' or (isinstance(x, float) and np.isnan(x)) else f'{x:.2f}'
            print(f"  {m:26s} {f(e['sanity']['top1']):>9s} {f(e['stress']['top1']):>9s} {f(e['stress']['top5']):>9s} "
                  f"{f(e['cti_ttp'].get('cti_score')):>12s} {f(e['cti30'].get('cti_score')):>8s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', action='append', help='label=folder (repeatable)')
    ap.add_argument('--thesis', action='append', help='label=thesis tool folder (repeatable)')
    ap.add_argument('--tool-root', default=os.path.join(HERE, '..', '..', '..', 'FinalToolsPhD_6feat'))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'tables', 'sweep_metrics.csv'))
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.results and all('=' not in s for s in args.results):
        args.results = [f'run={s}' for s in args.results]
    rows = collect(args)
    if args.quick:
        quick(rows)
    else:
        write_csv(rows, args.out)
        print(f"[analyse] {len(rows)} rows -> {args.out}")


if __name__ == '__main__':
    main()
