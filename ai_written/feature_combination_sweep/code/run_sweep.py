#!/usr/bin/env python3
"""
run_sweep.py -- run attribution methods for feature configurations and store full rank matrices.

For every (configuration, method) it stores, in <out>/ranks/<method>/<config>.npz
    full, divided   181 x 181 mid-ranks of the sanity check (full profiles) and the stress test (shuffled & divided)
    cti_ranks       30 x 181 mid-ranks of the 30 CTI reports restricted to the configuration (NaN row = the report keeps
                    no item known to the dataset in this configuration)
Nothing in the tool folder is written.  Finished (configuration, method) cells are skipped, so a run can be stopped and
restarted.

  python run_sweep.py --tool-root <.../FinalToolsPhD_6feat> --out <results dir> --configs ttp
  python run_sweep.py ... --configs all --methods jaccard,featureMatching,tvd,rf
  python run_sweep.py ... --configs 6f,3f --validate --saved-6f <.../FinalToolsPhD_6feat> --saved-3f <.../FinalToolsPhD_3feat>

--validate compares the configurations 6f and 3f with the thesis tools' own saved rank matrices (sanity check, stress test
and CTI-30): the set-based methods must be identical to the last digit.
"""
import os
import sys
import json
import time
import argparse
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                         # noqa: E402
from sweep_core import (Sweep, cfg_families, cfg_key, parse_configs, save_result, load_result, summarize,   # noqa: E402
                        true_ranks, CLOUD_METHODS, NEURAL_METHODS, METHODS, TOP_K, LETTER)

LABELS = {'xgboost': 'XGBoost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'Random Forest',
          'jaccard': 'Jaccard', 'tvd': 'TVD', 'featureMatching': 'Feature Matching'}


def result_path(out, method, key):
    return os.path.join(out, 'ranks', method, f'{key}.npz')


def ext_path(out, method, key):
    return os.path.join(out, 'ranks_external', method, f'{key}.npz')


def versions():
    v = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('numpy', 'pandas', 'sklearn', 'scipy', 'torch', 'xgboost', 'torch_geometric'):
        mod = sys.modules.get(name)
        if mod is not None:
            v[name] = getattr(mod, '__version__', '?')
    return v


def run(args):
    sw = Sweep(args.tool_root)
    sw.use_saved_rf = args.rf_saved
    template = args.cti_template or os.path.join(sw.root, 'CTI_eval', 'CTI_Eval_empty_template.json')
    if not os.path.exists(template):
        template = os.path.join(HERE, '..', 'inputs', 'CTI_Eval_empty_template.json')
    reports = sw.encode_reports(template)
    configs = parse_configs(args.configs)
    if args.shard:                                   # e.g. 1/2: every second configuration (parallel runs)
        i, n = (int(x) for x in args.shard.split('/'))
        configs = configs[i - 1::n]
    methods = [m for m in args.methods.split(',') if m]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method {m}")
    if 'xgboost' in methods:
        # the XGBoost model was pickled from train.py run as __main__ (as CTI_eval.py, expose its wrapper class here)
        try:
            setattr(sys.modules['__main__'], 'PicklableXGBWrapper', sw.module('xgboost').PicklableXGBWrapper)
        except Exception as exc:
            print(f"[sweep] !! XGBoost cannot be loaded here ({type(exc).__name__}: {exc}); it is skipped", flush=True)
            methods = [m for m in methods if m != 'xgboost']
    ext_ids, ext_sets = None, None
    if args.external:
        with open(args.external) as fh:
            ext = json.load(fh)
        if ext.get('apt_names') and ext['apt_names'] != sw.canon:
            sys.exit(f"{args.external}: built with another list of groups")
        ext_ids = sorted(ext['sets'])
        enc = sw.encode_sets(ext['sets'])
        ext_sets = [enc[s] for s in ext_ids]
    os.makedirs(args.out, exist_ok=True)
    print(f"[sweep] tool {sw.root}\n[sweep] {len(configs)} configurations x {methods}; {len(reports)} CTI reports"
          f"{'; %d external sets' % len(ext_ids) if ext_ids else ''}", flush=True)
    t_all = time.time()
    for ci, key in enumerate(configs, 1):
        S = cfg_families(key)
        stats = sw.profile_stats(S)
        for method in methods:
            path = result_path(args.out, method, key)
            epath = ext_path(args.out, method, key)
            if os.path.exists(path) and not args.force and (ext_ids is None or os.path.exists(epath)):
                continue
            t0 = time.time()
            try:
                scorer = sw.make_scorer(method, S)
                t_train = time.time() - t0
                # neural methods: the CTI reports go through their own hybrid_query, as in CTI_eval.py
                module = sw.module(method) if method in NEURAL_METHODS else None
                if not os.path.exists(path) or args.force:
                    full, div, res = sw.attribution(scorer, LABELS[method])
                    cti_r, cti_t, cti_n = sw.cti(scorer, reports, S, module=module)
                else:
                    old = load_result(path)
                    full, div, cti_r, cti_t, cti_n = old['full'], old['divided'], old['cti_ranks'], old['cti_true_idx'], old['cti_items']
                if ext_ids is not None:
                    e_r, _, e_n = sw.cti(scorer, ext_sets, S, module=module)
                    os.makedirs(os.path.dirname(epath), exist_ok=True)
                    np.savez_compressed(epath + '.tmp.npz', set_ids=np.array(ext_ids), ranks=e_r, items=e_n,
                                        apt_names=np.array(sw.canon))
                    os.replace(epath + '.tmp.npz', epath)
            except Exception:
                import traceback
                print(f"[sweep] !! {key} {method} failed:\n{traceback.format_exc()}", flush=True)
                continue
            meta = {'config': key, 'families': list(S), 'method': method, 'seconds': time.time() - t0,
                    'seconds_setup': t_train, 'versions': versions(), 'stats': stats,
                    'tool_root': sw.root, 'cti_path': 'hybrid_query' if module is not None else 'encoded once + score'}
            save_result(path, full, div, cti_r, cti_t, cti_n, meta)
            n_grp = full.shape[1]                               # number of groups in the dataset
            f, d = summarize(true_ranks(full), n_grp), summarize(true_ranks(div), n_grp)
            c = summarize(cti_r[np.arange(len(cti_t)), cti_t], n_grp)
            print(f"[sweep] {ci:2d}/{len(configs)} {key:7s} {method:16s} full top1 {f['top1']:6.2f} | divided top1 {d['top1']:6.2f} "
                  f"top5 {d['top5']:6.2f} | CTI n={c['n']:2d} {c['cti']:6.2f} | {time.time() - t0:5.1f}s "
                  f"(elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
    sw.restore()
    print(f"[sweep] done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return sw, configs, methods


# ------------------------------------------------------------------------------ validation against the thesis tools
def validate(args, sw, methods):
    """Compare configurations 6f and 3f with the saved matrices of the thesis tools."""
    from sweep_core import cfg_families
    P = sw.P
    report = {}
    for key, saved_root in (('TSAWVP', args.saved_6f), ('TSV', args.saved_3f)):
        if not saved_root:
            continue
        for method in methods:
            path = result_path(args.out, method, key)
            if not os.path.exists(path):
                continue
            mine = load_result(path)
            d = os.path.join(saved_root, P.METHOD_DIRS[method], 'results')
            row = {}
            for scen, name in (('full', 'attribution_ranks_full.npy'), ('divided', 'attribution_ranks_divided.npy')):
                f = os.path.join(d, name)
                if not os.path.exists(f):
                    row[scen] = 'saved file missing'
                    continue
                a, b = mine[scen].astype(np.float64), np.load(f).astype(np.float64)
                same_rows = int((np.abs(a - b).max(axis=1) == 0).sum())
                row[scen] = {'rows_identical': same_rows, 'rows': int(a.shape[0]), 'max_abs_rank_diff': float(np.abs(a - b).max()),
                             'top1_mine': float(100 * np.mean(np.diag(a) <= 1)), 'top1_saved': float(100 * np.mean(np.diag(b) <= 1)),
                             'top5_mine': float(100 * np.mean(np.diag(a) <= 5)), 'top5_saved': float(100 * np.mean(np.diag(b) <= 5))}
            # CTI-30 against the tool's CTI_method_ranks.npz
            cf = os.path.join(saved_root, 'CTI_eval', 'CTI_method_ranks.npz')
            if os.path.exists(cf):
                z = np.load(cf, allow_pickle=False)
                saved_methods = [str(x) for x in z['methods']]
                keymap = {(str(a), str(l)): i for i, (a, l) in enumerate(zip(z['apt'], z['report_link']))}
                j = saved_methods.index(method)
                tpl = os.path.join(saved_root, 'CTI_eval', 'CTI_Eval_empty_template.json')
                if not os.path.exists(tpl):       # the template is the same file in both tools (md5 8f502de0...)
                    tpl = args.cti_template or os.path.join(sw.root, 'CTI_eval', 'CTI_Eval_empty_template.json')
                reports = sw.encode_reports(tpl)
                n = same = 0
                worst = 0.0
                if reports is not None:
                    for i, rep in enumerate(reports):
                        k = (rep['apt'], rep['link'])
                        if k not in keymap or np.isnan(mine['cti_ranks'][i]).any():
                            continue
                        b = np.asarray(z['ranks'][keymap[k], j], dtype=np.float64)
                        a = mine['cti_ranks'][i].astype(np.float64)
                        n += 1
                        dd = float(np.abs(a - b).max())
                        worst = max(worst, dd)
                        same += int(dd == 0.0)
                row['cti'] = {'reports_compared': n, 'identical': same, 'max_abs_rank_diff': worst}
            report[f'{key}:{method}'] = row
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default='ttp')
    ap.add_argument('--methods', default=','.join(CLOUD_METHODS))
    ap.add_argument('--cti-template', default=None)
    ap.add_argument('--external', default=None, help='external_sets.json of the external evaluation (optional)')
    ap.add_argument('--rf-saved', action='store_true', help="RF: use the tool's saved model instead of retraining "
                                                            "(only meaningful for the tool's own configuration)")
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--shard', default=None, help='i/n: run every n-th configuration starting at i (parallel runs)')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--saved-6f', default=None)
    ap.add_argument('--saved-3f', default=None)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    sw, configs, methods = run(args)
    if args.validate:
        rep = validate(args, sw, methods)
        path = os.path.join(args.out, 'validation.json')
        with open(path, 'w') as fh:
            json.dump(rep, fh, indent=1)
        print(f"\n[validation] against the thesis tools' saved matrices ({path})")
        for k, v in rep.items():
            parts = []
            for scen in ('full', 'divided'):
                x = v.get(scen)
                if isinstance(x, dict):
                    parts.append(f"{scen}: {x['rows_identical']}/{x['rows']} rows identical (max |diff| {x['max_abs_rank_diff']:g})")
            if 'cti' in v:
                parts.append(f"CTI: {v['cti']['identical']}/{v['cti']['reports_compared']} reports identical (max |diff| {v['cti']['max_abs_rank_diff']:g})")
            print(f"  {k:22s} " + " | ".join(parts))


if __name__ == '__main__':
    main()
