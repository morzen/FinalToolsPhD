#!/usr/bin/env python3
"""
run_evidence_only.py -- the thesis' own tool, unchanged (its profiles and its SAVED, trained models), evaluated with the
same three tests as Tables 4.5 / 4.6 (sanity check, stress test, the 30 CTI reports), except that every query keeps
only the evidence of the chosen feature types (default: TTPs only).

  <venv>/bin/python run_evidence_only.py --tool-root .../FinalToolsPhD_6feat --evidence T --out <dir>
  <venv>/bin/python run_evidence_only.py --tool-root .../FinalToolsPhD_3feat --evidence T --out <dir>

What changes and what does not
  * The tool: nothing.  Its profiles keep every feature type, and every method uses the model saved in the tool folder
    (XGBoost, GNN, Siamese, MIL, RF), exactly as for Tables 4.5 / 4.6.  Nothing is retrained, nothing is written there.
  * The evidence: the thesis' own protocol builds it (common/protocol.py: full profile for the sanity check, the
    seeded halves for the stress test); a thin wrapper then hands the method only the items of the chosen feature
    types.  The halves are the protocol's own (seed 42), so the TTP half of every query is the one of Tables 4.5 /
    4.6.  CTI reports: only the chosen feature types of each report are kept (CTI_eval.py's path for the neural
    methods: their own hybrid_query); a report left with nothing is not scored (it counts at chance in the 30-report
    score, as for the TTP-only configuration).
  * A group whose query is empty (the seven APTs without any TTP, for TTP-only evidence) is still queried, with the
    empty evidence, as the protocol does; the method returns whatever it returns for nothing (all groups tied for the
    set-based measures).
Results: <out>/ranks/<method>/<configuration>.npz, the same format as run_sweep.py (metrics and hybrids by
analyse_sweep.py).  Finished methods are skipped (--force to redo).
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                     # noqa: E402
from sweep_core import (Sweep, cfg_families, cfg_key, save_result, summarize, true_ranks,   # noqa: E402
                        NEURAL_METHODS, METHODS, FAMILIES)

LABELS = {'xgboost': 'XGBoost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'Random Forest',
          'jaccard': 'Jaccard', 'tvd': 'TVD', 'featureMatching': 'Feature Matching'}


class EvidenceFilter:
    """The tool's scorer, handed only the evidence of the chosen feature types (profiles and model unchanged)."""

    def __init__(self, scorer, families):
        self._scorer = scorer
        self._families = set(families)

    def score(self, evidence):
        return self._scorer.score({f: v for f, v in evidence.items() if f in self._families})

    def __getattr__(self, name):                       # apt_names and everything else: the scorer's own
        return getattr(self._scorer, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool-root', required=True, help='FinalToolsPhD_6feat or FinalToolsPhD_3feat (only read)')
    ap.add_argument('--evidence', default='T', help='feature types kept in every query (letters; default T = TTPs)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--cti-template', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    sw = Sweep(args.tool_root)
    sw.use_saved_rf = True                             # the tool's own saved Random Forest, like every other model
    S = tuple(sw.families)                             # the tool's own configuration: all its feature types
    key = cfg_key(S)
    keep = tuple(f for f in S if f in set(cfg_families(args.evidence)))
    template = args.cti_template or os.path.join(sw.root, 'CTI_eval', 'CTI_Eval_empty_template.json')
    if not os.path.exists(template):                   # the template is the same file in both tools
        template = os.path.join(HERE, '..', '..', '..', 'FinalToolsPhD_6feat', 'CTI_eval', 'CTI_Eval_empty_template.json')
    reports = sw.encode_reports(template)
    methods = [m for m in args.methods.split(',') if m]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method {m}")
    if 'xgboost' in methods:                           # the XGBoost model was pickled from train.py run as __main__
        try:
            setattr(sys.modules['__main__'], 'PicklableXGBWrapper', sw.module('xgboost').PicklableXGBWrapper)
        except Exception as exc:
            print(f"[evidence] !! XGBoost cannot be loaded here ({type(exc).__name__}: {exc}); it is skipped", flush=True)
            methods = [m for m in methods if m != 'xgboost']
    print(f"[evidence] tool {sw.root} (configuration {key}, saved models); evidence kept: {', '.join(keep)}; "
          f"{len(reports)} CTI reports; methods {methods}", flush=True)
    t_all = time.time()
    for method in methods:
        path = os.path.join(out, 'ranks', method, f'{key}.npz')
        if os.path.exists(path) and not args.force:
            print(f"[evidence] {method}: already done, skipped", flush=True)
            continue
        t0 = time.time()
        try:
            scorer = sw.make_scorer(method, S)
            module = sw.module(method) if method in NEURAL_METHODS else None
            full, div, _ = sw.attribution(EvidenceFilter(scorer, keep), LABELS[method])
            cti_r, cti_t, cti_n = sw.cti(scorer, reports, keep, module=module)
        except Exception:
            import traceback
            print(f"[evidence] !! {method} failed:\n{traceback.format_exc()}", flush=True)
            continue
        meta = {'config': key, 'families': list(S), 'evidence': list(keep), 'method': method,
                'models': 'saved in the tool folder (not retrained)', 'seconds': time.time() - t0,
                'tool_root': sw.root, 'cti_path': 'hybrid_query' if module is not None else 'encoded once + score'}
        save_result(path, full, div, cti_r, cti_t, cti_n, meta)
        n_grp = full.shape[1]                                   # number of groups in the dataset
        f, d = summarize(true_ranks(full), n_grp), summarize(true_ranks(div), n_grp)
        c = summarize(cti_r[np.arange(len(cti_t)), cti_t], n_grp)
        print(f"[evidence] {key} {method:16s} sanity top1 {f['top1']:6.2f} | stress top1 {d['top1']:6.2f} "
              f"top5 {d['top5']:6.2f} | CTI n={c['n']:2d} {c['cti']:6.2f} | {time.time() - t0:5.1f}s", flush=True)
    sw.restore()
    print(f"[evidence] done in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
