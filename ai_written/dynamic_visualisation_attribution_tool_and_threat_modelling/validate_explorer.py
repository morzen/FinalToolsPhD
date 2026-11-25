#!/usr/bin/env python3
"""
validate_explorer.py -- checks that the explorer shows exactly what the thesis tools compute.

For the CTI reports and the five combined-evidence sets of CTI_eval.py, and for the halved profile of every
group, it compares three rankings of all groups, for each of the eight approaches:
  explorer    the explorer's path (the method's own scorer object, scorer.score, protocol.ranked_results)
  entry point the method's own hybrid_query(), the call CTI_eval.py makes
  thesis      the ranks saved by the thesis run in CTI_eval/CTI_method_ranks.npz (CTI sets only)
and the three WRRF hybrids of the explorer with the hybrid ranks saved in CTI_eval/CTI_Eval_*.json.
It also reports whether every "how" panel recomputed the tool's own scores.

    ../../FinalToolsPhD_6feat/venv/bin/python validate_explorer.py            (or --tool 3f)
Nothing is written.
"""
import sys
sys.dont_write_bytecode = True

import os                                              # noqa: E402
import json                                            # noqa: E402
import argparse                                        # noqa: E402
import time                                            # noqa: E402

import numpy as np                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    parent = os.path.dirname(os.path.dirname(HERE))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool', default='6f')
    ap.add_argument('--methods', default=None, help='comma-separated subset (testing only)')
    ap.add_argument('--profiles', type=int, default=20, help='number of halved profiles to check (spread over the groups)')
    args = ap.parse_args()
    tool = {'6f': os.path.join(parent, 'FinalToolsPhD_6feat'),
            '3f': os.path.join(parent, 'FinalToolsPhD_3feat')}.get(args.tool, args.tool)
    from engine import Engine
    t0 = time.time()
    E = Engine(tool, dump_dir=None, methods=args.methods.split(',') if args.methods else None)
    P = E.P
    print(f"loaded {len(E.methods)} approaches in {time.time() - t0:.0f} s; not loaded: {E.load_errors or 'none'}")

    cti_dir = os.path.join(tool, 'CTI_eval')
    with open(os.path.join(cti_dir, 'CTI_Eval_empty_template.json')) as fh:
        template = json.load(fh)
    saved = np.load(os.path.join(cti_dir, 'CTI_method_ranks.npz'), allow_pickle=False)
    saved_methods = list(saved['methods'])
    saved_names = list(saved['apt_names'])
    key_of = {}
    for s in range(len(saved['apt'])):
        key_of[(str(saved['apt'][s]), str(saved['report_link'][s]))] = s
    hybrid_saved = {}
    for cfg in P.HYBRID_WEIGHTS:
        path = os.path.join(cti_dir, f"CTI_Eval_{cfg}.json")
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            for apt, reps in data.items():
                for rep in reps:
                    hybrid_saved[(cfg, apt, rep['report_link'])] = rep['results_ranks']['hybrid']

    sets = []
    for apt, reports in template.items():
        combined = {f: [] for f in E.features}
        for rep in reports:
            if rep.get('report_link') == 'combination_of_evidence':
                continue
            ev = {f: list(rep['Data'].get(f, [])) for f in E.features}
            for f in E.features:
                combined[f].extend(ev[f])
            sets.append((apt, rep['report_link'], ev))
        sets.append((apt, 'combination_of_evidence', {f: list(dict.fromkeys(v)) for f, v in combined.items()}))

    stats = {m: {'entry': 0, 'thesis': 0, 'n': 0, 'thesis_n': 0, 'panel_max': 0.0} for m in E.methods}
    hyb = {cfg: [0, 0] for cfg in P.HYBRID_WEIGHTS}
    order_saved = [saved_names.index(n) for n in E.names] if saved_names else None

    def compare(apt, link, ev, with_thesis):
        res = E.attribute(ev)
        for m in E.methods:
            mine = np.array(res['methods'][m]['ranks'], dtype=float)
            ref = E.rank_with_tool_entry_point(m, ev)
            ref_v = np.full(len(E.names), np.nan)
            for it in ref:
                ref_v[E.index[it['apt_name']]] = it['rank']
            st = stats[m]
            st['n'] += 1
            st['entry'] += int(np.array_equal(mine, ref_v))
            chk = res['explain'][m].get('check')
            if chk is None and 'error' in res['explain'][m]:
                st['panel_max'] = float('inf')
                print(f"  ! {m}: panel error {res['explain'][m]['error']}")
            elif chk is not None:
                st['panel_max'] = max(st['panel_max'], chk)
            if with_thesis and (apt, link) in key_of and m in saved_methods:
                s = key_of[(apt, link)]
                th = np.asarray(saved['ranks'][s, saved_methods.index(m)], dtype=float)[order_saved]
                st['thesis_n'] += 1
                st['thesis'] += int(np.allclose(mine, th, atol=1e-9, equal_nan=False))
        for cfg in res['hybrid']:
            k = (cfg, apt, link)
            if with_thesis and k in hybrid_saved:
                mine = res['hybrid'][cfg]['ranks'][E.index[apt]]
                hyb[cfg][1] += 1
                hyb[cfg][0] += int(abs(mine - float(hybrid_saved[k])) < 1e-9)

    t1 = time.time()
    for apt, link, ev in sets:
        compare(apt, link, ev, True)
    print(f"CTI sets: {len(sets)} ({time.time() - t1:.0f} s)")
    t1 = time.time()
    picks = sorted(set(np.linspace(0, len(E.names) - 1, min(args.profiles, len(E.names))).round().astype(int)))
    for i in picks:
        compare(E.names[i], 'half', E.profile_evidence(E.names[i], 'half'), False)
    print(f"halved profiles: {len(picks)} ({time.time() - t1:.0f} s)")

    print()
    print(f"{'approach':18s} {'= own hybrid_query':>20s} {'= thesis CTI ranks':>20s} {'panel max |diff|':>18s}")
    ok = True
    for m in E.methods:
        st = stats[m]
        good = st['entry'] == st['n'] and st['thesis'] == st['thesis_n'] and st['panel_max'] < 1e-9
        ok &= good
        print(f"{m:18s} {st['entry']:>9d} / {st['n']:<8d} {st['thesis']:>9d} / {st['thesis_n']:<8d} "
              f"{st['panel_max']:>18.2e}")
    for cfg, (a, n) in hyb.items():
        if n:
            ok &= a == n
            print(f"hybrid {cfg:11s}: true-group rank equal to the thesis CTI results in {a} / {n} sets")
    print("\nALL IDENTICAL" if ok else "\nDIFFERENCES FOUND (see above)")


if __name__ == '__main__':
    main()
