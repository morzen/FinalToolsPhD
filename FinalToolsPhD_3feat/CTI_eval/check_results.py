"""
Per-group and overall CTI results from a CTI_Eval_*.json file produced by CTI_eval.py.

Average (mid-)rank and thesis CTI score, score = 100 (1 - (rank - 1) / (N - 1)), over the CTI
reports (combined-evidence entries are listed separately).

Usage:  python check_results.py [CTI_Eval_no_weights.json | CTI_Eval_self_eval_weights.json | CTI_Eval_CTI_boosting.json]
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P
N_APTS = len(P.apt_names())            # number of groups in the dataset
path = sys.argv[1] if len(sys.argv) > 1 else "CTI_Eval_no_weights.json"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path)) as fh:
    data = json.load(fh)


def score(rank):
    return 100.0 * (1.0 - (rank - 1.0) / (N_APTS - 1.0))


def summarise(reports):
    acc = {}
    for rep in reports:
        if rep['report_link'] == "combination_of_evidence":
            continue
        for method, rank in rep['results_ranks'].items():
            acc.setdefault(method, []).append(rank)
    return {m: (sum(r) / len(r), sum(score(x) for x in r) / len(r), len(r)) for m, r in acc.items()}


print("=" * 70)
print(f"CTI RESULTS BY APT GROUP ({path})")
print("=" * 70)
for group, reports in data.items():
    res = summarise(reports)
    if res:
        print(f"\n{group} ({next(iter(res.values()))[2]} reports):")
        for m, (avg, sc, n) in sorted(res.items(), key=lambda x: x[1][0]):
            print(f"  {m:16s} mean rank {avg:7.2f}   CTI score {sc:6.2f}%")
    for rep in reports:
        if rep['report_link'] == "combination_of_evidence":
            print("  combined evidence: " + ", ".join(f"{m}:{r:g}" for m, r in rep['results_ranks'].items()))

print("\n" + "=" * 70)
print("ALL GROUPS")
print("=" * 70)
total = summarise([r for reps in data.values() for r in reps])
for m, (avg, sc, n) in sorted(total.items(), key=lambda x: x[1][0]):
    print(f"{m:16s} mean rank {avg:7.2f}   CTI score {sc:6.2f}%   ({n} reports)")
