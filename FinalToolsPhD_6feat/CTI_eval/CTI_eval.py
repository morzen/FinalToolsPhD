"""
CTI-based evaluation (thesis §4.4.3 / §4.6).

Every CTI report in CTI_Eval_empty_template.json is queried with the hybrid_query() of each of the
eight methods (the same scorers used by eval.py and the false-flag scripts). The rank of the
attributed APT (mid-rank for ties) is stored per method, and the hybrid (WRRF, Appendix C.3) is
computed for the three weight configurations of the thesis in one run:

    CTI_Eval_no_weights.json, CTI_Eval_self_eval_weights.json, CTI_Eval_CTI_boosting.json

(same structure as before). A combined-evidence entry is appended for every APT group.
CTI_method_ranks.npz keeps every method's full ranking (mid-ranks of all APTs) for every report and
combined-evidence set, so other hybrid weightings can be evaluated later without re-running the models.
CTI_summary.txt / .json give, per method, the thesis CTI score
        score = 100 * (1 - (rank - 1) / (N - 1)),  N = number of APTs (163)
averaged over the 29 CTI reports of the template. The script stops before
running anything if a report contains no item known to the dataset.

Run from this folder:  python CTI_eval.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import io
import sys
import copy
import json
import contextlib
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import protocol as P

TEMPLATE = os.path.join(HERE, "CTI_Eval_empty_template.json")
JSON_KEY = {'xgboost': 'XGboost', 'gnn': 'GNN', 'siamese': 'Siamesenetwork', 'mil': 'MIL', 'rf': 'RF',
            'jaccard': 'Jacccards', 'tvd': 'TVD', 'featureMatching': 'featureMatching'}
QUERY_FILES = {'xgboost': 'query.py', 'gnn': 'query.py', 'siamese': 'query.py', 'mil': 'query.py',
               'rf': 'query.py', 'jaccard': 'Jaccards_correlation.py', 'tvd': 'TVD_featureIng.py',
               'featureMatching': 'featureMatching.py'}
VERBOSE = "--verbose" in sys.argv


def load_query_modules():
    modules = {}
    for method in P.METHODS:
        folder = os.path.join(ROOT, P.METHOD_DIRS[method])
        spec = importlib.util.spec_from_file_location(f"{method}_query", os.path.join(folder, QUERY_FILES[method]))
        module = importlib.util.module_from_spec(spec)
        cwd = os.getcwd()
        os.chdir(folder)
        sys.path.insert(0, folder)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(folder)
            os.chdir(cwd)
        modules[method] = module
    return modules


MODULES = None
RANK_ROWS = []      # one entry per evidence set: (apt, report_link, {method: rank vector over all APTs})


def all_rankings(evidence):
    """{method: ranked list of all APTs} for one piece of human-readable evidence."""
    out = {}
    for method, module in MODULES.items():
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink if not VERBOSE else sys.stdout):
            out[method] = module.hybrid_query(evidence)
        if len(out[method]) != len(P.apt_names()):
            raise RuntimeError(f"{method}: hybrid_query returned {len(out[method])} APTs")
    return out


def rank_of(ranked, apt):
    for item in ranked:
        if item['apt_name'] == apt:
            return float(item['rank'])
    raise KeyError(apt)


def rank_vector(ranked):
    """rank of every APT, in the canonical (sorted-name) order"""
    idx = {n: i for i, n in enumerate(P.apt_names())}
    r = np.full(len(idx), np.nan)
    for item in ranked:
        r[idx[item['apt_name']]] = float(item['rank'])
    return r


def evaluate(evidence, apt, report_link):
    rankings = all_rankings(evidence)
    RANK_ROWS.append((apt, report_link, {m: rank_vector(rankings[m]) for m in P.METHODS}))
    ranks = {JSON_KEY[m]: rank_of(r, apt) for m, r in rankings.items()}
    hybrid = {cfg: rank_of(P.wrrf_ranked(rankings, w), apt) for cfg, w in P.HYBRID_WEIGHTS.items()}
    recognised = sum(len(v) for v in P.encode_origin_evidence(evidence).values())
    return ranks, hybrid, recognised


def main():
    global MODULES
    with open(TEMPLATE) as fh:
        template = json.load(fh)
    unusable = [(apt, rep['report_link']) for apt, reports in template.items() for rep in reports
                if rep.get('report_link') != "combination_of_evidence"
                and sum(len(v) for v in P.encode_origin_evidence(rep['Data']).values()) == 0]
    if unusable:
        sys.exit("These CTI reports contain no item known to the dataset, so no method can attribute them.\n"
                 "Remove them from CTI_Eval_empty_template.json:\n  " +
                 "\n  ".join(f"{a}: {l}" for a, l in unusable))
    print("Loading the eight methods...")
    MODULES = load_query_modules()

    outputs = {cfg: {} for cfg in P.HYBRID_WEIGHTS}
    per_report = []
    n_apts = len(P.apt_names())
    for apt, reports in template.items():
        for cfg in outputs:
            outputs[cfg][apt] = []
        combined = {f: [] for f in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']}
        for rep in reports:
            if rep.get('report_link') == "combination_of_evidence":
                continue
            evidence = rep['Data']
            for f in combined:
                combined[f].extend(evidence.get(f, []))
            ranks, hybrid, recognised = evaluate(evidence, apt, rep['report_link'])
            per_report.append({'apt': apt, 'report_link': rep['report_link'], 'recognised_items': recognised,
                               'ranks': ranks, 'hybrid': hybrid})
            for cfg in outputs:
                entry = copy.deepcopy(rep)
                entry['results_ranks'] = dict(ranks, hybrid=hybrid[cfg])
                entry['recognised_items'] = recognised
                outputs[cfg][apt].append(entry)
            print(f"{apt:15s} recognised items {recognised:3d} | " +
                  " ".join(f"{k}:{v:g}" for k, v in ranks.items()) +
                  " | hybrid " + " ".join(f"{c}:{v:g}" for c, v in hybrid.items()), flush=True)
        combined = {f: list(dict.fromkeys(v)) for f, v in combined.items()}
        ranks, hybrid, recognised = evaluate(combined, apt, "combination_of_evidence")
        for cfg in outputs:
            outputs[cfg][apt].append({'report_link': "combination_of_evidence", 'Data': combined,
                                      'results_ranks': dict(ranks, hybrid=hybrid[cfg]),
                                      'recognised_items': recognised})
        print(f"{apt:15s} COMBINED evidence | " + " ".join(f"{k}:{v:g}" for k, v in ranks.items()) +
              " | hybrid " + " ".join(f"{c}:{v:g}" for c, v in hybrid.items()), flush=True)

    for cfg, data in outputs.items():
        with open(os.path.join(HERE, f"CTI_Eval_{cfg}.json"), "w") as fh:
            json.dump(data, fh, indent=2)
    names = P.apt_names()
    np.savez_compressed(os.path.join(HERE, "CTI_method_ranks.npz"),
                        methods=np.array(P.METHODS), apt_names=np.array(names),
                        apt=np.array([a for a, _, _ in RANK_ROWS]),
                        true_index=np.array([names.index(a) for a, _, _ in RANK_ROWS]),
                        report_link=np.array([l for _, l, _ in RANK_ROWS]),
                        combined=np.array([l == "combination_of_evidence" for _, l, _ in RANK_ROWS]),
                        ranks=np.array([[r[m] for m in P.METHODS] for _, _, r in RANK_ROWS]))   # (sets, methods, APTs)

    # ---- summary (thesis CTI score) -----------------------------------------------------
    columns = list(JSON_KEY.values()) + [f"hybrid_{c}" for c in P.HYBRID_WEIGHTS]

    def rank_list(rows, col):
        if col.startswith("hybrid_"):
            return [r['hybrid'][col[len("hybrid_"):]] for r in rows]
        return [r['ranks'][col] for r in rows]

    summary = {'n_reports': len(per_report),
               'n_reports_per_group': {apt: sum(1 for r in per_report if r['apt'] == apt) for apt in template},
               'cti_score': {}, 'mean_rank': {}, 'hybrid_weights': P.HYBRID_WEIGHTS}
    lines = [f"CTI evaluation: {len(per_report)} reports, {len(template)} APT groups, N = {n_apts} APTs", "",
             f"{'method':28s} {'CTI score':>10s} {'mean rank':>10s}"]
    for col in columns:
        ranks = rank_list(per_report, col)
        score = float(np.mean([P.cti_score(r, n_apts) for r in ranks]))
        summary['cti_score'][col] = score
        summary['mean_rank'][col] = float(np.mean(ranks))
        lines.append(f"{col:28s} {score:9.2f}% {np.mean(ranks):10.2f}")
    lines += ["", "Hybrid weights (common/protocol.py, HYBRID_WEIGHTS):"] + P.hybrid_weights_lines()
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(HERE, "CTI_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(HERE, "CTI_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


# The XGBoost model was pickled from train.py run as __main__: expose its wrapper class here.
def _expose_xgb_wrapper():
    spec = importlib.util.spec_from_file_location("_xgb_wrapper_source", os.path.join(ROOT, "1.XGboost", "query.py"))
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module.PicklableXGBWrapper


PicklableXGBWrapper = _expose_xgb_wrapper()

if __name__ == "__main__":
    main()
