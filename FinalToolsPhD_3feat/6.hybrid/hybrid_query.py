"""
Hybrid attribution query (thesis Chapter 5 / Appendix 8.3).

The evidence is ranked by the eight methods (each through its own hybrid_query, i.e. the same
scorers used everywhere else) and the rankings are fused with weighted reciprocal-rank fusion:

        WRRF(d) = sum_r  w_r / (k + rank_r(d)),   k = 60

Weight configurations are defined once in common/protocol.py (HYBRID_WEIGHTS):
'no_weights', 'self_eval_weights' and 'CTI_boosting'.

Run:  python hybrid_query.py [no_weights|self_eval_weights|CTI_boosting]   (uses ../query_input.json)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import io
import sys
import json
import contextlib
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import protocol as P

QUERY_FILES = {'xgboost': 'query.py', 'gnn': 'query.py', 'siamese': 'query.py', 'mil': 'query.py',
               'rf': 'query.py', 'jaccard': 'Jaccards_correlation.py', 'tvd': 'TVD_featureIng.py',
               'featureMatching': 'featureMatching.py'}

_MODULES = None


def _load(method):
    folder = os.path.join(ROOT, P.METHOD_DIRS[method])
    spec = importlib.util.spec_from_file_location(f"hybrid_{method}_query", os.path.join(folder, QUERY_FILES[method]))
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(folder)
    sys.path.insert(0, folder)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        os.chdir(cwd)
    return module


def modules():
    global _MODULES
    if _MODULES is None:
        _MODULES = {m: _load(m) for m in P.METHODS}
    return _MODULES


def all_rankings(evidence, quiet=True):
    out = {}
    for method, module in modules().items():
        sink = io.StringIO() if quiet else sys.stdout
        with contextlib.redirect_stdout(sink):
            out[method] = module.hybrid_query(evidence)
    return out


def remotequery(evidence, weights="no_weights"):
    """Hybrid ranked list (mid-ranks) of all APTs for human-readable evidence."""
    w = P.HYBRID_WEIGHTS[weights] if isinstance(weights, str) else weights
    return P.wrrf_ranked(all_rankings(evidence), w)


# The XGBoost model was pickled from train.py run as __main__: expose its wrapper class here.
PicklableXGBWrapper = modules()['xgboost'].PicklableXGBWrapper if __name__ == "__main__" else None


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "no_weights"
    with open(os.path.join(ROOT, "query_input.json")) as fh:
        evidence = json.load(fh)
    rankings = all_rankings(evidence)
    result = P.wrrf_ranked(rankings, P.HYBRID_WEIGHTS[config])
    print(f"Hybrid (WRRF, k={P.WRRF_K}, weights: {config})")
    for item in result[:30]:
        per_method = " ".join(f"{m}:{next(x['rank'] for x in r if x['apt_name'] == item['apt_name']):g}"
                              for m, r in rankings.items())
        print(f"  {item['rank']:6g}  {item['apt_name']:35s} | {per_method}")


if __name__ == "__main__":
    main()
