import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Use spawn – safe on macOS (avoids fork crash)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import json
import sys
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import xgboost as xgb

# ============================================================================
# 🔽  PicklableXGBWrapper – so pickle can load the XGBoost model
# ============================================================================
class PicklableXGBWrapper:
    def __init__(self, booster, num_classes, seed):
        self._Booster = booster
        self.n_classes_ = num_classes
        self.objective = 'multi:softprob'
        self.seed = seed
        self.classes_ = np.arange(num_classes)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X, missing=np.nan)
        return self._Booster.predict(dtest)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
# ============================================================================

# ======================
# Weighted RRF – EXACTLY like hybrid_query.py’s remotequery
# ======================
# HYBRID_WEIGHTS = {
#     "xgboost": 1,          # weight_xgboost
#     "gnn": 1,              # weight_gnn
#     "siamese": 1,          # weight_siamese
#     "mil": 1,              # weight_mil
#     "rf": 1,               # weight_rf
#     "jaccard": 1,          # weight_jaccard
#     "tvd": 1,              # weight_tvd
#     "featureMatching": 1,  # weight_featureMatching
# }

#self eval weights
# HYBRID_WEIGHTS = {
#     "xgboost": 6/8,          # weight_xgboost
#     "gnn": 2/8,              # weight_gnn
#     "siamese": 4/8,          # weight_siamese
#     "mil": 1/8,              # weight_mil
#     "rf": 3/8,               # weight_rf
#     "jaccard": 8/8,          # weight_jaccard
#     "tvd": 5/8,              # weight_tvd
#     "featureMatching": 7/8  # weight_featureMatching
# }

#cti boosting
HYBRID_WEIGHTS = {
    "xgboost": 1/8,          # weight_xgboost
    "gnn": 1/8,              # weight_gnn
    "siamese": 1/8,          # weight_siamese
    "mil": 8/8,              # weight_mil
    "rf": 1/8,               # weight_rf
    "jaccard": 8/8,          # weight_jaccard
    "tvd": 1/8,              # weight_tvd
    "featureMatching": 8/8,  # weight_featureMatching
}


K = 60    # standard RRF constant (same as hybrid script)
#k=60 is  the standard constant used in Reciprocal Rank Fusion (RRF) - for justification purpouse in thesis

def weighted_rrf(lists, weights_dict, k=K):
    """Weighted Reciprocal Rank Fusion – identical to hybrid_rank_aggregate in hybrid_query.py."""
    model_order = list(weights_dict.keys())   # fixed order
    weights = np.array([weights_dict[m] for m in model_order], dtype=float)
    weights /= weights.sum()

    rank_dict = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            r = item['rank']
            rank_dict.setdefault(name, {})[list_idx] = r

    results = []
    for name, ranked in rank_dict.items():
        score = 0.0
        for i, (model, w) in enumerate(zip(model_order, weights)):
            if i in ranked:
                score += w / (k + ranked[i])
        results.append({'name': name, 'rrf_score': score})

    results.sort(key=lambda x: (-x['rrf_score'], x['name']))
    return [{'apt_name': item['name'], 'rank': i+1} for i, item in enumerate(results)]

# ======================
# Suppress output during module loading
# ======================
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# ======================
# Worker module cache (loaded once per spawned process)
# ======================
_worker_modules = None

def _init_worker_modules():
    global _worker_modules
    if _worker_modules is not None:
        return
    with SuppressOutput():
        jaccard_dir = Path("..") / "Jaccards_Correlation"
        jaccard_file = jaccard_dir / "Jaccards_correlation.py"
        spec = importlib.util.spec_from_file_location("jaccards_module", jaccard_file)
        jaccard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jaccard_module)

        TVD_dir = Path("..") / "TVD_Total_Variation_Distance"
        TVD_file = TVD_dir / "TVD_featureIng.py"
        spec = importlib.util.spec_from_file_location("TVD_module", TVD_file)
        TVD_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(TVD_module)

        featureMatching_dir = Path("..") / "featureMatching"
        featureMatching_file = featureMatching_dir / "featureMatching.py"
        spec = importlib.util.spec_from_file_location("featureMatching_module", featureMatching_file)
        featureMatching_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(featureMatching_module)

        model_dirs = {
            "xgboost": Path("..") / "1.XGboost",
            "gnn": Path("..") / "2.GNN",
            "siamese": Path("..") / "3.SiameseNetwork",
            "mil": Path("..") / "4.Attention-Based_MIL",
            "rf": Path("..") / "5.RF_FeatureEngineering",
        }
        hybrid_queries = {}
        for name, path in model_dirs.items():
            if path.exists():
                query_file = path / 'query.py'
                spec = importlib.util.spec_from_file_location(f"{name}_query", query_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                hybrid_queries[name] = module.hybrid_query

        _worker_modules = {
            'jaccard_module': jaccard_module,
            'TVD_module': TVD_module,
            'featureMatching_module': featureMatching_module,
            'hybrid_queries': hybrid_queries,
        }

def process_element(apt_name, idx, element):
    """Process a single evidence element (runs in spawned worker)."""
    _init_worker_modules()
    m = _worker_modules
    evidence = element['Data']

    with SuppressOutput():
        xgboost_result = m['hybrid_queries']['xgboost'](evidence)
        gnn_result = m['hybrid_queries']['gnn'](evidence)           # direct call – safe in spawn
        siamese_results = m['hybrid_queries']['siamese'](evidence)
        mil_results = m['hybrid_queries']['mil'](evidence)
        rf_results = m['hybrid_queries']['rf'](evidence)
        Jaccards_result = m['jaccard_module'].hybrid_query(evidence)
        TVD_result = m['TVD_module'].hybrid_query(evidence)
        featureMatching_result = m['featureMatching_module'].hybrid_query(evidence)

        # Weighted RRF hybrid (identical to remotequery logic)
        all_lists = {
            "xgboost": xgboost_result,
            "gnn": gnn_result,
            "siamese": siamese_results,
            "mil": mil_results,
            "rf": rf_results,
            "jaccard": Jaccards_result,
            "tvd": TVD_result,
            "featureMatching": featureMatching_result,
        }
        valid = [lst for lst in all_lists.values() if lst]
        if len(valid) >= 2:
            hybrid_result = weighted_rrf(valid, HYBRID_WEIGHTS)
        else:
            hybrid_result = []

    def get_rank(lst):
        for item in lst:
            if item['apt_name'] == apt_name:
                return int(item['rank'])
        return None

    updated = element.copy()
    updated.setdefault('results_ranks', {})
    updated['results_ranks']['XGboost'] = get_rank(xgboost_result)
    updated['results_ranks']['GNN'] = get_rank(gnn_result)
    updated['results_ranks']['Siamesenetwork'] = get_rank(siamese_results)
    updated['results_ranks']['MIL'] = get_rank(mil_results)
    updated['results_ranks']['RF'] = get_rank(rf_results)
    updated['results_ranks']['hybrid'] = get_rank(hybrid_result)
    updated['results_ranks']['Jacccards'] = get_rank(Jaccards_result)
    updated['results_ranks']['TVD'] = get_rank(TVD_result)
    updated['results_ranks']['featureMatching'] = get_rank(featureMatching_result)

    return apt_name, idx, updated

def combine_and_evaluate_evidence(apt_name, evidence_list, modules):
    """Combine all evidence for an APT (sequential, main process)."""
    combined_evidence = {
        "techniques": [], "software": [], "capec": [], "cwe": [], "cve": [], "pulse": []
    }
    for element in evidence_list:
        ev = element['Data']
        for key in combined_evidence:
            combined_evidence[key].extend(ev.get(key, []))
    for key in combined_evidence:
        combined_evidence[key] = list(dict.fromkeys(combined_evidence[key]))

    with SuppressOutput():
        xgboost_result = modules['hybrid_queries']['xgboost'](combined_evidence)
        gnn_result = modules['hybrid_queries']['gnn'](combined_evidence)
        siamese_results = modules['hybrid_queries']['siamese'](combined_evidence)
        mil_results = modules['hybrid_queries']['mil'](combined_evidence)
        rf_results = modules['hybrid_queries']['rf'](combined_evidence)
        Jaccards_result = modules['jaccard_module'].hybrid_query(combined_evidence)
        TVD_result = modules['TVD_module'].hybrid_query(combined_evidence)
        featureMatching_result = modules['featureMatching_module'].hybrid_query(combined_evidence)

        all_lists = {
            "xgboost": xgboost_result,
            "gnn": gnn_result,
            "siamese": siamese_results,
            "mil": mil_results,
            "rf": rf_results,
            "jaccard": Jaccards_result,
            "tvd": TVD_result,
            "featureMatching": featureMatching_result,
        }
        valid = [lst for lst in all_lists.values() if lst]
        if len(valid) >= 2:
            hybrid_result = weighted_rrf(valid, HYBRID_WEIGHTS)
        else:
            hybrid_result = []

    def get_rank(lst):
        for item in lst:
            if item['apt_name'] == apt_name:
                return int(item['rank'])
        return None

    return {
        "report_link": "combination_of_evidence",
        "Data": combined_evidence,
        "results_ranks": {
            "XGboost": get_rank(xgboost_result),
            "GNN": get_rank(gnn_result),
            "Siamesenetwork": get_rank(siamese_results),
            "MIL": get_rank(mil_results),
            "RF": get_rank(rf_results),
            "hybrid": get_rank(hybrid_result),
            "Jacccards": get_rank(Jaccards_result),
            "TVD": get_rank(TVD_result),
            "featureMatching": get_rank(featureMatching_result)
        }
    }

if __name__ == "__main__":
    print("the code is running")

    json_file = 'CTI_Eval.json'
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            json.dump({}, f)

    with open(json_file, 'r') as f:
        data = json.load(f)

    tasks = []
    for apt_name, elements in data.items():
        for idx, element in enumerate(elements):
            tasks.append((apt_name, idx, element))

    if tasks:
        apt_names, indices, elements = zip(*tasks)
        with ProcessPoolExecutor(max_workers=9) as executor:
            results = list(executor.map(process_element, apt_names, indices, elements))
    else:
        results = []

    for apt_name, idx, updated_element in results:
        data[apt_name][idx] = updated_element

    # Main process – load modules for combined evaluation
    with SuppressOutput():
        jaccard_dir = Path("..") / "Jaccards_Correlation"
        jaccard_file = jaccard_dir / "Jaccards_correlation.py"
        spec = importlib.util.spec_from_file_location("jaccards_module", jaccard_file)
        jaccard_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jaccard_module)

        TVD_dir = Path("..") / "TVD_Total_Variation_Distance"
        TVD_file = TVD_dir / "TVD_featureIng.py"
        spec = importlib.util.spec_from_file_location("TVD_module", TVD_file)
        TVD_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(TVD_module)

        featureMatching_dir = Path("..") / "featureMatching"
        featureMatching_file = featureMatching_dir / "featureMatching.py"
        spec = importlib.util.spec_from_file_location("featureMatching_module", featureMatching_file)
        featureMatching_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(featureMatching_module)

        model_dirs = {
            "xgboost": Path("..") / "1.XGboost",
            "gnn": Path("..") / "2.GNN",
            "siamese": Path("..") / "3.SiameseNetwork",
            "mil": Path("..") / "4.Attention-Based_MIL",
            "rf": Path("..") / "5.RF_FeatureEngineering",
        }
        hybrid_queries = {}
        for name, path in model_dirs.items():
            if path.exists():
                query_file = path / 'query.py'
                spec = importlib.util.spec_from_file_location(f"{name}_query", query_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                hybrid_queries[name] = module.hybrid_query

    main_modules = {
        'jaccard_module': jaccard_module,
        'TVD_module': TVD_module,
        'featureMatching_module': featureMatching_module,
        'hybrid_queries': hybrid_queries,
    }

    for apt_name, elements in data.items():
        combined_entry = combine_and_evaluate_evidence(apt_name, elements, main_modules)
        data[apt_name].append(combined_entry)

        print(f"\n{'='*60}")
        print(f"Combined Evidence Results for {apt_name}")
        print(f"{'='*60}")
        res = {
            'XGBoost': combined_entry['results_ranks']['XGboost'],
            'GNN': combined_entry['results_ranks']['GNN'],
            'Siamese': combined_entry['results_ranks']['Siamesenetwork'],
            'MIL': combined_entry['results_ranks']['MIL'],
            'RF': combined_entry['results_ranks']['RF'],
            'hybrid': combined_entry['results_ranks']['hybrid'],
            'Jaccards': combined_entry['results_ranks']['Jacccards'],
            'TVD': combined_entry['results_ranks']['TVD'],
            'Feature Matching': combined_entry['results_ranks']['featureMatching']
        }
        for model, rank in res.items():
            print(f"  {model}: {rank}")

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)