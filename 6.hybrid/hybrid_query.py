import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # tolerate multiple OpenMP copies
os.environ["OMP_NUM_THREADS"] = "1"           # prevent thread oversubscription
import pandas as pd
import importlib.util
from pathlib import Path
import sys
import numpy as np
import xgboost as xgb
import gc
import multiprocessing



class PicklableXGBWrapper:
    """Wrapper to mimic XGBClassifier for predict_proba."""
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
# 🔽 TOP-LEVEL GNN WORKER (must be picklable) + SUBPROCESS WRAPPER
# ============================================================================
def _gnn_worker(evidence):
    """Run GNN hybrid_query in a clean subprocess."""
    import importlib.util
    from pathlib import Path
    gnn_dir = Path(__file__).parent.parent / "2.GNN"
    query_file = gnn_dir / "query.py"
    spec = importlib.util.spec_from_file_location("gnn_worker", query_file)
    gnn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gnn_mod)
    return gnn_mod.hybrid_query(evidence)

def run_gnn_in_subprocess(evidence):
    """Call GNN in a separate process to avoid memory fragmentation."""
    with multiprocessing.Pool(1) as pool:
        result = pool.apply_async(_gnn_worker, (evidence,))
        try:
            return result.get(timeout=300)   # 5 minutes
        except multiprocessing.TimeoutError:
            print("⚠ GNN subprocess timed out – skipping GNN results.")
            return []
# ============================================================================

# Path to Jaccards_correlation.py
jaccard_dir = Path("..") / "Jaccards_Correlation"
jaccard_file = jaccard_dir / "Jaccards_correlation.py"

spec = importlib.util.spec_from_file_location("jaccards_module", jaccard_file)
jaccard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jaccard_module)

# Path to TVD_featureIng.py
TVD_dir = Path("..") / "TVD_Total_Variation_Distance"
TVD_file = TVD_dir / "TVD_featureIng.py"

spec = importlib.util.spec_from_file_location("TVD_module", TVD_file)
TVD_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TVD_module)

# Path to featureMatching.py
featureMatching_dir = Path("..") / "featureMatching"
featureMatching_file = featureMatching_dir / "featureMatching.py"

spec = importlib.util.spec_from_file_location("featureMatching_module", featureMatching_file)
featureMatching_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(featureMatching_module)

# Store original state
original_dir = os.getcwd()
original_sys_path = sys.path.copy()

# Define all directories
directories = {
    "xgboost": Path("..") / "1.XGboost",
    "gnn": Path("..") / "2.GNN",
    "siamese": Path("..") / "3.SiameseNetwork",
    "mil": Path("..") / "4.Attention-Based_MIL",
    "rf": Path("..") / "5.RF_FeatureEngineering",
}

# Dictionary to store all imported modules
modules = {}

try:
    for module_name, dir_path in directories.items():
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            continue

        os.chdir(dir_path)
        sys.path.insert(0, os.getcwd())

        spec = importlib.util.spec_from_file_location(f"{module_name}_query", "query.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        modules[module_name] = module

        print(f"✓ Imported {module_name} from {os.getcwd()}/query.py")

        sys.path.remove(os.getcwd())

finally:
    os.chdir(original_dir)
    sys.path = original_sys_path

xgboost_query = modules.get("xgboost")
gnn_query = modules.get("gnn")
siamese_query = modules.get("siamese")
mil_query = modules.get("mil")
rf_query = modules.get("rf")

print(f"\n✓ Successfully imported {len(modules)} modules")
for name in modules.keys():
    print(f"  - {name}")

##########################################################################

def process_apt_data(csv_file_path, apt_name):
    df = pd.read_csv(csv_file_path)
    apt_data = df[df['APTname'] == apt_name]
    techniques = sorted(apt_data['techniques_str'].dropna().unique().tolist())
    software = sorted(apt_data['software_str'].dropna().unique().tolist())
    capec = sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1])
    cwe = sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1])
    cves = sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1])
    pulse = sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
    return {
        'techniques': techniques,
        'software': software,
        'capec': capec,
        'cwe': cwe,
        'cve': cves,
        'pulse': pulse
    }

def hybrid_rank_aggregate(lists, weight_xgboost, weight_gnn,
                          weight_siamese, weight_mil, weight_rf,
                          weight_featureMatching, weight_jaccard, weight_tvd):
    k = 60
    n_lists = len(lists)
    weights = [weight_xgboost, weight_gnn, weight_siamese, weight_mil,
               weight_rf, weight_jaccard, weight_tvd, weight_featureMatching]
    weights = np.array(weights) / np.sum(weights)

    rank_dict = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']
            if name not in rank_dict:
                rank_dict[name] = [None] * n_lists
            rank_dict[name][list_idx] = rank

    results = []
    for name, ranks in rank_dict.items():
        rrf_score = 0.0
        for i, rank in enumerate(ranks):
            if rank is not None:
                rrf_score += weights[i] / (k + rank)
        results.append({'name': name, 'rrf_score': rrf_score})

    results.sort(key=lambda x: (-x['rrf_score'], x['name']))
    return [(item['name'], i+1) for i, item in enumerate(results)]

def automated_query_for_eval(evidence_first_batch):
    try:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        XGboost_results = xgboost_query.hybrid_query(evidence_first_batch)
        GNN_results = run_gnn_in_subprocess(evidence_first_batch)
        Siamese_results = siamese_query.hybrid_query(evidence_first_batch)
        MIL_results = mil_query.hybrid_query(evidence_first_batch)
        RF_results = rf_query.hybrid_query(evidence_first_batch)
        Jaccards_result = jaccard_module.hybrid_query(evidence_first_batch)
        TVD_result = TVD_module.hybrid_query(evidence_first_batch)
        featureMatching_result = featureMatching_module.hybrid_query(evidence_first_batch)

        all_lists = [XGboost_results, GNN_results, Siamese_results, MIL_results,
                     RF_results, Jaccards_result, TVD_result, featureMatching_result]
        result = hybrid_rank_aggregate(all_lists)

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        return result
    except Exception as e:
        print(f"Error: {e}")

def remotequery(evidence_first_batch):
    gc.collect()
    XGboost_results = xgboost_query.hybrid_query(evidence_first_batch)
    gc.collect()
    GNN_results = run_gnn_in_subprocess(evidence_first_batch)
    gc.collect()
    Siamese_results = siamese_query.hybrid_query(evidence_first_batch)
    gc.collect()
    MIL_results = mil_query.hybrid_query(evidence_first_batch)
    gc.collect()
    RF_results = rf_query.hybrid_query(evidence_first_batch)
    gc.collect()
    Jaccards_result = jaccard_module.hybrid_query(evidence_first_batch)
    TVD_result = TVD_module.hybrid_query(evidence_first_batch)
    featureMatching_result = featureMatching_module.hybrid_query(evidence_first_batch)

    all_lists = [XGboost_results, GNN_results, Siamese_results, MIL_results,
                 RF_results, Jaccards_result, TVD_result, featureMatching_result]

    result = hybrid_rank_aggregate(
        all_lists,
        weight_xgboost=1, 
        weight_gnn=1, 
        weight_siamese=1,
        weight_mil=1, 
        weight_rf=1, 
        weight_featureMatching=1,
        weight_jaccard=1, 
        weight_tvd=1
    )


    #weight based only on self evaluation
    # result = hybrid_rank_aggregate(
    #         all_lists,
    #         weight_jaccard=8/8,
    #         weight_featureMatching=8/8,
    #         weight_siamese=1/8,
    #         weight_tvd=1/8,
    #         weight_rf=1/8,
    #         weight_xgboost=1/8,
    #         weight_gnn=1/8,
    #         weight_mil=1/8
    #     )

    #weight determined on precision evaluation through dataset and CTI eval testing
    # result = hybrid_rank_aggregate(
    #         all_lists,
    #         weight_featureMatching=8/8,
    #         weight_jaccard=7/8,
    #         weight_xgboost=7/8,
    #         weight_mil=6/8,
    #         weight_gnn=1/8,
    #         weight_tvd=3/8,
    #         weight_siamese=1/8,
    #         weight_rf=3/8
    #     )

    final_result = [{'rank': rank, 'apt_name': name} for name, rank in result]
    return final_result

def main():
    import json
    with open('../query_input.json', 'r') as f:
        evidence_first_batch = json.load(f)

    gc.collect()
    XGboost_results = xgboost_query.hybrid_query(evidence_first_batch)
    gc.collect()
    GNN_results = run_gnn_in_subprocess(evidence_first_batch)
    gc.collect()
    Siamese_results = siamese_query.hybrid_query(evidence_first_batch)
    gc.collect()
    MIL_results = mil_query.hybrid_query(evidence_first_batch)
    gc.collect()
    RF_results = rf_query.hybrid_query(evidence_first_batch)
    gc.collect()
    Jaccards_result = jaccard_module.hybrid_query(evidence_first_batch)
    TVD_result = TVD_module.hybrid_query(evidence_first_batch)
    featureMatching_result = featureMatching_module.hybrid_query(evidence_first_batch)

    all_lists = [XGboost_results, GNN_results, Siamese_results, MIL_results,
                 RF_results, Jaccards_result, TVD_result, featureMatching_result]

    result = hybrid_rank_aggregate(
        all_lists,
        weight_xgboost=1, 
        weight_gnn=1, 
        weight_siamese=1,
        weight_mil=1, 
        weight_rf=1, 
        weight_featureMatching=1,
        weight_jaccard=1, 
        weight_tvd=1
    )

    for name, rank in result:
        print(f"  {name}: {rank}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()