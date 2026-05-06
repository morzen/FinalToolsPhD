import os
# Fix OpenMP crash when multiple libraries ship their own libomp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import importlib.util
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import traceback
import contextlib
import multiprocessing as mp
import xgboost as xgb

# ============================================================================
# 🔽 MUST be defined before any pickle.load that uses it – 
#    same as in train.py / hybrid_query.py
# ============================================================================
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

# ------------------------------------------------------------------------------
# Module loading (unchanged)
# ------------------------------------------------------------------------------

# Suppress output during imports (temporarily)
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

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

# Thread lock for safe printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

try:
    for module_name, dir_path in directories.items():
        if not dir_path.exists():
            safe_print(f"⚠️  Directory not found: {dir_path}")
            continue

        os.chdir(dir_path)
        sys.path.insert(0, os.getcwd())

        spec = importlib.util.spec_from_file_location(f"{module_name}_query", "query.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        modules[module_name] = module
        safe_print(f"✓ Imported {module_name} from {os.getcwd()}/query.py")

        sys.path.remove(os.getcwd())
finally:
    os.chdir(original_dir)
    sys.path = original_sys_path

xgboost_query = modules.get("xgboost")
gnn_query = modules.get("gnn")
siamese_query = modules.get("siamese")
mil_query = modules.get("mil")
rf_query = modules.get("rf")

safe_print(f"\n✓ Successfully imported {len(modules)} modules")
for name in modules.keys():
    safe_print(f"  - {name}")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# ------------------------------------------------------------------------------
# Optimized functions
# ------------------------------------------------------------------------------

def preload_apt_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    apt_data = {}
    for apt_name, group in df.groupby('APTname'):
        apt_data[apt_name] = {
            'techniques': sorted(group['techniques_str'].dropna().unique().tolist()),
            'software': sorted(group['software_str'].dropna().unique().tolist()),
            'capec': sorted([x for x in group['CAPEC_ID'].dropna().unique().tolist() if x != -1]),
            'cwe': sorted([x for x in group['CWE_ID'].dropna().unique().tolist() if x != -1]),
            'cve': sorted([x for x in group['CVE_ID'].dropna().unique().tolist() if x != -1]),
            'pulse': sorted([x for x in group['Pulse_ID'].dropna().unique().tolist() if x != -1])
        }
    return apt_data

def shuffler_and_divider(evidence_dict):
    shuffled = {}
    for key, items in evidence_dict.items():
        if key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            half_length = len(shuffled_items) // 2
            shuffled[key] = shuffled_items[:half_length]
        else:
            shuffled[key] = items
    return shuffled

def hybrid_rank_aggregate(lists):
    n_lists = len(lists)
    rank_dict = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']
            rank_dict.setdefault(name, {})[list_idx] = rank

    df = pd.DataFrame(rank_dict).T.fillna(np.nan)
    avg = df.mean(axis=1)
    median = df.median(axis=1)
    min_rank = df.min(axis=1)
    rank_range = df.max(axis=1) - min_rank

    results_df = pd.DataFrame({'avg': avg, 'median': median, 'min': min_rank, 'range': rank_range})
    results_df.sort_values(by=['avg', 'median', 'min', 'range'], inplace=True)
    return [(name, i+1) for i, name in enumerate(results_df.index)]

def run_single_query(evidence, module):
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result = module.hybrid_query(evidence)
        return result
    except Exception as e:
        safe_print(f"  ⚠️ Module error: {str(e)[:50]}")
        return None

def automated_query_for_eval(evidence):
    modules_to_run = [
        ("xgboost", xgboost_query),
        ("gnn", gnn_query),
        ("siamese", siamese_query),
        ("mil", mil_query),
        ("rf", rf_query),
        ("jaccard", jaccard_module),
        ("tvd", TVD_module),
        ("featureMatching", featureMatching_module)
    ]
    valid_modules = [(name, module) for name, module in modules_to_run if module is not None]

    results = []
    for name, module in valid_modules:
        result = run_single_query(evidence, module)
        if result is not None:
            results.append(result)

    if len(results) >= 3:
        return hybrid_rank_aggregate(results)
    else:
        safe_print(f"  ⚠️ Only {len(results)} modules succeeded, skipping")
        return []

def process_single_apt(name, evidence, shuffle_flag):
    try:
        if shuffle_flag:
            evidence = shuffler_and_divider(evidence)
        result = automated_query_for_eval(evidence)
        return name, result
    except Exception as e:
        safe_print(f"✗ Error processing {name}: {str(e)[:100]}")
        return name, []

def eval_precision(unique_apt_names_list, givenEvidence):
    total = len(unique_apt_names_list)
    fulldata_top1 = fulldata_top5 = fulldata_top10 = fulldata_top20 = 0

    for name in unique_apt_names_list:
        ranking_list = givenEvidence.get(name, [])
        if not ranking_list:
            continue
        rank = None
        for apt_name, apt_rank in ranking_list:
            if apt_name == name:
                rank = apt_rank
                break
        if rank is not None:
            if rank == 1:
                fulldata_top1 += 1; fulldata_top5 += 1; fulldata_top10 += 1; fulldata_top20 += 1
            elif rank <= 5:
                fulldata_top5 += 1; fulldata_top10 += 1; fulldata_top20 += 1
            elif rank <= 10:
                fulldata_top10 += 1; fulldata_top20 += 1
            elif rank <= 20:
                fulldata_top20 += 1

    if total > 0:
        print(f"Top-1: {fulldata_top1}/{total} ({fulldata_top1/total*100:.2f}%)")
        print(f"Top-5: {fulldata_top5}/{total} ({fulldata_top5/total*100:.2f}%)")
        print(f"Top-10: {fulldata_top10}/{total} ({fulldata_top10/total*100:.2f}%)")
        print(f"Top-20: {fulldata_top20}/{total} ({fulldata_top20/total*100:.2f}%)")

def create_apt_ranking_heatmap(apt_dict):
    all_apt_names = sorted(apt_dict.keys())
    df = pd.DataFrame(index=all_apt_names, columns=all_apt_names).fillna(0)
    for query_apt in all_apt_names:
        rankings = apt_dict.get(query_apt, [])
        ranking_dict = dict(rankings)
        for ranked_apt in all_apt_names:
            if ranked_apt in ranking_dict:
                rank = ranking_dict[ranked_apt]
                inverted_score = 181 - rank + 1
                df.loc[query_apt, ranked_apt] = inverted_score

    plt.figure(figsize=(24, 20))
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    sns.heatmap(df, cmap=cmap, vmin=1, vmax=181, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'APT Ranking (Green=Best/Rank 1, Red=Worst/Rank 181)',
                          'ticks': [1, 90, 181]})
    plt.title("APT Ranking Similarity Matrix\n(Green = Rank 1, Yellow = Rank ~90, Red = Rank 181)",
              fontsize=18, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    cbar = plt.gcf().axes[-1]
    cbar.set_yticklabels(['Rank 181 (Red)', 'Rank ~90 (Yellow)', 'Rank 1 (Green)'])
    plt.tight_layout()
    plt.savefig('apt_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Heatmap saved as 'apt_ranking_heatmap.png'")
    plt.show()

def parallel_process_apts(apt_names, apt_data, shuffle_flag, max_workers=9):
    results = {}
    total = len(apt_names)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_apt, name, apt_data[name], shuffle_flag): name
                   for name in apt_names}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            safe_print(f"✓ Completed {name} (shuffled={shuffle_flag}) [{len(results)}/{total}]")
    return results

def main():
    csv_file_path = "../BaseData/TestFinal_df.csv"
    safe_print("Loading APT data...")
    apt_data = preload_apt_data(csv_file_path)
    apt_names = list(apt_data.keys())
    total_apts = len(apt_names)
    safe_print(f"Loaded data for {total_apts} APTs.")

    max_workers = min(9, total_apts)
    safe_print(f"Using {max_workers} parallel workers.")

    safe_print("\n" + "="*50)
    safe_print("Processing First Batch (Full Data)")
    safe_print("="*50)
    full_results = parallel_process_apts(apt_names, apt_data, shuffle_flag=False, max_workers=max_workers)

    safe_print("\n" + "="*50)
    safe_print("Processing Second Batch (Shuffled & Divided Data)")
    safe_print("="*50)
    shuffled_results = parallel_process_apts(apt_names, apt_data, shuffle_flag=True, max_workers=max_workers)

    print("\n" + "="*50)
    print("Results for Full Data")
    print("="*50)
    eval_precision(apt_names, full_results)

    print("\n" + "="*50)
    print("Results for Shuffled & Divided Data")
    print("="*50)
    eval_precision(apt_names, shuffled_results)

    if full_results:
        create_apt_ranking_heatmap(full_results)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)   # safe for Mac
    main()