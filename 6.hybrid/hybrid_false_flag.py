import os
# Fix OpenMP crash when multiple libraries each ship their own libomp
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
import pickle
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import time
import gc

# Set random seed for reproducibility
random.seed(42)

# ===== SUPPRESS MODULE IMPORT OUTPUT =====
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
        sys.path.remove(os.getcwd())
finally:
    os.chdir(original_dir)
    sys.path = original_sys_path

xgboost_query = modules.get("xgboost")
gnn_query = modules.get("gnn")
siamese_query = modules.get("siamese")
mil_query = modules.get("mil")
rf_query = modules.get("rf")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# ============================================================================
# 🔽 TOP-LEVEL GNN WORKER + SUBPROCESS WRAPPER (must be pickleable)
# ============================================================================
def _gnn_worker_eval(evidence):
    import importlib.util
    from pathlib import Path
    gnn_dir = Path(__file__).parent.parent / "2.GNN"
    query_file = gnn_dir / "query.py"
    spec = importlib.util.spec_from_file_location("gnn_worker", query_file)
    gnn_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gnn_mod)
    return gnn_mod.hybrid_query(evidence)

def run_gnn_in_subprocess(evidence):
    with mp.Pool(1) as pool:
        result = pool.apply_async(_gnn_worker_eval, (evidence,))
        try:
            return result.get(timeout=300)
        except mp.TimeoutError:
            print("⚠ GNN subprocess timed out – skipping GNN results.")
            return []
# ============================================================================

# ===== UNIFIED MIXING FUNCTIONS =====

def unified_full_mix(apt1, apt2, rng=None, attacker_fraction=0.3, max_count=10):
    if rng is None:
        rng = random.Random()
    categories = ['techniques', 'software', 'capec', 'cwe', 'cves', 'pulse']
    mixed = {}
    for cat in categories:
        atk_list = list(apt1.get(cat, []))
        mim_list = list(apt2.get(cat, []))
        if not atk_list and not mim_list:
            mixed[cat] = []
            continue
        result = []
        atk_copy = atk_list[:]
        mim_copy = mim_list[:]
        total_needed = min(max_count, len(atk_copy) + len(mim_copy))
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_copy:
                chosen = rng.choice(atk_copy)
                result.append(chosen)
                atk_copy.remove(chosen)
            elif mim_copy:
                chosen = rng.choice(mim_copy)
                result.append(chosen)
                mim_copy.remove(chosen)
            else:
                break
        mixed[cat] = result
    return mixed

def unified_partial_mix(full_mixed_dict, rng=None, volume_fraction=0.5):
    if rng is None:
        rng = random.Random()
    partial = {}
    for cat, feats in full_mixed_dict.items():
        feats = list(feats)
        rng.shuffle(feats)
        keep = max(1, int(len(feats) * volume_fraction)) if feats else 0
        partial[cat] = feats[:keep]
    return partial

# ===== DATA PROCESSING =====

def process_apt_data(apt_name, apt_data_cache=None):
    if apt_data_cache is not None:
        return apt_data_cache[apt_name]
    df = pd.read_csv(csv_file_path)
    apt_data = df[df['APTname'] == apt_name]
    return {
        'techniques': sorted(apt_data['techniques_str'].dropna().unique().tolist()),
        'software': sorted(apt_data['software_str'].dropna().unique().tolist()),
        'capec': sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1]),
        'cwe': sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1]),
        'cves': sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1]),
        'pulse': sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
    }

def hybrid_rank_aggregate(lists):
    n_lists = len(lists)
    rank_matrix = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']
            if name not in rank_matrix:
                rank_matrix[name] = [None] * n_lists
            rank_matrix[name][list_idx] = rank
    results = []
    for name, ranks in rank_matrix.items():
        ranks = [r for r in ranks if r is not None]
        avg_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        min_rank = min(ranks)
        max_rank = max(ranks)
        rank_range = max_rank - min_rank
        results.append({
            'name': name,
            'avg': avg_rank,
            'median': median_rank,
            'min': min_rank,
            'range': rank_range
        })
    results.sort(key=lambda x: (x['avg'], x['median'], x['min'], x['range'], x['name']))
    return [(item['name'], i+1) for i, item in enumerate(results)]

# ===== MODIFIED WORKER FUNCTION (GNN runs in subprocess) =====

def automated_query_for_eval(evidence):
    try:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        XGboost_results = xgboost_query.hybrid_query(evidence)
        gc.collect()
        GNN_results = run_gnn_in_subprocess(evidence)
        gc.collect()
        Siamese_results = siamese_query.hybrid_query(evidence)
        MIL_results = mil_query.hybrid_query(evidence)
        RF_results = rf_query.hybrid_query(evidence)
        Jaccards_result = jaccard_module.hybrid_query(evidence)
        TVD_result = TVD_module.hybrid_query(evidence)
        featureMatching_result = featureMatching_module.hybrid_query(evidence)

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

def worker_function(apt1, apt2, shuffle_n_divide, seed=None):
    rng = random.Random(seed) if seed is not None else random.Random()
    full_mixed = unified_full_mix(apt1, apt2, rng, attacker_fraction=0.3, max_count=10)
    if shuffle_n_divide:
        mixed_evidence = unified_partial_mix(full_mixed, rng, volume_fraction=0.5)
    else:
        mixed_evidence = full_mixed
    return automated_query_for_eval(mixed_evidence)

def eval_precision(unique_apt_names_list, givenEvidence):
    metrics = {
        'real': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0},
        'mimicked': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0},
        'both': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0},
        'neither': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0}
    }
    total_comparisons = 0
    for attacker_name, mimicked_dict in givenEvidence.items():
        for mimicked_name in mimicked_dict.keys():
            if attacker_name != mimicked_name:
                total_comparisons += 1
    thresholds = [1, 5, 10, 20]
    for attacker_name, mimicked_dict in givenEvidence.items():
        for mimicked_name, ranking_list in mimicked_dict.items():
            if attacker_name == mimicked_name:
                continue
            self_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == attacker_name:
                    self_rank = int(rank)
                    break
            if self_rank is None:
                self_rank = len(unique_apt_names_list)
            mimicked_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == mimicked_name:
                    mimicked_rank = int(rank)
                    break
            if mimicked_rank is None:
                mimicked_rank = len(unique_apt_names_list)
            for threshold in thresholds:
                self_in = 1 <= self_rank <= threshold
                other_in = 1 <= mimicked_rank <= threshold
                if self_in:
                    metrics['real'][f'top{threshold}'] += 1
                if other_in:
                    metrics['mimicked'][f'top{threshold}'] += 1
                if self_in and other_in:
                    metrics['both'][f'top{threshold}'] += 1
                if not self_in and not other_in:
                    metrics['neither'][f'top{threshold}'] += 1
    print("\n" + "="*60)
    print(" FALSE-FLAG ATTACK ANALYSIS - RANKING RESULTS")
    print("="*60)
    for threshold in thresholds:
        a = metrics['real'][f'top{threshold}']
        m = metrics['mimicked'][f'top{threshold}']
        b = metrics['both'][f'top{threshold}']
        n = metrics['neither'][f'top{threshold}']
        ap = a/total_comparisons*100 if total_comparisons else 0
        mp = m/total_comparisons*100 if total_comparisons else 0
        print(f"\n Top-{threshold}:")
        print(f"  Attacker (Detection): {a}/{total_comparisons} ({ap:.1f}%)")
        print(f"  Mimicked (Deception): {m}/{total_comparisons} ({mp:.1f}%)")
        print(f"  Both: {b}/{total_comparisons} ({b/total_comparisons*100:.1f}%)")
        print(f"  Neither: {n}/{total_comparisons} ({n/total_comparisons*100:.1f}%)")
        print(f"   Deception: {mp:.1f}% |  Detection: {ap:.1f}%")
    return metrics

def plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, apt_names):
    print("\n🔥 CREATING FALSE-FLAG DECEPTION HEATMAP...")
    n = len(apt_names)
    name_to_idx = {name: i for i, name in enumerate(apt_names)}
    deception_matrix = np.zeros((n, n), dtype=float)
    count_matrix = np.zeros((n, n), dtype=int)
    for attacker_name, mimicked_dict in Full_FalseFlag_evidencetable.items():
        if attacker_name not in name_to_idx:
            continue
        ai = name_to_idx[attacker_name]
        for mimicked_name, ranking_list in mimicked_dict.items():
            if mimicked_name not in name_to_idx or attacker_name == mimicked_name:
                continue
            mi = name_to_idx[mimicked_name]
            self_rank = next((r for a, r in ranking_list if a == attacker_name), n)
            mim_rank = next((r for a, r in ranking_list if a == mimicked_name), n)
            raw = self_rank - mim_rank
            norm = raw / (n - 1)
            deception_matrix[ai, mi] += norm
            count_matrix[ai, mi] += 1
    mask = count_matrix > 0
    deception_matrix[mask] /= count_matrix[mask]
    np.fill_diagonal(deception_matrix, 0)
    plt.figure(figsize=(20, 18))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Deception Score'})
    plt.title('False-Flag Deception Heatmap')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    from datetime import datetime
    path = f'false_flag_deception_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    return deception_matrix

def process_pair(args):
    name1, name2, apt_cache = args
    if name1 == name2:
        return None
    try:
        seed = hash((name1, name2)) % (2**32)
        apt1 = apt_cache[name1]
        apt2 = apt_cache[name2]
        full = worker_function(apt1, apt2, False, seed)
        shuffled = worker_function(apt1, apt2, True, seed)
        return (name1, name2, full, shuffled)
    except Exception as e:
        print(f"Error: {e}")
        return None

def precompute_apt_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    apt_names = df['APTname'].unique().tolist()
    cache = {}
    for apt in tqdm(apt_names, desc="Caching APTs"):
        ad = df[df['APTname'] == apt]
        cache[apt] = {
            'techniques': sorted(ad['techniques_str'].dropna().unique().tolist()),
            'software': sorted(ad['software_str'].dropna().unique().tolist()),
            'capec': sorted([x for x in ad['CAPEC_ID'].dropna().unique().tolist() if x != -1]),
            'cwe': sorted([x for x in ad['CWE_ID'].dropna().unique().tolist() if x != -1]),
            'cves': sorted([x for x in ad['CVE_ID'].dropna().unique().tolist() if x != -1]),
            'pulse': sorted([x for x in ad['Pulse_ID'].dropna().unique().tolist() if x != -1])
        }
    return cache, apt_names

def main():
    csv_file_path = "../BaseData/TestFinal_df.csv"
    checkpoint_file = 'false_flag_checkpoint.pkl'
    apt_cache, unique_apt_names_list = precompute_apt_data(csv_file_path)
    pairs = [(n1, n2) for n1 in unique_apt_names_list for n2 in unique_apt_names_list if n1 != n2]
    total_pairs = len(pairs)
    print(f"Total pairs: {total_pairs}")
    Full = {n: {} for n in unique_apt_names_list}
    Shuffled = {n: {} for n in unique_apt_names_list}
    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            ck = pickle.load(f)
            Full = ck['full']
            Shuffled = ck['shuffled']
            processed = set(ck['processed_pairs'])
        print(f"Loaded {len(processed)} processed pairs")
    remaining = [p for p in pairs if p not in processed]
    if remaining:
        args = [(n1, n2, apt_cache) for n1, n2 in remaining]
        start = time.time()
        with Pool(processes=8) as pool:
            with tqdm(total=len(remaining), desc="Processing pairs") as pbar:
                for res in pool.imap_unordered(process_pair, args):
                    if res:
                        n1, n2, full, shuffled = res
                        Full[n1][n2] = full
                        Shuffled[n1][n2] = shuffled
                        processed.add((n1, n2))
                        pbar.update(1)
                        if len(processed) % 50 == 0:
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump({'full': Full, 'shuffled': Shuffled,
                                             'processed_pairs': list(processed)}, f)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'full': Full, 'shuffled': Shuffled,
                         'processed_pairs': list(processed)}, f)
        print(f"Done in {(time.time()-start)/60:.1f} min")
    print("\n=== FULL RESULTS ===")
    eval_precision(unique_apt_names_list, Full)
    print("\n=== SHUFFLED RESULTS ===")
    eval_precision(unique_apt_names_list, Shuffled)
    plot_false_flag_deception_heatmap(Full, unique_apt_names_list)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()