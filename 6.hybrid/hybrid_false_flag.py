import os
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
import contextlib
import xgboost as xgb

# ============================================================================
# 🔽  PicklableXGBWrapper – MUST be present so pickle can load the model
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

random.seed(42)

# --------------------------------------------------------------------------
# 🔽  USE ALL 181 APTs (change to a number for testing)
# --------------------------------------------------------------------------
NUM_APTS = None   # all 181

# --------------------------------------------------------------------------
# Suppress import output
# --------------------------------------------------------------------------
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Load external modules
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

original_dir = os.getcwd()
original_sys_path = sys.path.copy()

directories = {
    "xgboost": Path("..") / "1.XGboost",
    "gnn": Path("..") / "2.GNN",
    "siamese": Path("..") / "3.SiameseNetwork",
    "mil": Path("..") / "4.Attention-Based_MIL",
    "rf": Path("..") / "5.RF_FeatureEngineering",
}

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

# ---------- Evidence mixing ----------
def unified_full_mix(apt1, apt2, rng=None, attacker_fraction=0.3, max_count=10):
    if rng is None: rng = random.Random()
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
                result.append(rng.choice(atk_copy))
                atk_copy.remove(result[-1])
            elif mim_copy:
                result.append(rng.choice(mim_copy))
                mim_copy.remove(result[-1])
            else:
                break
        mixed[cat] = result
    return mixed

def unified_partial_mix(full_mixed_dict, rng=None, volume_fraction=0.5):
    if rng is None: rng = random.Random()
    partial = {}
    for cat, feats in full_mixed_dict.items():
        feats = list(feats)
        rng.shuffle(feats)
        keep = max(1, int(len(feats) * volume_fraction)) if feats else 0
        partial[cat] = feats[:keep]
    return partial

# ---------- Rank aggregation ----------
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
        valid = [r for r in ranks if r is not None]
        if not valid: continue
        avg_rank = np.mean(valid)
        median_rank = np.median(valid)
        min_rank = min(valid)
        rank_range = max(valid) - min_rank
        results.append({'name': name, 'avg': avg_rank, 'median': median_rank,
                        'min': min_rank, 'range': rank_range})
    results.sort(key=lambda x: (x['avg'], x['median'], x['min'], x['range'], x['name']))
    return [(item['name'], i+1) for i, item in enumerate(results)]

# ---------- Safe module query (no debug prints) ----------
def safe_query(evidence, module_name):
    global xgboost_query, gnn_query, siamese_query, mil_query, rf_query, jaccard_module, TVD_module, featureMatching_module
    modules_dict = {
        "xgboost": xgboost_query, "gnn": gnn_query, "siamese": siamese_query,
        "mil": mil_query, "rf": rf_query, "jaccard": jaccard_module,
        "tvd": TVD_module, "featureMatching": featureMatching_module,
    }
    module = modules_dict.get(module_name)
    if module is None:
        return []
    try:
        with open(os.devnull, 'w') as devnull, \
             contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return module.hybrid_query(evidence)
    except Exception as e:
        return []

# ---------- Ensemble query ----------
def automated_query_for_eval(evidence):
    all_lists = []
    for mod_name in ["xgboost", "gnn", "siamese", "mil", "rf", "jaccard", "tvd", "featureMatching"]:
        res = safe_query(evidence, mod_name)
        all_lists.append(res)
        gc.collect()
    valid = [lst for lst in all_lists if lst]
    return hybrid_rank_aggregate(valid) if len(valid) >= 3 else []

def worker_function(apt1, apt2, shuffle_n_divide, seed=None):
    rng = random.Random(seed) if seed is not None else random.Random()
    full_mixed = unified_full_mix(apt1, apt2, rng, attacker_fraction=0.3, max_count=10)
    if shuffle_n_divide:
        mixed_evidence = unified_partial_mix(full_mixed, rng, volume_fraction=0.5)
    else:
        mixed_evidence = full_mixed
    return automated_query_for_eval(mixed_evidence)

# ---------- Evaluation ----------
def eval_precision(unique_apt_names_list, givenEvidence):
    metrics = {'real': {'top1':0,'top5':0,'top10':0,'top20':0},
               'mimicked': {'top1':0,'top5':0,'top10':0,'top20':0},
               'both': {'top1':0,'top5':0,'top10':0,'top20':0},
               'neither': {'top1':0,'top5':0,'top10':0,'top20':0}}
    total_comparisons = 0
    for att, mdict in givenEvidence.items():
        for mim, rlist in mdict.items():
            if rlist is not None and att != mim:
                total_comparisons += 1
    if total_comparisons == 0:
        print("\n⚠ No valid comparisons.")
        return metrics
    thresholds = [1,5,10,20]
    for att, mdict in givenEvidence.items():
        for mim, rlist in mdict.items():
            if rlist is None or att == mim: continue
            self_rank = None
            for a,r in rlist:
                if a == att: self_rank = int(r); break
            if self_rank is None: self_rank = len(unique_apt_names_list)
            mim_rank = None
            for a,r in rlist:
                if a == mim: mim_rank = int(r); break
            if mim_rank is None: mim_rank = len(unique_apt_names_list)
            for th in thresholds:
                s_in = 1 <= self_rank <= th
                m_in = 1 <= mim_rank <= th
                if s_in: metrics['real'][f'top{th}'] += 1
                if m_in: metrics['mimicked'][f'top{th}'] += 1
                if s_in and m_in: metrics['both'][f'top{th}'] += 1
                if not s_in and not m_in: metrics['neither'][f'top{th}'] += 1
    print("\n" + "="*60)
    print(" FALSE-FLAG ATTACK ANALYSIS - RANKING RESULTS")
    print("="*60)
    for th in thresholds:
        a = metrics['real'][f'top{th}']
        m = metrics['mimicked'][f'top{th}']
        b = metrics['both'][f'top{th}']
        n = metrics['neither'][f'top{th}']
        ap = a/total_comparisons*100 if total_comparisons else 0
        mp = m/total_comparisons*100 if total_comparisons else 0
        print(f"\n Top-{th}:")
        print(f"  Attacker (Detection): {a}/{total_comparisons} ({ap:.1f}%)")
        print(f"  Mimicked (Deception): {m}/{total_comparisons} ({mp:.1f}%)")
        print(f"  Both: {b}/{total_comparisons} ({b/total_comparisons*100:.1f}%)")
        print(f"  Neither: {n}/{total_comparisons} ({n/total_comparisons*100:.1f}%)")
    return metrics

def plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, apt_names):
    print("\n🔥 CREATING FALSE-FLAG DECEPTION HEATMAP...")
    n = len(apt_names)
    name_to_idx = {name:i for i,name in enumerate(apt_names)}
    deception_matrix = np.zeros((n,n))
    count_matrix = np.zeros((n,n), dtype=int)
    for att, mdict in Full_FalseFlag_evidencetable.items():
        if att not in name_to_idx: continue
        ai = name_to_idx[att]
        for mim, rlist in mdict.items():
            if rlist is None or mim not in name_to_idx or att == mim: continue
            mi = name_to_idx[mim]
            self_rank = next((r for a,r in rlist if a==att), n)
            mim_rank = next((r for a,r in rlist if a==mim), n)
            raw = self_rank - mim_rank
            norm = raw / (n-1)
            deception_matrix[ai,mi] += norm
            count_matrix[ai,mi] += 1
    mask = count_matrix > 0
    deception_matrix[mask] /= count_matrix[mask]
    np.fill_diagonal(deception_matrix,0)
    plt.figure(figsize=(20,18))
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

def process_pair(args):
    name1, name2, apt_cache = args
    if name1 == name2: return None
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

    if NUM_APTS is not None:
        unique_apt_names_list = unique_apt_names_list[:NUM_APTS]
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️  Deleted old checkpoint (testing with {NUM_APTS} APTs).")

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
                        if full is not None: Full[n1][n2] = full
                        if shuffled is not None: Shuffled[n1][n2] = shuffled
                        processed.add((n1, n2))
                        pbar.update(1)
                        if len(processed) % 50 == 0:
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump({'full':Full, 'shuffled':Shuffled,
                                             'processed_pairs':list(processed)}, f)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'full':Full, 'shuffled':Shuffled,
                         'processed_pairs':list(processed)}, f)
        print(f"Done in {(time.time()-start)/60:.1f} min")

    print("\n=== FULL RESULTS ===")
    eval_precision(unique_apt_names_list, Full)
    print("\n=== SHUFFLED RESULTS ===")
    eval_precision(unique_apt_names_list, Shuffled)
    plot_false_flag_deception_heatmap(Full, unique_apt_names_list)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()