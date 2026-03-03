import pandas as pd
import importlib.util
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import random
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import time

# Set random seed for reproducibility
random.seed(42)

# ===== LOAD MODULES (same as your original) =====
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

# ===== YOUR ORIGINAL FUNCTIONS =====
def process_apt_data(apt_name, apt_data_cache=None):
    """Modified to use pre-computed cache"""
    if apt_data_cache is not None:
        return apt_data_cache[apt_name]
    
    # Fallback to original method
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

def shuffler_and_divider(evidence_dict):
    shuffled = {}
    for key, items in evidence_dict.items():
        if key in ['techniques', 'software', 'capec', 'cwe', 'cves', 'pulse']:
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            half_length = len(shuffled_items) // 2
            shuffled[key] = shuffled_items[:half_length]
        else:
            shuffled[key] = items
    return shuffled

def shuffle_remove_and_combine(apt1, apt2, apt1_ratio=0.3, apt2_ratio=0.7):
    combined = {}
    for category in apt1.keys():
        apt1_items = apt1[category]
        apt1_count = int(len(apt1_items) * apt1_ratio)
        apt1_sample = apt1_items[:apt1_count]
        
        apt2_items = apt2[category]
        apt2_count = int(len(apt2_items) * apt2_ratio)
        apt2_sample = apt2_items[-apt2_count:] if apt2_count > 0 else []
        
        combined_category = []
        seen = set()
        for item in apt1_sample + apt2_sample:
            if item not in seen:
                seen.add(item)
                combined_category.append(item)
        
        combined[category] = combined_category
    return combined

def automated_query_for_eval(evidence):
    try:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        XGboost_results = xgboost_query.hybrid_query(evidence)
        GNN_results = gnn_query.hybrid_query(evidence)
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

def worker_function(apt1, apt2, shuffle_n_divide):
    combined = shuffle_remove_and_combine(apt1, apt2)
    if shuffle_n_divide:
        combined = shuffler_and_divider(combined)
    return automated_query_for_eval(combined)

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
            
            # Find where the REAL ATTACKER appears in THIS ranking list
            self_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == attacker_name:  # Look for the attacker in the results
                    self_rank = int(rank)
                    break
            
            if self_rank is None:
                self_rank = len(unique_apt_names_list)
            
            # Find where the MIMICKED APT appears in THIS ranking list
            mimicked_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == mimicked_name:
                    mimicked_rank = int(rank)
                    break
            
            if mimicked_rank is None:
                mimicked_rank = len(unique_apt_names_list)
            
            for threshold in thresholds:
                self_in_threshold = 1 <= self_rank <= threshold
                other_in_threshold = 1 <= mimicked_rank <= threshold
                
                if self_in_threshold:
                    metrics['real'][f'top{threshold}'] += 1
                if other_in_threshold:
                    metrics['mimicked'][f'top{threshold}'] += 1
                if self_in_threshold and other_in_threshold:
                    metrics['both'][f'top{threshold}'] += 1
                if not self_in_threshold and not other_in_threshold:
                    metrics['neither'][f'top{threshold}'] += 1

    print("\n" + "="*60)
    print(" FALSE-FLAG ATTACK ANALYSIS - RANKING RESULTS")
    print("="*60)

    for threshold in thresholds:
        attacker = metrics['real'][f'top{threshold}']
        mimicked = metrics['mimicked'][f'top{threshold}']
        both = metrics['both'][f'top{threshold}']
        neither = metrics['neither'][f'top{threshold}']
        
        attacker_pct = (attacker / total_comparisons * 100) if total_comparisons > 0 else 0
        mimicked_pct = (mimicked / total_comparisons * 100) if total_comparisons > 0 else 0
        both_pct = (both / total_comparisons * 100) if total_comparisons > 0 else 0
        neither_pct = (neither / total_comparisons * 100) if total_comparisons > 0 else 0
        
        print(f"\n Top-{threshold}:")
        print(f"  Attacker (Detection): {attacker}/{total_comparisons} ({attacker_pct:.1f}%)")
        print(f"  Mimicked (Deception): {mimicked}/{total_comparisons} ({mimicked_pct:.1f}%)")
        print(f"  Both: {both}/{total_comparisons} ({both_pct:.1f}%)")
        print(f"  Neither: {neither}/{total_comparisons} ({neither_pct:.1f}%)")
        print(f"   Deception: {mimicked_pct:.1f}% |  Detection: {attacker_pct:.1f}%")

    print(f"\nTotal comparisons analyzed: {total_comparisons:,}")
    return metrics

def plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, apt_names):
    print("\n🔥 CREATING FALSE-FLAG DECEPTION HEATMAP...")
    
    n = len(apt_names)
    name_to_idx = {name: i for i, name in enumerate(apt_names)}
    
    deception_matrix = np.zeros((n, n), dtype=float)
    count_matrix = np.zeros((n, n), dtype=int)
    
    total_pairs = 0
    for attacker_name, mimicked_dict in Full_FalseFlag_evidencetable.items():
        if attacker_name not in name_to_idx:
            continue
            
        attacker_idx = name_to_idx[attacker_name]
        
        for mimicked_name, ranking_list in mimicked_dict.items():
            if mimicked_name not in name_to_idx or attacker_name == mimicked_name:
                continue
                
            mimicked_idx = name_to_idx[mimicked_name]
            
            # Find attacker's rank in this ranking list
            self_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == attacker_name:
                    self_rank = rank
                    break
            
            if self_rank is None:
                self_rank = n
            
            # Find mimicked APT's rank in this ranking list
            mimicked_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == mimicked_name:
                    mimicked_rank = rank
                    break
                    
            if mimicked_rank is None:
                mimicked_rank = n
            
            # Calculate normalized deception score
            # Positive = deception success (mimicked ranks higher than attacker)
            # Negative = detection success (attacker ranks higher than mimicked)
            raw_score = self_rank - mimicked_rank
            max_possible = n - 1  # Maximum possible difference
            
            # Normalize to [-1, 1]
            normalized_score = raw_score / max_possible
            
            deception_matrix[attacker_idx, mimicked_idx] += normalized_score
            count_matrix[attacker_idx, mimicked_idx] += 1
            total_pairs += 1
    
    # Average the scores
    mask = count_matrix > 0
    deception_matrix[mask] = deception_matrix[mask] / count_matrix[mask]
    np.fill_diagonal(deception_matrix, 0)
    
    # Set symmetric color scale
    vmax = 1.0  # Max possible normalized score
    vmin = -1.0
    
    print(f"📊 Normalized deception score range: [{deception_matrix.min():.2f}, {deception_matrix.max():.2f}]")
    print(f"📊 Total attacker-mimicked pairs analyzed: {total_pairs}")
    
    plt.figure(figsize=(20, 18))
    cmap = 'RdBu_r'  # Red for deception, Blue for detection
    
    ax = sns.heatmap(deception_matrix, 
                     xticklabels=apt_names, 
                     yticklabels=apt_names,
                     cmap=cmap, 
                     center=0, 
                     vmin=vmin, 
                     vmax=vmax, 
                     square=True,
                     annot=False,
                     cbar_kws={
                         'label': 'Deception Score (Normalized)\n🔴 Deception Success | 🔵 Detection Success',
                         'shrink': 0.8,
                         'aspect': 30
                     })
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['-1.0\nDetection', '-0.5', '0\nEqual', '0.5', '1.0\nDeception'])
    
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title('False-Flag Deception Heatmap (30/70 Mix) - Normalized Scores', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT', fontsize=12, labelpad=10)
    plt.ylabel('Actual Attacker', fontsize=12, labelpad=10)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'false_flag_deception_heatmap_normalized_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.show()
    
    # Calculate statistics
    deception_success = np.sum(deception_matrix > 0.1)  # Threshold for meaningful deception
    detection_success = np.sum(deception_matrix < -0.1)  # Threshold for meaningful detection
    neutral = np.sum(np.abs(deception_matrix) <= 0.1)
    total_cells = n * n - n
    
    print("\n📈 Deception Heatmap Statistics:")
    print(f"   🔴 Deception success (>0.1): {deception_success} ({deception_success/total_cells*100:.1f}%)")
    print(f"   🔵 Detection success (<-0.1): {detection_success} ({detection_success/total_cells*100:.1f}%)")
    print(f"   ⚪ Neutral (|≤0.1|): {neutral} ({neutral/total_cells*100:.1f}%)")
    print(f"   📊 Average deception score: {np.mean(deception_matrix[mask]):.3f}")
    
    return deception_matrix

# ===== OPTIMIZED PROCESS PAIR =====
def process_pair(args):
    """Process a single pair - uses pre-computed data"""
    name1, name2, apt_cache = args
    if name1 == name2:
        return None
    
    try:
        start = time.time()
        apt1 = apt_cache[name1]
        apt2 = apt_cache[name2]
        
        # Process both conditions
        full_result = worker_function(apt1, apt2, False)
        shuffled_result = worker_function(apt1, apt2, True)
        
        elapsed = time.time() - start
        return (name1, name2, full_result, shuffled_result, elapsed)
    except Exception as e:
        print(f"Error: {e}")
        return None

def precompute_apt_data(csv_file_path):
    """Pre-compute all APT data once"""
    print("📊 Pre-computing APT data...")
    df = pd.read_csv(csv_file_path)
    apt_names = df['APTname'].unique().tolist()
    
    apt_cache = {}
    for apt_name in tqdm(apt_names, desc="Caching APTs"):
        apt_data = df[df['APTname'] == apt_name]
        apt_cache[apt_name] = {
            'techniques': sorted(apt_data['techniques_str'].dropna().unique().tolist()),
            'software': sorted(apt_data['software_str'].dropna().unique().tolist()),
            'capec': sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1]),
            'cwe': sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1]),
            'cves': sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1]),
            'pulse': sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
        }
    
    return apt_cache, apt_names

def main():
    csv_file_path = "../BaseData/TestFinal_df.csv"
    checkpoint_file = 'false_flag_checkpoint.pkl'
    
    # Pre-compute APT data once
    apt_cache, unique_apt_names_list = precompute_apt_data(csv_file_path)
    
    # Generate all pairs
    pairs = [(name1, name2) for name1 in unique_apt_names_list 
             for name2 in unique_apt_names_list if name1 != name2]
    
    total_pairs = len(pairs)
    print(f"Total pairs to process: {total_pairs}")
    
    # Initialize data structures
    Full_FalseFlag_evidencetable = {name: {} for name in unique_apt_names_list}
    shuffled_divided_FalseFlag_evidence_table = {name: {} for name in unique_apt_names_list}
    processed_pairs = set()
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        print(f"📦 Found checkpoint file. Loading existing results...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            Full_FalseFlag_evidencetable = checkpoint_data['full']
            shuffled_divided_FalseFlag_evidence_table = checkpoint_data['shuffled']
            processed_pairs = set(checkpoint_data['processed_pairs'])
        print(f"✅ Loaded {len(processed_pairs)} already processed pairs")
        
        pairs_to_process = [p for p in pairs if p not in processed_pairs]
        print(f"🔄 Remaining pairs to process: {len(pairs_to_process)}/{total_pairs}")
        
        # If all pairs are processed, skip processing and go directly to evaluation
        if not pairs_to_process:
            print("✨ All pairs already processed! Loading data for evaluation...")
            
            # Run evaluation
            print("\n" + "="*60)
            print("RESULT FOR FULL DATA ")
            eval_precision(unique_apt_names_list, Full_FalseFlag_evidencetable)
            
            print("\n" + "="*60)
            print("RESULT FOR DATA SHUFFLED AND DIVIDED by 2")
            eval_precision(unique_apt_names_list, shuffled_divided_FalseFlag_evidence_table)
            
            plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, unique_apt_names_list)

            print("\n💾 Saving complete ranking data to CSV files...")

            # Save Full_FalseFlag_evidencetable with complete rankings
            full_rows = []
            for attacker, mimicked_dict in Full_FalseFlag_evidencetable.items():
                for mimicked, rank_list in mimicked_dict.items():
                    for position, (apt_name, rank) in enumerate(rank_list, 1):
                        full_rows.append({
                            'attacker': attacker,
                            'mimicked': mimicked,
                            'rank_position': position,
                            'apt_name': apt_name,
                            'rank_value': rank
                        })

            pd.DataFrame(full_rows).to_csv('full_false_flag_complete.csv', index=False)

            # Save shuffled_divided_FalseFlag_evidence_table with complete rankings
            shuffled_rows = []
            for attacker, mimicked_dict in shuffled_divided_FalseFlag_evidence_table.items():
                for mimicked, rank_list in mimicked_dict.items():
                    for position, (apt_name, rank) in enumerate(rank_list, 1):
                        shuffled_rows.append({
                            'attacker': attacker,
                            'mimicked': mimicked,
                            'rank_position': position,
                            'apt_name': apt_name,
                            'rank_value': rank
                        })

            pd.DataFrame(shuffled_rows).to_csv('shuffled_false_flag_complete.csv', index=False)

            print("✅ CSV files saved")
            
            return  # Exit early since no processing needed
    else:
        print(f"🆕 No checkpoint found. Starting fresh...")
        pairs_to_process = pairs
    
        # Prepare arguments with cached data
        args_list = [(name1, name2, apt_cache) for name1, name2 in pairs_to_process]
        
        # Process in parallel
        global_start = time.time()
        
        with Pool(processes=10) as pool:
            with tqdm(total=len(pairs_to_process), desc="Processing pairs") as pbar:
                results = []
                for result in pool.imap_unordered(process_pair, args_list):
                    if result:
                        results.append(result)
                        name1, name2, full, shuffled, elapsed = result
                        
                        Full_FalseFlag_evidencetable[name1][name2] = full
                        shuffled_divided_FalseFlag_evidence_table[name1][name2] = shuffled
                        processed_pairs.add((name1, name2))
                        
                        pbar.update(1)
                        
                        # Calculate and display ETA
                        elapsed_total = time.time() - global_start
                        rate = len(processed_pairs) / elapsed_total
                        remaining = len(pairs_to_process) - pbar.n
                        eta = remaining / rate if rate > 0 else 0
                        
                        pbar.set_postfix({
                            'Rate': f'{rate:.2f}/s',
                            'ETA': f'{eta/60:.1f}min',
                            'Avg': f'{elapsed:.1f}s'
                        })
                        
                        # Save checkpoint every 50 pairs with visible message
                        if len(processed_pairs) % 50 == 0:
                            checkpoint = {
                                'full': Full_FalseFlag_evidencetable,
                                'shuffled': shuffled_divided_FalseFlag_evidence_table,
                                'processed_pairs': list(processed_pairs)
                            }
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump(checkpoint, f)
                            print(f"\n💾 Checkpoint saved at {len(processed_pairs)} pairs")
        
        # Final checkpoint
        checkpoint = {
            'full': Full_FalseFlag_evidencetable,
            'shuffled': shuffled_divided_FalseFlag_evidence_table,
            'processed_pairs': list(processed_pairs)
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"\n💾 Final checkpoint saved at {len(processed_pairs)} pairs")
        
        global_end = time.time()
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETED")
        print("="*50)
        print(f"Total pairs processed: {len(results)}/{total_pairs}")
        print(f"Total time: {global_end - global_start:.2f} seconds ({ (global_end - global_start)/60:.2f} minutes)")
        print(f"Average time per pair: {(global_end - global_start)/len(results):.2f} seconds")
        print(f"Processing rate: {len(results)/(global_end - global_start):.2f} pairs/second")
        
        # Run evaluation
        print("\n" + "="*60)
        print("RESULT FOR FULL DATA ")
        eval_precision(unique_apt_names_list, Full_FalseFlag_evidencetable)
        
        print("\n" + "="*60)
        print("RESULT FOR DATA SHUFFLED AND DIVIDED by 2")
        eval_precision(unique_apt_names_list, shuffled_divided_FalseFlag_evidence_table)
        
        plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, unique_apt_names_list)



if __name__ == "__main__":
    from datetime import datetime
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()