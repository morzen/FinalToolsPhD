# import pandas as pd
# import importlib.util
# from pathlib import Path
# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
# import random
# from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
# import threading
# from functools import lru_cache
# import time
# import traceback

# # Suppress output during imports
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

# # Path to Jaccards_correlation.py
# jaccard_dir = Path("..") / "Jaccards_Correlation"
# jaccard_file = jaccard_dir / "Jaccards_correlation.py"

# spec = importlib.util.spec_from_file_location("jaccards_module", jaccard_file)
# jaccard_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(jaccard_module)


# # Path to TVD_featureIng.py
# TVD_dir = Path("..") / "TVD_Total_Variation_Distance"
# TVD_file = TVD_dir / "TVD_featureIng.py"

# spec = importlib.util.spec_from_file_location("TVD_module", TVD_file)
# TVD_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(TVD_module)


# # Path to featureMatching.py
# featureMatching_dir = Path("..") / "featureMatching"
# featureMatching_file = featureMatching_dir / "featureMatching.py"

# spec = importlib.util.spec_from_file_location("featureMatching_module", featureMatching_file)
# featureMatching_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(featureMatching_module)


# # Store original state
# original_dir = os.getcwd()
# original_sys_path = sys.path.copy()

# # Define all directories
# directories = {
#     "xgboost": Path("..") / "1.XGboost",
#     "gnn": Path("..") / "2.GNN", 
#     "siamese": Path("..") / "3.SiameseNetwork",
#     "mil": Path("..") / "4.Attention-Based_MIL",
#     "rf": Path("..") / "5.RF_FeatureEngineering",
# }

# # Dictionary to store all imported modules
# modules = {}

# # Thread lock for safe printing
# print_lock = threading.Lock()

# def safe_print(*args, **kwargs):
#     """Thread-safe print function"""
#     with print_lock:
#         print(*args, **kwargs)

# try:
#     for module_name, dir_path in directories.items():
#         # Check if directory exists
#         if not dir_path.exists():
#             safe_print(f"⚠️  Directory not found: {dir_path}")
#             continue
            
#         # Change to the module's directory
#         os.chdir(dir_path)
        
#         # Add current directory to path
#         sys.path.insert(0, os.getcwd())
        
#         # Import from the current directory
#         spec = importlib.util.spec_from_file_location(f"{module_name}_query", "query.py")
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)
        
#         # Store the module
#         modules[module_name] = module
        
#         safe_print(f"✓ Imported {module_name} from {os.getcwd()}/query.py")
        
#         # IMPORTANT: Reset path before next import
#         sys.path.remove(os.getcwd())
        
# finally:
#     # Always restore original state
#     os.chdir(original_dir)
#     sys.path = original_sys_path

# # Now you have all modules in the dictionary
# xgboost_query = modules.get("xgboost")
# gnn_query = modules.get("gnn")
# siamese_query = modules.get("siamese")
# mil_query = modules.get("mil")
# rf_query = modules.get("rf")

# safe_print(f"\n✓ Successfully imported {len(modules)} modules")
# for name in modules.keys():
#     safe_print(f"  - {name}")

# sys.stdout = sys.__stdout__
# sys.stderr = sys.__stderr__

# ##########################################################################################

# ##########################################################################################
# # Cache for processed APT data
# processed_data_cache = {}

# def process_apt_data(csv_file_path, apt_name):
#     """
#     Process CSV file and extract all 7 features for a specific APT
#     """
#     # Check cache first
#     cache_key = f"{csv_file_path}_{apt_name}"
#     if cache_key in processed_data_cache:
#         return processed_data_cache[cache_key]
    
#     # Read CSV data from file
#     df = pd.read_csv(csv_file_path)
    
#     # Filter data for the specified APT
#     apt_data = df[df['APTname'] == apt_name]
    
#     # Extract unique techniques (excluding empty values and -1)
#     techniques = sorted(apt_data['techniques_str'].dropna().unique().tolist())
    
#     # Extract unique software (excluding empty values and -1)
#     software = sorted(apt_data['software_str'].dropna().unique().tolist())
    
#     # Extract unique CAPEC_ID (excluding empty values and -1)
#     capec = sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1])
    
#     # Extract unique CWE_ID (excluding empty values and -1)
#     cwe = sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1])
    
#     # Extract unique CVEs (excluding empty values and -1)
#     cves = sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1])
    
#     # Extract unique Pulse_ID (excluding empty values and -1)
#     pulse = sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
    
#     result = {
#         'techniques': techniques,
#         'software': software,
#         'capec': capec,
#         'cwe': cwe,
#         'cve': cves,
#         'pulse': pulse
#     }
    
#     # Cache the result
#     processed_data_cache[cache_key] = result
#     return result


# def hybrid_rank_aggregate(lists):
#     """
#     Advanced rank aggregation with explicit tie-breaking
#     """
#     n_lists = len(lists)
    
#     # Create rank matrix
#     rank_matrix = {}
#     for list_idx, ranking in enumerate(lists):
#         for item in ranking:
#             name = item['apt_name']
#             rank = item['rank']
#             if name not in rank_matrix:
#                 rank_matrix[name] = [None] * n_lists
#             rank_matrix[name][list_idx] = rank
    
#     # Calculate statistics
#     results = []
#     for name, ranks in rank_matrix.items():
#         ranks = [r for r in ranks if r is not None]
#         if ranks:  # Only if we have ranks
#             avg_rank = np.mean(ranks)
#             median_rank = np.median(ranks)
#             min_rank = min(ranks)
#             max_rank = max(ranks)
#             rank_range = max_rank - min_rank
            
#             results.append({
#                 'name': name,
#                 'avg': avg_rank,
#                 'median': median_rank,
#                 'min': min_rank,
#                 'range': rank_range
#             })
    
#     # Sort with multiple tie-breakers
#     results.sort(key=lambda x: (
#         x['avg'],           # Primary: average rank
#         x['median'],        # Secondary: median
#         x['min'],          # Tertiary: best rank
#         x['range'],        # Quaternary: consistency
#         x['name']          # Final: alphabetical
#     ))
    
#     return [(item['name'], i+1) for i, item in enumerate(results)]


# def run_single_query(evidence, module, module_name, timeout_seconds=20):
#     """Run a single query with a specific module - with timeout"""
#     try:
#         # Save original outputs
#         original_stdout = sys.stdout
#         original_stderr = sys.stderr
        
#         # Redirect to null
#         sys.stdout = open(os.devnull, 'w')
#         sys.stderr = open(os.devnull, 'w')
        
#         # Run the query with timeout
#         result = None
#         start_time = time.time()
        
#         # Execute in a separate thread with timeout
#         def target():
#             nonlocal result
#             try:
#                 result = module.hybrid_query(evidence)
#             except Exception as e:
#                 safe_print(f"  ⚠️ {module_name} error: {str(e)[:50]}")
#                 result = None
        
#         thread = threading.Thread(target=target)
#         thread.daemon = True
#         thread.start()
#         thread.join(timeout=timeout_seconds)
        
#         if thread.is_alive():
#             safe_print(f"  ⚠️ {module_name} timed out after {timeout_seconds}s")
#             result = None
        
#         # Restore outputs
#         sys.stdout.close()
#         sys.stderr.close()
#         sys.stdout = original_stdout
#         sys.stderr = original_stderr
        
#         return result
#     except Exception as e:
#         safe_print(f"  ⚠️ {module_name} failed: {str(e)[:50]}")
#         # Ensure outputs are restored
#         try:
#             sys.stdout.close()
#             sys.stderr.close()
#             sys.stdout = original_stdout
#             sys.stderr = original_stderr
#         except:
#             pass
#         return None


# def automated_query_for_eval(evidence_first_batch):
#     """
#     Parallelized version of automated_query_for_eval with better error handling
#     """
#     # Define all modules to run
#     modules_to_run = [
#         ("xgboost", xgboost_query),
#         ("gnn", gnn_query),
#         ("siamese", siamese_query),
#         ("mil", mil_query),
#         ("rf", rf_query),
#         ("jaccard", jaccard_module),
#         ("tvd", TVD_module),
#         ("featureMatching", featureMatching_module)
#     ]
    
#     # Filter out None modules
#     valid_modules = [(name, module) for name, module in modules_to_run if module is not None]
    
#     if not valid_modules:
#         return []
    
#     results = []
    
#     # Run queries sequentially but with timeout protection
#     # This is safer than parallel for this use case
#     for name, module in valid_modules:
#         try:
#             # Save original outputs
#             original_stdout = sys.stdout
#             original_stderr = sys.stderr
            
#             # Redirect to null
#             sys.stdout = open(os.devnull, 'w')
#             sys.stderr = open(os.devnull, 'w')
            
#             # Run with timeout using signal (simpler approach)
#             import signal
            
#             def timeout_handler(signum, frame):
#                 raise TimeoutError(f"{name} query timed out")
            
#             # Set timeout
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(20)  # 20 second timeout
            
#             try:
#                 result = module.hybrid_query(evidence_first_batch)
#                 signal.alarm(0)  # Disable alarm
#                 if result is not None:
#                     results.append(result)
#             except TimeoutError:
#                 safe_print(f"  ⚠️ {name} timed out")
#                 signal.alarm(0)
#             except Exception as e:
#                 safe_print(f"  ⚠️ {name} error: {str(e)[:50]}")
#                 signal.alarm(0)
            
#             # Restore outputs
#             sys.stdout.close()
#             sys.stderr.close()
#             sys.stdout = original_stdout
#             sys.stderr = original_stderr
            
#         except Exception as e:
#             safe_print(f"  ⚠️ Error in {name}: {str(e)[:50]}")
#             # Ensure outputs are restored
#             try:
#                 sys.stdout.close()
#                 sys.stderr.close()
#                 sys.stdout = original_stdout
#                 sys.stderr = original_stderr
#             except:
#                 pass
    
#     # Aggregate results if we have any
#     if len(results) >= 3:  # Need at least 3 results for meaningful aggregation
#         return hybrid_rank_aggregate(results)
#     else:
#         safe_print(f"  ⚠️ Only {len(results)} modules succeeded, skipping")
#         return []


# def shuffler_and_divider(evidence_dict):
#     """
#     Shuffle the evidence items for each category and keep only half.
#     """
#     shuffled_and_divided = {}
    
#     for key, items in evidence_dict.items():
#         if key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
#             # Make a copy and shuffle
#             shuffled_items = items.copy()
#             random.shuffle(shuffled_items)
            
#             # Keep only half
#             half_length = len(shuffled_items) // 2
#             shuffled_and_divided[key] = shuffled_items[:half_length]
#         else:
#             # Keep other keys as-is (if any)
#             shuffled_and_divided[key] = items
    
#     return shuffled_and_divided


# def eval_precision(unique_apt_names_list, givenEvidence):
#     """
#     Evaluate precision at different ranks
#     """
#     total = len(unique_apt_names_list)
    
#     fulldata_top1 = 0
#     fulldata_top5 = 0
#     fulldata_top10 = 0
#     fulldata_top20 = 0
    
#     for name in unique_apt_names_list:
#         # Get the list of (apt_name, rank) tuples for this query
#         ranking_list = givenEvidence.get(name, [])
        
#         if not ranking_list:
#             continue
        
#         # Find the tuple where the apt_name matches the query name
#         rank = None
#         for apt_name, apt_rank in ranking_list:
#             if apt_name == name:
#                 rank = apt_rank
#                 break
        
#         if rank is not None:
#             if rank == 1:
#                 fulldata_top1 += 1
#                 fulldata_top5 += 1
#                 fulldata_top10 += 1
#                 fulldata_top20 += 1
#             elif rank <= 5:
#                 fulldata_top5 += 1
#                 fulldata_top10 += 1
#                 fulldata_top20 += 1
#             elif rank <= 10:
#                 fulldata_top10 += 1
#                 fulldata_top20 += 1
#             elif rank <= 20:
#                 fulldata_top20 += 1

#     if total > 0:
#         print(f"Top-1: {fulldata_top1}/{total} ({fulldata_top1/total*100:.2f}%)")
#         print(f"Top-5: {fulldata_top5}/{total} ({fulldata_top5/total*100:.2f}%)")
#         print(f"Top-10: {fulldata_top10}/{total} ({fulldata_top10/total*100:.2f}%)")
#         print(f"Top-20: {fulldata_top20}/{total} ({fulldata_top20/total*100:.2f}%)")


# def process_single_apt(args):
#     """Process a single APT - for parallel execution"""
#     name, csv_file_path, is_shuffled, index, total = args
    
#     batch_type = "2nd_batch" if is_shuffled else "1st_batch"
#     safe_print(f"\n[{index+1}/{total}] {batch_type}_{name}")
    
#     try:
#         # Process the data
#         evidence_first_batch = process_apt_data(csv_file_path, name)
        
#         if is_shuffled:
#             evidence_to_use = shuffler_and_divider(evidence_first_batch)
#         else:
#             evidence_to_use = evidence_first_batch
        
#         result = automated_query_for_eval(evidence_to_use)
        
#         if result:
#             safe_print(f"  ✓ {name} completed")
#             return name, result
#         else:
#             safe_print(f"  ⚠️ {name} returned no results")
#             return name, []
#     except Exception as e:
#         safe_print(f"  ✗ Error processing {name}: {str(e)[:100]}")
#         traceback.print_exc(file=sys.stdout)
#         return name, []


# def create_apt_ranking_heatmap(apt_dict):
#     """
#     Create a single heatmap from APT ranking dictionary.
#     """
#     # Get all unique APT names in a consistent order
#     all_apt_names = sorted(apt_dict.keys())
    
#     # Create DataFrame with consistent row and column order
#     df = pd.DataFrame(index=all_apt_names, columns=all_apt_names)
#     df = df.fillna(0)
    
#     # Fill the DataFrame
#     for query_apt in all_apt_names:
#         rankings = apt_dict.get(query_apt, [])
#         ranking_dict = dict(rankings)
        
#         for ranked_apt in all_apt_names:
#             if ranked_apt in ranking_dict:
#                 # Invert rank: 1 becomes max value (green), 181 becomes min value (red)
#                 rank = ranking_dict[ranked_apt]
#                 inverted_score = 181 - rank + 1
#                 df.loc[query_apt, ranked_apt] = inverted_score
    
#     # Create heatmap
#     plt.figure(figsize=(24, 20))
    
#     # Red (min) -> Yellow -> Green (max)
#     colors = ["red", "yellow", "green"]
#     cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    
#     # Create heatmap
#     sns.heatmap(df,
#                 cmap=cmap,
#                 vmin=1,
#                 vmax=181,
#                 square=True,
#                 xticklabels=True,
#                 yticklabels=True,
#                 cbar_kws={
#                     'shrink': 0.8,
#                     'label': 'APT Ranking (Green=Best/Rank 1, Red=Worst/Rank 181)',
#                     'ticks': [1, 90, 181]
#                 })
    
#     # Customize
#     plt.title("APT Ranking Similarity Matrix\n(Green = Rank 1, Yellow = Rank ~90, Red = Rank 181)", 
#               fontsize=18, 
#               fontweight='bold')
#     plt.xticks(rotation=90, fontsize=8)
#     plt.yticks(rotation=0, fontsize=10)
    
#     # Update colorbar labels
#     cbar = plt.gcf().axes[-1]
#     cbar.set_yticklabels(['Rank 181 (Red)', 'Rank ~90 (Yellow)', 'Rank 1 (Green)'])
    
#     plt.tight_layout()
#     plt.savefig('apt_ranking_heatmap.png', dpi=300, bbox_inches='tight')
#     print("\n✓ Heatmap saved as 'apt_ranking_heatmap.png'")
#     plt.show()


# def main():
#     csv_file_path = "../BaseData/TestFinal_df.csv"
    
#     # Read APT names
#     df = pd.read_csv(csv_file_path)
#     unique_apt_names = df['APTname'].unique()
#     unique_apt_names_list = unique_apt_names.tolist()
#     total_apts = len(unique_apt_names_list)
    
#     print(f"Processing {total_apts} APTs...")
    
#     # Process first batch (full data)
#     print("\n" + "="*50)
#     print("Processing First Batch (Full Data)")
#     print("="*50)
    
#     Fullevidencetable = {}
    
#     # Process sequentially to avoid threading issues
#     # This is slower but more reliable
#     for i, name in enumerate(unique_apt_names_list):
#         safe_print(f"\n[{i+1}/{total_apts}] Processing {name} (Full Data)")
#         try:
#             evidence = process_apt_data(csv_file_path, name)
#             result = automated_query_for_eval(evidence)
#             if result:
#                 Fullevidencetable[str(name)] = result
#                 safe_print(f"  ✓ {name} completed")
#             else:
#                 safe_print(f"  ⚠️ {name} returned no results")
#         except Exception as e:
#             safe_print(f"  ✗ Error: {str(e)[:100]}")
    
#     # Process second batch (shuffled data)
#     print("\n" + "="*50)
#     print("Processing Second Batch (Shuffled & Divided Data)")
#     print("="*50)
    
#     shuffled_divided_evidence_table = {}
    
#     for i, name in enumerate(unique_apt_names_list):
#         safe_print(f"\n[{i+1}/{total_apts}] Processing {name} (Shuffled)")
#         try:
#             evidence = process_apt_data(csv_file_path, name)
#             shuffled_evidence = shuffler_and_divider(evidence)
#             result = automated_query_for_eval(shuffled_evidence)
#             if result:
#                 shuffled_divided_evidence_table[str(name)] = result
#                 safe_print(f"  ✓ {name} completed")
#             else:
#                 safe_print(f"  ⚠️ {name} returned no results")
#         except Exception as e:
#             safe_print(f"  ✗ Error: {str(e)[:100]}")
    
#     # Evaluate results
#     print("\n" + "="*50)
#     print("Results for Full Data")
#     print("="*50)
#     eval_precision(unique_apt_names_list, Fullevidencetable)
    
#     print("\n" + "="*50)
#     print("Results for Shuffled & Divided Data")
#     print("="*50)
#     eval_precision(unique_apt_names_list, shuffled_divided_evidence_table)
    
#     # Create the heatmap for full data check
#     if Fullevidencetable:
#         create_apt_ranking_heatmap(Fullevidencetable)


# if __name__ == "__main__":
#     main()


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


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
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import traceback
import contextlib

# ------------------------------------------------------------------------------
# Module loading (unchanged from original)
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
        # Check if directory exists
        if not dir_path.exists():
            safe_print(f"⚠️  Directory not found: {dir_path}")
            continue

        # Change to the module's directory
        os.chdir(dir_path)

        # Add current directory to path
        sys.path.insert(0, os.getcwd())

        # Import from the current directory
        spec = importlib.util.spec_from_file_location(f"{module_name}_query", "query.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Store the module
        modules[module_name] = module

        safe_print(f"✓ Imported {module_name} from {os.getcwd()}/query.py")

        # IMPORTANT: Reset path before next import
        sys.path.remove(os.getcwd())

finally:
    # Always restore original state
    os.chdir(original_dir)
    sys.path = original_sys_path

# Now you have all modules in the dictionary
xgboost_query = modules.get("xgboost")
gnn_query = modules.get("gnn")
siamese_query = modules.get("siamese")
mil_query = modules.get("mil")
rf_query = modules.get("rf")

safe_print(f"\n✓ Successfully imported {len(modules)} modules")
for name in modules.keys():
    safe_print(f"  - {name}")

# Restore stdout/stderr after imports
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# ------------------------------------------------------------------------------
# Optimized functions
# ------------------------------------------------------------------------------

def preload_apt_data(csv_file_path):
    """
    Read the CSV once and precompute the seven features for each APT.
    Returns a dictionary: apt_name -> {techniques, software, capec, cwe, cve, pulse}
    """
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
    """Shuffle and keep half of the items for each category."""
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
    """
    Advanced rank aggregation with explicit tie-breaking (vectorized using pandas).
    """
    n_lists = len(lists)
    # Build a dictionary of ranks per APT name
    rank_dict = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']
            rank_dict.setdefault(name, {})[list_idx] = rank

    # Convert to DataFrame
    df = pd.DataFrame(rank_dict).T.fillna(np.nan)

    # Compute statistics
    avg = df.mean(axis=1)
    median = df.median(axis=1)
    min_rank = df.min(axis=1)
    rank_range = df.max(axis=1) - min_rank

    # Create a DataFrame of results and sort
    results_df = pd.DataFrame({
        'avg': avg,
        'median': median,
        'min': min_rank,
        'range': rank_range
    })
    results_df.sort_values(by=['avg', 'median', 'min', 'range'], inplace=True)

    # Return list of (name, rank) tuples
    return [(name, i+1) for i, name in enumerate(results_df.index)]


def run_single_query(evidence, module):
    """
    Run a single query with a specific module, redirecting stdout/stderr.
    No timeout – we let the module run until completion.
    """
    try:
        # Redirect outputs to null
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result = module.hybrid_query(evidence)
        return result
    except Exception as e:
        safe_print(f"  ⚠️ Module error: {str(e)[:50]}")
        return None


def automated_query_for_eval(evidence):
    """
    Run all available modules sequentially (no inner parallelism) and aggregate results.
    """
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
    """
    Process one APT: optionally shuffle evidence, run query, return (name, result).
    This function is called in parallel processes.
    """
    try:
        if shuffle_flag:
            evidence = shuffler_and_divider(evidence)
        result = automated_query_for_eval(evidence)
        return name, result
    except Exception as e:
        safe_print(f"✗ Error processing {name}: {str(e)[:100]}")
        traceback.print_exc(file=sys.stdout)
        return name, []


def eval_precision(unique_apt_names_list, givenEvidence):
    """
    Evaluate precision at different ranks.
    """
    total = len(unique_apt_names_list)

    fulldata_top1 = 0
    fulldata_top5 = 0
    fulldata_top10 = 0
    fulldata_top20 = 0

    for name in unique_apt_names_list:
        ranking_list = givenEvidence.get(name, [])
        if not ranking_list:
            continue

        # Find rank for this APT
        rank = None
        for apt_name, apt_rank in ranking_list:
            if apt_name == name:
                rank = apt_rank
                break

        if rank is not None:
            if rank == 1:
                fulldata_top1 += 1
                fulldata_top5 += 1
                fulldata_top10 += 1
                fulldata_top20 += 1
            elif rank <= 5:
                fulldata_top5 += 1
                fulldata_top10 += 1
                fulldata_top20 += 1
            elif rank <= 10:
                fulldata_top10 += 1
                fulldata_top20 += 1
            elif rank <= 20:
                fulldata_top20 += 1

    if total > 0:
        print(f"Top-1: {fulldata_top1}/{total} ({fulldata_top1/total*100:.2f}%)")
        print(f"Top-5: {fulldata_top5}/{total} ({fulldata_top5/total*100:.2f}%)")
        print(f"Top-10: {fulldata_top10}/{total} ({fulldata_top10/total*100:.2f}%)")
        print(f"Top-20: {fulldata_top20}/{total} ({fulldata_top20/total*100:.2f}%)")


def create_apt_ranking_heatmap(apt_dict):
    """
    Create a single heatmap from APT ranking dictionary.
    """
    all_apt_names = sorted(apt_dict.keys())

    df = pd.DataFrame(index=all_apt_names, columns=all_apt_names)
    df = df.fillna(0)

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

    sns.heatmap(df,
                cmap=cmap,
                vmin=1,
                vmax=181,
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={
                    'shrink': 0.8,
                    'label': 'APT Ranking (Green=Best/Rank 1, Red=Worst/Rank 181)',
                    'ticks': [1, 90, 181]
                })

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
    """
    Process a list of APTs in parallel using a process pool.
    Returns a dictionary mapping APT name to result list.
    """
    results = {}
    total = len(apt_names)

    # Prepare arguments
    args = [(name, apt_data[name], shuffle_flag) for name in apt_names]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_apt, name, evidence, shuffle_flag): name
                   for name, evidence in zip(apt_names, [apt_data[name] for name in apt_names])}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
            safe_print(f"✓ Completed {name} (shuffled={shuffle_flag}) [{len(results)}/{total}]")
    return results


def main():
    csv_file_path = "../BaseData/TestFinal_df.csv"

    # Preload all APT data once
    safe_print("Loading APT data...")
    apt_data = preload_apt_data(csv_file_path)
    apt_names = list(apt_data.keys())
    total_apts = len(apt_names)
    safe_print(f"Loaded data for {total_apts} APTs.")

    # Determine maximum number of workers (limit to 9)
    max_workers = min(9, total_apts)
    safe_print(f"Using {max_workers} parallel workers.")

    # Process first batch (full data)
    safe_print("\n" + "="*50)
    safe_print("Processing First Batch (Full Data)")
    safe_print("="*50)
    full_results = parallel_process_apts(apt_names, apt_data, shuffle_flag=False, max_workers=max_workers)

    # Process second batch (shuffled data)
    safe_print("\n" + "="*50)
    safe_print("Processing Second Batch (Shuffled & Divided Data)")
    safe_print("="*50)
    shuffled_results = parallel_process_apts(apt_names, apt_data, shuffle_flag=True, max_workers=max_workers)

    # Evaluate results
    print("\n" + "="*50)
    print("Results for Full Data")
    print("="*50)
    eval_precision(apt_names, full_results)

    print("\n" + "="*50)
    print("Results for Shuffled & Divided Data")
    print("="*50)
    eval_precision(apt_names, shuffled_results)

    # Create heatmap for full data if results exist
    if full_results:
        create_apt_ranking_heatmap(full_results)


if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    import multiprocessing as mp
    mp.freeze_support()
    main()