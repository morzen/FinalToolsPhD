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

# Set random seed for reproducibility
random.seed(42)

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
        # Check if directory exists
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
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
        
        print(f"✓ Imported {module_name} from {os.getcwd()}/query.py")
        
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

print(f"\n✓ Successfully imported {len(modules)} modules")
for name in modules.keys():
    print(f"  - {name}")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

##########################################################################################

##########################################################################################
def process_apt_data(csv_file_path, apt_name):
    """
    Process CSV file and extract all 7 features for a specific APT
    """
    # Read CSV data from file
    df = pd.read_csv(csv_file_path)
    
    # Filter data for the specified APT
    apt_data = df[df['APTname'] == apt_name]
    
    # Extract unique techniques (excluding empty values and -1)
    techniques = sorted(apt_data['techniques_str'].dropna().unique().tolist())
    
    # Extract unique software (excluding empty values and -1)
    software = sorted(apt_data['software_str'].dropna().unique().tolist())
    
    # Extract unique CAPEC_ID (excluding empty values and -1)
    capec = sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique CWE_ID (excluding empty values and -1)
    cwe = sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique CVEs (excluding empty values and -1)
    cves = sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique Pulse_ID (excluding empty values and -1)
    pulse = sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
    
    return {
        'techniques': techniques,
        'software': software,
        'capec': capec,
        'cwe': cwe,
        'cve': cves,
        'pulse': pulse
    }


def hybrid_rank_aggregate(lists):
    """
    Advanced rank aggregation with explicit tie-breaking
    """
    n_lists = len(lists)
    
    # Create rank matrix
    rank_matrix = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']
            if name not in rank_matrix:
                rank_matrix[name] = [None] * n_lists
            rank_matrix[name][list_idx] = rank
    
    # Calculate statistics
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
    
    # Sort with multiple tie-breakers
    results.sort(key=lambda x: (
        x['avg'],           # Primary: average rank
        x['median'],        # Secondary: median
        x['min'],          # Tertiary: best rank
        x['range'],        # Quaternary: consistency
        x['name']          # Final: alphabetical
    ))
    
    return [(item['name'], i+1) for i, item in enumerate(results)]



def automated_query_for_eval(evidence_first_batch):
    try:
        # # Save original outputs
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect to null
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        XGboost_results = xgboost_query.hybrid_query(evidence_first_batch)

        GNN_results = gnn_query.hybrid_query(evidence_first_batch)

        Siamese_results = siamese_query.hybrid_query(evidence_first_batch)

        MIL_results = mil_query.hybrid_query(evidence_first_batch)

        RF_results = rf_query.hybrid_query(evidence_first_batch)

        Jaccards_result = jaccard_module.hybrid_query(evidence_first_batch)

        TVD_result = TVD_module.hybrid_query(evidence_first_batch)

        featureMatching_result = featureMatching_module.hybrid_query(evidence_first_batch)

        all_lists = [XGboost_results, GNN_results, Siamese_results, MIL_results, RF_results, Jaccards_result, TVD_result, featureMatching_result]

        result = hybrid_rank_aggregate(all_lists)

        # # Restore outputs
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # print(result)
        return result

    except Exception as e:
        print(f"Error: {e}")



def process_apt_data(csv_file_path, apt_name):
    """
    Process CSV file and extract all 7 features for a specific APT
    """
    # Read CSV data from file
    df = pd.read_csv(csv_file_path)
    
    # Filter data for the specified APT
    apt_data = df[df['APTname'] == apt_name]
    
    # Extract unique techniques (excluding empty values and -1)
    techniques = sorted(apt_data['techniques_str'].dropna().unique().tolist())
    
    # Extract unique software (excluding empty values and -1)
    software = sorted(apt_data['software_str'].dropna().unique().tolist())
    
    # Extract unique CAPEC_ID (excluding empty values and -1)
    capec = sorted([x for x in apt_data['CAPEC_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique CWE_ID (excluding empty values and -1)
    cwe = sorted([x for x in apt_data['CWE_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique CVEs (excluding empty values and -1)
    cves = sorted([x for x in apt_data['CVE_ID'].dropna().unique().tolist() if x != -1])
    
    # Extract unique Pulse_ID (excluding empty values and -1)
    pulse = sorted([x for x in apt_data['Pulse_ID'].dropna().unique().tolist() if x != -1])
    
    return {
        'techniques': techniques,
        'software': software,
        'capec': capec,
        'cwe': cwe,
        'cve': cves,
        'pulse': pulse
    }


def create_apt_ranking_heatmap(apt_dict):
    """
    Create a single heatmap from APT ranking dictionary.
    Color scale: Red = Rank 181 (lowest), Green = Rank 1 (highest)
    """
    
    # Convert dictionary to DataFrame
    df_list = []
    
    for query_apt, rankings in apt_dict.items():
        row = {'Query_APT': query_apt}
        for ranked_apt, rank in rankings:
            # Invert rank: 1 becomes max value (green), 181 becomes min value (red)
            inverted_score = 181 - rank + 1  # 1->181, 181->1
            row[ranked_apt] = inverted_score
        df_list.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(df_list).set_index('Query_APT')
    df = df.fillna(0)
    
    # Create heatmap
    plt.figure(figsize=(24, 20))
    
    # Red (min) -> Yellow -> Green (max)
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    
    # Create heatmap
    sns.heatmap(df,
                cmap=cmap,
                vmin=1,        # Red = Rank 181
                vmax=181,      # Green = Rank 1
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={
                    'shrink': 0.8,
                    'label': 'APT Ranking (Green=Best/Rank 1, Red=Worst/Rank 181)',
                    'ticks': [1, 90, 181]
                })
    
    # Customize
    plt.title("APT Ranking Similarity Matrix\n(Green = Rank 1, Yellow = Rank ~90, Red = Rank 181)", 
              fontsize=18, 
              fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    
    # Update colorbar labels
    cbar = plt.gcf().axes[-1]
    cbar.set_yticklabels(['Rank 181 (Red)', 'Rank ~90 (Yellow)', 'Rank 1 (Green)'])
    
    plt.tight_layout()
    plt.savefig('apt_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Heatmap saved as 'apt_ranking_heatmap.png'")
    plt.show()

    # Save the figure
    plt.savefig('fulldataheatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Heatmap saved as 'fulldataheatmap.png'")


import random

def shuffler_and_divider(evidence_dict):
    """
    Shuffle the evidence items for each category and keep only half.
    
    Parameters:
    -----------
    evidence_dict : dict
        Dictionary with keys 'techniques', 'software', 'capec', 'cwe', 'cves', 'pulse'
        each containing a list of items
    
    Returns:
    --------
    dict : New dictionary with shuffled and halved evidence
    """
    
    shuffled_and_divided = {}
    
    for key, items in evidence_dict.items():
        if key in ['techniques', 'software', 'capec', 'cwe', 'cves', 'pulse']:
            # Make a copy and shuffle
            shuffled_items = items.copy()
            random.shuffle(shuffled_items)
            
            # Keep only half
            half_length = len(shuffled_items) // 2
            shuffled_and_divided[key] = shuffled_items[:half_length]
        else:
            # Keep other keys as-is (if any)
            shuffled_and_divided[key] = items
    
    return shuffled_and_divided


def eval_precision(unique_apt_names_list, givenEvidence):

    # Initialize counters using dictionary for cleaner code
    metrics = {
        'real': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0},
        'mimicked': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0},
        'both': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0},  # Added top1
        'neither': {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0}  # Added top1
    }

    # Calculate total number of valid comparisons (excluding self-comparisons)
    total_comparisons = 0
    for name1, outer_dict in givenEvidence.items():
        for name2 in outer_dict.keys():
            if name1 != name2:
                total_comparisons += 1

    # Pre-define rank thresholds
    thresholds = [1, 5, 10, 20]

    for name1, outer_dict in givenEvidence.items():
        # Get self-rank once
        self_rank = int(outer_dict.get(name1, 0))
        
        for name2, rank in outer_dict.items():
            if name1 == name2:
                continue
                
            rank_val = int(rank)
            total_comparisons  # Already counted above
            
            # Check each threshold
            for threshold in thresholds:
                # Check if within threshold
                self_in_threshold = 1 <= self_rank <= threshold
                other_in_threshold = 1 <= rank_val <= threshold
                
                # Update individual counters
                if self_in_threshold:
                    metrics['real'][f'top{threshold}'] += 1
                if other_in_threshold:
                    metrics['mimicked'][f'top{threshold}'] += 1
                
                # Update both/neither counters
                if self_in_threshold and other_in_threshold:
                    metrics['both'][f'top{threshold}'] += 1
                if not self_in_threshold and not other_in_threshold:
                    metrics['neither'][f'top{threshold}'] += 1

    # PRINT IN EXACT FORMAT REQUESTED
    print("\n" + "="*60)
    print(" FALSE-FLAG ATTACK ANALYSIS - RANKING RESULTS")
    print("="*60)

    for threshold in thresholds:
        attacker = metrics['real'][f'top{threshold}']
        mimicked = metrics['mimicked'][f'top{threshold}']
        both = metrics['both'][f'top{threshold}']
        neither = metrics['neither'][f'top{threshold}']
        
        # Calculate percentages
        attacker_pct = (attacker / total_comparisons * 100) if total_comparisons > 0 else 0
        mimicked_pct = (mimicked / total_comparisons * 100) if total_comparisons > 0 else 0
        both_pct = (both / total_comparisons * 100) if total_comparisons > 0 else 0
        neither_pct = (neither / total_comparisons * 100) if total_comparisons > 0 else 0
        
        # Deception = Mimicked in top-X, Detection = Attacker in top-X
        deception_pct = mimicked_pct
        detection_pct = attacker_pct
        
        print(f"\n Top-{threshold}:")
        print(f"  Attacker: {attacker}/{total_comparisons} ({attacker_pct:.1f}%)")
        print(f"  Mimicked: {mimicked}/{total_comparisons} ({mimicked_pct:.1f}%)")
        print(f"  Both: {both}/{total_comparisons} ({both_pct:.1f}%)")
        print(f"  Neither: {neither}/{total_comparisons} ({neither_pct:.1f}%)")
        print(f"   Deception: {deception_pct:.1f}% |  Detection: {detection_pct:.1f}%")

    print(f"\nTotal comparisons analyzed: {total_comparisons:,}")

    return(metrics)








def shuffle_remove_and_combine(apt1, apt2, apt1_ratio=0.3, apt2_ratio=0.7):
    combined = {}
    
    # Process each category
    for category in apt1.keys():
        combined[category] = []
        
        # Sample from APT1 (high → low - take from beginning)
        apt1_items = apt1[category]
        apt1_count = int(len(apt1_items) * apt1_ratio)
        # Take from start for high → low
        apt1_sample = apt1_items[:apt1_count]
        
        # Sample from APT2 (low → high - take from end)
        apt2_items = apt2[category]
        apt2_count = int(len(apt2_items) * apt2_ratio)
        # Take from end for low → high
        apt2_sample = apt2_items[-apt2_count:] if apt2_count > 0 else []
        
        # Combine and deduplicate while preserving order
        combined_category = []
        seen = set()
        
        for item in apt1_sample + apt2_sample:
            if item not in seen:
                seen.add(item)
                combined_category.append(item)
        
        combined[category] = combined_category
    
    return combined

def plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, apt_names):
    """
    Create deception heatmap from FalseFlag evidence table.
    
    Args:
        Full_FalseFlag_evidencetable: dict {attacker_name: {mimicked_name: [(apt, rank), ...], ...}}
        apt_names: list of APT display names in the order you want them displayed
    """
    print("\n🔥 CREATING FALSE-FLAG DECEPTION HEATMAP...")
    
    n = len(apt_names)
    
    # Create mapping from APT name to index
    name_to_idx = {name: i for i, name in enumerate(apt_names)}
    
    # Initialize deception matrix
    deception_matrix = np.zeros((n, n), dtype=float)
    count_matrix = np.zeros((n, n), dtype=int)
    
    # Process each attacker and their mimicked results
    total_pairs = 0
    for attacker_name, mimicked_dict in Full_FalseFlag_evidencetable.items():
        if attacker_name not in name_to_idx:
            print(f"⚠ Attacker {attacker_name} not found in apt_names")
            continue
            
        attacker_idx = name_to_idx[attacker_name]
        
        # Get self-rank (attacker vs itself - should be rank 1)
        self_rank = None
        if attacker_name in mimicked_dict:
            ranking_list = mimicked_dict[attacker_name]
            for apt_name, rank in ranking_list:
                if apt_name == attacker_name:
                    self_rank = rank
                    break
        
        if self_rank is None:
            self_rank = n  # Default to worst rank if not found
            
        # Process each mimicked APT
        for mimicked_name, ranking_list in mimicked_dict.items():
            if mimicked_name not in name_to_idx:
                continue
                
            if attacker_name == mimicked_name:
                continue  # Skip self-comparisons (diagonal)
                
            mimicked_idx = name_to_idx[mimicked_name]
            
            # Find the rank of the mimicked APT in this ranking
            mimicked_rank = None
            for apt_name, rank in ranking_list:
                if apt_name == mimicked_name:
                    mimicked_rank = rank
                    break
                    
            if mimicked_rank is None:
                mimicked_rank = n  # Default to worst rank if not found
            
            # Deception score: Attacker Rank - Mimicked Rank
            # >0 = Deception success (mimicked ranked higher than attacker)
            # <0 = Detection success (attacker ranked higher than mimicked)
            deception_score = self_rank - mimicked_rank
            
            deception_matrix[attacker_idx, mimicked_idx] += deception_score
            count_matrix[attacker_idx, mimicked_idx] += 1
            total_pairs += 1
    
    # Average the scores
    mask = count_matrix > 0
    deception_matrix[mask] = deception_matrix[mask] / count_matrix[mask]
    
    # Set diagonal to 0 (self-comparisons)
    np.fill_diagonal(deception_matrix, 0)
    
    # Find non-zero values for color scaling
    non_zero = deception_matrix[np.abs(deception_matrix) > 0.01]
    if non_zero.size == 0:
        print("⚠ No deception signal detected")
        return deception_matrix
    
    # Symmetric color scaling
    vmax = np.max(np.abs(non_zero))
    vmin = -vmax
    
    print(f"📊 Deception score range: [{vmin:.2f}, {vmax:.2f}]")
    print(f"📊 Total attacker-mimicked pairs analyzed: {total_pairs}")
    
    # Create heatmap
    plt.figure(figsize=(20, 18))
    
    # Use the same colormap as your example
    cmap = 'RdBu_r'  # Red-Blue reversed (Red=positive/Deception, Blue=negative/Detection)
    
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
                         'label': 'Deception Score (Attacker Rank - Mimicked Rank)\n🔴 Deception Success | 🔵 Detection Success',
                         'shrink': 0.8,
                         'aspect': 30
                     })
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f'{vmin:.1f}\nDetection', '0\nEqual', f'{vmax:.1f}\nDeception'])
    
    # Rotate labels for better readability
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    # Title and labels
    plt.title(f'False-Flag Deception Heatmap (30/70 Mix)\nRows=Actual Attacker (30%) | Columns=Mimicked APT (70%)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT (Target of Impersonation)', fontsize=12, labelpad=10)
    plt.ylabel('Actual Attacker (Source of False-Flag)', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    
    # Save figure with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'false_flag_deception_heatmap_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    
    plt.show()
    
    # Print statistics
    deception_success = np.sum(deception_matrix > 0.01)
    detection_success = np.sum(deception_matrix < -0.01)
    neutral = np.sum(np.abs(deception_matrix) <= 0.01)
    total_cells = n * n - n  # Exclude diagonal
    
    print("\n📈 Deception Heatmap Statistics:")
    print(f"   🔴 Deception success cells: {deception_success} ({deception_success/total_cells*100:.1f}%)")
    print(f"   🔵 Detection success cells: {detection_success} ({detection_success/total_cells*100:.1f}%)")
    print(f"   ⚪ Neutral cells: {neutral} ({neutral/total_cells*100:.1f}%)")
    print(f"   📊 Average deception score: {np.mean(deception_matrix[mask]):.3f}")
    
    return deception_matrix

def main():
    csv_file_path = "../BaseData/TestFinal_df.csv"
    
    df = pd.read_csv(csv_file_path)
    unique_apt_names = df['APTname'].unique()

    # Convert to list if you need a list
    unique_apt_names_list = unique_apt_names.tolist()

    Full_FalseFlag_evidencetable = {}
    total_len = len(unique_apt_names_list) * len(unique_apt_names_list) - len(unique_apt_names_list)
    start_count = 0
    for name1 in unique_apt_names_list:
        evidence_first_batch_n1 = process_apt_data(csv_file_path, name1)
        TempFalseFlag = {}
        for name2 in unique_apt_names_list:
            if(name1 == name2):
                pass
            else:
                start_count = start_count + 1
                print(str(start_count)+"/"+str(total_len))
                # print("\n"+"1st_batch:  "+name1+" __ "+name2)
                try:
                    # Process the data
                    # Get APT name from user
                    evidence_first_batch_n2 = process_apt_data(csv_file_path, name2)

                    custom_falsflag_mimick_test = shuffle_remove_and_combine(evidence_first_batch_n1, evidence_first_batch_n2)

                    result = automated_query_for_eval(custom_falsflag_mimick_test)
                    # print(result)
                    TempFalseFlag[str(name2)] = result

                    # print(TempFalseFlag)
                    # print(Fullevidencetable)
                except FileNotFoundError:
                    print(f"Error: File '{csv_file_path}' not found.")
                except Exception as e:
                    print(f"Error: {e}")

        Full_FalseFlag_evidencetable[str(name1)] = TempFalseFlag
        # print(Full_FalseFlag_evidencetable)
    


    

    print("\n BATCH 2 \n")
    shuffled_divided_FalseFlag_evidence_table = {}
    total_len = len(unique_apt_names_list) * len(unique_apt_names_list) - len(unique_apt_names_list)
    start_count = 0
    for name1 in unique_apt_names_list:
        evidence_first_batch_n1 = process_apt_data(csv_file_path, name1)
        evidence_first_batch_n1 = shuffler_and_divider(evidence_first_batch_n1)
        TempFalseFlag = {}
        for name2 in unique_apt_names_list:
            if(name1 == name2):
                pass
            else:
                start_count = start_count + 1
                print(str(start_count)+"/"+str(total_len))
                # print("\n"+"2st_batch:  "+name1+" __ "+name2)
                try:
                    # Process the data
                    # Get APT name from user
                    evidence_first_batch_n2 = process_apt_data(csv_file_path, name2)

                    custom_falsflag_mimick_test = shuffle_remove_and_combine(evidence_first_batch_n1, evidence_first_batch_n2)
                    shuffled_and_divided_evidence = shuffler_and_divider(custom_falsflag_mimick_test)
                    result = automated_query_for_eval(shuffled_and_divided_evidence)
                    # print(result)
                    TempFalseFlag[str(name2)] = result

                    # print(Fullevidencetable)
                except FileNotFoundError:
                    print(f"Error: File '{csv_file_path}' not found.")
                except Exception as e:
                    print(f"Error: {e}")

        shuffled_divided_FalseFlag_evidence_table[str(name1)] = TempFalseFlag

    print("RESULT FOR FULL DATA ")

    eval_precision(unique_apt_names_list, Full_FalseFlag_evidencetable)

    print("RESULT FOR DATA SHUFFLED AND DIVIDED by 2")

    eval_precision(unique_apt_names_list, shuffled_divided_FalseFlag_evidence_table)

    plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable)

    # Create the heatmap for full data check
    # create_apt_ranking_heatmap(Full_FalseFlag_evidencetable)




if __name__ == "__main__":
    main()




##################################






