import pandas as pd
import importlib.util
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


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

    fulldata_top1 = 0
    fulldata_top5 = 0
    fulldata_top10 = 0
    fulldata_top20 = 0
    total = 181
    for name in unique_apt_names_list:
        if int(givenEvidence[name][1]) == 1:
            fulldata_top1 = fulldata_top1 + 1
        elif 1 <= int(givenEvidence[name][1]) <= 5 :
            fulldata_top5 = fulldata_top5 + 1
        elif 1 <= int(givenEvidence[name][1]) <= 10 :
            fulldata_top10 = fulldata_top10 + 1
        elif 1 <= int(givenEvidence[name][1]) <= 20 :
            fulldata_top20 = fulldata_top20 + 1
        else:
            pass

    print(f"fulldata_top1: {fulldata_top1} success, {total - fulldata_top1} failure, overall: {fulldata_top1}/{total} ({fulldata_top1/total*100:.2f}%)")
    print(f"fulldata_top5: {fulldata_top5} success, {total - fulldata_top5} failure, overall: {fulldata_top5}/{total} ({fulldata_top5/total*100:.2f}%)")
    print(f"fulldata_top10: {fulldata_top10} success, {total - fulldata_top10} failure, overall: {fulldata_top10}/{total} ({fulldata_top10/total*100:.2f}%)")
    print(f"fulldata_top20: {fulldata_top20} success, {total - fulldata_top20} failure, overall: {fulldata_top20}/{total} ({fulldata_top20/total*100:.2f}%)")


def main():
    csv_file_path = "../BaseData/TestFinal_df.csv"
    
    df = pd.read_csv(csv_file_path)
    unique_apt_names = df['APTname'].unique()

    # Convert to list if you need a list
    unique_apt_names_list = unique_apt_names.tolist()

    Fullevidencetable = {}
    for name in unique_apt_names_list:
        print("\n"+"1st_batch_"+name)
        try:
            # Process the data
            # Get APT name from user
            apt_name = name
            evidence_first_batch = process_apt_data(csv_file_path, apt_name)

            result = automated_query_for_eval(evidence_first_batch)
            # print(result)
            Fullevidencetable[str(name)] = result

            # print(Fullevidencetable)
        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    


    


    shuffled_divided_evidence_table = {}
    for name in unique_apt_names_list:
        print("\n"+"2nd_batch_"+name)
        try:
            # Process the data
            # Get APT name from user
            apt_name = name
            evidence_first_batch = process_apt_data(csv_file_path, apt_name)
            shuffled_and_divided_evidence = shuffler_and_divider(evidence_first_batch)
            result = automated_query_for_eval(shuffled_and_divided_evidence)
            # print(result)
            shuffled_divided_evidence_table[str(name)] = result

            # print(shuffled_divided_evidence_table)
        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
        except Exception as e:
            print(f"Error: {e}")



    eval_precision(unique_apt_names_list, Fullevidencetable)

    print("RESULT FOR DATA SHUFFLED AND DIVIDED by 2")

    eval_precision(unique_apt_names_list, shuffled_divided_evidence_table)


    # Create the heatmap for full data check
    create_apt_ranking_heatmap(Fullevidencetable)




if __name__ == "__main__":
    main()




##################################






