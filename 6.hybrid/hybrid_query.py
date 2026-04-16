import pandas as pd
import importlib.util
from pathlib import Path
import os
import sys
import numpy as np

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


def hybrid_rank_aggregate(lists, weight_xgboost, weight_gnn, 
                          weight_siamese, weight_mil, weight_rf, 
                          weight_featureMatching, weight_jaccard, weight_tvd):
    
    """
    Reciprocal Rank Fusion (RRF) with per-model weights.
    RRF score = sum_i ( weight_i / (k + rank_i) )
    Lower rank (1 = best) gives higher contribution.
    """
    k = 60  # standard constant, can be tuned
    n_lists = len(lists)
    
    # Create weights list in the same order as the lists
    weights = [weight_xgboost, weight_gnn, weight_siamese, weight_mil, 
               weight_rf, weight_jaccard, weight_tvd, weight_featureMatching]
    
    # Normalize weights (so they sum to 1, but not strictly necessary for RRF)
    weights = np.array(weights) / np.sum(weights)
    
    # For each APT, collect its rank from each model (if present)
    rank_dict = {}
    for list_idx, ranking in enumerate(lists):
        for item in ranking:
            name = item['apt_name']
            rank = item['rank']   # 1-indexed, lower is better
            if name not in rank_dict:
                rank_dict[name] = [None] * n_lists
            rank_dict[name][list_idx] = rank
    
    # Calculate RRF score for each APT
    results = []
    for name, ranks in rank_dict.items():
        rrf_score = 0.0
        for i, rank in enumerate(ranks):
            if rank is not None:
                rrf_score += weights[i] / (k + rank)
        results.append({
            'name': name,
            'rrf_score': rrf_score
        })
    
    # Sort by RRF score descending (higher = better)
    # If scores are equal (extremely rare), break ties by alphabetical order
    results.sort(key=lambda x: (-x['rrf_score'], x['name']))
    
    return [(item['name'], i+1) for i, item in enumerate(results)]


def automated_query_for_eval(evidence_first_batch):
    
    # print("IN")
    try:
        # Save original outputs
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

        # Restore outputs
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # print(result)
        return result

    except Exception as e:
        print(f"Error: {e}")


def remotequery(evidence_first_batch):

    XGboost_results = xgboost_query.hybrid_query(evidence_first_batch)

    GNN_results = gnn_query.hybrid_query(evidence_first_batch)

    Siamese_results = siamese_query.hybrid_query(evidence_first_batch)

    MIL_results = mil_query.hybrid_query(evidence_first_batch)

    RF_results = rf_query.hybrid_query(evidence_first_batch)

    Jaccards_result = jaccard_module.hybrid_query(evidence_first_batch)

    TVD_result = TVD_module.hybrid_query(evidence_first_batch)

    featureMatching_result = featureMatching_module.hybrid_query(evidence_first_batch)

    # print("RESULTS ::::::::")
    # i = 0
    # print(str("XGboost")+" :: "+str(XGboost_results[i]))
    # print(str("GNN")+" :: "+str(GNN_results[i]))
    # print(str("Siamese")+" :: "+str(Siamese_results[i]))
    # print(str("MIL")+" :: "+str(MIL_results[i]))
    # print(str("RF")+" :: "+str(RF_results[i]))

    # print(str("Jaccard")+" :: "+str(Jaccards_result[i]))
    # print(str("TVD")+" :: "+str(TVD_result[i]))
    # print(str("featureMatching_result")+" :: "+str(featureMatching_result[i]))


    all_lists = [XGboost_results, GNN_results, Siamese_results, MIL_results, RF_results, Jaccards_result, TVD_result, featureMatching_result]

    #no weights
    # result = hybrid_rank_aggregate(
    #         all_lists,
    #         weight_xgboost=1,
    #         weight_gnn=1,
    #         weight_siamese=1,
    #         weight_mil=1,
    #         weight_rf=1,
    #         weight_featureMatching=1,
    #         weight_jaccard=1,
    #         weight_tvd=1
    #     )

    #weight based only on self evaluation
    # result = hybrid_rank_aggregate(
    #         all_lists,
    #         weight_jaccard=5,
    #         weight_xgboost=5,
    #         weight_featureMatching=5,
    #         weight_siamese=1,
    #         weight_tvd=2,
    #         weight_rf=1,
    #         weight_gnn=1,
    #         weight_mil=1
    #     )

    #weight determined on precision evaluation through dataset and CTI eval testing
    result = hybrid_rank_aggregate(
            all_lists,
            weight_featureMatching=5,
            weight_jaccard=4,
            weight_xgboost=4,
            weight_mil=4,
            weight_gnn=1,
            weight_tvd=2,
            weight_siamese=1,
            weight_rf=2
        )


    # Transform to match first list format (only rank and apt_name)
    final_result = [
        {'rank': rank, 'apt_name': name}
        for name, rank in result
    ]

    return final_result

def main():
    # Get CSV file path from user
    csv_file_path = "../BaseData/TestFinal_df.csv"
    
    



    import json
    # Read evidence from JSON file
    with open('../query_input.json', 'r') as f:
        evidence_first_batch = json.load(f)

    XGboost_results = xgboost_query.hybrid_query(evidence_first_batch)

    GNN_results = gnn_query.hybrid_query(evidence_first_batch)

    Siamese_results = siamese_query.hybrid_query(evidence_first_batch)

    MIL_results = mil_query.hybrid_query(evidence_first_batch)

    RF_results = rf_query.hybrid_query(evidence_first_batch)

    Jaccards_result = jaccard_module.hybrid_query(evidence_first_batch)

    TVD_result = TVD_module.hybrid_query(evidence_first_batch)

    featureMatching_result = featureMatching_module.hybrid_query(evidence_first_batch)

    # print("RESULTS ::::::::")
    # i = 0
    # print(str("XGboost")+" :: "+str(XGboost_results[i]))
    # print(str("GNN")+" :: "+str(GNN_results[i]))
    # print(str("Siamese")+" :: "+str(Siamese_results[i]))
    # print(str("MIL")+" :: "+str(MIL_results[i]))
    # print(str("RF")+" :: "+str(RF_results[i]))

    # print(str("Jaccard")+" :: "+str(Jaccards_result[i]))
    # print(str("TVD")+" :: "+str(TVD_result[i]))
    # print(str("featureMatching_result")+" :: "+str(featureMatching_result[i]))


    all_lists = [XGboost_results, GNN_results, Siamese_results, MIL_results, RF_results, Jaccards_result, TVD_result, featureMatching_result]

    #no weights
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
    
    # #weight determined on precision evaluation through dataset
    # result = hybrid_rank_aggregate(
    #     all_lists,
    #     weight_xgboost=9,
    #     weight_gnn=1,
    #     weight_siamese=1,
    #     weight_mil=1,
    #     weight_rf=8,
    #     weight_featureMatching=10,
    #     weight_jaccard=10,
    #     weight_tvd=10
    # )
    
    for name, rank in result:
        print(f"  {name}: {rank}")
        
    # except FileNotFoundError:
    #     print(f"Error: File '{csv_file_path}' not found.")
    # except Exception as e:
    #     print(f"Error: {e}")



if __name__ == "__main__":
    main()