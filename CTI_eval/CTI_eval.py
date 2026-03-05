import json
import sys
import os
import importlib.util
from pathlib import Path

# Completely suppress stdout/stderr during module loading
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

# Suppress all output during module loading
with SuppressOutput():
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

    # Path to hybrid_query.py
    hybrid_dir = Path("..") / "6.hybrid"
    hybrid_file = hybrid_dir / "hybrid_query.py"
    spec = importlib.util.spec_from_file_location("hybrid_module", hybrid_file)
    hybrid_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_module)

    # Define all model directories
    model_dirs = {
        "xgboost": Path("..") / "1.XGboost",
        "gnn": Path("..") / "2.GNN", 
        "siamese": Path("..") / "3.SiameseNetwork",
        "mil": Path("..") / "4.Attention-Based_MIL",
        "rf": Path("..") / "5.RF_FeatureEngineering",
    }

    # Dictionary to store all hybrid_query functions
    hybrid_queries = {}

    # Load each module using importlib
    for name, path in model_dirs.items():
        if path.exists():
            query_file = path / 'query.py'
            spec = importlib.util.spec_from_file_location(f"{name}_query", query_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            hybrid_queries[name] = module.hybrid_query


def combine_and_evaluate_evidence(apt_name, evidence_list):
        """Combine all evidence for an APT and evaluate"""
        # Combine all evidence types across all reports
        combined_evidence = {
            "techniques": [],
            "software": [],
            "capec": [],
            "cwe": [],
            "cve": [],
            "pulse": []
        }
        
        # Collect all evidence
        for element in evidence_list:
            evidence = element['Data']
            for key in combined_evidence.keys():
                combined_evidence[key].extend(evidence.get(key, []))
        
        # Remove duplicates while preserving order
        for key in combined_evidence.keys():
            combined_evidence[key] = list(dict.fromkeys(combined_evidence[key]))
        
        # Evaluate combined evidence
        with SuppressOutput():
            xgboost_result = hybrid_queries['xgboost'](combined_evidence)
            gnn_result = hybrid_queries['gnn'](combined_evidence)
            siamese_results = hybrid_queries['siamese'](combined_evidence)
            mil_results = hybrid_queries['mil'](combined_evidence)
            rf_results = hybrid_queries['rf'](combined_evidence)
            hybrid_result = hybrid_module.remotequery(combined_evidence)
            
            Jaccards_result = jaccard_module.hybrid_query(combined_evidence)
            TVD_result = TVD_module.hybrid_query(combined_evidence)
            featureMatching_result = featureMatching_module.hybrid_query(combined_evidence)
        
        # Filter results
        matching_xgboost = [item for item in xgboost_result if item['apt_name'] == apt_name]
        matching_gnn = [item for item in gnn_result if item['apt_name'] == apt_name]
        matching_siamese = [item for item in siamese_results if item['apt_name'] == apt_name]
        matching_mil = [item for item in mil_results if item['apt_name'] == apt_name]
        matching_rf = [item for item in rf_results if item['apt_name'] == apt_name]
        matching_hybrid = [item for item in hybrid_result if item['apt_name'] == apt_name]
        
        matching_Jaccards = [item for item in Jaccards_result if item['apt_name'] == apt_name]
        matching_TVD = [item for item in TVD_result if item['apt_name'] == apt_name]
        matching_featureMatching = [item for item in featureMatching_result if item['apt_name'] == apt_name]
        
        # Create combined results entry
        combined_entry = {
            "report_link": "combination_of_evidence",
            "Data": combined_evidence,
            "results_ranks": {
                "XGboost": int(matching_xgboost[0]['rank']) if matching_xgboost else None,
                "GNN": int(matching_gnn[0]['rank']) if matching_gnn else None,
                "Siamesenetwork": int(matching_siamese[0]['rank']) if matching_siamese else None,
                "MIL": int(matching_mil[0]['rank']) if matching_mil else None,
                "RF": int(matching_rf[0]['rank']) if matching_rf else None,
                "hybrid": int(matching_hybrid[0]['rank']) if matching_hybrid else None,
                "Jacccards": int(matching_Jaccards[0]['rank']) if matching_Jaccards else None,
                "TVD": int(matching_TVD[0]['rank']) if matching_TVD else None,
                "featureMatching": int(matching_featureMatching[0]['rank']) if matching_featureMatching else None
            }
        }
        
        # Print combined results
        print(f"\n{'='*60}")
        print(f"Combined Evidence Results for {apt_name}")
        print(f"{'='*60}")
        print("\nRanks for Combined Evidence:")
        results = {
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
        
        for model, rank in results.items():
            print(f"  {model}: {rank}")
        
        return combined_entry


# Load JSON from a file
with open('CTI_Eval.json', 'r') as file:
    data = json.load(file)

# Process each element
for apt in data:
    length_apt_data = len(data[apt])
    
    # Call the function to combine and evaluate all evidence for this APT
    combined_result = combine_and_evaluate_evidence(apt, data[apt])
    
    # Append the combined result to the APT's data
    data[apt].append(combined_result)
    

    for i, element in enumerate(data[apt]):
        # Print only the element header
        print(f"\n{'='*60}")
        print(f"Processing {apt} - Element {i} / {length_apt_data}")
        print(f"{'='*60}")
        
        evidence = element['Data']

        # Suppress output during query execution
        with SuppressOutput():
            # Execute each query
            xgboost_result = hybrid_queries['xgboost'](evidence)
            gnn_result = hybrid_queries['gnn'](evidence)
            siamese_results = hybrid_queries['siamese'](evidence)
            mil_results = hybrid_queries['mil'](evidence)
            rf_results = hybrid_queries['rf'](evidence)
            hyrbid_result = hybrid_module.remotequery(evidence)
            
            Jaccards_result = jaccard_module.hybrid_query(evidence)
            TVD_result = TVD_module.hybrid_query(evidence)
            featureMatching_result = featureMatching_module.hybrid_query(evidence)

        # Filter results
        matching_xgboost_result = [item for item in xgboost_result if item['apt_name'] == apt]
        matching_gnn_result = [item for item in gnn_result if item['apt_name'] == apt]
        matching_siamese_results = [item for item in siamese_results if item['apt_name'] == apt]
        matching_mil_results = [item for item in mil_results if item['apt_name'] == apt]
        matching_rf_results = [item for item in rf_results if item['apt_name'] == apt]
        matching_hybrid_results = [item for item in hyrbid_result if item['apt_name'] == apt]
        
        matching_Jaccards_result = [item for item in Jaccards_result if item['apt_name'] == apt]
        matching_TVD_result = [item for item in TVD_result if item['apt_name'] == apt]
        matching_featureMatching_result = [item for item in featureMatching_result if item['apt_name'] == apt]

        # Print only the ranks in a clean format
        print("\nRanks:")
        results = {
            'XGBoost': matching_xgboost_result[0]['rank'] if matching_xgboost_result else 'N/A',
            'GNN': matching_gnn_result[0]['rank'] if matching_gnn_result else 'N/A',
            'Siamese': matching_siamese_results[0]['rank'] if matching_siamese_results else 'N/A',
            'MIL': matching_mil_results[0]['rank'] if matching_mil_results else 'N/A',
            'RF': matching_rf_results[0]['rank'] if matching_rf_results else 'N/A',
            'hybrid': matching_hybrid_results[0]['rank'] if matching_hybrid_results else 'N/A',

            'Jaccards': matching_Jaccards_result[0]['rank'] if matching_Jaccards_result else 'N/A',
            'TVD': matching_TVD_result[0]['rank'] if matching_TVD_result else 'N/A',
            'Feature Matching': matching_featureMatching_result[0]['rank'] if matching_featureMatching_result else 'N/A'
        }
        
        for model, rank in results.items():
            print(f"  {model}: {rank}")

        # Update the results_ranks
        if matching_xgboost_result:
            data[apt][i]['results_ranks']['XGboost'] = int(matching_xgboost_result[0]['rank'])
        if matching_gnn_result:
            data[apt][i]['results_ranks']['GNN'] = int(matching_gnn_result[0]['rank'])
        if matching_siamese_results:
            data[apt][i]['results_ranks']['Siamesenetwork'] = int(matching_siamese_results[0]['rank'])
        if matching_mil_results:
            data[apt][i]['results_ranks']['MIL'] = int(matching_mil_results[0]['rank'])
        if matching_rf_results:
            data[apt][i]['results_ranks']['RF'] = int(matching_rf_results[0]['rank'])
        if matching_hybrid_results:
            data[apt][i]['results_ranks']['hybrid'] = int(matching_hybrid_results[0]['rank'])

        if matching_Jaccards_result:
            data[apt][i]['results_ranks']['Jacccards'] = int(matching_Jaccards_result[0]['rank'])
        if matching_TVD_result:
            data[apt][i]['results_ranks']['TVD'] = int(matching_TVD_result[0]['rank'])
        if matching_featureMatching_result:
            data[apt][i]['results_ranks']['featureMatching'] = int(matching_featureMatching_result[0]['rank'])

    #     break
    # break

# Save the modified data
with open('CTI_Eval_modified.json', 'w') as file:
    json.dump(data, file, indent=2)