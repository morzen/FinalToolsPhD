import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
from itertools import islice

# Load saved model and data from single file
def load_complete_model():
    with open('./complete_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    return (model_data['model'], model_data['apt_profiles'], 
            model_data['apt_name_dict'], model_data['df_mapping'])

# Evidence processing functions
def sanitize_evidence(evidence, df_mapping):
    sanitized = {}
    missing = {}
    
    column_mapping = {
        'techniques_str': ('techniques_Origin', 'techniques_Encoded'),
        'software_str': ('software_Origin', 'software_Encoded'),
        'CAPEC_ID': ('CAPEC_ID_Origin', 'CAPEC_ID_Encoded'),
        'CWE_ID': ('CWE_ID_Origin', 'CWE_ID_Encoded'),
        'CVE_ID': ('CVE_ID_Origin', 'CVE_ID_Encoded'),
        'Pulse_ID': ('Pulse_ID_Origin', 'Pulse_ID_Encoded')
    }
    
    for evidence_key, (origin_col, _) in column_mapping.items():
        if evidence_key in evidence:
            valid_values = set(df_mapping[origin_col].dropna())
            valid_values.discard('')
            
            valid = []
            invalid = []
            
            for item in evidence[evidence_key]:
                if item == '':
                    continue
                if str(item) in valid_values:
                    valid.append(item)
                else:
                    invalid.append(item)
            
            if valid:
                sanitized[evidence_key] = valid
            if invalid:
                missing[evidence_key] = invalid
    
    if missing:
        print("\nWARNING: These features were not found in our database:")
        for feature_type, items in missing.items():
            print(f"- {feature_type}: {items[:5]}")
            if len(items) > 5:
                print(f"  (and {len(items)-5} more)")
        print()
    
    return sanitized

def translate_to_encoded(evidence, df_mapping):
    encoded_evidence = {}
    
    column_mapping = {
        'techniques_str': ('techniques_Encoded', 'techniques'),
        'software_str': ('software_Encoded', 'software'), 
        'CAPEC_ID': ('CAPEC_ID_Encoded', 'capec'),
        'CWE_ID': ('CWE_ID_Encoded', 'cwe'),
        'CVE_ID': ('CVE_ID_Encoded', 'cve'),
        'Pulse_ID': ('Pulse_ID_Encoded', 'pulse')
    }
    
    for evidence_key, (encoded_col, output_key) in column_mapping.items():
        if evidence_key in evidence:
            origin_col = encoded_col.replace('_Encoded', '_Origin')
            mapping = dict(zip(df_mapping[origin_col], df_mapping[encoded_col]))
            
            encoded_values = []
            for val in evidence[evidence_key]:
                if val in mapping:
                    encoded_values.append(mapping[val])
            
            if encoded_values:
                encoded_evidence[output_key] = encoded_values
    
    return encoded_evidence

# Bayesian network probability calculation
def calculate_apt_probabilities(encoded_evidence, model, inference, apt_profiles):
    results = []
    min_prob = 1e-10
    
    # Get all APT names
    all_apt_names = list(apt_profiles.keys())
    
    for apt_name in all_apt_names:
        total_log_prob = 0
        features_processed = 0
        
        for feature_type, features in encoded_evidence.items():
            for feature in features:
                try:
                    # Query for P(APT | Feature_Type, Feature)
                    query = inference.query(
                        variables=['APT'],
                        evidence={
                            'Feature_Type': feature_type,
                            'Feature': feature
                        }
                    )
                    
                    # Get the probability for this specific APT
                    apt_index = all_apt_names.index(apt_name)
                    prob = query.values[apt_index]
                    
                    # Avoid log(0) issues
                    if prob <= 0:
                        prob = min_prob
                    
                    total_log_prob += np.log(prob)
                    features_processed += 1
                    
                except Exception as e:
                    # Skip features that cause errors
                    continue
        
        # Calculate final probability using geometric mean
        if features_processed > 0:
            final_prob = np.exp(total_log_prob / features_processed)
        else:
            final_prob = min_prob
        
        results.append({
            'APT': apt_name,
            'Probability': final_prob,
            'Encoded_ID': apt_profiles[apt_name]['encoded_id'],
            'Features_Processed': features_processed
        })
    
    return pd.DataFrame(results)

# Create similarity matrix for heatmap
def create_similarity_matrix(apt_profiles, encoded_evidence, apt_name_dict):
    apt_names = list(apt_profiles.keys())
    num_apts = len(apt_names)
    similarity_matrix = np.zeros((num_apts, num_apts))
    
    for i in range(num_apts):
        for j in range(num_apts):
            if i <= j:
                vec_i = []
                vec_j = []
                
                for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
                    if feature_type in encoded_evidence:
                        features_i = set(apt_profiles[apt_names[i]][feature_type])
                        features_j = set(apt_profiles[apt_names[j]][feature_type])
                        all_features = set(encoded_evidence[feature_type])
                        
                        vec_i.extend([1 if f in features_i else 0 for f in all_features])
                        vec_j.extend([1 if f in features_j else 0 for f in all_features])
                
                if len(vec_i) > 0 and len(vec_j) > 0:
                    # Calculate Jaccard similarity
                    intersection = sum([1 for a, b in zip(vec_i, vec_j) if a == 1 and b == 1])
                    union = sum([1 for a, b in zip(vec_i, vec_j) if a == 1 or b == 1])
                    
                    if union > 0:
                        similarity = intersection / union
                    else:
                        similarity = 0
                    
                    similarity_matrix[i,j] = similarity
                    similarity_matrix[j,i] = similarity
    
    # Create DataFrame with APT names
    apt_display_names = [apt_name_dict[apt_profiles[name]['encoded_id']] for name in apt_names]
    similarity_df = pd.DataFrame(similarity_matrix, index=apt_display_names, columns=apt_display_names)
    
    return similarity_df

# Enhanced heatmap visualization
def plot_enhanced_similarity_matrix(similarity_df):
    plt.figure(figsize=(24, 20))
    
    # Custom green to red colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("green_red", colors)
    
    ax = sns.heatmap(
        similarity_df, 
        cmap=cmap, 
        vmin=0, 
        vmax=1,
        square=True,
        annot=False,
        cbar_kws={'shrink': 0.8, 'label': 'Jaccard Similarity'}
    )
    
    plt.title("APT Group Similarity Matrix\n(Green = High Similarity, Red = Low Similarity)", 
              fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=90, fontsize=7, ha='right')
    plt.yticks(fontsize=7)
    
    plt.tight_layout()
    plt.savefig('apt_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Simple ranking based only on Bayesian probability
def calculate_simple_ranking(results_df, apt_profiles, apt_name_dict):
    # Sort by probability descending
    final_df = results_df.sort_values('Probability', ascending=False)
    final_df['Rank'] = range(1, len(final_df) + 1)
    
    # Add APT names
    final_df['APT_Name'] = final_df['APT'].apply(lambda x: apt_name_dict[apt_profiles[x]['encoded_id']])
    
    return final_df

# ===== OPTIMIZED EVALUATION USING EXISTING SIMILARITY MATRIX =====
def evaluate_with_similarity_matrix(similarity_df, top_x_values=[1, 5, 10, 20]):
    """
    Optimized evaluation using the pre-computed similarity matrix
    """
    print(f"\n{'='*60}")
    print("OPTIMIZED EVALUATION USING EXISTING SIMILARITY MATRIX")
    print(f"{'='*60}")
    print("Using 100% of profile data (full similarity matrix)")
    
    apt_names = similarity_df.index.tolist()
    success_rates = {topX: 0 for topX in top_x_values}
    total_tests = len(apt_names)
    
    # For each APT, check its ranking in the similarity matrix
    for i, target_apt_name in enumerate(apt_names):
        # Get similarity scores for this APT against all others
        apt_similarities = similarity_df[target_apt_name].sort_values(ascending=False)
        
        # Check for each topX value
        for topX in top_x_values:
            top_apts = apt_similarities.head(topX).index.tolist()
            if target_apt_name in top_apts:
                success_rates[topX] += 1
        
        # Show progress
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{total_tests} APTs...")
    
    # Print results
    print(f"\nResults based on {total_tests} tests:")
    for topX in top_x_values:
        success_rate = (success_rates[topX] / total_tests) * 100
        print(f"Top {topX}: {success_rate:.1f}% success rate")
    
    return success_rates

# ===== EVALUATION WITH DIVIDED PROFILES =====
def evaluate_divided_profiles(apt_profiles, apt_name_dict, encoded_evidence, 
                             divider=2, top_x_values=[1, 5, 10, 20]):
    """
    Evaluate with divided profiles using Jaccard similarity
    """
    print(f"\n{'='*60}")
    print("EVALUATION WITH DIVIDED PROFILES")
    print(f"{'='*60}")
    print(f"Using {100/divider:.1f}% of profile data")
    
    apt_ids = list(apt_profiles.keys())
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    success_rates = {topX: 0 for topX in top_x_values}
    total_tests = len(apt_names)
    
    # For each APT, create divided profile and check similarity
    for i, target_apt_name in enumerate(apt_names):
        # Get the target APT's data
        target_apt_id = next(apt for apt in apt_ids if apt_name_dict[apt_profiles[apt]['encoded_id']] == target_apt_name)
        target_data = apt_profiles[target_apt_id]
        
        # Create divided evidence
        divided_features = {}
        for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            if feature_type in target_data and target_data[feature_type]:
                feature_list = target_data[feature_type].copy()
                random.shuffle(feature_list)
                division_size = max(1, len(feature_list) // divider)
                divided_features[feature_type] = set(feature_list[:division_size])
        
        # Calculate Jaccard similarity with all other APTs
        similarities = {}
        for other_apt_id in apt_ids:
            other_apt_name = apt_name_dict[apt_profiles[other_apt_id]['encoded_id']]
            other_data = apt_profiles[other_apt_id]
            
            total_similarity = 0
            feature_count = 0
            
            for feature_type, divided_set in divided_features.items():
                if feature_type in other_data:
                    other_set = set(other_data[feature_type])
                    
                    if divided_set:  # Only calculate if we have divided features
                        intersection = len(divided_set & other_set)
                        union = len(divided_set | other_set)
                        
                        if union > 0:
                            similarity = intersection / union
                            total_similarity += similarity
                            feature_count += 1
            
            # Average similarity across feature types
            if feature_count > 0:
                similarities[other_apt_name] = total_similarity / feature_count
            else:
                similarities[other_apt_name] = 0
        
        # Sort by similarity
        sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
        
        # Check for each topX value
        for topX in top_x_values:
            top_apts = list(sorted_similarities.keys())[:topX]
            if target_apt_name in top_apts:
                success_rates[topX] += 1
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_tests} APTs...")
    
    # Print results
    print(f"\nResults based on {total_tests} tests:")
    for topX in top_x_values:
        success_rate = (success_rates[topX] / total_tests) * 100
        print(f"Top {topX}: {success_rate:.1f}% success rate")
    
    return success_rates

def main():
    # Load the complete model from single file
    model, apt_profiles, apt_name_dict, df_mapping = load_complete_model()
    
    # Create inference engine
    inference = VariableElimination(model)
    
    # Get all APT IDs
    apt_ids = list(apt_profiles.keys())
    
    # Example evidence for Lazarus Group
    evidence = {
    'techniques_str': ['T1001.002', 'T1003.002', 'T1003.004', 'T1003.006', 'T1005', 'T1016.001', 'T1018', 'T1021.001', 
                      'T1021.002', 'T1021.006', 'T1021.007', 'T1027.001', 'T1027.002', 'T1027.003', 'T1027.006', 
                      'T1036.004', 'T1036.005', 'T1037', 'T1037.004', 'T1047', 'T1048.002', 'T1053.005', 'T1057', 
                      'T1059.001', 'T1059.003', 'T1059.005', 'T1059.006', 'T1059.009', 'T1068', 'T1069.002', 
                      'T1070.004', 'T1070.006', 'T1070.008', 'T1071.001', 'T1074.002', 'T1078', 'T1078.002', 
                      'T1078.003', 'T1078.004', 'T1082', 'T1083', 'T1087.002', 'T1087.004', 'T1090.001', 'T1090.002', 
                      'T1090.003', 'T1090.004', 'T1098.001', 'T1098.002', 'T1098.003', 'T1098.005', 'T1102.002', 
                      'T1105', 'T1110.001', 'T1110.003', 'T1114.002', 'T1133', 'T1136.003', 'T1140', 'T1190', 
                      'T1195.002', 'T1199', 'T1203', 'T1204.001', 'T1204.002', 'T1213', 'T1213.003', 'T1218.005', 
                      'T1218.011', 'T1482', 'T1484.002', 'T1505.003', 'T1528', 'T1539', 'T1546.003', 'T1546.008', 
                      'T1547.001', 'T1548.002', 'T1550.001', 'T1550.003', 'T1550.004', 'T1552.004', 'T1553.002', 
                      'T1553.005', 'T1555.003', 'T1556.007', 'T1558.003', 'T1560.001', 'T1562.001', 'T1562.002', 
                      'T1562.004', 'T1562.008', 'T1566.001', 'T1566.002', 'T1566.003', 'T1568', 'T1573', 'T1583.001', 
                      'T1583.006', 'T1584.001', 'T1585.001', 'T1586.002', 'T1586.003', 'T1587.001', 'T1587.003', 
                      'T1588.002', 'T1589.001', 'T1595.002', 'T1606.001', 'T1606.002', 'T1621', 'T1649', 'T1651', 
                      'T1665'],
    'software_str': ['S0037', 'S0559', 'S0633', 'S0002', 'S0046', 'S0050', 'S0154', 'S0357', 'S0677', 'S0048', 
                    'S0512', 'S0514', 'S0515', 'S0634', 'S0661', 'S0597', 'S0039', 'S0521', 'S0552', 'S0684', 
                    'S0029', 'S0053', 'S0513', 'S0565', 'S0588', 'S0139', 'S0511', 'S0518', 'S0560', 'S0562', 
                    'S0589', 'S1084', 'S0516', 'S0049', 'S0057', 'S0150', 'S0195', 'S0051', 'S0052', 'S0054', 
                    'S0635', 'S0636', 'S0682', 'S0096', 'S0183', 'S0175', 'S0637', 'S0100'],
    'CAPEC_ID': ['CAPEC-204', 'CAPEC-37', 'CAPEC-545', 'CAPEC-647', 'CAPEC-292', 'CAPEC-561', 'CAPEC-572', 'CAPEC-655', 
                'CAPEC-35', 'CAPEC-504', 'CAPEC-543', 'CAPEC-616', 'CAPEC-564', 'CAPEC-573', 'CAPEC-560', 'CAPEC-70', 
                'CAPEC-312', 'CAPEC-313', 'CAPEC-580', 'CAPEC-127', 'CAPEC-497', 'CAPEC-465', 'CAPEC-481', 'CAPEC-49', 
                'CAPEC-565', 'CAPEC-555', 'CAPEC-186', 'CAPEC-443', 'CAPEC-523', 'CAPEC-669', 'CAPEC-673', 'CAPEC-678', 
                'CAPEC-691', 'CAPEC-150', 'CAPEC-650', 'CAPEC-21', 'CAPEC-31', 'CAPEC-558', 'CAPEC-593', 'CAPEC-645', 
                'CAPEC-60', 'CAPEC-474', 'CAPEC-485', 'CAPEC-639', 'CAPEC-206', 'CAPEC-473', 'CAPEC-68', 'CAPEC-509', 
                'CAPEC-578', 'CAPEC-268', 'CAPEC-571', 'CAPEC-163', 'CAPEC-542', 'CAPEC-464'],
    'CWE_ID': ['CWE-524', 'CWE-311', 'CWE-1258', 'CWE-226', 'CWE-312', 'CWE-318', 'CWE-1278', 'CWE-285', 'CWE-200', 
              'CWE-522', 'CWE-294', 'CWE-521', 'CWE-94', 'CWE-272', 'CWE-59', 'CWE-282', 'CWE-270', 'CWE-1021', 
              'CWE-284', 'CWE-307', 'CWE-798', 'CWE-204', 'CWE-208', 'CWE-425', 'CWE-288', 'CWE-732', 'CWE-276', 
              'CWE-693', 'CWE-441', 'CWE-923', 'CWE-257', 'CWE-494', 'CWE-552', 'CWE-287', 'CWE-290', 'CWE-302', 
              'CWE-346', 'CWE-539', 'CWE-384', 'CWE-664', 'CWE-602', 'CWE-113', 'CWE-20', 'CWE-472', 'CWE-488', 
              'CWE-330', 'CWE-327', 'CWE-325', 'CWE-117', 'CWE-451', 'CWE-359'],
    'CVE_ID': ['CVE-2021-44228'],  # Kept your example CVE
    'Pulse_ID': ['']
    }

    # Process evidence
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)

    # Calculate probabilities
    results_df = calculate_apt_probabilities(encoded_evidence, model, inference, apt_profiles)

    # Calculate simple ranking based only on probability
    final_results = calculate_simple_ranking(results_df, apt_profiles, apt_name_dict)

    # Create similarity matrix for heatmap
    similarity_df = create_similarity_matrix(apt_profiles, encoded_evidence, apt_name_dict)

    # Display ALL results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    print("\nBayesian Network Results (Ranked by Probability):")
    print("=" * 80)
    print(final_results[['Rank', 'APT_Name', 'Probability', 'Features_Processed']].head(20))
    
    print(f"\nTotal APTs analyzed: {len(final_results)}")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"Highest probability: {final_results['Probability'].max():.6f}")
    print(f"Lowest probability: {final_results['Probability'].min():.6f}")
    print(f"Average probability: {final_results['Probability'].mean():.6f}")

    # Plot the similarity heatmap
    print("\nGenerating similarity heatmap...")
    plot_enhanced_similarity_matrix(similarity_df)

    # ===== OPTIMIZED EVALUATION USING EXISTING MATRIX =====
    top_x_values = [1, 5, 10, 20]
    
    # 1. Evaluation with full profile using existing similarity matrix
    print(f"\n{'='*80}")
    print("EVALUATION WITH FULL PROFILE (using existing similarity matrix)")
    print(f"{'='*80}")
    full_profile_results = evaluate_with_similarity_matrix(similarity_df, top_x_values)
    
    dividerVariable = 2
    # 2. Evaluation with divided profiles
    print(f"\n{'='*80}")
    print("EVALUATION WITH DIVIDED PROFILES (divider = "+ str(dividerVariable)+" )")
    print(f"{'='*80}")
    divided_profile_results = evaluate_divided_profiles(
        apt_profiles, apt_name_dict, encoded_evidence, 
        divider=dividerVariable, top_x_values=top_x_values
    )

if __name__ == "__main__":
    main()