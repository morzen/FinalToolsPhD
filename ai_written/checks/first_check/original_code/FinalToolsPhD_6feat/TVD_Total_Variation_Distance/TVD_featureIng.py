import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import random
import math
import json

# Create APT profiles with actual names as keys
def create_apt_profiles_with_names(df_encoded, name_mapping):
    profiles = {}
    
    for _, row in df_encoded.iterrows():
        apt_encoded = row['APTname']
        apt_name = name_mapping.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
        
        if apt_name not in profiles:
            profiles[apt_name] = {
                'encoded_id': apt_encoded,
                'techniques': set(),
                'software': set(),
                'capec': set(),
                'cwe': set(),
                'cve': set(),
                'pulse': set()
            }
        
        if row['techniques_str'] != -1:
            profiles[apt_name]['techniques'].add(row['techniques_str'])
        if row['software_str'] != -1:
            profiles[apt_name]['software'].add(row['software_str'])
        if row['CAPEC_ID'] != -1:
            profiles[apt_name]['capec'].add(row['CAPEC_ID'])
        if row['CWE_ID'] != -1:
            profiles[apt_name]['cwe'].add(row['CWE_ID'])
        if row['CVE_ID'] != -1:
            profiles[apt_name]['cve'].add(row['CVE_ID'])
        if row['Pulse_ID'] != -1:
            profiles[apt_name]['pulse'].add(row['Pulse_ID'])
    
    return profiles


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
        print("WARNING: These features were not found in our database:")
        for feature_type, items in missing.items():
            print(f"- {feature_type}: {items[:5]}")
            if len(items) > 5:
                print(f"  (and {len(items)-5} more)")
    
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

# Total Variation Distance implementation
def calculate_tvd_similarity(encoded_evidence, apt_profiles, feature_weights):
    """Calculate similarity using Total Variation Distance with feature weighting"""
    similarities = []
    
    # First get all possible features to create probability distributions
    all_features = {
        'techniques': set(),
        'software': set(),
        'capec': set(),
        'cwe': set(),
        'cve': set(),
        'pulse': set()
    }
    
    # Collect all unique features across all APTs
    for apt_name in apt_profiles:
        for feature_type in all_features:
            all_features[feature_type].update(apt_profiles[apt_name][feature_type])
    
    # Convert to lists for indexing
    feature_indices = {ft: list(features) for ft, features in all_features.items()}
    
    for apt_name in apt_profiles:
        apt_data = apt_profiles[apt_name]
        total_similarity = 0
        total_weight = 0
        
        for feature_type, weight in feature_weights.items():
            # Skip if evidence doesn't have this feature type
            if feature_type not in encoded_evidence:
                continue
                
            # Create probability distributions
            evidence_features = set(encoded_evidence[feature_type])
            profile_features = set(apt_data[feature_type])
            all_possible = feature_indices[feature_type]
            
            # Create binary vectors (1 if feature present, 0 otherwise)
            evidence_vec = np.array([1 if f in evidence_features else 0 for f in all_possible])
            profile_vec = np.array([1 if f in profile_features else 0 for f in all_possible])
            
            # Normalize to probability distributions
            evidence_dist = evidence_vec / evidence_vec.sum() if evidence_vec.sum() > 0 else evidence_vec
            profile_dist = profile_vec / profile_vec.sum() if profile_vec.sum() > 0 else profile_vec
            
            # Calculate TVD (half the L1 distance between distributions)
            tvd = 0.5 * np.sum(np.abs(evidence_dist - profile_dist))
            
            # Convert to similarity (1 - TVD)
            feature_similarity = (1 - tvd) * weight
            total_similarity += feature_similarity
            total_weight += weight
        
        # Normalize by total weight
        normalized_similarity = total_similarity / total_weight if total_weight > 0 else 0
        similarities.append(normalized_similarity)
    
    return similarities


# ==============================================
# SIMILARITY MATRIX GENERATION
# ==============================================

def calculate_tvd_matrix(apt_profiles, feature_weights):
    """Calculate TVD similarity matrix for all APT pairs"""
    num_apts = len(apt_profiles)
    apt_names = list(apt_profiles.keys())
    similarity_matrix = np.zeros((num_apts, num_apts))
    
    # First get all possible features to create probability distributions
    all_features = {
        'techniques': set(),
        'software': set(),
        'capec': set(),
        'cwe': set(),
        'cve': set(),
        'pulse': set()
    }
    
    # Collect all unique features across all APTs
    for apt_name in apt_profiles:
        for feature_type in all_features:
            all_features[feature_type].update(apt_profiles[apt_name][feature_type])
    
    # Convert to lists for indexing
    feature_indices = {ft: list(features) for ft, features in all_features.items()}
    
    print("Computing TVD similarity matrix...")
    for i in range(num_apts):
        apt1_name = apt_names[i]
        apt1_data = apt_profiles[apt1_name]
        
        for j in range(i, num_apts):  # Only compute upper triangle
            apt2_name = apt_names[j]
            apt2_data = apt_profiles[apt2_name]
            
            total_similarity = 0
            total_weight = 0
            
            for feature_type, weight in feature_weights.items():
                # Create probability distributions
                apt1_features = set(apt1_data[feature_type])
                apt2_features = set(apt2_data[feature_type])
                all_possible = feature_indices[feature_type]
                
                # Create binary vectors (1 if feature present, 0 otherwise)
                apt1_vec = np.array([1 if f in apt1_features else 0 for f in all_possible])
                apt2_vec = np.array([1 if f in apt2_features else 0 for f in all_possible])
                
                # Normalize to probability distributions
                apt1_dist = apt1_vec / apt1_vec.sum() if apt1_vec.sum() > 0 else apt1_vec
                apt2_dist = apt2_vec / apt2_vec.sum() if apt2_vec.sum() > 0 else apt2_vec
                
                # Calculate TVD (half the L1 distance between distributions)
                tvd = 0.5 * np.sum(np.abs(apt1_dist - apt2_dist))
                
                # Convert to similarity (1 - TVD)
                feature_similarity = (1 - tvd) * weight
                total_similarity += feature_similarity
                total_weight += weight
            
            # Normalize by total weight
            normalized_similarity = total_similarity / total_weight if total_weight > 0 else 0
            similarity_matrix[i, j] = normalized_similarity
            similarity_matrix[j, i] = normalized_similarity  # Symmetric matrix
    
    print("TVD similarity matrix computation completed!")
    return similarity_matrix


# ==============================================
# EVALUATION FUNCTIONS
# ==============================================

def evaluate_similarity_matrix(similarity_df, top_x_values=[1, 5, 10, 20]):
    """Evaluate if each APT appears in its own top-X most similar APTs"""
    print("Using 100% of profile data (full similarity matrix)")
    
    apt_names = similarity_df.index.tolist()
    resultdict = {}
    
    # For each APT, get its similarity scores from the matrix
    for target_apt_name in apt_names:
        similarities = similarity_df[target_apt_name].to_dict()
        resultdict[target_apt_name] = similarities
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs
            top_x_similar = dict(islice(sorted_similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success = success + 1
            else:
                failure = failure + 1
        
        print(f"Top {topX}: {success}/{failure} (Success rate: {(success/lengthResultDict)*100:.2f}%)")

def evaluate_profile_division(feature_weights, apt_profiles, apt_name_dict, df_mapping, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate model performance with divided profile data"""
    print(f"Using {100/divider:.1f}% of profile data")
    
    resultdict = {}
    apt_names = list(apt_profiles.keys())
    
    # Show progress
    total_apts = len(apt_names)
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_names):
        
            
        target_data = apt_profiles[target_apt_name]
        
        # Create evidence with divided data for each feature type
        divided_evidence = {}
        
        for feature_type, evidence_key in [('techniques', 'techniques_str'),
                                         ('software', 'software_str'),
                                         ('capec', 'CAPEC_ID'),
                                         ('cwe', 'CWE_ID'),
                                         ('cve', 'CVE_ID'),
                                         ('pulse', 'Pulse_ID')]:
            
            if feature_type in target_data and target_data[feature_type]:
                feature_list = target_data[feature_type].copy()
                random.shuffle(feature_list)
                division_size = max(1, len(feature_list) // divider)
                divided_features = feature_list[:division_size]
                
                # Convert encoded features back to original format for sanitization
                if feature_type == 'techniques':
                    mapping = dict(zip(df_mapping['techniques_Encoded'], df_mapping['techniques_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'software':
                    mapping = dict(zip(df_mapping['software_Encoded'], df_mapping['software_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'capec':
                    mapping = dict(zip(df_mapping['CAPEC_ID_Encoded'], df_mapping['CAPEC_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'cwe':
                    mapping = dict(zip(df_mapping['CWE_ID_Encoded'], df_mapping['CWE_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'cve':
                    mapping = dict(zip(df_mapping['CVE_ID_Encoded'], df_mapping['CVE_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
                elif feature_type == 'pulse':
                    mapping = dict(zip(df_mapping['Pulse_ID_Encoded'], df_mapping['Pulse_ID_Origin']))
                    divided_evidence[evidence_key] = [mapping.get(f, f) for f in divided_features if f in mapping]
        
        # Process the divided evidence
        clean_evidence = sanitize_evidence(divided_evidence, df_mapping)
        encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
        
        # Calculate similarities with divided data using TVD
        similarities = calculate_tvd_similarity(encoded_evidence, apt_profiles, feature_weights)
        apt_names_list = list(apt_profiles.keys())
        similarity_dict = {apt_names_list[i]: similarities[i] for i in range(len(apt_names_list))}
        
        resultdict[target_apt_name] = similarity_dict
    
    # Evaluate for multiple top-X values
    print(f"\nResults based on {len(resultdict)} tests:")
    for topX in top_x_values:
        success = 0
        failure = 0
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs
            top_x_similar = dict(islice(sorted_similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success = success + 1
            else:
                failure = failure + 1
        
        success_rate = (success / len(resultdict)) * 100
        print(f"Top {topX}: {success_rate:.1f}% success rate")

def hybrid_query(hybridEvidence):


    key_mapping = {
        'techniques' : 'techniques_str',
        'software' : 'software_str',
        'capec' : 'CAPEC_ID',
        'cwe' : 'CWE_ID',
        'cve' : 'CVE_ID',
        'pulse' : 'Pulse_ID'
    }
    
    # Replace keys using mapping
    renamed_evidence = {}
    for old_key, new_key in key_mapping.items():
        if old_key in hybridEvidence:
            renamed_evidence[new_key] = hybridEvidence[old_key]


    # Load data files
    pwd = "../BaseData/"
    df_encoded = pd.read_csv(pwd+'EncodedData.csv')
    df_mapping = pd.read_csv(pwd+'comparison_Encoded_df.csv')

    # Create APT name mapping
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))

    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)

    # Convert sets to lists and add counts
    for apt_name in apt_profiles:
        profile = apt_profiles[apt_name]
        for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            profile[key] = list(profile[key])
            profile[f'{key}_count'] = len(profile[key])


    # Feature weights for TVD calculation
    feature_weights = {
        'techniques': 6/6,
        'software': 5/6,
        'capec': 4/6,
        'cwe': 3/6,
        'cve': 2/6,
        'pulse': 1/6
    }

    # Process evidence
    clean_evidence = sanitize_evidence(renamed_evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)

    # Calculate similarities using TVD
    tvd_similarities = calculate_tvd_similarity(encoded_evidence, apt_profiles, feature_weights)

    # Create ranked results
    apt_ids = list(apt_profiles.keys())
    ranked_results = sorted(zip(apt_ids, tvd_similarities), key=lambda x: x[1], reverse=True)
    transformed_list = [
    {'rank': idx + 1, 'apt_name': item[0], 'Score': item[1]} 
    for idx, item in enumerate(ranked_results)
    ]
    return transformed_list

def main():
    # Load data files
    pwd = "../BaseData/"
    df_encoded = pd.read_csv(pwd+'EncodedData.csv')
    df_mapping = pd.read_csv(pwd+'comparison_Encoded_df.csv')

    # Create APT name mapping
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))

    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)

    # Convert sets to lists and add counts
    for apt_name in apt_profiles:
        profile = apt_profiles[apt_name]
        for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            profile[key] = list(profile[key])
            profile[f'{key}_count'] = len(profile[key])


    # Feature weights for TVD calculation
    feature_weights = {
        'techniques': 6/6,
        'software': 5/6,
        'capec': 4/6,
        'cwe': 3/6,
        'cve': 2/6,
        'pulse': 1/6
    }

    # Load evidence from query_input.json
    try:
        with open('../query_input.json', 'r') as f:
            evidence_data = json.load(f)
        
        # Transform the evidence to match expected format
        evidence = {
            'techniques_str': evidence_data.get('techniques', []),
            'software_str': evidence_data.get('software', []),
            'CAPEC_ID': evidence_data.get('capec', []),
            'CWE_ID': evidence_data.get('cwe', []),
            'CVE_ID': evidence_data.get('cve', []),
            'Pulse_ID': evidence_data.get('pulse', [])
        }
        print("Successfully loaded evidence from query_input.json")
        print(f"Found: {len(evidence['techniques_str'])} techniques, {len(evidence['software_str'])} software, "
              f"{len(evidence['CAPEC_ID'])} capec, {len(evidence['CWE_ID'])} cwe, "
              f"{len(evidence['CVE_ID'])} cve, {len(evidence['Pulse_ID'])} pulse")
    except FileNotFoundError:
        print("Warning: query_input.json not found, using default evidence")
        evidence = {
            'techniques_str': [
                'T0865', 'T1001.003', 'T1005', 'T1008', 'T1010', 'T1012', 'T1016', 
                'T1021.001', 'T1021.002', 'T1021.004', 'T1027.002', 'T1027.007', 
                'T1027.013', 'T1033', 'T1036.003', 'T1036.004', 'T1036.005', 
                'T1036.008', 'T1041', 'T1046', 'T1047', 'T1048.003', 'T1049', 
                'T1053.005', 'T1055.001', 'T1056.001', 'T1057', 'T1059.001', 
                'T1059.003', 'T1059.005', 'T1070', 'T1070.003', 'T1070.004', 
                'T1070.006', 'T1071.001', 'T1074.001', 'T1078', 'T1082', 'T1083', 
                'T1087.002', 'T1090.001', 'T1090.002', 'T1098', 'T1102.002', 
                'T1104', 'T1105', 'T1106', 'T1110.003', 'T1124', 'T1132.001', 
                'T1134.002', 'T1140', 'T1189', 'T1202', 'T1203', 'T1204.001', 
                'T1204.002', 'T1218', 'T1218.005', 'T1218.010', 'T1218.011', 
                'T1220', 'T1221', 'T1485', 'T1489', 'T1491.001', 'T1497.001', 
                'T1497.003', 'T1505.004', 'T1529', 'T1534', 'T1542.003', 
                'T1543.003', 'T1547.001', 'T1547.009', 'T1553.002', 'T1557.001', 
                'T1560', 'T1560.001', 'T1560.002', 'T1560.003', 'T1561.001', 
                'T1561.002', 'T1562.001', 'T1562.004', 'T1564.001', 'T1566.001', 
                'T1566.002', 'T1566.003', 'T1567.002', 'T1571', 'T1573.001', 
                'T1574.002', 'T1574.013', 'T1583.001', 'T1583.004', 'T1583.006', 
                'T1584.001', 'T1584.004', 'T1585.001', 'T1585.002', 'T1587.001', 
                'T1587.002', 'T1588.002', 'T1588.003', 'T1588.004', 'T1589.002', 
                'T1591', 'T1591.004', 'T1593.001', 'T1608.001', 'T1608.002', 
                'T1614.001', 'T1620', 'T1622', 'T1656'
            ],
            'software_str': [
                'S0181', 'S0239', 'S0245', 'S0246', 'S0586', 'S0238', 'S0498', 
                'S0520', 'S0567', 'S0665', 'S0694', 'S0376', 'S0431', 'S0180', 
                'S0241', 'S0103', 'S0271', 'S0366', 'S0678', 'S0263', 'S0347', 
                'S0497', 'S0584', 'S0593', 'S0364', 'S0174', 'S0108'
            ],
            'CAPEC_ID': [
                'CAPEC-204', 'CAPEC-37', 'CAPEC-545', 'CAPEC-647', 'CAPEC-309', 
                'CAPEC-561', 'CAPEC-577', 'CAPEC-165', 'CAPEC-504', 'CAPEC-543', 
                'CAPEC-616', 'CAPEC-300', 'CAPEC-568', 'CAPEC-573', 'CAPEC-268', 
                'CAPEC-560', 'CAPEC-70', 'CAPEC-312', 'CAPEC-313', 'CAPEC-580', 
                'CAPEC-127', 'CAPEC-497', 'CAPEC-465', 'CAPEC-565', 'CAPEC-295', 
                'CAPEC-196', 'CAPEC-698', 'CAPEC-163', 'CAPEC-552', 'CAPEC-478', 
                'CAPEC-132', 'CAPEC-206', 'CAPEC-473', 'CAPEC-68', 'CAPEC-578', 
                'CAPEC-641', 'CAPEC-640', 'CAPEC-542'
            ],
            'CWE_ID': [
                'CWE-524', 'CWE-311', 'CWE-1258', 'CWE-226', 'CWE-312', 'CWE-318', 
                'CWE-1278', 'CWE-285', 'CWE-200', 'CWE-522', 'CWE-294', 'CWE-521', 
                'CWE-1021', 'CWE-117', 'CWE-307', 'CWE-798', 'CWE-204', 'CWE-208', 
                'CWE-425', 'CWE-288', 'CWE-732', 'CWE-276', 'CWE-693', 'CWE-441', 
                'CWE-384', 'CWE-664', 'CWE-829', 'CWE-451', 'CWE-284', 'CWE-59', 
                'CWE-20', 'CWE-327', 'CWE-290', 'CWE-325', 'CWE-706', 'CWE-114'
            ],
            'CVE_ID': [],  # No CVE data provided in the original
            'Pulse_ID': []  # No Pulse data provided in the original
        }
    except json.JSONDecodeError:
        print("Error: query_input.json contains invalid JSON")
        return  

    # Process evidence
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)

    # Calculate similarities using TVD
    tvd_similarities = calculate_tvd_similarity(encoded_evidence, apt_profiles, feature_weights)

    # Create ranked results
    apt_ids = list(apt_profiles.keys())
    ranked_results = sorted(zip(apt_ids, tvd_similarities), key=lambda x: x[1], reverse=True)

    # Format and print results
    print("\nAPT Ranking using Total Variation Distance:")
    print("="*110)
    print(f"{'Rank':<5} | {'APT':<30} | {'Encoded ID':<10} | {'Score':<8} | {'Techniques':<10} | {'Software':<8} | {'CAPEC':<6} | {'CWE':<4} | {'CVE':<4} | {'Pulse':<5}")
    print("-"*110)

    for rank, (apt_name, score) in enumerate(ranked_results, 1):
        apt_data = apt_profiles[apt_name]
        common_features = {
            'techniques': len(set(encoded_evidence.get('techniques', [])) & set(apt_data['techniques'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'capec': len(set(encoded_evidence.get('capec', [])) & set(apt_data['capec'])),
            'cwe': len(set(encoded_evidence.get('cwe', [])) & set(apt_data['cwe'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve'])),
            'pulse': len(set(encoded_evidence.get('pulse', [])) & set(apt_data['pulse']))
        }
        
        # Truncate long APT names to fit column width
        display_name = (apt_name[:27] + '..') if len(apt_name) > 29 else apt_name
        print(f"{rank:<5} | {display_name:<30} | {apt_data['encoded_id']:<10} | {score:.4f} | {common_features['techniques']:<10} | {common_features['software']:<8} | {common_features['capec']:<6} | {common_features['cwe']:<4} | {common_features['cve']:<4} | {common_features['pulse']:<5}")


    # Calculate the full similarity matrix
    tvd_similarity_matrix = calculate_tvd_matrix(apt_profiles, feature_weights)

    # Get display names for APTs
    display_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_profiles.keys()]

    # ==============================================
    # VISUALIZATION
    # ==============================================

    # Create the heatmap
    plt.figure(figsize=(16, 14))
    sns.set(font_scale=0.7)

    # Create custom green-to-red colormap (reversed YlOrRd)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]  # Red for low, green for high
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)

    ax = sns.heatmap(
        tvd_similarity_matrix,
        cmap=cmap,
        vmin=0,  # Minimum similarity value
        vmax=1,  # Maximum similarity value
        center=0.5,  # Center the colormap
        xticklabels=display_names,
        yticklabels=display_names,
        annot=False,  # Don't show values to avoid clutter
        square=True,
        cbar_kws={'label': 'Similarity Score', 'shrink': 0.8}
    )

    # Adjust layout
    plt.title("APT Group Similarity Matrix (Total Variation Distance)\nGreen = High Similarity, Red = Low Similarity", 
            fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # Show the plot
    plt.savefig('./TVD_similarity_heatmap.png')  # adjust path as needed
    plt.show()

    # Also show a text version of the top 10x10 for verification
    print("\nText version of TVD similarity matrix (first 10x10):")
    print("="*90)
    print(f"{'':<20}", end="")
    for name in display_names[:10]:
        print(f"{name[:8]:<8}", end="")
    print("\n" + "-"*90)

    for i in range(10):
        print(f"{display_names[i][:20]:<20}", end="")
        for j in range(10):
            print(f"{tvd_similarity_matrix[i,j]:.2f}  ", end="")
        print()

    # Convert the numpy array to a DataFrame
    similarity_df = pd.DataFrame(
        tvd_similarity_matrix,
        index=display_names,
        columns=display_names
    )


    print("\n" + "="*80)
    print("EVALUATION WITH FULL PROFILE")
    print("="*80)
    evaluate_similarity_matrix(similarity_df)

    print("\n" + "="*80)
    print("EVALUATION WITH DIVIDED PROFILES")
    print("="*80)

    # Set your custom divider value here
    CUSTOM_DIVIDER = 2  # Change this to your desired value
    # Change this line in main():
    evaluate_profile_division(feature_weights, apt_profiles, apt_name_dict, df_mapping, divider=CUSTOM_DIVIDER)


if __name__ == "__main__":
    main()