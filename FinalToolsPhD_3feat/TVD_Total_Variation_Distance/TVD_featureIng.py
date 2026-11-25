import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
import random
import math
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P   # shared evaluation protocol (profiles, ranking, divided, false flag)

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
                'cve': set()
            }
        
        if row['techniques_str'] != -1:
            profiles[apt_name]['techniques'].add(row['techniques_str'])
        if row['software_str'] != -1:
            profiles[apt_name]['software'].add(row['software_str'])
        if row['CVE_ID'] != -1:
            profiles[apt_name]['cve'].add(row['CVE_ID'])
    
    return profiles


# Evidence processing functions
def sanitize_evidence(evidence, df_mapping):
    sanitized = {}
    missing = {}
    
    column_mapping = {
        'techniques_str': ('techniques_Origin', 'techniques_Encoded'),
        'software_str': ('software_Origin', 'software_Encoded'),
        'CVE_ID': ('CVE_ID_Origin', 'CVE_ID_Encoded')
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
        'CVE_ID': ('CVE_ID_Encoded', 'cve')
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
    """Weighted TVD similarity between the evidence and every APT profile (thesis App. 8.3.1).

    For each feature type the evidence (Q) and the profile (P) are uniform distributions over
    their feature sets, and TVD(P, Q) = 1/2 * sum_x |P(x) - Q(x)|. With E = evidence set,
    A = profile set and I = |E n A| this sum has the exact closed form
        TVD = 1/2 * ( (|E| - I)/|E| + (|A| - I)/|A| + I * | 1/|E| - 1/|A| | )
    (identical values to building the full 0/1 vectors, much faster). The per-type similarity
    is 1 - TVD, combined with the feature weights. Feature types without evidence are skipped:
    an empty evidence set has no distribution."""
    all_features = {}
    for feature_type in feature_weights:
        union = set()
        for apt_name in apt_profiles:
            union.update(apt_profiles[apt_name][feature_type])
        all_features[feature_type] = union

    evidence_sets = {}
    for feature_type in feature_weights:
        e = set(encoded_evidence.get(feature_type, [])) & all_features[feature_type]
        if e:
            evidence_sets[feature_type] = e

    similarities = []
    for apt_name in apt_profiles:
        apt_data = apt_profiles[apt_name]
        total_similarity = 0
        total_weight = 0
        for feature_type, weight in feature_weights.items():
            if feature_type not in evidence_sets:
                continue
            E = evidence_sets[feature_type]
            A = set(apt_data[feature_type])
            if A:
                inter = len(E & A)
                tvd = 0.5 * ((len(E) - inter) / len(E) + (len(A) - inter) / len(A)
                             + inter * abs(1.0 / len(E) - 1.0 / len(A)))
            else:
                tvd = 0.5          # evidence mass has no counterpart in an empty profile
            total_similarity += (1 - tvd) * weight
            total_weight += weight
        similarities.append(total_similarity / total_weight if total_weight > 0 else 0)
    return similarities


# ==============================================
# SINGLE SCORING PATH (used by hybrid_query, evaluation and false flag)
# ==============================================
FEATURE_WEIGHTS = {'techniques': 6/6, 'software': 5/6, 'cve': 2/6}   # rank-sum weights of the six levels, 6:5:4:3:2:1 (same rule in every approach; thesis Appendix C)


class TVDScorer:
    """Scores encoded evidence against every APT profile with TVD."""

    def __init__(self):
        self.profiles = P.load_profiles()            # -1 (missing) is never a feature
        self.apt_names = list(self.profiles.keys())
        self.weights = FEATURE_WEIGHTS
        self._df_mapping = None

    @property
    def df_mapping(self):
        if self._df_mapping is None:
            self._df_mapping = pd.read_csv(os.path.join(P.BASE, 'comparison_Encoded_df.csv'))
        return self._df_mapping

    def score(self, encoded_evidence):
        return np.asarray(calculate_tvd_similarity(P.non_empty(encoded_evidence), self.profiles, self.weights),
                          dtype=float)


_SCORER = None


def get_scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = TVDScorer()
    return _SCORER



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
        'cve': set()
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



def hybrid_query(evidence_forhybrid):
    """CTI / hybrid entry point: human-readable evidence -> ranked list of all APTs (mid-ranks)."""
    key_mapping = {'techniques': 'techniques_str', 'software': 'software_str', 'cve': 'CVE_ID'}
    renamed_evidence = {}
    for old_key, new_key in key_mapping.items():
        if old_key in evidence_forhybrid:
            renamed_evidence[new_key] = evidence_forhybrid[old_key]

    scorer = get_scorer()
    clean_evidence = sanitize_evidence(renamed_evidence, scorer.df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, scorer.df_mapping)
    scores = scorer.score(encoded_evidence)
    return P.ranked_results(scorer.apt_names, scores, score_key='Score')

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
        for key in ['techniques', 'software', 'cve']:
            profile[key] = list(profile[key])
            profile[f'{key}_count'] = len(profile[key])


    # Feature weights for TVD calculation
    feature_weights = dict(FEATURE_WEIGHTS)   # the single definition above (rank-sum weights)

    # Load evidence from query_input.json
    try:
        with open('../query_input.json', 'r') as f:
            evidence_data = json.load(f)
        
        # Transform the evidence to match expected format
        evidence = {
            'techniques_str': evidence_data.get('techniques', []),
            'software_str': evidence_data.get('software', []),
            'CVE_ID': evidence_data.get('cve', [])
        }
        print("Successfully loaded evidence from query_input.json")
        print(f"Found: {len(evidence['techniques_str'])} techniques, {len(evidence['software_str'])} software, "
              f"{len(evidence['CVE_ID'])} cve")
    
    except FileNotFoundError:
        print("Warning: query_input.json not found, using default evidence")
    
    except json.JSONDecodeError:
        print("Error: query_input.json contains invalid JSON")
        return  

        # evidence = {
        #     'techniques_str': [
        #         'T0865', 'T1001.003', 'T1005', 'T1008', 'T1010', 'T1012', 'T1016', 
        #         'T1021.001', 'T1021.002', 'T1021.004', 'T1027.002', 'T1027.007', 
        #         'T1027.013', 'T1033', 'T1036.003', 'T1036.004', 'T1036.005', 
        #         'T1036.008', 'T1041', 'T1046', 'T1047', 'T1048.003', 'T1049', 
        #         'T1053.005', 'T1055.001', 'T1056.001', 'T1057', 'T1059.001', 
        #         'T1059.003', 'T1059.005', 'T1070', 'T1070.003', 'T1070.004', 
        #         'T1070.006', 'T1071.001', 'T1074.001', 'T1078', 'T1082', 'T1083', 
        #         'T1087.002', 'T1090.001', 'T1090.002', 'T1098', 'T1102.002', 
        #         'T1104', 'T1105', 'T1106', 'T1110.003', 'T1124', 'T1132.001', 
        #         'T1134.002', 'T1140', 'T1189', 'T1202', 'T1203', 'T1204.001', 
        #         'T1204.002', 'T1218', 'T1218.005', 'T1218.010', 'T1218.011', 
        #         'T1220', 'T1221', 'T1485', 'T1489', 'T1491.001', 'T1497.001', 
        #         'T1497.003', 'T1505.004', 'T1529', 'T1534', 'T1542.003', 
        #         'T1543.003', 'T1547.001', 'T1547.009', 'T1553.002', 'T1557.001', 
        #         'T1560', 'T1560.001', 'T1560.002', 'T1560.003', 'T1561.001', 
        #         'T1561.002', 'T1562.001', 'T1562.004', 'T1564.001', 'T1566.001', 
        #         'T1566.002', 'T1566.003', 'T1567.002', 'T1571', 'T1573.001', 
        #         'T1574.002', 'T1574.013', 'T1583.001', 'T1583.004', 'T1583.006', 
        #         'T1584.001', 'T1584.004', 'T1585.001', 'T1585.002', 'T1587.001', 
        #         'T1587.002', 'T1588.002', 'T1588.003', 'T1588.004', 'T1589.002', 
        #         'T1591', 'T1591.004', 'T1593.001', 'T1608.001', 'T1608.002', 
        #         'T1614.001', 'T1620', 'T1622', 'T1656'
        #     ],
        #     'software_str': [
        #         'S0181', 'S0239', 'S0245', 'S0246', 'S0586', 'S0238', 'S0498', 
        #         'S0520', 'S0567', 'S0665', 'S0694', 'S0376', 'S0431', 'S0180', 
        #         'S0241', 'S0103', 'S0271', 'S0366', 'S0678', 'S0263', 'S0347', 
        #         'S0497', 'S0584', 'S0593', 'S0364', 'S0174', 'S0108'
        #     ],
        #     'CAPEC_ID': [
        #         'CAPEC-204', 'CAPEC-37', 'CAPEC-545', 'CAPEC-647', 'CAPEC-309', 
        #         'CAPEC-561', 'CAPEC-577', 'CAPEC-165', 'CAPEC-504', 'CAPEC-543', 
        #         'CAPEC-616', 'CAPEC-300', 'CAPEC-568', 'CAPEC-573', 'CAPEC-268', 
        #         'CAPEC-560', 'CAPEC-70', 'CAPEC-312', 'CAPEC-313', 'CAPEC-580', 
        #         'CAPEC-127', 'CAPEC-497', 'CAPEC-465', 'CAPEC-565', 'CAPEC-295', 
        #         'CAPEC-196', 'CAPEC-698', 'CAPEC-163', 'CAPEC-552', 'CAPEC-478', 
        #         'CAPEC-132', 'CAPEC-206', 'CAPEC-473', 'CAPEC-68', 'CAPEC-578', 
        #         'CAPEC-641', 'CAPEC-640', 'CAPEC-542'
        #     ],
        #     'CWE_ID': [
        #         'CWE-524', 'CWE-311', 'CWE-1258', 'CWE-226', 'CWE-312', 'CWE-318', 
        #         'CWE-1278', 'CWE-285', 'CWE-200', 'CWE-522', 'CWE-294', 'CWE-521', 
        #         'CWE-1021', 'CWE-117', 'CWE-307', 'CWE-798', 'CWE-204', 'CWE-208', 
        #         'CWE-425', 'CWE-288', 'CWE-732', 'CWE-276', 'CWE-693', 'CWE-441', 
        #         'CWE-384', 'CWE-664', 'CWE-829', 'CWE-451', 'CWE-284', 'CWE-59', 
        #         'CWE-20', 'CWE-327', 'CWE-290', 'CWE-325', 'CWE-706', 'CWE-114'
        #     ],
        #     'CVE_ID': [],  # No CVE data provided in the original
        #     'Pulse_ID': []  # No Pulse data provided in the original
        # }


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
    print(f"{'Rank':<5} | {'APT':<30} | {'Encoded ID':<10} | {'Score':<8} | {'Techniques':<10} | {'Software':<8} | {'CVE':<4}")
    print("-"*110)

    for rank, (apt_name, score) in enumerate(ranked_results, 1):
        apt_data = apt_profiles[apt_name]
        common_features = {
            'techniques': len(set(encoded_evidence.get('techniques', [])) & set(apt_data['techniques'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve']))
        }
        
        # Truncate long APT names to fit column width
        display_name = (apt_name[:27] + '..') if len(apt_name) > 29 else apt_name
        print(f"{rank:<5} | {display_name:<30} | {apt_data['encoded_id']:<10} | {score:.4f} | {common_features['techniques']:<10} | {common_features['software']:<8} | {common_features['cve']:<4}")


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



    # ==============================================
    # ATTRIBUTION EVALUATION (shared protocol: full data and seeded shuffled & divided)
    # ==============================================
    print("\n" + "="*80)
    print("ATTRIBUTION EVALUATION (full profile and shuffled & divided by 2)")
    print("="*80)
    if "--heatmaps-only" not in sys.argv:        # --heatmaps-only: only the heatmap above
        P.evaluate_attribution(get_scorer(), "TVD", results_dir="results", smoke=P.smoke_flag())


if __name__ == "__main__":
    main()