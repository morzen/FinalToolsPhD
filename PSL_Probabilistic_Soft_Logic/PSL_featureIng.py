import pandas as pd
import numpy as np
import math
from collections import defaultdict

# ==============================================
# DATA LOADING AND PROCESSING (from your original code)
# ==============================================

pwd = "./DataForPSL/"
df_encoded = pd.read_csv(pwd+'data_for_PSL.csv')
df_mapping = pd.read_csv(pwd+'comparison_Encodedtest_df.csv')

# Create APT profiles
apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))

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

apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)

# Convert sets to lists and add counts
for apt_name in apt_profiles:
    profile = apt_profiles[apt_name]
    for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
        profile[key] = list(profile[key])
        profile[f'{key}_count'] = len(profile[key])

# ==============================================
# EVIDENCE PROCESSING (from your original code)
# ==============================================

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

clean_evidence = sanitize_evidence(evidence, df_mapping)
encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)

# ==============================================
# PURE PYTHON PSL IMPLEMENTATION
# ==============================================

class PSLEngine:
    def __init__(self, apt_profiles, feature_weights):
        self.apt_profiles = apt_profiles
        self.feature_weights = feature_weights
        self.apt_names = list(apt_profiles.keys())
    
    @staticmethod
    def _logistic(x):
        """Logistic function to simulate PSL's probabilistic approach"""
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def _lukasiewicz(a, b):
        """Lukasiewicz t-norm for soft logical AND"""
        return max(0, a + b - 1)
    
    def _calculate_feature_similarity(self, evidence_set, profile_set, weight):
        """Calculate similarity for a single feature type"""
        common = len(evidence_set & profile_set)
        total_evidence = len(evidence_set)
        total_profile = len(profile_set)
        
        if total_evidence == 0 or total_profile == 0:
            return 0.0
        
        # Calculate Jaccard similarity weighted by feature importance
        jaccard = common / (total_evidence + total_profile - common)
        return jaccard * weight
    
    def infer_similarity(self, encoded_evidence):
        """Main inference method that mimics PSL"""
        results = {}
        
        # Calculate total possible weight for normalization
        total_possible_weight = sum(
            weight for feature_type, weight in self.feature_weights.items() 
            if feature_type in encoded_evidence
        )
        
        for apt_name in self.apt_names:
            profile = self.apt_profiles[apt_name]
            weighted_sum = 0.0
            
            # Calculate weighted similarity for each feature type
            for feature_type, weight in self.feature_weights.items():
                if feature_type in encoded_evidence and feature_type in profile:
                    evidence_set = set(encoded_evidence[feature_type])
                    profile_set = set(profile[feature_type])
                    
                    similarity = self._calculate_feature_similarity(
                        evidence_set, profile_set, weight
                    )
                    weighted_sum += similarity
            
            # Normalize by total possible weight
            normalized_score = weighted_sum / total_possible_weight if total_possible_weight > 0 else 0
            results[apt_name] = min(1.0, max(0.0, normalized_score))  # Ensure score is between 0 and 1
        
        return results

# Feature weights
feature_weights = {
    'techniques': 1.5,
    'software': 1.3,
    'capec': 1.0,
    'cwe': 1.0,
    'cve': 0.8,
    'pulse': 0.7
}

# Create and run PSL engine
psl_engine = PSLEngine(apt_profiles, feature_weights)
similarity_scores = psl_engine.infer_similarity(encoded_evidence)

# ==============================================
# RESULTS PROCESSING AND DISPLAY
# ==============================================

# Prepare results with detailed feature matches
results = []
for apt_name, score in similarity_scores.items():
    profile = apt_profiles[apt_name]
    matches = {}
    
    # Calculate matches per feature type
    for feature_type in feature_weights:
        if feature_type in encoded_evidence and feature_type in profile:
            evidence_set = set(encoded_evidence[feature_type])
            profile_set = set(profile[feature_type])
            matches[feature_type] = len(evidence_set & profile_set)
        else:
            matches[feature_type] = 0
    
    results.append({
        'APT': apt_name,
        'Score': score,
        'Encoded_ID': profile['encoded_id'],
        **matches
    })

# Sort by score descending
results.sort(key=lambda x: x['Score'], reverse=True)

# Add ranks
for i, result in enumerate(results):
    result['Rank'] = i + 1

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Display results
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("\nAPT Similarity Results using PSL-inspired Approach:")
print("="*120)
print(f"{'Rank':<5} | {'APT':<30} | {'Encoded ID':<10} | {'Score':<8} | {'Techniques':<10} | {'Software':<8} | {'CAPEC':<6} | {'CWE':<4} | {'CVE':<4} | {'Pulse':<5}")
print("-"*120)

for _, row in results_df.iterrows():
    apt_name = (row['APT'][:27] + '..') if len(row['APT']) > 29 else row['APT']
    print(f"{row['Rank']:<5} | {apt_name:<30} | {row['Encoded_ID']:<10} | {row['Score']:.4f} | {row['techniques']:<10} | {row['software']:<8} | {row['capec']:<6} | {row['cwe']:<4} | {row['cve']:<4} | {row['pulse']:<5}")

# Add these imports at the top of the script if not already present
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# SIMILARITY MATRIX GENERATION
# ==============================================

# Generate full similarity matrix for all APT pairs
num_apts = len(apt_profiles)
apt_names = list(apt_profiles.keys())
similarity_matrix = np.zeros((num_apts, num_apts))

print("Computing PSL similarity matrix...")
for i in range(num_apts):
    apt1_features = {
        'techniques': apt_profiles[apt_names[i]]['techniques'],
        'software': apt_profiles[apt_names[i]]['software'],
        'capec': apt_profiles[apt_names[i]]['capec'],
        'cwe': apt_profiles[apt_names[i]]['cwe'],
        'cve': apt_profiles[apt_names[i]]['cve'],
        'pulse': apt_profiles[apt_names[i]]['pulse']
    }
    
    for j in range(i, num_apts):  # Only compute upper triangle
        apt2_features = {
            'techniques': apt_profiles[apt_names[j]]['techniques'],
            'software': apt_profiles[apt_names[j]]['software'],
            'capec': apt_profiles[apt_names[j]]['capec'],
            'cwe': apt_profiles[apt_names[j]]['cwe'],
            'cve': apt_profiles[apt_names[j]]['cve'],
            'pulse': apt_profiles[apt_names[j]]['pulse']
        }
        
        # Create temporary evidence for comparison
        temp_evidence = {
            'techniques': apt2_features['techniques'],
            'software': apt2_features['software'],
            'capec': apt2_features['capec'],
            'cwe': apt2_features['cwe'],
            'cve': apt2_features['cve'],
            'pulse': apt2_features['pulse']
        }
        
        # Get similarity score
        score = psl_engine.infer_similarity(temp_evidence)[apt_names[i]]
        similarity_matrix[i, j] = score
        similarity_matrix[j, i] = score  # Symmetric matrix

# Get display names for APTs
display_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_names]

# Convert the numpy array to a DataFrame
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=display_names,
    columns=display_names
)

print("Similarity matrix computation completed!")
print(f"Similarity DataFrame shape: {similarity_df.shape}")

# ==============================================
# VISUALIZATION
# ==============================================

# Create the heatmap with green for high similarity, red for low similarity
plt.figure(figsize=(16, 14))
sns.set(font_scale=0.7)

# Create custom green-to-red colormap (reversed YlOrRd)
from matplotlib.colors import LinearSegmentedColormap
colors = ["red", "yellow", "green"]  # Red for low, green for high
n_bins = 100
cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)

ax = sns.heatmap(
    similarity_matrix,
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
plt.title("APT Group Similarity Matrix (PSL-inspired Approach)\nGreen = High Similarity, Red = Low Similarity", 
          fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

# Show the plot
plt.show()

# Also show a text version of the top 10x10 for verification
print("\nText version of similarity matrix (first 10x10):")
print("="*90)
print(f"{'':<20}", end="")
for name in display_names[:10]:
    print(f"{name[:8]:<8}", end="")
print("\n" + "-"*90)

for i in range(10):
    print(f"{display_names[i][:20]:<20}", end="")
    for j in range(10):
        print(f"{similarity_matrix[i,j]:.2f}  ", end="")
    print()

# ==============================================
# EVALUATION FUNCTIONS
# ==============================================

from itertools import islice
import random

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

def evaluate_profile_division(apt_profiles, apt_name_dict, df_mapping, divider=2, top_x_values=[1, 5, 10, 20]):
    """Evaluate model performance with divided profile data"""
    print(f"Using {100/divider:.1f}% of profile data")
    
    resultdict = {}
    apt_names = list(apt_profiles.keys())
    
    # Show progress
    total_apts = len(apt_names)
    print(f"Processing {total_apts} APTs...")
    
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_names):
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{total_apts} APTs...")
            
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
        
        # Calculate similarities with divided data
        psl_engine = PSLEngine(apt_profiles, feature_weights)
        similarities = psl_engine.infer_similarity(encoded_evidence)
        
        resultdict[target_apt_name] = similarities
    
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

# ==============================================
# EVALUATION CALLS
# ==============================================

print("\n" + "="*80)
print("EVALUATION WITH FULL PROFILE")
print("="*80)
evaluate_similarity_matrix(similarity_df)

print("\n" + "="*80)
print("EVALUATION WITH DIVIDED PROFILES")
print("="*80)

# Set your custom divider value here
CUSTOM_DIVIDER = 2  # Change this to your desired value
evaluate_profile_division(apt_profiles, apt_name_dict, df_mapping, divider=CUSTOM_DIVIDER)