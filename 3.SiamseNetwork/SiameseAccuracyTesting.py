# siamese_query.py
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ---------- Recreate the APTSiameseNetwork definition (same as in training) ----------
import torch.nn as nn
class APTSiameseNetwork(nn.Module):
    def __init__(self, num_apts, embedding_dim=64):
        super().__init__()
        self.apt_embedding = nn.Embedding(num_apts, embedding_dim)
        
    def forward(self, apt_idx1, apt_idx2):
        emb1 = self.apt_embedding(apt_idx1)
        emb2 = self.apt_embedding(apt_idx2)
        return emb1, emb2
    
    def get_similarity(self, emb1, emb2):
        # Use PyTorch cosine (works for single vectors)
        # emb1, emb2 are 1D tensors; ensure shape consistency
        emb1 = emb1.view(1, -1)
        emb2 = emb2.view(1, -1)
        return F.cosine_similarity(emb1, emb2).squeeze()
    
    def get_all_embeddings(self):
        return self.apt_embedding.weight

# ---------- Helper functions (re-used from your GNN script) ----------
def sanitize_evidence(evidence, df_mapping):
    """Keep only evidence features that exist in our database"""
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
            valid_values = set(df_mapping[origin_col].dropna().astype(str))
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
    """Convert sanitized evidence (orig values) to encoded IDs expected by apt profiles"""
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
            mapping = dict(zip(df_mapping[origin_col].astype(str), df_mapping[encoded_col]))
            
            encoded_values = []
            for val in evidence[evidence_key]:
                if val in mapping:
                    encoded_values.append(mapping[val])
            if encoded_values:
                encoded_evidence[output_key] = encoded_values
    
    return encoded_evidence

def calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids):
    """Calculate weighted similarity scores for all APTs (exactly your GNN pipeline)"""
    feature_weights = {
        'techniques': 1.5,
        'software': 1.3,
        'capec': 1.0,
        'cwe': 1.0,
        'cve': 0.8,
        'pulse': 0.7
    }
    
    similarities = []
    
    for apt_name in apt_ids:
        apt_data = apt_profiles[apt_name]
        total_score = 0
        total_weight = 0
        
        for feature_type, weight in feature_weights.items():
            if feature_type in encoded_evidence and feature_type in apt_data:
                evidence_set = set(encoded_evidence[feature_type])
                profile_set = set(apt_data[feature_type])
                common = len(evidence_set & profile_set)
                
                total_score += common * weight
                total_weight += len(encoded_evidence[feature_type]) * weight
        
        similarity = total_score / total_weight if total_weight > 0 else 0
        similarities.append(similarity)
    
    return torch.tensor(similarities)

def create_similarity_matrix(apt_embeddings, apt_name_dict, apt_profiles, apt_ids):
    """Create similarity DataFrame from embeddings (same as GNN code)"""
    # apt_embeddings: torch.Tensor or np.array (n_apts, dim)
    if torch.is_tensor(apt_embeddings):
        emb_np = apt_embeddings.detach().cpu().numpy()
    else:
        emb_np = np.array(apt_embeddings)
    
    similarity_matrix = cosine_similarity(emb_np)
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    similarity_df = pd.DataFrame(similarity_matrix, index=apt_names, columns=apt_names)
    return similarity_df

def plot_similarity_heatmap(similarity_df):
    plt.figure(figsize=(16, 14))
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)
    sns.heatmap(similarity_df, mask=None, cmap=cmap, center=0.5, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={'shrink':0.8, 'label':'Similarity Score'})
    plt.title("APT Group Similarity Matrix (Siamese embeddings)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

# ---------- Main querying functions ----------
def load_siamese_complete(path="./DataForSiamese/siamese_complete_model.pkl"):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # data contains model_state_dict, model_config, apt_embeddings (torch), apt_profiles, apt_name_dict, apt_ids, apt_to_idx, df_mapping
    model = APTSiameseNetwork(data['model_config']['num_apts'], data['model_config']['embedding_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    # apt_embeddings may be stored as torch tensor or numpy; ensure torch tensor
    apt_embeddings = data.get('apt_embeddings')
    if isinstance(apt_embeddings, np.ndarray):
        apt_embeddings = torch.tensor(apt_embeddings, dtype=torch.float)
    elif torch.is_tensor(apt_embeddings):
        apt_embeddings = apt_embeddings.float()
    else:
        # fallback: try to get from model
        apt_embeddings = model.get_all_embeddings().detach().float()
    
    return model, data, apt_embeddings

def get_similarity_between_apts(model, apt_to_idx, apt_a, apt_b):
    """Return siamese model cosine similarity between two APT names (apt_a, apt_b are keys in apt_to_idx)."""
    idx1 = torch.tensor(apt_to_idx[apt_a], dtype=torch.long)
    idx2 = torch.tensor(apt_to_idx[apt_b], dtype=torch.long)
    with torch.no_grad():
        emb1, emb2 = model(idx1, idx2)
        sim = model.get_similarity(emb1, emb2).item()
    return sim

# ---------- Example script usage ----------
def main():
    pwd = "./DataForSiamese/"
    model_path = os.path.join(pwd, 'siamese_complete_model.pkl')
    print("Loading siamese complete model & metadata...")
    model, data, apt_embeddings = load_siamese_complete(model_path)
    apt_profiles = data['apt_profiles']          # dict keyed by apt name (string) -> profile
    apt_name_dict = data['apt_name_dict']       # mapping encoded->original
    apt_ids = data['apt_ids']                   # list of apt names (order used when saving)
    apt_to_idx = data['apt_to_idx']             # name -> index
    df_mapping = data['df_mapping']             # dataframe used for sanitization/encoding
    
    # 1) Create full similarity DataFrame & show a sample
    print("Creating similarity matrix (from siamese embeddings)...")
    similarity_df = create_similarity_matrix(apt_embeddings, apt_name_dict, apt_profiles, apt_ids)
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # 2) Plot heatmap (same style as GNN)
    print("Plotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)
    
    # 3) Process the same evidence block you used for GNN (define or import here)
    # Example placeholder evidence - replace with your real evidence block or load from file
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
    'CVE_ID': ['CVE-2021-44228', 'ewhoiehe'],  # Kept your example CVE
    'Pulse_ID': ['ksjdgh']
    }

    print("Sanitizing evidence against mapping...")
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
    print("Encoded evidence counts:", {k: len(v) for k, v in encoded_evidence.items()})
    
    # 4) Score by weighted feature overlap (identical to GNN pipeline)
    weighted_similarities = calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids)
    
    # Create results in same display format as GNN
    full_results = []
    for idx in weighted_similarities.sort(descending=True).indices:
        apt_name = apt_ids[idx]
        apt_data = apt_profiles[apt_name]
        common_features = {
            'techniques': len(set(encoded_evidence.get('techniques', [])) & set(apt_data['techniques'])),
            'software': len(set(encoded_evidence.get('software', [])) & set(apt_data['software'])),
            'capec': len(set(encoded_evidence.get('capec', [])) & set(apt_data['capec'])),
            'cwe': len(set(encoded_evidence.get('cwe', [])) & set(apt_data['cwe'])),
            'cve': len(set(encoded_evidence.get('cve', [])) & set(apt_data['cve'])),
            'pulse': len(set(encoded_evidence.get('pulse', [])) & set(apt_data['pulse']))
        }
        full_results.append({
            'Rank': len(full_results)+1,
            'APT': apt_name_dict[apt_data['encoded_id']],
            'Score': weighted_similarities[idx].item(),
            'Encoded_ID': apt_data['encoded_id'],
            **common_features
        })
    
    ranked_apts = pd.DataFrame(full_results)
    print("\nTop ranked results (Siamese embeddings used only for the global matrix; scoring uses weighted features):")
    print("\nRanked results (all APTs):")
    print(ranked_apts.to_string(index=False))
    
    



    # ===== EVALUATE SIMILARITY MATRIX (Top-1, 5, 10, 20) =====
    print("\n" + "="*80)
    print("EVALUATING SIMILARITY MATRIX (Top-1, 5, 10, 20)")
    print("="*80)

    # Get APT names from the similarity matrix
    apt_names_sim = list(similarity_df.index)
    sim_matrix_results = {}

    # For each APT, get its similarity scores from the matrix
    for target_apt in apt_names_sim:
        similarities = similarity_df.loc[target_apt].to_dict()
        sim_matrix_results[target_apt] = similarities

    # Evaluate for multiple top-X values
    top_x_values_sim = [1, 5, 10, 20]

    for topX in top_x_values_sim:
        lengthResultDict = len(sim_matrix_results)
        success = 0
        failure = 0
        
        # Iterate through each APT's results
        for target_apt, similarities in sim_matrix_results.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs (exclude self if needed, but include for fair comparison)
            top_x_similar = dict(list(sorted_similarities.items())[:topX])
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success += 1
            else:
                failure += 1
        
        print(f"\nSimilarity Matrix - TopX={topX}:")
        print(f"Success: {success}/{lengthResultDict} | Failure: {failure}/{lengthResultDict}")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")



    # ===== PROPER PROFILE DATA DIVISION EVALUATION (Siamese similarity matrix) =====
    from itertools import islice
    import random

    divider = 2  # 1 = 100% of profile, 2 = 50% of profile, etc.
    top_x_values = [1, 5, 10, 20]

    apt_names = list(similarity_df.index)
    resultdict = {}

    print(f"\nEvaluating Siamese similarity matrix with divider={divider} (using {100/divider:.1f}% of profile data)")

    # ---------- PROPER: Divide actual profile data, then recalc similarity ----------
    def divide_apt_profile(apt_name, apt_profiles, apt_name_dict, apt_ids, df_mapping, divider=2):
        """Divide an APT's profile data and return the subset evidence"""
        # Find the APT ID from the name
        target_apt_id = None
        for apt_id in apt_ids:
            if apt_name_dict[apt_profiles[apt_id]['encoded_id']] == apt_name:
                target_apt_id = apt_id
                break
        
        if not target_apt_id:
            return {}
        
        target_data = apt_profiles[target_apt_id]
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
                
                # Convert encoded features back to original format
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
        
        return divided_evidence

    # For each APT, divide its profile and calculate similarities
    for target_apt_name in apt_names:
        print(f"Processing {target_apt_name}...")
        
        # Divide the profile
        divided_evidence = divide_apt_profile(target_apt_name, apt_profiles, apt_name_dict, apt_ids, df_mapping, divider)
        
        # Process the divided evidence
        clean_evidence = sanitize_evidence(divided_evidence, df_mapping)
        encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
        
        # Calculate weighted similarities (same as your original approach)
        weighted_similarities = calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids)
        
        # Create ranked results
        ranked_results = {}
        for idx in weighted_similarities.sort(descending=True).indices:
            apt_name = apt_ids[idx]
            apt_real_name = apt_name_dict[apt_profiles[apt_name]['encoded_id']]
            ranked_results[apt_real_name] = weighted_similarities[idx].item()
        
        resultdict[target_apt_name] = ranked_results

    # ---------- Evaluate Top-X ----------
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SuccessFailDict = {}

        for target_apt, similarities in resultdict.items():
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            top_x_similar = dict(islice(sorted_similarities.items(), topX))

            if target_apt in top_x_similar:
                success += 1
                SuccessFailDict[target_apt] = "Pass"
            else:
                failure += 1
                SuccessFailDict[target_apt] = "Fail"

        print(f"\nSettings: TopX={topX}, Divider={divider}")
        print(f"Siamese model success/failure: {success}/{failure}  (Total={lengthResultDict})")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")




if __name__ == "__main__":
    main()
