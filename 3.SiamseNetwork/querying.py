# siamese_query.py
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

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
        emb1 = emb1.view(1, -1)
        emb2 = emb2.view(1, -1)
        return F.cosine_similarity(emb1, emb2).squeeze()
    
    def get_all_embeddings(self):
        return self.apt_embedding.weight

# ---------- Helper functions ----------
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
    """Calculate weighted similarity scores for all APTs"""
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

# ---------- Main querying functions ----------
def load_siamese_complete(path="./siamese_complete_model.pkl"):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    model = APTSiameseNetwork(data['model_config']['num_apts'], data['model_config']['embedding_dim'])
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    
    apt_embeddings = data.get('apt_embeddings')
    if isinstance(apt_embeddings, np.ndarray):
        apt_embeddings = torch.tensor(apt_embeddings, dtype=torch.float)
    elif torch.is_tensor(apt_embeddings):
        apt_embeddings = apt_embeddings.float()
    else:
        apt_embeddings = model.get_all_embeddings().detach().float()
    
    return model, data, apt_embeddings

# ---------- Custom query function ----------
def query_apt_similarity(evidence, model_path="./siamese_complete_model.pkl", top_k=None):
    """
    Query the Siamese model with custom evidence and get similar APTs
    
    Parameters:
    - evidence: dict with keys like 'techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID'
    - model_path: path to the saved siamese model
    - top_k: number of top similar APTs to return (None returns all APTs)
    
    Returns:
    - ranked_results: DataFrame with similar APTs and their scores
    """
    
    # Load model and data
    print("Loading siamese complete model & metadata...")
    model, data, _ = load_siamese_complete(model_path)
    apt_profiles = data['apt_profiles']
    apt_name_dict = data['apt_name_dict']
    apt_ids = data['apt_ids']
    df_mapping = data['df_mapping']
    
    # Process the evidence
    print("Sanitizing evidence against mapping...")
    clean_evidence = sanitize_evidence(evidence, df_mapping)
    encoded_evidence = translate_to_encoded(clean_evidence, df_mapping)
    print("Encoded evidence counts:", {k: len(v) for k, v in encoded_evidence.items()})
    
    # Calculate weighted similarity scores
    weighted_similarities = calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids)
    
    # Create ranked results
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
    
    # Return all results if top_k is None, otherwise return top_k
    if top_k is None:
        ranked_apts = pd.DataFrame(full_results)
        print(f"\nReturning all {len(full_results)} APTs")
    else:
        ranked_apts = pd.DataFrame(full_results).head(top_k)
        print(f"\nReturning top {top_k} APTs out of {len(full_results)} total")
    
    return ranked_apts



# ---------- Example usage ----------
def main():
   
    
    # Example evidence - replace with your actual evidence
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
    'CVE_ID': ['CVE-2021-44228'],
    'Pulse_ID': ['']
    }
    
    print("\n" + "=" * 80)
    print("QUERYING WITH CUSTOM EVIDENCE")
    print("=" * 80)
    
    # Query the model - get ALL APTs (no top_k limit)
    ranked_results = query_apt_similarity(
        evidence=evidence,
        model_path="./siamese_complete_model.pkl",
        top_k=None  # This will return all APTs
    )
    
    # Display results
    print("\nAll APTs ranked by similarity score:")
    print(ranked_results.to_string(index=False))


if __name__ == "__main__":
    main()