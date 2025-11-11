import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle

class APTSimilarityGNN(nn.Module):
    def __init__(self, num_apts, hidden_dim=64):
        super().__init__()
        self.apt_embedding = nn.Embedding(num_apts, hidden_dim)
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, num_apts):
        num_nodes = x.size(0)
        num_features = num_nodes - num_apts
        
        # Create mask for edges pointing to feature nodes
        edge_mask = edge_index[1] >= num_apts
        feature_edges = edge_index[:, edge_mask]
        
        # Normalize feature indices to 0-based
        feature_edges[1] -= num_apts
        
        # Process feature nodes
        feature_x = x[num_apts:].unsqueeze(-1)
        h_features = F.relu(self.conv1(feature_x, feature_edges))
        h_features = self.conv2(h_features, feature_edges)
        
        # Get APT embeddings
        h_apts = self.apt_embedding(torch.arange(num_apts))
        
        # Aggregate features for each APT
        apt_feature_indices = edge_index[1, :num_apts] - num_apts
        
        return h_apts + h_features[apt_feature_indices]

def get_weighted_gnn_similarity(query_features, apt_embeddings, apt_profiles, apt_ids, feature_to_idx):
    """GNN similarity with manual weights for different feature types"""
    
    # Manual weights - techniques and software are more important
    feature_weights = {
        'techniques': 1.5,
        'software': 1.3, 
        'capec': 1.0,
        'cwe': 1.0,
        'cve': 0.8,
        'pulse': 0.7
    }
    
    with torch.no_grad():
        query_embedding = torch.zeros_like(apt_embeddings[0])
        total_weight = 0
        
        for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            weight = feature_weights[feature_type]
            
            for feature_id in query_features.get(feature_type, []):
                feature_node = f"{feature_type}_{feature_id}"
                if feature_node in feature_to_idx:
                    # Find all APTs that have this feature
                    apts_with_feature = []
                    for apt_idx, apt_name in enumerate(apt_ids):
                        apt_data = apt_profiles[apt_name]
                        if feature_id in apt_data[feature_type]:
                            apts_with_feature.append(apt_embeddings[apt_idx])
                    
                    if apts_with_feature:
                        # Average embeddings of APTs that share this feature, weighted by feature importance
                        feature_embedding = torch.stack(apts_with_feature).mean(dim=0)
                        query_embedding += feature_embedding * weight
                        total_weight += weight
        
        if total_weight > 0:
            query_embedding = query_embedding / total_weight
        else:
            query_embedding = apt_embeddings.mean(dim=0)
        
        # Calculate cosine similarity between weighted query and all APT embeddings
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0), 
            apt_embeddings,
            dim=-1
        )
    
    return similarities

def main():
    # Load trained model and data
    with open('./trained_model_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    model = APTSimilarityGNN(loaded_data['num_apts'])
    model.load_state_dict(loaded_data['model_state_dict'])
    model.eval()
    
    apt_embeddings = loaded_data['apt_embeddings']
    apt_profiles = loaded_data['apt_profiles']
    apt_name_dict = loaded_data['apt_name_dict']
    apt_ids = loaded_data['apt_ids']
    df_mapping = loaded_data['df_mapping']
    num_apts = loaded_data['num_apts']
    
    # Recreate feature_to_idx mapping
    feature_to_idx = {}
    current_idx = len(apt_ids)
    for apt, features in apt_profiles.items():
        for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            for feature_id in features[feature_type]:
                feature_node = f"{feature_type}_{feature_id}"
                if feature_node not in feature_to_idx:
                    feature_to_idx[feature_node] = current_idx
                    current_idx += 1

    print(f"Loaded GNN model with {num_apts} APTs")

    # Your evidence
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

    # Convert evidence to encoded format
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

    print("Calculating weighted GNN similarities...")
    print("Feature weights: Techniques=1.5, Software=1.3, CAPEC=1.0, CWE=1.0, CVE=0.8, Pulse=0.7")
    
    similarities = get_weighted_gnn_similarity(
        encoded_evidence, apt_embeddings, apt_profiles, apt_ids, feature_to_idx
    )

    # Create results with feature overlap counts for display
    results = []
    for idx in similarities.sort(descending=True).indices:
        apt_name = apt_ids[idx]
        apt_data = apt_profiles[apt_name]
        
        # Calculate feature overlaps for display
        common_features = {}
        for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            evidence_set = set(encoded_evidence.get(feature_type, []))
            profile_set = set(apt_data[feature_type])
            common_features[feature_type] = len(evidence_set & profile_set)
        
        results.append({
            'Rank': len(results)+1,
            'APT': apt_name_dict[apt_data['encoded_id']],
            'GNN_Similarity': similarities[idx].item(),
            'Encoded_ID': apt_data['encoded_id'],
            **common_features
        })

    # Display ALL APTs
    print(f"\n{'Rank':<5} | {'APT':<30} | {'Similarity':<10} | {'Tech':<5} | {'Soft':<5} | {'CAPEC':<6} | {'CWE':<4} | {'CVE':<4} | {'Pulse':<5}")
    print("-" * 95)
    for row in results:  # Show ALL APTs
        apt_name = (row['APT'][:27] + '..') if len(row['APT']) > 29 else row['APT']
        print(f"{row['Rank']:<5} | {apt_name:<30} | {row['GNN_Similarity']:.4f} | {row['techniques']:<5} | {row['software']:<5} | {row['capec']:<6} | {row['cwe']:<4} | {row['cve']:<4} | {row['pulse']:<5}")

    # Show summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total APTs analyzed: {len(results)}")
    print(f"Average similarity: {similarities.mean().item():.4f}")
    print(f"Maximum similarity: {similarities.max().item():.4f}")
    print(f"Minimum similarity: {similarities.min().item():.4f}")
    print(f"APTs with similarity > 0: {(similarities > 0).sum().item()}")

if __name__ == "__main__":
    main()