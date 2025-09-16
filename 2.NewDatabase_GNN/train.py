import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from collections import defaultdict
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
        
        # Verify edge indices
        assert edge_index.max() < num_nodes, f"Edge index {edge_index.max()} exceeds node count {num_nodes}"
        
        # Create mask for edges pointing to feature nodes
        edge_mask = edge_index[1] >= num_apts
        feature_edges = edge_index[:, edge_mask]
        
        # Normalize feature indices to 0-based
        feature_edges[1] -= num_apts
        assert feature_edges[1].max() < num_features, \
            f"Feature index {feature_edges[1].max()} exceeds feature count {num_features}"
        
        # Process feature nodes
        feature_x = x[num_apts:].unsqueeze(-1)
        h_features = F.relu(self.conv1(feature_x, feature_edges))
        h_features = self.conv2(h_features, feature_edges)
        
        # Get APT embeddings
        h_apts = self.apt_embedding(torch.arange(num_apts))
        
        # Aggregate features for each APT
        apt_feature_indices = edge_index[1, :num_apts] - num_apts
        assert apt_feature_indices.max() < num_features, \
            f"Aggregation index {apt_feature_indices.max()} exceeds features {num_features}"
        
        return h_apts + h_features[apt_feature_indices]

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

def main():
    pwd = "./DataForGNN/"
    
    # Load data
    df_encoded = pd.read_csv(pwd + 'data_for_GNN.csv')
    df_mapping = pd.read_csv(pwd + 'comparison_Encodedtest_df.csv')
    
    # Create APT name mapping
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))
    
    # Create APT profiles
    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)
    
    # Convert sets to lists and add counts
    for apt_name in apt_profiles:
        profile = apt_profiles[apt_name]
        for key in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
            profile[key] = list(profile[key])
            profile[f'{key}_count'] = len(profile[key])
    
    # Create graph structure
    apt_ids = list(apt_profiles.keys())
    feature_types = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
    
    apt_to_idx = {apt: i for i, apt in enumerate(apt_ids)}
    feature_to_idx = {}
    current_idx = len(apt_ids)
    
    # Build edge connections
    edge_index = []
    for apt, features in apt_profiles.items():
        for feature_type in feature_types:
            for feature_id in features[feature_type]:
                feature_node = f"{feature_type}_{feature_id}"
                
                if feature_node not in feature_to_idx:
                    feature_to_idx[feature_node] = current_idx
                    current_idx += 1
                    
                edge_index.append([apt_to_idx[apt], feature_to_idx[feature_node]])
                edge_index.append([feature_to_idx[feature_node], apt_to_idx[apt]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create node features
    apt_x = torch.arange(len(apt_ids), dtype=torch.long)
    
    feature_x = []
    for feature_node in feature_to_idx:
        _, feature_id = feature_node.split('_')
        feature_x.append([int(feature_id)])
        
    feature_x = torch.tensor(feature_x, dtype=torch.float)
    feature_x = (feature_x - feature_x.mean()) / feature_x.std()
    
    x = torch.cat([torch.zeros(len(apt_ids)), feature_x.squeeze()])
    
    # Create and train model
    model = APTSimilarityGNN(len(apt_ids))
    apt_embeddings = model(x, edge_index, len(apt_ids))
    
    # Save everything needed for inference
    save_data = {
        'model_state_dict': model.state_dict(),
        'apt_embeddings': apt_embeddings,
        'apt_profiles': apt_profiles,
        'apt_name_dict': apt_name_dict,
        'apt_ids': apt_ids,
        'x': x,
        'edge_index': edge_index,
        'df_mapping': df_mapping,
        'num_apts': len(apt_ids)  # Save this for model initialization
    }
    
    with open(pwd + 'trained_model_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print("Model trained and saved successfully!")
    print(f"Saved: model weights, embeddings, profiles, and mapping data")
    print(f"Number of APTs: {len(apt_ids)}")

if __name__ == "__main__":
    main()