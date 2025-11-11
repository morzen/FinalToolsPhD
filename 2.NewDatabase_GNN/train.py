import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from collections import defaultdict
import pickle
import numpy as np

class WeightedAPTSimilarityGNN(nn.Module):
    def __init__(self, num_apts, hidden_dim=64):
        super().__init__()
        self.apt_embedding = nn.Embedding(num_apts, hidden_dim)
        
        # Manual feature weights (fixed, not learnable)
        self.feature_weights = {
            'techniques': 1.5,
            'software': 1.3, 
            'capec': 1.0,
            'cwe': 1.0,
            'cve': 0.8,
            'pulse': 0.7
        }
        
        # Feature type mapping
        self.feature_type_map = {
            'techniques': 0,
            'software': 1, 
            'capec': 2,
            'cwe': 3,
            'cve': 4,
            'pulse': 5
        }
        
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, num_apts, feature_type_indices=None):
        num_nodes = x.size(0)
        num_features = num_nodes - num_apts
        
        # Create mask for edges pointing to feature nodes
        edge_mask = edge_index[1] >= num_apts
        feature_edges = edge_index[:, edge_mask]
        
        # Normalize feature indices to 0-based
        feature_edges[1] -= num_apts
        
        # Process feature nodes with manual weights
        feature_x = x[num_apts:].unsqueeze(-1)
        
        if feature_type_indices is not None:
            # Apply manual feature type weights
            weights = torch.ones_like(feature_x)
            for feature_type, weight_value in self.feature_weights.items():
                type_idx = self.feature_type_map[feature_type]
                mask = (feature_type_indices == type_idx).unsqueeze(-1)
                weights[mask] = weight_value
            
            weighted_feature_x = feature_x * weights
        else:
            weighted_feature_x = feature_x
        
        h_features = F.relu(self.conv1(weighted_feature_x, feature_edges))
        h_features = self.conv2(h_features, feature_edges)
        
        # Get APT embeddings
        h_apts = self.apt_embedding(torch.arange(num_apts))
        
        # Aggregate features for each APT
        apt_feature_indices = edge_index[1, :num_apts] - num_apts
        
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

def contrastive_loss(embeddings, apt_profiles, margin=1.0):
    """Contrastive loss that pulls similar APTs together and pushes dissimilar ones apart"""
    losses = []
    
    apt_embeddings = embeddings
    apt_names = list(apt_profiles.keys())
    
    for i, apt1 in enumerate(apt_names):
        for j, apt2 in enumerate(apt_names):
            if i >= j:  # Avoid duplicate pairs
                continue
                
            # Calculate feature overlap similarity
            overlap = 0
            for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']:
                set1 = set(apt_profiles[apt1][feature_type])
                set2 = set(apt_profiles[apt2][feature_type])
                overlap += len(set1 & set2)
            
            # Normalize similarity score
            max_possible = sum([
                len(apt_profiles[apt1][feature_type]) + len(apt_profiles[apt2][feature_type]) 
                for feature_type in ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
            ]) / 2
            similarity = overlap / max_possible if max_possible > 0 else 0
            
            # Contrastive loss
            distance = F.pairwise_distance(apt_embeddings[i].unsqueeze(0), 
                                         apt_embeddings[j].unsqueeze(0))
            
            if similarity > 0.3:  # Similar pairs
                loss = distance.pow(2)
            else:  # Dissimilar pairs
                loss = F.relu(margin - distance).pow(2)
            
            losses.append(loss)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)

def main():
    pwd = "../BaseData/"

    # Load data
    df_encoded = pd.read_csv(pwd + 'EncodedData.csv')
    df_mapping = pd.read_csv(pwd + 'comparison_Encoded_df.csv')
    
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
    feature_type_indices = []  # Track feature types for weighting
    current_idx = len(apt_ids)
    
    # Build edge connections and feature type indices
    edge_index = []
    feature_type_map = {
        'techniques': 0,
        'software': 1, 
        'capec': 2,
        'cwe': 3,
        'cve': 4,
        'pulse': 5
    }
    
    for apt, features in apt_profiles.items():
        for feature_type in feature_types:
            for feature_id in features[feature_type]:
                feature_node = f"{feature_type}_{feature_id}"
                
                if feature_node not in feature_to_idx:
                    feature_to_idx[feature_node] = current_idx
                    feature_type_indices.append(feature_type_map[feature_type])
                    current_idx += 1
                    
                edge_index.append([apt_to_idx[apt], feature_to_idx[feature_node]])
                edge_index.append([feature_to_idx[feature_node], apt_to_idx[apt]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    feature_type_indices = torch.tensor(feature_type_indices, dtype=torch.long)
    
    # Create node features
    apt_x = torch.arange(len(apt_ids), dtype=torch.long)
    
    feature_x = []
    for feature_node in feature_to_idx:
        _, feature_id = feature_node.split('_')
        feature_x.append([int(feature_id)])
        
    feature_x = torch.tensor(feature_x, dtype=torch.float)
    feature_x = (feature_x - feature_x.mean()) / feature_x.std()
    
    x = torch.cat([torch.zeros(len(apt_ids)), feature_x.squeeze()])
    
    # Create and train model with manual weights
    model = WeightedAPTSimilarityGNN(len(apt_ids))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training GNN model with manual feature weights...")
    print("Manual feature weights:")
    for feature_type, weight in model.feature_weights.items():
        print(f"  {feature_type}: {weight}")
    
    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        embeddings = model(x, edge_index, len(apt_ids), feature_type_indices)
        loss = contrastive_loss(embeddings, apt_profiles)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")
    
    # Get final embeddings
    model.eval()
    with torch.no_grad():
        apt_embeddings = model(x, edge_index, len(apt_ids), feature_type_indices)
    
    # Save everything needed for inference
    save_data = {
        'model_state_dict': model.state_dict(),
        'apt_embeddings': apt_embeddings,
        'apt_profiles': apt_profiles,
        'apt_name_dict': apt_name_dict,
        'apt_ids': apt_ids,
        'x': x,
        'edge_index': edge_index,
        'feature_type_indices': feature_type_indices,
        'df_mapping': df_mapping,
        'num_apts': len(apt_ids),
        'feature_to_idx': feature_to_idx,
        'manual_weights': model.feature_weights  # Save the manual weights
    }
    
    with open('./trained_model_data.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print("\nModel trained and saved successfully!")
    print(f"Number of APTs: {len(apt_ids)}")
    print(f"Number of feature nodes: {len(feature_to_idx)}")
    print(f"Total nodes in graph: {x.size(0)}")

if __name__ == "__main__":
    main()