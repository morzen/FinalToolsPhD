import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import islice
import random

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
    """Convert sanitized evidence to encoded IDs"""
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

def create_similarity_matrix(apt_embeddings, apt_name_dict, apt_profiles, apt_ids):
    """Create similarity matrix between all APTs"""
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(apt_embeddings.detach().numpy())
    
    # Get APT names in order
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    
    # Create DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=apt_names,
        columns=apt_names
    )
    
    return similarity_df

def plot_similarity_heatmap(similarity_df):
    """Plot heatmap of similarity matrix with complete square and red-green color scheme"""
    plt.figure(figsize=(16, 14))
    
    # Create custom red-to-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)
    
    # Plot complete heatmap (no mask)
    sns.heatmap(similarity_df, 
                mask=None,  # Remove the triangular mask
                cmap=cmap,
                center=0.5,  # Center the colormap at 0.5
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'Similarity Score'})
    
    plt.title("APT Group Similarity Matrix", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

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
    
    # 1. Create and display similarity matrix for all APTs
    print("Creating similarity matrix for all APTs...")
    similarity_df = create_similarity_matrix(apt_embeddings, apt_name_dict, apt_profiles, apt_ids)
    
    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")
    
    # ===== EVALUATE SIMILARITY MATRIX (FULL DATA) =====
    print("\n" + "="*80)
    print("EVALUATING SIMILARITY MATRIX (FULL DATA)")
    print("="*80)
    
    # Get APT names in order (same as similarity_df index)
    apt_names = list(similarity_df.index)
    
    # Evaluate for multiple top-X values
    top_x_values = [1, 5, 10, 20]
    
    for top_x in top_x_values:
        success = 0
        failure = 0
        
        # For each APT, check if it appears in its own top X most similar APTs
        for i, apt_name in enumerate(apt_names):
            # Get similarity scores for this APT
            similarities = similarity_df.iloc[i]
            
            # Get top X most similar APTs (including self)
            top_similar = similarities.nlargest(top_x)
            
            # Check if the APT itself is in its own top X most similar
            if apt_name in top_similar.index:
                success += 1
            else:
                failure += 1
        
        total = len(apt_names)
        print(f"\nSettings: Top {top_x} (Full Data)")
        print(f"Success/Failure: {success}/{failure} Total: {total}")
        print(f"Success rate: {(success/total)*100:.2f}%")
    
    # Optional: Save the similarity matrix to CSV
    # similarity_df.to_csv("./apt_similarity_matrix.csv", index=True)
    # print(f"Similarity matrix saved to ./apt_similarity_matrix.csv")
    
    # 2. Plot heatmap
    print("Plotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)
    
    # ===== PROPER PROFILE DATA DIVISION EVALUATION =====
    # Configuration
    divider = 2  # Start with complete profile (divider = 1 means 100% of data)
    top_x_values = [1, 5, 10, 20]
    
    resultdict = {}
    apt_names = [apt_name_dict[apt_profiles[apt]['encoded_id']] for apt in apt_ids]
    
    print(f"\nEvaluating with divider: {divider} (using {100/divider:.1f}% of profile data)")
    
    # For each APT, create divided profile and calculate similarities
    for i, target_apt_name in enumerate(apt_names):
        # Get the target APT's original data
        target_apt_id = next(apt for apt in apt_ids if apt_name_dict[apt_profiles[apt]['encoded_id']] == target_apt_name)
        target_data = apt_profiles[target_apt_id]
        
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
                division_size = max(1, len(feature_list) // divider)  # At least 1 item
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
        weighted_similarities = calculate_weighted_similarity(encoded_evidence, apt_profiles, apt_ids)
        
        # Create ranked results (INCLUDE self this time!)
        ranked_results = {}
        for idx in weighted_similarities.sort(descending=True).indices:
            apt_name = apt_ids[idx]
            apt_real_name = apt_name_dict[apt_profiles[apt_name]['encoded_id']]
            ranked_results[apt_real_name] = weighted_similarities[idx].item()
        
        resultdict[target_apt_name] = ranked_results
    
    # Evaluate for multiple top-X values
    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SucessFailDict = {}
        
        # Iterate through each APT's results
        for target_apt, similarities in resultdict.items():
            # Sort by similarity score (descending)
            sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
            
            # Get top X most similar APTs
            top_x_similar = dict(islice(sorted_similarities.items(), topX))
            
            # Check if target APT is in its own top X most similar
            if target_apt in top_x_similar:
                success = success + 1
                SucessFailDict[target_apt] = "Pass"
            else:
                failure = failure + 1
                SucessFailDict[target_apt] = "Fail"
        
        print("\nsettings: Topx: " + str(topX) + " Divider: " + str(divider))
        print("GNN for this set had a success failure rate of ")
        print(str(success) + "/" + str(failure) + " Total: " + str(lengthResultDict))
        print("in percentage success rate: " + str((success/lengthResultDict)*100))

if __name__ == "__main__":
    main()