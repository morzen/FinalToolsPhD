import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class APTGNN(nn.Module):
    """
    Simple 2-layer GCN over the heterogeneous APT-TTP-SW-CAPEC-CWE-CVE-Pulse graph.
    APT embeddings are just the node embeddings of the first num_apts nodes.
    """
    def __init__(self, in_channels, hidden_dim=256, out_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.normalize(x, p=2, dim=-1)
        return x


def load_trained_model():
    print(" LOADING TRAINED MODEL...")
    with open('3feat_trained_apt_gnn_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data


def convert_evidence_to_encoded(evidence, mapping_file="../BaseData/comparison_Encoded_df.csv"):
    df_mapping = pd.read_csv(mapping_file)

    technique_mapping = {}
    software_mapping = {}
    cve_mapping = {}

    for _, row in df_mapping.iterrows():
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            technique_mapping[row['techniques_Origin']] = int(row['techniques_Encoded'])
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            software_mapping[row['software_Origin']] = int(row['software_Encoded'])
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            cve_mapping[row['CVE_ID_Origin']] = int(row['CVE_ID_Encoded'])

    encoded_evidence = {
        'techniques': [technique_mapping.get(t) for t in evidence.get('techniques', []) if t in technique_mapping],
        'software': [software_mapping.get(s) for s in evidence.get('software', []) if s in software_mapping],
        'cve': [cve_mapping.get(c) for c in evidence.get('cve', []) if c in cve_mapping],
    }

    return encoded_evidence


def query_apt_similarity(evidence, saved_data):
    """
    Query APT similarity using the trained GNN with proper graph structure
    """
    # Load the saved graph structure
    if "edge_index" not in saved_data:
        raise ValueError("Graph structure not saved in model file. Please retrain with updated save function.")
    
    graph_info = saved_data['graph_info']
    edge_index = saved_data['edge_index']
    edge_weight = saved_data['edge_weight']
    x = saved_data['x']
    
    technique_to_idx = graph_info['technique_to_idx']
    software_to_idx = graph_info['software_to_idx']
    cve_to_idx = graph_info['cve_to_idx']

    num_apts = graph_info['num_apts']

    technique_offset = graph_info['technique_offset']
    software_offset = graph_info['software_offset']
    cve_offset = graph_info['cve_offset']
    
    feature_weights = graph_info['feature_weights']

    techniques = evidence.get('techniques', [])
    software = evidence.get('software', [])
    cve = evidence.get('cve', [])

    # Get valid indices (filter out None or 0)
    tech_indices = [technique_to_idx.get(t) for t in techniques if t in technique_to_idx]
    soft_indices = [software_to_idx.get(s) for s in software if s in software_to_idx]
    cve_indices = [cve_to_idx.get(c) for c in cve if c in cve_to_idx]

    # Add one new node for the query
    total_nodes = x.shape[0] + 1
    query_node_idx = x.shape[0]  # Index of the new node
    
    # Create new node features (copy existing + add one more)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_x = torch.ones(total_nodes, x.shape[1], device=device)
    new_x[:x.shape[0]] = x.to(device)
    
    # Copy existing edges
    new_edge_index = edge_index.clone().to(device)
    new_edge_weight = edge_weight.clone().to(device) if edge_weight is not None else None

    # Add edges from query node to matched features
    new_edges = []
    new_weights = []

    # Helper function to add edges
    def add_feature_edges(feature_indices, offset, weight):
        for feat_idx in feature_indices:
            if feat_idx is not None:
                feat_node = feat_idx + offset
                new_edges.append([query_node_idx, feat_node])
                new_edges.append([feat_node, query_node_idx])
                new_weights.extend([weight, weight])

    # Add edges for each feature type
    add_feature_edges(tech_indices, technique_offset, feature_weights["techniques"])
    add_feature_edges(soft_indices, software_offset, feature_weights["software"])
    add_feature_edges(cve_indices, cve_offset, feature_weights["cve"])

    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(device)
        new_weights_tensor = torch.tensor(new_weights, dtype=torch.float).to(device)
        
        # Append new edges to existing edges
        new_edge_index = torch.cat([new_edge_index, new_edges_tensor], dim=1)
        if new_edge_weight is not None:
            new_edge_weight = torch.cat([new_edge_weight, new_weights_tensor])
        else:
            new_edge_weight = new_weights_tensor

    # Load model
    model_config = saved_data['model_config']
    model = APTGNN(model_config['in_channels'], model_config['hidden_dim'], model_config['out_dim'])
    model.load_state_dict(saved_data['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        all_embeddings = model(new_x, new_edge_index, new_edge_weight)
        query_embedding = all_embeddings[query_node_idx:query_node_idx+1]
        apt_embeddings = all_embeddings[:num_apts].cpu()
        
        # Calculate cosine similarity
        similarities = F.cosine_similarity(query_embedding.cpu(), apt_embeddings)

    apt_ids = saved_data['apt_ids']
    
    return similarities, apt_ids


def create_similarity_matrix(saved_data):
    print("Creating similarity matrix using trained GNN model...")

    apt_embeddings = saved_data['apt_embeddings']
    apt_ids = saved_data['apt_ids']

    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}

    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]

    similarity_matrix = cosine_similarity(apt_embeddings.detach().numpy())

    similarity_df = pd.DataFrame(similarity_matrix, index=apt_names, columns=apt_names)

    return similarity_df, apt_names


def plot_similarity_heatmap(similarity_df):
    plt.figure(figsize=(16, 14))
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["red", "yellow", "green"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=n_bins)

    sns.heatmap(similarity_df,
                mask=None,
                cmap=cmap,
                center=0.5,
                square=True,
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': 'Similarity Score'})

    plt.title("APT Group Similarity Matrix (GNN Embeddings)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('apt_GNN_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_similarity_matrix(similarity_df, apt_names):
    print("\n" + "=" * 80)
    print("EVALUATING SIMILARITY MATRIX (FULL DATA)")
    print("=" * 80)

    top_x_values = [1, 5, 10, 20]

    for top_x in top_x_values:
        success = 0
        failure = 0
        for i, apt_name in enumerate(apt_names):
            similarities = similarity_df.iloc[i]
            top_similar = similarities.nlargest(top_x)
            if apt_name in top_similar.index:
                success += 1
            else:
                failure += 1
        total = len(apt_names)
        print(f"\nSettings: Top {top_x} (Full Data)")
        print(f"Success/Failure: {success}/{failure} Total: {total}")
        print(f"Success rate: {(success/total)*100:.2f}%")


def evaluate_profile_division(saved_data, divider=2, top_x_values=[1, 5, 10, 20]):
    import random
    from itertools import islice

    profiles = saved_data['profiles']
    apt_ids = saved_data['apt_ids']

    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}

    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]

    print(f"\nEvaluating with divider: {divider} (using {100/divider:.1f}% of profile data)")

    resultdict = {}

    for i, target_apt_name in enumerate(apt_names):
        target_apt_id = apt_ids[i]
        target_profile = profiles[target_apt_id]

        divided_evidence = {'techniques': [], 'software': [], 'cve': []}

        if target_profile['techniques']:
            techniques_list = list(target_profile['techniques'])
            random.shuffle(techniques_list)
            division_size = max(1, len(techniques_list) // divider)
            divided_evidence['techniques'] = techniques_list[:division_size]

        if target_profile['software']:
            software_list = list(target_profile['software'])
            random.shuffle(software_list)
            division_size = max(1, len(software_list) // divider)
            divided_evidence['software'] = software_list[:division_size]

        if target_profile['cve']:
            cve_list = list(target_profile['cve'])
            random.shuffle(cve_list)
            division_size = max(1, len(cve_list) // divider)
            divided_evidence['cve'] = cve_list[:division_size]

        similarities, query_apt_ids = query_apt_similarity(divided_evidence, saved_data)

        ranked_results = {}
        sorted_indices = torch.argsort(similarities, descending=True)

        for idx in sorted_indices:
            apt_id = query_apt_ids[idx.item()]
            apt_real_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
            ranked_results[apt_real_name] = similarities[idx].item()

        resultdict[target_apt_name] = ranked_results

    for topX in top_x_values:
        lengthResultDict = len(resultdict)
        success = 0
        failure = 0
        SucessFailDict = {}

        for target_apt, similarities in resultdict.items():
            top_x_similar = dict(islice(similarities.items(), topX))
            if target_apt in top_x_similar:
                success += 1
                SucessFailDict[target_apt] = "Pass"
            else:
                failure += 1
                SucessFailDict[target_apt] = "Fail"

        print(f"\nSettings: Top {topX}, Divider: {divider}")
        print(f"Success/Failure: {success}/{failure} Total: {lengthResultDict}")
        print(f"Success rate: {(success/lengthResultDict)*100:.2f}%")


def test_sample_queries(saved_data):
    print("\n" + "=" * 80)
    print("TESTING SAMPLE QUERIES")
    print("=" * 80)

    test_evidence =  {
        'techniques': ['faketest', 'T1001.002', 'T1003.002'],
        'software': ['faketest','S0002', 'S0029', 'S0037'],
        'cve': ['faketest','CVE-2012-0433', 'CVE-2013-10007']
    }

    print("Testing with evidence:")
    print(f"  Techniques: {test_evidence['techniques']}")
    print(f"  Software: {test_evidence['software']}")
    print(f"  cve: {test_evidence['cve']}")

    encoded_evidence = convert_evidence_to_encoded(test_evidence)
    similarities, apt_ids = query_apt_similarity(encoded_evidence, saved_data)

    top5_indices = torch.topk(similarities, 5).indices

    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}

    print("\nTop 5 most similar APTs:")
    for i, idx in enumerate(top5_indices):
        apt_id = apt_ids[idx.item()]
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")
        similarity = similarities[idx].item()
        print(f"  {i + 1}. {apt_name}: {similarity:.4f}")

def evaluate_full_profile_queries(saved_data, top_x_values=[1, 5, 10, 20]):
    """
    For each APT, use its full profile as a query and check if the target APT appears
    in the top X most similar APTs.
    """
    print("\n" + "=" * 80)
    print("EVALUATING FULL PROFILE QUERIES (each APT queried with its own complete profile)")
    print("=" * 80)

    profiles = saved_data['profiles']
    apt_ids = saved_data['apt_ids']

    # Map encoded APT IDs to readable names
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {row['APTnameEncoded']: row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]

    # Store results for each top X
    for topX in top_x_values:
        success = 0
        failure = 0
        for i, apt_id in enumerate(apt_ids):
            target_name = apt_names[i]
            full_profile = profiles[apt_id]   # dict with keys: techniques, software, cve

            # Query with the full profile (original string IDs)
            similarities, result_apt_ids = query_apt_similarity(full_profile, saved_data)

            # Get indices of top X similarities
            top_indices = torch.topk(similarities, topX).indices.tolist()
            top_apt_names = [apt_name_mapping.get(apt_ids[idx], f"APT{apt_ids[idx]}") for idx in top_indices]

            if target_name in top_apt_names:
                success += 1
            else:
                failure += 1

        total = len(apt_ids)
        print(f"\nSettings: Top {topX}")
        print(f"Success/Failure: {success}/{failure} Total: {total}")
        print(f"Success rate: {(success/total)*100:.2f}%")

def main():
    print(" GNN APT SIMILARITY MODEL EVALUATION")
    print("=" * 60)

    saved_data = load_trained_model()

    print(f" Model loaded successfully!")
    print(f"   APTs: {len(saved_data['apt_ids'])}")
    print(f"   Embedding dimension: {saved_data['apt_embeddings'].shape[1]}")
    print(f"   Profiles: {len(saved_data['profiles'])}")

    test_sample_queries(saved_data)

    print("\n" + "=" * 80)
    print("CREATING SIMILARITY MATRIX")
    print("=" * 80)

    similarity_df, apt_names = create_similarity_matrix(saved_data)

    print("Similarity DataFrame sample (first 5x5):")
    print(similarity_df.iloc[:5, :5])
    print(f"\nMatrix shape: {similarity_df.shape}")


    evaluate_similarity_matrix(similarity_df, apt_names)

    print("\nEVALUATING full PROFILE data ")
    # NEW: Full profile query evaluation
    evaluate_full_profile_queries(saved_data, top_x_values=[1, 5, 10, 20])

    print("\n" + "=" * 80)
    print("EVALUATING PROFILE DIVISION PERFORMANCE")
    print("=" * 80)

    evaluate_profile_division(saved_data, divider=2, top_x_values=[1, 5, 10, 20])

    print("\nPlotting similarity heatmap...")
    plot_similarity_heatmap(similarity_df)

    

if __name__ == "__main__":
    main()


