import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


###################################################################
# GLOBAL CONFIG: FEATURE WEIGHTS (TTPs > SW > CVE)
###################################################################

FEATURE_WEIGHTS = {
    "techniques": 100.0,   # MUCH higher emphasis on TTPs
    "software": 3.0,     # medium importance
    "capec": 7.0,
    "cwe": 6.0,
    "cve": 5.0,      # almost negligible
    "pulse": 0.1
}


###################################################################
# DATA PREPROCESSING AND PROFILE BUILDING
###################################################################

def load_training_data(pwd="../BaseData/"):
    """Load only the training data (EncodedData.csv)."""
    print("Loading training data...")
    df_encoded = pd.read_csv(os.path.join(pwd, "EncodedData.csv"))

    print(f"Training data: {len(df_encoded)} records")
    print("\nSample training data:")
    print(df_encoded.head(10))

    return df_encoded


def create_apt_profiles_simple(df_encoded):
    """Create APT profiles directly from encoded data."""
    print("\n📊 CREATING APT PROFILES FROM ENCODED DATA")
    print("=" * 50)

    unique_apts = sorted(df_encoded["APTname"].unique())
    print(f"Found {len(unique_apts)} unique APTs")

    profiles = {}
    for apt_id in unique_apts:
        apt_records = df_encoded[df_encoded["APTname"] == apt_id]

        profiles[apt_id] = {
            "encoded_id": apt_id,
            "techniques": set(),
            "software": set(),
            "capec": set(),
            "cwe": set(),
            "cve": set(),
            "pulse": set(),
        }

        for _, row in apt_records.iterrows():
            if row["techniques_str"] != -1:
                profiles[apt_id]["techniques"].add(row["techniques_str"])
            if row["software_str"] != -1:
                profiles[apt_id]["software"].add(row["software_str"])
            if row["CAPEC_ID"] != -1:
                profiles[apt_id]["capec"].add(row["CAPEC_ID"])
            if row["CWE_ID"] != -1:
                profiles[apt_id]["cwe"].add(row["CWE_ID"])
            if row["CVE_ID"] != -1:
                profiles[apt_id]["cve"].add(row["CVE_ID"])
            if row["Pulse_ID"] != -1:
                profiles[apt_id]["pulse"].add(row["Pulse_ID"])

    print(f"\n📈 PROFILE STATISTICS:")
    total_tech = sum(len(p["techniques"]) for p in profiles.values())
    total_soft = sum(len(p["software"]) for p in profiles.values())
    total_capec = sum(len(p["capec"]) for p in profiles.values())
    total_cwe = sum(len(p["cwe"]) for p in profiles.values())
    total_cve = sum(len(p["cve"]) for p in profiles.values())
    total_pulse = sum(len(p["pulse"]) for p in profiles.values())

    print(f"Total APTs: {len(profiles)}")
    print(f"Total techniques: {total_tech}")
    print(f"Total software: {total_soft}")
    print(f"Total CAPEC: {total_capec}")
    print(f"Total CWEs: {total_cwe}")
    print(f"Total CVEs: {total_cve}")
    print(f"Total Pulse: {total_pulse}")

    print(f"\n🧩 SAMPLE PROFILES (first 3 APTs):")
    for i, (apt_id, profile) in enumerate(list(profiles.items())[:3]):
        print(f"  APT {apt_id}:")
        print(f"    Techniques: {len(profile['techniques'])} - sample: {list(profile['techniques'])[:3]}")
        print(f"    Software: {len(profile['software'])} - sample: {list(profile['software'])[:3]}")
        print(f"    CAPEC: {len(profile['capec'])} - sample: {list(profile['capec'])[:3]}")
        print(f"    CWEs: {len(profile['cwe'])} - sample: {list(profile['cwe'])[:3]}")
        print(f"    CVEs: {len(profile['cve'])} - sample: {list(profile['cve'])[:3]}")
        print(f"    Pulse: {len(profile['pulse'])} - sample: {list(profile['pulse'])[:3]}")

    return profiles, unique_apts


###################################################################
# GRAPH BUILDING (USED BY THE GNN)
###################################################################

def build_graph_structure(profiles, apt_ids):
    """
    Build the heterogeneous graph:
      - APT nodes
      - Technique nodes
      - Software nodes
      - CVE nodes
    Edges are undirected, weighted according to FEATURE_WEIGHTS.
    """
    print("\n🕸️ BUILDING GRAPH STRUCTURE")
    print("=" * 50)

    apt_to_idx = {apt_id: i for i, apt_id in enumerate(apt_ids)}

    all_techniques = set()
    all_software = set()
    all_capec = set()
    all_cwe = set()
    all_cve = set()
    all_pulse = set()

    for profile in profiles.values():
        all_techniques.update(profile["techniques"])
        all_software.update(profile["software"])
        all_capec.update(profile["capec"])
        all_cwe.update(profile["cwe"])
        all_cve.update(profile["cve"])
        all_pulse.update(profile["pulse"])

    technique_to_idx = {tech: i for i, tech in enumerate(all_techniques)}
    software_to_idx = {soft: i for i, soft in enumerate(all_software)}
    capec_to_idx = {soft: i for i, soft in enumerate(all_capec)}
    cwe_to_idx = {soft: i for i, soft in enumerate(all_cwe)}
    cve_to_idx = {cve: i for i, cve in enumerate(all_cve)}
    pulse_to_idx = {soft: i for i, soft in enumerate(all_pulse)}

    num_apts = len(apt_ids)
    num_techniques = len(all_techniques)
    num_software = len(all_software)
    num_capec = len(all_capec)
    num_cwe = len(all_cwe)
    num_cve = len(all_cve)
    num_pulse = len(all_pulse)

    technique_offset = num_apts
    software_offset = technique_offset + num_techniques
    capec_offset = software_offset + num_software
    cwe_offset = capec_offset + num_capec
    cve_offset = cwe_offset + num_cwe
    pulse_offset = cve_offset + num_pulse

    total_nodes = num_apts + num_techniques + num_software + num_capec + num_cwe + num_cve + num_pulse

    print("Graph Structure:")
    print(f"  APT nodes: {num_apts}")
    print(f"  Technique nodes: {num_techniques}")
    print(f"  Software nodes: {num_software}")
    print(f"  CAPEC nodes: {num_capec}")
    print(f"  CWE nodes: {num_cwe}")
    print(f"  CVE nodes: {num_cve}")
    print(f"  Pulse nodes: {num_pulse}")
    print(f"  TOTAL NODES: {total_nodes}")

    edge_list = []
    edge_weights = []
    fw = FEATURE_WEIGHTS

    for apt_id, profile in profiles.items():
        apt_idx = apt_to_idx[apt_id]

        for technique in profile["techniques"]:
            tech_idx = technique_to_idx[technique] + technique_offset
            edge_list.append([apt_idx, tech_idx])
            edge_list.append([tech_idx, apt_idx])
            edge_weights.extend([fw["techniques"], fw["techniques"]])

        for software in profile["software"]:
            soft_idx = software_to_idx[software] + software_offset
            edge_list.append([apt_idx, soft_idx])
            edge_list.append([soft_idx, apt_idx])
            edge_weights.extend([fw["software"], fw["software"]])

        for capec in profile["capec"]:
            capec_idx = capec_to_idx[capec] + capec_offset
            edge_list.append([apt_idx, capec_idx])
            edge_list.append([capec_idx, apt_idx])
            edge_weights.extend([fw["capec"], fw["capec"]])

        for cwe in profile["cwe"]:
            cwe_idx = cwe_to_idx[cwe] + cwe_offset
            edge_list.append([apt_idx, cwe_idx])
            edge_list.append([cwe_idx, apt_idx])
            edge_weights.extend([fw["cwe"], fw["cwe"]])

        for cve in profile["cve"]:
            cve_idx = cve_to_idx[cve] + cve_offset
            edge_list.append([apt_idx, cve_idx])
            edge_list.append([cve_idx, apt_idx])
            edge_weights.extend([fw["cve"], fw["cve"]])

        for pulse in profile["pulse"]:
            pulse_idx = pulse_to_idx[pulse] + pulse_offset
            edge_list.append([apt_idx, pulse_idx])
            edge_list.append([pulse_idx, apt_idx])
            edge_weights.extend([fw["pulse"], fw["pulse"]])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)

    # Simple node features: 1 scalar per node; turned into [N, 1] matrix
    x = torch.ones(total_nodes, dtype=torch.float).view(-1, 1)

    print(f"\n✅ Graph built:")
    print(f"Nodes: {x.shape[0]}")
    print(f"Edges: {edge_index.shape[1]}")

    graph_info = {
        "num_apts": num_apts,
        "apt_to_idx": apt_to_idx,
        "technique_to_idx": technique_to_idx,
        "software_to_idx": software_to_idx,
        "capec_to_idx": capec_to_idx,
        "cwe_to_idx": cwe_to_idx,
        "cve_to_idx": cve_to_idx,
        "pulse_to_idx": pulse_to_idx,
        "technique_offset": technique_offset,
        "software_offset": software_offset,
        "capec_offset": capec_offset,
        "cwe_offset": cwe_offset,
        "cve_offset": cve_offset,
        "pulse_offset": pulse_offset,
        "num_techniques": num_techniques,
        "num_software": num_software,
        "num_capec": num_capec,
        "num_cwe": num_cwe,
        "num_cve": num_cve,
        "num_pulse": num_pulse,
        "feature_weights": FEATURE_WEIGHTS.copy(),
    }

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    return data, graph_info


###################################################################
# PURE GNN MODEL FOR NODE EMBEDDINGS
###################################################################

class APTGNN(nn.Module):
    """
    Simple 2-layer GCN over the heterogeneous APT–TTP–SW–CVE graph.
    APT embeddings are just the node embeddings of the first num_apts nodes.
    """
    def __init__(self, in_channels, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # Normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)
        return x


###################################################################
# TRAINING: TTP-WEIGHTED SIMILARITY SUPERVISION ON APT NODES
###################################################################

def compute_ground_truth_similarity_matrix(profiles, apt_ids):
    """
    Build the TTP-weighted similarity matrix y(i,j) in [0,1] for all APT pairs.
    """
    TECHNIQUE_WEIGHT = FEATURE_WEIGHTS["techniques"]
    SOFTWARE_WEIGHT = FEATURE_WEIGHTS["software"]
    CAPEC_WEIGHT = FEATURE_WEIGHTS["capec"]
    CWE_WEIGHT = FEATURE_WEIGHTS["cwe"]
    CVE_WEIGHT = FEATURE_WEIGHTS["cve"]
    PULSE_WEIGHT = FEATURE_WEIGHTS["pulse"]

    num_apts = len(apt_ids)
    gt = torch.zeros((num_apts, num_apts), dtype=torch.float32)

    for i, apt_i in enumerate(apt_ids):
        prof_i = profiles[apt_i]
        T_i, S_i, Ca_i, Cw_i, Cv_i, P_i = prof_i["techniques"], prof_i["software"], prof_i["capec"], prof_i["cwe"] , prof_i["cve"], prof_i["pulse"]

        max_i = (
            len(T_i) * TECHNIQUE_WEIGHT +
            len(S_i) * SOFTWARE_WEIGHT +
            len(Ca_i) * CAPEC_WEIGHT +
            len(Cw_i) * CWE_WEIGHT +
            len(Cv_i) * CVE_WEIGHT +
            len(P_i) * PULSE_WEIGHT 
        )

        for j, apt_j in enumerate(apt_ids):
            if i == j:
                continue

            prof_j = profiles[apt_j]
            T_j, S_j, Ca_j, Cw_j, Cv_j, P_j = prof_j["techniques"], prof_j["software"], prof_j["capec"], prof_j["cwe"], prof_j["cve"], prof_j["pulse"]

            tech_overlap = len(T_i & T_j)
            soft_overlap = len(S_i & S_j)
            capec_overlap = len(Ca_i & Ca_j)
            cwe_overlap = len(Cw_i & Cw_j)
            cve_overlap  = len(Cv_i & Cv_j)
            pulse_overlap = len(P_i & P_j)

            raw = (
                tech_overlap * TECHNIQUE_WEIGHT +
                soft_overlap * SOFTWARE_WEIGHT +
                capec_overlap * CAPEC_WEIGHT +
                cwe_overlap * CWE_WEIGHT +
                cve_overlap  * CVE_WEIGHT +
                pulse_overlap * PULSE_WEIGHT 
            )

            max_j = (
            len(T_j) * TECHNIQUE_WEIGHT +
            len(S_j) * SOFTWARE_WEIGHT +
            len(Ca_j) * CAPEC_WEIGHT +
            len(Cw_j) * CWE_WEIGHT +
            len(Cv_j) * CVE_WEIGHT +
            len(P_j) * PULSE_WEIGHT 
        )

            max_possible = min(max_i, max_j)
            if max_possible > 0:
                gt[i, j] = raw / max_possible
            else:
                gt[i, j] = 0.0

    return gt


def train_gnn(
    model,
    data,
    profiles,
    apt_ids,
    num_apts,
    epochs=200,
    device="cpu",
):
    """
    Train GNN so that cosine similarity between APT node embeddings
    approximates a TTP-weighted similarity derived from feature overlaps.
    """
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    apt_idx = torch.arange(num_apts, device=device)  # assume APT nodes = [0..num_apts-1]

    print(f"🚀 GNN TRAINING FOR {epochs} EPOCHS...")
    print(f"   Weights: TTP×{FEATURE_WEIGHTS['techniques']}, "
          f"SW×{FEATURE_WEIGHTS['software']}, CAPEC×{FEATURE_WEIGHTS['capec']}"
          f"CWE×{FEATURE_WEIGHTS['cwe']}, CVE×{FEATURE_WEIGHTS['cve']}"
          f"Pu×{FEATURE_WEIGHTS['pulse']}")

    gt_cpu = compute_ground_truth_similarity_matrix(profiles, apt_ids)
    gt = gt_cpu.to(device)

    mask = ~torch.eye(num_apts, dtype=bool, device=device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        node_emb = model(data.x, data.edge_index, data.edge_weight)  # [N, d]
        apt_emb = node_emb[apt_idx]  # [num_apts, d]

        sim = torch.matmul(apt_emb, apt_emb.t())  # cosine similarities

        loss = F.mse_loss(sim[mask], gt[mask])

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        node_emb = model(data.x, data.edge_index, data.edge_weight)
        apt_emb = node_emb[apt_idx].cpu()

    return apt_emb


###################################################################
# DEBUGGING / ANALYSIS (OPTIONAL)
###################################################################

def analyze_ranking_anomalies(profiles, apt_embeddings, apt_ids, apt_name_mapping):
    print("\n🔍 ANALYZING RANKING ANOMALIES")
    print("=" * 50)

    embedding_norms = []
    for i, apt_id in enumerate(apt_ids):
        embedding = apt_embeddings[i]
        profile = profiles[apt_id]
        apt_name = apt_name_mapping.get(apt_id, f"APT{apt_id}")

        embedding_norms.append({
            'apt_name': apt_name,
            'apt_id': apt_id,
            'embedding_norm': embedding.norm().item(),
            'techniques': len(profile['techniques']),
            'software': len(profile['software']),
            'capec': len(profile['capec']),
            'cwe': len(profile['cwe']),
            'cves': len(profile['cve']),
            'pulse': len(profile['pulse']),
            'profile_size': len(profile['techniques']) +
                            len(profile['software']) +
                            len(profile['capec']) +
                            len(profile['cwe']) +
                            len(profile['cve']) +
                            len(profile['pulse'])
        })

    embedding_norms.sort(key=lambda x: x['embedding_norm'], reverse=True)

    print("Top 10 APTs by Embedding Norm:")
    print(f"{'APT Name':<30} {'Embed Norm':<10} {'Techniques':<10} {'Software':<8} {'CAPEC':<8} {'CWEs':<8} {'CVEs':<8} {'Pulse':<8} {'Total':<6}")
    print("-" * 80)
    for item in embedding_norms[:10]:
        print(f"{item['apt_name']:<30} {item['embedding_norm']:<10.4f} "
              f"{item['techniques']:<10} {item['software']:<8} "
              f"{item['capec']:<10} {item['cwe']:<8} "
              f"{item['cves']:<8} {item['pulse']:<8} {item['profile_size']:<6}")


def debug_gnn_outputs(apt_embeddings, profiles, apt_ids, apt_name_mapping):
    print("\n🐛 GNN MODEL DEBUG INFO")
    print("=" * 50)

    print(f"APT embeddings shape: {apt_embeddings.shape}")
    print(f"Embedding stats - Mean: {apt_embeddings.mean():.4f}, Std: {apt_embeddings.std():.4f}")
    print(f"Min: {apt_embeddings.min():.4f}, Max: {apt_embeddings.max():.4f}")

    similarity = torch.matmul(apt_embeddings, apt_embeddings.t())
    print("\nCosine Similarity matrix stats:")
    print(f"Self-similarity mean: {torch.diag(similarity).mean():.4f}")
    mask = ~torch.eye(len(apt_ids), dtype=bool)
    print(f"Cross-similarity mean: {similarity[mask].mean():.4f}")
    print(f"Similarity range: {similarity.min():.4f} to {similarity.max():.4f}")

    analyze_ranking_anomalies(profiles, apt_embeddings, apt_ids, apt_name_mapping)


###################################################################
# SAVE TRAINED GNN MODEL
###################################################################

def save_trained_gnn_model(model, apt_embeddings, profiles, graph_info, apt_ids,
                           filename="trained_apt_gnn_model.pkl"):
    print("\n💾 SAVING TRAINED GNN MODEL")
    print("=" * 50)

    save_data = {
        "model_state_dict": model.state_dict(),
        "apt_embeddings": apt_embeddings,
        "profiles": profiles,
        "graph_info": graph_info,
        "apt_ids": apt_ids,
        "feature_weights": FEATURE_WEIGHTS.copy(),
        "model_config": {
            "in_channels": model.conv1.in_channels,
            "hidden_dim": model.conv1.out_channels,
            "out_dim": model.conv2.out_channels,
        },
    }

    with open(filename, "wb") as f:
        pickle.dump(save_data, f)

    print(f"✅ GNN model saved as '{filename}'")
    print(f"Saved: {len(apt_ids)} APT embeddings, {len(profiles)} profiles")


###################################################################
# MAIN
###################################################################

def main():
    print("PURE GNN APT ATTRIBUTION SYSTEM")
    print("=" * 60)

    # 1. Load data and build profiles
    df_encoded = load_training_data()
    profiles, apt_ids = create_apt_profiles_simple(df_encoded)

    # 2. Build heterogeneous graph
    data, graph_info = build_graph_structure(profiles, apt_ids)

    print("\n✅ READY FOR GNN MODEL CREATION!")
    print(f"We have {graph_info['num_apts']} APT nodes for the model")
    print(f"Graph structure: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")

    num_nodes = data.x.shape[0]
    in_channels = data.x.shape[1]  # 1
    num_apts = graph_info["num_apts"]

    # 3. Define GNN model
    model = APTGNN(in_channels=in_channels, hidden_dim=128, out_dim=128)

    # 4. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 5. Train GNN with TTP-weighted similarity supervision
    apt_embeddings = train_gnn(
        model,
        data,
        profiles,
        apt_ids,
        num_apts=num_apts,
        epochs=200,
        device=device,
    )
    print("\n✅ GNN TRAINING COMPLETE!")

    # 6. Load APT name mappings (if file exists)
    try:
        df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
        apt_name_mapping = {}
        for _, row in df_mapping.iterrows():
            encoded_id = row["APTnameEncoded"]
            original_name = row["APTnameOrigin"]
            if encoded_id not in apt_name_mapping:
                apt_name_mapping[encoded_id] = original_name
    except FileNotFoundError:
        print("Warning: comparison_Encoded_df.csv not found; using encoded IDs as names.")
        apt_name_mapping = {aid: f"APT{aid}" for aid in apt_ids}

    # 7. Debug / analyze embeddings
    debug_gnn_outputs(apt_embeddings, profiles, apt_ids, apt_name_mapping)

    # 8. Save trained GNN + embeddings
    save_trained_gnn_model(model, apt_embeddings, profiles, graph_info, apt_ids)


if __name__ == "__main__":
    main()

