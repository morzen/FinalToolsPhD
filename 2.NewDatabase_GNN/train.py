"""
STANDARDIZED GNN MODEL FOR APT SIMILARITY
=========================================
Author: [Your Name]
Date: [Current Date]
Description: Graph Neural Network for APT similarity detection
"""

# ============================
# 1. IMPORTS
# ============================
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

# ============================
# 2. CONFIGURATION
# ============================
class GNNConfig:
    """Configuration for GNN model"""
#variable shared by all models:
    DATA_PATH = "../BaseData/EncodedData.csv"
    MODEL_PATH = "trained_apt_gnn_model.pkl"
    
#variables share by some models:
    MAPPING_PATH = "../BaseData/comparison_Encoded_df.csv"
    FEATURE_WEIGHTS = {
        "techniques": 80.0,
        "software": 70.0,
        "capec": 60.0,
        "cwe": 50.0,
        "cve": 40.0,
        "pulse": 30.0
    }
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.1
    EPOCHS = 50
    
#variables unique to this model:
    OUT_DIM = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 3. DATA LOADING & PREPROCESSING
# ============================
class GNNDataProcessor:
    """Handles data processing for GNN"""
    
    def __init__(self):
        self.profiles = {}
        self.apt_ids = []
        
    def load_data(self):
        """Load and prepare training data"""
        print("Loading training data...")
        df_encoded = pd.read_csv(GNNConfig.DATA_PATH)
        print(f"Training data: {len(df_encoded)} records")
        return df_encoded
    
    def create_apt_profiles(self, df_encoded):
        """Create APT profiles from encoded data"""
        print("\nCREATING APT PROFILES")
        print("=" * 50)
        
        unique_apts = sorted(df_encoded["APTname"].unique())
        self.apt_ids = unique_apts
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
                self._add_feature(profiles[apt_id], "techniques", row["techniques_str"])
                self._add_feature(profiles[apt_id], "software", row["software_str"])
                self._add_feature(profiles[apt_id], "capec", row["CAPEC_ID"])
                self._add_feature(profiles[apt_id], "cwe", row["CWE_ID"])
                self._add_feature(profiles[apt_id], "cve", row["CVE_ID"])
                self._add_feature(profiles[apt_id], "pulse", row["Pulse_ID"])
        
        self.profiles = profiles
        self._print_statistics()
        return profiles, unique_apts
    
    def _add_feature(self, profile, feature_name, value):
        """Add feature value if valid"""
        if value != -1:
            profile[feature_name].add(value)
    
    def _print_statistics(self):
        """Print profile statistics"""
        print("\nPROFILE STATISTICS")
        print("=" * 50)
        
        stats = {
            'techniques': sum(len(p["techniques"]) for p in self.profiles.values()),
            'software': sum(len(p["software"]) for p in self.profiles.values()),
            'capec': sum(len(p["capec"]) for p in self.profiles.values()),
            'cwe': sum(len(p["cwe"]) for p in self.profiles.values()),
            'cve': sum(len(p["cve"]) for p in self.profiles.values()),
            'pulse': sum(len(p["pulse"]) for p in self.profiles.values()),
        }
        
        print(f"Total APTs: {len(self.profiles)}")
        for feature, count in stats.items():
            print(f"Total {feature}: {count}")
    
    def build_graph(self, profiles, apt_ids):
        """Build heterogeneous graph structure"""
        print("\nBUILDING GRAPH STRUCTURE")
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
        capec_to_idx = {capec: i for i, capec in enumerate(all_capec)}
        cwe_to_idx = {cwe: i for i, cwe in enumerate(all_cwe)}
        cve_to_idx = {cve: i for i, cve in enumerate(all_cve)}
        pulse_to_idx = {pulse: i for i, pulse in enumerate(all_pulse)}

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
        pulse_offset = cve_offset + num_cve

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
        fw = GNNConfig.FEATURE_WEIGHTS

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

        x = torch.ones(total_nodes, dtype=torch.float).view(-1, 1)

        print(f"\n Graph built:")
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
            "feature_weights": GNNConfig.FEATURE_WEIGHTS.copy(),
        }

        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return data, graph_info

# ============================
# 4. MODEL DEFINITION
# ============================
class APTGNNModel(nn.Module):
    """GNN model for APT similarity"""
    
    def __init__(self, in_channels, hidden_dim=GNNConfig.HIDDEN_DIM, out_dim=GNNConfig.OUT_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.normalize(x, p=2, dim=-1)
        return x

# ============================
# 5. TRAINING FUNCTION
# ============================
class GNNTrainer:
    """Handles GNN training"""
    
    def __init__(self, model, device=GNNConfig.DEVICE):
        self.model = model
        self.device = device
    
    def compute_ground_truth_similarity_matrix(self, profiles, apt_ids):
        """Build the TTP-weighted similarity matrix y(i,j) in [0,1] for all APT pairs"""
        fw = GNNConfig.FEATURE_WEIGHTS
        TECHNIQUE_WEIGHT = fw["techniques"]
        SOFTWARE_WEIGHT = fw["software"]
        CAPEC_WEIGHT = fw["capec"]
        CWE_WEIGHT = fw["cwe"]
        CVE_WEIGHT = fw["cve"]
        PULSE_WEIGHT = fw["pulse"]

        num_apts = len(apt_ids)
        gt = torch.zeros((num_apts, num_apts), dtype=torch.float32)

        for i, apt_i in enumerate(apt_ids):
            prof_i = profiles[apt_i]
            T_i, S_i, Ca_i, Cw_i, Cv_i, P_i = prof_i["techniques"], prof_i["software"], prof_i["capec"], prof_i["cwe"], prof_i["cve"], prof_i["pulse"]

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
                cve_overlap = len(Cv_i & Cv_j)
                pulse_overlap = len(P_i & P_j)

                raw = (
                    tech_overlap * TECHNIQUE_WEIGHT +
                    soft_overlap * SOFTWARE_WEIGHT +
                    capec_overlap * CAPEC_WEIGHT +
                    cwe_overlap * CWE_WEIGHT +
                    cve_overlap * CVE_WEIGHT +
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
    
    def train(self, data, profiles, apt_ids, num_apts, epochs=GNNConfig.EPOCHS):
        """Train the GNN model"""
        self.model = self.model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=GNNConfig.LEARNING_RATE)
        
        apt_idx = torch.arange(num_apts, device=self.device)

        print(f"GNN TRAINING FOR {epochs} EPOCHS...")

        gt_cpu = self.compute_ground_truth_similarity_matrix(profiles, apt_ids)
        gt = gt_cpu.to(self.device)

        gt_with_diag = gt.clone()
        gt_with_diag.fill_diagonal_(1.0)
        
        mask = torch.ones_like(gt_with_diag, dtype=bool, device=self.device)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            node_emb = self.model(data.x, data.edge_index, data.edge_weight)
            apt_emb = node_emb[apt_idx]

            sim = torch.matmul(apt_emb, apt_emb.t())
            loss = F.mse_loss(sim[mask], gt_with_diag[mask])

            loss.backward()
            optimizer.step()

            if epoch % 30 == 0:
                with torch.no_grad():
                    diag_mean = torch.diag(sim).mean().item()
                    print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Self-sim: {diag_mean:.4f}")

        self.model.eval()
        with torch.no_grad():
            node_emb = self.model(data.x, data.edge_index, data.edge_weight)
            apt_emb = node_emb[apt_idx].cpu()
            apt_emb = F.normalize(apt_emb, p=2, dim=1)

        print("\nGNN TRAINING COMPLETE!")
        return apt_emb

# ============================
# 6. UTILITY FUNCTIONS
# ============================
class GNNUtils:
    """Utility functions for GNN analysis"""
    
    @staticmethod
    def load_apt_mappings():
        """Load APT name mappings"""
        try:
            df_mapping = pd.read_csv(GNNConfig.MAPPING_PATH)
            apt_name_mapping = {}
            for _, row in df_mapping.iterrows():
                encoded_id = row["APTnameEncoded"]
                original_name = row["APTnameOrigin"]
                if encoded_id not in apt_name_mapping:
                    apt_name_mapping[encoded_id] = original_name
            return apt_name_mapping
        except FileNotFoundError:
            print("Warning: comparison_Encoded_df.csv not found; using encoded IDs as names.")
            return {}
    
    @staticmethod
    def debug_gnn_outputs(apt_embeddings, profiles, apt_ids, apt_name_mapping):
        """Debug GNN outputs and analyze embeddings"""
        print("\nGNN MODEL DEBUG INFO")
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
        
        GNNUtils._analyze_ranking_anomalies(apt_embeddings, profiles, apt_ids, apt_name_mapping)
    
    @staticmethod
    def _analyze_ranking_anomalies(apt_embeddings, profiles, apt_ids, apt_name_mapping):
        """Analyze ranking anomalies in embeddings"""
        print("\nANALYZING RANKING ANOMALIES")
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
                'cve': len(profile['cve']),
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
        print(f"{'APT Name':<30} {'Embed Norm':<10} {'Techniques':<10} {'Software':<8} {'CAPEC':<8} {'CWE':<8} {'CVE':<8} {'Pulse':<8} {'Total':<6}")
        print("-" * 80)
        for item in embedding_norms[:10]:
            print(f"{item['apt_name']:<30} {item['embedding_norm']:<10.4f} "
                  f"{item['techniques']:<10} {item['software']:<8} "
                  f"{item['capec']:<8} {item['cwe']:<8} {item['cve']:<8} "
                  f"{item['pulse']:<8} {item['profile_size']:<6}")

# ============================
# 7. MODEL SAVING
# ============================
def save_gnn_model(model, apt_embeddings, profiles, graph_info, apt_ids, 
                   apt_name_mapping, filename=GNNConfig.MODEL_PATH):
    """Save trained GNN model"""
    save_data = {
        "model_state_dict": model.state_dict(),
        "apt_embeddings": apt_embeddings,
        "profiles": profiles,
        "graph_info": graph_info,
        "apt_ids": apt_ids,
        "apt_name_mapping": apt_name_mapping,
        "feature_weights": GNNConfig.FEATURE_WEIGHTS.copy(),
        "model_config": {
            "in_channels": model.conv1.in_channels,
            "hidden_dim": model.conv1.out_channels,
            "out_dim": model.conv2.out_channels,
        },
    }
    
    with open(filename, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"\nGNN model saved as '{filename}'")
    return save_data

# ============================
# 8. MAIN EXECUTION
# ============================
def main():
    """Execute the GNN training pipeline"""
    print("GNN APT SIMILARITY PIPELINE")
    print("=" * 60)
    
    # Data processing
    processor = GNNDataProcessor()
    df_encoded = processor.load_data()
    profiles, apt_ids = processor.create_apt_profiles(df_encoded)
    
    # Graph building
    data, graph_info = processor.build_graph(profiles, apt_ids)
    
    # Model initialization
    in_channels = data.x.shape[1]
    model = APTGNNModel(in_channels=in_channels)
    
    # Training
    trainer = GNNTrainer(model)
    apt_embeddings = trainer.train(data, profiles, apt_ids, graph_info['num_apts'])
    
    # Load APT name mappings
    apt_name_mapping = GNNUtils.load_apt_mappings()
    if not apt_name_mapping:
        apt_name_mapping = {aid: f"APT{aid}" for aid in apt_ids}
    
    # Debug outputs
    GNNUtils.debug_gnn_outputs(apt_embeddings, profiles, apt_ids, apt_name_mapping)
    
    # Save model
    save_data = save_gnn_model(model, apt_embeddings, profiles, 
                              graph_info, apt_ids, apt_name_mapping)
    
    return save_data

if __name__ == "__main__":
    saved_data = main()