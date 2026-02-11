import time
import psutil
import GPUtil

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        print("\n" + "="*50)
        print("RESOURCE USAGE REPORT")
        print("="*50)
        
        # Time
        print(f"Execution Time: {end_time - self.start_time:.2f} seconds")
        
        # Memory
        print(f"Memory Usage: {end_memory - self.start_memory:.2f} MB")
        print(f"Current Memory: {end_memory:.2f} MB")
        
        # CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # System-wide CPU
        sys_cpu = psutil.cpu_percent(interval=0.1)
        print(f"System CPU: {sys_cpu}%")
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"\nGPU {gpu.id} ({gpu.name}):")
                print(f"  Utilization: {gpu.load*100:.1f}%")
                print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        except:
            print("\nGPU: Not available or not detected")

# Usage
monitor = ResourceMonitor()
monitor.start()





import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import islice
from matplotlib.colors import LinearSegmentedColormap

# ---------- FeatureEncoder ----------
class FeatureEncoder(nn.Module):
    def __init__(self, num_techniques, num_software, num_capec, num_cwe, num_cve, num_pulse, embed_dim=128):
        super().__init__()
        embed_per_feature = embed_dim // 6
        self.tech_embed = nn.Embedding(num_techniques+1, embed_per_feature)
        self.soft_embed = nn.Embedding(num_software+1, embed_per_feature)
        self.capec_embed = nn.Embedding(num_capec+1, embed_per_feature)
        self.cwe_embed = nn.Embedding(num_cwe+1, embed_per_feature)
        self.cve_embed = nn.Embedding(num_cve+1, embed_per_feature)
        self.pulse_embed = nn.Embedding(num_pulse+1, embed_per_feature)
        total_embed_size = embed_per_feature * 6
        self.fc1 = nn.Linear(total_embed_size, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        tech, soft, capec, cwe, cve, pulse = features[:, 0], features[:, 1], features[:, 2], features[:, 3], features[:, 4], features[:, 5]
        t_emb = self.tech_embed(tech.long() + 1)
        s_emb = self.soft_embed(soft.long() + 1)
        ca_emb = self.capec_embed(capec.long() + 1)
        cw_emb = self.cwe_embed(cwe.long() + 1)
        cv_emb = self.cve_embed(cve.long() + 1)
        p_emb = self.pulse_embed(pulse.long() + 1)
        x = torch.cat([t_emb, s_emb, ca_emb, cw_emb, cv_emb, p_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)

# ---------- EXACT SAME as XGBoost ----------
def load_apt_profiles_from_data():
    """EXACTLY like XGBoost - from EncodedData.csv"""
    print(" LOADING APT PROFILES FROM EncodedData.csv...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[int(apt_id)] = {
            'techniques': set(apt_data['techniques_str'].dropna().unique()),
            'software': set(apt_data['software_str'].dropna().unique()),
            'capec': set(apt_data['CAPEC_ID'].dropna().unique()),
            'cwe': set(apt_data['CWE_ID'].dropna().unique()),
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
            'pulse': set(apt_data['Pulse_ID'].dropna().unique()),
            'total_features': len(apt_data)
        }
    
    print(f" Loaded {len(profiles)} APTs - EXACTLY like XGBoost")
    return profiles

def load_model_and_mapping():
    """Load Siamese model + name mapping"""
    print(" LOADING SIAMESE MODEL...")
    with open('6feat_apt_siamese_feature_model.pkl', 'rb') as f:
        data = pickle.load(f)
    
    config = data['model_config']
    model = FeatureEncoder(
        config['num_techniques'], 
        config['num_software'], 
        config['num_capec'],
        config['num_cwe'],
        config['num_cve'],
        config['num_pulse'],
        config['embedding_dim']
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    
    # Name mapping ONLY for display
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        apt_name_mapping[int(row['APTnameEncoded'])] = row['APTnameOrigin']
    
    print(" Siamese model + name mapping loaded")
    return model, apt_name_mapping

# ---------- Siamese Similarity (MODEL FORWARD PASS) ----------
class SiameseSimilarityPredictor:
    def __init__(self, model):
        self.model = model
    
    def predict_similarity(self, features1, features2, n_samples=32):
        """Siamese cosine similarity - MODEL FORWARD PASS"""
        if features1[0] == -1 and features1[1] == -1 and features1[2] == -1 and features1[3] == -1 and features1[4] == -1 and features1[5] == -1:
            return 0.0
        if features2[0] == -1 and features2[1] == -1 and features2[2] == -1 and features2[3] == -1 and features2[4] == -1 and features2[5] == -1:
            return 0.0
        
        # Generate embeddings THROUGH MODEL
        emb1 = self._get_embedding(features1, n_samples)
        emb2 = self._get_embedding(features2, n_samples)
        
        # Cosine similarity of learned embeddings
        sim = F.cosine_similarity(emb1.mean(0).unsqueeze(0), emb2.mean(0).unsqueeze(0)).item()
        return np.clip(sim, 0.0, 1.0)
    
    def _get_embedding(self, features, n_samples=32):
        """Generate embedding via model forward pass"""
        vectors = []
        for _ in range(n_samples):
            # Sample single feature triplet
            tech, soft, capec, cwe, cve, pulse = features
            vectors.append([float(tech), float(soft), float(capec), float(cwe), float(cve), float(pulse)])
        
        vectors = torch.tensor(vectors, dtype=torch.float32)
        embeddings = self.model(vectors)
        return embeddings

# ---------- EXACT SAME as XGBoost ----------
def create_similarity_matrix_siamese(model, profiles, apt_name_mapping):
    """FULL 181x181 matrix - Siamese MODEL predictions"""
    print(" Creating FULL Siamese similarity matrix...")
    
    apt_ids = sorted(profiles.keys())
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    predictor = SiameseSimilarityPredictor(model)
    
    # Representative features per APT (EXACTLY like XGBoost)
    X_apts = []
    for apt_id in apt_ids:
        profile = profiles[apt_id]
        tech = next(iter(profile['techniques'])) if profile['techniques'] else -1
        soft = next(iter(profile['software'])) if profile['software'] else -1
        capec = next(iter(profile['capec'])) if profile['capec'] else -1
        cwe = next(iter(profile['cwe'])) if profile['cwe'] else -1
        cve = next(iter(profile['cve'])) if profile['cve'] else -1
        pulse = next(iter(profile['pulse'])) if profile['pulse'] else -1
        X_apts.append([int(tech), int(soft), int(capec), int(cwe), int(cve), int(pulse)])
    
    # FULL similarity matrix
    n = len(apt_ids)
    similarity_matrix = np.zeros((n, n))
    
    print("Computing Siamese similarities...")
    for i in range(n):
        if i % 20 == 0:
            print(f"  {i}/{n} APTs...")
        for j in range(n):
            similarity_matrix[i, j] = predictor.predict_similarity(X_apts[i], X_apts[j])
    
    similarity_df = pd.DataFrame(similarity_matrix, index=apt_names, columns=apt_names)
    return similarity_df, apt_names

# ---------- EXACT SAME as XGBoost ----------
def plot_similarity_heatmap_siamese(similarity_df):
    """FULL 181x181 heatmap - PROPER red=low, green=high scaling"""
    plt.figure(figsize=(20, 18))
    
    #  FIXED: Proper min/max scaling - RED=lowest, GREEN=highest
    vmin = similarity_df.min().min()  # Lowest value = RED
    vmax = similarity_df.max().max()  # Highest value = GREEN
    
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    
    sns.heatmap(similarity_df, 
                mask=None, 
                cmap=cmap, 
                center=None,  #  REMOVED center=0.5
                vmin=vmin,    #  RED at actual minimum
                vmax=vmax,    #  GREEN at actual maximum
                square=True,
                xticklabels=True, 
                yticklabels=True,
                cbar_kws={'shrink':0.8, 'label':'Siamese Similarity Score'})
    
    plt.title("APT Similarity Matrix (Siamese FeatureEncoder)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('siamese_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print(" PERFECT heatmap saved (Red=Min, Green=Max)!")
    plt.show()


def evaluate_similarity_matrix_siamese(similarity_df, apt_names):
    """EXACT SAME evaluation as XGBoost"""
    print("\n" + "="*80)
    print("SIAMESE SIMILARITY MATRIX EVALUATION")
    print("="*80)
    
    top_x_values = [1, 5, 10, 20]
    for top_x in top_x_values:
        success = 0
        for i, apt_name in enumerate(apt_names):
            similarities = similarity_df.iloc[i]
            top_similar = similarities.nlargest(top_x)
            if apt_name in top_similar.index:
                success += 1
        
        total = len(apt_names)
        print(f"Top-{top_x}: {success}/{total} ({success/total*100:.1f}%)")

# ---------- EXACT SAME profile division as XGBoost ----------
def evaluate_profile_division_siamese(model, divider=2):
    """EXACT SAME logic as XGBoost profile division"""
    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {int(row['APTnameEncoded']): row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    predictor = SiameseSimilarityPredictor(model)
    
    # Representative features (like XGBoost)
    X_apts = []
    for apt_id in apt_ids:
        profile = profiles[apt_id]
        tech = next(iter(profile['techniques'])) if profile['techniques'] else -1
        soft = next(iter(profile['software'])) if profile['software'] else -1
        capec = next(iter(profile['capec'])) if profile['capec'] else -1
        cwe = next(iter(profile['cwe'])) if profile['cwe'] else -1
        cve = next(iter(profile['cve'])) if profile['cve'] else -1
        pulse = next(iter(profile['pulse'])) if profile['pulse'] else -1
        X_apts.append([int(tech), int(soft), int(capec), int(cwe), int(cve), int(pulse)])
    
    print(f"\n SIAMESE PROFILE DIVISION (divider={divider})")
    resultdict = {}
    
    for i, target_apt_name in enumerate(apt_names):
        if i % 20 == 0: print(f"  {i}/{len(apt_names)}...")
        
        target_apt_id = next(apt_id for apt_id in apt_ids if apt_name_mapping.get(apt_id, f"APT{apt_id}") == target_apt_name)
        target_profile = profiles[target_apt_id]
        
        # EXACT SAME division logic as XGBoost
        divided_evidence = {'techniques': [], 'software': [], 'cve': []}
        for feature_type, key in [('techniques', 'techniques'), ('software', 'software'), ('capec', 'capec'), ('cwe', 'cwe'), ('cve', 'cve'), ('pulse', 'pulse')]:
            features = list(target_profile[feature_type])
            if features:
                random.shuffle(features)
                size = max(1, len(features) // divider)
                divided_evidence[key] = features[:size]
        
        # Query features (like XGBoost)
        tech = divided_evidence['techniques'][0] if divided_evidence['techniques'] else -1
        soft = divided_evidence['software'][0] if divided_evidence['software'] else -1
        capec = divided_evidence['capec'][0] if divided_evidence['capec'] else -1
        cwe = divided_evidence['cwe'][0] if divided_evidence['cwe'] else -1
        cve = divided_evidence['cve'][0] if divided_evidence['cve'] else -1
        pulse = divided_evidence['pulse'][0] if divided_evidence['pulse'] else -1
        query_features = [int(tech), int(soft), int(capec), int(cwe), int(cve), int(pulse)]
        
        # Rank all APTs
        similarities = [predictor.predict_similarity(query_features, apt_features) for apt_features in X_apts]
        ranked_results = {}
        for idx, sim in enumerate(similarities):
            ranked_results[apt_names[idx]] = sim
        
        resultdict[target_apt_name] = dict(sorted(ranked_results.items(), key=lambda x: x[1], reverse=True))
    
    # EXACT SAME evaluation
    for top_x in [1, 5, 10, 20]:
        success = sum(1 for target_apt, similarities in resultdict.items() 
                     if target_apt in dict(islice(similarities.items(), top_x)))
        print(f"Top-{top_x}: {success}/{len(resultdict)} ({success/len(resultdict)*100:.1f}%)")

# ---------- MAIN ----------
def main():
    print(" SIAMESE FEATURE ENCODER EVALUATION")
    print("=" * 60)
    
    model, apt_name_mapping = load_model_and_mapping()
    profiles = load_apt_profiles_from_data()
    
    # 1. FULL similarity matrix
    similarity_df, apt_names = create_similarity_matrix_siamese(model, profiles, apt_name_mapping)
    print(f"\n FULL {len(apt_names)}x{len(apt_names)} Siamese matrix created")
    
    # 2. FULL heatmap
    plot_similarity_heatmap_siamese(similarity_df)
    
    # 3. Matrix evaluation
    evaluate_similarity_matrix_siamese(similarity_df, apt_names)
    
    # 4. Profile division (EXACT XGBoost logic)
    evaluate_profile_division_siamese(model, divider=2)
    
    print("\n SIAMESE EVALUATION COMPLETE!")
    print(" Uses Siamese MODEL forward pass")
    print(" FULL 181x181 heatmap")
    print(" Exact XGBoost evaluation logic")

if __name__ == "__main__":
    main()




monitor.stop()