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

# ---------- FeatureEncoder (unchanged) ----------
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
    """Load Siamese model + name mapping + saved APT embeddings"""
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
    
    # Load saved APT embeddings and IDs (used in queries)
    apt_embeddings = data['apt_embeddings'].float()   # shape [num_apts, embed_dim]
    apt_ids = data['apt_ids']                          # list of encoded APT IDs
    
    print(" Siamese model + name mapping + saved embeddings loaded")
    return model, apt_name_mapping, apt_embeddings, apt_ids

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
    
    def get_profile_embedding(self, profile, n_samples=32):
        """Generate embedding for a full APT profile by sampling multiple feature combinations."""
        techs = list(profile['techniques']) if profile['techniques'] else [-1]
        softs = list(profile['software']) if profile['software'] else [-1]
        capecs = list(profile['capec']) if profile['capec'] else [-1]
        cwes = list(profile['cwe']) if profile['cwe'] else [-1]
        cves = list(profile['cve']) if profile['cve'] else [-1]
        pulses = list(profile['pulse']) if profile['pulse'] else [-1]
        
        vectors = []
        for _ in range(n_samples):
            tech = random.choice(techs)
            soft = random.choice(softs)
            capec = random.choice(capecs)
            cwe = random.choice(cwes)
            cve = random.choice(cves)
            pulse = random.choice(pulses)
            vectors.append([float(tech), float(soft), float(capec), float(cwe), float(cve), float(pulse)])
        
        vectors = torch.tensor(vectors, dtype=torch.float32)
        embeddings = self.model(vectors)
        return embeddings.mean(dim=0)  # average embedding

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
    
    vmin = similarity_df.min().min()
    vmax = similarity_df.max().max()
    
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    
    sns.heatmap(similarity_df, 
                mask=None, 
                cmap=cmap, 
                center=None,
                vmin=vmin,
                vmax=vmax,
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

# ---------- EXACT SAME profile division as XGBoost (fixed with all six features) ----------
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
        
        # EXACT SAME division logic as XGBoost (now includes all six features)
        divided_evidence = {'techniques': [], 'software': [], 'capec': [], 'cwe': [], 'cve': [], 'pulse': []}
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

def load_encoded_data():
    """Load the original encoded features and labels from EncodedData.csv."""
    print(" LOADING ENCODED DATA FOR FULL PROFILE EVALUATION...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    feature_columns = ['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID']
    X = df[feature_columns].values.astype(np.float32)
    y = df['APTname'].values.astype(int)
    print(f" Loaded {len(X)} rows, {len(np.unique(y))} APTs")
    return X, y

# ---------- CORRECTED FULL PROFILE EVALUATION (uses ALL rows) ----------
def evaluate_full_profile_queries_sampling(model, X, y, profiles, apt_name_mapping, saved_apt_embeddings, saved_apt_ids, top_x_values=[1, 5, 10, 20], n_samples=500):
    """
    For each APT, generate a query embedding by averaging `n_samples` random
    feature combinations drawn from the sets of actual values (ignoring -1).
    Compare with saved APT embeddings (which include -1 rows). This matches the query logic.
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING FULL PROFILE QUERIES (sampling, ignoring -1, {n_samples} samples per APT)")
    print("=" * 80)

    # Align saved embeddings with profile order
    apt_id_to_index = {apt_id: idx for idx, apt_id in enumerate(saved_apt_ids)}
    apt_ids_sorted = sorted(profiles.keys())
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids_sorted]
    embedding_order = [apt_id_to_index[apt_id] for apt_id in apt_ids_sorted]
    apt_embeddings = saved_apt_embeddings[embedding_order]  # [num_apts, embed_dim]

    # Convert y to numpy for masking
    y_np = np.array(y)
    device = next(model.parameters()).device

    resultdict = {}
    for i, target_name in enumerate(apt_names):
        if i % 20 == 0:
            print(f"  {i}/{len(apt_names)}...")

        target_id = apt_ids_sorted[i]
        # Get all rows of this APT
        mask = (y_np == target_id)
        apt_rows = X[mask]  # shape [num_rows, 6]

        # Collect sets of non‑-1 values for each feature type
        tech_set = set(apt_rows[:, 0][apt_rows[:, 0] != -1])
        soft_set = set(apt_rows[:, 1][apt_rows[:, 1] != -1])
        capec_set = set(apt_rows[:, 2][apt_rows[:, 2] != -1])
        cwe_set   = set(apt_rows[:, 3][apt_rows[:, 3] != -1])
        cve_set   = set(apt_rows[:, 4][apt_rows[:, 4] != -1])
        pulse_set = set(apt_rows[:, 5][apt_rows[:, 5] != -1])

        # If any set is empty, use [-1] (but this shouldn't happen for a full profile)
        techs = list(tech_set) if tech_set else [-1]
        softs = list(soft_set) if soft_set else [-1]
        capecs = list(capec_set) if capec_set else [-1]
        cwes   = list(cwe_set)   if cwe_set   else [-1]
        cves   = list(cve_set)   if cve_set   else [-1]
        pulses = list(pulse_set) if pulse_set else [-1]

        # Generate n_samples random combinations and average their embeddings
        vectors = []
        for _ in range(n_samples):
            tech = random.choice(techs)
            soft = random.choice(softs)
            capec = random.choice(capecs)
            cwe = random.choice(cwes)
            cve = random.choice(cves)
            pulse = random.choice(pulses)
            vectors.append([tech, soft, capec, cwe, cve, pulse])
        vec_tensor = torch.FloatTensor(vectors).to(device)
        with torch.no_grad():
            embs = model(vec_tensor)  # [n_samples, embed_dim]
        query_emb = embs.mean(dim=0)   # average over samples

        # Cosine similarities with all APT embeddings
        sims = F.cosine_similarity(query_emb.unsqueeze(0), apt_embeddings).cpu().numpy()
        ranked_indices = np.argsort(sims)[::-1]
        ranked_results = {apt_names[idx]: sims[idx] for idx in ranked_indices}
        resultdict[target_name] = ranked_results

    for top_x in top_x_values:
        success = sum(1 for target_name, ranked in resultdict.items() if target_name in list(ranked.keys())[:top_x])
        total = len(resultdict)
        print(f"Top-{top_x}: {success}/{total} ({success/total*100:.1f}%)")

# ---------- MAIN ----------
def main():
    print(" SIAMESE FEATURE ENCODER EVALUATION")
    print("=" * 60)
    # Load the full encoded data (X, y) for the ALL‑rows evaluation
    X, y = load_encoded_data()   
    
    model, apt_name_mapping, saved_apt_embeddings, saved_apt_ids = load_model_and_mapping()
    profiles = load_apt_profiles_from_data()
    
    # 1. FULL similarity matrix
    similarity_df, apt_names = create_similarity_matrix_siamese(model, profiles, apt_name_mapping)
    print(f"\n FULL {len(apt_names)}x{len(apt_names)} Siamese matrix created")
    
    # 2. FULL heatmap
    plot_similarity_heatmap_siamese(similarity_df)
    
    # 3. Matrix evaluation
    evaluate_similarity_matrix_siamese(similarity_df, apt_names)

    # 4. Full profile queries (using saved embeddings, averaging multiple samples)
    evaluate_full_profile_queries_sampling(model, X, y, profiles, apt_name_mapping, saved_apt_embeddings, saved_apt_ids)

    # 5. Profile division (EXACT XGBoost logic)
    evaluate_profile_division_siamese(model, divider=2)
    
    print("\n SIAMESE EVALUATION COMPLETE!")
    print(" Uses Siamese MODEL forward pass")
    print(" FULL 181x181 heatmap")
    print(" Corrected full‑profile evaluation (embedding averaging)")
    print(" Exact XGBoost evaluation logic")

if __name__ == "__main__":
    main()