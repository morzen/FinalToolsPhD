import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------- FeatureEncoder (unchanged) ----------
class FeatureEncoder(nn.Module):
    def __init__(self, num_techniques, num_software, num_cve, embed_dim=256):
        super().__init__()
        embed_per_feature = embed_dim // 3
        self.tech_embed = nn.Embedding(num_techniques+1, embed_per_feature)
        self.soft_embed = nn.Embedding(num_software+1, embed_per_feature)
        self.cve_embed = nn.Embedding(num_cve+1, embed_per_feature)
        total_embed_size = embed_per_feature * 3
        self.fc1 = nn.Linear(total_embed_size, 256)
        self.fc2 = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, features):
        tech, soft, cve = features[:, 0], features[:, 1], features[:, 2]
        t_emb = self.tech_embed(tech.long() + 1)
        s_emb = self.soft_embed(soft.long() + 1)
        cv_emb = self.cve_embed(cve.long() + 1)
        x = torch.cat([t_emb, s_emb, cv_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)

# ---------- Load model and raw data ----------
def load_model_and_data():
    print("LOADING saved model...")
    with open('3feat_apt_siamese_feature_model.pkl', 'rb') as f:
        data = pickle.load(f)
    config = data['model_config']
    model = FeatureEncoder(
        config['num_techniques'], config['num_software'], config['num_cve'], config['embedding_dim']
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {int(row['APTnameEncoded']): row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    
    # Saved full‑profile embeddings (averaged over ALL rows)
    apt_embeddings = data['apt_embeddings'].float()
    apt_ids = data['apt_ids']
    
    # Raw data (all rows)
    df_raw = pd.read_csv("../BaseData/EncodedData.csv")
    feature_cols = ['techniques_str', 'software_str', 'CVE_ID']
    X_raw = df_raw[feature_cols].values.astype(np.float32)
    y_raw = df_raw['APTname'].values.astype(int)
    
    print(f"Loaded {len(apt_ids)} APTs, {len(X_raw)} total rows")
    return model, apt_name_mapping, apt_ids, apt_embeddings, X_raw, y_raw

# ---------- Helper: compute embedding from rows (no sampling) ----------
def embedding_from_rows(model, rows, device):
    if len(rows) == 0:
        rows = np.array([[-1]*3])
    tensor = torch.FloatTensor(rows).to(device)
    with torch.no_grad():
        return model(tensor).mean(dim=0).cpu()

# ---------- 1. Full similarity matrix (full profile vs full profile) ----------
def full_similarity_matrix(apt_embeddings, apt_ids, apt_name_mapping):
    print("\n=== 1. Full vs Full Similarity Matrix (all rows) ===")
    n = len(apt_ids)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i,j] = F.cosine_similarity(apt_embeddings[i].unsqueeze(0),
                                                  apt_embeddings[j].unsqueeze(0)).item()
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    sim_df = pd.DataFrame(sim_matrix, index=apt_names, columns=apt_names)
    
    for top_x in [1,5,10,20]:
        success = 0
        for i, name in enumerate(apt_names):
            top = sim_df.iloc[i].nlargest(top_x)
            if name in top.index:
                success += 1
        print(f"Top-{top_x}: {success}/{n} ({success/n*100:.1f}%)")
    return sim_df

# ---------- 2. Complete profile analysis (no sampling: query = all rows, database = all rows) ----------
def complete_profile_analysis(model, X_raw, y_raw, apt_ids, apt_name_mapping, apt_embeddings, device):
    """
    For each APT, compute query embedding from ALL its rows.
    Database is the saved full‑profile embeddings (also from all rows).
    This is trivial (100% top‑1) but produces a similarity matrix.
    """
    print("\n=== 2. Complete Profile Analysis (no sampling, all rows for query) ===")
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    n = len(apt_ids)
    
    # Compute query embeddings from all rows (recompute to be safe, but they are identical to saved ones)
    query_embs = []
    for apt_id in apt_ids:
        mask = (y_raw == apt_id)
        rows = X_raw[mask]
        emb = embedding_from_rows(model, rows, device)
        query_embs.append(emb)
    query_embs = torch.stack(query_embs)
    query_embs = query_embs / query_embs.norm(dim=1, keepdim=True)
    
    # Database: saved full‑profile embeddings
    db_embs = apt_embeddings / apt_embeddings.norm(dim=1, keepdim=True)
    
    # Build similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i,j] = torch.dot(query_embs[i], db_embs[j]).item()
    
    sim_df = pd.DataFrame(sim_matrix, index=apt_names, columns=apt_names)
    
    plt.figure(figsize=(20,18))
    vmin, vmax = sim_df.min().min(), sim_df.max().max()
    cmap = LinearSegmentedColormap.from_list("red_green", ["red","yellow","green"], N=100)
    sns.heatmap(sim_df, cmap=cmap, vmin=vmin, vmax=vmax, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={'label':'Cosine Similarity'})
    plt.title("Complete Profile Analysis (all rows query vs all rows DB)", fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('complete_profile_analysis_heatmap.png', dpi=300)
    plt.show()
    
    # Evaluate top‑k
    for top_x in [1,5,10,20]:
        success = 0
        for i, name in enumerate(apt_names):
            top_indices = np.argsort(sim_matrix[i])[::-1][:top_x]
            top_names = [apt_names[idx] for idx in top_indices]
            if name in top_names:
                success += 1
        print(f"Top-{top_x}: {success}/{n} ({success/n*100:.1f}%)")

# ---------- 3. Profile division (half rows vs full profiles) ----------
def profile_division(model, X_raw, y_raw, apt_ids, apt_name_mapping, apt_embeddings, device, divider=2):
    print(f"\n=== 3. Profile Division (divider={divider}) – half rows vs full profiles ===")
    np.random.seed(42)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    n = len(apt_ids)
    db_embs = apt_embeddings / apt_embeddings.norm(dim=1, keepdim=True)
    
    success = {k:0 for k in [1,5,10,20]}
    for i, apt_id in enumerate(apt_ids):
        mask = (y_raw == apt_id)
        rows = X_raw[mask].copy()
        np.random.shuffle(rows)
        half_len = max(1, len(rows) // divider)
        query_rows = rows[:half_len]
        query_emb = embedding_from_rows(model, query_rows, device)
        query_emb = query_emb / query_emb.norm()
        sims = torch.mv(db_embs, query_emb)
        ranked = torch.argsort(sims, descending=True)
        ranked_names = [apt_names[idx] for idx in ranked]
        for k in success.keys():
            if apt_names[i] in ranked_names[:k]:
                success[k] += 1
        if i % 20 == 0:
            print(f"  {i}/{n}...")
    for k, val in success.items():
        print(f"Top-{k}: {val}/{n} ({val/n*100:.1f}%)")

# ---------- MAIN ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, apt_name_mapping, apt_ids, apt_embeddings, X_raw, y_raw = load_model_and_data()
    model = model.to(device)
    
    # 1. Full vs Full similarity matrix
    full_similarity_matrix(apt_embeddings, apt_ids, apt_name_mapping)
    
    # 2. Complete profile analysis (all rows query vs all rows DB) – produces heatmap
    complete_profile_analysis(model, X_raw, y_raw, apt_ids, apt_name_mapping, apt_embeddings, device)
    
    # 3. Profile division (half rows vs full profiles)
    profile_division(model, X_raw, y_raw, apt_ids, apt_name_mapping, apt_embeddings, device, divider=2)
    
    print("\n ALL TESTS COMPLETE – no sampling, no random combinations, just rows.")

if __name__ == "__main__":
    main()