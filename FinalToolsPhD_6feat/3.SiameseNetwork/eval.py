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
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P   # shared evaluation protocol (profiles, ranking, divided, false flag)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from query import SiameseScorer

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

# ---------- Load model and raw data ----------
def load_model_and_data():
    print("LOADING saved model...")
    with open('6feat_apt_siamese_feature_model.pkl', 'rb') as f:
        data = pickle.load(f)
    config = data['model_config']
    model = FeatureEncoder(
        config['num_techniques'], config['num_software'], config['num_capec'],
        config['num_cwe'], config['num_cve'], config['num_pulse'],
        config['embedding_dim']
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
    feature_cols = ['techniques_str', 'software_str', 'CAPEC_ID', 'CWE_ID', 'CVE_ID', 'Pulse_ID']
    X_raw = df_raw[feature_cols].values.astype(np.float32)
    y_raw = df_raw['APTname'].values.astype(int)
    
    print(f"Loaded {len(apt_ids)} APTs, {len(X_raw)} total rows")
    return model, apt_name_mapping, apt_ids, apt_embeddings, X_raw, y_raw

# ---------- Helper: compute embedding from rows (no sampling) ----------
def embedding_from_rows(model, rows, device):
    if len(rows) == 0:
        rows = np.array([[-1]*6])
    tensor = torch.FloatTensor(rows).to(device)
    with torch.no_grad():
        return model(tensor).mean(dim=0).cpu()

# ---------- 1. Full similarity matrix (full profile vs full profile) ----------

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

# ---------- 3. Profile division (half rows vs full profiles) ----------

# ---------- MAIN ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, apt_name_mapping, apt_ids, apt_embeddings, X_raw, y_raw = load_model_and_data()
    model = model.to(device)

    # Heatmap of the stored APT embeddings (figure only)
    complete_profile_analysis(model, X_raw, y_raw, apt_ids, apt_name_mapping, apt_embeddings, device)

    # Attribution evaluation with the shared protocol (full data, shuffled & divided by 2), using
    # the same query construction as query.py / hybrid_query
    P.evaluate_attribution(SiameseScorer(), "Siamese Network", results_dir="results", smoke=P.smoke_flag())

if __name__ == "__main__":
    main()