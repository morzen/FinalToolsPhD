import os
import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "3feat_apt_siamese_feature_model.pkl"
TOP_X_VALUES = [1, 5, 10, 20]
MAX_WORKERS = min(mp.cpu_count(), 8)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 32               # number of random vectors to average for query embedding
PARTIAL_KEEP_FRACTION = 0.5  # keep 50% of features (like MIL volume_fraction)

# ---------- FeatureEncoder (must match trained model) ----------
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

# ============================================================
# LOAD APT PROFILES (from EncodedData.csv)
# ============================================================
def load_apt_profiles_from_data():
    print(" LOADING APT PROFILES FROM EncodedData.csv...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[int(apt_id)] = {
            'techniques': set(apt_data['techniques_str'].dropna().unique()),
            'software': set(apt_data['software_str'].dropna().unique()),
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
            'total_features': len(apt_data)
        }
    print(f" Loaded {len(profiles)} APTs")
    return profiles

# ============================================================
# LOAD MODEL AND EMBEDDINGS
# ============================================================
def load_model_and_embeddings():
    print(" LOADING SIAMESE MODEL AND EMBEDDINGS...")
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    config = data['model_config']
    model = FeatureEncoder(
        config['num_techniques'],
        config['num_software'],
        config['num_cve'],
        config['embedding_dim']
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    
    apt_embeddings = data['apt_embeddings']
    apt_ids = data['apt_ids']
    apt_to_idx = data['apt_to_idx']
    
    # APT name mapping
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {int(row['APTnameEncoded']): row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    
    print(f" Model + {len(apt_ids)} embeddings loaded.")
    return model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx

# ============================================================
# 30/70 SAMPLING + PARTIAL (per‑feature dropout, like MIL)
# ============================================================
def sample_mixed_features(attacker_profile, mimicked_profile, rng=None,
                          attacker_fraction=0.3, partial=False, keep_fraction=0.5):
    """
    Returns a 3‑element array with 30% attacker / 70% mimicked per feature type.
    If partial=True, randomly set each feature to -1 (missing) with probability (1-keep_fraction).
    This mimics MIL's "shuffle and keep ~50% per category" at the per‑feature level.
    """
    if rng is None:
        rng = random
    
    def pick_one(atk_list, mim_list):
        if not atk_list and not mim_list:
            return -1
        use_attacker = (rng.random() < attacker_fraction)
        if use_attacker and atk_list:
            return rng.choice(list(atk_list))
        elif mim_list:
            return rng.choice(list(mim_list))
        elif atk_list:
            return rng.choice(list(atk_list))
        return -1
    
    features = np.array([
        pick_one(attacker_profile['techniques'], mimicked_profile['techniques']),
        pick_one(attacker_profile['software'], mimicked_profile['software']),
        pick_one(attacker_profile['cve'], mimicked_profile['cve'])
    ], dtype=np.float32)
    
    if partial:
        # Per‑feature dropout: keep with probability keep_fraction
        for i in range(3):
            if rng.random() > keep_fraction:
                features[i] = -1.0
    return features

def generate_query_embedding(model, features, n_samples=N_SAMPLES):
    """Average embedding over n_samples copies of the feature vector (to mimic training)."""
    batch = np.tile(features, (n_samples, 1))
    batch_tensor = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        emb = model(batch_tensor)
    return emb.mean(dim=0)

# ============================================================
# WORKER PROCESS
# ============================================================
def process_single_false_flag_siamese(args):
    (attacker_apt_id, mimicked_apt_id, model_state_dict, config,
     apt_embeddings, apt_ids, apt_name_mapping, profiles,
     apt_to_idx, partial, seed) = args
    
    rng = random.Random(seed)
    device = DEVICE  # use same device as main
    
    model = FeatureEncoder(
        config['num_techniques'],
        config['num_software'],
        config['num_cve'],
        config['embedding_dim']
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # Build false‑flag evidence (30/70 mix) with optional per‑feature dropout
    false_flag_features = sample_mixed_features(
        attacker_profile, mimicked_profile, rng,
        partial=partial, keep_fraction=PARTIAL_KEEP_FRACTION
    )
    
    query_emb = generate_query_embedding(model, false_flag_features, N_SAMPLES)
    
    # Compute similarities
    apt_emb_tensor = apt_embeddings.to(device)
    similarities = torch.matmul(query_emb.unsqueeze(0), apt_emb_tensor.t()).squeeze().cpu().numpy()
    
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_apts = [apt_ids[idx] for idx in sorted_indices]
    
    attacker_rank = ranked_apts.index(attacker_apt_id) + 1 if attacker_apt_id in ranked_apts else None
    mimicked_rank = ranked_apts.index(mimicked_apt_id) + 1 if mimicked_apt_id in ranked_apts else None
    attacker_sim = similarities[apt_to_idx[attacker_apt_id]]
    mimicked_sim = similarities[apt_to_idx[mimicked_apt_id]]
    
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")
    key = f"{attacker_name}→{mimicked_name}" + ("_partial" if partial else "")
    
    return {
        'key': key,
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_similarity': float(attacker_sim),
        'mimicked_similarity': float(mimicked_sim)
    }

# ============================================================
# EVALUATION FUNCTION (FULL / PARTIAL)
# ============================================================
def evaluate_false_flag_attacks_siamese(model, apt_embeddings, apt_ids, apt_names,
                                        apt_name_mapping, apt_to_idx, profiles,
                                        partial=False):
    scenario = "PARTIAL (per‑feature dropout, keep 50%)" if partial else "FULL (30/70 mix)"
    print(f"\n{'='*80}")
    print(f" SIAMESE FALSE-FLAG: {scenario}")
    print(f"{'='*80}")
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    print(f"Testing {total_combinations:,} combinations...")
    print(f"Using {mp.cpu_count()} cores")
    
    model_state_dict = model.state_dict()
    config = {
        'num_techniques': model.tech_embed.num_embeddings - 1,
        'num_software': model.soft_embed.num_embeddings - 1,
        'num_cve': model.cve_embed.num_embeddings - 1,
        'embedding_dim': model.fc2.out_features
    }
    
    args_list = []
    for i, (attacker_id, mimicked_id) in enumerate(all_combinations):
        args_list.append((
            attacker_id, mimicked_id, model_state_dict, config,
            apt_embeddings, apt_ids, apt_name_mapping, profiles,
            apt_to_idx, partial, i
        ))
    
    all_results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_false_flag_siamese, args): i for i, args in enumerate(args_list)}
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results[result['key']] = result
                completed += 1
                if completed % 500 == 0:
                    elapsed = time.time() - start_time
                    pct = completed / total_combinations * 100
                    eta = (elapsed / completed * total_combinations - elapsed) / 60
                    print(f"Progress: {completed:,}/{total_combinations:,} ({pct:.1f}%) ETA: {eta:.1f} min")
            except Exception as e:
                print(f"Worker error: {e}")
    
    total_time = time.time() - start_time
    print(f"\n Completed {len(all_results):,} evals in {total_time/60:.1f} min")
    
    # Print results
    print("\n" + "=" * 80)
    print(f"SIAMESE RESULTS - {scenario}")
    print("=" * 80)
    for top_x in TOP_X_VALUES:
        attacker_detected = mimicked_deceived = both_correct = neither_correct = 0
        for res in all_results.values():
            ar = res["attacker_rank"]
            mr = res["mimicked_rank"]
            attacker_in_top = ar is not None and ar <= top_x
            mimicked_in_top = mr is not None and mr <= top_x
            if attacker_in_top: attacker_detected += 1
            if mimicked_in_top: mimicked_deceived += 1
            if attacker_in_top and mimicked_in_top: both_correct += 1
            if not attacker_in_top and not mimicked_in_top: neither_correct += 1
        total_cases = len(all_results)
        print(f"\n Top-{top_x}:")
        print(f"  Attacker: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
    
    return all_results

# ============================================================
# HEATMAP (same as before)
# ============================================================
def plot_false_flag_deception_siamese(results, apt_names, apt_ids, partial=False):
    print("\n CREATING SIAMESE DECEPTION HEATMAP...")
    n = len(apt_ids)
    id_to_idx = {apt_id: i for i, apt_id in enumerate(apt_ids)}
    deception_matrix = np.zeros((n, n), dtype=float)
    
    for key, res in results.items():
        if partial and "_partial" not in key:
            continue
        if not partial and "_partial" in key:
            continue
        try:
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()
        except:
            continue
        attacker_id = next((aid for aid, name in zip(apt_ids, apt_names) if name == attacker_name), None)
        mimicked_id = next((aid for aid, name in zip(apt_ids, apt_names) if name == mimicked_name), None)
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue
        ar = res["attacker_rank"] or n
        mr = res["mimicked_rank"] or n
        deception_matrix[i, j] = ar - mr
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠ No deception signal")
        return
    vmin, vmax = -max(np.abs(non_zero)), max(np.abs(non_zero))
    plt.figure(figsize=(16, 14))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, square=True,
                cbar_kws={'label': 'Deception (Attacker Rank - Mimicked Rank)\nRed>0 Deception | Blue<0 Detection'})
    scenario = "PARTIAL" if partial else "FULL"
    plt.title(f'Siamese False-Flag Deception (30/70 Mix - {scenario})\nRows=Attacker | Columns=Mimicked', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT (70%)')
    plt.ylabel('Actual Attacker (30%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    output_path = f'3feat_false_flag_deception_matrix_siamese_30_70_{"partial" if partial else "full"}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {output_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    print(" SIAMESE FALSE-FLAG EVALUATION (30/70 MIX + PARTIAL DROPOUT)")
    print("=" * 70)
    model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx = load_model_and_embeddings()
    profiles = load_apt_profiles_from_data()
    
    # FULL (no dropout)
    full_results = evaluate_false_flag_attacks_siamese(
        model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx, profiles,
        partial=False
    )
    
    # PARTIAL (per‑feature dropout with keep_fraction = 0.5)
    partial_results = evaluate_false_flag_attacks_siamese(
        model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx, profiles,
        partial=True
    )
    
    # Heatmaps
    plot_false_flag_deception_siamese(full_results, apt_names, apt_ids, partial=False)

    

    plot_false_flag_deception_siamese(partial_results, apt_names, apt_ids, partial=True)
    
    print("\n SIAMESE 30/70 FALSE-FLAG COMPLETE!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    main()