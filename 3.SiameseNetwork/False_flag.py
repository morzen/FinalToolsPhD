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
MODEL_PATH = "6feat_apt_siamese_feature_model.pkl"
TOP_X_VALUES = [1, 5, 10, 20]
MAX_WORKERS = min(mp.cpu_count(), 8)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FEATURES_PER_CAT = 10   # max features to sample per category (unified with GNN/MIL/RF)

# ---------- FeatureEncoder (must match trained model) ----------
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
            'capec': set(apt_data['CAPEC_ID'].dropna().unique()),
            'cwe': set(apt_data['CWE_ID'].dropna().unique()),
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
            'pulse': set(apt_data['Pulse_ID'].dropna().unique()),
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
        config['num_capec'],
        config['num_cwe'],
        config['num_cve'],
        config['num_pulse'],
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
# UNIFIED PER‑FEATURE 30/70 MIXING (same as GNN/MIL/RF)
# ============================================================
def sample_mixed_features_unified(attacker_profile, mimicked_profile, rng=None,
                                  attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT):
    """
    For each category, sample up to `max_count` features,
    each chosen with probability attacker_fraction from attacker's set,
    otherwise from mimicked's set. Sampling without replacement.
    Returns dict with lists.
    """
    if rng is None:
        rng = random.Random()

    mixed = {}
    categories = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
    for cat in categories:
        atk_list = list(attacker_profile.get(cat, []))
        mim_list = list(mimicked_profile.get(cat, []))
        if not atk_list and not mim_list:
            mixed[cat] = []
            continue

        result = []
        atk_copy = atk_list[:]
        mim_copy = mim_list[:]
        total_needed = min(max_count, len(atk_copy) + len(mim_copy))
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_copy:
                chosen = rng.choice(atk_copy)
                result.append(chosen)
                atk_copy.remove(chosen)
            elif mim_copy:
                chosen = rng.choice(mim_copy)
                result.append(chosen)
                mim_copy.remove(chosen)
            else:
                break
        mixed[cat] = result
    return mixed

def sample_mixed_features_partial_unified(attacker_profile, mimicked_profile, rng=None,
                                          attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT,
                                          volume_fraction=0.5):
    """
    Partial evidence:
      1) Build full 30/70 mixed evidence (per-feature, up to max_count).
      2) Reduce volume per category: shuffle and keep volume_fraction (default 0.5).
    """
    full_mixed = sample_mixed_features_unified(attacker_profile, mimicked_profile, rng,
                                               attacker_fraction, max_count)
    partial_mixed = {}
    for cat, feats in full_mixed.items():
        feats = list(feats)
        rng.shuffle(feats)
        keep = max(1, int(len(feats) * volume_fraction)) if feats else 0
        partial_mixed[cat] = feats[:keep]
    return partial_mixed

# ============================================================
# Convert mixed lists into a list of rows (each row = 6 features)
# ============================================================
def mixed_lists_to_rows(mixed_lists):
    """
    mixed_lists: dict with keys 'techniques','software',..., each a list of feature IDs.
    Returns a numpy array of shape (n_rows, 6) where each row contains one feature from each category.
    Rows are created by taking the i-th element from each list (if exists, else -1).
    Number of rows = maximum length among the lists.
    """
    categories = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
    max_len = max(len(mixed_lists[cat]) for cat in categories) if mixed_lists else 0
    if max_len == 0:
        # empty evidence: return a single row of -1
        return np.array([[-1, -1, -1, -1, -1, -1]], dtype=np.float32)
    
    rows = []
    for i in range(max_len):
        row = []
        for cat in categories:
            lst = mixed_lists[cat]
            if i < len(lst):
                row.append(lst[i])
            else:
                row.append(-1)
        rows.append(row)
    return np.array(rows, dtype=np.float32)

# ============================================================
# Embedding from rows (exactly as in eval.py)
# ============================================================
def embedding_from_rows(model, rows, device):
    """
    rows: numpy array of shape (n_rows, 6)
    Returns averaged embedding (1D tensor)
    """
    if len(rows) == 0:
        rows = np.array([[-1]*6])
    tensor = torch.FloatTensor(rows).to(device)
    with torch.no_grad():
        embs = model(tensor)          # (n_rows, embed_dim)
        return embs.mean(dim=0).cpu() # (embed_dim,)

# ============================================================
# WORKER PROCESS (UPDATED: uses rows)
# ============================================================
def process_single_false_flag_siamese(args):
    (attacker_apt_id, mimicked_apt_id, model_state_dict, config,
     apt_embeddings, apt_ids, apt_name_mapping, profiles,
     apt_to_idx, partial, seed) = args
    
    rng = random.Random(seed)
    device = DEVICE
    
    model = FeatureEncoder(
        config['num_techniques'],
        config['num_software'],
        config['num_capec'],
        config['num_cwe'],
        config['num_cve'],
        config['num_pulse'],
        config['embedding_dim']
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # Generate mixed evidence (dict of lists)
    if partial:
        mixed_lists = sample_mixed_features_partial_unified(
            attacker_profile, mimicked_profile, rng,
            attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT, volume_fraction=0.5
        )
    else:
        mixed_lists = sample_mixed_features_unified(
            attacker_profile, mimicked_profile, rng,
            attacker_fraction=0.3, max_count=MAX_FEATURES_PER_CAT
        )
    
    # Convert to rows
    rows = mixed_lists_to_rows(mixed_lists)
    
    # Compute query embedding
    query_emb = embedding_from_rows(model, rows, device)
    query_emb = query_emb / query_emb.norm()
    
    # Database embeddings (already normalized)
    db_embs = apt_embeddings.to(device)
    db_embs = db_embs / db_embs.norm(dim=1, keepdim=True)
    
    # Similarities
    similarities = torch.mv(db_embs, query_emb.to(device)).cpu().numpy()
    
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
    scenario = "PARTIAL (unified 30/70 + 50% volume reduction)" if partial else "FULL (unified 30/70 per‑feature mix)"
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
        'num_capec': model.capec_embed.num_embeddings - 1,
        'num_cwe': model.cwe_embed.num_embeddings - 1,
        'num_cve': model.cve_embed.num_embeddings - 1,
        'num_pulse': model.pulse_embed.num_embeddings - 1,
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
        print(f"  ⚖ Net Advantage: {mimicked_deceived/total_cases*100 - attacker_detected/total_cases*100:+.1f}%")
    
    return all_results

# ============================================================
# HEATMAP (unchanged)
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
    plt.title(f'Siamese False-Flag Deception (Unified 30/70 Mix - {scenario})\nRows=Attacker | Columns=Mimicked', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT (70%)')
    plt.ylabel('Actual Attacker (30%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    output_path = f'false_flag_deception_matrix_siamese_unified_30_70_{"partial" if partial else "full"}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {output_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    print(" SIAMESE FALSE-FLAG EVALUATION (UNIFIED PER‑FEATURE 30/70 MIX, ROW‑BASED)")
    print("=" * 70)
    print("Generates false‑flag evidence with 30% attacker + 70% mimicked features")
    print("(up to 10 features per category), then constructs rows (each row = 6 features)")
    print("and computes embedding by averaging over rows – exactly like eval.py")
    print("=" * 70)
    
    model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx = load_model_and_embeddings()
    profiles = load_apt_profiles_from_data()
    
    # FULL (no volume reduction)
    full_results = evaluate_false_flag_attacks_siamese(
        model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx, profiles,
        partial=False
    )
    
    # PARTIAL (volume reduction 50%)
    partial_results = evaluate_false_flag_attacks_siamese(
        model, apt_embeddings, apt_ids, apt_names, apt_name_mapping, apt_to_idx, profiles,
        partial=True
    )
    
    # Heatmaps
    plot_false_flag_deception_siamese(full_results, apt_names, apt_ids, partial=False)
    plot_false_flag_deception_siamese(partial_results, apt_names, apt_ids, partial=True)
    
    print("\n SIAMESE UNIFIED 30/70 FALSE-FLAG COMPLETE!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    main()