# PERFECT siamese_false_flag.py - Matches XGBoost/GNN EXACTLY
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
from itertools import islice
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# CONFIG - SAME AS XGBoost/GNN
# ============================================================
MODEL_PATH = "apt_siamese_feature_model.pkl"
TOP_X_VALUES = [1, 5, 10, 20]
MAX_WORKERS = min(mp.cpu_count(), 8)

# ---------- YOUR EXACT FeatureEncoder from eval.py ----------
class FeatureEncoder(nn.Module):
    def __init__(self, num_techniques, num_software, num_capec, num_cwes, num_cves, num_pulse, embed_dim=128):
        super().__init__()
        embed_per_feature = embed_dim // 6
        self.tech_embed = nn.Embedding(num_techniques+1, embed_per_feature)
        self.soft_embed = nn.Embedding(num_software+1, embed_per_feature)
        self.capec_embed = nn.Embedding(num_capec+1, embed_per_feature)
        self.cwe_embed = nn.Embedding(num_cwes+1, embed_per_feature)
        self.cve_embed = nn.Embedding(num_cves+1, embed_per_feature)
        self.pulse_embed = nn.Embedding(num_pulse+1, embed_per_feature)
        total_embed_size = embed_per_feature * 6
        self.fc1 = nn.Linear(total_embed_size, 64)
        self.fc2 = nn.Linear(64, embed_dim)
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
# EXACT SAME LOADERS AS XGBoost
# ============================================================
def load_apt_profiles_from_data():
    """EXACTLY like XGBoost - from EncodedData.csv"""
    print("📊 LOADING APT PROFILES FROM EncodedData.csv...")
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
    
    print(f"✅ Loaded {len(profiles)} APTs - EXACTLY like XGBoost")
    return profiles

def load_model_and_mappings():
    """Load Siamese FeatureEncoder + mappings"""
    print("🔍 LOADING SIAMESE FEATURE ENCODER...")
    
    # Load YOUR model
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    config = data['model_config']
    model = FeatureEncoder(
        config['num_techniques'], 
        config['num_software'], 
        config['num_capec'],
        config['num_cwes'],
        config['num_cves'],
        config['num_pulse'],
        config['embedding_dim']
    )
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    
    # Representative features (EXACTLY like XGBoost)
    df = pd.read_csv("../BaseData/EncodedData.csv")
    apt_ids = sorted(df['APTname'].unique().astype(int))
    profiles = load_apt_profiles_from_data()
    
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
    X_apts = np.array(X_apts)
    
    # Name mapping
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {int(row['APTnameEncoded']): row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(apt_id, f"APT{apt_id}") for apt_id in apt_ids]
    
    print(f"✅ Model + {len(apt_ids)} APTs + features ready")
    return model, X_apts, apt_ids, apt_names, apt_name_mapping, profiles

# ============================================================
# SiameseSimilarityPredictor - EXACT XGBoost structure
# ============================================================
class SiameseSimilarityPredictor:
    def __init__(self, model):
        self.model = model
    
    def predict_similarity(self, features1, features2, n_samples=32):
        """EXACTLY like XGBoost - pure model cosine similarity"""
        if features1[0] == -1 and features1[1] == -1 and features1[2] == -1 and features1[3] == -1 and features1[4] == -1 and features1[5] == -1:
            return 0.0
        if features2[0] == -1 and features2[1] == -1 and features2[2] == -1 and features2[3] == -1 and features2[4] == -1 and features2[5] == -1:
            return 0.0
        
        # Generate embeddings THROUGH MODEL (n_samples like XGBoost prob dist)
        emb1 = self._get_embedding(features1, n_samples)
        emb2 = self._get_embedding(features2, n_samples)
        
        # Pure cosine similarity (like XGBoost prob1·prob2)
        sim = F.cosine_similarity(emb1.mean(0).unsqueeze(0), emb2.mean(0).unsqueeze(0)).item()
        return np.clip(sim, 0.0, 1.0)
    
    def _get_embedding(self, features, n_samples=32):
        """Model forward pass - multiple samples"""
        vectors = []
        for _ in range(n_samples):
            tech, soft, capec, cwe, cve, pulse = features
            vectors.append([float(tech), float(soft), float(capec), float(cwe), float(cve), float(pulse)])
        
        vectors = torch.tensor(vectors, dtype=torch.float32)
        return self.model(vectors)

# ============================================================
# EXACT SAME 30/70 SAMPLING AS XGBoost
# ============================================================
def sample_mixed_features(attacker_profile, mimicked_profile, rng=None, attacker_fraction=0.3):
    """EXACTLY like XGBoost 30/70 sampling"""
    if rng is None:
        rng = random
    
    def pick_feature(atk_list, mim_list):
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
    
    technique = pick_feature(attacker_profile['techniques'], mimicked_profile['techniques'])
    software = pick_feature(attacker_profile['software'], mimicked_profile['software'])
    capec = pick_feature(attacker_profile['capec'], mimicked_profile['capec'])
    cwe = pick_feature(attacker_profile['cwe'], mimicked_profile['cwe'])
    cve = pick_feature(attacker_profile['cve'], mimicked_profile['cve'])
    pulse = pick_feature(attacker_profile['pulse'], mimicked_profile['pulse'])
    
    return np.array([int(technique), int(software), int(capec), int(cwe), int(cve), int(pulse)])

# ============================================================
# WORKER - EXACT XGBoost structure
# ============================================================
def process_single_false_flag_siamese(args):
    """EXACT XGBoost worker structure"""
    attacker_apt_id, mimicked_apt_id, model, X_apts, apt_ids, apt_name_mapping, profiles, partial, seed = args
    
    rng = random.Random(seed)
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # 30/70 mix (EXACTLY like XGBoost)
    if partial:
        full_mix = sample_mixed_features(attacker_profile, mimicked_profile, rng)
        mask = [0, 1, 2, 3, 4, 5]
        rng.shuffle(mask)
        false_flag_features = full_mix.copy()
        false_flag_features[mask[0]] = -1  # Drop 1 slot
    else:
        false_flag_features = sample_mixed_features(attacker_profile, mimicked_profile, rng)
    
    # Siamese model prediction (like XGBoost predict_proba)
    predictor = SiameseSimilarityPredictor(model)
    similarities = [predictor.predict_similarity(false_flag_features, apt_features) 
                   for apt_features in X_apts]
    
    # Ranking (EXACTLY like XGBoost)
    sorted_indices = np.argsort(similarities)[::-1]
    ranked_results = {apt_ids[idx]: similarities[idx] for idx in sorted_indices}
    
    attacker_rank = list(ranked_results.keys()).index(attacker_apt_id) + 1 if attacker_apt_id in ranked_results else None
    mimicked_rank = list(ranked_results.keys()).index(mimicked_apt_id) + 1 if mimicked_apt_id in ranked_results else None
    
    attacker_name = apt_name_mapping.get(attacker_apt_id, f"APT{attacker_apt_id}")
    mimicked_name = apt_name_mapping.get(mimicked_apt_id, f"APT{mimicked_apt_id}")
    key = f"{attacker_name}→{mimicked_name}" + ("_partial" if partial else "")
    
    return {
        'key': key,
        'attacker_rank': attacker_rank,
        'mimicked_rank': mimicked_rank,
        'attacker_similarity': ranked_results.get(attacker_apt_id),
        'mimicked_similarity': ranked_results.get(mimicked_apt_id)
    }

# ============================================================
# MAIN EVALUATION - EXACT XGBoost format
# ============================================================
def evaluate_false_flag_attacks_siamese(model, X_apts, apt_ids, apt_names, apt_name_mapping, profiles, partial=False):
    """EXACT XGBoost evaluation structure"""
    scenario = "PARTIAL (~33% reduction)" if partial else "FULL (30/70 mix)"
    print(f"\n{'='*80}")
    print(f"🎯 SIAMESE FALSE-FLAG: 30% Attacker + 70% Mimicked ({scenario})")
    print(f"{'='*80}")
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations:,} combinations...")
    print(f"Using {mp.cpu_count()} cores")
    
    args_list = [(attacker_id, mimicked_id, model, X_apts, apt_ids, apt_name_mapping, profiles, partial, i) 
                for i, (attacker_id, mimicked_id) in enumerate(all_combinations)]
    
    all_results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_false_flag_siamese, args): i for i, args in enumerate(args_list)}
        
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_results[result['key']] = {
                        'attacker_rank': result['attacker_rank'],
                        'mimicked_rank': result['mimicked_rank'],
                        'attacker_similarity': result['attacker_similarity'],
                        'mimicked_similarity': result['mimicked_similarity']
                    }
                completed += 1
                if completed % 500 == 0:
                    elapsed = time.time() - start_time
                    pct = completed/total_combinations*100
                    eta = (elapsed/completed * total_combinations - elapsed)/60
                    print(f"Progress: {completed:,}/{total_combinations:,} ({pct:.1f}%) ETA: {eta:.1f} min")
            except Exception as e:
                print(f"Worker error: {e}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed {len(all_results):,} evals in {total_time/60:.1f} min")
    
    # EXACT XGBoost analysis format
    print("\n" + "=" * 80)
    print(f"SIAMESE RESULTS - 30/70 ({'PARTIAL' if partial else 'FULL'})")
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
        print(f"\n📊 Top-{top_x}:")
        print(f"  Attacker: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"  🎭 Deception: {mimicked_deceived/total_cases*100:.1f}% | 🔍 Detection: {attacker_detected/total_cases*100:.1f}%")
    
    return all_results

# ============================================================
# EXACT SAME HEATMAP AS XGBoost/GNN
# ============================================================
def plot_false_flag_deception_siamese(results, apt_names, apt_ids, partial=False):
    """EXACT XGBoost/GNN deception heatmap"""
    print("\n📈 CREATING SIAMESE DECEPTION HEATMAP...")
    
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
        
        # Find APT IDs from names
        attacker_id = next((aid for aid, name in zip(apt_ids, apt_names) if name == attacker_name), None)
        mimicked_id = next((aid for aid, name in zip(apt_ids, apt_names) if name == mimicked_name), None)
        
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue
        
        ar = res["attacker_rank"] or n
        mr = res["mimicked_rank"] or n
        deception_matrix[i, j] = ar - mr  # >0 = deception success
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠️ No deception signal")
        return
    
    vmin, vmax = -max(np.abs(non_zero)), max(np.abs(non_zero))
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(deception_matrix, 
                xticklabels=apt_names, 
                yticklabels=apt_names,
                cmap='RdBu_r', 
                center=0, 
                vmin=vmin, 
                vmax=vmax, 
                square=True,
                cbar_kws={'label': 'Deception (Attacker Rank - Mimicked Rank)\nRed>0 Deception | Blue<0 Detection'})
    
    scenario = "PARTIAL" if partial else "FULL"
    plt.title(f'Siamese False-Flag Deception (30/70 Mix - {scenario})\nRows=Attacker | Columns=Mimicked', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT (70%)'); plt.ylabel('Actual Attacker (30%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    output_path = f'false_flag_deception_matrix_siamese_30_70_{"partial" if partial else "full"}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {output_path}")

# ============================================================
# MAIN - EXACT XGBoost structure
# ============================================================
def main():
    print("🚨 SIAMESE FALSE-FLAG EVALUATION: 30/70 MIX")
    print("=" * 70)
    
    model, X_apts, apt_ids, apt_names, apt_name_mapping, profiles = load_model_and_mappings()
    
    # FULL 30/70 (like XGBoost)
    full_results = evaluate_false_flag_attacks_siamese(model, X_apts, apt_ids, apt_names, apt_name_mapping, profiles, partial=False)
    plot_false_flag_deception_siamese(full_results, apt_names, apt_ids, partial=False)
    
    # PARTIAL 30/70 (like XGBoost partial)
    print("\n" + "="*80)
    partial_results = evaluate_false_flag_attacks_siamese(model, X_apts, apt_ids, apt_names, apt_name_mapping, profiles, partial=True)
    plot_false_flag_deception_siamese(partial_results, apt_names, apt_ids, partial=True)
    
    print("\n🎉 SIAMESE 30/70 FALSE-FLAG COMPLETE!")
    print("✅ Exact XGBoost/GNN format + structure")
    print("✅ Uses FeatureEncoder forward pass")
    print("✅ false_flag_deception_matrix_siamese_30_70_*.png")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    main()
