import os
import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as torch_mp
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "3feat_apt_similarity_model_complete.pth"
TOP_X_VALUES = [1, 5, 10, 20]
MAX_WORKERS = min(mp.cpu_count(), 8)
DEVICE = torch.device("cpu")


# ============================================================
# MODEL DEFINITION (AS IN YOUR CURRENT MIL CODE)
# ============================================================

import torch.nn as nn
import torch.nn.functional as F


class OptimizedAttentionMILSimilarity(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature embedding layers
        self.technique_embed = nn.Embedding(100000, hidden_dim)
        self.software_embed = nn.Embedding(100000, hidden_dim)
        self.cve_embed = nn.Embedding(100000, hidden_dim)
        
        # Attention networks
        self.attention_nets = nn.ModuleDict({
            'technique': self._build_attention(),
            'software': self._build_attention(),
            'cve': self._build_attention()
        })
        
        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_dim*3*2 + 6, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 3),
            nn.Sigmoid()
        )
        
        # Initialize weights with higher technique importance
        self.feature_weights = nn.Parameter(torch.tensor([6/6, 5/6, 2/6]))
        self.perfect_match_boost = 2.0
        
        self.init_weights()

    def _build_attention(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

    def init_weights(self):
        for embed in [self.technique_embed, self.software_embed, self.cve_embed]:
            nn.init.xavier_uniform_(embed.weight)
        
        for net in self.attention_nets.values():
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        for layer in self.similarity_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _calculate_match_ratios(self, apt_dict, evidence_dict):
        ratios = []
        perfect_matches = []
        feature_types = ['technique', 'software', 'cve']
        
        for feat_type in feature_types:
            dict_key = feat_type
            apt_features = set(apt_dict.get(dict_key, []))
            evidence_features = set(evidence_dict.get(dict_key, []))
            
            if len(evidence_features) == 0:
                ratios.append(0.01)
                perfect_matches.append(0)
            else:
                coverage = len(apt_features & evidence_features) / len(evidence_features)
                is_perfect = 1 if coverage == 1.0 else 0
                perfect_matches.append(is_perfect)
                ratios.append(max(coverage, 0.01))
                
        return torch.tensor(ratios), torch.tensor(perfect_matches)

    def forward(self, x1_dict, x2_dict):
        ratios_12, _ = self._calculate_match_ratios(x1_dict, x2_dict)
        ratios_21, _ = self._calculate_match_ratios(x2_dict, x1_dict)
        emb1 = self._forward_single(x1_dict, ratios_12)
        emb2 = self._forward_single(x2_dict, ratios_21)
        both_ratios = torch.cat([ratios_12, ratios_21])   # length 12
        combined = torch.cat([emb1, emb2, both_ratios.unsqueeze(0)], dim=-1)
        feature_similarities = self.similarity_net(combined)
        
        weights = F.softmax(self.feature_weights, dim=0)
        # Final score
        similarity = (feature_similarities * weights).sum(dim=-1, keepdim=True)
        return similarity

    def _forward_single(self, x_dict, match_ratios):
        features = []
        feature_types = ['technique', 'software', 'cve']
        
        for i, feat_type in enumerate(feature_types):
            dict_key = feat_type
            feat = x_dict.get(dict_key, [])
            
            if len(feat) > 0:
                embedded = getattr(self, f'{feat_type}_embed')(torch.tensor(feat, dtype=torch.long))
                attn_weights = F.softmax(self.attention_nets[feat_type](embedded), dim=0)
                attended = (embedded * attn_weights).sum(dim=0, keepdim=True)
                weight = torch.log1p(torch.tensor(len(feat))) * (match_ratios[i] + 0.1)
                features.append(attended * weight)
            else:
                features.append(torch.zeros(1, self.hidden_dim))
                
        return torch.cat(features, dim=-1)


# ============================================================
# LOADING MODEL + DATA - UPDATED FOR PICKLE FORMAT
# ============================================================

def load_mil_model_and_data(model_path=MODEL_PATH):
    print(" LOADING MIL MODEL AND DATA...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found.")
    
    # Try pickle first (since model is saved with pickle)
    import pickle
    
    try:
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        print("✓ Model loaded with pickle")
        
    except Exception as e:
        print(f"Pickle load failed: {e}")
        print("Trying torch.load as fallback...")
        
        # Fallback to torch.load
        save_data = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Initialize the model
    model = OptimizedAttentionMILSimilarity()
    model.load_state_dict(save_data['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Extract other data
    apt_profiles = save_data['apt_profiles']
    apt_name_dict = save_data['apt_name_dict']
    df_mapping = save_data['df_mapping']
    apt_ids = save_data['apt_ids']
    
    print(f" MIL model and data loaded from {model_path}")
    print(f"   APTs in profiles: {len(apt_profiles)}")
    print(f"   Available CPU cores: {mp.cpu_count()}")
    return model, apt_profiles, apt_name_dict, df_mapping, apt_ids


# ============================================================
# EVIDENCE BUILDING – GNN-STYLE 30/70 + PARTIAL
# ============================================================

def build_apt_evidence_dict_from_profile(profile):
    """
    Build x_dict expected by MIL model:
      {'technique': [...], 'software': [...], 'cve': [...]}
    """
    return {
        "technique": profile.get("technique", []),
        "software":   profile.get("software", []),
        "cve":        profile.get("cve", []),
    }

def build_false_flag_evidence_full(attacker_profile, mimicked_profile, rng=None,
                                   attacker_fraction=0.3, max_count=10):
    """
    FULL evidence:
      For each category, sample up to max_count features with 30% attacker / 70% mimicked ratio,
      just like sample_mixed_evidence in the GNN code.
    """
    if rng is None:
        rng = random

    def sample_mixed_features(atk_feats, mim_feats, max_count):
        atk_list = list(atk_feats)
        mim_list = list(mim_feats)
        
        if not atk_list and not mim_list:
            return []
        
        total_needed = min(max_count, len(atk_list) + len(mim_list))
        result = []
        
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_list:
                choice = rng.choice(atk_list)
                result.append(choice)
                atk_list.remove(choice)
            elif mim_list:
                choice = rng.choice(mim_list)
                result.append(choice)
                mim_list.remove(choice)
            else:
                break
        
        return result[:max_count]

    evidence = {
        "technique": sample_mixed_features(attacker_profile.get("technique", []),
                                            mimicked_profile.get("technique", []),
                                            max_count),
        "software":   sample_mixed_features(attacker_profile.get("software", []),
                                            mimicked_profile.get("software", []),
                                            max_count),
        "cve":        sample_mixed_features(attacker_profile.get("cve", []),
                                            mimicked_profile.get("cve", []),
                                            max_count),
    }
    return evidence

def build_false_flag_evidence_partial(attacker_profile, mimicked_profile, rng=None,
                                      attacker_fraction=0.3, max_count=10,
                                      volume_fraction=0.5):
    """
    PARTIAL evidence:
      1) Build full 30/70 mixed evidence per category (as above).
      2) Shuffle and keep ~50% per category (at least 1 if non-empty).
    """
    if rng is None:
        rng = random
    
    full_mixed = build_false_flag_evidence_full(
        attacker_profile, mimicked_profile, rng,
        attacker_fraction=attacker_fraction,
        max_count=max_count
    )
    
    partial = {}
    for key, feats in full_mixed.items():
        feats = list(feats)
        rng.shuffle(feats)
        if len(feats) == 0:
            partial[key] = []
        else:
            keep_count = max(1, int(len(feats) * volume_fraction))
            partial[key] = feats[:keep_count]
    
    return partial


# ============================================================
# SINGLE COMBINATION EVALUATION (FULL / PARTIAL)
# ============================================================

def process_single_false_flag_mil(args):
    attacker_name, mimicked_name, model_state_dict, model_class, apt_profiles, partial, seed = args
    
    rng = random.Random(seed)
    
    # Rebuild model in worker
    model = model_class()
    model.load_state_dict(model_state_dict)
    model.to("cpu")
    model.eval()
    
    attacker_profile = apt_profiles[attacker_name]
    mimicked_profile = apt_profiles[mimicked_name]
    
    # 30/70 mixed evidence, full vs partial EXACTLY like GNN
    if partial:
        false_flag_evidence = build_false_flag_evidence_partial(
            attacker_profile, mimicked_profile, rng,
            attacker_fraction=0.3, max_count=10, volume_fraction=0.5
        )
    else:
        false_flag_evidence = build_false_flag_evidence_full(
            attacker_profile, mimicked_profile, rng,
            attacker_fraction=0.3, max_count=10
        )
    
    # Pre-build candidate evidence dicts
    apt_evidence_dicts = {
        name: build_apt_evidence_dict_from_profile(apt_profiles[name])
        for name in apt_profiles.keys()
    }
    apt_ids = list(apt_profiles.keys())
    
    # Score against all APTs
    similarities = []
    with torch.no_grad():
        for name in apt_ids:
            target_evidence = apt_evidence_dicts[name]
            score = model(false_flag_evidence, target_evidence)
            similarities.append(score.item())
    
    similarities = np.array(similarities)
    sorted_idx = np.argsort(similarities)[::-1]
    
    ranked_results = {apt_ids[idx]: float(similarities[idx]) for idx in sorted_idx}
    ranked_names = list(ranked_results.keys())
    
    attacker_rank = ranked_names.index(attacker_name) + 1 if attacker_name in ranked_names else None
    mimicked_rank = ranked_names.index(mimicked_name) + 1 if mimicked_name in ranked_names else None
    
    key = f"{attacker_name}→{mimicked_name}"
    if partial:
        key += "_partial"
    
    return {
        "key": key,
        "attacker_rank": attacker_rank,
        "mimicked_rank": mimicked_rank,
        "attacker_similarity": ranked_results.get(attacker_name),
        "mimicked_similarity": ranked_results.get(mimicked_name)
    }


# ============================================================
# PARALLEL EVALUATION (FULL / PARTIAL) WITH ETA
# ============================================================

def evaluate_false_flag_attacks_mil_parallel(model, model_class, apt_profiles,
                                             top_x_values=TOP_X_VALUES, partial=False):
    print("=" * 80)
    scenario = "PARTIAL (50% volume from 30/70 mix)" if partial else "FULL (30/70 mixed evidence)"
    print(" FALSE-FLAG: 30% Attacker + 70% Mimicked features across ALL categories")
    print(f"({scenario})")
    print("=" * 80)
    
    apt_ids = list(apt_profiles.keys())
    model_state_dict = model.state_dict()
    
    all_combinations = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    total_combinations = len(all_combinations)
    
    print(f"Testing {total_combinations} false-flag combinations (30/70 mix)...")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    args_list = []
    for i, (attacker_name, mimicked_name) in enumerate(all_combinations):
        args_list.append((attacker_name, mimicked_name, model_state_dict, model_class, apt_profiles, partial, i))
    
    all_results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_false_flag_mil, args): i for i, args in enumerate(args_list)}
        
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results[result["key"]] = {
                    "attacker_rank": result["attacker_rank"],
                    "mimicked_rank": result["mimicked_rank"],
                    "attacker_similarity": result["attacker_similarity"],
                    "mimicked_similarity": result["mimicked_similarity"],
                }
                completed += 1
                
                if completed % 100 == 0 or completed == total_combinations:
                    elapsed = time.time() - start_time
                    est_total = (elapsed / completed) * total_combinations if completed > 0 else 0
                    remaining = max(0, est_total - elapsed)
                    print(f"Progress: {completed}/{total_combinations} "
                          f"({completed/total_combinations*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} minutes")
            except Exception as e:
                print(f"Worker error: {e}")
    
    total_time = time.time() - start_time
    print(f"\n Completed {len(all_results)} evaluations in {total_time/60:.1f} minutes")
    
    # Top-k analysis (same structure as GNN)
    print("\n" + "=" * 80)
    print(f"FALSE-FLAG RESULTS - MIL 30/70 MIX ({'PARTIAL' if partial else 'FULL'})")
    print("=" * 80)
    
    for top_x in top_x_values:
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
        print(f"  Actual attacker: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked APT: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
    
    return all_results, apt_ids


# ============================================================
# CONTINUOUS DECEPTION HEATMAP (MIL) – MATCHING GNN
# ============================================================

def plot_false_flag_deception_mil(results, apt_ids, output_path="false_flag_deception_matrix_mil_30_70_continuous.png"):
    """
    MIL deception heatmap analogous to GNN:
      deception_score = attacker_rank - mimicked_rank
      >0: mimicked ranked better (deception success)
      <0: attacker ranked better (detection success)
    """
    print("\n CREATING FALSE-FLAG DECEPTION HEATMAP (MIL 30/70, CONTINUOUS)...")
    
    n = len(apt_ids)
    apt_names = apt_ids
    id_to_idx = {name: i for i, name in enumerate(apt_ids)}
    
    deception_matrix = np.zeros((n, n), dtype=float)
    
    for key, res in results.items():
        if "_partial" in key.lower():
            continue
        
        try:
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()
        except Exception:
            continue
        
        i = id_to_idx.get(attacker_name)
        j = id_to_idx.get(mimicked_name)
        if i is None or j is None:
            continue
        
        ar = res["attacker_rank"]
        mr = res["mimicked_rank"]
        a_r = ar if ar is not None else n
        m_r = mr if mr is not None else n
        
        deception_matrix[i, j] = a_r - m_r  # >0 => mimicked better
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠ All deception scores are zero; nothing to plot.")
        return
    
    low = np.percentile(non_zero, 5)
    high = np.percentile(non_zero, 95)
    max_abs = max(abs(low), abs(high))
    vmin, vmax = -max_abs, max_abs
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        deception_matrix,
        xticklabels=apt_names,
        yticklabels=apt_names,
        cmap='RdBu_r',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={
            'label': 'Deception Score (Attacker Rank - Mimicked Rank)\n'
                     '>0: Deception success, <0: Detection success'
        }
    )
    
    plt.title(
        'MIL False-Flag Deception (30/70 Mixed Evidence, Continuous Score)\n'
        'Rows=Actual Attacker | Columns=Mimicked APT',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Saved: {output_path}")


# ============================================================
# MAIN – PARALLEL TO GNN SCRIPT
# ============================================================

def main():
    print(" MIL FALSE-FLAG EVALUATION: 30/70 MIXED EVIDENCE")
    print("=" * 70)
    print("Optimized Attention MIL + cosine-like similarity")
    print("30% attacker + 70% mimicked features ACROSS ALL CATEGORIES")
    print("=" * 70)
    
    model, apt_profiles, apt_name_dict, df_mapping, apt_ids = load_mil_model_and_data()
    
    # FULL 30/70 evidence
    full_results, apt_ids_order = evaluate_false_flag_attacks_mil_parallel(
        model, model.__class__, apt_profiles, partial=False
    )
    
    
    # PARTIAL 30/70 + volume reduction
    partial_results, _ = evaluate_false_flag_attacks_mil_parallel(
        model, model.__class__, apt_profiles, partial=True
    )
    

    plot_false_flag_deception_mil(full_results, apt_ids_order)
    print("\n MIL 30/70 FALSE-FLAG EVALUATION COMPLETE!")
    print("Saved: false_flag_deception_matrix_mil_30_70_continuous.png")


if __name__ == "__main__":
    torch_mp.set_start_method("spawn", force=True)
    main()

