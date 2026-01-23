import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import warnings
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

class APTGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.normalize(x, p=2, dim=-1)

def load_trained_gnn_model(path="trained_apt_gnn_model.pkl"):
    print("🔍 LOADING ENHANCED TRAINED GNN MODEL...")
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    model = APTGNN(saved["model_config"]["in_channels"], 
                   saved["model_config"]["hidden_dim"], 
                   saved["model_config"]["out_dim"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    print(f"✅ Loaded {path}")
    print(f"✅ GNN model loaded successfully!")
    print(f"   APTs: {len(saved['apt_ids'])}")
    print(f"   Embedding dimension: {saved['apt_embeddings'].shape[1]}")
    print(f"   Feature weights: TTP×{saved['feature_weights']['techniques']:.1f}, "
          f"SW×{saved['feature_weights']['software']:.1f}, CAPEC×{saved['feature_weights']['capec']:.1f}, CWE×{saved['feature_weights']['cwe']:.1f}, CVE×{saved['feature_weights']['cve']:.1f}, Pulse×{saved['feature_weights']['pulse']:.1f}")
    print(f"   Available CPU cores: {mp.cpu_count()}")
    print("   Model uses: Pure GNN cosine similarity on 30/70 mixed evidence")
    return model, saved

def load_apt_profiles_from_data():
    print("📊 LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].dropna().unique()),
            'software': set(apt_data['software_str'].dropna().unique()), 
            'capec': set(apt_data['CAPEC_ID'].dropna().unique()),
            'cwe': set(apt_data['CWE_ID'].dropna().unique()),
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
            'pulse': set(apt_data['Pulse_ID'].dropna().unique()),
        }
    print(f"✅ Loaded profiles for {len(profiles)} APTs")
    return profiles

def load_mappings():
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row["APTnameEncoded"]
        original_name = row["APTnameOrigin"]
        if encoded_id not in apt_name_mapping:
            apt_name_mapping[encoded_id] = original_name
    return apt_name_mapping

def sample_mixed_evidence(attacker_profile, mimicked_profile, rng=None, attacker_fraction=0.3, max_count=10):
    """
    Sample 30% attacker / 70% mimicked evidence for each feature type
    Returns dict with same structure as original create_false_flag_evidence
    """
    if rng is None:
        rng = random

    def sample_mixed_features(atk_feats, mim_feats, max_count):
        """Sample up to max_count features with 30/70 attacker/mimicked ratio"""
        atk_list = list(atk_feats)
        mim_list = list(mim_feats)
        
        if not atk_list and not mim_list:
            return []
        
        total_needed = min(max_count, len(atk_list) + len(mim_list))
        result = []
        
        # Fill with 30/70 ratio until max_count
        while len(result) < total_needed:
            use_attacker = (rng.random() < attacker_fraction)
            if use_attacker and atk_list:
                result.append(rng.choice(atk_list))
                atk_list.pop(atk_list.index(result[-1]))  # remove to avoid duplicates
            elif mim_list:
                result.append(rng.choice(mim_list))
                mim_list.pop(mim_list.index(result[-1]))
            else:
                break
        
        return result[:max_count]

    return {
        'techniques': sample_mixed_features(attacker_profile['techniques'], mimicked_profile['techniques'], max_count),
        'software': sample_mixed_features(attacker_profile['software'], mimicked_profile['software'], max_count),
        'capec': sample_mixed_features(attacker_profile['capec'], mimicked_profile['capec'], max_count),
        'cwe': sample_mixed_features(attacker_profile['cwe'], mimicked_profile['cwe'], max_count),
        'cve': sample_mixed_features(attacker_profile['cve'], mimicked_profile['cve'], max_count),
        'pulse': sample_mixed_features(attacker_profile['pulse'], mimicked_profile['pulse'], max_count),
    }

def create_partial_mixed_evidence(attacker_profile, mimicked_profile, rng=None):
    """Create partial evidence from 30/70 mixed base (50% volume reduction)"""
    if rng is None:
        rng = random
    
    full_mixed = sample_mixed_evidence(attacker_profile, mimicked_profile, rng, attacker_fraction=0.3, max_count=10)
    
    # Reduce volume by ~50% per category
    partial = {}
    for key in full_mixed:
        feats = full_mixed[key]
        rng.shuffle(feats)
        keep_count = max(1, len(feats) // 2)
        partial[key] = feats[:keep_count]
    
    return partial

def build_temp_graph_for_evidence(evidence, graph_info):
    num_apts = graph_info['num_apts']
    technique_offset = graph_info['technique_offset']
    software_offset = graph_info['software_offset'] 
    capec_offset = graph_info['capec_offset']
    cwe_offset = graph_info['cwe_offset']
    cve_offset = graph_info['cve_offset']
    pulse_offset = graph_info['pulse_offset']
    
    total_nodes = num_apts + graph_info['num_techniques'] + graph_info['num_software'] + graph_info['num_capec'] + graph_info['num_cwe'] + graph_info['num_cve'] + graph_info['num_pulse']
    
    edge_list = []
    edge_weights = []
    fw = graph_info['feature_weights']
    
    temp_apt_idx = 0
    
    for technique in evidence['techniques']:
        if technique in graph_info['technique_to_idx']:
            tech_idx = graph_info['technique_to_idx'][technique] + technique_offset
            edge_list.extend([[temp_apt_idx, tech_idx], [tech_idx, temp_apt_idx]])
            edge_weights.extend([fw['techniques'], fw['techniques']])
    
    for software in evidence['software']:
        if software in graph_info['software_to_idx']:
            soft_idx = graph_info['software_to_idx'][software] + software_offset
            edge_list.extend([[temp_apt_idx, soft_idx], [soft_idx, temp_apt_idx]])
            edge_weights.extend([fw['software'], fw['software']])
    
    for capec in evidence['capec']:
        if capec in graph_info['capec_to_idx']:
            capec_idx = graph_info['capec_to_idx'][capec] + capec_offset
            edge_list.extend([[temp_apt_idx, capec_idx], [capec_idx, temp_apt_idx]])
            edge_weights.extend([fw['capec'], fw['capec']])

    for cwe in evidence['cwe']:
        if cwe in graph_info['cwe_to_idx']:
            cwe_idx = graph_info['cwe_to_idx'][cwe] + cwe_offset
            edge_list.extend([[temp_apt_idx, cwe_idx], [cwe_idx, temp_apt_idx]])
            edge_weights.extend([fw['cwe'], fw['cwe']])

    for cve in evidence['cve']:
        if cve in graph_info['cve_to_idx']:
            cve_idx = graph_info['cve_to_idx'][cve] + cve_offset
            edge_list.extend([[temp_apt_idx, cve_idx], [cve_idx, temp_apt_idx]])
            edge_weights.extend([fw['cve'], fw['cve']])

    for pulse in evidence['pulse']:
        if pulse in graph_info['pulse_to_idx']:
            pulse_idx = graph_info['pulse_to_idx'][pulse] + pulse_offset
            edge_list.extend([[temp_apt_idx, pulse_idx], [pulse_idx, temp_apt_idx]])
            edge_weights.extend([fw['pulse'], fw['pulse']])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
    
    x = torch.ones(total_nodes, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

def process_single_false_flag_gnn(args):
    attacker_apt_id, mimicked_apt_id, model, saved_data, profiles, apt_ids, \
    apt_name_mapping, graph_info, device, partial, seed = args
    
    rng = random.Random(seed)  # Reproducible randomness
    
    attacker_profile = profiles[attacker_apt_id]
    mimicked_profile = profiles[mimicked_apt_id]
    
    # 30/70 MIXED EVIDENCE
    if partial:
        evidence = create_partial_mixed_evidence(attacker_profile, mimicked_profile, rng)
    else:
        evidence = sample_mixed_evidence(attacker_profile, mimicked_profile, rng, attacker_fraction=0.3)
    
    with torch.no_grad():
        temp_data = build_temp_graph_for_evidence(evidence, graph_info)
        temp_data = temp_data.to(device)
        evidence_emb = model(temp_data.x, temp_data.edge_index, temp_data.edge_weight)[0]
    
    apt_embeddings = saved_data['apt_embeddings'].to(device)
    similarities = torch.matmul(evidence_emb.unsqueeze(0), apt_embeddings.t()).squeeze()
    
    ranked_indices = torch.argsort(similarities, descending=True)
    ranked_results = {apt_ids[idx.item()]: similarities[idx.item()].item() for idx in ranked_indices}
    
    attacker_rank = None
    mimicked_rank = None
    if attacker_apt_id in ranked_results:
        attacker_rank = list(ranked_results.keys()).index(attacker_apt_id) + 1
    if mimicked_apt_id in ranked_results:
        mimicked_rank = list(ranked_results.keys()).index(mimicked_apt_id) + 1
    
    key = f"{apt_name_mapping.get(attacker_apt_id, attacker_apt_id)}→{apt_name_mapping.get(mimicked_apt_id, mimicked_apt_id)}"
    if partial:
        key += "_partial"
    
    return (key, attacker_rank, mimicked_rank, evidence)  # Return evidence for debugging

def evaluate_false_flag_attacks_gnn_parallel(model, saved_data, profiles, partial=False):
    print("=" * 80)
    scenario = "PARTIAL (50% volume from 30/70 mix)" if partial else "FULL (30/70 mixed evidence)"
    print(f"🎯 FALSE-FLAG: 30% Attacker + 70% Mimicked features across ALL categories")
    print(f"({scenario})")
    print("=" * 80)
    
    apt_ids = saved_data['apt_ids']
    apt_name_mapping = load_mappings()
    graph_info = saved_data['graph_info']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    total_combinations = 0
    args_list = []
    for i, (attacker_id, mimicked_id) in enumerate([(a, m) for a in apt_ids for m in apt_ids if a != m]):
        args_list.append((attacker_id, mimicked_id, model, saved_data, 
                         profiles, apt_ids, apt_name_mapping, graph_info, device, partial, i))
        total_combinations += 1
    
    print(f"Testing {total_combinations} false-flag combinations (30/70 mix)...")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    all_results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
        future_to_combination = {executor.submit(process_single_false_flag_gnn, args): i 
                               for i, args in enumerate(args_list)}
        
        completed = 0
        for future in as_completed(future_to_combination):
            try:
                result = future.result()
                all_results[result[0]] = result[1:3]  # (key, attacker_rank, mimicked_rank)
                completed += 1
                
                if completed % 100 == 0 or completed == total_combinations:
                    elapsed_time = time.time() - start_time
                    estimated_total = (elapsed_time / completed) * total_combinations if completed > 0 else 0
                    remaining = max(0, estimated_total - elapsed_time)
                    print(f"Progress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%) "
                          f"ETA: {remaining/60:.1f} minutes")
            except Exception as e:
                print(f"Worker error: {e}")
                continue
    
    total_time = time.time() - start_time
    print(f"\n✅ Completed {len(all_results)} evaluations in {total_time/60:.1f} minutes")
    return all_results

def analyze_results(results, scenario="FULL"):
    print("\n" + "=" * 80)
    print(f"FALSE-FLAG RESULTS - GNN 30/70 MIX ({scenario})")
    print("=" * 80)
    
    topx_values = [1, 3, 5, 10, 20]
    for top_x in topx_values:
        attacker_detected = 0
        mimicked_deceived = 0
        both_correct = 0
        neither_correct = 0
        
        for result in results.values():
            attacker_rank, mimicked_rank = result
            attacker_in_top = attacker_rank is not None and attacker_rank <= top_x
            mimicked_in_top = mimicked_rank is not None and mimicked_rank <= top_x
            
            if attacker_in_top: attacker_detected += 1
            if mimicked_in_top: mimicked_deceived += 1
            if attacker_in_top and mimicked_in_top: both_correct += 1
            if not attacker_in_top and not mimicked_in_top: neither_correct += 1
        
        total_cases = len(results)
        print(f"\n📊 Top-{top_x} Analysis:")
        print(f"  Actual attacker: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked APT: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"  🎭 Deception: {mimicked_deceived/total_cases*100:.1f}% | 🔍 Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖️ Net Advantage: {(mimicked_deceived - attacker_detected)/total_cases*100:+.1f}%")

def plot_false_flag_deception_gnn(results, apt_name_mapping, saved_data):
    """
    GNN deception heatmap with continuous deception score:
    deception_score = attacker_rank - mimicked_rank

    Interpretation (rank 1 is best):
      > 0 : mimicked ranked better (smaller rank)  -> successful deception
      < 0 : attacker ranked better                -> successful detection
      = 0 : tie or equal effectiveness
    """
    print("\n📈 CREATING FALSE-FLAG DECEPTION HEATMAP (GNN 30/70, CONTINUOUS)...")

    apt_ids = saved_data['apt_ids']
    apt_names = [apt_name_mapping.get(a, f"APT{a}") for a in apt_ids]
    n = len(apt_ids)

    # Continuous matrix: attacker_rank - mimicked_rank
    deception_matrix = np.zeros((n, n), dtype=float)
    id_to_idx = {apt_id: i for i, apt_id in enumerate(apt_ids)}

    for key, (attacker_rank, mimicked_rank) in results.items():
        if "_partial" in key.lower():
            continue

        try:
            # Parse name-based keys from process_single_false_flag_gnn
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()

            # Find corresponding APT IDs
            attacker_id = next((aid for aid, name in apt_name_mapping.items() if name == attacker_name), None)
            mimicked_id = next((aid for aid, name in apt_name_mapping.items() if name == mimicked_name), None)
        except Exception:
            continue

        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue

        # Substitute n for missing ranks so "not in list" behaves like very bad rank
        a_r = attacker_rank if attacker_rank is not None else n
        m_r = mimicked_rank if mimicked_rank is not None else n

        # Positive => mimicked better (smaller rank) => deception success
        # Negative => attacker better => detection success
        deception_matrix[i, j] = a_r - m_r

    # If everything is zero, nothing to plot
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠️ All deception scores are zero; nothing to plot.")
        return

    # Robust symmetric color scaling (5th–95th percentile)
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
        'GNN False-Flag Deception (30/70 Mixed Evidence, Continuous Score)\n'
        '(Positive = Mimicked ranked higher than Attacker)',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    out_name = 'false_flag_deception_matrix_gnn_30_70_continuous.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {out_name}")



if __name__ == "__main__":
    print("🚨 GNN FALSE-FLAG EVALUATION: 30/70 MIXED EVIDENCE")
    print("=" * 70)
    print("Pure GNN embeddings + cosine similarity")
    print("30% attacker + 70% mimicked features ACROSS ALL CATEGORIES")
    print("=" * 70)
    
    model, saved_data = load_trained_gnn_model()
    profiles = load_apt_profiles_from_data()
    
    # FULL 30/70 evidence
    full_results = evaluate_false_flag_attacks_gnn_parallel(model, saved_data, profiles, partial=False)
    analyze_results(full_results, "FULL 30/70")
    plot_false_flag_deception_gnn(full_results, load_mappings(), saved_data)
    
    # PARTIAL 30/70 + volume reduction
    partial_results = evaluate_false_flag_attacks_gnn_parallel(model, saved_data, profiles, partial=True)
    analyze_results(partial_results, "PARTIAL 30/70")
    
    print("\n🎉 GNN 30/70 FALSE-FLAG EVALUATION COMPLETE!")
    print("Saved: false_flag_deception_matrix_gnn_30_70.png")
