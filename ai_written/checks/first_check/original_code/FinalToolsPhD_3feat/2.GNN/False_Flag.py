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
    def __init__(self, in_channels, hidden_dim=256, out_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.normalize(x, p=2, dim=-1)

def load_trained_gnn_model(path=None):
    path = path or MODEL_PATH
    print(" LOADING ENHANCED TRAINED GNN MODEL...")
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    model = APTGNN(saved["model_config"]["in_channels"], 
                   saved["model_config"]["hidden_dim"], 
                   saved["model_config"]["out_dim"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    print(f" Loaded {path}")
    print(f" GNN model loaded successfully!")
    print(f"   APTs: {len(saved['apt_ids'])}")
    print(f"   Embedding dimension: {saved['apt_embeddings'].shape[1]}")
    print(f"   Feature weights: TTP×{saved['feature_weights']['techniques']:.1f}, "
          f"SW×{saved['feature_weights']['software']:.1f}, CVE×{saved['feature_weights']['cve']:.1f}")
    print(f"   Available CPU cores: {mp.cpu_count()}")
    print("   Model uses: Pure GNN cosine similarity on 30/70 mixed evidence")
    return model, saved

def load_apt_profiles_from_data():
    print(" LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].dropna().unique()),
            'software': set(apt_data['software_str'].dropna().unique()),
            'cve': set(apt_data['CVE_ID'].dropna().unique()),
        }
    print(f" Loaded profiles for {len(profiles)} APTs")
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
        'cve': sample_mixed_features(attacker_profile['cve'], mimicked_profile['cve'], max_count),
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

# ============================================================
# FIX (2026-10-22): score the evidence as a NEW node added to the FULL
# training graph -- the same method as eval.py, query.py and Appendix 8.3.
#
# The previous version built a separate graph containing ONLY the evidence
# and its own edges. Because every node's input is 1 (x = ones), the GCN could
# then only see HOW MANY features of each type the evidence contained, not
# WHICH ones, so the attacker was detected at chance level.
#
# Evidence sampling and seeds are unchanged, so this run uses exactly the same
# 30/70 evidence sets as the previous one; only the scoring differs.
# ============================================================

MODEL_PATH = "3feat_trained_apt_gnn_model.pkl"
TEST_PAIRS = None      # e.g. 300 for a quick sanity check; None = all pairs

# evidence key -> prefix used in graph_info ('technique_to_idx', 'technique_offset', ...)
GRAPH_PREFIX = {'techniques': 'technique', 'software': 'software', 'capec': 'capec',
                'cwe': 'cwe', 'cve': 'cve', 'pulse': 'pulse'}

_WORKER = {}


def _init_worker(model_path, profiles, apt_name_mapping):
    """Runs once per worker process: load model + full graph a single time."""
    torch.set_num_threads(1)
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    cfg = saved["model_config"]
    model = APTGNN(cfg["in_channels"], cfg["hidden_dim"], cfg["out_dim"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    gi = saved['graph_info']
    ew = saved['edge_weight']
    _WORKER.update(
        model=model,
        x=saved['x'].float().cpu(),
        edge_index=saved['edge_index'].long().cpu(),
        edge_weight=None if ew is None else ew.float().cpu(),
        graph_info=gi,
        num_apts=gi['num_apts'],
        apt_ids=list(saved['apt_ids']),
        profiles=profiles,
        apt_name_mapping=apt_name_mapping,
        feature_types=[ft for ft, p in GRAPH_PREFIX.items() if f"{p}_to_idx" in gi],
    )


def score_evidence_full_graph(evidence):
    """Cosine similarity between the evidence node and every APT node,
    after adding the evidence as a new node to the full training graph."""
    W = _WORKER
    gi = W['graph_info']
    fw = gi['feature_weights']
    q = W['x'].shape[0]                       # index of the new evidence node

    new_edges, new_weights = [], []
    for ft in W['feature_types']:
        to_idx = gi[f"{GRAPH_PREFIX[ft]}_to_idx"]
        offset = gi[f"{GRAPH_PREFIX[ft]}_offset"]
        for feat in evidence.get(ft, []):
            if feat in to_idx:
                node = to_idx[feat] + offset
                new_edges += [[q, node], [node, q]]
                new_weights += [fw[ft], fw[ft]]

    x = torch.ones(q + 1, W['x'].shape[1])
    x[:q] = W['x']
    edge_index, edge_weight = W['edge_index'], W['edge_weight']
    if new_edges:
        edge_index = torch.cat([edge_index, torch.tensor(new_edges, dtype=torch.long).t().contiguous()], dim=1)
        nw = torch.tensor(new_weights, dtype=torch.float)
        edge_weight = nw if edge_weight is None else torch.cat([edge_weight, nw])

    with torch.no_grad():
        out = W['model'](x, edge_index, edge_weight)
        sims = F.cosine_similarity(out[q].unsqueeze(0), out[:W['num_apts']])
    return sims


def process_single_false_flag_gnn(args):
    attacker_apt_id, mimicked_apt_id, partial, seed = args
    W = _WORKER
    rng = random.Random(seed)  # Reproducible randomness (same seeds as before)

    attacker_profile = W['profiles'][attacker_apt_id]
    mimicked_profile = W['profiles'][mimicked_apt_id]

    # 30/70 MIXED EVIDENCE (unchanged)
    if partial:
        evidence = create_partial_mixed_evidence(attacker_profile, mimicked_profile, rng)
    else:
        evidence = sample_mixed_evidence(attacker_profile, mimicked_profile, rng, attacker_fraction=0.3)

    sims = score_evidence_full_graph(evidence)
    order = torch.argsort(sims, descending=True).tolist()
    ranked_ids = [W['apt_ids'][i] for i in order]

    attacker_rank = ranked_ids.index(attacker_apt_id) + 1 if attacker_apt_id in ranked_ids else None
    mimicked_rank = ranked_ids.index(mimicked_apt_id) + 1 if mimicked_apt_id in ranked_ids else None

    names = W['apt_name_mapping']
    key = f"{names.get(attacker_apt_id, attacker_apt_id)}→{names.get(mimicked_apt_id, mimicked_apt_id)}"
    if partial:
        key += "_partial"

    return (key, attacker_rank, mimicked_rank, evidence)  # Return evidence for debugging


def evaluate_false_flag_attacks_gnn_parallel(model, saved_data, profiles, partial=False):
    print("=" * 80)
    scenario = "PARTIAL (50% volume from 30/70 mix)" if partial else "FULL (30/70 mixed evidence)"
    print(f" FALSE-FLAG: 30% Attacker + 70% Mimicked features across ALL categories")
    print(f"({scenario})")
    print(" Scoring: evidence added as a new node to the FULL training graph")
    print("=" * 80)

    apt_ids = saved_data['apt_ids']
    apt_name_mapping = load_mappings()

    pairs = [(a, m) for a in apt_ids for m in apt_ids if a != m]
    if TEST_PAIRS:
        pairs = pairs[:TEST_PAIRS]
        print(f" TEST MODE: only the first {TEST_PAIRS} pairs")
    args_list = [(a, m, partial, i) for i, (a, m) in enumerate(pairs)]
    total_combinations = len(args_list)

    n_workers = min(mp.cpu_count(), 8)
    print(f"Testing {total_combinations} false-flag combinations (30/70 mix)...")
    print(f"Using {n_workers} worker processes")

    all_results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(MODEL_PATH, profiles, apt_name_mapping)) as executor:
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
    print(f"\n Completed {len(all_results)} evaluations in {total_time/60:.1f} minutes")
    if len(all_results) != total_combinations:
        print(f" WARNING: {total_combinations - len(all_results)} pairs failed -- check the worker errors above")
    return all_results

def analyze_results(results, scenario="FULL"):
    print("\n" + "=" * 80)
    print(f"FALSE-FLAG RESULTS - GNN 30/70 MIX ({scenario})")
    print("=" * 80)
    
    topx_values = [1, 5, 10, 20]
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
        print(f"\n Top-{top_x} Analysis:")
        print(f"  Actual attacker: {attacker_detected}/{total_cases} ({attacker_detected/total_cases*100:.1f}%)")
        print(f"  Mimicked APT: {mimicked_deceived}/{total_cases} ({mimicked_deceived/total_cases*100:.1f}%)")
        print(f"  Both: {both_correct}/{total_cases} ({both_correct/total_cases*100:.1f}%)")
        print(f"  Neither: {neither_correct}/{total_cases} ({neither_correct/total_cases*100:.1f}%)")
        print(f"   Deception: {mimicked_deceived/total_cases*100:.1f}% |  Detection: {attacker_detected/total_cases*100:.1f}%")
        print(f"  ⚖ Net Advantage: {(mimicked_deceived - attacker_detected)/total_cases*100:+.1f}%")

def plot_false_flag_deception_gnn(results, apt_name_mapping, saved_data):
    """
    GNN deception heatmap with continuous deception score:
    deception_score = attacker_rank - mimicked_rank

    Interpretation (rank 1 is best):
      > 0 : mimicked ranked better (smaller rank)  -> successful deception
      < 0 : attacker ranked better                -> successful detection
      = 0 : tie or equal effectiveness
    """
    print("\n CREATING FALSE-FLAG DECEPTION HEATMAP (GNN 30/70, CONTINUOUS)...")

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
        print("⚠ All deception scores are zero; nothing to plot.")
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
    print(f" Saved: {out_name}")



if __name__ == "__main__":
    print(" GNN FALSE-FLAG EVALUATION: 30/70 MIXED EVIDENCE")
    print("=" * 70)
    print("Pure GNN embeddings + cosine similarity")
    print("30% attacker + 70% mimicked features ACROSS ALL CATEGORIES")
    print("=" * 70)
    
    model, saved_data = load_trained_gnn_model()
    profiles = load_apt_profiles_from_data()
    
    # FULL 30/70 evidence
    full_results = evaluate_false_flag_attacks_gnn_parallel(model, saved_data, profiles, partial=False)
    analyze_results(full_results, "FULL 30/70")
    
    
    # PARTIAL 30/70 + volume reduction
    partial_results = evaluate_false_flag_attacks_gnn_parallel(model, saved_data, profiles, partial=True)
    analyze_results(partial_results, "PARTIAL 30/70")
    
    plot_false_flag_deception_gnn(full_results, load_mappings(), saved_data)
    print("\n GNN 30/70 FALSE-FLAG EVALUATION COMPLETE!")
    print("Saved: false_flag_deception_matrix_gnn_30_70.png")