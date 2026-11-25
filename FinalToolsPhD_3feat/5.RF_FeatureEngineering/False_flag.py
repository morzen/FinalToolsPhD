"""
Random Forest false-flag evaluation (thesis Chapter 6, Section 6.1).

False-flag evidence: the first 30% of the true attacker's profile (ordered from high-level to
low-level features) plus the last 70% of the mimicked APT's profile. It is generated once by common/protocol.py into
BaseData/false_flag_evidence.pkl, so every method and the hybrid are scored on identical
evidence. PARTIAL keeps 50% of each feature type of the full evidence. Ranks use mid-ranks
for ties; AD / MA / Both / Neither are reported for Top-1/5/10/20.

Scoring uses exactly the same scorer as query.py (hybrid_query), loaded once per worker.

Run:  python False_flag.py            full run (both scenarios), results in ./results/
      python False_flag.py --smoke    40 pairs only, quick check (results in ./results/smoke/)
      python False_flag.py --heatmaps-only   only redraw the original heatmap(s) from the saved results
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import protocol as P


def make_scorer():
    from query import RFScorer
    return RFScorer()



# =====================================================================================
# HEATMAP CODE OF THE FIRST IMPLEMENTATION, kept unchanged.
# The functions below are unchanged. Only original_heatmaps() is new: it gives them the
# FINAL false-flag results (the ranks saved in results/ff_ranks_*.npy by the run above),
# in exactly the format the original script built, then calls them as its main() did.
# =====================================================================================
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # draw to file only (plt.show() then opens no window)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

MODEL_PATH = "3feat_apt_similarity_model.joblib"

def load_efficient_rf_model(path=MODEL_PATH):
    """
    Load your RF similarity joblib model.
    Expects keys from your training script:
      - 'rf_model'
      - 'binarizers'
      - 'all_features'
      - 'apt_profiles' (APT name -> {encoded_id, techniques, software, cve})
      - 'apt_names'
      - 'apt_name_dict'
      - 'X'
      - 'feature_types' (['techniques','software','cve'])
      - 'df_mapping'
    """
    print(" LOADING EFFICIENT RF MODEL...")
    if not os.path.exists(path):
        print(f" {path} not found! Please run your RF training script first.")
        return None
    
    model_data = joblib.load(path)
    
    print(f" Loaded {path}")
    print(f"   APTs: {len(model_data['apt_names'])}")
    print(f"   Feature types: {model_data['feature_types']}")
    print(f"   Feature matrix shape: {model_data['X'].shape}")
    print(f"   Available CPU cores: {mp.cpu_count()}")
    
    return model_data

def plot_false_flag_deception_rf(results, apt_name_mapping, apt_ids,
                                 output_path="false_flag_deception_matrix_rf_30_70_continuous.png"):
    """
    RF deception heatmap:
      deception_score = attacker_rank - mimicked_rank
      >0: mimicked ranked better (deception success)
      <0: attacker ranked better (detection success)
    """
    print("\n CREATING FALSE-FLAG DECEPTION HEATMAP (RF 30/70, CONTINUOUS)...")
    
    n = len(apt_ids)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    id_to_idx = {aid: i for i, aid in enumerate(apt_ids)}
    
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
        
        attacker_id = next((aid for aid, nm in apt_name_mapping.items() if nm == attacker_name), None)
        mimicked_id = next((aid for aid, nm in apt_name_mapping.items() if nm == mimicked_name), None)
        if attacker_id is None or mimicked_id is None:
            continue
        
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
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
        'RF False-Flag Deception (30/70 Mixed Evidence, Continuous Score)\n'
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


def original_heatmaps():
    """Draw the original false-flag heatmap(s) from the final saved results (no model is re-run)."""
    os.chdir(HERE)
    model_data = load_efficient_rf_model(MODEL_PATH)
    # APT order and names exactly as in evaluate_false_flag_attacks_rf_parallel():
    apt_ids = [model_data['apt_profiles'][name]['encoded_id'] for name in model_data['apt_names']]
    apt_name_mapping = model_data['apt_name_dict']
    full_results = P.original_ff_results("full", os.path.join(HERE, "results", "ff_ranks_full.npy"), os.path.join(HERE, "results", "false_flag_summary.json"))
    plot_false_flag_deception_rf(full_results, apt_name_mapping, apt_ids)


if __name__ == "__main__":
    os.chdir(HERE)
    if "--heatmaps-only" not in sys.argv:
        P.run_false_flag(make_scorer, "Random Forest", results_dir="results", smoke=P.smoke_flag())
    if not P.smoke_flag():
        original_heatmaps()
