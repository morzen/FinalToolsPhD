"""
MIL false-flag evaluation (thesis Chapter 6, Section 6.1).

False-flag evidence: the first 30% of the true attacker's profile (ordered from high-level to
low-level features) plus the last 70% of the mimicked APT's profile. It is generated once by common/protocol.py into
BaseData/false_flag_evidence.pkl, so every method and the hybrid are scored on identical
evidence. PARTIAL keeps 50% of each feature type of the full evidence. Ranks use mid-ranks
for ties; AD / MA / Both / Neither are reported for Top-1/5/10/20.

Scoring uses exactly the same scorer as query.py (hybrid_query), loaded once per worker.

Run:  python False_Flag.py            full run (both scenarios), results in ./results/
      python False_Flag.py --smoke    40 pairs only, quick check (results in ./results/smoke/)
      python False_Flag.py --heatmaps-only   only redraw the original heatmap(s) from the saved results
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import protocol as P


def make_scorer():
    from query import MILScorer
    return MILScorer()



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


MODEL_PATH = "6feat_apt_similarity_model_complete.pth"

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


def original_heatmaps():
    """Draw the original false-flag heatmap(s) from the final saved results (no model is re-run)."""
    os.chdir(HERE)
    # APT order exactly as in evaluate_false_flag_attacks_mil_parallel(): the keys of the
    # 'apt_profiles' stored in the model file (loaded as in load_mil_model_and_data())
    try:
        with open(MODEL_PATH, 'rb') as f:
            save_data = pickle.load(f)
    except Exception:
        import torch
        save_data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    apt_ids_order = list(save_data['apt_profiles'].keys())
    full_results = P.original_ff_results("full", os.path.join(HERE, "results", "ff_ranks_full.npy"), os.path.join(HERE, "results", "false_flag_summary.json"))
    plot_false_flag_deception_mil(full_results, apt_ids_order)


if __name__ == "__main__":
    os.chdir(HERE)
    if "--heatmaps-only" not in sys.argv:
        P.run_false_flag(make_scorer, "MIL", results_dir="results", smoke=P.smoke_flag())
    if not P.smoke_flag():
        original_heatmaps()
