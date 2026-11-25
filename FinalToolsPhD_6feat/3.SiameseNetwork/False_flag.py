"""
Siamese Network false-flag evaluation (thesis Chapter 6, Section 6.1).

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
    from query import SiameseScorer
    return SiameseScorer()



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


MODEL_PATH = "6feat_apt_siamese_feature_model.pkl"

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


def original_heatmaps():
    """Draw the original false-flag heatmap(s) from the final saved results (no model is re-run)."""
    os.chdir(HERE)
    plt.rcParams['font.family'] = 'DejaVu Sans'          # as in the original __main__
    # apt_ids / apt_names exactly as in load_model_and_embeddings():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    apt_ids = data['apt_ids']
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {int(row['APTnameEncoded']): row['APTnameOrigin'] for _, row in df_mapping.iterrows()}
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    full_results = P.original_ff_results("full", os.path.join(HERE, "results", "ff_ranks_full.npy"), os.path.join(HERE, "results", "false_flag_summary.json"))
    partial_results = P.original_ff_results("partial", os.path.join(HERE, "results", "ff_ranks_partial.npy"), os.path.join(HERE, "results", "false_flag_summary.json"))
    plot_false_flag_deception_siamese(full_results, apt_names, apt_ids, partial=False)
    plot_false_flag_deception_siamese(partial_results, apt_names, apt_ids, partial=True)


if __name__ == "__main__":
    os.chdir(HERE)
    if "--heatmaps-only" not in sys.argv:
        P.run_false_flag(make_scorer, "Siamese Network", results_dir="results", smoke=P.smoke_flag())
    if not P.smoke_flag():
        original_heatmaps()
