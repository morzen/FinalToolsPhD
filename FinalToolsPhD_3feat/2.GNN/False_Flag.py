"""
GNN false-flag evaluation (thesis Chapter 6, Section 6.1).

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
    from query import GNNScorer
    return GNNScorer()



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


def load_mappings():
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        encoded_id = row["APTnameEncoded"]
        original_name = row["APTnameOrigin"]
        if encoded_id not in apt_name_mapping:
            apt_name_mapping[encoded_id] = original_name
    return apt_name_mapping

MODEL_PATH = "3feat_trained_apt_gnn_model.pkl"

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


def original_heatmaps():
    """Draw the original false-flag heatmap(s) from the final saved results (no model is re-run)."""
    os.chdir(HERE)
    # saved_data = the trained model file load_trained_gnn_model() read (the heatmap uses its 'apt_ids')
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    full_results = P.original_ff_results("full", os.path.join(HERE, "results", "ff_ranks_full.npy"), os.path.join(HERE, "results", "false_flag_summary.json"), as_tuple=True)
    plot_false_flag_deception_gnn(full_results, load_mappings(), saved_data)


if __name__ == "__main__":
    os.chdir(HERE)
    if "--heatmaps-only" not in sys.argv:
        P.run_false_flag(make_scorer, "GNN", results_dir="results", smoke=P.smoke_flag())
    if not P.smoke_flag():
        original_heatmaps()
