"""
TVD false-flag evaluation (thesis Chapter 6, Section 6.1).

False-flag evidence: the first 30% of the true attacker's profile (ordered from high-level to
low-level features) plus the last 70% of the mimicked APT's profile. It is generated once by common/protocol.py into
BaseData/false_flag_evidence.pkl, so every method and the hybrid are scored on identical
evidence. PARTIAL keeps 50% of each feature type of the full evidence. Ranks use mid-ranks
for ties; AD / MA / Both / Neither are reported for Top-1/5/10/20.

Scoring uses exactly the same scorer as TVD_featureIng.py (hybrid_query), loaded once per worker.

Run:  python false_flag.py            full run (both scenarios), results in ./results/
      python false_flag.py --smoke    40 pairs only, quick check (results in ./results/smoke/)
      python false_flag.py --heatmaps-only   only redraw the original heatmap(s) from the saved results
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import protocol as P


def make_scorer():
    from TVD_featureIng import TVDScorer
    return TVDScorer()



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


def create_apt_profiles_with_names(df_encoded, name_mapping):
    profiles = {}
    for _, row in df_encoded.iterrows():
        apt_encoded = row['APTname']
        apt_name = name_mapping.get(apt_encoded, f"Unknown_APT_{apt_encoded}")
        if apt_name not in profiles:
            profiles[apt_name] = {
                'encoded_id': apt_encoded,
                'techniques': set(),
                'software': set(),
                'cve': set()
            }
        if row['techniques_str'] != -1:
            profiles[apt_name]['techniques'].add(row['techniques_str'])
        if row['software_str'] != -1:
            profiles[apt_name]['software'].add(row['software_str'])
        if row['CVE_ID'] != -1:
            profiles[apt_name]['cve'].add(row['CVE_ID'])
    return profiles

def load_apt_profiles_and_mappings():
    pwd = "../BaseData/"
    df_encoded = pd.read_csv(pwd + 'EncodedData.csv')
    df_mapping = pd.read_csv(pwd + 'comparison_Encoded_df.csv')
    apt_name_mapping = df_mapping[['APTnameEncoded', 'APTnameOrigin']].drop_duplicates()
    apt_name_dict = dict(zip(apt_name_mapping['APTnameEncoded'], apt_name_mapping['APTnameOrigin']))
    apt_profiles = create_apt_profiles_with_names(df_encoded, apt_name_dict)
    return apt_profiles, apt_name_dict

def plot_deception_heatmap(results, apt_names, title_suffix="Full"):
    print("\n CREATING TVD DECEPTION HEATMAP...")
    n = len(apt_names)
    name_to_idx = {name: i for i, name in enumerate(apt_names)}
    deception_matrix = np.zeros((n, n), dtype=float)
    
    for key, res in results.items():
        if "_partial" in key:
            continue  # only plot full results (or we could pass a flag)
        try:
            attacker, mimicked = key.split("→")
            attacker = attacker.strip()
            mimicked = mimicked.strip()
        except:
            continue
        i = name_to_idx.get(attacker)
        j = name_to_idx.get(mimicked)
        if i is None or j is None:
            continue
        ar = res['attacker_rank'] if res['attacker_rank'] is not None else n
        mr = res['mimicked_rank'] if res['mimicked_rank'] is not None else n
        deception_matrix[i, j] = ar - mr   # positive = deception (mimicked ranks better)
    
    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("⚠ All deception scores zero; nothing to plot.")
        return
    low = np.percentile(non_zero, 5)
    high = np.percentile(non_zero, 95)
    max_abs = max(abs(low), abs(high))
    vmin, vmax = -max_abs, max_abs
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, square=True,
                cbar_kws={'label': 'Deception Score (Attacker Rank - Mimicked Rank)\n>0: Deception (RED), <0: Detection (BLUE)'})
    plt.title(f'TVD False‑Flag Deception – {title_suffix}\n30% Attacker + 70% Mimicked features per category',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mimicked APT (70% features)', fontsize=12)
    plt.ylabel('Actual Attacker APT (30% features)', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_name = f'false_flag_deception_matrix_tvd_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_name}")


def original_heatmaps():
    """Draw the original false-flag heatmap(s) from the final saved results (no model is re-run)."""
    os.chdir(HERE)
    apt_profiles, apt_name_dict = load_apt_profiles_and_mappings()
    apt_names = list(apt_profiles.keys())
    full_results = P.original_ff_results("full", os.path.join(HERE, "results", "ff_ranks_full.npy"), os.path.join(HERE, "results", "false_flag_summary.json"))
    plot_deception_heatmap(full_results, apt_names, title_suffix="Full")


if __name__ == "__main__":
    os.chdir(HERE)
    if "--heatmaps-only" not in sys.argv:
        P.run_false_flag(make_scorer, "TVD", results_dir="results", smoke=P.smoke_flag())
    if not P.smoke_flag():
        original_heatmaps()
