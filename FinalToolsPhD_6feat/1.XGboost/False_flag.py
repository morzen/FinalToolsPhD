"""
XGBoost false-flag evaluation (thesis Chapter 6, Section 6.1).

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
# The XGBoost model was pickled from train.py run as __main__: the wrapper class must be visible
# in __main__ (also in the worker processes) for pickle to load it.
from query import PicklableXGBWrapper  # noqa: F401


def make_scorer():
    from query import XGBScorer
    return XGBScorer()



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


def load_apt_profiles_from_data():
    print(" LOADING APT PROFILES FROM DATA...")
    df = pd.read_csv("../BaseData/EncodedData.csv")
    profiles = {}
    for apt_id in df['APTname'].unique():
        apt_data = df[df['APTname'] == apt_id]
        profiles[apt_id] = {
            'techniques': set(apt_data['techniques_str'].unique()),
            'software': set(apt_data['software_str'].unique()),
            'capec': set(apt_data['CAPEC_ID'].unique()),
            'cwe': set(apt_data['CWE_ID'].unique()),
            'cve': set(apt_data['CVE_ID'].unique()),
            'pulse': set(apt_data['Pulse_ID'].unique()),
            'total_features': len(apt_data)
        }
    print(f" Loaded profiles for {len(profiles)} APTs")
    return profiles

def load_mappings():
    """Load mappings and APT name mapping."""
    df_mapping = pd.read_csv("../BaseData/comparison_Encoded_df.csv")
    technique_mapping = {}
    software_mapping = {}
    capec_mapping = {}
    cwe_mapping = {}
    cve_mapping = {}
    pulse_mapping = {}

    for _, row in df_mapping.iterrows():
        if pd.notna(row['techniques_Origin']) and pd.notna(row['techniques_Encoded']):
            technique_mapping[row['techniques_Origin']] = row['techniques_Encoded']
        if pd.notna(row['software_Origin']) and pd.notna(row['software_Encoded']):
            software_mapping[row['software_Origin']] = row['software_Encoded']
        if pd.notna(row['CAPEC_ID_Origin']) and pd.notna(row['CAPEC_ID_Encoded']):
            capec_mapping[row['CAPEC_ID_Origin']] = row['CAPEC_ID_Encoded']
        if pd.notna(row['CWE_ID_Origin']) and pd.notna(row['CWE_ID_Encoded']):
            cwe_mapping[row['CWE_ID_Origin']] = row['CWE_ID_Encoded']
        if pd.notna(row['CVE_ID_Origin']) and pd.notna(row['CVE_ID_Encoded']):
            cve_mapping[row['CVE_ID_Origin']] = row['CVE_ID_Encoded']
        if pd.notna(row['Pulse_ID_Origin']) and pd.notna(row['Pulse_ID_Encoded']):
            pulse_mapping[row['Pulse_ID_Origin']] = row['Pulse_ID_Encoded']

    apt_name_mapping = {}
    for _, row in df_mapping.iterrows():
        if row['APTnameEncoded'] not in apt_name_mapping:
            apt_name_mapping[row['APTnameEncoded']] = row['APTnameOrigin']

    return technique_mapping, software_mapping, capec_mapping, cwe_mapping, cve_mapping, pulse_mapping, apt_name_mapping

def plot_false_flag_analysis_binary(results, apt_name_mapping, profiles, apt_ids):
    """Deception heatmap using rank difference (attacker - mimicked)."""
    n = len(apt_ids)
    apt_names = [apt_name_mapping.get(aid, f"APT{aid}") for aid in apt_ids]
    deception_matrix = np.zeros((n, n))
    id_to_idx = {aid: i for i, aid in enumerate(apt_ids)}

    for key, res in results.items():
        try:
            attacker_name, mimicked_name = key.split("→")
            attacker_name = attacker_name.strip()
            mimicked_name = mimicked_name.strip()
            attacker_id = next((aid for aid, name in apt_name_mapping.items() if name == attacker_name), None)
            mimicked_id = next((aid for aid, name in apt_name_mapping.items() if name == mimicked_name), None)
        except:
            continue
        i = id_to_idx.get(attacker_id)
        j = id_to_idx.get(mimicked_id)
        if i is None or j is None:
            continue
        a_r = res['attacker_rank'] if res['attacker_rank'] is not None else n
        m_r = res['mimicked_rank'] if res['mimicked_rank'] is not None else n
        deception_matrix[i, j] = a_r - m_r

    non_zero = deception_matrix[np.nonzero(deception_matrix)]
    if non_zero.size == 0:
        print("All deception scores zero; nothing to plot.")
        return

    low = np.percentile(non_zero, 5)
    high = np.percentile(non_zero, 95)
    max_abs = max(abs(low), abs(high))
    plt.figure(figsize=(16, 14))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=-max_abs, vmax=max_abs, square=True,
                cbar_kws={'label': 'Deception Score (Att Rank - Mim Rank)\n>0: Deception, <0: Detection'})
    plt.title('XGBoost False-Flag Deception (Model Probabilities, Full Evidence)')
    plt.xlabel('Mimicked APT (70% features)')
    plt.ylabel('Actual Attacker (30% features)')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('false_flag_deception_matrix_xgboost_probabilities.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(" Heatmap saved.")


def original_heatmaps():
    """Draw the original false-flag heatmap(s) from the final saved results (no model is re-run)."""
    os.chdir(HERE)
    # APT order and names exactly as in evaluate_false_flag_attacks_binary_parallel():
    profiles = load_apt_profiles_from_data()
    apt_ids = sorted(profiles.keys())
    _, _, _, _, _, _, apt_name_mapping = load_mappings()
    results_full = P.original_ff_results("full", os.path.join(HERE, "results", "ff_ranks_full.npy"), os.path.join(HERE, "results", "false_flag_summary.json"))
    plot_false_flag_analysis_binary(results_full, apt_name_mapping, profiles, apt_ids)


if __name__ == "__main__":
    os.chdir(HERE)
    if "--heatmaps-only" not in sys.argv:
        P.run_false_flag(make_scorer, "XGBoost", results_dir="results", smoke=P.smoke_flag())
    if not P.smoke_flag():
        original_heatmaps()
