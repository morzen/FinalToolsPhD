"""
Hybrid false-flag evaluation (thesis Chapter 6).

Every method's false-flag script has scored exactly the same evidence
(BaseData/false_flag_evidence.pkl: first 30% of the attacker profile + last 70% of the
mimicked profile) and saved
the full rank matrix of every pair in <method>/results/ff_ranks_{full,partial}.npy. The hybrid is
the weighted reciprocal-rank fusion (WRRF, k = 60) of those rankings, for the three weight
configurations (common/protocol.py, HYBRID_WEIGHTS). No model is re-run, so this takes seconds instead of weeks.

Run (after the eight false-flag scripts):  python hybrid_false_flag.py   (--smoke: uses the --smoke outputs)
     python hybrid_false_flag.py --heatmaps-only   only redraw the original heatmaps from the saved results
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import protocol as P


def load_method_ranks(scenario, n_pairs, fingerprint, smoke=False):
    ranks = {}
    for m in P.METHODS:
        d = P.method_results_dir(m, smoke)
        path = os.path.join(d, f"ff_ranks_{scenario}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run the false-flag script of {P.METHOD_DIRS[m]} first")
        with open(os.path.join(d, "false_flag_summary.json")) as fh:
            info = json.load(fh)
        if info.get('evidence_fingerprint') != fingerprint or info.get('n_pairs') != n_pairs:
            raise ValueError(f"{m}: false-flag results were computed on different evidence; re-run it")
        ranks[m] = np.load(path).astype(np.float64)
    return ranks


def main():
    smoke = P.smoke_flag()
    if "--heatmaps-only" in sys.argv:
        original_heatmaps(os.path.join(HERE, "results"))
        return
    data = P.load_false_flag_evidence()
    pairs = [data['pairs'][i] for i in P.smoke_indices(len(data['pairs']), P.SMOKE_PAIRS)] if smoke else data['pairs']
    fingerprint = P.evidence_fingerprint()
    out_dir = os.path.join(HERE, "results", "smoke") if smoke else os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    P.print_hybrid_weights()
    summary = {}
    for scenario in ("full", "partial"):
        method_ranks = load_method_ranks(scenario, len(pairs), fingerprint, smoke)
        for cfg, weights in P.HYBRID_WEIGHTS.items():
            fused = P.wrrf_scores(method_ranks, weights)
            R = np.vstack([P.midranks(row) for row in fused])
            label = f"Hybrid ({cfg})"
            res, ar, mr = P.summarise_false_flag(R, pairs, label, scenario)
            summary.setdefault(cfg, {})[scenario] = res
            np.save(os.path.join(out_dir, f"hybrid_{cfg}_ff_ranks_{scenario}.npy"), R.astype(np.float32))
            cfg_dir = os.path.join(out_dir, cfg)
            os.makedirs(cfg_dir, exist_ok=True)
            if not smoke:
                P.plot_false_flag_heatmap(ar, mr, pairs, label, scenario, cfg_dir)
    with open(os.path.join(out_dir, "hybrid_false_flag_summary.json"), "w") as fh:
        json.dump({'n_pairs': len(pairs), 'n_excluded': len(data['excluded']),
                   'evidence_fingerprint': fingerprint, 'weights': P.HYBRID_WEIGHTS, 'results': summary}, fh, indent=2)
    if not smoke:
        original_heatmaps(out_dir)


# =====================================================================================
# HEATMAP CODE OF THE FIRST IMPLEMENTATION, kept unchanged.
# The functions below are unchanged. Only original_heatmaps() is new: it gives them the
# FINAL hybrid results saved by main(), in the format the original script built, once per
# weight configuration (the original had one configuration), saved in results/<configuration>/.
# =====================================================================================
import matplotlib
matplotlib.use("Agg")                  # draw to file only (plt.show() then opens no window)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def plot_false_flag_deception_heatmap(Full_FalseFlag_evidencetable, apt_names):
    print("\n🔥 CREATING FALSE-FLAG DECEPTION HEATMAP...")
    n = len(apt_names)
    name_to_idx = {name:i for i,name in enumerate(apt_names)}
    deception_matrix = np.zeros((n,n))
    count_matrix = np.zeros((n,n), dtype=int)
    for att, mdict in Full_FalseFlag_evidencetable.items():
        if att not in name_to_idx: continue
        ai = name_to_idx[att]
        for mim, rlist in mdict.items():
            if rlist is None or mim not in name_to_idx or att == mim: continue
            mi = name_to_idx[mim]
            self_rank = next((r for a,r in rlist if a==att), n)
            mim_rank = next((r for a,r in rlist if a==mim), n)
            raw = self_rank - mim_rank
            norm = raw / (n-1)
            deception_matrix[ai,mi] += norm
            count_matrix[ai,mi] += 1
    mask = count_matrix > 0
    deception_matrix[mask] /= count_matrix[mask]
    np.fill_diagonal(deception_matrix,0)
    plt.figure(figsize=(20,18))
    sns.heatmap(deception_matrix, xticklabels=apt_names, yticklabels=apt_names,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Deception Score'})
    plt.title('False-Flag Deception Heatmap')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    from datetime import datetime
    path = f'false_flag_deception_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def precompute_apt_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    apt_names = df['APTname'].unique().tolist()
    cache = {}
    for apt in tqdm(apt_names, desc="Caching APTs"):
        ad = df[df['APTname'] == apt]
        cache[apt] = {
            'techniques': sorted(ad['techniques_str'].dropna().unique().tolist()),
            'software': sorted(ad['software_str'].dropna().unique().tolist()),
            'capec': sorted([x for x in ad['CAPEC_ID'].dropna().unique().tolist() if x != -1]),
            'cwe': sorted([x for x in ad['CWE_ID'].dropna().unique().tolist() if x != -1]),
            'cves': sorted([x for x in ad['CVE_ID'].dropna().unique().tolist() if x != -1]),
            'pulse': sorted([x for x in ad['Pulse_ID'].dropna().unique().tolist() if x != -1])
        }
    return cache, apt_names


def original_heatmaps(out_dir):
    """Draw the original hybrid false-flag heatmap from the final hybrid false-flag ranks (full
    scenario, as the original main() did), one per weight configuration, in results/<configuration>/."""
    os.chdir(HERE)
    with open(os.path.join(out_dir, "hybrid_false_flag_summary.json")) as fh:
        if json.load(fh).get('weights') != P.HYBRID_WEIGHTS:
            raise ValueError("hybrid false-flag results were computed with other weights: run hybrid_false_flag.py first")
    apt_cache, unique_apt_names_list = precompute_apt_data("../BaseData/TestFinal_df.csv")
    for cfg in P.HYBRID_WEIGHTS:
        res = P.original_ff_results("full", os.path.join(out_dir, f"hybrid_{cfg}_ff_ranks_full.npy"),
                                    os.path.join(out_dir, "hybrid_false_flag_summary.json"))
        # same format as the original main(): Full[attacker][mimicked] = ranked list of (APT, rank);
        # the heatmap only reads the ranks of the attacker and of the mimicked APT from it
        Full = {n: {} for n in unique_apt_names_list}
        for key, r in res.items():
            att, mim = key.split("→")
            Full[att][mim] = [(att, r['attacker_rank']), (mim, r['mimicked_rank'])]
        cfg_dir = os.path.join(out_dir, cfg)
        os.makedirs(cfg_dir, exist_ok=True)
        os.chdir(cfg_dir)
        print(f"\n[{cfg}]")
        plot_false_flag_deception_heatmap(Full, unique_apt_names_list)
        os.chdir(HERE)


if __name__ == "__main__":
    main()
