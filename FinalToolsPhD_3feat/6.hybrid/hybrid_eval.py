"""
Hybrid attribution evaluation (thesis Chapter 5), full data and shuffled & divided.

The hybrid is the weighted reciprocal-rank fusion (WRRF, k = 60) of the eight methods. Each method's
eval.py has already ranked all APTs for exactly the same full and divided evidence (shared
protocol, fixed seed) and saved the rank matrices in <method>/results/. This script fuses them for
the three weight configurations (common/protocol.py, HYBRID_WEIGHTS), so no model has to be re-run.

Run (after the eight eval.py):  python hybrid_eval.py      (--smoke: uses the eval.py --smoke outputs)
     python hybrid_eval.py --heatmaps-only   only redraw the original heatmaps from the saved results
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import protocol as P


def load_method_ranks(scenario, smoke=False):
    """Rank matrices of the 8 methods (rows = query APTs, in the order of `query_indices`)."""
    ranks, meta = {}, {}
    for m in P.METHODS:
        d = P.method_results_dir(m, smoke)
        path = os.path.join(d, f"attribution_ranks_{scenario}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run {P.METHOD_DIRS[m]}/eval.py first")
        ranks[m] = np.load(path).astype(np.float64)
        with open(os.path.join(d, "attribution_eval.json")) as fh:
            meta[m] = json.load(fh)
    names = P.apt_names()
    for m, info in meta.items():
        if info['apt_names'] != names or info['divided_seed'] != P.DIVIDED_SEED or info['divider'] != P.DIVIDER \
                or info.get('n_queries') != next(iter(ranks.values())).shape[0]:
            raise ValueError(f"{m}: attribution results were produced with a different protocol; re-run its eval.py")
    query_idx = [info.get('query_indices', list(range(info['n_queries']))) for info in meta.values()]
    if any(q != query_idx[0] for q in query_idx):
        raise ValueError("the methods were evaluated on different query APTs; re-run their eval.py")
    return ranks, query_idx[0]


def plot_rank_heatmap(R, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    names = P.apt_names()
    n = len(names)
    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "yellow", "green"], N=100)
    fig, ax = plt.subplots(figsize=(24, 20))
    im = ax.imshow(n - R + 1, cmap=cmap, vmin=1, vmax=n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_title(title, fontsize=16, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[1, (n + 1) / 2, n])
    cbar.ax.set_yticklabels([f'Rank {n} (Red)', f'Rank ~{(n + 1) // 2} (Yellow)', 'Rank 1 (Green)'])
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    smoke = P.smoke_flag()
    if "--heatmaps-only" in sys.argv:
        original_heatmaps(os.path.join(HERE, "results"))
        return
    names = P.apt_names()
    out_dir = os.path.join(HERE, "results", "smoke") if smoke else os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    P.print_hybrid_weights()
    summary = {}
    for scenario in ("full", "divided"):
        method_ranks, query_idx = load_method_ranks(scenario, smoke)
        for cfg, weights in P.HYBRID_WEIGHTS.items():
            fused = P.wrrf_scores(method_ranks, weights)
            R = np.vstack([P.midranks(row) for row in fused])
            target = R[np.arange(len(query_idx)), query_idx]   # rank of the queried APT itself
            label = "FULL DATA (data division 1)" if scenario == "full" else "SHUFFLED & DIVIDED (data division 2)"
            print(f"\nHybrid [{cfg}] - {label}")
            res = {}
            for k in P.TOP_K:
                hits = int((target <= k).sum())
                res[k] = 100.0 * hits / len(target)
                print(f"  Top {k}: {hits}/{len(target)} ({res[k]:.2f}%)")
            summary.setdefault(cfg, {})[scenario] = res
            np.save(os.path.join(out_dir, f"hybrid_{cfg}_ranks_{scenario}.npy"), R.astype(np.float32))
            if scenario == "full" and not smoke:
                plot_rank_heatmap(R, f"Hybrid ({cfg}) APT ranking matrix, full profiles\n"
                                     f"(Green = Rank 1, Yellow = Rank ~{(len(names) + 1) // 2}, Red = Rank {len(names)})",
                                  os.path.join(out_dir, f"apt_ranking_heatmap_{cfg}.png"))
    summary['_weights'] = P.HYBRID_WEIGHTS          # record of the weights used (checked by collect_results.py)
    with open(os.path.join(out_dir, "hybrid_attribution_eval.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
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
from matplotlib.colors import LinearSegmentedColormap

def create_apt_ranking_heatmap(apt_dict):
    all_apt_names = sorted(apt_dict.keys())
    n_apts = len(all_apt_names)                     # number of groups in the dataset
    df = pd.DataFrame(index=all_apt_names, columns=all_apt_names).fillna(0)
    for query_apt in all_apt_names:
        rankings = apt_dict.get(query_apt, [])
        ranking_dict = dict(rankings)
        for ranked_apt in all_apt_names:
            if ranked_apt in ranking_dict:
                rank = ranking_dict[ranked_apt]
                inverted_score = n_apts - rank + 1
                df.loc[query_apt, ranked_apt] = inverted_score

    plt.figure(figsize=(24, 20))
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("red_green", colors, N=100)
    sns.heatmap(df, cmap=cmap, vmin=1, vmax=n_apts, square=True,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': 0.8, 'label': f'APT Ranking (Green=Best/Rank 1, Red=Worst/Rank {n_apts})',
                          'ticks': [1, n_apts // 2, n_apts]})
    plt.title(f"APT Ranking Similarity Matrix\n(Green = Rank 1, Yellow = Rank ~{n_apts // 2}, Red = Rank {n_apts})",
              fontsize=18, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    cbar = plt.gcf().axes[-1]
    cbar.set_yticklabels([f'Rank {n_apts} (Red)', f'Rank ~{n_apts // 2} (Yellow)', 'Rank 1 (Green)'])
    plt.tight_layout()
    plt.savefig('apt_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Heatmap saved as 'apt_ranking_heatmap.png'")
    plt.show()


def original_heatmaps(out_dir):
    """Draw the original hybrid ranking heatmap from the final hybrid ranks (full profiles),
    one per weight configuration, in results/<configuration>/apt_ranking_heatmap.png."""
    os.chdir(HERE)
    names = P.apt_names()
    with open(os.path.join(out_dir, "hybrid_attribution_eval.json")) as fh:
        if json.load(fh).get('_weights') != P.HYBRID_WEIGHTS:
            raise ValueError("hybrid results were computed with other weights: run hybrid_eval.py first")
    for cfg in P.HYBRID_WEIGHTS:
        R = np.load(os.path.join(out_dir, f"hybrid_{cfg}_ranks_full.npy"))
        if R.shape != (len(names), len(names)):
            raise ValueError(f"hybrid_{cfg}_ranks_full.npy: unexpected shape {R.shape}")
        # same format as parallel_process_apts(): {queried APT: [(APT, rank), ...] best first}
        full_results = {names[i]: [(names[j], P._plain_rank(R[i, j])) for j in np.argsort(R[i], kind='stable')]
                        for i in range(len(names))}
        cfg_dir = os.path.join(out_dir, cfg)
        os.makedirs(cfg_dir, exist_ok=True)
        os.chdir(cfg_dir)
        print(f"\n[{cfg}]")
        create_apt_ranking_heatmap(full_results)
        os.chdir(HERE)


if __name__ == "__main__":
    main()
