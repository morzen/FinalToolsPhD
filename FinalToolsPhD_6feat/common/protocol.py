"""
Shared evaluation protocol used by EVERY attribution method in this folder.

It implements, in one place, the evaluation rules stated in the thesis so that all
methods are evaluated on exactly the same evidence and ranked the same way:

  * APT profiles      : sets of encoded feature IDs per feature type; -1 marks a
                        missing value and is never treated as a feature.
  * Full data (§4.4.2): each APT queried with its complete profile.
  * Shuffled & divided: each feature type of the profile is shuffled and divided into
                        two equal parts (one half, at least one item, is used).
                        Fixed seed, identical subsets for every method.
  * False flag (§6.1) : each APT profile is ordered from the highest-level feature type
                        (TTPs) to the lowest (FEATURES order). The synthetic profile is
                        the first 30% of the true attacker's ordered profile plus the
                        last 70% of the mimicked APT's ordered profile, whatever the
                        feature types. Partial = 50% of each feature type of it.
                        Generated once into BaseData/false_flag_evidence.pkl so every
                        method (and the hybrid) is scored on identical evidence.
  * Ranking           : ties share the average (mid) rank of the tied positions;
                        a Top-k hit means mid-rank <= k.
  * CTI score (§4.6)  : score = 100 * (1 - (rank - 1) / (N - 1)), N = number of APTs.

Every evaluation also saves its full rank/score matrices in <method>/results/ so the
hybrid (WRRF) can be computed by fusing them, without re-running any model.
"""
import os
import sys
import json
import math
import time
from fractions import Fraction
import pickle
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # FinalToolsPhD_xfeat
BASE = os.path.join(ROOT, "BaseData")

FEATURES = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
COLUMNS = {'techniques': 'techniques_str', 'software': 'software_str', 'capec': 'CAPEC_ID',
           'cwe': 'CWE_ID', 'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
MISSING = -1

TOP_K = [1, 5, 10, 20]
DIVIDER = 2
DIVIDED_SEED = 42

FF_SEED = 2026
FF_ATTACKER_SHARE = 0.30      # first 30% of the true attacker's profile (from the high-level end)
FF_MIMICKED_SHARE = 0.70      # last 70% of the mimicked APT's profile (from the low-level end)
FF_PARTIAL_FRACTION = 0.5
FF_EVIDENCE_FILE = os.path.join(BASE, "false_flag_evidence.pkl")
SMOKE_APTS = 10              # --smoke: APTs queried (spread over the alphabetical list)
SMOKE_PAIRS = 40             # --smoke: false-flag pairs (spread over 40 different attackers)

MAX_WORKERS = int(os.environ.get("PHD_WORKERS", min(mp.cpu_count(), 8)))


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
_PROFILE_CACHE = None


def load_profiles():
    """Canonical APT profiles: {apt_name: {'encoded_id': int, feature: sorted list of IDs}}.
    Ordered by APT name. -1 (missing) is removed."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    df = pd.read_csv(os.path.join(BASE, "EncodedData.csv"),
                     usecols=['APTname'] + [COLUMNS[f] for f in FEATURES])
    names = pd.read_csv(os.path.join(BASE, "comparison_Encoded_df.csv"),
                        usecols=['APTnameEncoded', 'APTnameOrigin']).drop_duplicates()
    name_of = dict(zip(names['APTnameEncoded'], names['APTnameOrigin']))
    profiles = {}
    for enc, group in df.groupby('APTname'):
        prof = {'encoded_id': int(enc)}
        for f in FEATURES:
            vals = pd.unique(group[COLUMNS[f]].dropna())
            prof[f] = sorted(int(v) for v in vals if int(v) != MISSING)
        profiles[str(name_of.get(enc, f"APT{enc}"))] = prof
    _PROFILE_CACHE = dict(sorted(profiles.items()))
    return _PROFILE_CACHE


def apt_names():
    return list(load_profiles().keys())


def non_empty(evidence):
    """Keep only the feature types that actually contain evidence."""
    return {f: list(v) for f, v in evidence.items() if f in FEATURES and len(v) > 0}


def full_evidence(profile):
    return non_empty({f: profile[f] for f in FEATURES})


def divided_evidence(apt_name, profile, divider=DIVIDER, seed=DIVIDED_SEED):
    """Shuffle each feature type and keep one of `divider` equal parts (at least one item).
    The RNG is seeded per APT, so every method receives exactly the same subset."""
    rng = random.Random(f"{seed}|{apt_name}")
    ev = {}
    for f in FEATURES:
        items = list(profile[f])
        if items:
            rng.shuffle(items)
            ev[f] = sorted(items[:max(1, len(items) // divider)])
    return ev


# ------------------------------------------------------------------------------
# False-flag evidence (thesis §6.1)
# ------------------------------------------------------------------------------
def ff_params():
    """Parameters of the false-flag construction (stored with the evidence file; a file built
    with other parameters is rebuilt automatically)."""
    return {'rule': 'first 30% of the attacker profile + last 70% of the mimicked profile',
            'order': 'feature types from high to low level (FEATURES order); fixed random order '
                     '(seeded per APT and feature type) within a feature type',
            'seed': FF_SEED, 'attacker_share': FF_ATTACKER_SHARE, 'mimicked_share': FF_MIMICKED_SHARE,
            'partial_fraction': FF_PARTIAL_FRACTION, 'features': list(FEATURES),
            'version': 2}     # 2: exact half-up rounding of the 30% / 70% shares


def ordered_profile(apt_name, profile, seed=FF_SEED):
    """The whole APT profile as ONE list of (feature type, item), from the highest-level feature
    type (TTPs) down to the lowest (FEATURES order).
    Items of the same feature type have no level order between them, so they are placed in a
    fixed random order (seeded per APT and type): where the 30% / 70% cut falls inside a type,
    the items kept are therefore not chosen by their ID number."""
    ordered = []
    for f in FEATURES:
        items = list(profile[f])
        random.Random(f"{seed}|{apt_name}|{f}").shuffle(items)
        ordered.extend((f, x) for x in items)
    return ordered


def share_count(n, fraction):
    """Number of items making up `fraction` of an n-item profile: rounded to the nearest
    integer (halves up), and at least one item when the profile is not empty. Computed with
    exact fractions: in floating point 45 x 0.7 = 31.499999999999996 would round down."""
    if n == 0:
        return 0
    return max(1, int((2 * n * Fraction(str(fraction)) + 1) // 2))


def smoke_indices(n, k):
    """k indices spread evenly over range(n) (used by --smoke runs)."""
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()))


def false_flag_parts(attacker_ordered, mimicked_ordered):
    """(top, bottom): the first 30% of the attacker's ordered profile (high-level end) and the
    last 70% of the mimicked APT's ordered profile (low-level end). Each percentage is taken of
    that APT's own profile size, whatever feature types the items belong to."""
    top = attacker_ordered[:share_count(len(attacker_ordered), FF_ATTACKER_SHARE)]
    bottom = mimicked_ordered[len(mimicked_ordered) - share_count(len(mimicked_ordered), FF_MIMICKED_SHARE):]
    return top, bottom


def false_flag_evidence(attacker_name, attacker, mimicked_name, mimicked, seed=FF_SEED, ordered=None):
    """Synthetic false-flag profile for one (true attacker, mimicked APT) pair.

    Example: attacker A,B,C,D,E,F,G,H,I,J (10 items, high to low level) and mimicked APT
    Q,R,S,T,V,W,X,Y,Z (9 items) give A,B,C (30% of 10) + T,V,W,X,Y,Z (70% of 9 = 6.3 -> 6).
    An item present in both parts is kept once.

    full    : the synthetic profile, split back into feature types;
    partial : a random half (at least one item) of each feature type of `full`.
    Returns (full, partial, (top, bottom)) or None if either profile is empty.
    `ordered` may hold precomputed ordered_profile() lists by APT name."""
    ordered = ordered or {}
    att = ordered[attacker_name] if attacker_name in ordered else ordered_profile(attacker_name, attacker, seed)
    mim = ordered[mimicked_name] if mimicked_name in ordered else ordered_profile(mimicked_name, mimicked, seed)
    if not att or not mim:
        return None
    top, bottom = false_flag_parts(att, mim)
    merged = {}
    for f, x in top + bottom:
        merged.setdefault(f, set()).add(x)
    full = {f: sorted(merged[f]) for f in FEATURES if f in merged}

    rng = random.Random(f"{seed}|{attacker_name}|{mimicked_name}|partial")
    partial = {}
    for f, items in full.items():
        items = list(items)
        rng.shuffle(items)
        partial[f] = sorted(items[:max(1, int(len(items) * FF_PARTIAL_FRACTION))])
    return full, partial, (top, bottom)


class PackedEvidence:
    """Many evidence dicts stored compactly as numpy arrays (the synthetic profiles hold tens of
    millions of items in total; Python lists would need several GB in every worker).
    packed[i] -> {feature type: [IDs]} with only the non-empty feature types."""

    def __init__(self, raw):
        self.items = raw['items']
        self.offsets = raw['offsets']            # (n, n_features + 1) start of each type, then end
        self.features = list(raw['features'])

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        o = self.offsets[i]
        return {f: self.items[o[j]:o[j + 1]].tolist() for j, f in enumerate(self.features) if o[j + 1] > o[j]}

    def sizes(self):
        return self.offsets[:, -1] - self.offsets[:, 0]


def _pack(evidences):
    counts = np.array([[len(ev.get(f, ())) for f in FEATURES] for ev in evidences], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts.ravel())])
    n, nf = counts.shape
    offsets = starts[np.arange(n)[:, None] * nf + np.arange(nf + 1)[None, :]]
    flat = [x for ev in evidences for f in FEATURES for x in ev.get(f, ())]
    dtype = np.int16 if (not flat or max(flat) < 2 ** 15) else np.int32
    return {'items': np.asarray(flat, dtype=dtype), 'offsets': offsets, 'features': list(FEATURES)}


def build_false_flag_evidence(path=FF_EVIDENCE_FILE, force=False):
    """Generate (once) the false-flag evidence for all ordered (attacker, mimicked) APT pairs."""
    if os.path.exists(path) and not force:
        return load_false_flag_evidence(path)
    profiles = load_profiles()
    names = list(profiles)
    ordered = {n: ordered_profile(n, profiles[n]) for n in names}
    pairs, full, partial, composition, excluded = [], [], [], [], []
    for a in names:
        for m in names:
            if a == m:
                continue
            res = false_flag_evidence(a, profiles[a], m, profiles[m], ordered=ordered)
            if res is None:
                excluded.append((a, m, "empty profile"))
                continue
            top, bottom = res[2]
            pairs.append((a, m))
            full.append(res[0])
            partial.append(res[1])
            composition.append([[sum(1 for g, _ in part if g == f) for f in FEATURES] for part in (top, bottom)])
    data = {'pairs': pairs, 'full': _pack(full), 'partial': _pack(partial),
            'composition': np.asarray(composition, dtype=np.int32).reshape(len(pairs), 2, len(FEATURES)),
            'excluded': excluded, 'apt_names': names, 'params': ff_params()}
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    os.replace(tmp, path)
    return _unpack(data)


def _unpack(data):
    data = dict(data)
    for scenario in ("full", "partial"):
        data[scenario] = PackedEvidence(data[scenario])
    return data


def load_false_flag_evidence(path=FF_EVIDENCE_FILE):
    """{'pairs', 'full', 'partial', 'composition', 'excluded', 'apt_names', 'params'};
    data['full'][i] / data['partial'][i] is the evidence dict of pair i."""
    if not os.path.exists(path):
        return build_false_flag_evidence(path)
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict) or data.get('params') != ff_params() or data.get('apt_names') != apt_names():
        return build_false_flag_evidence(path, force=True)
    return _unpack(data)


# ------------------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------------------
def _canonical(scores):
    """Round to 12 significant digits so mathematically equal scores tie exactly."""
    s = np.asarray(scores, dtype=np.float64).copy()
    s[np.isnan(s)] = -np.inf                  # NaN ranks last; +inf / -inf keep their order
    out = s.copy()
    nz = np.isfinite(s) & (s != 0)
    if nz.any():
        mag = 10.0 ** np.floor(np.log10(np.abs(s[nz])))
        out[nz] = np.round(s[nz] / mag, 11) * mag
    return out


def midranks(scores):
    """Rank 1 = highest score; tied scores share the average of their positions."""
    s = _canonical(scores)
    try:
        from scipy.stats import rankdata
        return rankdata(-s, method='average')
    except ImportError:
        order = np.argsort(-s, kind='mergesort')
        ranks = np.empty(len(s), dtype=np.float64)
        sorted_s = s[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return ranks


def ranked_results(names, scores, score_key='Score'):
    """Ranked list (best first) with mid-ranks, in the format the hybrid/CTI code reads."""
    r = midranks(scores)
    order = np.lexsort((np.arange(len(names)), r))
    return [{'rank': float(r[i]), 'apt_name': names[i], score_key: float(scores[i])} for i in order]


def cti_score(rank, n_apts):
    """Thesis §4.6: score = 100 (1 - (rank - 1) / (N - 1))."""
    return 100.0 * (1.0 - (rank - 1.0) / (n_apts - 1.0))


# ------------------------------------------------------------------------------
# Scorer alignment
# ------------------------------------------------------------------------------
def alignment(scorer_names):
    """Index array mapping a scorer's APT order to the canonical (sorted-name) order."""
    canon = apt_names()
    pos = {n: i for i, n in enumerate(scorer_names)}
    missing = [n for n in canon if n not in pos]
    if missing or len(scorer_names) != len(canon):
        raise ValueError(f"Scorer APT names do not match the dataset: missing {missing[:5]}")
    return np.array([pos[n] for n in canon])


def _results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ------------------------------------------------------------------------------
# Attribution evaluation: full data and shuffled & divided
# ------------------------------------------------------------------------------
def _scorer_folder(scorer):
    """Folder of the file that defines the scorer class (the method's folder)."""
    try:
        return os.path.dirname(os.path.abspath(sys.modules[type(scorer).__module__].__file__))
    except Exception:
        return os.getcwd()


def plot_attribution_heatmap(S, method_label, path):
    """Heatmap of the scores behind the Top-k numbers: row = APT queried with its full profile,
    column = score the method gives to each APT."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as exc:
        print(f"(heatmap skipped: {exc})")
        return
    names = apt_names()
    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "yellow", "green"], N=100)
    # Plain matplotlib style whatever the calling script set before: Jaccard / Feature Matching / TVD
    # call sns.set() for their own heatmap, which otherwise adds white grid lines and another font here.
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(20, 18))
        im = ax.imshow(S, cmap=cmap, vmin=float(np.min(S)), vmax=float(np.max(S)), aspect='equal')
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=4)
        ax.set_yticklabels(names, fontsize=4)
        ax.set_xlabel('APT profile scored')
        ax.set_ylabel('APT queried with its full profile')
        ax.set_title(f'{method_label}: scores used for the full-data attribution results')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='score')
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print(f"Saved {path}")


def evaluate_attribution(scorer, method_label, results_dir="results", smoke=False):
    """Query every APT with (1) its full profile and (2) its seeded divided profile.
    Prints Top-k precision and saves rank/score matrices for the hybrid. A relative results_dir
    is taken inside the method's own folder, whatever the current directory."""
    profiles = load_profiles()
    all_names = list(profiles)
    query_idx = smoke_indices(len(all_names), SMOKE_APTS) if smoke else list(range(len(all_names)))
    names = [all_names[i] for i in query_idx]
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_scorer_folder(scorer), results_dir)
    if smoke:
        results_dir = os.path.join(results_dir, "smoke")
    align = alignment(scorer.apt_names)
    canon = apt_names()
    out = {}
    for scenario in ("full", "divided"):
        t0 = time.time()
        S = np.zeros((len(names), len(canon)), dtype=np.float64)
        R = np.zeros_like(S)
        hits = {k: 0 for k in TOP_K}
        target_ranks = []
        for i, name in enumerate(names):
            ev = full_evidence(profiles[name]) if scenario == "full" else divided_evidence(name, profiles[name])
            s = np.asarray(scorer.score(ev), dtype=np.float64)[align]
            r = midranks(s)
            S[i], R[i] = s, r
            tr = r[canon.index(name)]
            target_ranks.append(float(tr))
            for k in TOP_K:
                hits[k] += int(tr <= k)
        label = "FULL DATA (data division 1)" if scenario == "full" else "SHUFFLED & DIVIDED (data division 2)"
        print(f"\n{method_label} - {label}  [{time.time() - t0:.1f}s]")
        for k in TOP_K:
            print(f"  Top {k}: {hits[k]}/{len(names)} ({100.0 * hits[k] / len(names):.2f}%)")
        out[scenario] = {'top_k': {k: 100.0 * hits[k] / len(names) for k in TOP_K},
                         'hits': hits, 'n': len(names), 'mean_rank': float(np.mean(target_ranks))}
        d = _results_dir(results_dir)
        np.save(os.path.join(d, f"attribution_scores_{scenario}.npy"), S.astype(np.float32))
        np.save(os.path.join(d, f"attribution_ranks_{scenario}.npy"), R.astype(np.float32))
        if scenario == "full" and not smoke:
            plot_attribution_heatmap(S, method_label, os.path.join(d, "attribution_heatmap_full.png"))
    meta = {'method': method_label, 'apt_names': canon, 'divider': DIVIDER,
            'divided_seed': DIVIDED_SEED, 'n_queries': len(names), 'query_indices': query_idx,
            'results': out}
    with open(os.path.join(results_dir, "attribution_eval.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return out


# ------------------------------------------------------------------------------
# False-flag evaluation (parallel, model loaded once per worker)
# ------------------------------------------------------------------------------
_W = {}


def _ff_init(factory, scenario, evidence_path, rows=None):
    _W['rows'] = rows
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _W['scorer'] = factory()
    _W['align'] = alignment(_W['scorer'].apt_names)
    with open(evidence_path, 'rb') as fh:
        _W['evidence'] = PackedEvidence(pickle.load(fh)[scenario])


def _ff_chunk(bounds):
    start, end = bounds
    scorer, align, evs = _W['scorer'], _W['align'], _W['evidence']
    rows = _W['rows']
    S = np.empty((end - start, len(align)), dtype=np.float64)
    for k in range(start, end):
        i = rows[k] if rows is not None else k
        S[k - start] = np.asarray(scorer.score(evs[i]), dtype=np.float64)[align]
    return start, S


def summarise_false_flag(R, pairs, method_label, scenario):
    canon = apt_names()
    idx = {n: i for i, n in enumerate(canon)}
    a_idx = np.array([idx[a] for a, _ in pairs])
    m_idx = np.array([idx[m] for _, m in pairs])
    rows = np.arange(len(pairs))
    ar, mr = R[rows, a_idx], R[rows, m_idx]
    n = len(pairs)
    summary = {}
    title = "FULL (first 30% of the attacker profile + last 70% of the mimicked profile)" \
        if scenario == "full" else "PARTIAL (half of each feature type of the full false-flag profile)"
    print("\n" + "=" * 80)
    print(f"FALSE-FLAG RESULTS - {method_label} - {title}")
    print("=" * 80)
    for k in TOP_K:
        ad, ma = ar <= k, mr <= k
        res = {'AD': 100.0 * ad.mean(), 'MA': 100.0 * ma.mean(),
               'Both': 100.0 * (ad & ma).mean(), 'Neither': 100.0 * (~ad & ~ma).mean()}
        summary[k] = res
        print(f"\n Top-{k}:")
        print(f"  Attacker detected (AD): {int(ad.sum())}/{n} ({res['AD']:.1f}%)")
        print(f"  Mimicked APT (MA):      {int(ma.sum())}/{n} ({res['MA']:.1f}%)")
        print(f"  Both:                   {int((ad & ma).sum())}/{n} ({res['Both']:.1f}%)")
        print(f"  Neither (N):            {int((~ad & ~ma).sum())}/{n} ({res['Neither']:.1f}%)")
    return summary, ar, mr


def plot_false_flag_heatmap(ar, mr, pairs, method_label, scenario, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(heatmap skipped: {exc})")
        return
    canon = apt_names()
    idx = {n: i for i, n in enumerate(canon)}
    M = np.full((len(canon), len(canon)), np.nan)
    for (a, m), x, y in zip(pairs, ar, mr):
        M[idx[a], idx[m]] = x - y                       # > 0: mimicked APT ranked above attacker
    vals = M[np.isfinite(M)]
    lim = max(abs(np.percentile(vals, 5)), abs(np.percentile(vals, 95)), 1.0)
    fig, ax = plt.subplots(figsize=(16, 14))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('lightgrey')
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-lim, vmax=lim, aspect='equal')
    ax.set_xticks(range(len(canon)))
    ax.set_yticks(range(len(canon)))
    ax.set_xticklabels(canon, rotation=90, fontsize=4)
    ax.set_yticklabels(canon, fontsize=4)
    ax.set_xlabel('Mimicked APT (last 70% of its profile, low-level end)')
    ax.set_ylabel('True attacker (first 30% of its profile, high-level end)')
    ax.set_title(f'{method_label} false-flag deception ({scenario})\n'
                 f'attacker rank - mimicked rank (>0 red: deceived, <0 blue: attacker ranked higher)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(results_dir, f"false_flag_heatmap_{scenario}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def run_false_flag(factory, method_label, results_dir="results", smoke=False, workers=MAX_WORKERS,
                   chunk=100):
    """Score the shared false-flag evidence with `factory()` (a scorer), in parallel."""
    data = load_false_flag_evidence()
    rows = smoke_indices(len(data['pairs']), SMOKE_PAIRS) if smoke else None
    pairs = [data['pairs'][i] for i in rows] if smoke else data['pairs']
    n_total = len(pairs)
    print(f"{method_label}: {n_total} false-flag pairs "
          f"({len(data['excluded'])} ordered pairs excluded: empty profile), {workers} workers")
    results_dir = _results_dir(os.path.join(results_dir, "smoke") if smoke else results_dir)
    summaries = {}
    for scenario in ("full", "partial"):
        t0 = time.time()
        S = np.empty((n_total, len(apt_names())), dtype=np.float64)
        bounds = [(s, min(s + chunk, n_total)) for s in range(0, n_total, chunk)]
        if workers <= 1:
            _ff_init(factory, scenario, FF_EVIDENCE_FILE, rows)
            for b in bounds:
                s, block = _ff_chunk(b)
                S[s:s + len(block)] = block
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_ff_init,
                                     initargs=(factory, scenario, FF_EVIDENCE_FILE, rows)) as ex:
                futures = [ex.submit(_ff_chunk, b) for b in bounds]
                done = 0
                for fut in as_completed(futures):
                    s, block = fut.result()
                    S[s:s + len(block)] = block
                    done += len(block)
                    el = time.time() - t0
                    print(f"  [{scenario}] {done}/{n_total} ({100.0 * done / n_total:.1f}%) "
                          f"ETA {el / done * (n_total - done) / 60:.1f} min", flush=True)
        R = np.vstack([midranks(row) for row in S])
        summary, ar, mr = summarise_false_flag(R, pairs[:n_total], method_label, scenario)
        summaries[scenario] = summary
        np.save(os.path.join(results_dir, f"ff_scores_{scenario}.npy"), S.astype(np.float32))
        np.save(os.path.join(results_dir, f"ff_ranks_{scenario}.npy"), R.astype(np.float32))
        if not smoke:
            plot_false_flag_heatmap(ar, mr, pairs[:n_total], method_label, scenario, results_dir)
        print(f"[{scenario}] completed in {(time.time() - t0) / 60:.1f} min")
    with open(os.path.join(results_dir, "false_flag_summary.json"), "w") as fh:
        json.dump({'method': method_label, 'n_pairs': n_total, 'pair_indices': rows,
                   'evidence_fingerprint': evidence_fingerprint(),
                   'n_excluded': len(data['excluded']), 'params': data['params'],
                   'results': summaries}, fh, indent=2)
    return summaries


# ------------------------------------------------------------------------------
# Input for the ORIGINAL heatmap functions (kept in each false-flag script and in 6.hybrid)
# ------------------------------------------------------------------------------
def _plain_rank(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def original_ff_results(scenario, ranks_file, summary_file, partial_suffix=True, as_tuple=False):
    """Final false-flag results in the format the original scripts gave to their heatmap functions:
    {"attacker→mimicked": {'attacker_rank': r_a, 'mimicked_rank': r_m}}, "_partial" appended to the
    keys of the partial scenario (as the original scripts did), or (r_a, r_m) tuples if as_tuple (GNN).
    The ranks are read from the saved rank matrix (one row per false-flag pair, mid-ranks), after
    checking that it is the full run on the current false-flag evidence."""
    with open(summary_file) as fh:
        info = json.load(fh)
    pairs = load_false_flag_evidence()['pairs']
    if info.get('evidence_fingerprint') != evidence_fingerprint() or info.get('n_pairs') != len(pairs) \
            or info.get('pair_indices') is not None:
        raise ValueError(f"{summary_file}: not the full false-flag run on the current evidence; re-run it")
    R = np.load(ranks_file)
    names = apt_names()
    if R.shape != (len(pairs), len(names)):
        raise ValueError(f"{ranks_file}: unexpected shape {R.shape}")
    idx = {n: i for i, n in enumerate(names)}
    suffix = "_partial" if (scenario == "partial" and partial_suffix) else ""
    results = {}
    for row, (a, m) in enumerate(pairs):
        ar, mr = _plain_rank(R[row, idx[a]]), _plain_rank(R[row, idx[m]])
        results[f"{a}→{m}{suffix}"] = (ar, mr) if as_tuple else {'attacker_rank': ar, 'mimicked_rank': mr}
    return results


# ------------------------------------------------------------------------------
# Hybrid: weighted reciprocal-rank fusion (thesis §5.3 / Appendix 8.3; weights below)
# ------------------------------------------------------------------------------
WRRF_K = 60
METHOD_DIRS = {
    'xgboost': '1.XGboost', 'gnn': '2.GNN', 'siamese': '3.SiameseNetwork',
    'mil': '4.Attention-Based_MIL', 'rf': '5.RF_FeatureEngineering',
    'jaccard': 'Jaccards_Correlation', 'tvd': 'TVD_Total_Variation_Distance',
    'featureMatching': 'featureMatching',
}
METHODS = list(METHOD_DIRS)
# Weight configurations (hybrid ensemble approach), in eighths. Both sets were derived on the
# 6-feature version and are used unchanged in the 3-feature version (same weights in both versions).
# Found by 6.hybrid/weight_search.py on the 163-group dataset (the figures below are those of
# 6.hybrid/results/weight_search.txt):
#   self_eval_weights: chosen from the self-evaluation tests ONLY (sanity check = full profiles, stress
#       test = shuffled & divided profiles). Of the 16,777,216 weightings with every weight in 1/8..8/8,
#       2,617,721 rank the correct APT first for all 163 queries of BOTH tests (100% Top-1). Among these,
#       this weighting gives the largest mean relative margin between the correct APT and the runner-up
#       ((WRRF(correct) - WRRF(runner-up)) / WRRF(correct), averaged over the queries of both tests).
#   CTI_boosting: chosen from the CTI evaluation ONLY: the weighting (each weight 1/8..8/8) with the best
#       mean CTI score over the 29 CTI reports (6-feature version: 91.59).
#   No false-flag result was used to choose either set.
HYBRID_WEIGHTS = {
    'no_weights': {m: 1.0 for m in METHODS},
    'self_eval_weights': {'xgboost': 1/8, 'gnn': 2/8, 'siamese': 2/8, 'mil': 4/8,
                          'rf': 5/8, 'featureMatching': 8/8, 'jaccard': 1/8, 'tvd': 3/8},
    'CTI_boosting': {'xgboost': 1/8, 'gnn': 1/8, 'siamese': 1/8, 'mil': 1/8,
                     'rf': 1/8, 'featureMatching': 8/8, 'jaccard': 1/8, 'tvd': 1/8},
}


def hybrid_weights_lines():
    """The weight configurations as readable lines (methods from highest to lowest weight)."""
    def eighths(x):
        return f"{round(8 * x)}/8" if abs(8 * x - round(8 * x)) < 1e-12 else f"{x:g}"
    out = []
    for cfg, w in HYBRID_WEIGHTS.items():
        order = sorted(w, key=lambda m: (-w[m], METHODS.index(m)))
        out.append(f"  {cfg:18s} " + ", ".join(f"{m} {eighths(w[m])}" for m in order))
    return out


def print_hybrid_weights():
    print("Hybrid weights (common/protocol.py, HYBRID_WEIGHTS):")
    print("\n".join(hybrid_weights_lines()), flush=True)


def wrrf_scores(ranks_by_method, weights, k=WRRF_K):
    """WRRF(d) = sum_r w_r / (k + rank_r(d)). `ranks_by_method` maps method -> rank array
    (any shape, same for all methods). Weights are normalised to sum to 1 (ranking unchanged)."""
    total = sum(weights[m] for m in ranks_by_method)
    score = None
    for m, ranks in ranks_by_method.items():
        term = (weights[m] / total) / (k + np.asarray(ranks, dtype=np.float64))
        score = term if score is None else score + term
    return score


def wrrf_ranked(results_by_method, weights, k=WRRF_K):
    """Fuse ranked lists ({'apt_name', 'rank'}) from several methods into one ranked list."""
    names = apt_names()
    idx = {n: i for i, n in enumerate(names)}
    ranks = {}
    for m, lst in results_by_method.items():
        r = np.full(len(names), np.nan)
        for item in lst:
            r[idx[item['apt_name']]] = item['rank']
        if np.isnan(r).any():
            raise ValueError(f"{m}: ranked list does not cover all {len(names)} APTs")
        ranks[m] = r
    return ranked_results(names, wrrf_scores(ranks, weights, k), score_key='wrrf_score')


def method_results_dir(method, smoke=False):
    d = os.path.join(ROOT, METHOD_DIRS[method], "results")
    return os.path.join(d, "smoke") if smoke else d


def evidence_fingerprint(path=FF_EVIDENCE_FILE):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


_ORIGIN_MAPS = None


def encode_origin_evidence(evidence):
    """Human-readable evidence -> encoded IDs, using the dataset mapping (comparison_Encoded_df).
    Used to tell which CTI reports contain any evidence recognised by the dataset."""
    global _ORIGIN_MAPS
    if _ORIGIN_MAPS is None:
        cols = {'techniques': 'techniques', 'software': 'software', 'capec': 'CAPEC_ID',
                'cwe': 'CWE_ID', 'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
        dfm = pd.read_csv(os.path.join(BASE, "comparison_Encoded_df.csv"))
        _ORIGIN_MAPS = {}
        for f in FEATURES:
            pair = dfm[[f"{cols[f]}_Origin", f"{cols[f]}_Encoded"]].dropna()
            pair = pair[pair[f"{cols[f]}_Encoded"].astype(int) != MISSING]
            _ORIGIN_MAPS[f] = dict(zip(pair[f"{cols[f]}_Origin"].astype(str), pair[f"{cols[f]}_Encoded"].astype(int)))
    return {f: [_ORIGIN_MAPS[f][str(x)] for x in evidence.get(f, []) if str(x) in _ORIGIN_MAPS[f]] for f in FEATURES}


def smoke_flag():
    return "--smoke" in sys.argv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shared evaluation protocol utilities")
    ap.add_argument("command", choices=["build-ff-evidence", "describe"])
    args = ap.parse_args()
    if args.command == "build-ff-evidence":
        d = build_false_flag_evidence(force=True)
        print(f"Saved {FF_EVIDENCE_FILE}: {len(d['pairs'])} pairs, {len(d['excluded'])} excluded")
    else:
        d = load_false_flag_evidence()
        full_sizes, part_sizes = d['full'].sizes(), d['partial'].sizes()
        comp = d['composition']                                   # (pairs, [attacker, mimicked], features)
        n_att, n_mim = comp[:, 0].sum(axis=1), comp[:, 1].sum(axis=1)
        overlap = n_att + n_mim - full_sizes
        print(f"False-flag evidence: {len(d['pairs'])} ordered pairs, {len(d['excluded'])} excluded")
        print(f"  rule: {d['params']['rule']}")
        print(f"  full profile size    : mean {full_sizes.mean():.1f}, median {np.median(full_sizes):.0f}, "
              f"min {full_sizes.min()}, max {full_sizes.max()}")
        print(f"  partial profile size : mean {part_sizes.mean():.1f}, median {np.median(part_sizes):.0f}")
        print(f"  items from attacker  : mean {n_att.mean():.1f} | from mimicked APT: mean {n_mim.mean():.1f} | "
              f"in both (kept once): mean {overlap.mean():.1f}")
        print(f"  attacker share of the synthetic profile: mean {100 * np.mean(n_att / (n_att + n_mim)):.1f}%")
        print("  mean items per feature type     " + " ".join(f"{f:>10s}" for f in FEATURES))
        print("    from the attacker (first 30%) " + " ".join(f"{v:10.1f}" for v in comp[:, 0].mean(axis=0)))
        print("    from the mimicked (last 70%)  " + " ".join(f"{v:10.1f}" for v in comp[:, 1].mean(axis=0)))
