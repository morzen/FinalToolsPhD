"""
Hybrid weight search: try every combination of weights and keep the best one.

Each of the eight approaches gets a weight of 1/8, 2/8, ..., 8/8, so there are 8^8 = 16,777,216
combinations. Every combination is tried on results that are already saved (no model is re-run):
the hybrid ranking is rebuilt with WRRF (k = 60, exactly as in common/protocol.py) from each
approach's saved ranks, and scored.

SELF-EVALUATION weights - only the sanity check (full profiles) and the stress test (shuffled &
divided profiles) are used:
    1. keep the combinations that rank the correct APT first for every query of BOTH tests
       (100% Top-1 in both);
    2. among those, take the one where the correct APT wins with the largest lead over the
       second-ranked APT:  lead = (WRRF(correct) - WRRF(runner-up)) / WRRF(correct),
       averaged over all queries of the two tests. A large lead = a robust attribution.

CTI weights - only the CTI reports are used: the combination with the best mean CTI score
(thesis formula 100 (1 - (rank - 1) / (N - 1))). To check that this is not luck on a small set,
the choice is repeated leaving one report out each time (and leaving one APT group out each
time), and the left-out report is scored with the weights chosen without it.

If several combinations are equally good, the one closest to equal weights is kept.
The false-flag results are never used.

Needs (already produced by the normal run):
    <method>/results/attribution_ranks_full.npy and attribution_ranks_divided.npy  (eval.py)
    CTI_eval/CTI_method_ranks.npz                                                (CTI_eval.py)

The thesis uses the weights found in the 6-feature version for BOTH versions (run it in
FinalToolsPhD_6feat; run in FinalToolsPhD_3feat it gives that version's own optimum, for comparison).

Run from this version's folder:
    venv/bin/python 6.hybrid/weight_search.py                      full search (a few minutes)
    venv/bin/python 6.hybrid/weight_search.py --levels 1,2,4,8     quick test (65,536 combinations)
Results: printed, and saved in 6.hybrid/results/weight_search.json / weight_search.txt
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker process (the workers run in parallel)

import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import protocol as P

SHORT = {'xgboost': 'XGB', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'RF',
         'featureMatching': 'FM', 'jaccard': 'Jaccard', 'tvd': 'TVD'}
TABLE_ORDER = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'featureMatching', 'jaccard', 'tvd']
K = P.WRRF_K
CHUNK = 1 << 20                              # combinations handed to one worker at a time
BLOCK = 1 << 16                              # combinations computed together inside a worker (memory)

# ----------------------------------------------------------------------------------------------
# The grid of combinations: combination number i -> one weight level per method (P.METHODS order)
# ----------------------------------------------------------------------------------------------
_W = {}                                      # data shared with the worker processes


def _init(levels, sets):
    _W['lev'] = np.asarray(levels, dtype=np.float64)
    _W['pow'] = len(levels) ** np.arange(len(P.METHODS) - 1, -1, -1)
    _W.update(sets)


def weights_of(idx):
    idx = np.asarray(idx, dtype=np.int64)
    return _W['lev'][(idx[:, None] // _W['pow']) % len(_W['lev'])]


def prepare(ranks, targets):
    """For every query: how the correct APT compares with every other APT in every method.
    d[m, j] = 1/(k + rank of correct APT) - 1/(k + rank of APT j) in method m, so the fused
    WRRF score of the correct APT minus that of APT j is simply  weights . d[:, j].
    APTs that every method puts behind (or ahead of) the correct APT need no computing."""
    x = 1.0 / (K + ranks)                                  # (methods, queries, APTs)
    out = []
    for q, t in enumerate(targets):
        d = x[:, q, t][:, None] - np.delete(x[:, q, :], t, axis=1)
        neg, pos = (d < 0).any(0), (d > 0).any(0)
        out.append({'ahead': int((neg & ~pos).sum()),     # beat the correct APT whatever the weights
                    'tied': int((~neg & ~pos).sum()),      # identical in every method
                    'd': np.ascontiguousarray(d[:, neg & pos]), 'xt': x[:, q, t]})
    return out


def _diff(W, qd):
    """fused(correct) - fused(other) for the undecided APTs, and the rounding tolerance"""
    return W @ qd['d'], (1e-10 * (W @ qd['xt']))[:, None]


def _perfect_chunk(bounds):
    """combinations in [start, stop) that put the correct APT strictly first for every query"""
    alive = np.arange(*bounds, dtype=np.int64)
    for qd in _W['self_queries']:                          # hardest queries first
        if qd['d'].shape[1] == 0 or len(alive) == 0:
            continue
        keep = np.empty(len(alive), dtype=bool)
        for s in range(0, len(alive), BLOCK):
            diff, eps = _diff(weights_of(alive[s:s + BLOCK]), qd)
            keep[s:s + BLOCK] = (diff > eps).all(1)
        alive = alive[keep]
    return alive


def _lead_chunk(idx):
    """mean (and smallest) relative lead of the correct APT over the runner-up, over all queries"""
    W = weights_of(idx)
    W = W / W.sum(1, keepdims=True)
    total, worst = np.zeros(len(idx)), np.full(len(idx), np.inf)
    for xq, t in _W['self_x']:
        S = W @ xq
        st = S[:, t].copy()
        S[:, t] = -np.inf
        lead = (st - S.max(1)) / st
        total += lead
        worst = np.minimum(worst, lead)
    return total / len(_W['self_x']), worst


def _cti_chunk(bounds):
    """2 x mid-rank of the correct APT in every CTI report, for the combinations in [start, stop)"""
    start, stop = bounds
    out = np.empty((stop - start, len(_W['cti_queries'])), dtype=np.uint16)
    for s in range(start, stop, BLOCK):
        e = min(stop, s + BLOCK)
        W = weights_of(np.arange(s, e, dtype=np.int64))
        for r, qd in enumerate(_W['cti_queries']):
            base = 2 + 2 * qd['ahead'] + qd['tied']
            if qd['d'].shape[1] == 0:
                out[s - start:e - start, r] = base
                continue
            diff, eps = _diff(W, qd)
            behind = np.count_nonzero(diff < -eps, axis=1)
            not_ahead = np.count_nonzero(diff <= eps, axis=1)
            out[s - start:e - start, r] = base + 2 * behind + (not_ahead - behind)     # ties count half
    return out


def closest_to_equal(cands, W):
    p = W / W.sum(1, keepdims=True)
    return cands[np.argmin(((p - 1.0 / len(P.METHODS)) ** 2).sum(1))]


def best_cti(rank_sums):
    cands = np.flatnonzero(rank_sums == rank_sums.min())
    return closest_to_equal(cands, weights_of(cands)), len(cands)


def as_dict(w, levels):
    top = max(levels)
    return {m: float(w[k]) / top for k, m in enumerate(P.METHODS)}


def fmt(wd):
    top = 8
    return ", ".join(f"{SHORT[m]} {round(wd[m] * top)}/{top}" for m in TABLE_ORDER)


# ----------------------------------------------------------------------------------------------
# Exact check of a weighting with the protocol's own fusion and ranking functions
# ----------------------------------------------------------------------------------------------
def exact_scores(wd, attr, cti):
    out = {}
    n = len(P.apt_names())
    for scen in ('full', 'divided'):
        R = np.vstack([P.midranks(row) for row in P.wrrf_scores(attr[scen], wd)])
        out[scen] = 100.0 * np.mean(R[np.arange(n), np.arange(n)] <= 1)
    ranks = np.array([P.midranks(P.wrrf_scores({m: cti['ranks'][m][r] for m in P.METHODS}, wd))[t]
                      for r, t in enumerate(cti['targets'])])
    out['cti_score'] = float(np.mean([P.cti_score(x, n) for x in ranks]))
    out['cti_mean_rank'] = float(ranks.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3,4,5,6,7,8", help="weight levels (in eighths) to try")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    n_comb = len(levels) ** len(P.METHODS)
    t_start = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # ---- saved results -------------------------------------------------------------------
    names = P.apt_names()
    n = len(names)
    attr = {scen: {m: np.load(os.path.join(P.method_results_dir(m), f"attribution_ranks_{scen}.npy")).astype(np.float64)
                   for m in P.METHODS} for scen in ('full', 'divided')}
    for scen in attr:
        for m in P.METHODS:
            if attr[scen][m].shape != (n, n):
                sys.exit(f"{m}: attribution_ranks_{scen}.npy has shape {attr[scen][m].shape}; run eval.py first")
    z = np.load(os.path.join(ROOT, "CTI_eval", "CTI_method_ranks.npz"))
    if list(z['methods']) != P.METHODS or list(z['apt_names']) != names:
        sys.exit("CTI_method_ranks.npz does not match this folder; run CTI_eval/CTI_eval.py first")
    keep = ~z['combined']                                   # the 30 reports (not the combined evidence)
    cti = {'ranks': {m: z['ranks'][keep, k, :] for k, m in enumerate(P.METHODS)},
           'targets': z['true_index'][keep], 'groups': z['apt'][keep]}
    n_rep = len(cti['targets'])

    say(f"Hybrid weight search - {len(P.FEATURES)}-feature version")
    say(f"weights tried for every approach: {', '.join(f'{x}/8' for x in levels)} -> {n_comb:,} combinations")
    say(f"self-evaluation: {n} queries x 2 tests;  CTI: {n_rep} reports;  false flag: not used")

    stack = lambda d: np.stack([d[m] for m in P.METHODS])                   # (methods, queries, APTs)
    self_queries = prepare(stack(attr['full']), np.arange(n)) + prepare(stack(attr['divided']), np.arange(n))
    if any(q['ahead'] or q['tied'] for q in self_queries):
        sys.exit("Some query cannot be ranked first by any combination: 100% Top-1 is impossible here.")
    # hardest queries first (failure rate on a fixed random sample), so most combinations are discarded early
    _init(levels, {})
    sample = weights_of(np.random.default_rng(0).choice(n_comb, min(n_comb, 20000), replace=False))
    fail = []
    for q in self_queries:
        if q['d'].shape[1] == 0:
            fail.append(0.0)
        else:
            diff, eps = _diff(sample, q)
            fail.append(float(np.mean(~(diff > eps).all(1))))
    self_queries = [self_queries[i] for i in np.argsort(fail)[::-1]]
    xs = {scen: 1.0 / (K + stack(attr[scen])) for scen in ('full', 'divided')}
    self_x = [(np.ascontiguousarray(xs[scen][:, q, :]), q) for scen in ('full', 'divided') for q in range(n)]
    cti_queries = prepare(np.stack([cti['ranks'][m] for m in P.METHODS]), cti['targets'])
    shared = {'self_queries': self_queries, 'self_x': self_x, 'cti_queries': cti_queries}
    chunks = [(s, min(n_comb, s + CHUNK)) for s in range(0, n_comb, CHUNK)]

    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(chunks))),
                             initializer=_init, initargs=(levels, shared)) as pool:
        _init(levels, shared)
        # ---- SELF-EVALUATION weights -----------------------------------------------------
        t0 = time.time()
        perfect = np.concatenate(list(pool.map(_perfect_chunk, chunks)))
        say()
        say("SELF-EVALUATION WEIGHTS (sanity check + stress test only)")
        say(f"  step 1: {len(perfect):,} of {n_comb:,} combinations rank the correct APT first for all "
            f"{2 * n} queries (100% Top-1 in both tests)   [{time.time() - t0:.0f} s]")
        if len(perfect) == 0:
            sys.exit("No combination reaches 100% Top-1 in both tests.")
        t0 = time.time()
        parts = np.array_split(perfect, max(1, len(perfect) // 50000))
        res = list(pool.map(_lead_chunk, parts))
        lead = np.concatenate([r[0] for r in res])
        worst = np.concatenate([r[1] for r in res])
        cands = np.flatnonzero(lead >= lead.max() - 1e-12)
        i_self = perfect[closest_to_equal(cands, weights_of(perfect[cands]))]
        w_self = as_dict(weights_of([i_self])[0], [8])
        say(f"  step 2: largest mean lead of the correct APT over the runner-up: {100 * lead.max():.3f}% "
            f"(reached by {len(cands)} combination(s))   [{time.time() - t0:.0f} s]")
        say(f"  BEST SELF-EVALUATION WEIGHTS: {fmt(w_self)}")
        say("  the 5 best:")
        for j in np.argsort(-lead, kind='stable')[:5]:
            say(f"     lead {100 * lead[j]:.3f}%  (smallest {100 * worst[j]:.3f}%)   {fmt(as_dict(weights_of([perfect[j]])[0], [8]))}")
        lead_self = float(lead.max())

        # ---- CTI weights -------------------------------------------------------------------
        t0 = time.time()
        rk2 = np.empty((n_comb, n_rep), dtype=np.uint16)              # 2 x rank, every combination x report
        for (s, e), part in zip(chunks, pool.map(_cti_chunk, chunks)):
            rk2[s:e] = part
    sums = rk2.sum(1, dtype=np.int64)
    i_cti, n_tie = best_cti(sums)
    w_cti = as_dict(weights_of([i_cti])[0], [8])
    score = lambda r: 100.0 * (1.0 - (np.asarray(r, dtype=np.float64) - 1.0) / (n - 1.0))
    loo = [score(rk2[best_cti(sums - rk2[:, r])[0], r] / 2.0) for r in range(n_rep)]
    logo = []
    for g in np.unique(cti['groups']):
        mask = cti['groups'] == g
        logo += list(score(rk2[best_cti(sums - rk2[:, mask].sum(1, dtype=np.int64))[0], mask] / 2.0))
    say()
    say("CTI WEIGHTS (CTI reports only)")
    say(f"  best mean CTI score: {score(sums[i_cti] / 2.0 / n_rep):.2f} (mean rank {sums[i_cti] / 2.0 / n_rep:.2f}), "
        f"reached by {n_tie} combination(s)   [{time.time() - t0:.0f} s]")
    say(f"  BEST CTI WEIGHTS: {fmt(w_cti)}")
    say(f"  check against luck - choice redone without the report(s) it is tested on:")
    say(f"     leave one report out: CTI {np.mean(loo):.2f};  leave one APT group out: CTI {np.mean(logo):.2f}")
    say("  the 5 best:")
    for j in np.argsort(sums, kind='stable')[:5]:
        say(f"     CTI {score(sums[j] / 2.0 / n_rep):.2f} (mean rank {sums[j] / 2.0 / n_rep:.2f})   {fmt(as_dict(weights_of([j])[0], [8]))}")

    # ---- exact check with the protocol's own functions -------------------------------------
    say()
    say("EXACT CHECK with common/protocol.py (wrrf_scores + midranks)")
    say(f"  {'weights':34s} {'sanity Top-1':>12s} {'stress Top-1':>12s} {'CTI score':>10s} {'mean rank':>10s}")
    rows = [("found: self-evaluation", w_self), ("found: CTI", w_cti)] + \
           [(f"in protocol.py now: {c}", w) for c, w in P.HYBRID_WEIGHTS.items()]
    checks = {}
    for label, wd in rows:
        e = exact_scores(wd, attr, cti)
        checks[label] = e
        say(f"  {label:34s} {e['full']:12.2f} {e['divided']:12.2f} {e['cti_score']:10.2f} {e['cti_mean_rank']:10.2f}")
    assert checks["found: self-evaluation"]['full'] == 100 and checks["found: self-evaluation"]['divided'] == 100
    assert abs(checks["found: CTI"]['cti_score'] - score(sums[i_cti] / 2.0 / n_rep)) < 1e-9
    say()
    say("To use them, in common/protocol.py:")
    say("    'self_eval_weights': {" + ", ".join(f"'{m}': {round(w_self[m] * 8)}/8" for m in P.METHODS) + "},")
    say("    'CTI_boosting': {" + ", ".join(f"'{m}': {round(w_cti[m] * 8)}/8" for m in P.METHODS) + "},")
    say(f"total time {time.time() - t_start:.0f} s")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "weight_search.json"), "w") as fh:
        json.dump({'levels_in_eighths': levels, 'n_combinations': n_comb,
                   'self_eval_weights': w_self, 'self_eval_n_perfect': int(len(perfect)), 'self_eval_mean_lead': lead_self,
                   'CTI_boosting': w_cti, 'cti_score': score(sums[i_cti] / 2.0 / n_rep),
                   'cti_leave_one_report_out': float(np.mean(loo)), 'cti_leave_one_group_out': float(np.mean(logo)),
                   'exact_check': checks}, fh, indent=2)
    with open(os.path.join(HERE, "results", "weight_search.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
