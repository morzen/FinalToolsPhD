"""
Check, on YOUR real models, that the new code gives the same answers as your ORIGINAL code
(the copies kept in _backup_before_uniform_protocol_2026-10-22/).

For each of the 8 methods, in its own Python process:
  * a sample of false-flag pairs (full and partial scenario, spread over many different attackers)
    and a sample of APTs (full profile and divided profile) are scored by
      (a) your ORIGINAL code from the backup folder, fed with the human-readable evidence
          (the names, e.g. 'T1059.001', as a CTI report would give them), and
      (b) the NEW scorer used by eval.py, the false-flag scripts, the hybrid and CTI_eval;
    the scores and the ranks of all APTs are compared;
  * if a full run has been done (results/ of the method up to date), the ranks SAVED by that run
    are compared with the ranks from the original code as well.

GNN and XGBoost: the original evidence encoding (convert_evidence_to_encoded) takes ~30 s per call,
so it is checked on 3 samples and the original scoring function is then called on the encoded
evidence. In the other originals only file loading is cached (they re-read the CSV / model files on
every call); the first query is also run without any caching and must give the same result.
Siamese: the original draws its random combinations without a seed, so it can only agree up to
sampling noise; the report compares that noise (original vs original with another seed) with the
original-vs-new difference.

Run from this folder:  venv/bin/python verify_against_original.py        (about 20-40 min)
Options: --pairs N (per scenario, default 12)  --apts N (per scenario, default 12)
         --methods gnn,xgboost,...   --backup <folder with the original files>
The report is printed and saved in verify_report.txt.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import sys
import json
import time
import pickle
import argparse
import contextlib
import subprocess
import importlib.util

import numpy as np
import pandas as pd

# the tool folder to check: given by run_first_check.sh (PHD_TOOL_DIR); by default the folder of this file
HERE = os.path.abspath(os.environ.get("PHD_TOOL_DIR") or os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import protocol as P

FILES = {'xgboost': ('query.py', 'XGBScorer'), 'gnn': ('query.py', 'GNNScorer'),
         'siamese': ('query.py', 'SiameseScorer'), 'mil': ('query.py', 'MILScorer'),
         'rf': ('query.py', 'RFScorer'), 'jaccard': ('Jaccards_correlation.py', 'JaccardScorer'),
         'tvd': ('TVD_featureIng.py', 'TVDScorer'), 'featureMatching': ('featureMatching.py', 'FeatureMatchingScorer')}
ORDER = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'jaccard', 'tvd', 'featureMatching']
DEFAULT_BACKUP = os.path.join(HERE, "_backup_before_uniform_protocol_2026-10-22")
TAG = "VERIFY|"


# ------------------------------------------------------------------------------ helpers
def load_module(path, name, workdir, file_attr=None, extra_path=()):
    """Import a .py file as a module, run from `workdir` (its relative paths: ../BaseData, models)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if file_attr:
        module.__file__ = file_attr          # original code that locates its model via __file__
    cwd = os.getcwd()
    added = [p for p in list(extra_path) + [workdir] if p not in sys.path]
    for p in reversed(added):
        sys.path.insert(0, p)
    os.chdir(workdir)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
        for p in added:
            sys.path.remove(p)
    return module


class CachedRead:
    """Stands in for `pd` inside the original module: same pandas, but each CSV is read only once."""
    def __init__(self):
        self.cache = {}

    def __getattr__(self, k):
        return getattr(pd, k)

    def read_csv(self, path, *a, **kw):
        key = (os.path.abspath(str(path)), repr(a), repr(sorted(kw.items())))
        if key not in self.cache:
            self.cache[key] = pd.read_csv(path, *a, **kw)
        return self.cache[key]


def memoize(fn):
    store = {}

    def wrapped(*a, **kw):
        key = repr((a, sorted(kw.items()))) if all(isinstance(x, (str, int, float, type(None))) for x in a) else id(a[0])
        if key not in store:
            store[key] = fn(*a, **kw)
        return store[key]
    return wrapped


def decoder():
    dfm = pd.read_csv(os.path.join(P.BASE, "comparison_Encoded_df.csv"))
    col = {'techniques': 'techniques', 'software': 'software', 'capec': 'CAPEC_ID', 'cwe': 'CWE_ID',
           'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
    maps = {}
    for f in P.FEATURES:
        pr = dfm[[f"{col[f]}_Origin", f"{col[f]}_Encoded"]].dropna().drop_duplicates()
        pr = pr[pr[f"{col[f]}_Encoded"].astype(int) != P.MISSING]
        maps[f] = dict(zip(pr[f"{col[f]}_Encoded"].astype(int), pr[f"{col[f]}_Origin"]))
    return lambda ev: {f: [maps[f][int(x)] for x in v] for f, v in ev.items()}


def score_dict(results):
    """Original result list -> {apt_name: score}."""
    out = {}
    for r in results:
        for k in ('Score', 'similarity', 'score'):
            if k in r:
                out[r['apt_name']] = float(r[k])
                break
    return out


def spearman(a, b):
    ra, rb = P.midranks(a), P.midranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------ one method (child process)
def check(method, n_pairs, n_apts, backup):
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    sub = P.METHOD_DIRS[method]
    live = os.path.join(HERE, sub)
    fname, cls = FILES[method]
    orig_path = os.path.join(backup, sub, fname)
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"original file not found: {orig_path}")
    main = sys.modules['__main__']

    new = load_module(os.path.join(live, fname), f"new_{method}", live)
    orig = load_module(orig_path, f"orig_{method}", live, file_attr=os.path.join(live, fname),
                       extra_path=[os.path.join(backup, sub)])
    if method == 'xgboost':                  # the XGBoost pickle refers to __main__.PicklableXGBWrapper
        main.PicklableXGBWrapper = new.PicklableXGBWrapper
    cwd = os.getcwd()
    os.chdir(live)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scorer = getattr(new, cls)()
    finally:
        os.chdir(cwd)
    if method == 'xgboost':
        main.PicklableXGBWrapper = orig.PicklableXGBWrapper
    align = P.alignment(scorer.apt_names)
    names = P.apt_names()
    decode = decoder()
    notes = []

    # --- how the original code is called -------------------------------------------------
    def run_orig(fn, *args):
        os.chdir(live)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                res = fn(*args)
        finally:
            os.chdir(cwd)
        return res, buf.getvalue()

    if method in ('gnn', 'xgboost'):
        mapping_file = "../BaseData/comparison_Encoded_df.csv"

        def orig_scores(ev):
            if method == 'gnn':
                res, _ = run_orig(lambda e: orig.query_apt_similarity_enhanced(e, "hybrid")[0], ev)
            else:
                res, _ = run_orig(lambda e: orig.query_apt_similarity_efficient(e)[0], ev)
            return score_dict(res)
    else:
        def orig_scores(ev):
            res, text = run_orig(orig.hybrid_query, decode(ev))
            if method == 'siamese' and "Exact match found" in text:
                orig_scores.shortcut += 1
            return score_dict(res)
        orig_scores.shortcut = 0

    data = P.load_false_flag_evidence()
    n_all = len(data['pairs'])
    pair_idx = sorted(set(np.linspace(0, n_all - 1, n_pairs).round().astype(int).tolist()))
    apt_idx = sorted(set(np.linspace(0, len(names) - 1, n_apts).round().astype(int).tolist()))
    profiles = P.load_profiles()

    # --- evidence encoding of the original (GNN / XGBoost), 3 samples, no caching -----------
    if method in ('gnn', 'xgboost'):
        ok = 0
        t0 = time.time()
        for i in pair_idx[:3]:
            ev = data['full'][i]
            enc, _ = run_orig(orig.convert_evidence_to_encoded, decode(ev), mapping_file)
            ok += int(all(set(map(int, enc.get(f, []))) == set(ev.get(f, [])) for f in P.FEATURES))
        notes.append(f"original evidence encoding identical on {ok}/3 "
                     f"(original encoding alone: {(time.time() - t0) / 3:.1f} s per query)")
        encoding_ok = ok == 3
    else:
        # first call fully original (no caching), then cache file loading only
        ev0 = data['full'][pair_idx[0]]
        if method == 'siamese':
            np.random.seed(1)
        t0 = time.time()
        first = orig_scores(ev0)
        notes.append(f"one fully original query (files re-read as in the original): {time.time() - t0:.1f} s")
        orig.pd = CachedRead()
        for fn in ('create_apt_profiles_with_names', 'load_model', 'load_complete_model',
                   'load_model_and_data', 'load_mapping_and_profiles'):
            if hasattr(orig, fn):
                setattr(orig, fn, memoize(getattr(orig, fn)))
        if method == 'siamese':
            np.random.seed(1)
        again = orig_scores(ev0)
        if method == 'siamese':
            orig_scores.shortcut = 0
        same = all(abs(first[n] - again[n]) <= 1e-12 for n in names)
        notes.append("cached original == uncached original" if same else "CACHED ORIGINAL DIFFERS")
        encoding_ok = same

    # --- saved results of a full run, if up to date ------------------------------------------
    rdir = os.path.join(live, "results")
    saved_ff = {}
    try:
        with open(os.path.join(rdir, "false_flag_summary.json")) as fh:
            summ = json.load(fh)
        if summ.get('evidence_fingerprint') == P.evidence_fingerprint() and summ.get('n_pairs') == n_all:
            saved_ff = {s: np.load(os.path.join(rdir, f"ff_ranks_{s}.npy")) for s in ("full", "partial")}
    except (OSError, ValueError):
        pass
    saved_attr = {}
    try:
        with open(os.path.join(rdir, "attribution_eval.json")) as fh:
            meta = json.load(fh)
        if meta.get('n_queries') == len(names) and meta.get('divided_seed') == P.DIVIDED_SEED:
            saved_attr = {s: np.load(os.path.join(rdir, f"attribution_ranks_{s}.npy")) for s in ("full", "divided")}
    except (OSError, ValueError):
        pass

    # --- the comparison ------------------------------------------------------------------------
    items = []
    for scen in ("full", "partial"):
        items += [(f"ff_{scen}", data[scen][i], saved_ff.get(scen, [None] * n_all)[i] if saved_ff else None) for i in pair_idx]
    for scen in ("full", "divided"):
        for i in apt_idx:
            ev = P.full_evidence(profiles[names[i]]) if scen == "full" else P.divided_evidence(names[i], profiles[names[i]])
            items.append((f"attr_{scen}", ev, saved_attr[scen][i] if saved_attr else None))

    TOL = 1e-6      # scores closer than this count as tied (float rounding, e.g. MIL batched in float32)

    def inversions(scores, ranks):
        """Pairs of APTs ranked in the wrong order although their scores differ by more than TOL."""
        better = (scores[:, None] - scores[None, :]) > TOL
        return int(np.sum(better & (ranks[:, None] > ranks[None, :])))

    max_diff, rank_same, inv_free, saved_ok, saved_n, rho, rho_noise, top1_same = 0.0, 0, 0, 0, 0, [], [], 0
    t_orig = t_new = 0.0
    for kind, ev, saved_row in items:
        if method == 'siamese':
            np.random.seed(1)
        t0 = time.time()
        o = orig_scores(ev)
        t_orig += time.time() - t0
        o = np.array([o[n] for n in names])
        t0 = time.time()
        s = np.asarray(scorer.score(ev), dtype=float)[align]
        t_new += time.time() - t0
        ro, rs = P.midranks(o), P.midranks(s)
        max_diff = max(max_diff, float(np.max(np.abs(o - s))))
        rank_same += int(np.array_equal(ro, rs))
        inv_free += int(inversions(o, rs) == 0)
        top1_same += int(np.argmax(o) == np.argmax(s))
        if saved_row is not None:
            saved_n += 1
            ref = s if method == 'siamese' else o      # Siamese: the saved run must equal the new scorer
            saved_ok += int(inversions(ref, np.asarray(saved_row, dtype=np.float64)) == 0)
        if method == 'siamese':
            np.random.seed(2)
            o2 = orig_scores(ev)
            o2 = np.array([o2[n] for n in names])
            rho.append(spearman(o, s))
            rho_noise.append(spearman(o, o2))

    n = len(items)
    res = {'method': method, 'queries': n, 'max_score_diff': max_diff, 'ranks_identical': rank_same,
           'ranks_consistent': inv_free, 'top1_identical': top1_same, 'saved_checked': saved_n,
           'saved_consistent': saved_ok, 'orig_s_per_query': t_orig / n, 'new_ms_per_query': 1000 * t_new / n,
           'notes': notes}
    if method == 'siamese':
        res['spearman_orig_vs_new'] = float(np.mean(rho))
        res['spearman_orig_vs_orig_other_seed'] = float(np.mean(rho_noise))
        res['original_exact_match_shortcut_used'] = orig_scores.shortcut
        verdict_ok = encoding_ok and res['spearman_orig_vs_new'] >= res['spearman_orig_vs_orig_other_seed'] - 0.05 \
            and saved_ok == saved_n
    else:
        verdict_ok = encoding_ok and inv_free == n and max_diff < 1e-5 and saved_ok == saved_n
    res['ok'] = bool(verdict_ok)
    print(TAG + json.dumps(res), flush=True)
    return 0 if verdict_ok else 1


# ------------------------------------------------------------------------------ parent
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method")
    ap.add_argument("--methods", default=",".join(ORDER))
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--apts", type=int, default=12)
    ap.add_argument("--backup", default=DEFAULT_BACKUP)
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("PHD_VERIFY_TIMEOUT", "3600")))
    args = ap.parse_args()
    if args.method:
        return check(args.method, args.pairs, args.apts, args.backup)

    lines = [f"Verification against the original code ({args.backup})",
             f"Features: {P.FEATURES}; false-flag evidence fingerprint {P.evidence_fingerprint()[:12]}",
             f"Per method: {args.pairs} false-flag pairs x 2 scenarios + {args.apts} APTs x 2 scenarios", ""]
    print("\n".join(lines), flush=True)
    all_ok = True
    for m in args.methods.split(","):
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--method", m, "--pairs", str(args.pairs),
                                "--apts", str(args.apts), "--backup", args.backup], cwd=HERE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
            out = r.stdout.splitlines()
        except subprocess.TimeoutExpired:
            out = []
        found = [json.loads(l[len(TAG):]) for l in out if l.startswith(TAG)]
        if not found:
            all_ok = False
            msg = [f"{m:16s} FAILED or no answer; last lines:"] + ["    " + l for l in out[-12:]]
        else:
            x = found[-1]
            all_ok &= x['ok']
            msg = [f"{m:16s} {'OK ' if x['ok'] else 'DIFFERENT'}  {x['queries']} queries | max |score diff| "
                   f"{x['max_score_diff']:.1e} | ranks of all APTs identical {x['ranks_identical']}/{x['queries']}, "
                   f"same order up to ties < 1e-6 {x['ranks_consistent']}/{x['queries']} | top-1 identical "
                   f"{x['top1_identical']}/{x['queries']} | saved results consistent "
                   f"{x['saved_consistent']}/{x['saved_checked']} | original {x['orig_s_per_query']:.2f} s/query, "
                   f"new {x['new_ms_per_query']:.0f} ms/query  [{time.time() - t0:.0f}s]"]
            msg += [f"{'':16s} {n}" for n in x['notes']]
            if m == 'siamese':
                msg.append(f"{'':16s} rank correlation original vs new {x['spearman_orig_vs_new']:.3f}; original vs "
                           f"original with another seed {x['spearman_orig_vs_orig_other_seed']:.3f} (sampling noise); "
                           f"original exact-match shortcut used in {x['original_exact_match_shortcut_used']} queries")
        print("\n".join(msg), flush=True)
        lines += msg
    lines += ["", "RESULT: " + ("every method gives the same results as the original code"
                                if all_ok else "SEE THE LINES MARKED DIFFERENT / FAILED")]
    print(lines[-1])
    with open(os.path.join(HERE, "verify_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
