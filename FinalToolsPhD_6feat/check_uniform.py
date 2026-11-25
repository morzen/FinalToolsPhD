"""
Quick check (a few minutes) that every method is wired to the shared protocol, on the REAL models.

  * the false-flag evidence file exists (first 30% of the attacker profile + last 70% of the
    mimicked profile);
  * each method's scorer loads once and covers the same APTs as the protocol;
  * GNN / XGBoost: the scorer gives exactly the scores of the original query function;
  * MIL: the batched scorer is checked against calculate_similarities (printed at start-up);
  * Siamese: the query is reproducible (fixed seed);
  * every hybrid_query returns all APTs with mid-ranks;
  * time per evidence -> estimated false-flag run time.

Each method is checked in its own Python process, as eval.py and the false-flag scripts run, so
PyTorch and XGBoost are never loaded together (on macOS their OpenMP libraries can freeze a
process that holds both), and a method that does not answer within PHD_CHECK_TIMEOUT seconds
(default 900) is reported instead of blocking the run.

Run from this folder:  venv/bin/python check_uniform.py        (exit code 1 if a check fails)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import sys
import time
import json
import contextlib
import subprocess
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import protocol as P

FILES = {'xgboost': ('query.py', 'XGBScorer'), 'gnn': ('query.py', 'GNNScorer'),
         'siamese': ('query.py', 'SiameseScorer'), 'mil': ('query.py', 'MILScorer'),
         'rf': ('query.py', 'RFScorer'), 'jaccard': ('Jaccards_correlation.py', 'JaccardScorer'),
         'tvd': ('TVD_featureIng.py', 'TVDScorer'), 'featureMatching': ('featureMatching.py', 'FeatureMatchingScorer')}
TIMEOUT = int(os.environ.get("PHD_CHECK_TIMEOUT", "900"))
ROW, PROBLEM = "ROW|", "PROBLEM|"


def load(method):
    folder = os.path.join(HERE, P.METHOD_DIRS[method])
    spec = importlib.util.spec_from_file_location(f"check_{method}", os.path.join(folder, FILES[method][0]))
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(folder)
    sys.path.insert(0, folder)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        os.chdir(cwd)
    return module


def check_method(method):
    """Runs in a child process: checks one method, prints its table row and any problem."""
    try:
        import torch
        torch.set_num_threads(1)               # as in the false-flag workers
    except ImportError:
        pass
    module = load(method)
    if method == 'xgboost':
        # The XGBoost model was pickled from train.py run as __main__: expose its wrapper class here.
        globals()['PicklableXGBWrapper'] = module.PicklableXGBWrapper
    problems = []
    data = P.load_false_flag_evidence()
    with open(os.path.join(HERE, "CTI_eval", "CTI_Eval_empty_template.json")) as fh:
        report = json.load(fh)['APT29'][0]['Data']
    profiles = P.load_profiles()
    names = P.apt_names()
    test_evidence = [data['full'][0], data['partial'][len(data['pairs']) // 2],
                     P.divided_evidence(names[3], profiles[names[3]])]
    n_ff = 2 * len(data['pairs'])

    folder = os.path.join(HERE, P.METHOD_DIRS[method])
    cwd = os.getcwd()
    t0 = time.time()
    os.chdir(folder)
    try:
        scorer = getattr(module, FILES[method][1])()
    finally:
        os.chdir(cwd)
    t_load = time.time() - t0
    notes = []
    if sorted(scorer.apt_names) != names:
        problems.append(f"{method}: APT names differ from the dataset")
        notes.append("NAMES MISMATCH")
    t0 = time.time()
    scores = [np.asarray(scorer.score(e), dtype=float) for e in test_evidence]
    per = (time.time() - t0) / len(test_evidence)
    if method == 'gnn':
        with contextlib.redirect_stdout(io.StringIO()):
            os.chdir(folder)
            res, *_ = module.query_apt_similarity_enhanced({f: test_evidence[0].get(f, []) for f in P.FEATURES}, "nothybrid")
            os.chdir(cwd)
        ref = {r['apt_id']: r['similarity'] for r in res}
        diff = max(abs(ref[a] - v) for a, v in zip(scorer.apt_ids, scores[0]))
        notes.append(f"vs query_apt_similarity_enhanced {diff:.1e}")
        if diff > 1e-5:
            problems.append(f"gnn: scorer differs from query_apt_similarity_enhanced ({diff:.2e})")
    if method == 'xgboost':
        with contextlib.redirect_stdout(io.StringIO()):
            os.chdir(folder)
            res, _ = module.query_apt_similarity_efficient({f: test_evidence[0].get(f, []) for f in P.FEATURES})
            os.chdir(cwd)
        ref = {r['apt_id']: r['similarity'] for r in res}
        diff = max(abs(ref[a] - v) for a, v in zip(scorer.apt_ids, scores[0]))
        notes.append(f"vs query_apt_similarity_efficient {diff:.1e}")
        if diff > 1e-6:
            problems.append(f"xgboost: scorer differs from query_apt_similarity_efficient ({diff:.2e})")
        vocab_has_missing = any(-1 in scorer.feature_metadata[f"all_{f}"] for f in P.FEATURES)
        notes.append("model trained WITHOUT -1" if not vocab_has_missing else "model still has -1 features: run train.py")
    if method == 'siamese':
        same = np.array_equal(scorer.score(test_evidence[0]), scores[0])
        notes.append(f"reproducible {same}")
        if not same:
            problems.append("siamese: query not reproducible")
    if method == 'mil':
        notes.append("batched path" if scorer.fast else "ORIGINAL path (slower)")
    with contextlib.redirect_stdout(io.StringIO()):
        os.chdir(folder)
        ranked = module.hybrid_query(report)
        os.chdir(cwd)
    if len(ranked) != len(names) or not all('rank' in r and 'apt_name' in r for r in ranked):
        problems.append(f"{method}: hybrid_query output incomplete")
    est = n_ff * per / min(P.MAX_WORKERS, 8) / 60
    print(f"{ROW}{method:16s} {t_load:6.1f}s {1000 * per:11.1f} {est:24.0f} min  {'; '.join(notes)}", flush=True)
    for p in problems:
        print(f"{PROBLEM}{p}", flush=True)
    return problems


def main():
    if "--method" in sys.argv:
        return 1 if check_method(sys.argv[sys.argv.index("--method") + 1]) else 0

    data = P.load_false_flag_evidence()
    print(f"Features: {P.FEATURES}")
    print(f"False-flag evidence: {len(data['pairs'])} ordered pairs ({len(data['excluded'])} excluded), "
          f"fingerprint {P.evidence_fingerprint()[:12]}")
    workers = min(P.MAX_WORKERS, 8)
    print(f"\n{'method':16s} {'load':>7s} {'ms/evidence':>12s} {f'est. false-flag ({workers} workers)':>28s}  checks",
          flush=True)
    problems = []
    limit = f"{TIMEOUT // 60} min" if TIMEOUT >= 60 else f"{TIMEOUT} s"
    for method in P.METHODS:
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--method", method], cwd=HERE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"{method:16s} NO ANSWER after {limit}: stopped", flush=True)
            problems.append(f"{method}: the check did not finish within {limit}")
            continue
        lines = r.stdout.splitlines()
        for line in lines:
            if line.startswith("[MILScorer]"):
                print(line)
        rows = [line[len(ROW):] for line in lines if line.startswith(ROW)]
        found = [line[len(PROBLEM):] for line in lines if line.startswith(PROBLEM)]
        if rows:
            print(rows[-1], flush=True)
        if found:
            problems += found
        elif r.returncode != 0 or not rows:
            print(f"{method:16s} FAILED (exit code {r.returncode}); last lines of its output:")
            print("    " + "\n    ".join(lines[-15:]), flush=True)
            problems.append(f"{method}: the check crashed (exit code {r.returncode})")
    print("\nRESULT:", "ALL CHECKS PASSED" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
