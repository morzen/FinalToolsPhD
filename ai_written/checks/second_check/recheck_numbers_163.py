"""
Second check of Appendix F, for the final results (163 ATT&CK groups).

Independent recomputation of every reported number from the files saved by the run
(numpy + pandas only; no project code imported).

    python3 -B recheck_numbers_163.py <FinalToolsPhD_6feat or FinalToolsPhD_3feat folder>      (or run_second_check.sh)

It only reads: the saved rank and score matrices of the eight approaches, the saved hybrid ranks, the result
files (.json), the CTI result files and the summary of the last archived run. It writes nothing.

The hybrid weights typed below are those of Table 5.1 of the thesis. They are typed here from the thesis, not
read from common/protocol.py, so that a wrong weight in the code would be reported.
Written by an AI model (Claude, Anthropic).
"""
import sys, os, json, pickle, hashlib
import numpy as np, pandas as pd

F = sys.argv[1]
R_ = lambda *p: os.path.join(F, *p)
METHODS = {'xgboost': '1.XGboost', 'gnn': '2.GNN', 'siamese': '3.SiameseNetwork', 'mil': '4.Attention-Based_MIL',
           'rf': '5.RF_FeatureEngineering', 'jaccard': 'Jaccards_Correlation', 'tvd': 'TVD_Total_Variation_Distance',
           'featureMatching': 'featureMatching'}
# hybrid weights (eighths), typed here independently of common/protocol.py (same in both versions)
VERSION = '6feat' if '6feat' in os.path.basename(os.path.normpath(F)) else '3feat' if '3feat' in os.path.basename(os.path.normpath(F)) else None
assert VERSION, "folder name must contain 6feat or 3feat"
W = {'no_weights': {m: 8 for m in METHODS},          # same weights in both versions (derived on 6feat)
     'self_eval_weights': {'xgboost': 1, 'gnn': 2, 'siamese': 2, 'mil': 4, 'rf': 5, 'featureMatching': 8, 'jaccard': 1, 'tvd': 3},
     'CTI_boosting': {'xgboost': 1, 'gnn': 1, 'siamese': 1, 'mil': 1, 'rf': 1, 'featureMatching': 8, 'jaccard': 1, 'tvd': 1}}
def recorded_ok(rec, label):
    """the weights recorded in a result file must be these (as fractions of 8)"""
    if not rec: bad(f"{label}: no weights recorded (computed before the weights were recorded)"); return
    for cfg, w in W.items():
        if any(abs(8 * float(rec[cfg][m]) - w[m]) > 1e-9 for m in METHODS): bad(f"{label}: recorded {cfg} weights differ: {rec[cfg]}")
K = [1, 5, 10, 20]
problems = []
def bad(msg): problems.append(msg); print("  PROBLEM:", msg)

def midranks(s):
    """average ranks, rank 1 = highest; same 12-significant-digit canonicalisation as the protocol"""
    s = np.asarray(s, dtype=np.float64).copy()
    s[np.isnan(s)] = -np.inf
    nz = np.isfinite(s) & (s != 0)
    mag = 10.0 ** np.floor(np.log10(np.abs(s[nz])))
    s[nz] = np.round(s[nz] / mag, 11) * mag
    order = np.argsort(-s, kind='mergesort'); ss = s[order]
    r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        r[order[i:j + 1]] = (i + j + 2) / 2.0; i = j + 1
    return r

# canonical APT names straight from the mapping file
names = sorted(set(pd.read_csv(R_("BaseData", "comparison_Encoded_df.csv"), usecols=['APTnameOrigin'])['APTnameOrigin'].astype(str)))
N = len(names); idx = {n: i for i, n in enumerate(names)}
RANKSUM = N * (N + 1) / 2
print(f"{F}\n  {N} APTs; hybrid weights checked: " + "; ".join(f"{c} " + " ".join(f"{m}={w[m]}" for m in sorted(w, key=lambda m: -w[m])) for c, w in W.items() if c != 'no_weights'))

# ---------------- evidence / pairs
ev_path = R_("BaseData", "false_flag_evidence.pkl")
fp = hashlib.md5(open(ev_path, 'rb').read()).hexdigest()
ev = pickle.load(open(ev_path, 'rb'))
pairs = ev['pairs']; a_idx = np.array([idx[a] for a, _ in pairs]); m_idx = np.array([idx[m] for _, m in pairs]); rows = np.arange(len(pairs))
print(f"  evidence fingerprint {fp[:12]}, {len(pairs)} pairs")
if ev['apt_names'] != names: bad("evidence APT order differs from the sorted names")

def ff_summary(R):
    out = {}
    ar, mr = R[rows, a_idx], R[rows, m_idx]
    for k in K:
        ad, ma = ar <= k, mr <= k
        out[k] = {'AD': 100 * ad.mean(), 'MA': 100 * ma.mean(), 'Both': 100 * (ad & ma).mean(), 'Neither': 100 * (~ad & ~ma).mean()}
    return out

def check_matrix(R, S, label):
    if R.shape[1] != N: bad(f"{label}: {R.shape[1]} columns")
    sums = R.sum(axis=1)
    if not np.allclose(sums, RANKSUM, atol=1e-3): bad(f"{label}: {int(np.sum(~np.isclose(sums, RANKSUM, atol=1e-3)))} rows are not a ranking of all APTs")
    if R.min() < 1 or R.max() > N: bad(f"{label}: ranks outside 1..{N}")
    if S is not None:
        if not np.all(np.isfinite(S)): bad(f"{label}: non-finite scores")
        # ranks must follow the scores (scores saved in float32: tolerate differences below 1e-6 relative)
        sample = np.linspace(0, len(R) - 1, min(len(R), 3000)).astype(int)
        inv = 0
        for i in sample:
            s, r = S[i].astype(np.float64), R[i]
            tol = 1e-6 * max(1e-12, np.max(np.abs(s)))
            inv += int(np.sum(((s[:, None] - s[None, :]) > tol) & (r[:, None] > r[None, :])))
        if inv: bad(f"{label}: {inv} rank inversions vs saved scores on {len(sample)} rows")

summary = {'attr': {}, 'ff': {}}
attr_R, ff_R = {}, {}
for m, d in METHODS.items():
    rd = R_(d, "results")
    meta = json.load(open(os.path.join(rd, "attribution_eval.json")))
    if meta['apt_names'] != names: bad(f"{m}: attribution APT order differs")
    if meta.get('n_queries') != N: bad(f"{m}: attribution on {meta.get('n_queries')} queries")
    attr_R[m] = {}
    for scen in ("full", "divided"):
        R = np.load(os.path.join(rd, f"attribution_ranks_{scen}.npy")).astype(np.float64)
        S = np.load(os.path.join(rd, f"attribution_scores_{scen}.npy"))
        check_matrix(R, S, f"{m} attribution {scen}")
        tr = R[np.arange(N), np.arange(N)]                     # query i is APT i
        top = {k: 100 * np.mean(tr <= k) for k in K}
        rep = {int(k): v for k, v in meta['results'][scen]['top_k'].items()}
        if any(abs(top[k] - rep[k]) > 1e-9 for k in K): bad(f"{m} attribution {scen}: recomputed {top} != reported {rep}")
        attr_R[m][scen] = R
        summary['attr'].setdefault(m, {})[scen] = top
    s = json.load(open(os.path.join(rd, "false_flag_summary.json")))
    if s['evidence_fingerprint'] != fp: bad(f"{m}: false-flag results from other evidence")
    if s['n_pairs'] != len(pairs): bad(f"{m}: false flag on {s['n_pairs']} pairs")
    ff_R[m] = {}
    for scen in ("full", "partial"):
        R = np.load(os.path.join(rd, f"ff_ranks_{scen}.npy")).astype(np.float64)
        S = np.load(os.path.join(rd, f"ff_scores_{scen}.npy"))
        if R.shape[0] != len(pairs): bad(f"{m} ff {scen}: {R.shape[0]} rows")
        check_matrix(R, S, f"{m} ff {scen}")
        mine = ff_summary(R)
        for k in K:
            for x in ('AD', 'MA', 'Both', 'Neither'):
                if abs(mine[k][x] - s['results'][scen][str(k)][x]) > 1e-9: bad(f"{m} ff {scen} top{k} {x}: {mine[k][x]} != {s['results'][scen][str(k)][x]}")
        ff_R[m][scen] = R
        summary['ff'].setdefault(m, {})[scen] = mine
    print(f"  {m:16s} checked: attribution 2x{N} rows, false flag 2x{len(pairs)} rows")

# ---------------- hybrid, recomputed from the eight methods' saved ranks
hd = R_("6.hybrid", "results")
ha = json.load(open(os.path.join(hd, "hybrid_attribution_eval.json")))
hf = json.load(open(os.path.join(hd, "hybrid_false_flag_summary.json")))
if hf['evidence_fingerprint'] != fp: bad("hybrid false flag from other evidence")
recorded_ok(ha.get('_weights'), "hybrid_attribution_eval.json"); recorded_ok(hf.get('weights'), "hybrid_false_flag_summary.json")
for cfg, w in W.items():
    tot = sum(w.values())
    for scen in ("full", "divided"):
        fused = sum((w[m] / tot) / (60.0 + attr_R[m][scen]) for m in METHODS)
        R = np.vstack([midranks(r) for r in fused])
        saved = np.load(os.path.join(hd, f"hybrid_{cfg}_ranks_{scen}.npy")).astype(np.float64)
        if not np.array_equal(R, saved): bad(f"hybrid {cfg} attribution {scen}: {int(np.sum(R != saved))} ranks differ from saved")
        tr = R[np.arange(N), np.arange(N)]
        top = {k: 100 * np.mean(tr <= k) for k in K}
        if any(abs(top[k] - ha[cfg][scen][str(k)]) > 1e-9 for k in K): bad(f"hybrid {cfg} attribution {scen}: {top} != {ha[cfg][scen]}")
        summary['attr'][f"hybrid:{cfg}"] = summary['attr'].get(f"hybrid:{cfg}", {}); summary['attr'][f"hybrid:{cfg}"][scen] = top
    for scen in ("full", "partial"):
        fused = sum((w[m] / tot) / (60.0 + ff_R[m][scen]) for m in METHODS)
        R = np.vstack([midranks(r) for r in fused])
        saved = np.load(os.path.join(hd, f"hybrid_{cfg}_ff_ranks_{scen}.npy")).astype(np.float64)
        if not np.array_equal(R, saved): bad(f"hybrid {cfg} ff {scen}: {int(np.sum(R != saved))} ranks differ from saved")
        mine = ff_summary(R)
        for k in K:
            for x in ('AD', 'MA', 'Both', 'Neither'):
                if abs(mine[k][x] - hf['results'][cfg][scen][str(k)][x]) > 1e-9: bad(f"hybrid {cfg} ff {scen} top{k} {x}")
        summary['ff'][f"hybrid:{cfg}"] = summary['ff'].get(f"hybrid:{cfg}", {}); summary['ff'][f"hybrid:{cfg}"][scen] = mine
    print(f"  hybrid {cfg:18s} recomputed independently from the 8 methods: attribution + false flag")

# ---------------- CTI
cti = {cfg: json.load(open(R_("CTI_eval", f"CTI_Eval_{cfg}.json"))) for cfg in W}
cs = json.load(open(R_("CTI_eval", "CTI_summary.json")))
recorded_ok(cs.get('hybrid_weights'), "CTI_summary.json")
tmpl = json.load(open(R_("CTI_eval", "CTI_Eval_empty_template.json")))
keys = ['XGboost', 'GNN', 'Siamesenetwork', 'MIL', 'RF', 'Jacccards', 'TVD', 'featureMatching']
ranks = {k: [] for k in keys}; hyb = {cfg: [] for cfg in W}
n_rep = 0
for apt, reps in cti['no_weights'].items():
    tl = [r['report_link'] for r in tmpl[apt]]
    real = [r for r in reps if r['report_link'] != "combination_of_evidence"]
    if [r['report_link'] for r in real] != tl: bad(f"CTI {apt}: reports differ from the template")
    for i, r in enumerate(real):
        n_rep += 1
        for k in keys:
            vals = {cfg: cti[cfg][apt][i]['results_ranks'][k] for cfg in W}
            if len(set(vals.values())) != 1: bad(f"CTI {apt} report {i} {k}: differs between the three files")
            ranks[k].append(r['results_ranks'][k])
        for cfg in W:
            hyb[cfg].append(cti[cfg][apt][i]['results_ranks']['hybrid'])
score = lambda r: float(np.mean([100 * (1 - (x - 1) / (N - 1)) for x in r]))
if n_rep != cs['n_reports']: bad(f"CTI: {n_rep} reports vs {cs['n_reports']}")
for k in keys:
    if abs(score(ranks[k]) - cs['cti_score'][k]) > 1e-9 or abs(np.mean(ranks[k]) - cs['mean_rank'][k]) > 1e-9: bad(f"CTI {k}: recomputed score differs")
for cfg in W:
    if abs(score(hyb[cfg]) - cs['cti_score'][f"hybrid_{cfg}"]) > 1e-9: bad(f"CTI hybrid {cfg}: recomputed score differs")
print(f"  CTI: {n_rep} reports, scores recomputed for 8 methods + 3 hybrids")

# ---------------- the summary.txt and excel_blocks.csv given to the user
arch = sorted(d for d in os.listdir(R_("results_archive")) if not d.startswith("smoke"))[-1]
txt = open(R_("results_archive", arch, "summary.txt")).read().splitlines()
csvt = open(R_("results_archive", arch, "excel_blocks.csv")).read()
label = {'xgboost': 'Xgboost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'RF', 'featureMatching': 'FeatureMatch',
         'jaccard': "Jaccard's", 'tvd': 'TVD', 'hybrid:no_weights': 'Hybrid no weight', 'hybrid:self_eval_weights': 'weighted_self_eval',
         'hybrid:CTI_boosting': 'weighted_Cti_boosting'}
def section(title):
    i = next(j for j, l in enumerate(txt) if l.startswith(title)); out = []
    for l in txt[i + 2:]:
        if not l.strip(): break
        out.append(l)
    return out
n_cells = 0
for scen, title in (("full", "ATTRIBUTION - FULL DATA"), ("divided", "ATTRIBUTION - SHUFFLED & DIVIDED")):
    lines = {l[:26].strip(): l[26:].split() for l in section(title)}
    for m, lab in label.items():
        for j, k in enumerate(K):
            n_cells += 1
            if lines[lab][j] != f"{summary['attr'][m][scen][k]:.2f}": bad(f"summary.txt {title} {lab} top{k}: {lines[lab][j]}")
ffl = {'xgboost': 'Xgboost', 'gnn': 'GNN', 'siamese': 'Siamese', 'mil': 'MIL', 'rf': 'RF', 'featureMatching': 'Feature match',
       'jaccard': 'jaccards', 'tvd': 'TVD', 'hybrid:no_weights': 'hybrid', 'hybrid:self_eval_weights': 'hybrid_self_eval_weights',
       'hybrid:CTI_boosting': 'hybrid_CTI_boosting'}
for scen, title in (("full", "FALSE FLAG - FULL"), ("partial", "FALSE FLAG - PARTIAL")):
    lines = {l[:26].strip(): l[26:].split() for l in section(title)}
    for m, lab in ffl.items():
        vals = [summary['ff'][m][scen][k][x] for k in K for x in ('AD', 'MA', 'Both', 'Neither')]
        for j, v in enumerate(vals):
            n_cells += 1
            if lines[lab][j] != f"{v:.2f}": bad(f"summary.txt {title} {lab} col{j}: {lines[lab][j]} vs {v:.2f}")
# excel csv: every 4-decimal fraction must appear
for m in summary['attr']:
    for scen in summary['attr'][m]:
        for k in K:
            if f"{summary['attr'][m][scen][k] / 100:.4f}" not in csvt: bad(f"excel_blocks.csv lacks attribution {m} {scen} {k}")
print(f"  summary.txt: {n_cells} numbers match the independent recomputation; excel_blocks.csv checked")
print("  RESULT:", "ALL CONSISTENT" if not problems else f"{len(problems)} PROBLEMS")
