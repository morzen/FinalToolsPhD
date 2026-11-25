#!/usr/bin/env python3
"""
sweep_core.py -- helpers shared by the feature-combination sweep.

IDEA
  A *configuration* S is a subset of the six feature types (techniques, software, CAPEC, CWE, CVE, Pulse).
  "The tool for S" is the thesis tool (FinalToolsPhD_6feat, its code untouched) run on data in which every
  feature type outside S is blank (-1 = missing, exactly how the thesis marks a missing value).  Blank types are
  invisible to the shared protocol (common/protocol.py): they are not in the profiles, not in the full / divided
  evidence, not in the false-flag construction, and not in any vector the models see.  So
        S = all six                      is the thesis' six-feature tool,
        S = {techniques, software, CVE}  is the thesis' three-feature tool (this is checked: identical ranks),
        S = {techniques}                 is the TTP-only tool.
  Nothing in the tool folders is written: profiles are blanked in memory, models are retrained in memory (RF) or in
  a throw-away copy of the tool (the neural methods, see run_neural_config.py).

CONVENTIONS
  * Configurations are named by letters in the protocol's order:  T techniques, S software, A CAPEC (attack
    patterns), W CWE (weaknesses), V CVE (vulnerabilities), P Pulse (OTX).   'TSV' = techniques + software + CVE.
  * Every result is stored as full rank matrices (mid-ranks of all 181 groups for every query), so every metric
    and every WRRF hybrid can be recomputed later without re-running a model.
"""
import os
import sys
import io
import json
import time
import itertools
import contextlib
import importlib.util
import warnings

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

FAMILIES = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']            # = P.FEATURES of the six-feature tool
LETTER = {'techniques': 'T', 'software': 'S', 'capec': 'A', 'cwe': 'W', 'cve': 'V', 'pulse': 'P'}
FROM_LETTER = {v: k for k, v in LETTER.items()}
LABEL = {'techniques': 'TTPs (techniques)', 'software': 'Software', 'capec': 'CAPEC', 'cwe': 'CWE',
         'cve': 'CVE', 'pulse': 'OTX pulses'}
QUERY_FILES = {'xgboost': 'query.py', 'gnn': 'query.py', 'siamese': 'query.py', 'mil': 'query.py', 'rf': 'query.py',
               'jaccard': 'Jaccards_correlation.py', 'tvd': 'TVD_featureIng.py', 'featureMatching': 'featureMatching.py'}
METHODS = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'jaccard', 'tvd', 'featureMatching']      # = P.METHODS
CLOUD_METHODS = ['jaccard', 'featureMatching', 'tvd', 'rf']       # need no PyTorch / XGBoost
NEURAL_METHODS = ['xgboost', 'gnn', 'siamese', 'mil']
TOP_K = [1, 5, 10, 20]


# ------------------------------------------------------------------------------ configurations
def cfg_key(families):
    fs = set(families)
    return ''.join(LETTER[f] for f in FAMILIES if f in fs)


def cfg_families(key):
    bad = [c for c in key if c not in FROM_LETTER]
    if bad or not key:
        raise ValueError(f"bad configuration '{key}' (use letters from {''.join(LETTER.values())})")
    return tuple(f for f in FAMILIES if LETTER[f] in key)


def all_configs():
    """The 63 non-empty subsets, ordered by size and then by the protocol's feature order."""
    return [cfg_key(c) for r in range(1, 7) for c in itertools.combinations(FAMILIES, r)]


def parse_configs(spec):
    """'ttp' | '3f' | '6f' | 'singles' | 'all' | 'T,TS,TSV' (comma separated, any order of letters)."""
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        low = part.lower()
        if low == 'all':
            out += all_configs()
        elif low == 'ttp':
            out.append('T')
        elif low == '3f':
            out.append('TSV')
        elif low == '6f':
            out.append('TSAWVP')
        elif low == 'singles':
            out += [LETTER[f] for f in FAMILIES]
        else:
            out.append(cfg_key(cfg_families(part.upper())))
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


# ------------------------------------------------------------------------------ small utilities
@contextlib.contextmanager
def quiet():
    """Silence print() output of the thesis modules (they are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_module(path, name, chdir=True):
    """Import a thesis script from its own folder (as CTI_eval.py does)."""
    folder = os.path.dirname(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    if chdir:
        os.chdir(folder)
    sys.path.insert(0, folder)
    try:
        with quiet():
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        os.chdir(cwd)
    return module


# ------------------------------------------------------------------------------ the tool wrapper
class Sweep:
    """One thesis tool folder (read only) + an in-memory feature configuration."""

    def __init__(self, tool_root):
        self.root = os.path.abspath(tool_root)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        from common import protocol as P
        if not set(P.FEATURES) <= set(FAMILIES):
            raise RuntimeError(f"{self.root}: unexpected feature types {P.FEATURES}")
        self.families = [f for f in FAMILIES if f in P.FEATURES]      # six (6f tool) or three (3f tool, --check only)
        self.P = P
        P.plot_attribution_heatmap = lambda *a, **k: None          # the heatmaps are not needed (and are slow)
        self.canon = P.apt_names()
        self.orig = P.load_profiles()
        self._orig_cache = P._PROFILE_CACHE
        self._modules = {}
        self._enc_df = None
        self._map_df = None
        self._cti = None

    # ---- data ---------------------------------------------------------------------------
    @property
    def enc_df(self):
        import pandas as pd
        if self._enc_df is None:
            self._enc_df = pd.read_csv(os.path.join(self.P.BASE, 'EncodedData.csv'))
        return self._enc_df

    @property
    def map_df(self):
        import pandas as pd
        if self._map_df is None:
            self._map_df = pd.read_csv(os.path.join(self.P.BASE, 'comparison_Encoded_df.csv'))
        return self._map_df

    def blank_df(self, S, dedup=True):
        """EncodedData.csv with every feature type outside S set to -1.  dedup: identical rows removed (profiles are
        sets, so the Random Forest, XGBoost, GNN and MIL profiles do not change; the first appearance of every group
        keeps its place).  The Siamese network samples ROWS, so the copy written for the neural tools keeps them."""
        d = self.enc_df.copy()
        for f in FAMILIES:
            if f not in S and self.P.COLUMNS[f] in d.columns:
                d[self.P.COLUMNS[f]] = -1
        return d.drop_duplicates() if dedup else d

    def set_config(self, S):
        """Make the protocol see only the feature types in S (profiles blanked in memory)."""
        S = set(S)
        if not S <= set(self.families):
            raise ValueError(f"configuration {sorted(S)} is not available in this tool ({self.families})")
        blanked = {}
        for name, prof in self.orig.items():
            d = {'encoded_id': prof['encoded_id']}
            for f in self.families:
                d[f] = list(prof[f]) if f in S else []
            blanked[name] = d
        self.P._PROFILE_CACHE = blanked
        self.S = tuple(f for f in FAMILIES if f in S)
        return blanked

    def restore(self):
        self.P._PROFILE_CACHE = self._orig_cache

    def profile_stats(self, S):
        """Dimension and coverage of the configuration (counted from the original profiles)."""
        S = [f for f in FAMILIES if f in set(S)]
        vocab = {f: len(set(x for p in self.orig.values() for x in p[f])) for f in S}
        sizes = np.array([sum(len(p[f]) for f in S) for p in self.orig.values()])
        return {'families': S, 'dimension': int(sum(vocab.values())), 'vocabulary': vocab,
                'n_groups': len(sizes), 'n_groups_empty': int((sizes == 0).sum()),
                'profile_size_mean': float(sizes.mean()), 'profile_size_median': float(np.median(sizes))}

    # ---- modules and scorers -----------------------------------------------------------------
    def module(self, method):
        if method not in self._modules:
            folder = os.path.join(self.root, self.P.METHOD_DIRS[method])
            self._modules[method] = load_module(os.path.join(folder, QUERY_FILES[method]), f"sweep_{method}_query")
        return self._modules[method]

    def make_scorer(self, method, S, rf_model=None):
        """The thesis' own scorer class, built after set_config(S)."""
        self.set_config(S)
        mod = self.module(method)
        if method == 'jaccard':
            sc = mod.JaccardScorer()
        elif method == 'featureMatching':
            sc = mod.FeatureMatchingScorer()
        elif method == 'tvd':
            sc = mod.TVDScorer()
        elif method == 'rf':
            if rf_model is None and getattr(self, 'use_saved_rf', False):
                with quiet():
                    sc = mod.get_scorer()                    # the tool's own saved model (--check runs)
            else:
                sc = mod.RFScorer(rf_model if rf_model is not None else self.train_rf(S))
        else:                                                    # neural methods: model trained in this tool folder
            with quiet():
                sc = mod.get_scorer()
        if hasattr(sc, '_df_mapping'):
            sc._df_mapping = self.map_df                          # one shared copy of the 100 MB mapping
        return sc

    def train_rf(self, S):
        """Retrain the thesis' Random Forest for S in memory, with the classes of 5.RF_FeatureEngineering/train.py."""
        tr = load_module(os.path.join(self.root, self.P.METHOD_DIRS['rf'], 'train.py'), 'sweep_rf_train')
        df = self.blank_df(S)
        with quiet():
            loader = tr.RFDataLoader()
            loader.df_encoded, loader.df_mapping = df, self.map_df
            apt_name_dict = loader.create_apt_name_mapping()
            profiles = tr.APTProfileCreator(apt_name_dict).create_profiles_from_dataframe(df)
            fe = tr.FeatureEngineer()
            binarizers, all_features = fe.prepare_feature_binarizers(profiles)
            apt_features = fe.create_feature_vectors(profiles)
            X, y, apt_names = fe.create_feature_matrix(profiles, apt_features)
            rf = tr.RandomForestTrainer().train(X, y)
        return {'rf_model': rf, 'binarizers': binarizers, 'all_features': all_features, 'apt_profiles': profiles,
                'apt_names': apt_names, 'apt_name_dict': apt_name_dict, 'X': X,
                'feature_types': tr.RandomForestConfig.FEATURE_TYPES, 'df_mapping': self.map_df,
                'config': {'n_estimators': tr.RandomForestConfig.N_ESTIMATORS,
                           'random_state': tr.RandomForestConfig.RANDOM_STATE, 'model_type': 'RandomForestClassifier'}}

    # ---- evidence ---------------------------------------------------------------------------------
    def encode_reports(self, template_path):
        """The 30 CTI reports as encoded evidence, built once with the thesis' own sanitize / translate functions
        (identical in every method's hybrid_query).  Returns a list of dicts (apt, link, enc)."""
        import json as _json
        mod = self.module('jaccard')
        keymap = {'techniques': 'techniques_str', 'software': 'software_str', 'capec': 'CAPEC_ID', 'cwe': 'CWE_ID',
                  'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
        with open(template_path) as fh:
            template = _json.load(fh)
        out = []
        for apt, reports in template.items():
            for rep in reports:
                if rep.get('report_link') == 'combination_of_evidence':
                    continue
                renamed = {keymap[k]: v for k, v in rep['Data'].items() if k in keymap and v}
                with quiet():
                    clean = mod.sanitize_evidence(renamed, self.map_df)
                    enc = mod.translate_to_encoded(clean, self.map_df)
                out.append({'apt': apt, 'link': rep['report_link'], 'enc': enc,
                            'raw': {k: list(v) for k, v in rep['Data'].items() if v}})
        return out

    def encode_sets(self, sets):
        """Encoded evidence of an {id: human-readable evidence} dict (external evidence sets), same functions."""
        mod = self.module('jaccard')
        keymap = {'techniques': 'techniques_str', 'software': 'software_str', 'capec': 'CAPEC_ID', 'cwe': 'CWE_ID',
                  'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
        out = {}
        for sid, ev in sets.items():
            renamed = {keymap[k]: v for k, v in ev.items() if k in keymap and v}
            with quiet():
                enc = mod.translate_to_encoded(mod.sanitize_evidence(renamed, self.map_df), self.map_df)
            out[sid] = {'enc': enc, 'raw': {k: list(v) for k, v in ev.items() if v}}
        return out

    # ---- evaluation ----------------------------------------------------------------------------
    def attribution(self, scorer, label):
        """The thesis' evaluate_attribution (full data + shuffled & divided), nothing re-implemented."""
        import tempfile
        P = self.P
        with tempfile.TemporaryDirectory() as tmp:
            with quiet():
                res = P.evaluate_attribution(scorer, label, results_dir=tmp, smoke=False)
            full = np.load(os.path.join(tmp, 'attribution_ranks_full.npy')).astype(np.float32)
            div = np.load(os.path.join(tmp, 'attribution_ranks_divided.npy')).astype(np.float32)
        return full, div, res

    def cti(self, scorer, reports, S, module=None):
        """Rank vectors (mid-ranks over the 181 groups, canonical order) of evidence sets restricted to S.
        Rows of sets that keep no item known to the dataset are NaN (the thesis' CTI_eval refuses such reports).
        module=None: scorer.score() on the evidence encoded once (checked identical to hybrid_query for Jaccard,
        Feature Matching, TVD and RF).  module given: the method's own hybrid_query() on the human-readable
        evidence, exactly as CTI_eval.py (used for the neural methods)."""
        P = self.P
        align = P.alignment(scorer.apt_names)
        idx = {n: i for i, n in enumerate(self.canon)}
        S = set(S)
        ranks = np.full((len(reports), len(self.canon)), np.nan, dtype=np.float32)
        n_items = np.zeros(len(reports), dtype=np.int32)
        if module is not None and hasattr(module, '_SCORER'):
            module._SCORER = scorer                          # hybrid_query() uses this very scorer
        for i, rep in enumerate(reports):
            ev = {f: v for f, v in rep['enc'].items() if f in S and len(v) > 0}
            n_items[i] = sum(len(v) for v in ev.values())
            if not ev:
                continue
            if module is None:
                s = np.asarray(scorer.score(ev), dtype=np.float64)[align]
                ranks[i] = P.midranks(s)
            else:
                raw = {f: v for f, v in rep['raw'].items() if f in S and v}
                with quiet():
                    ranked = module.hybrid_query(raw)
                for item in ranked:
                    ranks[i, idx[item['apt_name']]] = float(item['rank'])
        true_idx = np.array([self.canon.index(r['apt']) if 'apt' in r else -1 for r in reports])
        return ranks, true_idx, n_items


# ------------------------------------------------------------------------------ metrics
def true_ranks(R):
    """Rank of the queried group in each row of a full attribution rank matrix (query i = canonical group i)."""
    return np.diag(np.asarray(R, dtype=np.float64))


def cti_score(rank, n):
    return 100.0 * (1.0 - (np.asarray(rank, dtype=np.float64) - 1.0) / (n - 1.0))


def summarize(ranks, n_groups):                 # n_groups = number of groups in the dataset (was fixed at 181)
    """Top-k hit rates (%), mean rank and mean CTI score of an array of ranks of the true group."""
    r = np.asarray(ranks, dtype=np.float64)
    r = r[~np.isnan(r)]
    out = {'n': int(len(r))}
    if len(r) == 0:
        return out | {f'top{k}': float('nan') for k in TOP_K} | {'mean_rank': float('nan'), 'cti': float('nan')}
    for k in TOP_K:
        out[f'top{k}'] = float(100.0 * np.mean(r <= k))
    out['mean_rank'] = float(r.mean())
    out['cti'] = float(cti_score(r, n_groups).mean())
    return out


def save_result(path, full, div, cti_ranks, cti_true_idx, cti_items, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, full=full, divided=div, cti_ranks=cti_ranks, cti_true_idx=cti_true_idx,
                        cti_items=cti_items, meta=json.dumps(meta))
    os.replace(tmp, path)


def load_result(path):
    z = np.load(path, allow_pickle=False)
    return {'full': z['full'], 'divided': z['divided'], 'cti_ranks': z['cti_ranks'], 'cti_true_idx': z['cti_true_idx'],
            'cti_items': z['cti_items'], 'meta': json.loads(str(z['meta']))}
