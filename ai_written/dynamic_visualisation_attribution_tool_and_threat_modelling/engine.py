"""
engine.py -- runs the thesis attribution tools for the explorer, without changing them.

Every ranking comes from the method's own CTI entry point, hybrid_query(evidence) (the call CTI_eval.py
makes), and the hybrid from common/protocol.py (wrrf_ranked, HYBRID_WEIGHTS), so the explorer shows exactly
what the thesis tools compute.  The "how" panels re-run the same scorer objects and keep their intermediate
values; each panel checks that its recomputed scores equal the tool's own scores and says so.

Nothing is written into the tool folders (no bytecode, no results files).
"""
import os
import sys
import io
import json
import math
import time
import contextlib
import importlib.util
import threading

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                                    # noqa: E402

import names as names_mod                                             # noqa: E402

QUERY_FILES = {'xgboost': 'query.py', 'gnn': 'query.py', 'siamese': 'query.py', 'mil': 'query.py',
               'rf': 'query.py', 'jaccard': 'Jaccards_correlation.py', 'tvd': 'TVD_featureIng.py',
               'featureMatching': 'featureMatching.py'}
LEVEL_LABEL = {'techniques': 'TTPs', 'software': 'Software', 'capec': 'CAPEC', 'cwe': 'CWE', 'cve': 'CVE',
               'pulse': 'Pulse'}
MAP_COLUMN = {'techniques': 'techniques', 'software': 'software', 'capec': 'CAPEC_ID', 'cwe': 'CWE_ID',
              'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
METHOD_LABEL = {'xgboost': 'XGBoost', 'gnn': 'GNN', 'siamese': 'Siamese Network', 'mil': 'MIL',
                'rf': 'Random Forest', 'jaccard': 'Jaccard', 'tvd': 'TVD', 'featureMatching': 'Feature Matching'}
WEIGHTING_LABEL = {'no_weights': 'Equal', 'self_eval_weights': 'Self-eval', 'CTI_boosting': 'CTI-based'}
TOP = 10


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@contextlib.contextmanager
def _in_folder(folder):
    cwd = os.getcwd()
    os.chdir(folder)
    sys.path.insert(0, folder)
    try:
        yield
    finally:
        if folder in sys.path:
            sys.path.remove(folder)
        os.chdir(cwd)


def _f(x):
    """JSON-safe float."""
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def _pca2(points, k=2):
    """Principal-component coordinates (first k components) of the rows of `points` (numpy SVD)."""
    X = np.asarray(points, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    out = Xc @ vt[:k].T
    if out.shape[1] < k:                               # fewer dimensions than asked: pad with zeros
        out = np.hstack([out, np.zeros((out.shape[0], k - out.shape[1]))])
    return out


class Engine:
    def __init__(self, tool_root, dump_dir=None, methods=None, log=print):
        self.root = os.path.abspath(tool_root)
        self.lock = threading.Lock()
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        from common import protocol as P                             # the tool's own protocol
        self.P = P
        self.features = list(P.FEATURES)
        self.names = P.apt_names()                                   # canonical APT order
        self.index = {n: i for i, n in enumerate(self.names)}
        self.profiles = P.load_profiles()
        self.methods, self.modules, self.scorers, self.load_errors = [], {}, {}, {}
        self.load_seconds = {}
        for m in P.METHODS:
            if methods is not None and m not in methods:
                continue
            t0 = time.time()
            try:
                self._load(m)
                self.methods.append(m)
                self.load_seconds[m] = round(time.time() - t0, 1)
                log(f"[explorer] {METHOD_LABEL[m]:17s} loaded in {self.load_seconds[m]:.1f} s")
            except Exception as exc:                                  # noqa: BLE001
                self.load_errors[m] = f"{type(exc).__name__}: {exc}"
                log(f"[explorer] {METHOD_LABEL[m]:17s} NOT loaded: {self.load_errors[m]}")
        self._build_maps()
        # which groups hold each item (encoded ID -> group indices), for the 3-D view
        self.members = {f: {} for f in self.features}
        for gi, g in enumerate(self.names):
            for f in self.features:
                for e in self.profiles[g][f]:
                    self.members[f].setdefault(int(e), []).append(gi)
        self.labels = names_mod.load(dump_dir) if dump_dir else None
        self._prepare_explainers()
        self.cache = {}

    # ------------------------------------------------------------------ loading
    def _load(self, m):
        folder = os.path.join(self.root, self.P.METHOD_DIRS[m])
        spec = importlib.util.spec_from_file_location(f"explorer_{m}_query", os.path.join(folder, QUERY_FILES[m]))
        module = importlib.util.module_from_spec(spec)
        with _in_folder(folder), _quiet():
            spec.loader.exec_module(module)
            if m == 'xgboost':
                # the model was pickled from train.py run as __main__: the class must be found there
                setattr(sys.modules['__main__'], 'PicklableXGBWrapper', module.PicklableXGBWrapper)
            scorer = module.get_scorer()
        self.modules[m] = module
        self.scorers[m] = scorer

    def _build_maps(self):
        """origin ID <-> encoded ID per level, from the dataset mapping (comparison_Encoded_df.csv)."""
        import pandas as pd
        cols = []
        for f in self.features:
            cols += [f"{MAP_COLUMN[f]}_Origin", f"{MAP_COLUMN[f]}_Encoded"]
        dfm = pd.read_csv(os.path.join(self.P.BASE, "comparison_Encoded_df.csv"), usecols=cols)
        self.to_enc, self.to_origin = {}, {}
        for f in self.features:
            pair = dfm[[f"{MAP_COLUMN[f]}_Origin", f"{MAP_COLUMN[f]}_Encoded"]].dropna()
            pair = pair[pair[f"{MAP_COLUMN[f]}_Encoded"].astype(int) != self.P.MISSING]
            pair = pair.drop_duplicates(subset=f"{MAP_COLUMN[f]}_Origin", keep="first")
            self.to_enc[f] = dict(zip(pair[f"{MAP_COLUMN[f]}_Origin"].astype(str),
                                      pair[f"{MAP_COLUMN[f]}_Encoded"].astype(int)))
            self.to_origin[f] = {}
            for o, e in self.to_enc[f].items():
                self.to_origin[f].setdefault(e, o)

    def origin(self, level, enc):
        return self.to_origin.get(level, {}).get(int(enc), f"#{enc}")

    def label(self, level, origin_id):
        """Readable name of an item (empty if unknown)."""
        if not self.labels:
            return ''
        if level == 'techniques':
            return self.labels['techniques'].get(origin_id, {}).get('name', '')
        if level == 'software':
            return self.labels['software'].get(origin_id, {}).get('name', '')
        if level in ('capec', 'cwe'):
            return self.labels[level].get(origin_id, {}).get('name', '')
        return ''

    def item(self, level, enc):
        o = self.origin(level, enc)
        return {'id': o, 'name': self.label(level, o)}

    # ------------------------------------------------------------------ meta for the page
    def meta(self):
        vocab = {}
        for f in self.features:
            ids = sorted(self.to_enc[f])
            vocab[f] = [[i, self.label(f, i)] for i in ids]
        presets = []
        tpl = os.path.join(self.root, 'CTI_eval', 'CTI_Eval_empty_template.json')
        if os.path.exists(tpl):
            with open(tpl) as fh:
                template = json.load(fh)
            for apt, reports in template.items():
                for k, rep in enumerate(reports, 1):
                    if rep.get('report_link') == 'combination_of_evidence':
                        continue
                    ev = {f: list(rep['Data'].get(f, [])) for f in self.features}
                    presets.append({'kind': 'cti', 'group': apt, 'link': rep.get('report_link', ''),
                                    'label': f"{apt} - CTI report {k}", 'evidence': ev})
        tactics = {}
        if self.labels:
            for t, v in self.labels['techniques'].items():
                tactics[t] = v.get('tactics', [])
        return {'tool': os.path.basename(self.root), 'features': self.features,
                'level_label': {f: LEVEL_LABEL[f] for f in self.features},
                'methods': self.methods, 'method_label': {m: METHOD_LABEL[m] for m in self.methods},
                'load_errors': {m: e for m, e in self.load_errors.items()},
                'groups': self.names, 'vocab': vocab, 'presets': presets, 'tactics': tactics,
                'weights': self.P.HYBRID_WEIGHTS, 'weighting_label': WEIGHTING_LABEL,
                'wrrf_k': self.P.WRRF_K, 'names_loaded': self.labels is not None,
                'group_info': self.group_info()}

    def group_info(self):
        """Per group: profile sizes and whether it comes from ATT&CK (else an AlienVault OTX entry)."""
        attack = {v.get('name') for v in (self.labels or {}).get('groups', {}).values()}
        gid = {v.get('name'): k for k, v in (self.labels or {}).get('groups', {}).items()}
        return [{'name': g, 'attack': g in attack, 'gid': gid.get(g, ''),
                 'sizes': {f: len(self.profiles[g][f]) for f in self.features}} for g in self.names]

    def profile_evidence(self, group, mode='full', imitated=None):
        """Evidence (origin IDs) built the way the thesis tests build it: the full profile (sanity check),
        one half per level (stress test, seed 42) or a synthetic false flag (Section 6.1)."""
        P = self.P
        prof = self.profiles[group]
        if mode == 'half':
            enc = P.divided_evidence(group, prof)
        elif mode == 'falseflag':
            enc = P.false_flag_evidence(group, prof, imitated, self.profiles[imitated])
            if isinstance(enc, tuple):
                enc = enc[0]
        else:
            enc = P.full_evidence(prof)
        return {f: [self.origin(f, e) for e in enc.get(f, [])] for f in self.features}

    # ------------------------------------------------------------------ attribution
    def attribute(self, evidence):
        with self.lock:
            return self._attribute(evidence)

    def encode(self, evidence):
        """Human-readable evidence -> encoded IDs, item by item in the given order, unknown items dropped.
        The dataset mapping is one-to-one (checked: no origin ID has two encoded IDs), so this is the encoding
        every method's own hybrid_query builds; validate_explorer.py checks the rankings are identical."""
        return {f: [self.to_enc[f][x] for x in evidence.get(f, []) if x in self.to_enc[f]] for f in self.features}

    def rank_with_tool_entry_point(self, m, evidence):
        """The method's own hybrid_query (the CTI evaluation path): slower, used by validate_explorer.py."""
        with _in_folder(os.path.join(self.root, self.P.METHOD_DIRS[m])), _quiet():
            return self.modules[m].hybrid_query({f: list(v) for f, v in evidence.items()})

    def _attribute(self, evidence):
        P = self.P
        t0 = time.time()
        # kept as given (order and repeats), as the tools receive it; repeats are only merged for display
        evidence = {f: [str(x).strip() for x in evidence.get(f, []) if str(x).strip()] for f in self.features}
        recognised = {f: [x for x in dict.fromkeys(evidence[f]) if x in self.to_enc[f]] for f in self.features}
        unknown = {f: [x for x in dict.fromkeys(evidence[f]) if x not in self.to_enc[f]] for f in self.features}
        enc = self.encode(evidence)
        rankings, per_method, seconds = {}, {}, {}
        for m in self.methods:
            t = time.time()
            sc = self.scorers[m]
            with _quiet():
                ranked = P.ranked_results(sc.apt_names, sc.score(enc), score_key='score')
            seconds[m] = round(time.time() - t, 3)
            rankings[m] = ranked
            ranks = np.full(len(self.names), np.nan)
            scores = np.full(len(self.names), np.nan)
            for it in ranked:
                i = self.index[it['apt_name']]
                ranks[i] = it['rank']
                scores[i] = next(v for k, v in it.items() if k not in ('rank', 'apt_name'))
            per_method[m] = {'ranks': [_f(r) for r in ranks], 'scores': [_f(s) for s in scores],
                             'top': [{'group': it['apt_name'], 'rank': _f(it['rank']),
                                      'score': _f(next(v for k, v in it.items() if k not in ('rank', 'apt_name')))}
                                     for it in ranked[:TOP]],
                             'seconds': seconds[m]}
        hybrid = {}
        if len(self.methods) == len(P.METHODS):
            for cfg, w in P.HYBRID_WEIGHTS.items():
                fused = P.wrrf_ranked(rankings, w)
                ranks = np.full(len(self.names), np.nan)
                scores = np.full(len(self.names), np.nan)
                for it in fused:
                    ranks[self.index[it['apt_name']]] = it['rank']
                    scores[self.index[it['apt_name']]] = it['wrrf_score']
                hybrid[cfg] = {'ranks': [_f(r) for r in ranks], 'scores': [_f(s) for s in scores],
                               'top': [{'group': it['apt_name'], 'rank': _f(it['rank']), 'score': _f(it['wrrf_score'])}
                                       for it in fused[:TOP]]}
        qid = f"q{int(time.time() * 1000)}"
        self.cache = {'qid': qid, 'evidence': evidence, 'per_method': per_method}
        explain = {}
        for m in self.methods:
            t = time.time()
            try:
                explain[m] = self._explain(m, enc, per_method[m])
            except Exception as exc:                                  # noqa: BLE001
                explain[m] = {'error': f"{type(exc).__name__}: {exc}"}
            explain[m]['seconds'] = round(time.time() - t, 3)
        membership = {f: {x: self.members[f].get(self.to_enc[f][x], []) for x in recognised[f]} for f in self.features}
        return {'qid': qid, 'evidence': evidence, 'recognised': recognised, 'unknown': unknown,
                'membership': membership,
                'recognised_named': {f: [{'id': x, 'name': self.label(f, x)} for x in recognised[f]]
                                     for f in self.features},
                'methods': per_method, 'hybrid': hybrid, 'explain': explain,
                'complete': len(self.methods) == len(P.METHODS), 'seconds': round(time.time() - t0, 2)}

    # ------------------------------------------------------------------ the "how" panels
    def _check(self, m, scores_by_name, pm):
        """Largest difference between the panel's recomputed scores and the tool's own scores."""
        diffs = [abs(scores_by_name[n] - pm['scores'][self.index[n]]) for n in scores_by_name
                 if pm['scores'][self.index[n]] is not None]
        return _f(max(diffs)) if diffs else None

    def _top_names(self, pm, k=TOP):
        return [t['group'] for t in pm['top'][:k]]

    def _prepare_explainers(self):
        """Things that do not depend on the evidence, computed once."""
        self.gnn_feature_apts = {}
        if 'gnn' in self.scorers:
            sc = self.scorers['gnn']
            ei = sc.edge_index.cpu().numpy()
            na = sc.num_apts
            for a, b in zip(ei[0], ei[1]):
                if a < na <= b:
                    self.gnn_feature_apts.setdefault(int(b), set()).add(int(a))
        if 'rf' in self.scorers:
            md = self.scorers['rf'].md
            self.rf_columns = []
            for ft in md['feature_types']:
                for c in md['binarizers'][ft].classes_:
                    self.rf_columns.append((ft, int(c)))

    def _explain(self, m, enc, pm):
        if m in ('jaccard', 'featureMatching', 'tvd'):
            return self._explain_setbased(m, enc, pm)
        return getattr(self, f"_explain_{m}")(enc, pm)

    # ---- Jaccard, Feature Matching, TVD: per-level terms -------------------------------------
    def _explain_setbased(self, m, enc, pm):
        sc, P = self.scorers[m], self.P
        ev = P.non_empty(enc)
        W = sc.weights
        if m == 'tvd':                         # TVD keeps only evidence items present in some profile
            ev2 = {}
            for f in W:
                union = set()
                for a in sc.profiles:
                    union.update(sc.profiles[a][f])
                e = set(ev.get(f, [])) & union
                if e:
                    ev2[f] = sorted(e)
            ev = ev2
        rows, recomputed = [], {}
        for apt in self._top_names(pm):
            prof = sc.profiles[apt]
            terms, num, den = [], 0.0, 0.0
            for f, w in W.items():
                if f not in ev:
                    continue
                E, A = set(ev[f]), set(prof[f])
                inter, union = len(E & A), len(E | A)
                if m == 'jaccard':
                    sim = inter / union if union else 0.0
                elif m == 'featureMatching':
                    sim = inter / len(E) if E else 0.0
                else:
                    if A:
                        tvd = 0.5 * ((len(E) - inter) / len(E) + (len(A) - inter) / len(A)
                                     + inter * abs(1.0 / len(E) - 1.0 / len(A)))
                    else:
                        tvd = 0.5
                    sim = 1 - tvd
                num += sim * w
                den += w
                terms.append({'level': f, 'weight': w, 'evidence': len(E), 'profile': len(A), 'shared': inter,
                              'union': union, 'similarity': _f(sim),
                              'shared_items': [self.item(f, x) for x in sorted(E & A)][:25]})
            recomputed[apt] = num / den if den else 0.0
            rows.append({'group': apt, 'score': _f(recomputed[apt]), 'terms': terms, 'weight_total': den})
        return {'kind': 'setbased', 'weights': W, 'rows': rows, 'check': self._check(m, recomputed, pm)}

    # ---- Random Forest: shared leaves ------------------------------------------------------------
    def _explain_rf(self, enc, pm):
        sc, P = self.scorers['rf'], self.P
        md = sc.md
        mod = self.modules['rf']
        vector = mod.create_evidence_vector(P.non_empty(enc), md['binarizers'], md['all_features'], md['feature_types'])
        X1 = vector.reshape(1, -1)
        leaves = np.array([tree.apply(X1)[0] for tree in sc.rf.estimators_])
        agree = sc.apt_leaves == leaves.reshape(-1, 1)                     # trees x APTs
        own = {n: i for i, n in enumerate(sc.apt_names)}
        rows, recomputed = [], {}
        top = self._top_names(pm)
        for apt in top:
            col = agree[:, own[apt]]
            recomputed[apt] = float(col.mean())
            rows.append({'group': apt, 'share': _f(col.mean()), 'trees': ''.join('1' if b else '0' for b in col)})
        # what routes the evidence to the top group's leaves: the tests on the evidence's own path in the trees
        # where it ends in the same leaf as the top group (present = the evidence has the item)
        counts = {}
        best = own[top[0]] if top else None
        for k, tree in enumerate(sc.rf.estimators_):
            if best is None or not agree[k, best]:
                continue
            t = tree.tree_
            node = 0
            while t.children_left[node] != t.children_right[node]:
                feat = int(t.feature[node])
                present = X1[0, feat] > t.threshold[node]
                key = (feat, bool(present))
                counts[key] = counts.get(key, 0) + 1
                node = t.children_right[node] if present else t.children_left[node]
        tests = []
        for (feat, present), n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
            lvl, e = self.rf_columns[feat]
            tests.append({'item': self.item(lvl, e), 'level': lvl, 'present': present, 'trees': n})
        return {'kind': 'rf', 'n_trees': int(len(sc.rf.estimators_)), 'rows': rows, 'path_tests': tests,
                'agreeing_trees_top': int(agree[:, best].sum()) if best is not None else 0,
                'n_tests': int(sum(counts.values())),
                'n_present_tests': int(sum(n for (_, p), n in counts.items() if p)),
                'check': self._check('rf', recomputed, pm)}

    # ---- XGBoost: class probabilities and contributions ------------------------------------------
    def _explain_xgboost(self, enc, pm):
        import xgboost as xgb
        sc, P = self.scorers['xgboost'], self.P
        mod = self.modules['xgboost']
        fm = sc.feature_metadata
        vector = mod.EfficientQueryModel.create_query_vector_from_evidence(P.non_empty(enc), fm)
        probs = np.asarray(sc.predictor.predict_proba_vector(vector))
        booster = sc.predictor.model._Booster
        dm = xgb.DMatrix(vector.reshape(1, -1), missing=np.nan)
        contrib = booster.predict(dm, pred_contribs=True, approx_contribs=True)
        margin = booster.predict(dm, output_margin=True)
        contrib = np.asarray(contrib).reshape(len(probs), -1)          # classes x (features + bias)
        margin = np.asarray(margin).reshape(-1)
        offsets = [(lvl, 0 if lvl == 'techniques' else fm[f'{lvl}_offset'], f'all_{lvl}')
                   for lvl in self.features if f'all_{lvl}' in fm]       # the tool's own vector layout

        def column(idx):
            for name, off, key in offsets:
                if off <= idx < off + len(fm[key]):
                    return name, fm[key][idx - off]
            return None, None

        present = np.flatnonzero(vector > 0)
        name_to_class = {sc.apt_names[i]: int(sc.class_index[i]) for i in range(len(sc.apt_names))}
        rows, recomputed = [], {}
        for apt in self._top_names(pm, 5):
            k = name_to_class[apt]
            c = contrib[k]
            recomputed[apt] = float(probs[k])
            items = []
            for idx in present:
                lvl, e = column(int(idx))
                items.append({'item': self.item(lvl, e), 'level': lvl, 'value': _f(c[idx])})
            items.sort(key=lambda d: -abs(d['value'] or 0))
            absent_sum = float(c[:-1].sum() - c[present].sum())
            rows.append({'group': apt, 'probability': _f(probs[k]), 'margin': _f(margin[k]),
                         'bias': _f(c[-1]), 'absent_total': _f(absent_sum),
                         'contrib_sum_check': _f(abs(c.sum() - margin[k])), 'items': items[:20]})
        return {'kind': 'xgboost', 'n_classes': int(len(probs)), 'n_features': int(len(vector)),
                'n_present': int(len(present)), 'rows': rows, 'check': self._check('xgboost', recomputed, pm)}

    # ---- GNN: the query node, its evidence nodes and the groups two hops away --------------------
    def _explain_gnn(self, enc, pm):
        import torch
        import torch.nn.functional as F
        sc = self.scorers['gnn']
        x = sc.x
        q = x.shape[0]
        new_x = torch.ones(x.shape[0] + 1, x.shape[1])
        new_x[:x.shape[0]] = x
        new_edges, new_weights, ev_nodes = [], [], []
        for f in self.P.FEATURES:
            w = sc.feature_weights[f]
            for feat in enc.get(f, []):
                if feat in sc.to_idx[f]:
                    node = sc.to_idx[f][feat] + sc.offset[f]
                    new_edges += [[q, node], [node, q]]
                    new_weights += [w, w]
                    ev_nodes.append((f, feat, node, w))
        edge_index, edge_weight = sc.edge_index, sc.edge_weight
        if new_edges:
            edge_index = torch.cat([edge_index, torch.tensor(new_edges, dtype=torch.long).t().contiguous()], dim=1)
            wt = torch.tensor(new_weights, dtype=torch.float)
            edge_weight = wt if edge_weight is None else torch.cat([edge_weight, wt])
        with torch.no_grad():
            emb = sc.model(new_x, edge_index, edge_weight)
            sims = F.cosine_similarity(emb[q].unsqueeze(0), emb[:sc.num_apts]).cpu().numpy()
        recomputed = {sc.apt_names[i]: float(sims[i]) for i in range(len(sc.apt_names))}
        top = self._top_names(pm)
        own = {n: i for i, n in enumerate(sc.apt_names)}
        rows = []
        for apt in top:
            i = own[apt]
            shared = {}
            for f, feat, node, w in ev_nodes:
                if i in self.gnn_feature_apts.get(node, ()):
                    shared.setdefault(f, []).append(self.item(f, feat))
            rows.append({'group': apt, 'cosine': _f(sims[i]), 'shared': shared,
                         'n_shared': sum(len(v) for v in shared.values())})
        evidence_nodes = [{'level': f, 'item': self.item(f, feat), 'edge_weight': w,
                           'degree': len(self.gnn_feature_apts.get(node, ()))} for f, feat, node, w in ev_nodes]
        pts = _pca2(np.vstack([emb[:sc.num_apts].cpu().numpy(), emb[q].cpu().numpy()[None, :]]), 3)
        points = [{'group': sc.apt_names[i], 'x': _f(pts[i, 0]), 'y': _f(pts[i, 1]), 'z': _f(pts[i, 2])}
                  for i in range(sc.num_apts)]
        return {'kind': 'gnn', 'level_weights': {f: sc.feature_weights[f] for f in self.P.FEATURES},
                'evidence_nodes': evidence_nodes[:200], 'n_evidence_nodes': len(evidence_nodes), 'rows': rows,
                'map': {'points': points, 'query': {'x': _f(pts[-1, 0]), 'y': _f(pts[-1, 1]), 'z': _f(pts[-1, 2])}},
                'check': self._check('gnn', {n: recomputed[n] for n in top}, pm)}

    # ---- Siamese: sampled combinations, query embedding, embedding map ---------------------------
    def _explain_siamese(self, enc, pm):
        import torch
        import torch.nn.functional as F
        sc, P = self.scorers['siamese'], self.P
        mod = self.modules['siamese']
        ev = P.non_empty(enc)
        total = sum(len(v) for v in ev.values())
        if total == 0:
            return {'kind': 'siamese', 'empty': True}
        rng = np.random.default_rng(mod.SIAMESE_QUERY_SEED)
        n = min(mod.SIAMESE_MAX_SAMPLES, max(1, total * 2))
        columns = [rng.choice(np.asarray(ev[f], dtype=np.float64), size=n) if f in ev else np.full(n, -1.0)
                   for f in P.FEATURES]
        rows_arr = np.stack(columns, axis=1)
        with torch.no_grad():
            q = sc.model(torch.tensor(rows_arr, dtype=torch.float32)).mean(dim=0)
            sims = F.cosine_similarity(q.unsqueeze(0), sc.apt_embeddings, dim=1).cpu().numpy()
        recomputed = {sc.apt_names[i]: float(sims[i]) for i in range(len(sc.apt_names))}
        samples = []
        for r in rows_arr[:8]:
            samples.append({f: (None if v < 0 else self.item(f, int(v))) for f, v in zip(P.FEATURES, r)})
        top = self._top_names(pm)
        own = {nm: i for i, nm in enumerate(sc.apt_names)}
        pts = _pca2(np.vstack([sc.apt_embeddings.cpu().numpy(), q.cpu().numpy()[None, :]]), 3)
        points = [{'group': sc.apt_names[i], 'x': _f(pts[i, 0]), 'y': _f(pts[i, 1]), 'z': _f(pts[i, 2])}
                  for i in range(len(sc.apt_names))]
        return {'kind': 'siamese', 'n_samples': int(n), 'samples': samples,
                'rows': [{'group': a, 'cosine': _f(sims[own[a]])} for a in top],
                'map': {'points': points, 'query': {'x': _f(pts[-1, 0]), 'y': _f(pts[-1, 1]), 'z': _f(pts[-1, 2])}},
                'check': self._check('siamese', {a: recomputed[a] for a in top}, pm)}

    # ---- MIL: attention over the evidence, per-level similarities --------------------------------
    def _explain_mil(self, enc, pm):
        import torch
        import torch.nn.functional as F
        sc = self.scorers['mil']
        model = sc.model
        ev = sc._evidence_dict(enc)
        types = sc.types
        level_of = {'technique': 'techniques', 'software': 'software', 'capec': 'capec', 'cwe': 'cwe',
                    'cve': 'cve', 'pulse': 'pulse'}
        attention = {}
        with torch.no_grad():
            for t in types:
                if ev[t]:
                    embedded = getattr(model, f'{t}_embed')(torch.tensor(ev[t], dtype=torch.long))
                    a = F.softmax(model.attention_nets[t](embedded), dim=0).reshape(-1).cpu().numpy()
                    order = np.argsort(-a)
                    attention[level_of[t]] = [{'item': self.item(level_of[t], ev[t][j]), 'weight': _f(a[j])}
                                              for j in order[:20]]
            type_w = sc.type_weights.cpu().numpy()
        own = {n: i for i, n in enumerate(sc.apt_names)}
        rows, recomputed = [], {}
        H = model.hidden_dim
        for apt in self._top_names(pm, 5):
            i = own[apt]
            A = sc.apt_sets[i]
            r12 = torch.zeros(1, len(types))
            r21 = torch.zeros(1, len(types))
            for j, t in enumerate(types):
                E = set(ev[t])
                if E:
                    r12[0, j] = max(len(A[t] & E) / len(E), 0.01)
                if A[t]:
                    r21[0, j] = max(len(E & A[t]) / len(A[t]), 0.01)
            with torch.no_grad():
                emb1, emb2 = [], []
                for j, t in enumerate(types):
                    w1 = sc.apt_log[t][i] * (r12[0, j] + 0.1)
                    emb1.append(sc.apt_att[t][i:i + 1] * w1 if sc.apt_has[t][i] else torch.zeros(1, H))
                    if ev[t]:
                        att_e = sc._attended(t, ev[t])
                        w2 = torch.log1p(torch.tensor(len(ev[t]))) * (r21[0, j] + 0.1)
                        emb2.append(att_e * w2)
                    else:
                        emb2.append(torch.zeros(1, H))
                combined = torch.cat(emb1 + emb2 + [r12, r21], dim=-1)
                fs = model.similarity_net(combined)[0].cpu().numpy()
            score = float((fs * type_w).sum())
            recomputed[apt] = score
            rows.append({'group': apt, 'score': _f(score),
                         'levels': [{'level': level_of[t], 'coverage': _f(r12[0, j]), 'profile_coverage': _f(r21[0, j]),
                                     'similarity': _f(fs[j]), 'weight': _f(type_w[j])} for j, t in enumerate(types)]})
        return {'kind': 'mil', 'attention': attention, 'rows': rows,
                'check': self._check('mil', recomputed, pm)}
