/* attr3d.js — the 3-D attribution view: the evidence on the six levels of abstraction (the tower), the groups
   around it (closer = ranked higher by the approach shown) and, for each approach, what it computes on the way. */
import { View3D, V, rgba, ease } from './engine3d.js';
import { LEVELS, LEVEL_COLOR, LEVEL_LABEL, LEVEL_Y, INK, INK2, MUTED, GRID, AXIS, ACCENT, $, h, fmt, fmtRank, pct, trunc, hash01, tip } from './common.js';

export const ORDER = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'featureMatching', 'jaccard', 'tvd'];
export const SHORT = { xgboost: 'XGBoost', gnn: 'GNN', siamese: 'Siamese', mil: 'MIL', rf: 'RF', featureMatching: 'FM', jaccard: 'Jaccard', tvd: 'TVD' };
const ABBR = { xgboost: 'XGB', gnn: 'GNN', siamese: 'Siam', mil: 'MIL', rf: 'RF', featureMatching: 'FM', jaccard: 'Jac', tvd: 'TVD' };
const NAME = { xgboost: 'XGBoost', gnn: 'Graph neural network (GCN)', siamese: 'Siamese network', mil: 'Attention-based MIL',
  rf: 'Random Forest', featureMatching: 'Feature Matching', jaccard: 'Jaccard similarity', tvd: 'Total Variation Distance' };
const KIND = { xgboost: 'machine-learning model', gnn: 'machine-learning model', siamese: 'machine-learning model', mil: 'machine-learning model',
  rf: 'machine-learning model', featureMatching: 'mathematical algorithm', jaccard: 'mathematical algorithm', tvd: 'mathematical algorithm' };
const HYB = [['no_weights', 'Equal'], ['self_eval_weights', 'Self-eval'], ['CTI_boosting', 'CTI-based']];
const DESC = {
  jaccard: 'For each level present in the evidence: the items the evidence E and the profile A share, out of all items in either, |E ∩ A| / |E ∪ A|. The score is the mean of these similarities weighted by level.',
  featureMatching: 'For each level present in the evidence: the share of the evidence found in the profile, |E ∩ A| / |E|. The score is the mean weighted by level.',
  tvd: 'For each level: evidence and profile as uniform distributions over their items; similarity = 1 − total variation distance (0.5 when the group has nothing at that level). The score is the mean weighted by level.',
  rf: 'A forest of 100 decision trees trained to recognise the groups. The evidence vector falls into one leaf of every tree; a group\'s score is the share of trees in which its own profile falls into the same leaf.',
  xgboost: 'Gradient-boosted trees with one class per group. The evidence becomes the 0/1 vector of every item the model knows; each item it contains raises or lowers each group\'s raw score, and the softmax of the raw scores gives the probability of each group.',
  gnn: 'A two-layer graph convolutional network over the graph that links each group to its items (edge weight by level). The evidence enters the graph as a new node linked to its items; after two rounds of message passing its embedding is compared (cosine) with each group\'s.',
  siamese: 'An encoder trained so that profiles of the same group lie close together. Random combinations of one evidence item per level are encoded and averaged; the mean is compared (cosine) with each group\'s embedding.',
  mil: 'Attention-based multiple-instance learning. Attention pools the items of each level; with how much of the evidence the profile covers and how much of the profile the evidence covers, a similarity network scores each level, and learned level weights combine them.',
};

const R_TOWER = 5.0, R_IN = 8.5, R_OUT = 21.0;
const FOREST_Y = -13.5;

export class AttrScene {
  constructor(canvas, meta, hooks = {}) {
    this.meta = meta;
    this.N = meta.groups.length;
    this.idx = new Map(meta.groups.map((g, i) => [g, i]));
    this.info = new Map((meta.group_info || []).map(g => [g.name, g]));
    this.hooks = hooks;
    this.methods = ORDER.filter(m => meta.methods.includes(m));
    this.complete = this.methods.length === ORDER.length;
    this.mode = this.complete ? 'hyb:CTI_boosting' : 'featureMatching';
    this.res = null; this.sel = null; this.userSel = false; this.follow = ''; this.embed = false;
    this.items = []; this.itemByKey = new Map();
    this.gcur = meta.groups.map((g, i) => this._ring(i, this.N));
    this.gtgt = this.gcur.map(p => [...p]);
    this.t0 = performance.now(); this.tPrev = performance.now();
    this.view = new View3D(canvas, {
      onHover: (id, x, y) => tip.show(id ? this._tipText(id) : null, x, y),
      onClick: id => { if (id && id.startsWith('g:')) this.select(id.slice(2)); },
    });
    this.view.cam = { theta: 0.78, phi: 0.3, dist: 41, target: [0, -1.2, 0] };
    this.view.setHome(this.view.cam);
    this.view.draw = v => this.draw(v);
    this._methodBar();
    this._legend();
    this.renderInfo();
  }

  /* ------------------------------------------------------------------ data */
  _ring(i, rank) {
    const a = (2 * Math.PI * i) / this.N;
    const t = Math.pow(Math.max(0, (rank - 1)) / (this.N - 1), 0.7);
    const r = R_IN + (R_OUT - R_IN) * t;
    return [r * Math.cos(a), 0, r * Math.sin(a)];
  }
  isHybrid(m = this.mode) { return m.startsWith('hyb:'); }
  cfg(m = this.mode) { return m.slice(4); }
  ranks(m = this.mode) {
    if (!this.res) return null;
    if (this.isHybrid(m)) return this.res.hybrid && this.res.hybrid[this.cfg(m)] ? this.res.hybrid[this.cfg(m)].ranks : null;
    return this.res.methods[m] ? this.res.methods[m].ranks : null;
  }
  scores(m = this.mode) {
    if (!this.res) return null;
    if (this.isHybrid(m)) return this.res.hybrid && this.res.hybrid[this.cfg(m)] ? this.res.hybrid[this.cfg(m)].scores : null;
    return this.res.methods[m] ? this.res.methods[m].scores : null;
  }
  top(m = this.mode, k = 10) {
    const r = this.ranks(m), s = this.scores(m);
    if (!r) return [];
    return this.meta.groups.map((g, i) => ({ group: g, i, rank: r[i], score: s ? s[i] : null }))
      .filter(x => x.rank !== null).sort((a, b) => a.rank - b.rank || a.group.localeCompare(b.group)).slice(0, k);
  }
  explain(m = this.mode) { return this.res && !this.isHybrid(m) ? this.res.explain[m] : null; }

  setResult(res) {
    this.res = res;
    const items = [];
    for (const lv of LEVELS) {
      const ids = (res.recognised && res.recognised[lv]) || [];
      const named = new Map(((res.recognised_named && res.recognised_named[lv]) || []).map(x => [x.id, x.name]));
      const n = ids.length;
      ids.forEach((id, k) => {
        // stable position from the item ID; large sets are packed on a sunflower so they stay readable
        let r, a;
        if (n > 60) { r = R_TOWER * Math.sqrt((k + 0.5) / n); a = k * 2.39996; }
        else { r = R_TOWER * Math.sqrt(0.08 + 0.92 * hash01(lv + id)); a = 2 * Math.PI * hash01(id + lv + '#'); }
        items.push({ level: lv, id, name: named.get(id) || '', pos: [r * Math.cos(a), LEVEL_Y[lv], r * Math.sin(a)],
          groups: (res.membership && res.membership[lv] && res.membership[lv][id]) || [], n });
      });
    }
    this.items = items;
    this.itemByKey = new Map(items.map(it => [`${it.level}|${it.id}`, it]));
    this._autoSelect();
    this.updateTargets();
    this.restart();
    this.renderInfo();
  }
  /* the best group is shown, unless the user picked a group that is still among the ten best */
  _autoSelect() {
    const t = this.top();
    if (!this.userSel || !t.some(x => x.group === this.sel)) { this.sel = t.length ? t[0].group : null; this.userSel = false; }
  }
  setMode(m) {
    this.mode = m;
    this.embed = (m === 'siamese');
    this._autoSelect();
    this.updateTargets();
    this.restart();
    this._methodBar();
    this.renderInfo();
  }
  /* no evidence: back to the empty scene (groups on the outer ring, no reasoning shown) */
  clear() {
    this.res = null; this.items = []; this.itemByKey = new Map();
    this.sel = null; this.userSel = false;
    this.updateTargets();
    this.restart();
    this.renderInfo();
  }
  select(g) { this.sel = g; this.userSel = true; this.updateTargets(); this.renderInfo(); }
  setFollow(g) { this.follow = g || ''; this.renderInfo(); }
  restart() { this.t0 = performance.now(); }
  toggleEmbed() { this.embed = !this.embed; this.updateTargets(); this.restart(); this._methodBar(); this.renderInfo(); }

  _embedPositions() {
    const ex = this.explain();
    if (!ex || !ex.map || !ex.map.points) return null;
    const pts = ex.map.points.map(p => [p.x || 0, p.y || 0, p.z || 0]);
    const q = ex.map.query ? [ex.map.query.x || 0, ex.map.query.y || 0, ex.map.query.z || 0] : [0, 0, 0];
    const all = pts.concat([q]);
    const mean = [0, 1, 2].map(k => all.reduce((a, p) => a + p[k], 0) / all.length);
    const spread = Math.max(1e-9, ...all.map(p => Math.max(...[0, 1, 2].map(k => Math.abs(p[k] - mean[k])))));
    const sc = 13 / spread;
    const tr = p => [(p[0] - mean[0]) * sc, (p[1] - mean[1]) * sc, (p[2] - mean[2]) * sc];
    const byName = new Map(ex.map.points.map((p, i) => [p.group, tr(pts[i])]));
    return { byName, query: tr(q) };
  }
  updateTargets() {
    const r = this.ranks();
    const emb = this.embed && (this.mode === 'gnn' || this.mode === 'siamese') ? this._embedPositions() : null;
    this.embPos = emb;
    for (let i = 0; i < this.N; i++) {
      const g = this.meta.groups[i];
      if (emb && emb.byName.has(g)) this.gtgt[i] = emb.byName.get(g);
      else this.gtgt[i] = this._ring(i, r ? (r[i] || this.N) : this.N);
    }
  }

  /* ------------------------------------------------------------------ drawing */
  draw(v) {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.tPrev) / 1000); this.tPrev = now;
    const k = 1 - Math.exp(-dt * 4.5);
    for (let i = 0; i < this.N; i++) this.gcur[i] = V.lerp(this.gcur[i], this.gtgt[i], k);
    const T = (now - this.t0) / 1000;                      // seconds since the reasoning started
    const inEmbed = !!this.embPos;
    if (!inEmbed) this._drawTower(v, T);
    if (this.res) {
      if (!inEmbed) {
        if (this.mode === 'mil') {
          const att = this._milAttention();
          this._drawEvidence(v, T, it => { const w = att.get(`${it.level}|${it.id}`); return w === undefined ? 0.12 : 0.16 + 0.75 * Math.sqrt(w); },
            it => { const w = att.get(`${it.level}|${it.id}`) || 0; return w > 0.25 ? w * 1.5 : 0; });
        } else this._drawEvidence(v, T);
      }
      const m = this.mode;
      if (this.isHybrid(m)) this._drawHybrid(v, T);
      else if (m === 'rf') this._drawRF(v, T);
      else if (m === 'xgboost') this._drawXGB(v, T);
      else if (m === 'gnn') this._drawGNN(v, T);
      else if (m === 'siamese') this._drawSiamese(v, T);
      else if (m === 'mil') this._drawMIL(v, T);
      else this._drawSet(v, T);
    }
    this._drawGroups(v, T);
  }

  _drawTower(v, T) {
    for (const lv of LEVELS) {
      const y = LEVEL_Y[lv];
      v.disc([0, y, 0], R_TOWER + 0.6, rgba(LEVEL_COLOR[lv], 0.07), rgba(LEVEL_COLOR[lv], 0.55), { width: 1.2, segments: 72 });
      // on the left of the tower whatever the camera angle
      const R = v._right || [1, 0, 0];
      v.label([-R[0] * (R_TOWER + 1.0), y, -R[2] * (R_TOWER + 1.0)], LEVEL_LABEL[lv], { color: LEVEL_COLOR[lv], size: 12, weight: 700, align: 'right', bg: 'rgba(26,26,25,0.7)', priority: 60, force: true, noFog: true });
    }
    v.line([0, LEVEL_Y.pulse - 0.6, 0], [0, LEVEL_Y.techniques + 0.6, 0], AXIS, 1, { alpha: 0.6 });
    // the plane of the groups
    v.disc([0, 0, 0], R_IN - 0.6, null, rgba(INK2, 0.18), { segments: 96, width: 1, zbias: 8 });
    v.disc([0, 0, 0], R_OUT + 0.6, null, rgba(INK2, 0.10), { segments: 128, width: 1, zbias: 8 });
  }

  _itemScale(T, it) { return ease((T - 0.05 * hash01(it.id)) / 0.45); }

  _drawEvidence(v, T, sizeOf = null, glowOf = null) {
    for (const it of this.items) {
      const s = this._itemScale(T, it);
      if (s <= 0) continue;
      const base = it.n > 300 ? 0.1 : it.n > 80 ? 0.15 : 0.22;
      const r = (sizeOf ? sizeOf(it) : base) * s;
      v.sphere(it.pos, r, LEVEL_COLOR[it.level], { pick: `e:${it.level}|${it.id}`, glow: glowOf ? glowOf(it) : 0, flat: it.n > 300 });
    }
  }

  _drawGroups(v, T) {
    const r = this.ranks();
    const topSet = new Set(this.top().map(x => x.group));
    for (let i = 0; i < this.N; i++) {
      const g = this.meta.groups[i];
      const p = this.gcur[i];
      const rank = r ? r[i] : null;
      const isTop = topSet.has(g), isSel = g === this.sel, isFollow = g === this.follow;
      const attack = (this.info.get(g) || {}).attack !== false;
      const col = isSel ? INK : isTop ? '#d8d6cf' : attack ? '#6f6d66' : '#57554f';
      const rad = isSel ? 0.62 : isTop ? 0.45 : 0.2;
      v.sphere(p, rad, col, { pick: `g:${g}`, ring: isFollow ? '#ffd166' : null, ringWidth: 2, glow: isSel ? 0.8 : 0 });
      if (isTop || isSel || isFollow) {
        const txt = rank !== null && rank !== undefined ? `${fmtRank(rank)}  ${trunc(g, 26)}` : g;
        v.label(V.add(p, [0, 1.15, 0]), txt, { size: isSel ? 12.5 : 11, weight: isSel ? 700 : 500,
          priority: isSel ? 30 : isFollow ? 25 : 20 - Math.min(19, rank || 19), color: isFollow && !isSel ? '#ffd166' : INK,
          border: isSel ? rgba(INK, 0.6) : null });
      }
    }
  }

  _edgesFrom(it, g, T, delay, o = {}) {
    const p = this.gcur[this.idx.get(g)];
    const mid = V.mul(V.add(it.pos, p), 0.5);
    const ctrl = [mid[0], mid[1] + (o.lift === undefined ? 2.5 : o.lift), mid[2]];
    const prog = ease((T - delay) / 0.9);
    if (prog <= 0) return;
    const col = o.color || LEVEL_COLOR[it.level];
    this.view.curve(it.pos, ctrl, p, col, o.width || 1.1, { alpha: o.alpha === undefined ? 0.35 : o.alpha, progress: prog, dash: o.dash });
    if (prog >= 1 && o.particles !== false) {
      const sp = o.speed || 0.35;
      const t = (((this.view.time * sp + hash01(it.id + g)) % 1) + 1) % 1;
      const q = this._bez(it.pos, ctrl, p, t);
      this.view.sphere(q, o.dot || 0.09, col, { flat: true, alpha: 0.95, noFog: true });
    }
  }
  _bez(a, c, b, t) { const u = 1 - t; return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1], u * u * a[2] + 2 * u * t * c[2] + t * t * b[2]]; }

  _column(v, g, segs, scale, labelText) {
    // a vertical stack of segments above a group (segs: [{color, value, label}])
    const p = this.gcur[this.idx.get(g)];
    let y = p[1] + 1.9;
    for (const s of segs) {
      const hgt = Math.max(0, s.value * scale);
      if (hgt <= 0.001) continue;
      const a = [p[0], y, p[2]], b = [p[0], y + hgt, p[2]];
      v.line(a, b, s.color, 7, { alpha: 0.95, zbias: -0.2 });
      if (s.label) v.label(V.add(b, [0.45, -hgt / 2, 0]), s.label, { size: 10.5, align: 'left', color: INK2, priority: 12 });
      y += hgt + 0.06;
    }
    if (labelText) v.label([p[0], y + 0.5, p[2]], labelText, { size: 11, weight: 700, priority: 28, force: true });
  }

  /* ---- set-based: shared items ------------------------------------------------------------ */
  _drawSet(v, T) {
    const top = this.top();
    const ex = this.explain();
    top.forEach((x, n) => {
      const isSel = x.group === this.sel;
      for (const it of this.items) {
        if (!it.groups.includes(x.i)) continue;
        this._edgesFrom(it, x.group, T, 0.35 + 0.04 * n, { alpha: isSel ? 0.75 : 0.18, width: isSel ? 1.5 : 0.9, particles: isSel || n < 3 });
      }
    });
    if (ex && ex.rows && T > 1.2) {
      const row = ex.rows.find(r => r.group === this.sel);
      if (row) {
        const W = row.weight_total || 1;
        const segs = row.terms.map(t => ({ color: LEVEL_COLOR[t.level], value: (t.similarity || 0) * t.weight / W,
          label: `${LEVEL_LABEL[t.level]} ${fmt(t.similarity, 2)} × ${fmt(t.weight, 1)}` }));
        this._column(v, this.sel, segs, 7, `score ${fmt(row.score, 3)}`);
      }
    }
  }

  /* ---- Random Forest: the trees that agree ------------------------------------------------- */
  _drawRF(v, T) {
    const ex = this.explain();
    if (!ex || !ex.rows) return;
    const row = ex.rows.find(r => r.group === this.sel) || ex.rows[0];
    if (!row) return;
    const bits = row.trees || '';
    const n = ex.n_trees || bits.length || 100, side = Math.ceil(Math.sqrt(n)), sp = 1.45;
    const gp = this.gcur[this.idx.get(row.group)];
    for (let k = 0; k < n; k++) {
      const x = (k % side - (side - 1) / 2) * sp, z = (Math.floor(k / side) - (side - 1) / 2) * sp;
      const base = [x, FOREST_Y, z];
      const shown = ease((T - 0.3 - k * 0.012) / 0.3);
      if (shown <= 0) continue;
      const agree = bits[k] === '1';
      v.line(base, V.add(base, [0, 0.35, 0]), '#6b5a43', 2, { alpha: 0.8 });
      v.cone(V.add(base, [0, 0.3, 0]), 1.1 * shown, 0.42, agree ? '#64c27f' : '#4a4945', { alpha: agree ? 1 : 0.7, pick: `t:${k}` });
      if (agree && T > 1.6) {
        const top = V.add(base, [0, 1.4, 0]);
        const ctrl = [(top[0] + gp[0]) / 2, (top[1] + gp[1]) / 2 + 2, (top[2] + gp[2]) / 2];
        const prog = ease((T - 1.6 - k * 0.006) / 0.8);
        v.curve(top, ctrl, gp, '#64c27f', 0.8, { alpha: 0.28, progress: prog, segments: 10 });
      }
    }
    v.label([0, FOREST_Y - 1.2, (side / 2 + 0.5) * sp], `${(row.trees.match(/1/g) || []).length} of ${n} trees put the evidence in the same leaf as ${row.group}`,
      { size: 12, weight: 600, priority: 40, force: true });
    // the evidence also reaches the selected group's leaves through its items: show the items it shares
    for (const it of this.items) if (it.groups.includes(this.idx.get(row.group))) this._edgesFrom(it, row.group, T, 0.4, { alpha: 0.25, particles: false });
  }

  /* ---- XGBoost: contributions -------------------------------------------------------------- */
  _drawXGB(v, T) {
    const ex = this.explain();
    if (!ex || !ex.rows) return;
    const row = ex.rows.find(r => r.group === this.sel) || ex.rows[0];
    const contrib = new Map((row.items || []).map(c => [`${c.level}|${c.item.id}`, c.value || 0]));
    const mx = Math.max(1e-6, ...[...contrib.values()].map(Math.abs));
    for (const [key, val] of contrib) {
      const it = this.itemByKey.get(key);
      if (!it) continue;
      const pos = val >= 0;
      this._edgesFrom(it, row.group, T, 0.35, { alpha: 0.25 + 0.6 * Math.abs(val) / mx, width: 0.8 + 2.2 * Math.abs(val) / mx,
        color: pos ? LEVEL_COLOR[it.level] : '#e66767', dash: pos ? null : [4, 4], speed: pos ? 0.4 : -0.25 });
      if (Math.abs(val) / mx > 0.45 && T > 1) v.label(V.add(it.pos, [0, 0.55, 0]), `${it.id} ${val >= 0 ? '+' : ''}${fmt(val, 2)}`,
        { size: 10.5, color: pos ? INK : '#ff9b9b', priority: 15 });
    }
    if (T > 1.2) {
      const p = row.probability || 0;
      this._column(v, row.group, [{ color: ACCENT, value: p }], 7, `p = ${fmt(p, 3)}`);
    }
  }

  /* ---- GNN: message passing / embedding ---------------------------------------------------- */
  _drawGNN(v, T) {
    const ex = this.explain();
    if (!ex) return;
    if (this.embPos) return this._drawEmbedding(v, T, ex, 'cosine');
    const Q = [0, LEVEL_Y.techniques + 4.2, 0];
    v.sphere(Q, 0.55, INK, { glow: 1, pick: 'q' });
    v.label(V.add(Q, [0, 1.1, 0]), 'evidence node', { size: 11, weight: 600, priority: 35, force: true });
    // hop 1: the evidence node sends to its items
    this.items.forEach(it => {
      const prog = ease((T - 0.3) / 0.7);
      if (prog <= 0) return;
      v.line(Q, V.lerp(Q, it.pos, prog), LEVEL_COLOR[it.level], 0.8, { alpha: 0.3 });
      if (prog >= 1 && T < 4.5) {
        const t = ((T - 1) * 0.6 + hash01(it.id)) % 1;
        v.sphere(V.lerp(Q, it.pos, t), 0.08, LEVEL_COLOR[it.level], { flat: true, noFog: true });
      }
    });
    // hop 2: the items to the groups that hold them (the ten best groups)
    const rows = ex.rows || [];
    rows.forEach((row, n) => {
      const isSel = row.group === this.sel;
      for (const [lv, list] of Object.entries(row.shared || {})) {
        for (const item of list) {
          const it = this.itemByKey.get(`${lv}|${item.id}`);
          if (it) this._edgesFrom(it, row.group, T, 1.3 + 0.05 * n, { alpha: isSel ? 0.7 : 0.16, width: isSel ? 1.4 : 0.8, particles: isSel || n < 2, speed: 0.5 });
        }
      }
    });
  }

  /* ---- Siamese: embedding space ------------------------------------------------------------- */
  _drawSiamese(v, T) {
    const ex = this.explain();
    if (!ex || ex.empty) return;
    if (this.embPos) return this._drawEmbedding(v, T, ex, 'cosine');
    // ring view: the items sampled into combinations go to the selected group
    for (const it of this.items) if (it.groups.includes(this.idx.get(this.sel))) this._edgesFrom(it, this.sel, T, 0.4, { alpha: 0.3 });
  }

  _drawEmbedding(v, T, ex, what) {
    const q = this.embPos.query;
    v.sphere(q, 0.6, INK, { glow: 1.2, pick: 'q' });
    v.label(V.add(q, [0, 1.2, 0]), 'evidence', { size: 11.5, weight: 700, priority: 40, force: true });
    // light axes of the principal components
    const L = 14;
    [[[-L, 0, 0], [L, 0, 0], 'PC1'], [[0, -L, 0], [0, L, 0], 'PC2'], [[0, 0, -L], [0, 0, L], 'PC3']].forEach(([a, b, t]) => {
      v.line(a, b, AXIS, 1, { alpha: 0.5 });
      v.label(b, t, { size: 10, color: MUTED, bg: null, priority: 1 });
    });
    const rows = (ex.rows || []).slice(0, 10);
    const sims = rows.map(r => r.cosine || 0);
    const mx = Math.max(...sims, 1e-6), mn = Math.min(...sims, 0);
    rows.forEach((row, n) => {
      const p = this.gcur[this.idx.get(row.group)];
      const prog = ease((T - 0.6 - 0.08 * n) / 0.6);
      if (prog <= 0) return;
      const w = (row.cosine - mn) / Math.max(1e-6, mx - mn);
      const isSel = row.group === this.sel;
      v.line(q, V.lerp(q, p, prog), isSel ? INK : ACCENT, isSel ? 2 : 0.8 + 1.6 * w, { alpha: isSel ? 0.9 : 0.35 + 0.4 * w });
      if (prog >= 1) v.label(V.lerp(q, p, 0.55), `${what} ${fmt(row.cosine, 3)}`, { size: 10, color: INK2, priority: isSel ? 26 : 8 });
    });
  }

  /* ---- MIL: attention ------------------------------------------------------------------------- */
  _milAttention() {
    const ex = this.explain('mil');
    if (this._attCache && this._attCache.ex === ex) return this._attCache.map;
    const att = new Map();
    for (const [lv, list] of Object.entries((ex && ex.attention) || {})) list.forEach(a => att.set(`${lv}|${a.item.id}`, a.weight || 0));
    this._attCache = { ex, map: att };
    return att;
  }

  _drawMIL(v, T) {
    const ex = this.explain();
    if (!ex) return;
    const att = this._milAttention();
    // the attention weights (softmax within each level) resize the evidence items (drawn in _drawEvidence)
    this.items.forEach(it => {
      const w = att.get(`${it.level}|${it.id}`);
      if (w !== undefined && w > 0.2 && T > 0.8) v.label(V.add(it.pos, [0, 0.6 + Math.sqrt(w) * 0.7, 0]), `${it.id} ${fmt(w, 2)}`, { size: 10.5, priority: 14 + w });
    });
    const sel = this.idx.get(this.sel);
    for (const it of this.items) if (it.groups.includes(sel)) this._edgesFrom(it, this.sel, T, 0.5, { alpha: 0.25 + 0.6 * Math.sqrt(att.get(`${it.level}|${it.id}`) || 0) });
    const row = (ex.rows || []).find(r => r.group === this.sel);
    if (row && T > 1.2) {
      const tot = row.levels.reduce((a, l) => a + Math.max(0, (l.similarity || 0) * (l.weight || 0)), 0) || 1;
      const segs = row.levels.map(l => ({ color: LEVEL_COLOR[l.level], value: Math.max(0, (l.similarity || 0) * (l.weight || 0)) / tot * Math.max(0, Math.min(1, row.score || 0)),
        label: `${LEVEL_LABEL[l.level]} sim ${fmt(l.similarity, 2)} × w ${fmt(l.weight, 2)}` }));
      this._column(v, this.sel, segs, 7, `score ${fmt(row.score, 3)}`);
    }
  }

  /* ---- Hybrid: the eight rankings fused --------------------------------------------------------- */
  _drawHybrid(v, T) {
    const cfg = this.cfg();
    const W = this.meta.weights[cfg] || {};
    const sumW = this.methods.reduce((a, m) => a + (W[m] || 0), 0) || 1;
    const K = this.meta.wrrf_k || 60;
    const top = this.top();
    const maxS = Math.max(...top.map(x => x.score || 0), 1e-9);
    top.forEach((x, n) => {
      const isSel = x.group === this.sel;
      if (!isSel && T < 1.0) return;
      const segs = this.methods.map((m, j) => {
        const r = this.res.methods[m].ranks[x.i];
        return { color: j % 2 ? '#8f8d86' : '#d8d6cf', value: ((W[m] || 0) / sumW) / (K + r), label: isSel ? `${ABBR[m]} #${fmtRank(r)}` : null };
      });
      if (isSel || n < 10) this._column(v, x.group, segs, 7 / maxS, isSel ? `WRRF ${fmt(x.score, 5)}` : null);
    });
    // where the selected group stands in each approach: ghosts on its radial line, pulled together by the fusion
    if (this.sel && T > 0.4) {
      const i = this.idx.get(this.sel);
      const fused = this.gcur[i];
      this.methods.forEach((m, j) => {
        const r = this.res.methods[m].ranks[i];
        const p = this._ring(i, r);
        const ghost = [p[0], -3.6 + j * 1.05, p[2]];
        const prog = ease((T - 0.4 - 0.08 * j) / 0.6);
        v.sphere(ghost, 0.24, '#8f8d86', { alpha: 0.85, pick: `h:${m}` });
        v.line(ghost, V.lerp(ghost, fused, prog), INK2, 1, { alpha: 0.45, dash: [3, 3] });
        v.label(V.add(ghost, [0.5, 0, 0]), `${ABBR[m]} #${fmtRank(r)}`, { size: 10, align: 'left', color: INK2, priority: 10, bg: 'rgba(20,20,19,0.6)' });
      });
    }
    // the evidence the selected group shares
    const si = this.idx.get(this.sel);
    for (const it of this.items) if (it.groups.includes(si)) this._edgesFrom(it, this.sel, T, 0.3, { alpha: 0.22, particles: false });
  }

  /* ------------------------------------------------------------------ hud and panels */
  _methodBar() {
    const bar = $('#methodbar');
    bar.innerHTML = '';
    const mk = (key, text, title) => h('button', { type: 'button', class: this.mode === key ? 'on' : '', title,
      onclick: () => { this.setMode(key); this.hooks.onMode && this.hooks.onMode(key); } }, text);
    if (this.complete) {
      bar.append(h('div', { class: 'grp' }, h('span', { class: 'lbl' }, 'Hybrid'),
        ...HYB.map(([c, l]) => mk(`hyb:${c}`, l, `Hybrid Attribution Framework, ${l} weights`))));
    }
    bar.append(h('div', { class: 'grp' }, h('span', { class: 'lbl' }, 'ML'),
      ...['xgboost', 'gnn', 'siamese', 'mil', 'rf'].filter(m => this.methods.includes(m)).map(m => mk(m, SHORT[m], NAME[m]))));
    bar.append(h('div', { class: 'grp' }, h('span', { class: 'lbl' }, 'Math'),
      ...['featureMatching', 'jaccard', 'tvd'].filter(m => this.methods.includes(m)).map(m => mk(m, SHORT[m], NAME[m]))));
    if (this.mode === 'gnn' || this.mode === 'siamese') {
      bar.append(h('div', { class: 'grp' }, h('button', { type: 'button', class: this.embed ? 'on' : '', onclick: () => this.toggleEmbed(),
        title: 'Show the groups at their learned embeddings (first three principal components)' }, 'Embedding space')));
    }
  }
  _legend() {
    const box = $('#legend-attr');
    box.innerHTML = '';
    LEVELS.forEach(lv => box.append(h('div', {}, h('span', { class: 'sw', style: `background:${LEVEL_COLOR[lv]}` }), `${LEVEL_LABEL[lv]} evidence`)));
    box.append(h('div', {}, h('span', { class: 'sw', style: 'background:#d8d6cf' }), 'ten best groups (closer = higher)'));
    box.append(h('div', {}, h('span', { class: 'sw', style: 'background:#6f6d66' }), 'other groups'));
    box.append(h('div', {}, h('span', { class: 'sw', style: 'background:transparent;border:2px solid #ffd166' }), 'group followed'));
  }

  narration() {
    const m = this.mode;
    if (!this.res) return `Add evidence on the left, or load a CTI report, to see how each approach ranks the ${this.N} groups.`;
    const t = this.top();
    const first = t.length ? `${t[0].group} is ranked first` : '';
    if (this.isHybrid(m)) {
      return `<b>Hybrid Attribution Framework, ${HYB.find(x => x[0] === this.cfg())[1]} weights.</b> Each group's WRRF score adds w / (${this.meta.wrrf_k} + rank) over the eight approaches. Beams: what each approach adds to the ten best groups; grey dots: where the selected group stands in each approach. ${first}.`;
    }
    const ex = this.explain();
    const sel = this.sel;
    switch (m) {
      case 'rf': {
        const row = ex && ex.rows ? ex.rows.find(r => r.group === sel) : null;
        return `<b>Random Forest.</b> Each tree sends the evidence to one leaf; green trees send it to the same leaf as ${sel}'s profile${row ? ` (${(row.trees.match(/1/g) || []).length} of ${ex.n_trees})` : ''}. ${first}.`;
      }
      case 'xgboost': return `<b>XGBoost.</b> Threads show how much each evidence item raises (solid) or lowers (dashed red) ${sel}'s raw score; the softmax of the raw scores gives the probabilities. ${first}.`;
      case 'gnn': return this.embPos ? `<b>GNN, embedding space.</b> After message passing, the evidence node lands near the groups whose neighbourhoods resemble it; lines give the cosine similarity to the ten best groups. ${first}.`
        : `<b>GNN.</b> The evidence joins the graph as a new node (top). Message passing: first to its items, then from the items to the groups that hold them. Its embedding is then compared with each group's. ${first}.`;
      case 'siamese': return this.embPos ? `<b>Siamese network, embedding space.</b> ${ex && ex.n_samples ? `${ex.n_samples} combinations of one item per level` : 'Combinations of one item per level'} are encoded; their mean (white) is compared (cosine) with each group's embedding. ${first}.`
        : `<b>Siamese network.</b> Threads: the evidence items in ${sel}'s profile. Switch to the embedding space to see the comparison itself. ${first}.`;
      case 'mil': return `<b>MIL.</b> Bigger, brighter items get more attention within their level; the column gives ${sel}'s per-level similarity times the learned level weight. ${first}.`;
      default: return `<b>${NAME[m]}.</b> Threads: each evidence item found in one of the ten best groups; the column gives ${sel}'s per-level similarity times the level weight, and their weighted mean is its score. ${first}.`;
    }
  }

  renderInfo() {
    const box = $('#info-attr');
    box.innerHTML = '';
    const n = $('#narration');
    if (n) n.innerHTML = this.narration();
    const m = this.mode;
    const hyb = this.isHybrid(m);
    box.append(h('div', { class: 'mkind' }, hyb ? 'decision-level fusion' : KIND[m]));
    box.append(h('div', { class: 'mtitle' }, hyb ? `Hybrid (${HYB.find(x => x[0] === this.cfg())[1]})` : NAME[m]));
    box.append(h('div', { class: 'mdesc' }, hyb ? `Weighted reciprocal rank fusion of the eight rankings: score = Σ w·1/(${this.meta.wrrf_k} + rank), weights normalised (Table 5.1).` : DESC[m]));
    if (!this.res) { box.append(h('p', { class: 'note' }, 'No evidence yet.')); return; }
    const ex = this.explain();
    if (ex && ex.check !== undefined && ex.check !== null) {
      const ok = ex.check < 1e-6;
      box.append(h('div', { class: `check${ok ? '' : ' bad'}` }, ok ? '✓ recomputed from the model: identical to the tool\'s scores' : `recomputed vs tool: max difference ${fmt(ex.check, 6)}`));
    }
    if (ex && ex.error) box.append(h('div', { class: 'check bad' }, ex.error));
    box.append(h('h3', {}, 'Ten best groups'));
    const t = this.top();
    const mx = Math.max(...t.map(x => Math.abs(x.score || 0)), 1e-9);
    const ul = h('ul', { class: 'toplist' });
    t.forEach(x => ul.append(h('li', { class: `${x.group === this.sel ? 'sel' : ''} ${x.group === this.follow ? 'follow' : ''}`, onclick: () => this.select(x.group) },
      h('span', { class: 'r' }, fmtRank(x.rank)), h('span', { class: 'g' }, x.group), h('span', { class: 'v' }, fmt(x.score, hyb ? 5 : 3)),
      h('span', { class: 'bar' }, h('i', { style: `width:${Math.max(2, 100 * Math.abs(x.score || 0) / mx)}%` })))));
    box.append(ul);
    if (this.follow) {
      const r = this.ranks();
      const i = this.idx.get(this.follow);
      box.append(h('p', { class: 'note' }, `Followed: ${this.follow} is ranked ${fmtRank(r ? r[i] : null)} of ${this.N} here.`));
    }
    box.append(h('h3', {}, `Why ${this.sel || ''}`));
    const d = h('div', {});
    if (hyb) this._infoHybrid(d); else if (ex) this._infoMethod(d, m, ex);
    box.append(d);
  }

  _infoHybrid(d) {
    const cfg = this.cfg();
    const W = this.meta.weights[cfg] || {};
    const sumW = this.methods.reduce((a, m) => a + (W[m] || 0), 0) || 1;
    const K = this.meta.wrrf_k || 60;
    const i = this.idx.get(this.sel);
    const tbl = h('table', { class: 't' }, h('tr', {}, h('th', { class: 'l' }, 'Approach'), h('th', {}, 'Weight'), h('th', {}, 'Rank'), h('th', {}, 'Adds')));
    let tot = 0;
    this.methods.forEach(m => {
      const r = this.res.methods[m].ranks[i];
      const add = ((W[m] || 0) / sumW) / (K + r);
      tot += add;
      tbl.append(h('tr', {}, h('td', { class: 'l' }, SHORT[m]), h('td', {}, `${Math.round((W[m] || 0) * 8)}/8`), h('td', {}, fmtRank(r)), h('td', {}, fmt(add, 5))));
    });
    tbl.append(h('tr', {}, h('td', { class: 'l' }, h('b', {}, 'WRRF score')), h('td', {}), h('td', {}), h('td', {}, h('b', {}, fmt(tot, 5)))));
    d.append(tbl);
    d.append(h('p', { class: 'small' }, 'A rank of 1 adds w/61, a rank of 100 adds w/160: the fusion rewards agreement at the top of several rankings.'));
  }

  _infoMethod(d, m, ex) {
    const sel = this.sel;
    if (['jaccard', 'featureMatching', 'tvd'].includes(m)) {
      const row = (ex.rows || []).find(r => r.group === sel);
      if (!row) return d.append(h('p', { class: 'note' }, 'Select one of the ten best groups.'));
      const tbl = h('table', { class: 't' }, h('tr', {}, h('th', { class: 'l' }, 'Level'), h('th', {}, '|E|'), h('th', {}, '|A|'), h('th', {}, 'shared'), h('th', {}, 'sim.'), h('th', {}, 'weight')));
      row.terms.forEach(tm => tbl.append(h('tr', {}, h('td', { class: 'l' }, h('span', { class: 'sw', style: `background:${LEVEL_COLOR[tm.level]};margin-right:5px` }), LEVEL_LABEL[tm.level]),
        h('td', {}, tm.evidence), h('td', {}, tm.profile), h('td', {}, tm.shared), h('td', {}, fmt(tm.similarity, 3)), h('td', {}, fmt(tm.weight, 1)))));
      d.append(tbl, h('p', { class: 'small' }, `score = Σ sim × weight / Σ weight = ${fmt(row.score, 4)}`));
      const shared = row.terms.flatMap(tm => (tm.shared_items || []).map(x => `${x.id}${x.name ? ' ' + x.name : ''}`));
      if (shared.length) d.append(h('div', { class: 'small' }, `Shared: ${shared.slice(0, 18).join(' · ')}${shared.length > 18 ? ' …' : ''}`));
    } else if (m === 'rf') {
      const row = (ex.rows || []).find(r => r.group === sel) || ex.rows[0];
      d.append(h('div', { class: 'kv' }, h('span', { class: 'k' }, 'Trees agreeing'), h('span', {}, `${(row.trees.match(/1/g) || []).length} of ${ex.n_trees} (${pct(row.share || 0)})`)));
      if (ex.path_tests && ex.path_tests.length) {
        d.append(h('div', { class: 'small' }, `Tests on the evidence's path in the trees that agree with ${ex.rows[0].group} (present / absent item):`));
        ex.path_tests.slice(0, 10).forEach(pt => d.append(h('div', { class: 'hbar' },
          h('span', { class: 'nm' }, `${pt.present ? 'has' : 'lacks'} ${pt.item.id} ${pt.item.name || ''}`), h('span', { class: 'track' }, h('i', { style: `left:0;width:${Math.min(100, pt.trees)}%;background:${LEVEL_COLOR[pt.level]}` })))));
      }
    } else if (m === 'xgboost') {
      const row = (ex.rows || []).find(r => r.group === sel);
      if (!row) return d.append(h('p', { class: 'note' }, 'Contributions are computed for the five best groups.'));
      d.append(h('div', { class: 'kv' }, h('span', { class: 'k' }, 'Probability'), h('span', {}, fmt(row.probability, 4)),
        h('span', { class: 'k' }, 'Raw score'), h('span', {}, fmt(row.margin, 3)), h('span', { class: 'k' }, 'Bias'), h('span', {}, fmt(row.bias, 3)),
        h('span', { class: 'k' }, 'Items absent'), h('span', {}, fmt(row.absent_total, 3))));
      const mx = Math.max(...row.items.map(c => Math.abs(c.value || 0)), 1e-9);
      row.items.slice(0, 14).forEach(c => d.append(h('div', { class: 'hbar' }, h('span', { class: 'nm' }, `${c.item.id} ${c.item.name || ''}`),
        h('span', { class: 'track' }, h('i', { style: `${c.value >= 0 ? 'left:50%' : `left:${50 - 50 * Math.abs(c.value) / mx}%`};width:${50 * Math.abs(c.value) / mx}%;background:${c.value >= 0 ? LEVEL_COLOR[c.level] : '#e66767'}` })))));
    } else if (m === 'gnn') {
      const row = (ex.rows || []).find(r => r.group === sel);
      d.append(h('div', { class: 'small' }, `Edge weights by level: ${Object.entries(ex.level_weights || {}).map(([k, w]) => `${LEVEL_LABEL[k]} ${w}`).join(', ')}.`));
      if (row) {
        d.append(h('div', { class: 'kv' }, h('span', { class: 'k' }, 'Cosine'), h('span', {}, fmt(row.cosine, 4)), h('span', { class: 'k' }, 'Items shared'), h('span', {}, String(row.n_shared))));
        const shared = Object.values(row.shared || {}).flat().map(x => x.id);
        if (shared.length) d.append(h('div', { class: 'small' }, shared.slice(0, 24).join(' · ')));
      }
    } else if (m === 'siamese') {
      const row = (ex.rows || []).find(r => r.group === sel);
      d.append(h('div', { class: 'kv' }, h('span', { class: 'k' }, 'Samples'), h('span', {}, String(ex.n_samples || '–')), h('span', { class: 'k' }, 'Cosine'), h('span', {}, row ? fmt(row.cosine, 4) : '–')));
      (ex.samples || []).slice(0, 5).forEach((smp, k) => d.append(h('div', { class: 'small' }, `#${k + 1}: ${LEVELS.filter(l => smp[l]).map(l => smp[l].id).join(' + ')}`)));
    } else if (m === 'mil') {
      const row = (ex.rows || []).find(r => r.group === sel);
      if (row) {
        const tbl = h('table', { class: 't' }, h('tr', {}, h('th', { class: 'l' }, 'Level'), h('th', {}, 'E in A'), h('th', {}, 'A in E'), h('th', {}, 'sim.'), h('th', {}, 'weight')));
        row.levels.forEach(l => tbl.append(h('tr', {}, h('td', { class: 'l' }, LEVEL_LABEL[l.level]), h('td', {}, fmt(l.coverage, 2)), h('td', {}, fmt(l.profile_coverage, 2)), h('td', {}, fmt(l.similarity, 3)), h('td', {}, fmt(l.weight, 3)))));
        d.append(tbl, h('p', { class: 'small' }, `score = Σ sim × weight = ${fmt(row.score, 4)}`));
      } else d.append(h('p', { class: 'note' }, 'Per-level details are computed for the five best groups.'));
      for (const [lv, list] of Object.entries(ex.attention || {})) {
        d.append(h('div', { class: 'small', style: 'margin-top:6px' }, `${LEVEL_LABEL[lv]}: attention`));
        list.slice(0, 5).forEach(a => d.append(h('div', { class: 'hbar' }, h('span', { class: 'nm' }, `${a.item.id} ${a.item.name || ''}`),
          h('span', { class: 'track' }, h('i', { style: `left:0;width:${100 * (a.weight || 0)}%;background:${LEVEL_COLOR[lv]}` })))));
      }
    }
  }

  _tipText(id) {
    if (id === 'q') return 'The evidence (a new node / point)';
    if (id.startsWith('g:')) {
      const g = id.slice(2), i = this.idx.get(g), r = this.ranks();
      const inf = this.info.get(g) || {};
      const sizes = inf.sizes ? LEVELS.map(l => `${LEVEL_LABEL[l]} ${inf.sizes[l]}`).join(', ') : '';
      return `${g}${inf.gid ? ` (${inf.gid})` : inf.attack === false ? ' (AlienVault OTX entry)' : ''}\nrank ${fmtRank(r ? r[i] : null)} of ${this.N}\n${sizes}`;
    }
    if (id.startsWith('e:')) {
      const it = this.itemByKey.get(id.slice(2));
      if (!it) return '';
      return `${LEVEL_LABEL[it.level]}: ${it.id}${it.name ? ` — ${it.name}` : ''}\nin ${it.groups.length} of ${this.N} profiles`;
    }
    if (id.startsWith('t:')) return `Tree ${Number(id.slice(2)) + 1}`;
    if (id.startsWith('h:')) {
      const m = id.slice(2), i = this.idx.get(this.sel);
      return `${NAME[m]}: ${this.sel} ranked ${fmtRank(this.res.methods[m].ranks[i])}`;
    }
    return '';
  }
}
