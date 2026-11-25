'use strict';
/* Attribution explorer front end. Every number shown comes from the local server, which runs the thesis tools. */

const LEVEL_COLOR = { techniques: '#2a78d6', software: '#eb6834', capec: '#1baf7a', cwe: '#eda100', cve: '#e87ba4', pulse: '#008300' };
const ORDER = ['xgboost', 'gnn', 'siamese', 'mil', 'rf', 'featureMatching', 'jaccard', 'tvd'];   // thesis table order
const SHORT = { xgboost: 'XGBoost', gnn: 'GNN', siamese: 'Siamese', mil: 'MIL', rf: 'RF', featureMatching: 'FM', jaccard: 'Jaccard', tvd: 'TVD' };
const KIND = { xgboost: 'ML model', gnn: 'ML model', siamese: 'ML model', mil: 'ML model', rf: 'ML model',
  featureMatching: 'Mathematical algorithm', jaccard: 'Mathematical algorithm', tvd: 'Mathematical algorithm' };
const DESC = {
  xgboost: 'Gradient-boosted trees with one class per group. The evidence becomes the 0/1 vector of every item the model knows; the probability it gives each group is the score.',
  gnn: 'Graph convolutional network over the graph linking groups to their items. The evidence joins the graph as a new node; after two rounds of message passing its embedding is compared with each group’s.',
  siamese: 'An encoder trained so that profiles of the same group lie close together. The evidence is sampled into combinations of one item per level, encoded and averaged, then compared with each group’s embedding.',
  mil: 'Attention-based multiple-instance learning: each level is a bag of items pooled with learned attention, compared with each group’s pooled profile and with how much each side covers the other.',
  rf: 'A random forest trained to recognise groups. A group’s score is the share of trees in which the evidence ends in the same leaf as the group’s profile.',
  featureMatching: 'Per level, the share of the evidence found in the group’s profile, |E ∩ A| / |E|, then a weighted mean over the levels present in the evidence.',
  jaccard: 'Per level, the items the evidence and the profile share out of all items in either, |E ∩ A| / |E ∪ A|, then a weighted mean over the levels present.',
  tvd: 'Per level, evidence and profile are uniform distributions over their items; similarity = 1 − total variation distance (0.5 when the group has no item at that level).'
};
const ID_PATTERNS = [
  [/\bT\d{4}(?:\.\d{3})?\b/g, 'techniques'], [/\bS\d{4}\b/g, 'software'], [/\bCAPEC-\d+\b/gi, 'capec'],
  [/\bCWE-\d+\b/gi, 'cwe'], [/\bCVE-\d{4}-\d{4,7}\b/gi, 'cve'], [/\b[0-9a-f]{24}\b/g, 'pulse']];
const HEAT = [[1, '#184f95', '#fff'], [5, '#2a78d6', '#fff'], [10, '#6da7ec', '#0b0b0b'], [20, '#9ec5f4', '#0b0b0b'],
  [50, '#cde2fb', '#0b0b0b'], [Infinity, '#f3f8fe', '#52514e']];

const S = { meta: null, ev: {}, follow: '', imitated: '', res: null, sel: null, weighting: 'no_weights', pick: {}, name: {}, byName: {} };

/* ------------------------------------------------------------------ helpers */
const $ = (q, r = document) => r.querySelector(q);
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k.nodeType ? k : String(k));
  return e;
}
const NS = 'http://www.w3.org/2000/svg';
function s(tag, attrs = {}, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);
const fmtRank = r => (r === null || r === undefined) ? '–' : (Number.isInteger(r) ? String(r) : r.toFixed(1));
const pct = x => `${(100 * x).toFixed(1)}%`;
const heat = r => HEAT.find(([lim]) => r <= lim);
const eighths = w => { const e = Math.round(w * 8); return Math.abs(w * 8 - e) < 1e-9 ? `${e}/8` : fmt(w, 2); };
const label = (lv, id) => S.name[lv]?.get(id) || '';
const itemText = it => it.name ? `${it.id} ${it.name}` : it.id;
function trunc(t, n) { return t.length > n ? t.slice(0, n - 1) + '…' : t; }
async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function groupIdx(g) { return S.meta.groups.indexOf(g); }
function rankOf(arr, g) { const i = groupIdx(g); return i < 0 ? null : arr[i]; }

/* tooltip */
const tip = $('#tip');
document.addEventListener('mouseover', e => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (t) { tip.textContent = t.getAttribute('data-tip'); tip.hidden = false; } else tip.hidden = true;
});
document.addEventListener('mousemove', e => {
  if (tip.hidden) return;
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > window.innerWidth - 8) x = e.clientX - w - 14;
  if (y + hh > window.innerHeight - 8) y = e.clientY - hh - 14;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
});

/* ------------------------------------------------------------------ input */
function renderLevels() {
  const box = $('#levels');
  box.innerHTML = '';
  for (const lv of S.meta.features) {
    const list = h('datalist', { id: `dl-${lv}` });
    for (const [id, nm] of S.meta.vocab[lv]) list.append(h('option', { value: nm ? `${id} — ${nm}` : id }));
    const input = h('input', { type: 'text', id: `in-${lv}`, list: `dl-${lv}`, placeholder: placeholder(lv), autocomplete: 'off' });
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); addFromText(lv, input.value); input.value = ''; }
    });
    input.addEventListener('change', () => { if (input.value.includes(' — ')) { addFromText(lv, input.value); input.value = ''; } });
    box.append(h('div', { class: 'level', id: `lv-${lv}` },
      h('div', { class: 'level-head' }, h('span', {}, h('span', { class: 'swatch', style: `background:${LEVEL_COLOR[lv]}` }), S.meta.level_label[lv]),
        h('span', { class: 'count', id: `cnt-${lv}` }, '')),
      h('div', { class: 'chips', id: `chips-${lv}` }), input, list));
    renderChips(lv);
  }
}
function placeholder(lv) {
  return { techniques: 'e.g. T1566.001 or Spearphishing', software: 'e.g. S0154 or Cobalt Strike', capec: 'e.g. CAPEC-112',
    cwe: 'e.g. CWE-79', cve: 'e.g. CVE-2021-44228', pulse: 'OTX pulse ID (24 hex characters)' }[lv];
}
function normalise(lv, raw) {
  let t = String(raw).trim();
  if (!t) return null;
  const cut = t.indexOf(' — ');
  if (cut > 0) t = t.slice(0, cut).trim();
  if (lv === 'pulse') return t.toLowerCase();
  const up = t.toUpperCase();
  if (S.name[lv].has(up)) return up;
  const byName = S.byName[lv].get(t.toLowerCase());
  return byName || up;
}
function addFromText(lv, text) {
  for (const part of String(text).split(/[,;\n]+/)) {
    const id = normalise(lv, part);
    if (id && !S.ev[lv].includes(id)) S.ev[lv].push(id);
  }
  renderChips(lv);
}
function renderChips(lv) {
  const box = $(`#chips-${lv}`);
  if (!box) return;
  box.innerHTML = '';
  const list = S.ev[lv];
  list.slice(0, 60).forEach(id => {
    const known = S.name[lv].has(id);
    box.append(h('span', { class: `chip${known ? '' : ' unknown'}`, 'data-tip': known ? `${id} ${label(lv, id)}` : `${id}: not in the dataset, ignored by every approach` },
      h('span', {}, id), known && label(lv, id) ? h('span', { class: 'nm' }, label(lv, id)) : null,
      h('button', { type: 'button', 'aria-label': `remove ${id}`, onclick: () => { S.ev[lv] = S.ev[lv].filter(x => x !== id); renderChips(lv); } }, '×')));
  });
  if (list.length > 60) box.append(h('span', { class: 'chip more' }, `+${list.length - 60} more`));
  const known = list.filter(id => S.name[lv].has(id)).length;
  $(`#cnt-${lv}`).textContent = list.length ? `${known} known${list.length > known ? `, ${list.length - known} unknown` : ''}` : '';
  updateRunNote();
}
function updateRunNote() {
  const n = S.meta.features.reduce((a, lv) => a + S.ev[lv].filter(id => S.name[lv].has(id)).length, 0);
  $('#run-note').textContent = n ? `${n} item(s) known to the dataset will be sent to the eight approaches.` : 'No known item yet.';
  $('#run').disabled = n === 0;
}
function setEvidence(ev, follow = '', imitated = '') {
  for (const lv of S.meta.features) S.ev[lv] = (ev[lv] || []).slice();
  S.follow = follow; S.imitated = imitated;
  $('#follow').value = follow;
  S.meta.features.forEach(renderChips);
}
function pasteIDs() {
  const text = $('#paste').value;
  let added = 0;
  for (const [re, lv] of ID_PATTERNS) {
    if (!S.meta.features.includes(lv)) continue;
    for (const m of text.matchAll(re)) {
      const id = normalise(lv, m[0]);
      if (id && !S.ev[lv].includes(id)) { S.ev[lv].push(id); added++; }
    }
  }
  S.meta.features.forEach(renderChips);
  $('#paste').value = '';
  $('#run-note').textContent += ` (${added} ID(s) added from the pasted text)`;
}

/* ------------------------------------------------------------------ presets */
function fillPresets() {
  const cti = $('#preset-cti');
  const byGroup = {};
  S.meta.presets.forEach((p, i) => { (byGroup[p.group] ||= []).push([i, p]); });
  for (const [g, arr] of Object.entries(byGroup)) {
    const og = h('optgroup', { label: g });
    arr.forEach(([i, p]) => og.append(h('option', { value: i }, `${p.label}${p.link ? '' : ''}`)));
    cti.append(og);
  }
  cti.addEventListener('change', () => {
    const p = S.meta.presets[cti.value];
    if (p) { setEvidence(p.evidence, p.group); $('#run-note').textContent += ` Source: ${p.link}`; }
  });
  for (const id of ['preset-group', 'ff-attacker', 'ff-imitated', 'follow']) {
    const sel = $(`#${id}`);
    S.meta.groups.forEach(g => sel.append(h('option', { value: g }, g)));
  }
  $('#follow').addEventListener('change', e => { S.follow = e.target.value; if (S.res) renderResults(false); });
  const loadProfile = async mode => {
    const g = $('#preset-group').value;
    if (!g) return;
    const r = await api(`/api/profile?group=${encodeURIComponent(g)}&mode=${mode}`);
    setEvidence(r.evidence, g);
  };
  $('#load-full').addEventListener('click', () => loadProfile('full'));
  $('#load-half').addEventListener('click', () => loadProfile('half'));
  $('#load-ff').addEventListener('click', async () => {
    const a = $('#ff-attacker').value, m = $('#ff-imitated').value;
    if (!a || !m || a === m) return;
    const r = await api(`/api/profile?group=${encodeURIComponent(a)}&mode=falseflag&imitated=${encodeURIComponent(m)}`);
    setEvidence(r.evidence, a, m);
  });
  $('#paste-add').addEventListener('click', pasteIDs);
  $('#clear').addEventListener('click', () => setEvidence({}, ''));
  $('#run').addEventListener('click', run);
}

/* ------------------------------------------------------------------ run */
async function run() {
  const btn = $('#run');
  btn.disabled = true; btn.textContent = 'Running the eight approaches…';
  renderPipeline(true);
  try {
    S.res = await api('/api/attribute', { evidence: S.ev });
    S.pick = {};
    if (!S.sel || !S.res.methods[S.sel]) S.sel = ORDER.find(m => S.res.methods[m]);
    $('#empty').hidden = true; $('#results').hidden = false;
    renderResults(true);
  } catch (err) {
    $('#run-note').textContent = `Error: ${err.message}`;
  } finally {
    btn.textContent = 'Attribute'; updateRunNote();
  }
}
function renderResults(animate) {
  renderPipeline(false, animate);
  renderReceived();
  renderCards(animate);
  renderDetail();
  renderHybrid();
}

function renderPipeline(busy = false, animate = false) {
  const p = $('#pipeline');
  p.innerHTML = '';
  const r = S.res;
  const nItems = r ? S.meta.features.reduce((a, lv) => a + r.recognised[lv].length, 0) : 0;
  const nLv = r ? S.meta.features.filter(lv => r.recognised[lv].length).length : 0;
  const hy = r && r.hybrid[S.weighting];
  const first = hy ? hy.top[0].group : '';
  const followRank = hy && S.follow ? rankOf(hy.ranks, S.follow) : null;
  const steps = [
    ['Evidence', r ? `${nItems} known items, ${nLv} level(s)` : 'waiting', '#sec-received'],
    ['Eight approaches', r ? `each ranks all ${S.meta.groups.length} groups` : '8 rankings', '#sec-approaches'],
    ['WRRF fusion', `${S.meta.weighting_label[S.weighting]} weights`, '#sec-hybrid'],
    ['Attribution', hy ? `#1 ${first}${S.follow ? ` · ${S.follow}: rank ${fmtRank(followRank)}` : ''}` : (r && !r.complete ? 'needs all 8 approaches' : 'ranked list'), '#sec-hybrid']];
  steps.forEach(([k, v, target], i) => {
    const d = h('div', { class: `pipe-step${busy ? ' busy' : ''}`, onclick: () => $(target)?.scrollIntoView({ behavior: 'smooth' }) },
      h('div', { class: 'k' }, `${i + 1}. ${k}`), h('div', { class: 'v' }, v));
    p.append(d);
    if (r && !busy) setTimeout(() => d.classList.add('on'), animate ? 220 * i : 0);
  });
}

function renderReceived() {
  const r = S.res, box = $('#received');
  box.innerHTML = '';
  const wrap = h('div', { class: 'received' });
  for (const lv of S.meta.features) {
    const rec = r.recognised_named[lv], unk = r.unknown[lv];
    if (!rec.length && !unk.length) continue;
    const chips = h('div', { class: 'items' });
    rec.slice(0, 30).forEach(it => chips.append(h('span', { class: 'item', 'data-tip': itemText(it) }, it.id)));
    if (rec.length > 30) chips.append(h('span', { class: 'item muted' }, `+${rec.length - 30} more`));
    const el = h('div', { class: 'recv-level' },
      h('div', { class: 'head' }, h('span', {}, h('span', { class: 'swatch', style: `background:${LEVEL_COLOR[lv]}` }), S.meta.level_label[lv]),
        h('span', { class: 'muted' }, `${rec.length} used${unk.length ? `, ${unk.length} not in the dataset` : ''}`)), chips);
    if (lv === 'techniques' && rec.length && Object.keys(S.meta.tactics).length) el.append(tacticBars(rec.map(x => x.id)));
    wrap.append(el);
  }
  box.append(wrap);
  box.append(h('p', { class: 'note' }, 'Items that are not in the dataset are dropped by every approach before scoring, as in the thesis tools. The levels left empty are simply absent from the evidence.'));
}
function tacticBars(ids) {
  const counts = {};
  ids.forEach(t => (S.meta.tactics[t] || ['(unknown)']).forEach(ta => { counts[ta] = (counts[ta] || 0) + 1; }));
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = Math.max(...rows.map(r => r[1]));
  const W = 250, rowH = 15, svg = s('svg', { width: W, height: rows.length * rowH + 18, role: 'img', 'aria-label': 'TTPs by tactic' });
  svg.append(s('text', { x: 0, y: 10, 'font-size': 11, fill: '#52514e' }, 'TTPs by ATT&CK tactic'));
  rows.forEach(([ta, n], i) => {
    const y = 16 + i * rowH;
    svg.append(s('text', { x: 118, y: y + 10, 'font-size': 10.5, 'text-anchor': 'end', fill: '#52514e' }, trunc(ta, 22)));
    svg.append(s('rect', { x: 122, y: y + 2, width: Math.max(2, (W - 150) * n / max), height: 9, rx: 2, fill: LEVEL_COLOR.techniques }));
    svg.append(s('text', { x: 126 + (W - 150) * n / max, y: y + 10, 'font-size': 10.5, fill: '#0b0b0b' }, n));
  });
  return svg;
}

/* ------------------------------------------------------------------ approach cards */
function renderCards(animate) {
  const box = $('#cards');
  box.innerHTML = '';
  ORDER.filter(m => S.res.methods[m]).forEach((m, i) => {
    const pm = S.res.methods[m], ex = S.res.explain[m] || {};
    const card = h('div', { class: `card${S.sel === m ? ' sel' : ''}${ex.error ? ' err' : ''}`, style: animate ? 'opacity:0' : '' },
      h('div', { class: 'tag' }, KIND[m]), h('div', { class: 'title' }, S.meta.method_label[m]),
      h('div', { class: 'desc' }, DESC[m]),
      miniBars(pm.top.slice(0, 5), x => x.score, m),
      S.follow ? h('div', { class: 'follow-line' }, h('b', {}, S.follow), `: rank ${fmtRank(rankOf(pm.ranks, S.follow))} of ${S.meta.groups.length}`,
        S.imitated ? h('span', { class: 'muted' }, ` · imitated ${S.imitated}: ${fmtRank(rankOf(pm.ranks, S.imitated))}`) : null) : null,
      h('button', { type: 'button', class: 'ghost', onclick: () => { S.sel = m; renderCards(false); renderDetail(); $('#sec-detail').scrollIntoView({ behavior: 'smooth' }); } }, 'How it ranked ▸'));
    box.append(card);
    if (animate) setTimeout(() => { card.style.transition = 'opacity .35s'; card.style.opacity = 1; }, 300 + 70 * i);
  });
  const missing = Object.keys(S.meta.load_errors || {});
  if (missing.length) box.append(h('div', { class: 'card err' }, h('div', { class: 'title' }, 'Not loaded'), h('div', { class: 'desc' }, missing.map(m => `${m}: ${S.meta.load_errors[m]}`).join('\n'))));
}
function miniBars(rows, val, m) {
  const W = 232, rowH = 17, vmax = Math.max(1e-12, ...rows.map(r => Math.abs(val(r))));
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${rows.length * rowH}`, role: 'img', 'aria-label': `${m} top five` });
  rows.forEach((r, i) => {
    const y = i * rowH, follow = r.group === S.follow, imit = r.group === S.imitated;
    const w = Math.max(1.5, 68 * Math.abs(val(r)) / vmax);
    svg.append(s('g', { 'data-tip': `${r.group}\nrank ${fmtRank(r.rank)} · score ${fmt(val(r), 4)}` },
      s('text', { x: 0, y: y + 12, 'font-size': 10.5, fill: '#52514e' }, fmtRank(r.rank)),
      s('text', { x: 20, y: y + 12, 'font-size': 11, 'font-weight': follow ? 700 : 400, fill: '#0b0b0b', 'text-decoration': imit ? 'underline' : null }, trunc(r.group, 17)),
      s('rect', { class: 'bar', x: 126, y: y + 4, width: w, height: 9, rx: 2, fill: follow ? '#0b0b0b' : '#2a78d6', style: `animation-delay:${80 * i}ms` }),
      s('text', { x: 130 + w, y: y + 12, 'font-size': 10, fill: '#52514e' }, fmt(val(r), 3))));
  });
  return svg;
}

/* ------------------------------------------------------------------ detail panel */
function renderDetail() {
  const box = $('#detail'), m = S.sel;
  box.innerHTML = '';
  if (!m) return;
  const ex = S.res.explain[m] || {};
  const head = h('div', { class: 'detail-head' },
    h('h2', {}, h('span', { class: 'step-no' }, '4'), `How ${S.meta.method_label[m]} ranked the groups`),
    ex.check === null || ex.check === undefined ? null :
      h('span', { class: `check${ex.check < 1e-9 ? '' : ' bad'}` }, ex.check < 1e-9 ? '✓ recomputed from the same model: identical to the tool’s scores' : `recomputed scores differ by up to ${ex.check.toExponential(2)}`));
  box.append(head);
  if (ex.error) { box.append(h('p', { class: 'note' }, `This panel could not be built: ${ex.error}. The ranking above is still the tool’s own.`)); return; }
  const k = ex.kind;
  if (k === 'setbased') detailSetBased(box, m, ex);
  else if (k === 'rf') detailRF(box, ex);
  else if (k === 'xgboost') detailXGB(box, ex);
  else if (k === 'gnn') detailGNN(box, ex);
  else if (k === 'siamese') detailSiamese(box, ex);
  else if (k === 'mil') detailMIL(box, ex);
}
function groupTabs(rows, key, onPick) {
  const cur = S.pick[key] || rows[0]?.group;
  const t = h('div', { class: 'tabs', role: 'tablist' });
  rows.forEach(r => t.append(h('button', { type: 'button', class: r.group === cur ? 'on' : '', onclick: () => { S.pick[key] = r.group; onPick(); } }, trunc(r.group, 24))));
  return [t, cur];
}
function levelLegend(levels) {
  return h('div', { class: 'legend' }, levels.map(lv => h('span', {}, h('span', { class: 'swatch', style: `background:${LEVEL_COLOR[lv]}` }), S.meta.level_label[lv] || lv)));
}

function detailSetBased(box, m, ex) {
  const W = ex.weights;
  const formula = {
    jaccard: 'simₗ = |Eₗ ∩ Aₗ| / |Eₗ ∪ Aₗ|        score = Σₗ wₗ · simₗ / Σₗ wₗ   (levels present in the evidence)',
    featureMatching: 'simₗ = |Eₗ ∩ Aₗ| / |Eₗ|        score = Σₗ wₗ · simₗ / Σₗ wₗ   (levels present in the evidence)',
    tvd: 'simₗ = 1 − ½ Σₓ |Pₗ(x) − Qₗ(x)|  (uniform over Aₗ and Eₗ; 0.5 if Aₗ is empty)        score = Σₗ wₗ · simₗ / Σₗ wₗ' }[m];
  box.append(h('div', { class: 'formula' }, formula));
  box.append(h('p', { class: 'hint' }, `E = the evidence at a level, A = the group’s profile at that level. Level weights: ${Object.entries(W).map(([lv, w]) => `${S.meta.level_label[lv] || lv} ${w}`).join(', ')}. Each bar is a group’s score split by level (wₗ · simₗ / Σw).`));
  const levels = Object.keys(W).filter(lv => ex.rows.some(r => r.terms.some(t => t.level === lv)));
  box.append(levelLegend(levels));
  const rows = ex.rows, maxS = Math.max(1e-12, ...rows.map(r => r.score));
  const Wd = 860, rowH = 22, lab = 210;
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${Wd} ${rows.length * rowH + 4}`, role: 'img', 'aria-label': 'score by level' });
  rows.forEach((r, i) => {
    const y = i * rowH, follow = r.group === S.follow;
    let x = lab;
    const g = s('g', { style: 'cursor:pointer', onclick: () => { S.pick[m] = r.group; renderDetail(); } });
    g.append(s('text', { x: 0, y: y + 15, 'font-size': 12, 'font-weight': follow ? 700 : 400, fill: '#0b0b0b' }, `${fmtRank(S.res.methods[m].top[i]?.rank)}  ${trunc(r.group, 26)}`));
    r.terms.forEach(t => {
      const part = t.weight * t.similarity / r.weight_total;
      const w = (Wd - lab - 70) * part / maxS;
      if (w > 0.2) g.append(s('rect', { class: 'seg', style: `animation-delay:${40 * i}ms`, x, y: y + 5, width: Math.max(0, w - 1), height: 12, fill: LEVEL_COLOR[t.level],
        'data-tip': `${r.group} · ${S.meta.level_label[t.level]}\nshared ${t.shared} of ${t.evidence} evidence items (profile ${t.profile}${m === 'jaccard' ? `, union ${t.union}` : ''})\nsim ${fmt(t.similarity, 4)} × w ${t.weight} / ${r.weight_total} = ${fmt(part, 4)}` }));
      x += w;
    });
    g.append(s('text', { x: x + 6, y: y + 15, 'font-size': 11, fill: '#52514e' }, fmt(r.score, 4)));
    svg.append(g);
  });
  box.append(svg);
  const [tabs, cur] = groupTabs(rows, m, renderDetail);
  box.append(h('h3', {}, 'Level by level for one group'), tabs);
  const row = rows.find(r => r.group === cur);
  if (!row) return;
  const tbl = h('table', { class: 't' }, h('tr', {}, h('th', { class: 'l' }, 'Level'), h('th', {}, 'evidence |E|'), h('th', {}, 'profile |A|'), h('th', {}, 'shared'),
    m === 'jaccard' ? h('th', {}, 'union') : null, h('th', {}, 'similarity'), h('th', {}, 'weight'), h('th', {}, 'contribution')));
  row.terms.forEach(t => tbl.append(h('tr', {}, h('td', { class: 'l' }, h('span', { class: 'swatch', style: `background:${LEVEL_COLOR[t.level]}` }), S.meta.level_label[t.level]),
    h('td', {}, t.evidence), h('td', {}, t.profile), h('td', {}, t.shared), m === 'jaccard' ? h('td', {}, t.union) : null,
    h('td', {}, fmt(t.similarity, 4)), h('td', {}, t.weight), h('td', {}, fmt(t.weight * t.similarity / row.weight_total, 4)))));
  tbl.append(h('tr', {}, h('td', { class: 'l' }, h('b', {}, 'score')), h('td', { colspan: m === 'jaccard' ? 6 : 5 }), h('td', {}, h('b', {}, fmt(row.score, 4)))));
  box.append(tbl);
  const shared = row.terms.filter(t => t.shared_items.length);
  if (shared.length) {
    box.append(h('h3', {}, `Evidence items ${cur} also has`));
    shared.forEach(t => box.append(h('div', { class: 'items', style: 'margin:4px 0' }, h('span', { class: 'muted', style: 'font-size:12px;margin-right:4px' }, `${S.meta.level_label[t.level]}:`),
      t.shared_items.map(it => h('span', { class: 'item', 'data-tip': itemText(it) }, it.id)), t.shared > t.shared_items.length ? h('span', { class: 'muted' }, `+${t.shared - t.shared_items.length}`) : null)));
  }
}

function detailRF(box, ex) {
  box.append(h('p', { class: 'hint' }, `The forest has ${ex.n_trees} trees. The evidence is turned into the forest’s 0/1 vector and dropped down every tree; a group gets one vote from each tree in which its profile vector ends in the same leaf. Filled squares are the trees that voted for the group.`));
  const Wd = 860, lab = 210, sq = 5.2, gap = 0.8, rowH = 20;
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${Wd} ${ex.rows.length * rowH}`, role: 'img', 'aria-label': 'tree votes' });
  ex.rows.forEach((r, i) => {
    const y = i * rowH, follow = r.group === S.follow;
    svg.append(s('text', { x: 0, y: y + 13, 'font-size': 12, 'font-weight': follow ? 700 : 400, fill: '#0b0b0b' }, trunc(r.group, 28)));
    [...r.trees].forEach((b, k) => svg.append(s('rect', { class: b === '1' ? 'vote' : null, x: lab + k * (sq + gap), y: y + 4, width: sq, height: 11, rx: 1,
      fill: b === '1' ? (follow ? '#0b0b0b' : '#2a78d6') : '#e9e8e3', style: b === '1' ? `animation-delay:${6 * k + 40 * i}ms` : null,
      'data-tip': `tree ${k + 1}: ${b === '1' ? 'same leaf as' : 'different leaf from'} ${r.group}` })));
    svg.append(s('text', { x: lab + ex.n_trees * (sq + gap) + 8, y: y + 13, 'font-size': 11, fill: '#52514e' }, `${pct(r.share)}`));
  });
  box.append(svg);
  if (ex.path_tests && ex.path_tests.length) {
    box.append(h('h3', {}, `What sends the evidence to ${ex.rows[0].group}’s leaves`));
    box.append(h('p', { class: 'hint' }, `The tests met on the evidence’s own path, in the ${ex.agreeing_trees_top} trees where it lands with ${ex.rows[0].group}: “present” = the evidence has the item, “absent” = it does not. ${ex.n_tests ? `Of all ${ex.n_tests} tests on these paths, ${ex.n_present_tests} concern items the evidence has.` : ''}`));
    const max = Math.max(...ex.path_tests.map(t => t.trees));
    const W2 = 860, rH = 18;
    const svg2 = s('svg', { width: '100%', viewBox: `0 0 ${W2} ${ex.path_tests.length * rH}`, role: 'img', 'aria-label': 'path tests' });
    ex.path_tests.forEach((t, i) => {
      const y = i * rH;
      svg2.append(s('rect', { x: 0, y: y + 5, width: 9, height: 9, rx: 2, fill: LEVEL_COLOR[t.level] }));
      svg2.append(s('text', { x: 14, y: y + 13, 'font-size': 11.5, fill: '#0b0b0b' }, trunc(itemText(t.item), 58)));
      svg2.append(s('text', { x: 430, y: y + 13, 'font-size': 11, fill: t.present ? '#0b0b0b' : '#898781' }, t.present ? 'present' : 'absent'));
      svg2.append(s('rect', { class: 'bar', x: 490, y: y + 4, width: Math.max(2, 300 * t.trees / max), height: 10, rx: 2, fill: t.present ? '#2a78d6' : '#c3c2b7' }));
      svg2.append(s('text', { x: 496 + 300 * t.trees / max, y: y + 13, 'font-size': 11, fill: '#52514e' }, `${t.trees} trees`));
    });
    box.append(svg2);
  }
}

function divergingItems(items, title) {
  const W = 860, rH = 18, mid = 560, half = 200;
  const vmax = Math.max(1e-12, ...items.map(it => Math.abs(it.value)));
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${items.length * rH + 18}`, role: 'img', 'aria-label': title });
  svg.append(s('line', { x1: mid, x2: mid, y1: 0, y2: items.length * rH + 4, stroke: '#c3c2b7' }));
  svg.append(s('text', { x: mid - 6, y: items.length * rH + 15, 'font-size': 10.5, 'text-anchor': 'end', fill: '#898781' }, '← lowers this group'));
  svg.append(s('text', { x: mid + 6, y: items.length * rH + 15, 'font-size': 10.5, fill: '#898781' }, 'raises this group →'));
  items.forEach((it, i) => {
    const y = i * rH, w = half * Math.abs(it.value) / vmax, pos = it.value >= 0;
    svg.append(s('rect', { x: 0, y: y + 5, width: 9, height: 9, rx: 2, fill: LEVEL_COLOR[it.level] }));
    svg.append(s('text', { x: 14, y: y + 13, 'font-size': 11.5, fill: '#0b0b0b' }, trunc(itemText(it.item), 50)));
    svg.append(s('rect', { class: pos ? 'bar' : 'bar neg', style: `animation-delay:${30 * i}ms`, x: pos ? mid : mid - w, y: y + 4, width: Math.max(1, w), height: 10, rx: 2, fill: pos ? '#2a78d6' : '#e34948', 'data-tip': `${itemText(it.item)}\ncontribution ${fmt(it.value, 4)}` }));
    svg.append(s('text', { x: pos ? mid + w + 4 : mid - w - 4, y: y + 13, 'font-size': 10.5, 'text-anchor': pos ? 'start' : 'end', fill: '#52514e' }, fmt(it.value, 3)));
  });
  return svg;
}
function detailXGB(box, ex) {
  box.append(h('p', { class: 'hint' }, `The evidence is the model’s 0/1 input vector (${ex.n_present} of ${ex.n_features} items present). Each group has its own boosted trees; their summed output is the group’s raw score, and a softmax over the ${ex.n_classes} groups turns these into the probabilities ranked below.`));
  const top = ex.rows;
  box.append(h('h3', {}, 'Probability given to each group'));
  box.append(barRows(top.map(r => ({ group: r.group, v: r.probability })), v => pct(v)));
  const [tabs, cur] = groupTabs(top, 'xgboost', renderDetail);
  box.append(h('h3', {}, 'Which evidence items drive one group’s raw score'), tabs);
  const r = top.find(x => x.group === cur);
  if (!r) return;
  box.append(h('div', { class: 'formula' }, `raw score ${fmt(r.margin, 3)} = base ${fmt(r.bias, 3)} + items present ${fmt(r.margin - r.bias - r.absent_total, 3)} + items absent ${fmt(r.absent_total, 3)}`));
  box.append(divergingItems(r.items, 'XGBoost contributions'));
  box.append(h('p', { class: 'note' }, 'Contributions follow each tree’s decision path (XGBoost’s approximate per-feature contributions). Absent items count too: the trees also split on items the evidence does not contain.'));
}
function barRows(rows, fmtv) {
  const W = 860, rH = 18, lab = 230, vmax = Math.max(1e-12, ...rows.map(r => Math.abs(r.v)));
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${rows.length * rH}`, role: 'img', 'aria-label': 'scores' });
  rows.forEach((r, i) => {
    const y = i * rH, follow = r.group === S.follow, w = Math.max(1.5, (W - lab - 80) * Math.abs(r.v) / vmax);
    svg.append(s('text', { x: 0, y: y + 13, 'font-size': 12, 'font-weight': follow ? 700 : 400, fill: '#0b0b0b' }, trunc(r.group, 30)));
    svg.append(s('rect', { class: 'bar', style: `animation-delay:${40 * i}ms`, x: lab, y: y + 4, width: w, height: 10, rx: 2, fill: follow ? '#0b0b0b' : '#2a78d6' }));
    svg.append(s('text', { x: lab + w + 5, y: y + 13, 'font-size': 11, fill: '#52514e' }, fmtv(r.v)));
  });
  return svg;
}
function embeddingMap(mp, topRows, title) {
  const W = 420, H = 320, pad = 18;
  const xs = mp.points.map(p => p.x).concat(mp.query.x), ys = mp.points.map(p => p.y).concat(mp.query.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = x => pad + (W - 2 * pad) * (x - x0) / Math.max(1e-12, x1 - x0);
  const Y = y => H - pad - (H - 2 * pad) * (y - y0) / Math.max(1e-12, y1 - y0);
  const topSet = new Map(topRows.map((r, i) => [r.group, i]));
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });
  svg.append(s('rect', { x: 0.5, y: 0.5, width: W - 1, height: H - 1, fill: '#fff', stroke: '#e1e0d9', rx: 6 }));
  mp.points.forEach(p => {
    const i = topSet.get(p.group), follow = p.group === S.follow;
    svg.append(s('circle', { cx: X(p.x), cy: Y(p.y), r: i !== undefined || follow ? 4.5 : 2.6, fill: follow ? '#0b0b0b' : (i !== undefined ? '#2a78d6' : '#c3c2b7'),
      stroke: '#fff', 'stroke-width': 1, 'data-tip': `${p.group}${i !== undefined ? ` · top ${i + 1}` : ''}` }));
  });
  mp.points.filter(p => (topSet.has(p.group) && topSet.get(p.group) < 5) || p.group === S.follow).forEach(p => {
    svg.append(s('text', { x: X(p.x) + 6, y: Y(p.y) - 5, 'font-size': 10.5, fill: '#0b0b0b' }, trunc(p.group, 20)));
  });
  const qx = X(mp.query.x), qy = Y(mp.query.y);
  const star = [];
  for (let k = 0; k < 10; k++) { const a = Math.PI / 5 * k - Math.PI / 2, rr = k % 2 ? 4 : 9; star.push(`${qx + rr * Math.cos(a)},${qy + rr * Math.sin(a)}`); }
  svg.append(s('polygon', { points: star.join(' '), fill: '#eb6834', stroke: '#fff', 'stroke-width': 1.2, 'data-tip': 'the evidence (query)' }));
  svg.append(s('text', { x: qx + 10, y: qy + 14, 'font-size': 10.5, 'font-weight': 700, fill: '#0b0b0b' }, 'evidence'));
  return svg;
}
function detailGNN(box, ex) {
  const lw = Object.entries(ex.level_weights).map(([lv, w]) => `${S.meta.level_label[lv] || lv} ${w}`).join(', ');
  box.append(h('p', { class: 'hint' }, `The trained graph links each group to its items. The evidence is added as a new node, linked to its ${ex.n_evidence_nodes} items with the level’s edge weight (${lw}). Two graph-convolution layers pass information evidence → items → groups, so the evidence embedding reflects the groups that share its items. Groups are then ranked by cosine similarity to it.`));
  const grid = h('div', { class: 'grid2' });
  grid.append(h('div', {}, h('h3', {}, 'Two hops: evidence → its items → the top groups'), gnnGraph(ex)));
  grid.append(h('div', {}, h('h3', {}, 'Embedding space after message passing'), embeddingMap(ex.map, ex.rows, 'GNN embeddings'),
    h('p', { class: 'hint small' }, 'First two principal components of the group embeddings and of the evidence embedding (same forward pass). Blue: top 10; black: the followed group.')));
  box.append(grid);
  box.append(h('h3', {}, 'Cosine similarity of each group to the evidence'));
  box.append(barRows(ex.rows.map(r => ({ group: r.group, v: r.cosine })), v => fmt(v, 4)));
}
function gnnGraph(ex) {
  const groups = ex.rows.slice(0, 5);
  const linked = new Map();
  groups.forEach((g, gi) => Object.entries(g.shared).forEach(([lv, items]) => items.forEach(it => {
    const key = `${lv}|${it.id}`;
    if (!linked.has(key)) linked.set(key, { lv, it, groups: new Set() });
    linked.get(key).groups.add(gi);
  })));
  let nodes = [...linked.values()].sort((a, b) => b.groups.size - a.groups.size || a.lv.localeCompare(b.lv)).slice(0, 26);
  const order = S.meta.features;
  nodes.sort((a, b) => order.indexOf(a.lv) - order.indexOf(b.lv));
  const W = 420, rowH = 13.5, H = Math.max(200, nodes.length * rowH + 30);
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'GNN two-hop graph' });
  const qx = 22, qy = H / 2, ix = 175, gx = 335;
  const iy = i => 18 + i * ((H - 36) / Math.max(1, nodes.length - 1 || 1));
  const gy = i => 30 + i * ((H - 60) / Math.max(1, groups.length - 1));
  const wmax = Math.max(...Object.values(ex.level_weights));
  nodes.forEach((n, i) => {
    const y = nodes.length === 1 ? H / 2 : iy(i);
    svg.append(s('line', { class: 'hop1', pathLength: 1, x1: qx, y1: qy, x2: ix, y2: y, stroke: LEVEL_COLOR[n.lv], 'stroke-opacity': 0.55, 'stroke-width': 0.6 + 2.2 * ex.level_weights[n.lv] / wmax }));
    n.groups.forEach(gi => svg.append(s('line', { class: 'hop2', pathLength: 1, x1: ix, y1: y, x2: gx, y2: gy(gi), stroke: '#c3c2b7', 'stroke-width': 0.8 })));
  });
  nodes.forEach((n, i) => {
    const y = nodes.length === 1 ? H / 2 : iy(i);
    svg.append(s('circle', { cx: ix, cy: y, r: 4, fill: LEVEL_COLOR[n.lv], stroke: '#fff', 'data-tip': `${itemText(n.it)}\n${S.meta.level_label[n.lv]} · linked to ${n.groups.size} of these groups` }));
    svg.append(s('text', { x: ix - 7, y: y + 3.5, 'font-size': 9.5, 'text-anchor': 'end', fill: '#52514e' }, trunc(n.it.id, 16)));
  });
  groups.forEach((g, i) => {
    const follow = g.group === S.follow;
    svg.append(s('rect', { x: gx - 5, y: gy(i) - 5, width: 10, height: 10, rx: 2, fill: follow ? '#0b0b0b' : '#2a78d6', 'data-tip': `${g.group}\ncosine ${fmt(g.cosine, 4)} · shares ${g.n_shared} evidence items` }));
    svg.append(s('text', { x: gx + 9, y: gy(i) + 4, 'font-size': 10.5, 'font-weight': follow ? 700 : 400, fill: '#0b0b0b' }, trunc(g.group, 12)));
  });
  svg.append(s('circle', { cx: qx, cy: qy, r: 9, fill: '#eb6834', stroke: '#fff', 'stroke-width': 1.5, 'data-tip': 'the evidence node' }));
  svg.append(s('text', { x: qx - 6, y: qy + 24, 'font-size': 10.5, 'font-weight': 700, fill: '#0b0b0b' }, 'evidence'));
  if (linked.size > nodes.length) svg.append(s('text', { x: ix, y: H - 4, 'font-size': 10, 'text-anchor': 'middle', fill: '#898781' }, `${nodes.length} of ${linked.size} shared items shown`));
  if (!nodes.length) svg.append(s('text', { x: W / 2, y: H / 2, 'font-size': 11, 'text-anchor': 'middle', fill: '#898781' }, 'The top groups share no evidence item directly'));
  return svg;
}
function detailSiamese(box, ex) {
  if (ex.empty) { box.append(h('p', { class: 'note' }, 'No usable evidence.')); return; }
  box.append(h('p', { class: 'hint' }, `The encoder takes one item per level. From the evidence, ${ex.n_samples} combinations are drawn (seed 0; “–” where the evidence has no item at that level), each is encoded, and the ${ex.n_samples} embeddings are averaged into the evidence embedding.`));
  const lvls = S.meta.features;
  const tbl = h('table', { class: 't' }, h('tr', {}, h('th', { class: 'l' }, '#'), lvls.map(lv => h('th', { class: 'l' }, S.meta.level_label[lv]))));
  ex.samples.forEach((smp, i) => tbl.append(h('tr', {}, h('td', { class: 'l' }, i + 1), lvls.map(lv => h('td', { class: 'l', 'data-tip': smp[lv] ? itemText(smp[lv]) : '' }, smp[lv] ? smp[lv].id : '–')))));
  tbl.append(h('tr', {}, h('td', { class: 'l muted', colspan: lvls.length + 1 }, `… ${ex.n_samples - ex.samples.length} more combinations, then the mean of all ${ex.n_samples} embeddings`)));
  const grid = h('div', { class: 'grid2' });
  grid.append(h('div', {}, h('h3', {}, 'Sampled combinations (first 8)'), tbl));
  grid.append(h('div', {}, h('h3', {}, 'Stored group embeddings and the evidence'), embeddingMap(ex.map, ex.rows, 'Siamese embeddings'),
    h('p', { class: 'hint small' }, 'First two principal components of the 256-dimensional embeddings. Blue: top 10; black: the followed group.')));
  box.append(grid);
  box.append(h('h3', {}, 'Cosine similarity of each group to the evidence'));
  box.append(barRows(ex.rows.map(r => ({ group: r.group, v: r.cosine })), v => fmt(v, 4)));
}
function detailMIL(box, ex) {
  box.append(h('p', { class: 'hint' }, 'Each level of the evidence is a bag of items. A learned attention gives every item a weight (shown below); the weighted average is the level’s embedding. For each group the network also sees how much of the evidence the group covers and how much of the group the evidence covers, outputs one similarity per level, and combines them with learned level weights.'));
  const att = ex.attention, levels = Object.keys(att);
  const grid = h('div', { class: 'grid2' });
  levels.forEach(lv => {
    const rows = att[lv].slice(0, 10), max = Math.max(...rows.map(r => r.weight));
    const W = 420, rH = 16;
    const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${rows.length * rH}`, role: 'img', 'aria-label': `attention ${lv}` });
    rows.forEach((r, i) => {
      const y = i * rH, w = Math.max(1.5, 170 * r.weight / max);
      svg.append(s('text', { x: 0, y: y + 12, 'font-size': 11, fill: '#0b0b0b' }, trunc(itemText(r.item), 34)));
      svg.append(s('rect', { class: 'bar', style: `animation-delay:${40 * i}ms`, x: 210, y: y + 3, width: w, height: 9, rx: 2, fill: LEVEL_COLOR[lv], 'data-tip': `${itemText(r.item)}\nattention ${fmt(r.weight, 4)}` }));
      svg.append(s('text', { x: 214 + w, y: y + 12, 'font-size': 10, fill: '#52514e' }, fmt(r.weight, 3)));
    });
    grid.append(h('div', {}, h('h3', {}, `${S.meta.level_label[lv]}: attention over the evidence${att[lv].length > 10 ? ' (top 10)' : ''}`), svg));
  });
  box.append(grid);
  const [tabs, cur] = groupTabs(ex.rows, 'mil', renderDetail);
  box.append(h('h3', {}, 'Level similarities for one group'), tabs);
  const r = ex.rows.find(x => x.group === cur);
  if (!r) return;
  const tbl = h('table', { class: 't' }, h('tr', {}, h('th', { class: 'l' }, 'Level'), h('th', {}, 'evidence covered by the group'), h('th', {}, 'group covered by the evidence'),
    h('th', {}, 'level similarity'), h('th', {}, 'level weight'), h('th', {}, 'contribution')));
  r.levels.forEach(l => tbl.append(h('tr', {}, h('td', { class: 'l' }, h('span', { class: 'swatch', style: `background:${LEVEL_COLOR[l.level]}` }), S.meta.level_label[l.level] || l.level),
    h('td', {}, pct(l.coverage)), h('td', {}, pct(l.profile_coverage)), h('td', {}, fmt(l.similarity, 4)), h('td', {}, fmt(l.weight, 4)), h('td', {}, fmt(l.similarity * l.weight, 4)))));
  tbl.append(h('tr', {}, h('td', { class: 'l' }, h('b', {}, 'score')), h('td', { colspan: 4 }), h('td', {}, h('b', {}, fmt(r.score, 4)))));
  box.append(tbl);
  box.append(h('p', { class: 'note' }, 'Coverages below 1% are set to 1% by the model, and levels absent from the evidence still get a similarity: the network output for an empty bag.'));
}

/* ------------------------------------------------------------------ hybrid */
function renderHybrid() {
  const box = $('#hybrid');
  box.innerHTML = '';
  const r = S.res;
  if (!r.complete || !r.hybrid[S.weighting]) {
    box.append(h('p', { class: 'note' }, 'The hybrid needs the eight approaches; some are not loaded.'));
    return;
  }
  const tabs = h('div', { class: 'tabs' });
  Object.keys(S.meta.weights).forEach(cfg => tabs.append(h('button', { type: 'button', class: cfg === S.weighting ? 'on' : '',
    onclick: () => { S.weighting = cfg; renderHybrid(); renderPipeline(false, false); } }, S.meta.weighting_label[cfg])));
  box.append(tabs);
  const w = S.meta.weights[S.weighting], wsum = ORDER.reduce((a, m) => a + w[m], 0), k = S.meta.wrrf_k;
  box.append(h('div', { class: 'formula' }, `WRRF(g) = Σₘ (wₘ / Σw) / (${k} + rankₘ(g))`,
    h('br'), `w (eighths): ${ORDER.map(m => `${SHORT[m]} ${eighths(w[m])}`).join(' · ')}`));
  box.append(h('p', { class: 'hint' }, 'Each column is one approach’s rank of the group (darker = closer to the top). A rank adds w/(60 + rank) to the group’s score, so a first place counts most, but every approach votes for every group. Hover a cell to see what it adds.'));
  const hy = r.hybrid[S.weighting];
  const rows = hy.top.map(t => t.group);
  [S.follow, S.imitated].filter(g => g && !rows.includes(g)).forEach(g => rows.push(g));
  const lab = 210, cw = 64, W = lab + ORDER.length * cw + 250, rH = 24, top = 44;
  const svg = s('svg', { width: '100%', viewBox: `0 0 ${W} ${top + rows.length * rH + 8}`, role: 'img', 'aria-label': 'WRRF matrix' });
  ORDER.forEach((m, j) => {
    const x = lab + j * cw + cw / 2;
    svg.append(s('text', { x, y: 14, 'font-size': 11, 'text-anchor': 'middle', 'font-weight': 600, fill: '#0b0b0b', 'data-tip': S.meta.method_label[m] }, SHORT[m]));
    svg.append(s('text', { x, y: 30, 'font-size': 10.5, 'text-anchor': 'middle', fill: '#52514e' }, `w ${eighths(w[m])}`));
  });
  const xS = lab + ORDER.length * cw + 12;
  svg.append(s('text', { x: xS, y: 14, 'font-size': 11, 'font-weight': 600, fill: '#0b0b0b' }, 'WRRF score'));
  const smax = Math.max(...hy.scores.filter(v => v !== null));
  rows.forEach((g, i) => {
    const y = top + i * rH, gi = groupIdx(g), follow = g === S.follow, imit = g === S.imitated;
    if (i === hy.top.length) svg.append(s('line', { x1: 0, x2: W, y1: y - 2, y2: y - 2, stroke: '#c3c2b7', 'stroke-dasharray': '3 3' }));
    const fr = hy.ranks[gi];
    svg.append(s('text', { x: 0, y: y + 15, 'font-size': 12, 'font-weight': follow ? 700 : 400, fill: '#0b0b0b', 'text-decoration': imit ? 'underline' : null },
      `${fmtRank(fr)}  ${trunc(g, 24)}${imit ? ' (imitated)' : ''}`));
    ORDER.forEach((m, j) => {
      const rk = r.methods[m].ranks[gi];
      const [, bg, fg] = heat(rk);
      const add = (w[m] / wsum) / (k + rk);
      const x = lab + j * cw;
      svg.append(s('rect', { class: 'cell', style: `animation-delay:${110 * j}ms`, x: x + 2, y: y + 2, width: cw - 4, height: rH - 4, rx: 3, fill: bg, 'data-tip': `${g} · ${S.meta.method_label[m]}\nrank ${fmtRank(rk)} of ${S.meta.groups.length}\nadds (${eighths(w[m])} / ${fmt(wsum, 3)}) / (${k} + ${fmtRank(rk)}) = ${add.toExponential(3)}` }));
      svg.append(s('text', { class: 'cell', style: `animation-delay:${110 * j}ms`, x: x + cw / 2, y: y + 16, 'font-size': 11, 'text-anchor': 'middle', fill: fg }, fmtRank(rk)));
    });
    const sc = hy.scores[gi], bw = 160 * sc / smax;
    svg.append(s('rect', { class: 'bar', style: `animation-delay:${110 * ORDER.length + 40 * i}ms`, x: xS, y: y + 7, width: Math.max(1, bw), height: 10, rx: 2, fill: follow ? '#0b0b0b' : '#2a78d6' }));
    svg.append(s('text', { x: xS + bw + 5, y: y + 16, 'font-size': 10.5, fill: '#52514e' }, sc.toFixed(5)));
  });
  box.append(h('div', { class: 'matrix-wrap' }, svg));
  const legend = h('div', { class: 'legend' }, 'Rank:', HEAT.map(([lim, bg], i) => h('span', {}, h('span', { class: 'swatch', style: `background:${bg};border:1px solid #e1e0d9` }),
    i === 0 ? '1' : (lim === Infinity ? `> ${HEAT[i - 1][0]}` : `≤ ${lim}`))));
  box.append(legend);
  if (S.follow) {
    const fr = rankOf(hy.ranks, S.follow);
    box.append(h('p', { class: 'note' }, h('b', {}, S.follow), ` is ranked ${fmtRank(fr)} of ${S.meta.groups.length} by the hybrid with ${S.meta.weighting_label[S.weighting]} weights; individually: `,
      ORDER.map(m => `${S.meta.method_label[m]} ${fmtRank(rankOf(r.methods[m].ranks, S.follow))}`).join(', '), '.'));
  }
}

/* ------------------------------------------------------------------ start */
async function init() {
  try {
    S.meta = await api('/api/meta');
  } catch (err) {
    $('#status').textContent = `The local server is not answering: ${err.message}`;
    return;
  }
  for (const lv of S.meta.features) {
    S.ev[lv] = [];
    S.name[lv] = new Map(S.meta.vocab[lv].map(([id, nm]) => [id, nm]));
    S.byName[lv] = new Map(S.meta.vocab[lv].filter(([, nm]) => nm).map(([id, nm]) => [nm.toLowerCase(), id]));
  }
  const nLoaded = S.meta.methods.length, errs = Object.keys(S.meta.load_errors || {});
  $('#status').innerHTML = '';
  $('#status').append(h('span', {}, h('span', {}, `Tool: ${S.meta.tool} (${S.meta.features.length} feature types) · ${S.meta.groups.length} groups · `),
    h('span', { class: nLoaded === 8 ? 'ok' : 'bad' }, `${nLoaded}/8 approaches loaded`),
    errs.length ? h('span', { class: 'bad' }, ` (missing: ${errs.join(', ')})`) : null,
    S.meta.names_loaded ? null : h('span', { class: 'muted' }, ' · names not loaded')));
  renderLevels();
  fillPresets();
  renderPipeline(false);
  updateRunNote();
}
init();
