/* studio.js — Attribution Studio: wiring of the 3-D attribution view and the attack generator.
   Every ranking, score and attack shown comes from the local server, which runs the thesis tools. */
import { $, h, api, LEVELS, LEVEL_COLOR, LEVEL_LABEL, tip } from './common.js';
import { AttrScene } from './attr3d.js';
import { AttackScene } from './attack3d.js';
import { GroupPicker } from './picker.js';

const ID_PATTERNS = [
  [/\bT\d{4}(?:\.\d{3})?\b/g, 'techniques'], [/\bS\d{4}\b/g, 'software'], [/\bCAPEC-\d+\b/gi, 'capec'],
  [/\bCWE-\d+\b/gi, 'cwe'], [/\bCVE-\d{4}-\d{4,7}\b/gi, 'cve'], [/\b[0-9a-f]{24}\b/g, 'pulse']];
const S = { meta: null, ev: {}, name: {}, byName: {}, follow: '', seq: 0, timer: null, attr: null, atk: null, groups: null, gmap: null,
  gindex: [], pk: {}, groupsP: null, atkInit: false, inc: {}, atkTac: null, atkSeq: 0 };
// the attack generator's tactics in kill-chain order (attack.py) and those an objective leaves out
const TACTIC_ORDER = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence', 'Privilege Escalation',
  'Defense Evasion', 'Evasion', 'Credential Access', 'Discovery', 'Lateral Movement', 'Collection', 'Command and Control', 'Exfiltration',
  'Inhibit Response Function', 'Impair Process Control', 'Impact'];
const OBJECTIVE_DROP = { espionage: ['Impact', 'Inhibit Response Function', 'Impair Process Control'], destruction: ['Exfiltration'] };

/* ------------------------------------------------------------------ status */
function status(html) { $('#status').innerHTML = html; }
function baseStatus(extra = '') {
  const m = S.meta;
  const errs = Object.entries(m.load_errors || {});
  status(`${m.tool} · <span class="${errs.length ? 'bad' : 'ok'}">${m.methods.length} of 8 approaches loaded</span>${errs.length ? ` (${errs.map(([k]) => k).join(', ')} failed)` : ''}${extra}`);
}

/* ------------------------------------------------------------------ evidence input */
function placeholder(lv) {
  return { techniques: 'T1566.001 or Spearphishing…', software: 'S0154 or Cobalt Strike…', capec: 'CAPEC-112…', cwe: 'CWE-79…',
    cve: 'CVE-2021-44228…', pulse: 'OTX pulse ID (24 hex)' }[lv];
}
function normalise(lv, raw) {
  let t = String(raw).trim();
  if (!t) return null;
  const cut = t.indexOf(' — ');
  if (cut > 0) t = t.slice(0, cut).trim();
  if (lv === 'pulse') return t.toLowerCase();
  const up = t.toUpperCase();
  if (S.name[lv].has(up)) return up;
  return S.byName[lv].get(t.toLowerCase()) || up;
}
function renderLevels() {
  const box = $('#levels');
  box.innerHTML = '';
  for (const lv of S.meta.features) {
    const list = h('datalist', { id: `dl-${lv}` });
    for (const [id, nm] of S.meta.vocab[lv]) list.append(h('option', { value: nm ? `${id} — ${nm}` : id }));
    const input = h('input', { type: 'text', id: `in-${lv}`, list: `dl-${lv}`, placeholder: placeholder(lv), autocomplete: 'off' });
    const add = () => { addFromText(lv, input.value); input.value = ''; };
    input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    input.addEventListener('change', () => { if (input.value.includes(' — ')) add(); });
    box.append(h('div', { class: 'level' },
      h('div', { class: 'level-head' }, h('span', {}, h('span', { class: 'sw', style: `background:${LEVEL_COLOR[lv]};margin-right:6px` }), LEVEL_LABEL[lv]),
        h('span', { class: 'count', id: `cnt-${lv}` }, '')),
      h('div', { class: 'chips', id: `chips-${lv}` }), input, list));
    renderChips(lv);
  }
}
function addFromText(lv, text) {
  for (const part of String(text).split(/[,;\n]+/)) {
    const id = normalise(lv, part);
    if (id && !S.ev[lv].includes(id)) S.ev[lv].push(id);
  }
  renderChips(lv);
  changed();
}
function renderChips(lv) {
  const box = $(`#chips-${lv}`);
  if (!box) return;
  box.innerHTML = '';
  const list = S.ev[lv];
  list.slice(0, 40).forEach(id => {
    const known = S.name[lv].has(id);
    const nm = S.name[lv].get(id) || '';
    box.append(h('span', { class: `chip${known ? '' : ' unknown'}`, 'data-tip': known ? `${id} ${nm}` : `${id}: not in the dataset, ignored by every approach` },
      h('span', {}, id), known && nm ? h('span', { class: 'nm' }, nm) : null,
      h('button', { type: 'button', 'aria-label': `remove ${id}`, onclick: () => { tip.show(null); S.ev[lv] = S.ev[lv].filter(x => x !== id); renderChips(lv); changed(); } }, '×')));
  });
  if (list.length > 40) box.append(h('span', { class: 'chip', style: 'background:transparent' }, `+${list.length - 40} more`));
  const known = list.filter(id => S.name[lv].has(id)).length;
  $(`#cnt-${lv}`).textContent = list.length ? `${known} known${list.length > known ? `, ${list.length - known} unknown` : ''}` : '';
}
function knownCount() { return S.meta.features.reduce((a, lv) => a + S.ev[lv].filter(id => S.name[lv].has(id)).length, 0); }
function setEvidence(ev, follow = null) {
  for (const lv of S.meta.features) S.ev[lv] = (ev[lv] || []).slice();
  if (follow !== null) { S.follow = follow; S.pk.follow.set(follow); S.attr.setFollow(follow); }
  S.meta.features.forEach(renderChips);
  changed(true);
}
function changed(now = false) {
  const n = knownCount();
  $('#run-note').textContent = n ? `${n} item(s) known to the dataset.` : 'No known item yet.';
  $('#run').disabled = n === 0;
  clearTimeout(S.timer);
  if (n === 0) { resetScene(); return; }
  if (!$('#auto').checked) return;
  S.timer = setTimeout(run, now ? 0 : 450);
}
function resetScene() {                              // no known evidence left: nothing to attribute
  S.seq++;                                           // an answer still on its way is dropped
  $('#busy').classList.remove('on');
  if (S.attr && S.attr.res) S.attr.clear();
  if (S.meta) baseStatus();
}
function clearAll() {
  $('#preset-cti').value = '';
  for (const k of ['group', 'ffA', 'ffB']) S.pk[k].set('');
  $('#paste').value = '';
  setEvidence({}, '');
}
function pasteIDs() {
  const text = $('#paste').value;
  for (const [re, lv] of ID_PATTERNS) {
    if (!S.meta.features.includes(lv)) continue;
    for (const m of text.matchAll(re)) {
      const id = normalise(lv, m[0]);
      if (id && !S.ev[lv].includes(id)) S.ev[lv].push(id);
    }
  }
  S.meta.features.forEach(renderChips);
  $('#paste').value = '';
  changed(true);
}
async function run() {
  if (knownCount() === 0) return;
  const seq = ++S.seq;
  $('#busy').classList.add('on');
  try {
    const t0 = performance.now();
    const res = await api('/api/attribute', { evidence: S.ev });
    if (seq !== S.seq) return;                         // a newer request is on its way
    S.attr.setResult(res);
    baseStatus(` · answered in ${((performance.now() - t0) / 1000).toFixed(2)} s`);
  } catch (e) {
    status(`<span class="bad">${e.message}</span>`);
  } finally {
    if (seq === S.seq) $('#busy').classList.remove('on');
  }
}

/* ------------------------------------------------------------------ presets */
function fillPresets() {
  const cti = $('#preset-cti');
  const byGroup = {};
  S.meta.presets.forEach((p, i) => { (byGroup[p.group] ||= []).push([i, p]); });
  for (const [g, arr] of Object.entries(byGroup)) {
    const og = h('optgroup', { label: g });
    arr.forEach(([i, p]) => og.append(h('option', { value: i }, p.label)));
    cti.append(og);
  }
  cti.addEventListener('change', () => {
    const p = S.meta.presets[cti.value];
    if (p) setEvidence(p.evidence, p.group);
  });
  const index = () => S.gindex;
  S.pk.follow = new GroupPicker($('#follow'), { index, onPick: g => { S.follow = g; S.attr.setFollow(g); } });
  S.pk.group = new GroupPicker($('#preset-group'), { index });
  S.pk.ffA = new GroupPicker($('#ff-attacker'), { index });
  S.pk.ffB = new GroupPicker($('#ff-imitated'), { index });
  const loadProfile = async mode => {
    const g = S.pk.group.value;
    if (!g) { S.pk.group.focus(); return; }
    const r = await api(`/api/profile?group=${encodeURIComponent(g)}&mode=${mode}`);
    setEvidence(r.evidence, g);
  };
  $('#load-full').addEventListener('click', () => loadProfile('full'));
  $('#load-half').addEventListener('click', () => loadProfile('half'));
  $('#load-ff').addEventListener('click', async () => {
    const a = S.pk.ffA.value, m = S.pk.ffB.value;
    if (!a) { S.pk.ffA.focus(); return; }
    if (!m || a === m) { S.pk.ffB.focus(); return; }
    const r = await api(`/api/profile?group=${encodeURIComponent(a)}&mode=falseflag&imitated=${encodeURIComponent(m)}`);
    setEvidence(r.evidence, a);
  });
  $('#paste-add').addEventListener('click', pasteIDs);
  $('#clear').addEventListener('click', clearAll);
  $('#run').addEventListener('click', run);
  $('#auto').addEventListener('change', () => changed(true));
}

/* ------------------------------------------------------------------ modes */
function setMode(mode) {
  document.querySelectorAll('.modes button').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  $('#mode-attr').classList.toggle('on', mode === 'attr');
  $('#mode-attack').classList.toggle('on', mode === 'attack');
  if (S.attr) { S.attr.view.active = mode === 'attr'; S.attr.view.resize(); }
  if (S.atk) { S.atk.view.active = mode === 'attack'; S.atk.view.resize(); }
  if (mode === 'attack' && !S.atkInit) { S.atkInit = true; initAttack(); }
  tip.show(null);
}

/* ------------------------------------------------------------------ attack generator */
/* the attack generator's list of groups (with their ATT&CK aliases), once it has read the records */
function attackGroups() {
  if (!S.groupsP) {
    S.groupsP = (async () => {
      for (let tries = 0; ; tries++) {
        const st = await api('/api/attack/status');
        if (st.status === 'ready') return (await api('/api/attack/groups')).groups;
        // 'not started' for long means the explorer was started with --no-attack
        if (st.status !== 'loading' && !(st.status === 'not started' && tries < 3)) throw new Error(`Attack generator ${st.status}`);
        await new Promise(r => setTimeout(r, 1500));
      }
    })();
    S.groupsP.catch(() => { S.groupsP = null; });
  }
  return S.groupsP;
}
function addAliases(groups) {                        // lets every group field find a group by alias too
  const by = new Map(S.gindex.map(g => [g.name, g]));
  for (const g of groups) {
    const e = by.get(g.name);
    if (e) { e.aliases = g.aliases || []; if (!e.gid) e.gid = g.gid || ''; }
  }
}
async function initAttack() {
  const note = $('#atk-note');
  const wait = setTimeout(() => { note.textContent = 'Reading the dataset records and the ATT&CK, CAPEC and CWE dumps (about 20 s)…'; }, 300);
  let groups;
  try { groups = await attackGroups(); } catch (e) { clearTimeout(wait); note.textContent = e.message; S.atkInit = false; return; }
  clearTimeout(wait);
  note.textContent = '';
  addAliases(groups);
  S.groups = groups;
  S.gmap = new Map(groups.map(g => [g.name, g]));
  S.atk = new AttackScene($('#cv-attack'), {
    onPlay: p => { $('#atk-play').textContent = p ? 'Pause' : 'Play'; },
    onFollow: on => $('#atk-follow').classList.toggle('on', on),
  });
  S.atk.view.active = $('#mode-attack').classList.contains('on');
  S.pk.atk = new GroupPicker($('#atk-group'), { index: () => S.gindex.filter(g => S.gmap.has(g.name)), allowEmpty: false, onPick: () => pickGroup(true) });
  S.pk.atk.set(S.gmap.has(S.follow) ? S.follow : 'APT29');
  pickGroup(true);
}
function currentGroup() {
  return S.gmap && S.pk.atk ? S.gmap.get(S.pk.atk.value) : null;
}
function pickGroup(generateNow = false) {
  const g = currentGroup();
  const kv = $('#atk-profile');
  kv.innerHTML = '';
  if (!g) { $('#atk-note').textContent = 'Unknown group: choose one from the list.'; return; }
  $('#atk-note').textContent = '';
  kv.append(h('span', { class: 'k' }, 'ATT&CK ID'), h('span', {}, g.gid || 'AlienVault OTX entry'));
  LEVELS.forEach(lv => kv.append(h('span', { class: 'k' }, LEVEL_LABEL[lv]), h('span', {}, String(g.sizes[lv]))));
  kv.append(h('span', { class: 'k' }, 'Tactics'), h('span', {}, String(Object.keys(g.tactics).length)));
  const sel = $('#atk-platform');
  const keep = sel.value;
  sel.innerHTML = '';
  sel.append(h('option', { value: 'any' }, 'any platform'));
  g.platforms.forEach(p => sel.append(h('option', { value: p }, p)));
  sel.value = g.platforms.includes(keep) ? keep : 'any';
  renderTactics(true);
  if (generateNow) generate();
}

/* tactics: a step is drawn from the profile for each tactic ticked (default: those the objective keeps) */
function groupTactics() {
  const g = currentGroup();
  if (!g) return [];
  const have = Object.keys(g.tactics);
  return TACTIC_ORDER.filter(t => have.includes(t)).concat(have.filter(t => !TACTIC_ORDER.includes(t)));
}
function defaultTactics() {
  const drop = OBJECTIVE_DROP[$('#atk-objective').value] || [];
  return groupTactics().filter(t => !drop.includes(t));
}
function renderTactics(reset = false) {
  const g = currentGroup();
  const box = $('#atk-tactics');
  box.innerHTML = '';
  if (!g) return;
  if (reset || !S.atkTac) S.atkTac = new Set(defaultTactics());
  for (const t of groupTactics()) {
    const cb = h('input', { type: 'checkbox', checked: S.atkTac.has(t) ? true : null });
    cb.addEventListener('change', () => { if (cb.checked) S.atkTac.add(t); else S.atkTac.delete(t); tacticsCount(); generate(currentSeed()); });
    box.append(h('label', {}, cb, h('span', {}, t),
      h('span', { class: 'small', 'data-tip': `${g.tactics[t]} technique${g.tactics[t] === 1 ? '' : 's'} of ${g.name}'s profile for this tactic (all platforms)` }, String(g.tactics[t]))));
  }
  tacticsCount();
}
function tacticsCount() {
  const all = groupTactics();
  $('#atk-tac-count').textContent = all.length ? `(${all.filter(t => S.atkTac && S.atkTac.has(t)).length} of ${all.length} ticked)` : '';
}
function tacticsParam() {                           // null when the ticks are the default ones: the plain generator
  const def = new Set(defaultTactics());
  const cur = S.atkTac || def;
  if (cur.size === def.size && [...def].every(t => cur.has(t))) return null;
  return TACTIC_ORDER.filter(t => cur.has(t)).concat([...cur].filter(t => !TACTIC_ORDER.includes(t)));
}

/* items to include in the attack: typed, from a CTI report of the thesis, or pasted */
function renderIncLevels() {
  const box = $('#atk-levels');
  box.innerHTML = '';
  for (const lv of S.meta.features) {
    const input = h('input', { type: 'text', list: `dl-${lv}`, placeholder: placeholder(lv), autocomplete: 'off' });
    const add = () => { addInc(lv, input.value); input.value = ''; };
    input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    input.addEventListener('change', () => { if (input.value.includes(' — ')) add(); });
    box.append(h('div', { class: 'level' },
      h('div', { class: 'level-head' }, h('span', {}, h('span', { class: 'sw', style: `background:${LEVEL_COLOR[lv]};margin-right:6px` }), LEVEL_LABEL[lv]),
        h('span', { class: 'count', id: `inc-cnt-${lv}` }, '')),
      h('div', { class: 'chips', id: `inc-chips-${lv}` }), input));
    renderIncChips(lv);
  }
}
function renderIncChips(lv) {
  const box = $(`#inc-chips-${lv}`);
  if (!box) return;
  box.innerHTML = '';
  const list = S.inc[lv];
  list.slice(0, 60).forEach(id => {
    const known = S.name[lv].has(id);
    const nm = S.name[lv].get(id) || '';
    box.append(h('span', { class: `chip${known ? '' : ' unknown'}`, 'data-tip': known ? `${id} ${nm}` : `${id}: not in the dataset` },
      h('span', {}, id), known && nm ? h('span', { class: 'nm' }, nm) : null,
      h('button', { type: 'button', 'aria-label': `remove ${id}`, onclick: () => { tip.show(null); S.inc[lv] = S.inc[lv].filter(x => x !== id); renderIncChips(lv); incChanged(); } }, '×')));
  });
  if (list.length > 60) box.append(h('span', { class: 'chip', style: 'background:transparent' }, `+${list.length - 60} more`));
  $(`#inc-cnt-${lv}`).textContent = list.length ? String(list.length) : '';
}
function incCount() { return S.meta ? S.meta.features.reduce((a, lv) => a + (S.inc[lv] || []).length, 0) : 0; }
function incChanged() {
  const n = incCount();
  $('#atk-inc-count').textContent = n ? `(${n} item${n === 1 ? '' : 's'})` : '';
  generate(currentSeed());
}
function addInc(lv, text) {
  for (const part of String(text).split(/[,;\n]+/)) {
    const id = normalise(lv, part);
    if (id && !S.inc[lv].includes(id)) S.inc[lv].push(id);
  }
  renderIncChips(lv);
  incChanged();
}
function setInc(ev) {
  for (const lv of S.meta.features) S.inc[lv] = [...new Set((ev[lv] || []).map(String))];
  S.meta.features.forEach(renderIncChips);
}
function incParam() {
  if (!incCount()) return null;
  return Object.fromEntries(S.meta.features.map(lv => [lv, S.inc[lv]]));
}
function fillAtkReports() {
  const sel = $('#atk-cti');
  const byGroup = {};
  S.meta.presets.forEach((p, i) => { (byGroup[p.group] ||= []).push([i, p]); });
  for (const [g, arr] of Object.entries(byGroup)) {
    const og = h('optgroup', { label: g });
    arr.forEach(([i, p]) => og.append(h('option', { value: i }, p.label)));
    sel.append(og);
  }
  sel.addEventListener('change', () => {
    const p = S.meta.presets[sel.value];
    if (!p) return;
    setInc(p.evidence);
    if (S.pk.atk && S.gmap && S.gmap.has(p.group) && S.pk.atk.value !== p.group) { S.pk.atk.set(p.group); pickGroup(false); }
    incChanged();
  });
}
function pasteInc() {
  const text = $('#atk-paste').value;
  for (const [re, lv] of ID_PATTERNS) {
    if (!S.meta.features.includes(lv)) continue;
    for (const m of text.matchAll(re)) {
      const id = normalise(lv, m[0]);
      if (id && !S.inc[lv].includes(id)) S.inc[lv].push(id);
    }
  }
  S.meta.features.forEach(renderIncChips);
  $('#atk-paste').value = '';
  incChanged();
}
function clearInc() {
  $('#atk-cti').value = '';
  setInc({});
  incChanged();
}
function currentSeed() {
  const v = parseInt($('#atk-seed').value, 10);
  return Number.isFinite(v) && v >= 0 ? v : null;
}
async function generate(seed = null) {
  const g = currentGroup();
  if (!g || !S.atk) return;
  const sd = seed === null ? Math.floor(Math.random() * 100000) : seed;
  const body = { group: g.name, seed: sd, platform: $('#atk-platform').value, objective: $('#atk-objective').value,
    tactics: tacticsParam(), include: incParam() };
  const seq = ++S.atkSeq;
  try {
    const a = await api('/api/attack', body);
    if (seq !== S.atkSeq) return;                     // a newer request is on its way
    $('#atk-seed').value = String(sd);
    S.atk.setAttack(a);
    const np = a.not_placed || [];
    $('#atk-note').textContent = np.length ? `Not placed: ${np.map(x => `${x.id} (${x.reason})`).join('; ')}.` : '';
    if (a.steps.length) { S.atk.playing = false; S.atk.togglePlay(); }
  } catch (e) { if (seq === S.atkSeq) $('#atk-note').textContent = e.message; }
}
function bindAttack() {
  $('#atk-platform').addEventListener('change', () => generate());
  $('#atk-objective').addEventListener('change', () => { renderTactics(true); generate(); });
  $('#atk-tac-all').addEventListener('click', () => { S.atkTac = new Set(groupTactics()); renderTactics(); generate(currentSeed()); });
  $('#atk-tac-none').addEventListener('click', () => { S.atkTac = new Set(); renderTactics(); generate(currentSeed()); });
  $('#atk-paste-add').addEventListener('click', pasteInc);
  $('#atk-inc-clear').addEventListener('click', clearInc);
  $('#atk-generate').addEventListener('click', () => generate());
  $('#atk-seed').addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    const v = parseInt(e.target.value, 10);
    if (Number.isFinite(v) && v >= 0) generate(v);
  });
  $('#atk-play').addEventListener('click', () => S.atk && S.atk.togglePlay());
  $('#atk-next').addEventListener('click', () => { if (S.atk) { S.atk.playing = false; $('#atk-play').textContent = 'Play'; S.atk.next(); } });
  $('#atk-prev').addEventListener('click', () => { if (S.atk) { S.atk.playing = false; $('#atk-play').textContent = 'Play'; S.atk.previous(); } });
  $('#atk-speed').addEventListener('input', e => { if (S.atk) S.atk.speed = Number(e.target.value); });
  $('#atk-follow').addEventListener('click', () => S.atk && S.atk.setFollow(!S.atk.follow));
  $('#atk-home').addEventListener('click', () => S.atk && S.atk.overview());
  document.addEventListener('keydown', e => {
    if (!$('#mode-attack').classList.contains('on') || !S.atk) return;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (e.key === ' ') { e.preventDefault(); S.atk.togglePlay(); }
    else if (e.key === 'ArrowRight') $('#atk-next').click();
    else if (e.key === 'ArrowLeft') $('#atk-prev').click();
  });
}

/* ------------------------------------------------------------------ start */
async function init() {
  try { S.meta = await api('/api/meta'); } catch (e) { status(`<span class="bad">${e.message}</span>`); return; }
  for (const lv of S.meta.features) {
    S.ev[lv] = [];
    S.inc[lv] = [];
    S.name[lv] = new Map(S.meta.vocab[lv]);
    S.byName[lv] = new Map(S.meta.vocab[lv].filter(([, n]) => n).map(([id, n]) => [n.toLowerCase(), id]));
  }
  baseStatus(S.meta.attack_status === 'ready' ? '' : ` · attack generator ${S.meta.attack_status}`);
  const info = new Map((S.meta.group_info || []).map(g => [g.name, g]));
  S.gindex = S.meta.groups.map(g => ({ name: g, gid: (info.get(g) || {}).gid || '', attack: (info.get(g) || {}).attack, aliases: [] }));
  attackGroups().then(addAliases).catch(() => {});
  S.attr = new AttrScene($('#cv-attr'), S.meta, {});
  renderLevels();
  fillPresets();
  renderIncLevels();
  fillAtkReports();
  changed();
  document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('#replay').addEventListener('click', () => S.attr.restart());
  $('#home').addEventListener('click', () => S.attr.view.reset());
  $('#spin').addEventListener('click', e => { S.attr.view.autoRotate = S.attr.view.autoRotate ? 0 : 0.12; e.target.classList.toggle('on', !!S.attr.view.autoRotate); });
  bindAttack();
  if (location.hash === '#attack') setMode('attack');
}
init();
