/* attack3d.js — the attack generator view: one possible attack of a group, step by step along the kill chain
   (left to right), each step dropping through the levels of abstraction (TTP, software, CAPEC, CWE, CVE, pulse). */
import { View3D, V, rgba, ease } from './engine3d.js';
import { LEVELS, LEVEL_COLOR, LEVEL_LABEL, LEVEL_Y, INK, INK2, MUTED, AXIS, ACCENT, $, h, fmt, trunc, tip } from './common.js';

const DX = 4.6;                 // distance between steps
const REVEAL = [['techniques', 0], ['software', 0.35], ['capec', 0.7], ['cwe', 1.0], ['cve', 1.3], ['pulse', 1.6]];
const DWELL = 3.4;              // seconds per step at speed 1
const AY = Object.fromEntries(LEVELS.map(lv => [lv, LEVEL_Y[lv] * 1.4]));   // level heights, spaced out more than in the attribution view

export class AttackScene {
  constructor(canvas, hooks = {}) {
    this.hooks = hooks;
    this.attack = null;
    this.step = 0; this.prev = 0; this.t0 = performance.now();
    this.playing = false; this.speed = 1; this.follow = true;
    this.view = new View3D(canvas, {
      onHover: (id, x, y) => tip.show(id ? this._tipText(id) : null, x, y),
      onClick: id => { if (id && id.startsWith('s:')) this.goto(Number(id.slice(2).split('|')[0])); },
      onUserMove: () => { if (this.follow) this.setFollow(false); },
    });
    this.view.cam = { theta: 0.42, phi: 0.18, dist: 40, target: [0, 0, 0] };
    this.view.setHome(this.view.cam);
    this.view.draw = v => this.draw(v);
    this._legend();
  }

  /* ------------------------------------------------------------------ control */
  setAttack(a) {
    this.attack = a;
    this.step = 0; this.prev = 0; this.t0 = performance.now();
    this._legend();
    this.view.flyTo({ theta: 0.42, phi: 0.18, dist: 40, target: [this.x(0), 0, 0] }, 900);
    this._homeDirty = true;           // the overview is fitted to the canvas at the next frame
    this.setFollow(true);
    this.render();
  }
  /* camera that shows the whole attack */
  _overviewCam() {
    const n = this.attack && this.attack.steps.length ? this.attack.steps.length : 1;
    const x0 = this.x(0) - 3.2, x1 = this.x(n - 1) + 3.2;
    const yLo = AY.pulse - 0.8, yHi = AY.techniques + 4.2;
    const pts = [];
    for (const x of [x0 - 3, x1]) for (const y of [yLo, yHi]) for (const z of [-1.9, 1.9]) pts.push([x, y, z]);
    const cam = { theta: 0.24, phi: 0.2, target: [(x0 - 3 + x1) / 2, (yLo + yHi) / 2 - 0.6, 0] };
    cam.dist = this.view.fitDist(pts, cam, { l: 24, r: 24, t: 46, b: 96 });
    return cam;
  }
  overview() {
    if (!this.attack) return;
    this.setFollow(false);
    this.view.setHome(this._overviewCam());
    this.view.reset(900);
  }
  x(i) { const n = this.attack ? this.attack.steps.length : 1; return (i - (n - 1) / 2) * DX; }
  goto(i) {
    if (!this.attack) return;
    const n = this.attack.steps.length;
    const j = Math.max(0, Math.min(n - 1, i));
    this.prev = this.step; this.step = j; this.t0 = performance.now();
    this.render();
  }
  next() { this.goto(this.step + 1); }
  previous() { this.goto(this.step - 1); }
  togglePlay() {
    if (!this.attack) return;
    if (!this.playing && this.step >= this.attack.steps.length - 1) this.goto(0);
    this.playing = !this.playing;
    this.t0 = performance.now();
    this.hooks.onPlay && this.hooks.onPlay(this.playing);
  }
  setFollow(on) { this.follow = on; this.hooks.onFollow && this.hooks.onFollow(on); }

  /* ------------------------------------------------------------------ drawing */
  draw(v) {
    const a = this.attack;
    if (!a || !a.steps.length) {
      LEVELS.forEach(lv => {
        const y = AY[lv];
        v.poly([[-14, y, -1.8], [14, y, -1.8], [14, y, 1.8], [-14, y, 1.8]], rgba(LEVEL_COLOR[lv], 0.06), rgba(LEVEL_COLOR[lv], 0.4));
        v.label([-15, y, 0], LEVEL_LABEL[lv], { color: LEVEL_COLOR[lv], weight: 700, align: 'right', bg: null, force: true });
      });
      return;
    }
    const n = a.steps.length;
    if (this._homeDirty && v.w > 200) { v.setHome(this._overviewCam()); this._homeDirty = false; }
    const el = (performance.now() - this.t0) / 1000 * this.speed;     // seconds since the step started
    if (this.playing && el > DWELL) {
      if (this.step < n - 1) this.goto(this.step + 1);
      else { this.playing = false; this.hooks.onPlay && this.hooks.onPlay(false); }
    }
    if (this.follow) {
      const tgt = [this.x(this.step), -0.5, 0];
      this.view.cam.target = V.lerp(this.view.cam.target, tgt, 0.06);
    }
    const x0 = this.x(0) - 3.2, x1 = this.x(n - 1) + 3.2;
    // the levels of abstraction
    for (const lv of LEVELS) {
      const y = AY[lv];
      v.poly([[x0, y, -1.9], [x1, y, -1.9], [x1, y, 1.9], [x0, y, 1.9]], rgba(LEVEL_COLOR[lv], 0.06), rgba(LEVEL_COLOR[lv], 0.45), { width: 1 });
      v.label([x0 - 0.4, y, 0], LEVEL_LABEL[lv], { color: LEVEL_COLOR[lv], weight: 700, align: 'right', bg: null, size: 12, force: true, priority: 3 });
    }
    a.steps.forEach((s, i) => this._drawStep(v, s, i, el));
    // the attacker moving from the previous step to the current one
    for (let i = 0; i < n - 1; i++) {
      const A = [this.x(i), AY.techniques, 0], B = [this.x(i + 1), AY.techniques, 0];
      const C = [(A[0] + B[0]) / 2, AY.techniques + 2.4, 0];
      const done = i < this.step;
      v.curve(A, C, B, done ? INK2 : AXIS, done ? 1.6 : 1, { alpha: done ? 0.85 : 0.35, arrow: done ? 8 : 6, dash: done ? null : [4, 5] });
    }
    if (this.step !== this.prev) {
      const t = ease(el / 0.8);
      const i0 = this.prev, i1 = this.step;
      if (t < 1) {
        const A = [this.x(i0), AY.techniques, 0], B = [this.x(i1), AY.techniques, 0];
        const C = [(A[0] + B[0]) / 2, AY.techniques + 2.4 + Math.abs(i1 - i0) * 0.4, 0];
        const u = 1 - t;
        const p = [u * u * A[0] + 2 * u * t * C[0] + t * t * B[0], u * u * A[1] + 2 * u * t * C[1] + t * t * B[1], 0];
        v.sphere(p, 0.42, '#ffd166', { glow: 1.4, noFog: true });
      }
    }
  }

  _drawStep(v, s, i, el) {
    const x = this.x(i);
    const cur = i === this.step, past = i < this.step, future = i > this.step;
    const fade = future ? 0.22 : past ? 0.75 : 1;
    const reveal = lv => {
      if (!cur) return future ? 0 : 1;
      const d = REVEAL.find(r => r[0] === lv)[1];
      return ease((el - 0.6 - d) / 0.35);
    };
    const P = lv => [x, AY[lv], 0];
    // technique
    const tt = cur ? Math.max(0.35, ease(el / 0.5)) : 1;
    v.sphere(P('techniques'), (cur ? 0.62 : 0.48) * tt, LEVEL_COLOR.techniques, { alpha: fade, glow: cur ? 1 : 0, pick: `s:${i}|techniques` });
    v.label(V.add(P('techniques'), [0, cur ? 3.6 : 3.2, 0]), `${i + 1}. ${s.tactic}${s.source === 'included' ? ' \u2605' : ''}`, { size: cur ? 12 : 10.5, weight: 700, color: cur ? INK : INK2,
      priority: cur ? 50 : 20, alpha: future ? 0.45 : 1, force: cur });
    if (!future) v.label(V.add(P('techniques'), [0, 1.25, 0]), cur ? `${s.technique.id}\n${trunc(s.technique.name, 34)}` : s.technique.id,
      { size: cur ? 11.5 : 10, priority: cur ? 49 : 18, force: cur, firstColor: INK, restColor: INK2 });
    if (future) return;
    // software used with the technique
    const sw = s.software.slice(0, 3);
    sw.forEach((w, k) => {
      const r = reveal('software');
      if (r <= 0) return;
      const pz = (k - (sw.length - 1) / 2) * 1.15;
      const p = [x + 0.0, AY.software, pz];
      v.line(P('techniques'), V.lerp(P('techniques'), p, r), LEVEL_COLOR.software, 1, { alpha: 0.55 * fade });
      v.sphere(p, 0.32 * r, LEVEL_COLOR.software, { alpha: fade, pick: `s:${i}|software|${k}` });
      if (cur && r >= 1) v.label(V.add(p, [0, 0.62, 0]), trunc(w.name, 18), { size: 10.5, priority: 40 - k });
    });
    // the attack path through the lower levels
    const c = s.chain;
    if (!c) {
      const r = reveal('capec');
      if (r > 0) {
        v.line(P('techniques'), V.lerp(P('techniques'), P('capec'), r), AXIS, 1, { dash: [3, 4], alpha: 0.6 * fade });
        if (cur && r >= 1) v.label(V.add(P('capec'), [0.75, 0, 0]), 'no CAPEC link for this technique', { size: 10.5, color: MUTED, priority: 30, align: 'left' });
      }
      return;
    }
    const chainLv = ['techniques', 'capec', 'cwe', 'cve', 'pulse'];
    for (let k = 1; k < chainLv.length; k++) {
      const lv = chainLv[k], r = reveal(lv);
      if (r <= 0) continue;
      const A = P(chainLv[k - 1]), B = P(lv);
      v.line(A, V.lerp(A, B, Math.min(1, r)), LEVEL_COLOR[lv], cur ? 2 : 1.2, { alpha: 0.8 * fade });
      const rad = lv === 'pulse' ? 0.22 + 0.12 * Math.log1p(c.pulses) : 0.42;
      v.sphere(B, rad * r, LEVEL_COLOR[lv], { alpha: fade, glow: cur && r >= 1 ? 0.5 : 0, pick: `s:${i}|${lv}` });
      if (cur && r < 1) v.sphere(V.lerp(A, B, r), 0.14, '#ffd166', { flat: true, noFog: true });
      if (cur && r >= 1) {
        const txt = lv === 'capec' ? `${c.capec.id}\n${trunc(c.capec.name, 30)}` : lv === 'cwe' ? `${c.cwe.id}\n${trunc(c.cwe.name, 30)}`
          : lv === 'cve' ? `${c.cve.id}${c.cve.exploits ? `\n${c.cve.exploits} public exploit(s)` : ''}` : `${c.pulses} OTX pulse${c.pulses === 1 ? '' : 's'}`;
        v.label(V.add(B, [0.75, 0, 0]), txt, { size: 10.5, align: 'left', priority: 45 - k, firstColor: INK, restColor: INK2, force: true });
      }
    }
  }

  /* ------------------------------------------------------------------ panels */
  _legend() {
    const box = $('#legend-attack');
    box.innerHTML = '';
    LEVELS.forEach(lv => box.append(h('div', {}, h('span', { class: 'sw', style: `background:${LEVEL_COLOR[lv]}` }), LEVEL_LABEL[lv])));
    box.append(h('div', {}, h('span', { class: 'sw', style: 'background:#ffd166' }), 'the attacker'));
    if (this.attack && this.attack.steps.some(s => s.source === 'included')) box.append(h('div', {}, h('span', { class: 'star' }, '\u2605'), 'step of an included item'));
  }

  render() {
    const a = this.attack;
    const list = $('#atk-steps');
    list.innerHTML = '';
    const info = $('#info-attack');
    info.innerHTML = '';
    const narr = $('#atk-narration');
    if (!a) return;
    if (!a.steps.length) {
      narr.textContent = a.constrained ? 'No step: tick at least one tactic, or include an item that can be placed.'
        : `${a.group} has no technique${a.platform !== 'any' ? ` on ${a.platform}` : ''} in the dataset, so no attack can be built from its profile.`;
      return;
    }
    a.steps.forEach((s, i) => list.append(h('li', { class: i === this.step ? 'cur' : i < this.step ? 'past' : 'future', onclick: () => this.goto(i) },
      h('span', { class: 'n' }, String(i + 1)),
      h('span', {}, h('div', { class: 'tac' }, s.tactic, s.source === 'included' ? h('span', { class: 'star', 'data-tip': `included: ${s.because.map(b => b.id).join(', ')}` }, ' \u2605') : null),
        h('div', { class: 't' }, `${s.technique.id} ${s.technique.name}`)))));
    const cur = list.querySelector('li.cur');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
    const s = a.steps[this.step];
    narr.innerHTML = `<b>Step ${this.step + 1} of ${a.steps.length} — ${s.tactic}${s.source === 'included' ? ' \u2605' : ''}:</b> ${s.technique.id} ${s.technique.name}.` +
      (s.how ? `<div class="step">${trunc(s.how, 210)}</div>` : '');
    // the step card
    const card = h('div', { class: 'card' },
      h('div', { class: 'tac' }, `Step ${this.step + 1} · ${s.tactic}`),
      h('div', { class: 'tech' }, `${s.technique.id} ${s.technique.name}`));
    const inc = s.source === 'included';
    const outside = inc && !s.in_profile;
    if (s.how) {
      card.append(h('div', { class: 'how' }, s.how),
        h('div', { class: 'src' }, s.how_source === 'group' ? `ATT&CK procedure example for ${a.group}`
          : s.how_own ? `ATT&CK procedure example for ${s.how_by}, software of ${a.group}` : `ATT&CK procedure example for ${s.how_by}, included software`));
    } else card.append(h('div', { class: 'src' }, outside ? `ATT&CK gives no procedure example of ${a.group} or of the software of this step for this technique.`
      : `ATT&CK gives no procedure example of ${a.group} or of its software for this technique: it is in the group's profile through its campaigns.`));
    if (inc) {
      card.append(h('div', { class: 'src inc', style: 'margin-top:6px' }, `\u2605 Included: ${s.because.map(b => `${b.id}${b.name && b.name !== b.id ? ` ${trunc(b.name, 40)}` : ''}`).join('; ')}.`));
      if (outside) card.append(h('div', { class: 'src' }, `This technique is not in ${a.group}'s profile.`));
    } else card.append(h('div', { class: 'src', style: 'margin-top:6px' },
      s.candidates === 1 ? 'The only technique of the profile for this tactic.' : `Drawn at random among the ${s.candidates} techniques of the profile for this tactic, each equally likely.`));
    info.append(card);
    const chain = h('div', { class: 'chain' });
    const row = (lv, txt) => chain.append(h('span', { class: 'lv' }, h('span', { class: 'sw', style: `background:${LEVEL_COLOR[lv]}` }), LEVEL_LABEL[lv]), h('span', {}, txt));
    row('software', s.software.length ? s.software.map(w => `${w.name} (${w.type || 'software'}${w.own === false ? `, not in ${a.group}'s profile` : ''})`).join(', ') + (s.n_software > s.software.length ? ` +${s.n_software - s.software.length}` : '') : 'none recorded with this technique');
    if (s.chain) {
      row('capec', `${s.chain.capec.id} ${s.chain.capec.name}${s.chain.capec.severity && s.chain.capec.severity !== 'not stated' ? ` · severity ${s.chain.capec.severity}` : ''}`);
      row('cwe', `${s.chain.cwe.id} ${s.chain.cwe.name}`);
      row('cve', `${s.chain.cve.id}${s.chain.cve.exploits ? ` · ${s.chain.cve.exploits} public exploit(s) recorded` : ''}`);
      row('pulse', `${s.chain.pulses} OTX pulse${s.chain.pulses === 1 ? '' : 's'} report this CVE${s.chain.pulse_ids.length ? ` (${s.chain.pulse_ids.slice(0, 2).join(', ')}…)` : ''}`);
    } else row('capec', 'no link from this technique to an attack pattern');
    info.append(h('h3', {}, 'Down the levels'), chain);
    if (s.chain && s.chain.own_records !== false) info.append(h('p', { class: 'small' }, `One of ${s.n_chains} paths of ${a.group}'s records from this technique (${s.n_capec} CAPEC, ${s.n_cwe} CWE, ${s.n_cve} CVE). The CVE is associated through the weakness, not reported as exploited by the group.`));
    else if (s.chain) info.append(h('p', { class: 'small' }, `This path is not in ${a.group}'s records: it comes from the records of other groups of the dataset. The CVE is associated through the weakness, not reported as exploited by the group.`));
    info.append(h('h3', {}, 'Mitigate'), h('div', { class: 'tagrow' }, ...(s.mitigations.length ? s.mitigations.map(m => h('span', { class: 'tag', 'data-tip': m.id }, `${m.id} ${m.name}`)) : [h('span', { class: 'small' }, 'ATT&CK lists no preventive mitigation for this technique.')])));
    info.append(h('h3', {}, 'Detect'), h('div', { class: 'tagrow' }, ...(s.detections.length ? s.detections.slice(0, 8).map(d => h('span', { class: 'tag' }, d.name)) : [h('span', { class: 'small' }, 'No detection listed.')])));
    // defence plan for the whole attack
    info.append(h('h3', {}, 'Defence plan for this attack'));
    const plan = (title, p) => {
      info.append(h('div', { class: 'small' }, title));
      p.chosen.forEach(c => info.append(h('div', { class: 'hbar' }, h('span', { class: 'nm' }, `${c.id} ${c.name}`),
        h('span', { class: 'track', 'data-tip': `covers step ${c.steps.map(x => x + 1).join(', ')}` }, h('i', { style: `left:0;width:${100 * c.steps.length / a.steps.length}%;background:${ACCENT}` })))));
      if (p.uncovered.length) info.append(h('div', { class: 'small' }, `Not covered: step ${p.uncovered.map(x => x + 1).join(', ')}`));
    };
    plan('Mitigations that cover most steps', a.plan.mitigations);
    plan('Detections (ATT&CK data sources) that cover most steps', a.plan.detections);
  }

  _tipText(id) {
    const [i, lv, k] = id.slice(2).split('|');
    const s = this.attack && this.attack.steps[Number(i)];
    if (!s) return '';
    if (lv === 'techniques') return `${s.tactic}\n${s.technique.id} ${s.technique.name}`;
    if (lv === 'software') { const w = s.software[Number(k)]; return w ? `${w.id} ${w.name} (${w.type})` : ''; }
    const c = s.chain;
    if (!c) return '';
    if (lv === 'capec') return `${c.capec.id} ${c.capec.name}\nseverity ${c.capec.severity}, likelihood ${c.capec.likelihood}`;
    if (lv === 'cwe') return `${c.cwe.id} ${c.cwe.name}`;
    if (lv === 'cve') return `${c.cve.id}${c.cve.exploits ? `\n${c.cve.exploits} public exploit(s) recorded` : ''}`;
    if (lv === 'pulse') return `${c.pulses} AlienVault OTX pulse(s) report ${c.cve.id}`;
    return '';
  }
}
