/* picker.js — type-to-search field for the groups. Matches the name (APT29, apt 29), the ATT&CK ID (G0016) or an
   alias (Cozy Bear), lists the best matches under the field and only ever commits a real group (or none). */
import { h } from './common.js';

const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');

export class GroupPicker {
  /* input: the <input>; index(): [{name, gid, attack, aliases}]; onPick(name) when the chosen group changes;
     allowEmpty: whether "no group" is a valid choice */
  constructor(input, { index, onPick = () => {}, allowEmpty = true, max = 12 } = {}) {
    Object.assign(this, { input, index, onPick, allowEmpty, max });
    this.value = '';
    this.matches = [];
    this.hi = -1;
    for (const [k, v] of Object.entries({ autocomplete: 'off', spellcheck: 'false', role: 'combobox', 'aria-autocomplete': 'list', 'aria-expanded': 'false' })) input.setAttribute(k, v);
    this.list = h('ul', { class: 'picker-list', role: 'listbox', hidden: true });
    document.body.append(this.list);
    input.addEventListener('input', () => this._update());
    input.addEventListener('focus', () => { this._fresh = true; input.select(); });   // typing replaces the group shown
    input.addEventListener('mouseup', e => { if (this._fresh) { e.preventDefault(); this._fresh = false; } });
    input.addEventListener('keydown', e => this._key(e));
    input.addEventListener('blur', () => this._blur());
    input.addEventListener('search', () => { if (input.value === '' && this.allowEmpty) this.commit(''); });   // the field's own clear button
    this.list.addEventListener('mousedown', e => {
      e.preventDefault();                                // keep the focus in the field
      const li = e.target.closest('li[data-i]');
      if (li) this.commit(this.matches[Number(li.dataset.i)].name);
    });
    const follow = () => { if (!this.list.hidden) this._place(); };
    window.addEventListener('resize', follow);
    window.addEventListener('scroll', follow, true);
  }

  set(name) { this.value = name || ''; this.input.value = this.value; this._close(); }
  focus() { this.input.focus(); }
  commit(name) {
    const changed = name !== this.value;
    this.set(name);
    if (changed) this.onPick(this.value);
  }

  /* groups matching the text, best first: exact name or ID, name or ID starting with it, exact alias, alias
     starting with it, name containing it, alias containing it */
  find(text) {
    const q = String(text || '').trim().toLowerCase(), qn = norm(q);
    if (!q) return [];
    const out = [];
    for (const g of this.index()) {
      const n = g.name.toLowerCase(), nn = norm(g.name), id = (g.gid || '').toLowerCase();
      let score = -1, alias = '';
      if (n === q || (qn && nn === qn) || id === q) score = 0;
      else if (n.startsWith(q) || (qn && nn.startsWith(qn)) || (id && id.startsWith(q))) score = 1;
      else {
        for (const a of g.aliases || []) {
          const al = a.toLowerCase(), an = norm(a);
          const s = (al === q || (qn && an === qn)) ? 2 : (al.startsWith(q) || (qn && an.startsWith(qn))) ? 3
            : (al.includes(q) || (qn && an.includes(qn))) ? 5 : -1;
          if (s >= 0 && (score < 0 || s < score)) { score = s; alias = a; }
        }
        if ((score < 0 || score > 4) && (n.includes(q) || (qn && nn.includes(qn)))) { score = 4; alias = ''; }
      }
      if (score >= 0) out.push({ name: g.name, gid: g.gid, attack: g.attack, score, alias });
    }
    out.sort((a, b) => a.score - b.score || a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
    return out.slice(0, this.max);
  }
  resolve(text) {                                        // the group the text names exactly, or null
    const m = this.find(text);
    return m.length && m[0].score <= 2 ? m[0].name : null;
  }

  _update() {
    if (!this.input.value.trim()) { this._close(); return; }
    this.matches = this.find(this.input.value);
    this.hi = this.matches.length ? 0 : -1;
    this._render();
  }
  _render() {
    const L = this.list;
    L.innerHTML = '';
    if (!this.matches.length) L.append(h('li', { class: 'empty' }, 'No group matches'));
    this.matches.forEach((m, i) => L.append(h('li', { 'data-i': i, role: 'option', class: i === this.hi ? 'hi' : null, 'aria-selected': i === this.hi ? 'true' : 'false' },
      h('span', { class: 'n' }, m.name),
      h('span', { class: 'm' }, m.alias ? `aka ${m.alias}` : m.gid || (m.attack === false ? 'AlienVault OTX entry' : '')))));
    L.hidden = false;
    this.input.setAttribute('aria-expanded', 'true');
    this._place();
    const cur = L.querySelector('li.hi');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  }
  _place() {
    const r = this.input.getBoundingClientRect(), L = this.list;
    L.style.left = `${Math.max(4, r.left)}px`;
    L.style.minWidth = `${r.width}px`;
    const below = window.innerHeight - r.bottom - 10, above = r.top - 10;
    if (below >= 180 || below >= above) { L.style.top = `${r.bottom + 4}px`; L.style.bottom = 'auto'; L.style.maxHeight = `${Math.min(320, below)}px`; }
    else { L.style.top = 'auto'; L.style.bottom = `${window.innerHeight - r.top + 4}px`; L.style.maxHeight = `${Math.min(320, above)}px`; }
  }
  _close() { this.list.hidden = true; this.hi = -1; this.input.setAttribute('aria-expanded', 'false'); }
  _key(e) {
    const open = !this.list.hidden;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!open) { this._update(); return; }
      if (!this.matches.length) return;
      const d = e.key === 'ArrowDown' ? 1 : -1;
      this.hi = (this.hi + d + this.matches.length) % this.matches.length;
      this._render();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (open && this.hi >= 0) this.commit(this.matches[this.hi].name);
      else {
        const g = this.resolve(this.input.value);
        if (g !== null) this.commit(g);
        else if (!this.input.value.trim() && this.allowEmpty) this.commit('');
      }
    } else if (e.key === 'Escape') {
      if (open || this.input.value !== this.value) { e.preventDefault(); e.stopPropagation(); this.set(this.value); }
    }
  }
  _blur() {
    if (!this.list.hidden && this.hi >= 0) { this.commit(this.matches[this.hi].name); return; }
    const t = this.input.value.trim();
    if (!t) { if (this.allowEmpty) this.commit(''); else this.set(this.value); return; }
    const g = this.resolve(t);
    if (g !== null) this.commit(g); else this.set(this.value);           // not a group: back to the last choice
  }
}
