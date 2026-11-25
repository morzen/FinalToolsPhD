/* common.js — shared constants and helpers of the studio. */
export const LEVELS = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse'];
// dark-surface steps of the validated categorical palette (checked with the dataviz validator on #1a1a19)
export const LEVEL_COLOR = { techniques: '#3987e5', software: '#d95926', capec: '#199e70', cwe: '#c98500', cve: '#d55181', pulse: '#008300' };
export const LEVEL_LABEL = { techniques: 'TTPs', software: 'Software', capec: 'CAPEC', cwe: 'CWE', cve: 'CVE', pulse: 'Pulse' };
export const LEVEL_Y = { techniques: 7.5, software: 4.5, capec: 1.5, cwe: -1.5, cve: -4.5, pulse: -7.5 };
export const INK = '#f0efec', INK2 = '#b5b3ab', MUTED = '#8c8a83', GRID = '#33332f', AXIS = '#4a4945', ACCENT = '#3987e5';

export const $ = (q, r = document) => r.querySelector(q);
export function h(tag, attrs = {}, ...kids) {
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
export const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);
export const fmtRank = r => (r === null || r === undefined) ? '–' : (Number.isInteger(r) ? String(r) : Number(r).toFixed(1));
export const pct = x => `${(100 * x).toFixed(1)}%`;
export function trunc(t, n) { t = String(t || ''); return t.length > n ? t.slice(0, n - 1) + '…' : t; }
export async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
export function hash01(s) {                      // stable pseudo-random number in [0, 1) from a string
  let h1 = 2166136261;
  for (let i = 0; i < s.length; i++) { h1 ^= s.charCodeAt(i); h1 = Math.imul(h1, 16777619); }
  return ((h1 >>> 0) % 100000) / 100000;
}

/* tooltip */
export const tip = {
  el: null,
  show(text, x, y) {
    if (!this.el) this.el = $('#tip');
    if (!text) { this.el.hidden = true; return; }
    this.el.textContent = text;
    this.el.hidden = false;
    const w = this.el.offsetWidth, hh = this.el.offsetHeight;
    let px = x + 14, py = y + 14;
    if (px + w > window.innerWidth - 8) px = x - w - 14;
    if (py + hh > window.innerHeight - 8) py = y - hh - 14;
    this.el.style.left = `${px}px`; this.el.style.top = `${py}px`;
  },
};
document.addEventListener('mousemove', e => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (t) tip.show(t.getAttribute('data-tip'), e.clientX, e.clientY);
  else if (e.target.tagName !== 'CANVAS') tip.show(null);
});
