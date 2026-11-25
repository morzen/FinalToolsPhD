/* engine3d.js — a small 3-D renderer on a 2-D canvas, written for the studio (no library, works offline).
   Orbit camera with perspective, depth-sorted spheres / lines / curves / polygons, labels with collision
   avoidance, hover picking, smooth camera moves.  Scenes are rebuilt every frame by a draw callback. */

export const V = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, k) => [a[0] * k, a[1] * k, a[2] * k],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: a => Math.hypot(a[0], a[1], a[2]),
  norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};
export const ease = t => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
export function bezier(a, c, b, t) {           // quadratic Bézier point
  const u = 1 - t;
  return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1],
    u * u * a[2] + 2 * u * t * c[2] + t * t * b[2]];
}
export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export function rgba(hex, a) { const [r, g, b] = hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; }
function shade(hex, k) {                        // k > 0 lighter, k < 0 darker
  const [r, g, b] = hexToRgb(hex);
  const f = c => Math.round(k >= 0 ? c + (255 - c) * k : c * (1 + k));
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}

export class View3D {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.bg = opts.bg || '#1a1a19';
    this.ink = opts.ink || '#f0efec';
    this.font = opts.font || 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
    this.fov = 0.9;
    this.cam = { theta: 0.6, phi: 0.35, dist: 46, target: [0, 0, 0] };
    this.home = { ...this.cam, target: [...this.cam.target] };
    this.fly = null;
    this.items = [];
    this.labels = [];
    this.picks = [];
    this.hover = null;
    this.onHover = opts.onHover || (() => {});
    this.onClick = opts.onClick || (() => {});
    this.onUserMove = opts.onUserMove || (() => {});
    this.draw = () => {};
    this.autoRotate = 0;
    this.time = 0;
    this._drag = null;
    this._bind();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.active = true;
    const loop = t => {
      this.time = t / 1000;
      if (this.active && !document.hidden) this.frame();
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(10, r.width); this.h = Math.max(10, r.height);
    this.canvas.width = Math.round(this.w * dpr); this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  setHome(cam) { this.home = { ...this.cam, ...cam, target: [...(cam.target || this.cam.target)] }; }
  /* smallest camera distance at which every point is inside the canvas, keeping margins in pixels {l, r, t, b} */
  fitDist(points, cam, m = { l: 40, r: 40, t: 40, b: 40 }) {
    const save = { ...this.cam, target: [...this.cam.target] };
    const fits = d => {
      this.cam = { ...save, ...cam, target: [...(cam.target || save.target)], dist: d };
      this._basis();
      return points.every(p => { const q = this.project(p); return q && q.x >= m.l && q.x <= this.w - m.r && q.y >= m.t && q.y <= this.h - m.b; });
    };
    let lo = 5, hi = 600;
    for (let k = 0; k < 32; k++) { const mid = (lo + hi) / 2; if (fits(mid)) hi = mid; else lo = mid; }
    this.cam = save; this._basis();
    return hi;
  }
  reset(ms = 700) { this.flyTo(this.home, ms); }
  flyTo(to, ms = 700) {
    const from = { ...this.cam, target: [...this.cam.target] };
    const dest = { ...this.cam, ...to, target: to.target ? [...to.target] : [...this.cam.target] };
    this.fly = { from, to: dest, t0: performance.now(), ms };
  }

  /* ---- camera basis */
  _basis() {
    const { theta, phi, dist, target } = this.cam;
    const eye = [target[0] + dist * Math.cos(phi) * Math.sin(theta), target[1] + dist * Math.sin(phi),
      target[2] + dist * Math.cos(phi) * Math.cos(theta)];
    const fwd = V.norm(V.sub(target, eye));
    let right = V.cross(fwd, [0, 1, 0]);
    if (V.len(right) < 1e-6) right = [1, 0, 0];
    right = V.norm(right);
    const up = V.cross(right, fwd);
    this._eye = eye; this._fwd = fwd; this._right = right; this._up = up;
    this._f = (this.h / 2) / Math.tan(this.fov / 2);
  }
  project(p) {
    const d = V.sub(p, this._eye);
    const z = V.dot(d, this._fwd);
    if (z < 0.5) return null;
    const x = V.dot(d, this._right), y = V.dot(d, this._up);
    return { x: this.w / 2 + this._f * x / z, y: this.h / 2 - this._f * y / z, z, s: this._f / z };
  }
  fog(z) { const n = this.cam.dist * 0.55, f = this.cam.dist * 1.9; return 1 - 0.55 * Math.min(1, Math.max(0, (z - n) / (f - n))); }

  /* ---- primitives (called from the draw callback) */
  sphere(p, r, color, o = {}) { this.items.push({ k: 's', p, r, color, o }); }
  line(a, b, color, width = 1, o = {}) { this.items.push({ k: 'l', pts: [a, b], color, width, o }); }
  polyline(pts, color, width = 1, o = {}) { this.items.push({ k: 'l', pts, color, width, o }); }
  curve(a, c, b, color, width = 1, o = {}) {
    const n = o.segments || 18, upto = o.progress === undefined ? 1 : Math.max(0, Math.min(1, o.progress));
    const pts = [];
    for (let i = 0; i <= n; i++) { const t = (i / n) * upto; pts.push(bezier(a, c, b, t)); }
    this.items.push({ k: 'l', pts, color, width, o });
  }
  poly(pts, fill, stroke, o = {}) { this.items.push({ k: 'p', pts, fill, stroke, o }); }
  label(p, text, o = {}) { this.labels.push({ p, text, o }); }
  disc(center, radius, fill, stroke, o = {}) {
    const n = o.segments || 64, pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; pts.push([center[0] + radius * Math.cos(a), center[1], center[2] + radius * Math.sin(a)]); }
    this.poly(pts, fill, stroke, o);
  }
  cone(base, height, radius, color, o = {}) { this.items.push({ k: 'c', p: base, h: height, r: radius, color, o }); }

  /* ---- frame */
  frame() {
    if (this.fly) {
      const t = ease((performance.now() - this.fly.t0) / this.fly.ms);
      const { from, to } = this.fly;
      let dth = to.theta - from.theta;
      while (dth > Math.PI) dth -= 2 * Math.PI;
      while (dth < -Math.PI) dth += 2 * Math.PI;
      this.cam.theta = from.theta + dth * t;
      this.cam.phi = from.phi + (to.phi - from.phi) * t;
      this.cam.dist = from.dist + (to.dist - from.dist) * t;
      this.cam.target = V.lerp(from.target, to.target, t);
      if (t >= 1) this.fly = null;
    } else if (this.autoRotate && !this._drag) {
      this.cam.theta += this.autoRotate / 60;
    }
    this._basis();
    this.items = []; this.labels = []; this.picks = [];
    this.draw(this);
    this.render();
  }

  render() {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = this.bg;
    ctx.fillRect(0, 0, this.w, this.h);
    const list = [];
    for (const it of this.items) {
      if (it.k === 's') {
        const q = this.project(it.p);
        if (!q) continue;
        list.push({ it, z: q.z + (it.o.zbias || 0), q });
      } else if (it.k === 'l') {
        const qs = it.pts.map(p => this.project(p));
        if (qs.some(q => !q)) continue;
        const z = qs.reduce((a, q) => a + q.z, 0) / qs.length;
        list.push({ it, z: z + (it.o.zbias || 0), qs });
      } else if (it.k === 'p') {
        const qs = it.pts.map(p => this.project(p));
        if (qs.some(q => !q)) continue;
        const z = qs.reduce((a, q) => a + q.z, 0) / qs.length;
        list.push({ it, z: z + (it.o.zbias === undefined ? 4 : it.o.zbias), qs });
      } else if (it.k === 'c') {
        const qb = this.project(it.p), qt = this.project(V.add(it.p, [0, it.h, 0]));
        if (!qb || !qt) continue;
        list.push({ it, z: qb.z, qb, qt });
      }
    }
    list.sort((a, b) => b.z - a.z);
    for (const d of list) this._drawItem(ctx, d);
    this._drawLabels(ctx);
    ctx.restore();
  }

  _drawItem(ctx, d) {
    const { it } = d;
    const o = it.o;
    const fogA = o.noFog ? 1 : this.fog(d.z);
    if (it.k === 's') {
      const { q } = d;
      const r = Math.max(o.minPx || 1.2, it.r * q.s);
      const a = (o.alpha === undefined ? 1 : o.alpha) * fogA;
      if (r > 600) return;
      ctx.globalAlpha = a;
      if (o.glow) {
        const g = ctx.createRadialGradient(q.x, q.y, r * 0.6, q.x, q.y, r * (2.4 + o.glow));
        g.addColorStop(0, rgba(o.glowColor || it.color, 0.45 * Math.min(1, o.glow)));
        g.addColorStop(1, rgba(o.glowColor || it.color, 0));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(q.x, q.y, r * (2.4 + o.glow), 0, Math.PI * 2); ctx.fill();
      }
      if (r > 3.2 && !o.flat) {
        const g = ctx.createRadialGradient(q.x - r * 0.35, q.y - r * 0.4, r * 0.1, q.x, q.y, r);
        g.addColorStop(0, shade(it.color, 0.55));
        g.addColorStop(0.55, it.color);
        g.addColorStop(1, shade(it.color, -0.35));
        ctx.fillStyle = g;
      } else ctx.fillStyle = it.color;
      ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, Math.PI * 2); ctx.fill();
      if (o.ring) {
        ctx.strokeStyle = o.ring; ctx.lineWidth = o.ringWidth || 2;
        ctx.beginPath(); ctx.arc(q.x, q.y, r + 3, 0, Math.PI * 2); ctx.stroke();
      }
      if (o.pick !== undefined) this.picks.push({ id: o.pick, x: q.x, y: q.y, r: Math.max(r, 5), z: q.z });
    } else if (it.k === 'l') {
      const { qs } = d;
      ctx.globalAlpha = (o.alpha === undefined ? 1 : o.alpha) * fogA;
      ctx.strokeStyle = it.color;
      ctx.lineWidth = o.perspective ? Math.max(0.5, it.width * qs[0].s * 0.06) : it.width;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      if (o.dash) ctx.setLineDash(o.dash); else ctx.setLineDash([]);
      ctx.beginPath();
      qs.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.stroke();
      ctx.setLineDash([]);
      if (o.arrow && qs.length >= 2) {
        const a = qs[qs.length - 2], b = qs[qs.length - 1];
        const ang = Math.atan2(b.y - a.y, b.x - a.x), L = o.arrow;
        ctx.fillStyle = it.color;
        ctx.beginPath();
        ctx.moveTo(b.x, b.y);
        ctx.lineTo(b.x - L * Math.cos(ang - 0.4), b.y - L * Math.sin(ang - 0.4));
        ctx.lineTo(b.x - L * Math.cos(ang + 0.4), b.y - L * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
      }
    } else if (it.k === 'p') {
      const { qs } = d;
      ctx.globalAlpha = (o.alpha === undefined ? 1 : o.alpha) * (o.noFog ? 1 : Math.max(0.55, fogA));
      ctx.beginPath();
      qs.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.closePath();
      if (it.fill) { ctx.fillStyle = it.fill; ctx.fill(); }
      if (it.stroke) { ctx.strokeStyle = it.stroke; ctx.lineWidth = o.width || 1; ctx.stroke(); }
    } else if (it.k === 'c') {
      const { qb, qt } = d;
      const r = Math.max(1.5, it.r * qb.s);
      ctx.globalAlpha = (o.alpha === undefined ? 1 : o.alpha) * fogA;
      ctx.fillStyle = it.color;
      ctx.beginPath();
      ctx.moveTo(qt.x, qt.y); ctx.lineTo(qb.x - r, qb.y); ctx.lineTo(qb.x + r, qb.y); ctx.closePath(); ctx.fill();
      if (o.ring) { ctx.strokeStyle = o.ring; ctx.lineWidth = 1.2; ctx.stroke(); }
      if (o.pick !== undefined) this.picks.push({ id: o.pick, x: (qb.x + qt.x) / 2, y: (qb.y + qt.y) / 2, r: Math.max(6, r), z: qb.z });
    }
    ctx.globalAlpha = 1;
  }

  _drawLabels(ctx) {
    const placed = [];
    const ls = this.labels.map(l => ({ ...l, q: this.project(l.p) })).filter(l => l.q);
    ls.sort((a, b) => (b.o.priority || 0) - (a.o.priority || 0) || a.q.z - b.q.z);
    for (const l of ls) {
      const o = l.o;
      const size = o.size || 11;
      ctx.font = `${o.weight || 500} ${size}px ${this.font}`;
      const lines = String(l.text).split('\n');
      const w = Math.max(...lines.map(t => ctx.measureText(t).width)) + (o.pad === undefined ? 8 : o.pad * 2);
      const hgt = lines.length * size * 1.25 + (o.pad === undefined ? 4 : o.pad);
      let x = l.q.x + (o.dx || 0), y = l.q.y + (o.dy || 0);
      const align = o.align || 'center';
      let x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
      const y0 = y - hgt / 2;
      const box = [x0, y0, x0 + w, y0 + hgt];
      if (!o.force && placed.some(b => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]))) continue;
      placed.push(box);
      ctx.globalAlpha = (o.alpha === undefined ? 1 : o.alpha) * (o.noFog ? 1 : Math.max(0.6, this.fog(l.q.z)));
      if (o.bg !== null) {
        ctx.fillStyle = o.bg || 'rgba(20,20,19,0.78)';
        const rr = 4;
        ctx.beginPath();
        ctx.moveTo(x0 + rr, y0); ctx.arcTo(box[2], y0, box[2], box[3], rr); ctx.arcTo(box[2], box[3], x0, box[3], rr);
        ctx.arcTo(x0, box[3], x0, y0, rr); ctx.arcTo(x0, y0, box[2], y0, rr); ctx.closePath(); ctx.fill();
        if (o.border) { ctx.strokeStyle = o.border; ctx.lineWidth = 1; ctx.stroke(); }
      }
      ctx.fillStyle = o.color || this.ink;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      const px = x0 + (o.pad === undefined ? 4 : o.pad);
      lines.forEach((t, i) => {
        if (i === 0 && o.firstColor) ctx.fillStyle = o.firstColor;
        else if (i > 0 && o.restColor) ctx.fillStyle = o.restColor;
        ctx.fillText(t, px, y0 + (o.pad === undefined ? 2 : o.pad / 2) + size * 1.25 * (i + 0.5));
      });
      ctx.globalAlpha = 1;
    }
  }

  /* ---- interaction */
  _bind() {
    const c = this.canvas;
    c.addEventListener('contextmenu', e => e.preventDefault());
    c.addEventListener('pointerdown', e => {
      c.setPointerCapture(e.pointerId);
      this._drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey, moved: 0 };
      this.fly = null;
    });
    c.addEventListener('pointermove', e => {
      const r = c.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      if (this._drag) {
        const dx = e.clientX - this._drag.x, dy = e.clientY - this._drag.y;
        this._drag.x = e.clientX; this._drag.y = e.clientY; this._drag.moved += Math.abs(dx) + Math.abs(dy);
        if (this._drag.btn === 2 || this._drag.shift) {
          const k = this.cam.dist / this._f;
          this.cam.target = V.add(this.cam.target, V.add(V.mul(this._right, -dx * k), V.mul(this._up, dy * k)));
        } else {
          this.cam.theta -= dx * 0.006;
          this.cam.phi = Math.max(-1.45, Math.min(1.45, this.cam.phi + dy * 0.006));
        }
        if (this._drag.moved > 3) this.onUserMove();
        return;
      }
      this._pick(mx, my, e.clientX, e.clientY);
    });
    c.addEventListener('pointerup', e => {
      const d = this._drag; this._drag = null;
      if (d && d.moved < 4) {
        const r = c.getBoundingClientRect();
        const hit = this._hit(e.clientX - r.left, e.clientY - r.top);
        this.onClick(hit ? hit.id : null);
      }
    });
    c.addEventListener('pointerleave', () => { if (this.hover) { this.hover = null; this.onHover(null); } });
    c.addEventListener('wheel', e => {
      e.preventDefault();
      const k = Math.exp((e.ctrlKey ? e.deltaY * 3 : e.deltaY) * 0.0012);
      this.cam.dist = Math.max(6, Math.min(160, this.cam.dist * k));
      this.fly = null;
      this.onUserMove();
    }, { passive: false });
    c.addEventListener('dblclick', () => this.reset());
  }
  _hit(mx, my) {
    let best = null;
    for (const p of this.picks) {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d <= p.r + 4 && (!best || p.z < best.z)) best = { ...p, d };
    }
    return best;
  }
  _pick(mx, my, cx, cy) {
    const hit = this._hit(mx, my);
    const id = hit ? hit.id : null;
    if (id !== (this.hover && this.hover.id)) { this.hover = hit ? { id } : null; }
    this.onHover(id, cx, cy);
    this.canvas.style.cursor = id ? 'pointer' : (this._drag ? 'grabbing' : 'grab');
  }
}
