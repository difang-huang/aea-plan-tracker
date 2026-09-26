/* Shared helpers: theme, formatting, tooltip, and two hand-rolled SVG charts.
   No chart library — the site must work offline from a static host. */

const SVGNS = "http://www.w3.org/2000/svg";

/* ---- theme ------------------------------------------------------------ */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* private mode */ }
  if (saved === "dark" || saved === "light") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const isDark = cur ? cur === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    document.dispatchEvent(new CustomEvent("themechange"));
  });
}

/* ---- formatting ------------------------------------------------------- */
const nf = new Intl.NumberFormat("en-US");
const num = (v) => (v === null || v === undefined || v === "") ? "—" : nf.format(v);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v === null || v === undefined) ? "—" : `${(+v).toFixed(0)}`;

const STAGE_COLORS = {
  registered: "var(--ord-1)", in_progress: "var(--ord-1)",
  completed_no_paper: "var(--ord-2)", candidate_paper: "var(--ord-3)",
  working_paper: "var(--ord-4)", published: "var(--ord-5)",
};
const ORD = ["var(--ord-1)", "var(--ord-2)", "var(--ord-3)", "var(--ord-4)", "var(--ord-5)"];

/* ---- tooltip ---------------------------------------------------------- */
let tipEl = null;
function tip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "tip";
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.style.opacity = "1";
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  t.style.left = Math.max(8, x) + "px";
  t.style.top = Math.max(8, y) + "px";
}
function hideTip() { if (tipEl) tipEl.style.opacity = "0"; }

/* ---- svg helpers ------------------------------------------------------ */
function el(name, attrs, parent) {
  const n = document.createElementNS(SVGNS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function niceMax(v) {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const step = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
    .find((s) => v / (s * mag) <= 1) || 10;
  return step * mag;
}

/* ---- bar chart (single series) ---------------------------------------- */
function barChart(host, data, opts = {}) {
  const { xKey = "x", yKey = "y", tipFn, color = "var(--series-1)",
          height = 190, labelEvery = 1, yTicks = 4 } = opts;
  host.innerHTML = "";
  if (!data.length) { host.innerHTML = '<div class="empty">No data</div>'; return; }

  const W = Math.max(host.clientWidth || 520, 320);
  const M = { t: 12, r: 10, b: 26, l: 42 };
  const iw = W - M.l - M.r, ih = height - M.t - M.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${height}`, width: "100%",
                          height, role: "img" }, host);

  const max = niceMax(Math.max(...data.map((d) => +d[yKey] || 0)));
  const y = (v) => M.t + ih - (v / max) * ih;
  const bw = iw / data.length;

  for (let i = 0; i <= yTicks; i++) {
    const v = (max / yTicks) * i;
    el("line", { class: "gridline", x1: M.l, x2: M.l + iw, y1: y(v), y2: y(v) }, svg);
    const tx = el("text", { class: "axis-label", x: M.l - 7, y: y(v) + 3,
                            "text-anchor": "end" }, svg);
    tx.textContent = v >= 1000 ? (v / 1000) + "k" : String(v);
  }
  el("line", { class: "axisline", x1: M.l, x2: M.l + iw,
               y1: M.t + ih, y2: M.t + ih }, svg);

  data.forEach((d, i) => {
    const v = +d[yKey] || 0;
    const h = Math.max(v > 0 ? 2 : 0, M.t + ih - y(v));
    const x = M.l + i * bw + 1;         /* 2px surface gap between bars */
    const w = Math.max(1, bw - 2);
    if (h > 0) {
      const r = el("rect", { x, y: M.t + ih - h, width: w, height: h,
                             rx: Math.min(4, w / 2), fill: d.color || color }, svg);
      r.addEventListener("mousemove", (ev) => showTip(
        tipFn ? tipFn(d) : `<b>${esc(d[xKey])}</b><br>${num(v)}`, ev));
      r.addEventListener("mouseleave", hideTip);
    }
    if (i % labelEvery === 0) {
      const t = el("text", { class: "axis-label", x: M.l + i * bw + bw / 2,
                             y: height - 8, "text-anchor": "middle" }, svg);
      t.textContent = d[xKey];
    }
  });
  return svg;
}

/* ---- line chart (single series, crosshair) ---------------------------- */
function lineChart(host, data, opts = {}) {
  const { xKey = "x", yKey = "y", tipFn, color = "var(--series-1)",
          height = 190, suffix = "", labelEvery = 1 } = opts;
  host.innerHTML = "";
  if (data.length < 2) { host.innerHTML = '<div class="empty">Not enough data</div>'; return; }

  const W = Math.max(host.clientWidth || 520, 320);
  const M = { t: 12, r: 12, b: 26, l: 42 };
  const iw = W - M.l - M.r, ih = height - M.t - M.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${height}`, width: "100%", height }, host);

  const max = niceMax(Math.max(...data.map((d) => +d[yKey] || 0)));
  const x = (i) => M.l + (i / (data.length - 1)) * iw;
  const y = (v) => M.t + ih - (v / max) * ih;

  for (let i = 0; i <= 4; i++) {
    const v = (max / 4) * i;
    el("line", { class: "gridline", x1: M.l, x2: M.l + iw, y1: y(v), y2: y(v) }, svg);
    const t = el("text", { class: "axis-label", x: M.l - 7, y: y(v) + 3,
                           "text-anchor": "end" }, svg);
    t.textContent = (Math.round(v * 10) / 10) + suffix;
  }
  el("line", { class: "axisline", x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih }, svg);

  const pts = data.map((d, i) => [x(i), y(+d[yKey] || 0)]);
  el("path", {
    d: `M${M.l},${M.t + ih} ` + pts.map((p) => `L${p[0]},${p[1]}`).join(" ") +
       ` L${M.l + iw},${M.t + ih} Z`,
    fill: color, opacity: .12,
  }, svg);
  el("path", { d: "M" + pts.map((p) => `${p[0]},${p[1]}`).join(" L"),
               fill: "none", stroke: color, "stroke-width": 2,
               "stroke-linejoin": "round" }, svg);

  data.forEach((d, i) => {
    if (i % labelEvery === 0) {
      const t = el("text", { class: "axis-label", x: x(i), y: height - 8,
                             "text-anchor": "middle" }, svg);
      t.textContent = d[xKey];
    }
  });

  const cross = el("line", { class: "axisline", y1: M.t, y2: M.t + ih,
                             opacity: 0 }, svg);
  const dot = el("circle", { r: 4.5, fill: color, stroke: "var(--surface)",
                             "stroke-width": 2, opacity: 0 }, svg);
  const hit = el("rect", { x: M.l, y: M.t, width: iw, height: ih,
                           fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (ev) => {
    const box = svg.getBoundingClientRect();
    const rel = (ev.clientX - box.left) * (W / box.width);
    let i = Math.round(((rel - M.l) / iw) * (data.length - 1));
    i = Math.max(0, Math.min(data.length - 1, i));
    const d = data[i];
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i));
    cross.setAttribute("opacity", .6);
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(+d[yKey] || 0));
    dot.setAttribute("opacity", 1);
    showTip(tipFn ? tipFn(d) : `<b>${esc(d[xKey])}</b><br>${d[yKey]}${suffix}`, ev);
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); hideTip();
  });
  return svg;
}

/* ---- density heatmap (sequential, one hue) ---------------------------- */
function heatmap(host, cells, opts = {}) {
  const { xLabel = "", yLabel = "", tipFn, height = 300, bins = 10 } = opts;
  host.innerHTML = "";
  if (!cells.length) { host.innerHTML = '<div class="empty">No data</div>'; return; }

  const W = Math.max(host.clientWidth || 520, 320);
  const M = { t: 10, r: 12, b: 40, l: 46 };
  const size = Math.min(W - M.l - M.r, height - M.t - M.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${height}`, width: "100%", height }, host);
  const cw = size / bins;
  const max = Math.max(...cells.map((c) => c.n), 1);
  const ramp = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)",
                "var(--seq-5)", "var(--seq-6)", "var(--seq-7)"];

  cells.forEach((c) => {
    const x = M.l + c.nx * cw;
    /* y grows upward: feasibility 0 at the bottom */
    const y = M.t + size - (c.fy + 1) * cw;
    const step = c.n === 0 ? -1 : Math.min(ramp.length - 1,
      Math.floor((c.n / max) ** 0.6 * ramp.length));
    const r = el("rect", {
      x: x + 1, y: y + 1, width: cw - 2, height: cw - 2, rx: 3,
      fill: step < 0 ? "var(--grid)" : ramp[step],
      opacity: step < 0 ? .45 : 1,
    }, svg);
    r.addEventListener("mousemove", (ev) => showTip(
      tipFn ? tipFn(c) : `${c.n} plans`, ev));
    r.addEventListener("mouseleave", hideTip);
  });

  [0, 25, 50, 75, 100].forEach((v) => {
    const t = el("text", { class: "axis-label", x: M.l + (v / 100) * size,
                           y: M.t + size + 15, "text-anchor": "middle" }, svg);
    t.textContent = v;
    const t2 = el("text", { class: "axis-label", x: M.l - 8,
                            y: M.t + size - (v / 100) * size + 4,
                            "text-anchor": "end" }, svg);
    t2.textContent = v;
  });
  const xl = el("text", { class: "axis-label", x: M.l + size / 2,
                          y: height - 6, "text-anchor": "middle" }, svg);
  xl.textContent = xLabel;
  const yl = el("text", { class: "axis-label", x: 12, y: M.t + size / 2,
                          "text-anchor": "middle",
                          transform: `rotate(-90 12 ${M.t + size / 2})` }, svg);
  yl.textContent = yLabel;

  /* legend: the ramp itself, low -> high */
  const lx = M.l + size + 14;
  if (lx + 26 < W) {
    ramp.forEach((c, i) => {
      el("rect", { x: lx, y: M.t + size - (i + 1) * 16, width: 12, height: 14,
                   rx: 2, fill: c }, svg);
    });
    const hi = el("text", { class: "axis-label", x: lx + 16,
                            y: M.t + size - ramp.length * 16 + 11 }, svg);
    hi.textContent = "more";
    const lo = el("text", { class: "axis-label", x: lx + 16, y: M.t + size - 4 }, svg);
    lo.textContent = "fewer";
  }
  return svg;
}

/* Redraw on resize and theme change — SVG text metrics depend on width. */
function responsive(fn) {
  let t;
  const run = () => { clearTimeout(t); t = setTimeout(fn, 120); };
  addEventListener("resize", run);
  document.addEventListener("themechange", fn);
  fn();
}

async function loadJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
