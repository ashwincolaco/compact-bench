(function () {
  "use strict";
  var D = window.COMPACT_DATA;
  var KEEP = D.keep; // [0.9, 0.75, 0.5, 0.25, 0.1]
  var NS = "http://www.w3.org/2000/svg";

  // ---------- small DOM helpers (all text goes through textContent) ----------
  function h(tag, attrs) {
    var e = document.createElement(tag);
    setAttrs(e, attrs);
    for (var i = 2; i < arguments.length; i++) add(e, arguments[i]);
    return e;
  }
  function s(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    setAttrs(e, attrs);
    for (var i = 2; i < arguments.length; i++) add(e, arguments[i]);
    return e;
  }
  function setAttrs(e, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") e.textContent = v;
      else if (k === "class") e.setAttribute("class", v);
      else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    });
  }
  function add(e, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { add(e, x); }); return; }
    e.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
  }
  var pct = function (x) { return x === null || x === undefined ? "–" : Math.round(x * 100) + "%"; };
  var f2 = function (x) { return x === null || x === undefined ? "–" : x.toFixed(2); };
  var kfmt = function (v) {
    if (v >= 1e6) return (v / 1e6).toFixed(v % 1e6 ? 1 : 0) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(v % 1e3 ? 1 : 0) + "k";
    return String(Math.round(v));
  };
  var comma = function (v) { return Math.round(v).toLocaleString("en-US"); };
  var mean = function (a) { return a.reduce(function (p, c) { return p + c; }, 0) / a.length; };

  // ---------- scales and frame ----------
  function linear(d0, d1, r0, r1) {
    var f = function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); };
    return f;
  }
  function logScale(d0, d1, r0, r1) {
    var a = Math.log(d0), b = Math.log(d1);
    return function (v) { return r0 + (Math.log(v) - a) / (b - a) * (r1 - r0); };
  }

  function frame(host, o) {
    var W = Math.max(300, Math.floor(host.clientWidth));
    var H = o.height || 320;
    var m = Object.assign({ t: 12, r: 16, b: 40, l: o.y.label ? 60 : 46 }, o.margin || {});
    var svg = s("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": o.aria || "" });
    var x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
    var X = o.x.log ? logScale(o.x.domain[0], o.x.domain[1], x0, x1) : linear(o.x.domain[0], o.x.domain[1], x0, x1);
    var Y = o.y.band ? null : linear(o.y.domain[0], o.y.domain[1], y0, y1);
    var g = s("g");
    svg.appendChild(g);
    if (Y) {
      (o.y.ticks || []).forEach(function (t) {
        var y = Math.round(Y(t)) + 0.5;
        g.appendChild(s("line", { class: t === o.y.domain[0] ? "baseline" : "gridline", x1: x0, x2: x1, y1: y, y2: y }));
        g.appendChild(s("text", { class: "tick", x: x0 - 8, y: y + 4, "text-anchor": "end", text: o.y.fmt ? o.y.fmt(t) : t }));
      });
    }
    (o.x.ticks || []).forEach(function (t) {
      var x = X(t);
      if (o.x.grid) g.appendChild(s("line", { class: "gridline", x1: Math.round(x) + 0.5, x2: Math.round(x) + 0.5, y1: y0, y2: y1 }));
      g.appendChild(s("text", { class: "tick", x: x, y: y0 + 16, "text-anchor": "middle", text: o.x.fmt ? o.x.fmt(t) : t }));
    });
    if (o.x.label) g.appendChild(s("text", { class: "axis-label", x: (x0 + x1) / 2, y: H - 6, "text-anchor": "middle", text: o.x.label }));
    if (o.y.label) g.appendChild(s("text", { class: "axis-label", x: 12, y: (y0 + y1) / 2, "text-anchor": "middle",
      transform: "rotate(-90 12 " + (y0 + y1) / 2 + ")", text: o.y.label }));
    host.replaceChildren(svg);
    return { svg: svg, g: g, X: X, Y: Y, W: W, H: H, x0: x0, x1: x1, y0: y0, y1: y1 };
  }

  function path(pts) {
    return pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join("");
  }
  function line(g, pts, color, width, opacity) {
    var p = s("path", { d: path(pts), fill: "none",
      style: "stroke:" + color + ";stroke-width:" + (width || 2) + "px;stroke-linejoin:round;stroke-linecap:round;" +
        (opacity !== undefined ? "opacity:" + opacity : "") });
    g.appendChild(p);
    return p;
  }
  function dot(g, x, y, color, r, shape, hollow) {
    r = r || 4;
    var st = hollow
      ? "fill:var(--surface);stroke:" + color + ";stroke-width:2px"
      : "fill:" + color + ";stroke:var(--surface);stroke-width:2px";
    var e = shape === "diamond"
      ? s("rect", { x: x - r * 0.9, y: y - r * 0.9, width: r * 1.8, height: r * 1.8, transform: "rotate(45 " + x + " " + y + ")", style: st })
      : s("circle", { cx: x, cy: y, r: r, style: st });
    g.appendChild(e);
    return e;
  }

  // ---------- tooltip ----------
  var tip = document.getElementById("tip");
  function showTip(pos, title, rows) {
    tip.replaceChildren();
    if (title) tip.appendChild(h("div", { class: "t", text: title }));
    rows.forEach(function (r) {
      var row = h("div", { class: "r" });
      if (r.color) { var k = h("span", { class: "key" }); k.style.background = r.color; row.appendChild(k); }
      row.appendChild(h("span", { class: "v", text: r.v }));
      if (r.l) row.appendChild(h("span", { class: "l", text: r.l }));
      tip.appendChild(row);
    });
    tip.classList.add("on");
    tip.setAttribute("aria-hidden", "false");
    var bw = tip.offsetWidth, bh = tip.offsetHeight;
    var x = pos.x + 14, y = pos.y + 14;
    if (x + bw > window.innerWidth - 8) x = pos.x - bw - 14;
    if (y + bh > window.innerHeight - 8) y = pos.y - bh - 14;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip.classList.remove("on"); tip.setAttribute("aria-hidden", "true"); }
  function rectPos(el) { var r = el.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }

  // Nearest-point hover over a whole plot, plus focusable hit targets for keyboard.
  function hoverLayer(F, pts, onClick, noKeys) {
    var ov = s("rect", { x: F.x0 - 10, y: F.y1 - 10, width: F.x1 - F.x0 + 20, height: F.y0 - F.y1 + 20, class: "hit" });
    var ring = s("circle", { r: 9, class: "focus-ring", style: "fill:none;stroke-width:1.5px;pointer-events:none", cx: 0, cy: 0, visibility: "hidden" });
    F.svg.appendChild(ov);
    F.svg.appendChild(ring);
    function nearest(evt) {
      var r = F.svg.getBoundingClientRect();
      var sx = F.W / r.width, px = (evt.clientX - r.left) * sx, py = (evt.clientY - r.top) * sx;
      var best = null, bd = 30 * 30;
      pts.forEach(function (p) {
        var d = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py);
        if (d < bd) { bd = d; best = p; }
      });
      return best;
    }
    function mark(p) {
      if (!p) { ring.setAttribute("visibility", "hidden"); ring.classList.remove("on"); return; }
      ring.setAttribute("cx", p.x); ring.setAttribute("cy", p.y); ring.setAttribute("visibility", "visible"); ring.classList.add("on");
    }
    ov.addEventListener("pointermove", function (e) {
      var p = nearest(e);
      mark(p);
      if (p) { showTip({ x: e.clientX, y: e.clientY }, p.title, p.rows); ov.style.cursor = onClick ? "pointer" : "default"; }
      else hideTip();
    });
    ov.addEventListener("pointerleave", function () { mark(null); hideTip(); });
    if (onClick) ov.addEventListener("click", function (e) { var p = nearest(e); if (p) onClick(p); });
    // keyboard: one focusable target per point
    if (noKeys) return;
    var kb = s("g");
    F.svg.appendChild(kb);
    pts.forEach(function (p) {
      var t = s("circle", { cx: p.x, cy: p.y, r: 8, class: "hit", tabindex: 0, role: "img", "aria-label": p.title + ": " + p.rows.map(function (r) { return r.v + " " + (r.l || ""); }).join(", ") });
      t.addEventListener("focus", function () { mark(p); showTip(rectPos(t), p.title, p.rows); });
      t.addEventListener("blur", function () { mark(null); hideTip(); });
      if (onClick) t.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(p); } });
      t.style.pointerEvents = "none";
      kb.appendChild(t);
    });
  }

  // Crosshair hover snapping to a shared set of x positions.
  function crosshair(F, xs, rowsAt, titleAt) {
    var ov = s("rect", { x: F.x0, y: F.y1, width: F.x1 - F.x0, height: F.y0 - F.y1, class: "hit" });
    var ln = s("line", { class: "crosshair", y1: F.y1, y2: F.y0, x1: 0, x2: 0, visibility: "hidden" });
    F.svg.appendChild(ln);
    F.svg.appendChild(ov);
    ov.addEventListener("pointermove", function (e) {
      var r = F.svg.getBoundingClientRect();
      var px = (e.clientX - r.left) * (F.W / r.width);
      var best = 0, bd = Infinity;
      xs.forEach(function (x, i) { var d = Math.abs(F.X(x) - px); if (d < bd) { bd = d; best = i; } });
      var X = Math.round(F.X(xs[best])) + 0.5;
      ln.setAttribute("x1", X); ln.setAttribute("x2", X); ln.setAttribute("visibility", "visible");
      showTip({ x: e.clientX, y: e.clientY }, titleAt(best), rowsAt(best));
    });
    ov.addEventListener("pointerleave", function () { ln.setAttribute("visibility", "hidden"); hideTip(); });
  }

  // ---------- controls ----------
  function seg(label, options, value, onChange) {
    var wrap = h("div", { class: "ctl" }, h("span", { text: label }));
    var box = h("div", { class: "seg", role: "group", "aria-label": label });
    var btns = options.map(function (o) {
      var b = h("button", { type: "button", "aria-pressed": String(o.value === value), text: o.label, disabled: o.disabled });
      b.addEventListener("click", function () {
        btns.forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onChange(o.value);
      });
      box.appendChild(b);
      return b;
    });
    wrap.appendChild(box);
    wrap.setDisabled = function (i, d) { btns[i].disabled = d; };
    wrap.press = function (val) { options.forEach(function (o, i) { btns[i].setAttribute("aria-pressed", String(o.value === val)); }); };
    return wrap;
  }
  function selectCtl(label, groups, value, onChange) {
    var sel = h("select", { "aria-label": label });
    groups.forEach(function (g) {
      var parent = g.label ? h("optgroup", { label: g.label }) : sel;
      g.options.forEach(function (o) { parent.appendChild(h("option", { value: o.value, text: o.label, selected: o.value === value })); });
      if (g.label) sel.appendChild(parent);
    });
    sel.addEventListener("change", function () { onChange(sel.value); });
    return h("div", { class: "ctl" }, h("span", { text: label }), sel);
  }
  function legend(items) {
    return h("div", { class: "legend" }, items.map(function (it) {
      var sw;
      if (it.kind === "line") sw = s("svg", { width: 16, height: 8 }, s("line", { x1: 1, x2: 15, y1: 4, y2: 4, style: "stroke:" + it.color + ";stroke-width:2px;stroke-linecap:round" }));
      else if (it.kind === "hollow") sw = s("svg", { width: 12, height: 12 }, s("circle", { cx: 6, cy: 6, r: 4, style: "fill:var(--surface);stroke:" + it.color + ";stroke-width:2px" }));
      else if (it.kind === "diamond") sw = s("svg", { width: 12, height: 12 }, s("rect", { x: 2.5, y: 2.5, width: 7, height: 7, transform: "rotate(45 6 6)", style: "fill:" + it.color }));
      else if (it.kind === "bar") sw = s("svg", { width: 12, height: 10 }, s("rect", { x: 0, y: 1, width: 12, height: 8, rx: 2, style: "fill:" + it.color }));
      else sw = s("svg", { width: 12, height: 12 }, s("circle", { cx: 6, cy: 6, r: 4.5, style: "fill:" + it.color }));
      return h("span", { class: "k" }, sw, it.label);
    }));
  }
  function tableToggle(buildTable) {
    var wrap = h("div", { class: "tablewrap", hidden: true });
    var btn = h("button", { type: "button", class: "linkish", text: "Show as table", "aria-expanded": "false" });
    btn.addEventListener("click", function () {
      var open = wrap.hidden;
      wrap.hidden = !open;
      btn.textContent = open ? "Hide table" : "Show as table";
      btn.setAttribute("aria-expanded", String(open));
      if (open) wrap.replaceChildren(buildTable());
    });
    return { btn: btn, wrap: wrap, refresh: function () { if (!wrap.hidden) wrap.replaceChildren(buildTable()); } };
  }
  function table(head, rows, numeric) {
    return h("table", { class: "data" },
      h("thead", null, h("tr", null, head.map(function (c, i) { return h("th", { class: numeric && numeric[i] ? "num" : null, text: c }); }))),
      h("tbody", null, rows.map(function (r) {
        return h("tr", null, r.map(function (c, i) { return h("td", { class: numeric && numeric[i] ? "num" : null, text: c }); }));
      })));
  }
  function panel(title, sub) {
    var chart = h("div", { class: "chart" });
    var el = h("div", { class: "panel" }, title ? h("h3", { text: title }) : null, sub ? h("p", { class: "sub", text: sub }) : null);
    el.chart = chart;
    el.appendChild(chart);
    return el;
  }

  var GEN = { earlier: "var(--s1)", newer: "var(--s2)" };
  var GEN_LABEL = { earlier: "Released Aug–Sep 2024", newer: "Released Nov 2024 onwards" };
  var byTag = {};
  D.frontiers.forEach(function (f) { byTag[f.tag] = f; });
  function modelLabel(f) { return f.name + (f.precision === "nf4" ? " (4-bit)" : ""); }

  // =====================================================================
  // 1. Same bytes, different answers
  // =====================================================================
  var FAM = [
    { fam: "eviction", label: "Eviction, best of six methods", color: "var(--s1)" },
    { fam: "quantization", label: "KV quantization", color: "var(--s2)" },
    { fam: "evict+quant", label: "Evict, then quantize (best)", color: "var(--s3)" },
    { fam: "prompt", label: "LLMLingua-2", color: "var(--s4)" },
    { fam: "summary", label: "Model's own summary", color: "var(--s5)" }
  ];
  function mechLabel(p) {
    if (p.family === "eviction") return p.method + ", keep " + pct(+p.setting) + " of tokens";
    if (p.family === "quantization") return p.setting + "-bit KV cache";
    if (p.family === "evict+quant") { var m = p.method.split("+"); return m[0] + ", keep " + pct(+p.setting) + ", stored at " + m[1].replace("bit", "-bit"); }
    if (p.family === "prompt") return "LLMLingua-2, target rate " + p.setting;
    if (p.family === "summary") return "summary, target " + pct(+p.setting) + " of tokens";
    return p.method;
  }
  function mechSeries(points) {
    var out = {};
    FAM.forEach(function (F) {
      var pts = points.filter(function (p) { return p.family === F.fam; });
      if (F.fam === "eviction" || F.fam === "evict+quant") {
        var best = {};
        pts.forEach(function (p) { var k = p.bpt.toFixed(3); if (!best[k] || p.acc > best[k].acc) best[k] = p; });
        pts = Object.keys(best).map(function (k) { return best[k]; });
      }
      out[F.fam] = pts.sort(function (a, b) { return a.bpt - b.bpt; });
    });
    return out;
  }

  function viewMech(root) {
    var st = { model: D.mechanisms.length - 1, all: false };
    var dek = h("p", { class: "dek" });
    var p = panel(null, null);
    var tt = tableToggle(function () {
      var pts = D.mechanisms[st.model].points.filter(function (q) { return q.family !== "full"; })
        .sort(function (a, b) { return b.bpt - a.bpt; });
      return table(["Mechanism", "Setting", "BPT (% of full cache)", "Accuracy", "n"],
        pts.map(function (q) { return [q.family, mechLabel(q), (q.bpt * 100).toFixed(1) + "%", f2(q.acc), q.n]; }),
        [0, 0, 1, 1, 1]);
    });
    var allBox = h("input", { type: "checkbox" });
    allBox.addEventListener("change", function () { st.all = allBox.checked; render(); });
    root.append(
      h("h2", { text: "Same bytes, different answers" }),
      dek,
      h("div", { class: "controls" },
        seg("Model", D.mechanisms.map(function (m, i) { return { label: m.name, value: i }; }), st.model, function (v) { st.model = v; render(); }),
        h("label", { class: "check" }, allBox, "Show all six eviction methods"),
        h("span", { class: "spacer" }), tt.btn),
      legend(FAM.map(function (F) { return { kind: "line", color: F.color, label: F.label }; })),
      p, tt.wrap);

    function render() {
      var M = D.mechanisms[st.model];
      var series = mechSeries(M.points);
      var low = M.points.filter(function (q) { return q.bpt <= 0.305 && q.family !== "full"; });
      var q = low.filter(function (x) { return x.family === "quantization"; }).sort(function (a, b) { return b.acc - a.acc; })[0];
      var o = low.filter(function (x) { return x.family !== "quantization"; }).sort(function (a, b) { return b.acc - a.acc; })[0];
      dek.replaceChildren("Accuracy on the same 48 questions against bytes kept per token of history. On " + M.name +
        ", at 30% of the full cache or less, the best KV quantization answers ", h("b", { text: f2(q.acc) }),
        " and the best of everything else ", h("b", { text: f2(o.acc) }), " (" + mechLabel(o) + "). The full cache answers every question.");
      var F = frame(p.chart, { height: 340, aria: "Accuracy against bytes per token for five compaction mechanisms on " + M.name,
        x: { domain: [5, 62], ticks: [10, 20, 30, 40, 50, 60], fmt: function (t) { return t + "%"; }, label: "bytes kept per token of history (% of full cache)" },
        y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (t) { return t.toFixed(2); }, label: "accuracy" } });
      var hov = [];
      if (st.all) {
        D.methods.forEach(function (m) {
          var pts = M.points.filter(function (x) { return x.family === "eviction" && x.method === m; }).sort(function (a, b) { return a.bpt - b.bpt; });
          line(F.g, pts.map(function (x) { return [F.X(x.bpt * 100), F.Y(x.acc)]; }), "var(--gray-line)", 1.25);
          pts.forEach(function (x) { hov.push({ x: F.X(x.bpt * 100), y: F.Y(x.acc), title: "Eviction", rows: [{ v: f2(x.acc), l: mechLabel(x) + ", " + (x.bpt * 100).toFixed(1) + "% of bytes", color: "var(--gray-line)" }] }); });
        });
      }
      FAM.forEach(function (Fm) {
        var pts = series[Fm.fam];
        line(F.g, pts.map(function (x) { return [F.X(x.bpt * 100), F.Y(x.acc)]; }), Fm.color, 2);
        pts.forEach(function (x) {
          var X = F.X(x.bpt * 100), Y = F.Y(x.acc);
          dot(F.g, X, Y, Fm.color, 4);
          hov.push({ x: X, y: Y, title: Fm.label, rows: [{ v: f2(x.acc), l: "accuracy", color: Fm.color }, { v: (x.bpt * 100).toFixed(1) + "%", l: "of full-cache bytes" }, { v: mechLabel(x), l: "" }] });
        });
      });
      hoverLayer(F, hov);
      tt.refresh();
    }
    return render;
  }

  // =====================================================================
  // 2. Generation, not size
  // =====================================================================
  var PAIRS = [["qwen0.5b", "qwen3-0.6b"], ["qwen1.5b", "qwen3-1.7b"], ["qwen3b", "qwen3-4b-nf4"],
    ["qwen7b-fp16", "qwen3-8b"], ["qwen14b-nf4", "qwen3-14b-nf4"]];
  var SCORERS = ["SnapKV", "StreamingLLM", "TOVA", "Knorm", "ExpectedAttn"];

  function curveOf(f, method) {
    var acc = KEEP.map(function (k, i) {
      if (method === "avg") return mean(SCORERS.map(function (m) { return f.acc[m][i]; }));
      return f.acc[method][i];
    });
    var pts = KEEP.map(function (k, i) { return [k, acc[i]]; });
    pts.unshift([1, f.baseline]);
    return pts.sort(function (a, b) { return a[0] - b[0]; });
  }

  function viewGen(root) {
    var st = { sel: ["qwen7b-fp16", "qwen3-8b"], xmode: "frac", method: "avg" };
    var left = panel("Where each model loses half its accuracy", "Share of its own full cache at which scorer-averaged accuracy halves. Lower means it tolerates more compaction. Click a model to trace its curve.");
    var right = panel("Accuracy as the cache shrinks", null);
    var tt = tableToggle(function () {
      return table(["Model", "Weights", "Released", "Parameters", "Full cache (bytes/token)", "Collapse point"],
        D.frontiers.map(function (f) { return [f.name, f.precision, f.released, f.params + "B", comma(f.fullBpt), (f.collapseAbove ? ">" : "") + pct(f.collapse)]; }),
        [0, 0, 0, 1, 1, 1]);
    });
    var xseg = seg("Budget axis", [{ label: "Share of own cache", value: "frac" }, { label: "Bytes per token", value: "bytes" }], st.xmode, function (v) { st.xmode = v; render(); });
    var msel = selectCtl("Scorer", [{ options: [{ label: "Average of five scorers", value: "avg" }].concat(D.methods.map(function (m) { return { label: m, value: m }; })) }], st.method, function (v) { st.method = v; render(); });
    var clear = h("button", { type: "button", class: "linkish", text: "Clear selection" });
    clear.addEventListener("click", function () { st.sel = []; render(); });
    root.append(
      h("h2", { text: "Tolerance to eviction follows model generation, not size" }),
      h("p", { class: "dek" }, "Twenty-one settings of sixteen models. In every same-family pair of similar size (grey links), the model released from November 2024 onwards keeps half its accuracy at a smaller share of its cache. Within each generation, size moves it far less."),
      h("div", { class: "controls" }, xseg, msel, clear, h("span", { class: "spacer" }), tt.btn),
      legend([{ color: GEN.earlier, label: GEN_LABEL.earlier }, { color: GEN.newer, label: GEN_LABEL.newer }, { kind: "diamond", color: "var(--muted)", label: "4-bit weights" }]),
      h("div", { class: "grid2" }, left, right), tt.wrap);

    function toggle(tag) {
      var i = st.sel.indexOf(tag);
      if (i >= 0) st.sel.splice(i, 1);
      else { st.sel.push(tag); if (st.sel.length > 4) st.sel.shift(); }
      render();
    }

    function render() {
      // scatter
      var F = frame(left.chart, { height: 330, margin: { l: 60, b: 40 }, aria: "Collapse point against model size",
        x: { domain: [0.38, 30], log: true, ticks: [0.5, 1, 2, 4, 8, 16], fmt: function (t) { return t + "B"; }, label: "parameters (log scale)", grid: true },
        y: { domain: [0.25, 0.95], ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fmt: pct, label: "collapse point" } });
      var pos = {};
      D.frontiers.forEach(function (f) {
        var jig = f.precision === "nf4" ? 1.07 : 1;
        pos[f.tag] = [F.X(f.params * jig), F.Y(f.collapse)];
      });
      PAIRS.forEach(function (pr) {
        if (pos[pr[0]] && pos[pr[1]]) line(F.g, [pos[pr[0]], pos[pr[1]]], "var(--gray-line)", 1.25);
      });
      var hov = [];
      D.frontiers.forEach(function (f) {
        var p = pos[f.tag];
        if (st.sel.indexOf(f.tag) >= 0) F.g.appendChild(s("circle", { cx: p[0], cy: p[1], r: 9, style: "fill:none;stroke:var(--ink);stroke-width:1.5px" }));
        dot(F.g, p[0], p[1], GEN[f.gen], 4.5, f.precision === "nf4" ? "diamond" : "circle");
        hov.push({ x: p[0], y: p[1], tag: f.tag, title: modelLabel(f),
          rows: [{ v: (f.collapseAbove ? ">" : "") + pct(f.collapse), l: "collapse point", color: GEN[f.gen] },
            { v: f.params + "B", l: "parameters" }, { v: f.released, l: "released" }] });
      });
      hoverLayer(F, hov, function (p) { toggle(p.tag); });

      // curves
      var bytes = st.xmode === "bytes";
      var all = D.frontiers.map(function (f) {
        return { f: f, pts: curveOf(f, st.method).map(function (q) { return [bytes ? q[0] * f.fullBpt : q[0], q[1]]; }) };
      });
      var xs = [];
      all.forEach(function (c) { c.pts.forEach(function (q) { xs.push(q[0]); }); });
      right.querySelector(".sub") || right.insertBefore(h("p", { class: "sub" }), right.chart);
      right.querySelector(".sub").textContent = (st.method === "avg" ? "Average of the five non-random scorers" : st.method) +
        (bytes ? ", against absolute bytes per token. Read side by side with the left axis: fractions line up by generation, bytes do not." : ", against the share of each model's own cache.");
      var G = frame(right.chart, { height: 330, margin: { l: 60, b: 40, r: 18 }, aria: "Accuracy against budget for every setting",
        x: bytes ? { domain: [Math.min.apply(null, xs) * 0.9, Math.max.apply(null, xs) * 1.1], log: true, ticks: [1e3, 1e4, 1e5, 1e6], fmt: kfmt, label: "bytes per token of history (log scale)", grid: true }
          : { domain: [0.08, 1.02], ticks: [0.1, 0.25, 0.5, 0.75, 1], fmt: pct, label: "budget (share of the model's own full cache)", grid: true },
        y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (t) { return t.toFixed(2); }, label: "accuracy" } });
      var hov2 = [], labels = [];
      all.forEach(function (c) {
        var chosen = st.sel.indexOf(c.f.tag) >= 0;
        if (chosen) return;
        line(G.g, c.pts.map(function (q) { return [G.X(q[0]), G.Y(q[1])]; }), GEN[c.f.gen], 1.1, 0.28);
      });
      all.forEach(function (c) {
        var chosen = st.sel.indexOf(c.f.tag) >= 0;
        var sp = c.pts.map(function (q) { return [G.X(q[0]), G.Y(q[1])]; });
        if (chosen) {
          line(G.g, sp, GEN[c.f.gen], 2.25);
          sp.forEach(function (q) { dot(G.g, q[0], q[1], GEN[c.f.gen], 3.5, c.f.precision === "nf4" ? "diamond" : "circle"); });
          labels.push({ f: c.f, sp: sp });
        }
        c.pts.forEach(function (q, i) {
          hov2.push({ x: sp[i][0], y: sp[i][1], tag: c.f.tag, title: modelLabel(c.f),
            rows: [{ v: f2(q[1]), l: "accuracy", color: GEN[c.f.gen] },
              { v: bytes ? comma(q[0]) : pct(q[0]), l: bytes ? "bytes per token" : "of its own cache" }] });
        });
      });
      // direct labels for selected curves at the half-budget point; skip on collision
      var placed = [];
      labels.forEach(function (L) {
        var cand = [2, 1]; // the 50% point, then the 25% point (points run from 10% to 100%)
        for (var j = 0; j < cand.length; j++) {
          var q = L.sp[cand[j]];
          if (!q) continue;
          var clash = placed.some(function (p) { return Math.abs(p[1] - q[1]) < 14 && Math.abs(p[0] - q[0]) < 90; });
          if (!clash) {
            placed.push(q);
            G.g.appendChild(s("text", { class: "dlabel strong", x: q[0] + 7, y: q[1] - 7, text: modelLabel(L.f) }));
            break;
          }
        }
      });
      hoverLayer(G, hov2, function (p) { toggle(p.tag); }, true);
      tt.refresh();
    }
    return render;
  }

  // =====================================================================
  // 3. Seeing the question
  // =====================================================================
  var READS = { SnapKV: true, TOVA: true };
  function viewQA(root) {
    var st = { model: 1, budget: 2 };
    var dek = h("p", { class: "dek" });
    var p = panel(null, null);
    var tt = tableToggle(function () {
      var M = D.questionAware[st.model];
      var rows = [];
      D.methods.forEach(function (m) {
        KEEP.forEach(function (k, i) { if (i < 4) rows.push([m, pct(k), f2(M.agnostic[m][i]), f2(M.aware[m][i]), (M.aware[m][i] - M.agnostic[m][i] >= 0 ? "+" : "") + f2(M.aware[m][i] - M.agnostic[m][i])]); });
      });
      return table(["Method", "Budget", "Question after compression", "Question before compression", "Change"], rows, [0, 1, 1, 1, 1]);
    });
    root.append(
      h("h2", { text: "Content scorers lose to random eviction because they cannot see the question" }),
      dek,
      h("div", { class: "controls" },
        selectCtl("Model", [{ options: D.questionAware.map(function (m, i) { return { label: m.name + (m.tag.indexOf("nf4") >= 0 ? " (4-bit)" : ""), value: String(i) }; }) }], String(st.model), function (v) { st.model = +v; render(); }),
        seg("Budget", KEEP.slice(0, 4).map(function (k, i) { return { label: pct(k), value: i }; }), st.budget, function (v) { st.budget = v; render(); }),
        h("span", { class: "spacer" }), tt.btn),
      legend([{ kind: "hollow", color: "var(--muted)", label: "Question arrives after compression (default)" },
        { color: "var(--s1)", label: "Question in the context before compression" }]),
      p, tt.wrap);

    function render() {
      var M = D.questionAware[st.model], b = st.budget;
      var order = ["SnapKV", "TOVA", "Knorm", "ExpectedAttn", "StreamingLLM", "Random"];
      var other = order.filter(function (m) { return !READS[m]; });
      var maxOther = Math.max.apply(null, other.map(function (m) { return Math.abs(M.aware[m][b] - M.agnostic[m][b]); }));
      dek.replaceChildren("On " + M.name + " at " + pct(KEEP[b]) + " of the cache, SnapKV goes from ", h("b", { text: f2(M.agnostic.SnapKV[b]) }),
        " to ", h("b", { text: f2(M.aware.SnapKV[b]) }), " and TOVA from ", h("b", { text: f2(M.agnostic.TOVA[b]) }), " to ",
        h("b", { text: f2(M.aware.TOVA[b]) }), " once the question is in the compressed context. The four methods that do not score against it move by at most " + f2(maxOther) + ".");
      var rowH = 40, top = 30;
      var H = top + order.length * rowH + 50;
      var F = frame(p.chart, { height: H, margin: { l: 150, r: 70, t: top, b: 40 }, aria: "Accuracy before and after the question is visible to the scorer",
        x: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (t) { return t.toFixed(2); }, label: "accuracy", grid: true }, y: { band: true } });
      F.g.appendChild(s("text", { class: "axis-label", x: 8, y: 16, text: "scores against the question" }));
      F.g.appendChild(s("text", { class: "axis-label", x: 8, y: top + 2 * rowH + 16, text: "does not" }));
      var hov = [];
      order.forEach(function (m, i) {
        var y = top + i * rowH + rowH / 2 + (i >= 2 ? 10 : 0);
        var a = M.agnostic[m][b], w = M.aware[m][b];
        F.g.appendChild(s("text", { class: "dlabel", x: F.x0 - 12, y: y + 4, "text-anchor": "end", text: m }));
        line(F.g, [[F.X(a), y], [F.X(w), y]], "var(--gray-line)", 2);
        dot(F.g, F.X(a), y, "var(--muted)", 4.5, "circle", true);
        dot(F.g, F.X(w), y, "var(--s1)", 4.5);
        F.g.appendChild(s("text", { class: "dlabel", x: F.X(Math.max(a, w)) + 12, y: y + 4, text: f2(a) + " → " + f2(w) }));
        var rows = [{ v: f2(a), l: "question after compression", color: "var(--muted)" }, { v: f2(w), l: "question before compression", color: "var(--s1)" }];
        hov.push({ x: F.X(a), y: y, title: m + " at " + pct(KEEP[b]) + " of the cache", rows: rows });
        hov.push({ x: F.X(w), y: y, title: m + " at " + pct(KEEP[b]) + " of the cache", rows: rows });
      });
      hoverLayer(F, hov);
      tt.refresh();
    }
    return render;
  }

  // =====================================================================
  // 4. What it knows it lost
  // =====================================================================
  var AUD_METHODS = ["StreamingLLM", "SnapKV", "Knorm", "Random"];
  function barRows(host, rows, opts) {
    var rowH = 30, top = 8;
    var F = frame(host, { height: top + rows.length * rowH + 44, margin: { l: opts.left || 140, r: 44, t: top, b: 38 }, aria: opts.aria,
      x: { domain: [0, 1], ticks: [0, 0.5, 1], fmt: pct, label: opts.xlabel, grid: true }, y: { band: true } });
    var hov = [];
    rows.forEach(function (r, i) {
      var y = top + i * rowH + rowH / 2;
      F.g.appendChild(s("text", { class: "dlabel", x: F.x0 - 10, y: y + 4, "text-anchor": "end", text: r.label }));
      if (r.divider) F.g.appendChild(s("line", { class: "gridline", x1: 6, x2: F.x1, y1: y - rowH / 2, y2: y - rowH / 2 }));
      if (r.value === null || r.value === undefined) {
        F.g.appendChild(s("text", { class: "tick", x: F.x0 + 6, y: y + 4, text: r.empty || "no cases" }));
        return;
      }
      if (r.dot) {
        dot(F.g, F.X(r.value), y, r.color, 5);
      } else {
        var w = Math.max(0, F.X(r.value) - F.x0), bh = 14;
        var rad = Math.min(4, w / 2);
        F.g.appendChild(s("path", { d: "M" + F.x0 + " " + (y - bh / 2) + "h" + Math.max(0, w - rad) + "a" + rad + " " + rad + " 0 0 1 " + rad + " " + rad +
          "v" + (bh - 2 * rad) + "a" + rad + " " + rad + " 0 0 1 " + (-rad) + " " + rad + "h" + (-Math.max(0, w - rad)) + "z", style: "fill:" + r.color }));
      }
      F.g.appendChild(s("text", { class: "dlabel", x: F.X(r.value) + (r.dot ? 10 : 6), y: y + 4, text: pct(r.value) }));
      hov.push({ x: F.X(r.value), y: y, title: r.label, rows: r.tip || [{ v: pct(r.value), l: "", color: r.color }] });
    });
    if (opts.ref !== undefined && opts.ref !== null) {
      var X = Math.round(F.X(opts.ref)) + 0.5;
      F.g.appendChild(s("line", { x1: X, x2: X, y1: F.y1, y2: F.y0, style: "stroke:var(--muted);stroke-width:1px" }));
    }
    hoverLayer(F, hov);
  }

  function viewAudit(root) {
    var valid = D.audit.filter(function (a) { return a.probeValid; });
    var invalid = D.audit.filter(function (a) { return !a.probeValid; });
    var st = { tag: (valid.filter(function (a) { return a.tag === "qwen7b-nf4"; })[0] || valid[0]).tag, ratio: "0.5" };
    var dek = h("p", { class: "dek" });
    var note = h("div", { class: "note", hidden: true });
    var pa = panel("Says it still has the fact", null);
    var pb = panel("What survived in the wrong answers", "Share of the fact's key tokens still in the cache, pooled over both compression ratios.");
    var pc = panel("Stated confidence on wrong answers", "Mean 0–100 confidence the model gave with a wrong answer, pooled over both ratios.");
    var tt = tableToggle(function () {
      var A = byAudit(st.tag), rows = [];
      AUD_METHODS.forEach(function (m) {
        var M = A.methods[m];
        ["0.5", "0.9"].forEach(function (r) { rows.push([m, r, f2(M[r].acc), pct(M[r].overclaim), M[r].nWrong]); });
      });
      return h("div", null,
        table(["Control", "Probe says yes", "Mean confidence"], [["Fact present, full cache", pct(A.presentYes), A.confOmitted ? "–" : f2(A.confPresent)], ["Fact absent, full cache", pct(A.absentYes), A.confOmitted ? "–" : f2(A.confAbsent)]], [0, 1, 1]),
        table(["Method", "Compression ratio", "Accuracy", "Probe yes on wrong answers", "Wrong answers"], rows, [0, 1, 1, 1, 1]),
        table(["Method", "Probe answer", "Key kept", "Value kept", "Wrong answers"], [].concat.apply([], AUD_METHODS.map(function (m) {
          var M = A.methods[m];
          return [[m, "yes", pct(M.survivalYes.key), pct(M.survivalYes.value), M.survivalYes.n], [m, "no", pct(M.survivalNo.key), pct(M.survivalNo.value), M.survivalNo.n]];
        })), [0, 0, 1, 1, 1]));
    });
    function byAudit(tag) { return D.audit.filter(function (a) { return a.tag === tag; })[0]; }
    root.append(
      h("h2", { text: "A fact that is partly kept looks present to the model" }),
      dek,
      h("div", { class: "controls" },
        selectCtl("Model", [
          { label: "Probe passes both controls", options: valid.map(function (a) { return { label: a.name + (a.tag.indexOf("nf4") >= 0 ? " (4-bit)" : ""), value: a.tag }; }) },
          { label: "Probe fails a control", options: invalid.map(function (a) { return { label: a.name + (a.tag.indexOf("nf4") >= 0 ? " (4-bit)" : ""), value: a.tag }; }) }
        ], st.tag, function (v) { st.tag = v; render(); }),
        seg("Compression ratio", [{ label: "0.5", value: "0.5" }, { label: "0.9", value: "0.9" }], st.ratio, function (v) { st.ratio = v; render(); }),
        h("span", { class: "spacer" }), tt.btn),
      note,
      h("div", { class: "grid3" }, pa, pb, pc), tt.wrap);

    function render() {
      var A = byAudit(st.tag);
      var snap = A.methods.SnapKV, sllm = A.methods.StreamingLLM;
      if (A.probeValid) {
        dek.replaceChildren("On " + A.name + ", the probe says the fact is still there on ", h("b", { text: pct(snap[st.ratio].overclaim) }),
          " of SnapKV's wrong answers and ", h("b", { text: pct(sllm[st.ratio].overclaim) }), " of StreamingLLM's. SnapKV scatters its evictions and leaves fragments behind; StreamingLLM removes a contiguous span, so the fact is either whole or gone.");
      } else {
        dek.replaceChildren("On " + A.name + " the probe does not pass its controls, so overclaim rates here say nothing about compaction. The two control rows show why.");
      }
      note.hidden = A.probeValid;
      note.replaceChildren(h("b", { text: "Control failed. " }), A.absentYes > 0.1
        ? "Asked about a fact that was never in its context, this model says it has it " + pct(A.absentYes) + " of the time."
        : "With the fact in plain view, this model says it has it only " + pct(A.presentYes) + " of the time.");
      pa.querySelector(".sub") || pa.insertBefore(h("p", { class: "sub" }), pa.chart);
      pa.querySelector(".sub").textContent = "Yes/no probe from the same cache as the answer. Controls on all contexts; methods on wrong answers at ratio " + st.ratio + ".";
      barRows(pa.chart, [
        { label: "Fact present (control)", value: A.presentYes, color: "var(--gray-line)" },
        { label: "Fact absent (control)", value: A.absentYes, color: "var(--gray-line)" }
      ].concat(AUD_METHODS.map(function (m, i) {
        var M = A.methods[m][st.ratio];
        return { label: m, value: M.nWrong ? M.overclaim : null, color: "var(--s1)", divider: i === 0, empty: "no wrong answers",
          tip: [{ v: pct(M.overclaim), l: "of wrong answers claimed", color: "var(--s1)" }, { v: String(M.nWrong), l: "wrong answers" }, { v: f2(M.acc), l: "accuracy" }] };
      })), { aria: "Probe says yes", xlabel: "share answering yes" });

      var rows = [];
      AUD_METHODS.forEach(function (m, i) {
        var M = A.methods[m];
        [["Yes", "said yes", "var(--s1)"], ["No", "said no", "var(--s2)"]].forEach(function (k, j) {
          var S = M["survival" + k[0]];
          rows.push({ label: j === 0 ? m + ", said yes" : "said no", value: S.n ? S.key : null, color: k[2], divider: i > 0 && j === 0, empty: "no cases",
            tip: [{ v: pct(S.key), l: "of key tokens kept", color: k[2] }, { v: pct(S.value), l: "of value tokens kept" }, { v: String(S.n), l: "wrong answers where the probe " + k[1] }] });
        });
      });
      pb.querySelector(".legend") || pb.insertBefore(legend([{ kind: "bar", color: "var(--s1)", label: "probe said yes" }, { kind: "bar", color: "var(--s2)", label: "probe said no" }]), pb.chart);
      barRows(pb.chart, rows, { aria: "Needle survival split by probe answer", xlabel: "key tokens kept", left: 130 });

      if (A.confOmitted) {
        pc.chart.replaceChildren(h("p", { class: "note", text: "This run capped generation at 50 tokens and scored a missing confidence as 50, so its confidence is not used." }));
      } else {
        barRows(pc.chart, [{ label: "Fact absent (control)", value: A.confAbsent, color: "var(--gray-line)", dot: true }].concat(AUD_METHODS.map(function (m, i) {
          return { label: m, value: A.methods[m].confWrong, color: "var(--s1)", dot: true, divider: i === 0 };
        })), { aria: "Stated confidence on wrong answers", xlabel: "stated confidence", ref: A.confAbsent });
      }
      tt.refresh();
    }
    return render;
  }

  // =====================================================================
  // 5. Exact vs summarized memory
  // =====================================================================
  function viewRev(root) {
    var st = { tag: "qwen3-1.7b", doc: "standard" };
    var dek = h("p", { class: "dek" });
    var p = panel(null, null);
    var docSeg;
    function R() { return D.reversibility.filter(function (r) { return r.tag === st.tag; })[0]; }
    var tt = tableToggle(function () {
      var r = R(), pts = r[st.doc] || [];
      return table(["Budget", "Rolling summary", "Raw buffer (exact)"], pts.map(function (q) { return [pct(q.budget), f2(q.summary), f2(q.raw)]; }), [1, 1, 1]);
    });
    var groups = [
      { label: "Released Aug–Sep 2024", options: [] }, { label: "Released Nov 2024 onwards", options: [] }
    ];
    D.reversibility.forEach(function (r) { groups[r.gen === "earlier" ? 0 : 1].options.push({ label: r.name + (r.dense ? "" : " (standard only)"), value: r.tag }); });
    var leg = h("div");
    docSeg = seg("Document", [{ label: "Standard: 12 facts", value: "standard" }, { label: "Dense: 48 facts", value: "dense" }], st.doc, function (v) { st.doc = v; render(); });
    root.append(
      h("h2", { text: "Whether exact memory beats a summary depends on how much the questions need" }),
      dek,
      h("div", { class: "controls" },
        selectCtl("Model", groups, st.tag, function (v) { st.tag = v; if (!R().dense) { st.doc = "standard"; docSeg.press("standard"); } render(); }),
        docSeg, h("span", { class: "spacer" }), tt.btn),
      leg, p, tt.wrap);

    function render() {
      var r = R();
      docSeg.setDisabled(1, !r.dense);
      leg.replaceChildren(legend([{ kind: "line", color: "var(--s1)", label: "Rolling summary (covers everything, lossy)" }, { kind: "line", color: "var(--s2)", label: "Raw buffer (exact, keeps only recent chunks)" }]
        .concat(st.doc === "dense" ? [{ kind: "line", color: "var(--gray-line)", label: "Most facts a summary this size could hold" }] : [])));
      var pts = r[st.doc];
      var cross = null;
      for (var i = 0; i + 1 < pts.length; i++) {
        var d0 = pts[i].raw - pts[i].summary, d1 = pts[i + 1].raw - pts[i + 1].summary;
        if (d0 <= 0 && d1 > 0 && pts[i + 1].budget < 1) { cross = pts[i].budget + (-d0) / (d1 - d0) * (pts[i + 1].budget - pts[i].budget); break; }
      }
      var lead = pts.filter(function (q) { return q.budget < 1; }).every(function (q) { return q.summary >= q.raw; });
      var msg = cross !== null ? ["On " + r.name + ", the exact raw buffer overtakes the summary at about ", h("b", { text: pct(cross) }), " of the budget."]
        : lead ? ["On " + r.name + ", the summary leads at every budget below the full history."]
          : ["On " + r.name + ", neither form leads throughout."];
      if (st.doc === "dense") msg.push(" With 48 facts, even the tersest summary needs 42% of the history to hold them all.");
      else msg.push(" Twelve facts in repeated filler fit in almost any summary.");
      if (r.earlierHarness) msg.push(" (Earlier harness with a shorter summary cap.)");
      if (st.doc === "dense" && r.denseNote) msg.push(" (Dense run with 4-bit weights.)");
      dek.replaceChildren.apply(dek, msg);
      var F = frame(p.chart, { height: 320, aria: "Accuracy of summary and raw buffer against budget",
        x: { domain: [0.08, 1.02], ticks: [0.1, 0.25, 0.5, 0.75, 1], fmt: pct, label: "memory budget (share of the history's tokens)", grid: true },
        y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: function (t) { return t.toFixed(2); }, label: "accuracy" } });
      if (st.doc === "dense") {
        var c = [0.1, 0.25, 0.42, 0.5, 0.75, 1].map(function (b) { return [F.X(b), F.Y(Math.min(1, b / D.denseFactShare))]; });
        line(F.g, c, "var(--gray-line)", 1.5);
      }
      [["summary", "var(--s1)"], ["raw", "var(--s2)"]].forEach(function (k) {
        var sp = pts.map(function (q) { return [F.X(q.budget), F.Y(q[k[0]])]; });
        line(F.g, sp, k[1], 2);
        sp.forEach(function (q) { dot(F.g, q[0], q[1], k[1], 4); });
      });
      crosshair(F, pts.map(function (q) { return q.budget; }), function (i) {
        var rows = [{ v: f2(pts[i].summary), l: "rolling summary", color: "var(--s1)" }, { v: f2(pts[i].raw), l: "raw buffer", color: "var(--s2)" }];
        if (st.doc === "dense") rows.push({ v: f2(Math.min(1, pts[i].budget / D.denseFactShare)), l: "ceiling", color: "var(--gray-line)" });
        return rows;
      }, function (i) { return pct(pts[i].budget) + " of the history"; });
      tt.refresh();
    }
    return render;
  }

  // =====================================================================
  // shell: tabs, routing, theme, resize
  // =====================================================================
  var VIEWS = [
    { id: "axis", title: "Same bytes, different answers", build: viewMech },
    { id: "generation", title: "Generation, not size", build: viewGen },
    { id: "question", title: "Seeing the question", build: viewQA },
    { id: "self-report", title: "What it knows it lost", build: viewAudit },
    { id: "reversibility", title: "Exact vs summarized memory", build: viewRev }
  ];
  var tablist = document.getElementById("tablist");
  var main = document.getElementById("views");
  var current = null;
  VIEWS.forEach(function (v, i) {
    v.tab = h("button", { class: "tab", role: "tab", id: "tab-" + v.id, "aria-controls": "view-" + v.id, "aria-selected": "false", tabindex: "-1" },
      h("span", { class: "n", text: String(i + 1).padStart(2, "0") }), v.title);
    v.tab.addEventListener("click", function () { go(v.id, true); });
    v.tab.addEventListener("keydown", function (e) {
      var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
      if (j === null) return;
      var nv = VIEWS[(j + VIEWS.length) % VIEWS.length];
      go(nv.id, true); nv.tab.focus();
    });
    tablist.appendChild(v.tab);
    v.el = h("section", { class: "view", id: "view-" + v.id, role: "tabpanel", "aria-labelledby": "tab-" + v.id, hidden: true });
    main.appendChild(v.el);
  });
  function go(id, push) {
    var v = VIEWS.filter(function (x) { return x.id === id; })[0] || VIEWS[0];
    VIEWS.forEach(function (x) {
      var on = x === v;
      x.tab.setAttribute("aria-selected", String(on));
      x.tab.tabIndex = on ? 0 : -1;
      x.el.hidden = !on;
    });
    hideTip();
    current = v;
    if (!v.render) v.render = v.build(v.el);
    v.render();
    if (push) { try { history.replaceState(null, "", "#" + v.id); } catch (e) { location.hash = v.id; } }
  }
  window.addEventListener("hashchange", function () { go(location.hash.slice(1), false); });

  var rt = null;
  var lastW = 0;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () {
      if (main.clientWidth !== lastW && current) { lastW = main.clientWidth; current.render(); }
    }, 120);
  });

  var tb = document.getElementById("theme-btn");
  function themeNow() {
    var t = document.documentElement.dataset.theme;
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function syncThemeBtn() { tb.textContent = themeNow() === "dark" ? "Light" : "Dark"; tb.setAttribute("aria-label", "Switch to " + tb.textContent.toLowerCase() + " theme"); }
  tb.addEventListener("click", function () {
    var next = themeNow() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("cb-theme", next); } catch (e) {}
    syncThemeBtn();
  });
  syncThemeBtn();

  var code = document.body.getAttribute("data-code-url");
  if (code) document.getElementById("code-link").setAttribute("href", code);

  var nModels = {};
  D.frontiers.forEach(function (f) { nModels[f.name] = 1; });
  var facts = document.getElementById("facts");
  [[String(D.frontiers.length), "model settings on the frontier, " + Object.keys(nModels).length + " distinct models from 0.5B to 24B"],
    ["48", "generations per cell, same needles at every budget"],
    ["5", "mechanisms priced in one unit: eviction, quantization, both, prompt compression, summaries"],
    [String(D.audit.length), "models audited against uncompacted present/absent controls"]].forEach(function (r) {
    facts.appendChild(h("li", null, h("b", { text: r[0] }), r[1]));
  });

  lastW = main.clientWidth;
  go(location.hash.slice(1), false);
})();
