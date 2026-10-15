// What each program draws as it runs on the network, for tlf.html. A run is a list of configurations of the program
// (state, head, tape), one per step, read off the network's transcript; each program turns that list into its own
// drawing: a counter's bits, a sort's passes, the values of n, a fractal's turns, a busy beaver's tape through time, the
// sieve's cells on an Ulam spiral. The same functions draw a direct run of the rules, for the gallery.
(function (root) {
  "use strict";
  var PR = root.TCT_PROGRAMS || (typeof require !== "undefined" && require("./programs.js"));
  var CU = root.TCT_CURVES || (typeof require !== "undefined" && require("./curves.js"));
  var PRI = root.TCT_PRIMES || (typeof require !== "undefined" && require("./primes.js"));
  var INK = "#0B1F33", ACC = "#F05A1A", PALE = "#E2E7EA", MUT = "#5A6A78", RULE = "#D3DADE", TINT = "#FDEBE2";
  var prog = function (id) { return PR.programs.filter(function (p) { return p.id === id; })[0]; };
  var curve = function (id) { return CU.curves.filter(function (c) { return c.id === id; })[0]; };
  function f(x) { return +x.toFixed(1); }
  function svg(w, h, body) { return "<svg viewBox='0 0 " + w + " " + h + "' preserveAspectRatio='xMidYMid meet'>" + body + "</svg>"; }
  function rect(x, y, w, h, c) { return "<rect x='" + f(x) + "' y='" + f(y) + "' width='" + f(w) + "' height='" + f(h) + "' fill='" + c + "'/>"; }
  function text(x, y, s, c, size, anchor) { return "<text x='" + f(x) + "' y='" + f(y) + "' font-family='IBM Plex Mono' font-size='" + (size || 11) + "' fill='" + (c || MUT) + "'" + (anchor ? " text-anchor='" + anchor + "'" : "") + ">" + s + "</text>"; }
  // a grid of cells, drawn as horizontal runs of one colour: grid[r][c] is a colour or null
  function runs(grid, x0, y0, cw, rh, gap) {
    var b = "";
    grid.forEach(function (row, r) {
      var c = 0;
      while (c < row.length) { var col = row[c], e = c; while (e < row.length && row[e] === col) e++; if (col) b += rect(x0 + c * cw, y0 + r * rh, (e - c) * cw, rh - gap, col); c = e; }
    });
    return b;
  }
  var comma = function (n) { return n.toLocaleString("en-US"); };
  function bits(n, k) { var s = ""; for (var i = k - 1; i >= 0; i--) s += (n >> i) & 1; return s; }
  // the values a program announces, from its configurations (programs.js value())
  function valuesOf(p, cfgs) { var out = []; cfgs.forEach(function (c) { var v = p.value(c); if (v !== null) out.push(v); }); return out; }
  var W = 600, H = 260;

  function bitGrid(vals, k, ncols, y0, h, x0) {
    x0 = x0 || 0;
    var cw = (W - x0) / ncols, rh = h / k;
    var grid = []; for (var j = 0; j < k; j++) grid.push(vals.map(function (n, i) { return (n >> j) & 1 ? (i === vals.length - 1 ? ACC : INK) : null; }));
    return runs(grid, x0, y0, cw, rh, Math.min(3, rh * 0.15));
  }

  var SPECS = {
    counter: {
      param: { label: "Bits", value: 8, min: 1, max: 10 }, input: function (k) { return prog("counter").input(k); },
      draw: function (cfgs, k, done) {
        var v = valuesOf(prog("counter"), cfgs), last = v[v.length - 1];
        return { svg: svg(W, H, bitGrid(v, k, Math.pow(2, k), 0, H)),
                 big: last === undefined ? "–" : comma(last), sub: last === undefined ? "" : "= " + bits(last, k) + " · every count so far, one column each, lowest bit on top" };
      }
    },
    sort: {
      param: { label: "String", value: "1011001110100110", pattern: "^[01]{1,24}$", hint: "1 to 24 digits of 0 and 1" }, input: function (s) { return prog("sort").input(s); },
      draw: function (cfgs, s, done, total, finalVals) {
        var v = valuesOf(prog("sort"), cfgs), n = s.length, rows = Math.max(finalVals ? finalVals.length : 1, v.length), cw = W / n, rh = Math.min(16, H / rows), b = "";
        var grid = v.map(function (row, r) { return row.split("").map(function (c) { return c === "1" ? (r === v.length - 1 ? ACC : INK) : PALE; }); });
        b += runs(grid, 0, 0, cw, rh, Math.min(2, rh * 0.25));
        for (var i = 1; i < n; i++) b += "<line x1='" + f(i * cw) + "' x2='" + f(i * cw) + "' y1='0' y2='" + f(rows * rh) + "' stroke='#FFFFFF' stroke-width='2'/>";
        return { svg: svg(W, Math.max(40, rows * rh), b), big: v.length ? v[v.length - 1] : "–", sub: comma(Math.max(0, v.length - 1)) + " swaps so far · each row is the string at the start of a scan" };
      }
    },
    count: {
      param: { label: "Ones", value: 40, min: 1, max: 80 }, input: function (n) { return prog("count").input(n); },
      draw: function (cfgs, n, done) {
        var v = valuesOf(prog("count"), cfgs), k = Math.max(1, Math.ceil(Math.log2(n + 1))), cw = W / n, b = "";
        for (var i = 0; i < n; i++) {
          var crossed = i < v.length - 1;
          b += rect(i * cw + 1, 0, cw - 2, 20, crossed ? PALE : INK);
          if (crossed) b += "<line x1='" + f(i * cw + 2) + "' y1='18' x2='" + f(i * cw + cw - 2) + "' y2='2' stroke='" + MUT + "' stroke-width='1.2'/>";
        }
        b += bitGrid(v.slice(1), k, n, 36, H - 36);
        var last = v[v.length - 1];
        return { svg: svg(W, H, b), big: last === undefined ? "–" : comma(last), sub: last === undefined ? "" : "= " + (last ? last.toString(2) : "0") + " · top, the ones crossed off; below, the counter after each" };
      }
    },
    add: {
      param: { label: "a + b", value: "1000+99", pattern: "^\\d{1,5}\\+\\d{1,3}$", hint: "like 1000+99: a up to 5 digits, b up to 3" }, input: function (s) { return prog("add").input(s); },
      draw: function (cfgs, s, done) {
        var v = valuesOf(prog("add"), cfgs), p = s.split("+"), A = +p[0], B = +p[1], ka = Math.max(1, Math.ceil(Math.log2(A + B + 1))), kb = Math.max(1, Math.ceil(Math.log2(B + 1)));
        var cols = B + 1, x0 = 24, rh = (H - 16) / (ka + kb), b = "";
        b += bitGrid(v.map(function (q) { return q[1]; }), kb, cols, 0, kb * rh, x0);
        b += bitGrid(v.map(function (q) { return q[0]; }), ka, cols, kb * rh + 16, ka * rh, x0);
        b += text(0, kb * rh / 2 + 4, "b", MUT, 13) + text(0, kb * rh + 16 + ka * rh / 2 + 4, "a", MUT, 13);
        var last = v[v.length - 1];
        return { svg: svg(W, H, b), big: last ? comma(last[0]) + " + " + comma(last[1]) : "–", sub: "a and b now · the bits of b counting down, above the bits of a counting up" };
      }
    },
    collatz: {
      param: { label: "Start from n =", value: 27, min: 1, max: 999 }, input: function (n) { return prog("collatz").input(n); },
      draw: function (cfgs, n, done, total, finalVals) {
        var v = valuesOf(prog("collatz"), cfgs), all = finalVals || v, mx = Math.max.apply(null, all.concat([2])), N = Math.max(all.length, 2);
        var X = function (i) { return 6 + i / (N - 1) * (W - 70); }, Y = function (x) { return H - 6 - x / mx * (H - 24); }, b = "";
        var step = Math.pow(10, Math.floor(Math.log10(mx))), g = mx / step > 5 ? step * 2 : mx / step > 2.5 ? step : step / 2;
        for (var t = 0; t <= mx; t += g) b += "<line x1='0' x2='" + (W - 58) + "' y1='" + f(Y(t)) + "' y2='" + f(Y(t)) + "' stroke='" + RULE + "'/>" + text(W, Y(t) + 4, comma(t), MUT, 11, "end");
        if (v.length > 1) b += "<polyline fill='none' stroke='" + INK + "' stroke-width='2' stroke-linejoin='round' points='" + v.map(function (x, i) { return f(X(i)) + "," + f(Y(x)); }).join(" ") + "'/>";
        if (v.length) { var pk = v.indexOf(Math.max.apply(null, v)); b += "<circle cx='" + f(X(pk)) + "' cy='" + f(Y(v[pk])) + "' r='4' fill='" + INK + "'/>"; b += "<circle cx='" + f(X(v.length - 1)) + "' cy='" + f(Y(v[v.length - 1])) + "' r='6' fill='" + ACC + "'/>"; }
        var last = v[v.length - 1], hi = v.length ? Math.max.apply(null, v) : 0;
        return { svg: svg(W, H, b), big: last === undefined ? "–" : "n = " + comma(last), sub: v.length ? comma(v.length - 1) + " halvings and 3n + 1s so far; the highest, " + comma(hi) : "" };
      }
    },
    fib: {
      param: { label: "Bits", value: 8, min: 4, max: 24 }, input: function (w) { return prog("fib").input(w); },
      draw: function (cfgs, w, done, total, finalVals) {
        var v = valuesOf(prog("fib"), cfgs).filter(function (x) { return x > 0; }), all = (finalVals || v).filter(function (x) { return x > 0; });
        function tile(vals) {
          var x0 = 0, y0 = 0, x1 = 1, y1 = 1, sq = vals.length ? [[0, 0, 1]] : [];
          for (var i = 1; i < vals.length; i++) {
            var s = vals[i], d = i % 4, x, y;
            if (d === 1) { x = x1; y = y0; } else if (d === 2) { x = x1 - s; y = y1; } else if (d === 3) { x = x0 - s; y = y1 - s; } else { x = x0; y = y0 - s; }
            sq.push([x, y, s]); x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + s); y1 = Math.max(y1, y + s);
          }
          return { sq: sq, box: [x0, y0, x1, y1] };
        }
        var T = tile(v), F = tile(all), bx = F.box, k = Math.min(W / (bx[2] - bx[0]), H / (bx[3] - bx[1])), ox = (W - k * (bx[2] - bx[0])) / 2, oy = (H - k * (bx[3] - bx[1])) / 2, b = "";
        T.sq.forEach(function (q, i) {
          var last = i === T.sq.length - 1, s = q[2] * k, x = ox + (q[0] - bx[0]) * k, y = oy + (q[1] - bx[1]) * k;
          b += "<rect x='" + f(x) + "' y='" + f(y) + "' width='" + f(s) + "' height='" + f(s) + "' fill='" + (last ? TINT : "#FFFFFF") + "' stroke='" + INK + "' stroke-width='1'/>";
          if (s > 24) b += text(x + s / 2, y + s / 2 + 4, comma(q[2]), last ? ACC : MUT, s > 70 ? 14 : 10, "middle");
        });
        var last = v[v.length - 1];
        return { svg: svg(W, H, b), big: last === undefined ? "–" : comma(last), sub: "the latest Fibonacci number · a square for each, tiled into the spiral" };
      }
    }
  };
  // the curves: turns announced by the program, drawn as a path
  function curveSpec(id, def, min, max) {
    return {
      param: { label: "Bits", value: def, min: min, max: max }, input: function (k) { return "_" + new Array(k + 1).join("0"); },
      draw: function (cfgs, k, done, total, finalVals) {
        var c = curve(id), vals = []; cfgs.forEach(function (g) { var x = c.emit(g.state); if (x !== null) vals.push(x); });
        var pts = c.path(vals), all = c.path(finalVals || vals), xs = all.map(function (p) { return p[0]; }), ys = all.map(function (p) { return p[1]; });
        var mnx = Math.min.apply(null, xs), mxx = Math.max.apply(null, xs), mny = Math.min.apply(null, ys), mxy = Math.max.apply(null, ys);
        var s = Math.min((W - 12) / (mxx - mnx || 1), (H - 12) / (mxy - mny || 1)), ox = (W - s * (mxx - mnx)) / 2, oy = (H - s * (mxy - mny)) / 2;
        var P = pts.map(function (p) { return [ox + (p[0] - mnx) * s, oy + (p[1] - mny) * s]; }), e = P[P.length - 1];
        var b = "<polyline fill='none' stroke='" + INK + "' stroke-width='" + (all.length > 600 ? 1.2 : 1.8) + "' stroke-linejoin='round' points='" + P.map(function (p) { return f(p[0]) + "," + f(p[1]); }).join(" ") + "'/>" +
                "<circle cx='" + f(e[0]) + "' cy='" + f(e[1]) + "' r='5' fill='" + ACC + "'/>";
        return { svg: svg(W, H, b), big: vals.length ? c.label(vals[vals.length - 1]) : "–", sub: comma(vals.length) + " turns read off the tape so far" };
      },
      finalOf: function (cfgs) { var c = curve(id), out = []; cfgs.forEach(function (g) { var x = c.emit(g.state); if (x !== null) out.push(x); }); return out; }
    };
  }
  ["collatz", "fib", "sort"].forEach(function (id) { SPECS[id].finalOf = function (cfgs) { return valuesOf(prog(id), cfgs); }; });
  SPECS.dragon = curveSpec("dragon", 8, 2, 11);
  SPECS.koch = curveSpec("koch", 8, 2, 11);
  SPECS.fibword = curveSpec("fibword", 11, 4, 16);
  // the busy beaver: the champion after a prelude that walks right over `pad` 0s, so it has room on the one-way tape
  var CHAMPS = {
    bb4: { name: "4 states", pad: 10, rules: ["A _ -> B 1 R", "A 1 -> B 1 L", "B _ -> A 1 L", "B 1 -> C _ L", "C _ -> qa 1 R", "C 1 -> D 1 L", "D _ -> D 1 R", "D 1 -> A _ R"] },
    bb3: { name: "3 states", pad: 4, rules: ["A _ -> B 1 R", "A 1 -> qa 1 R", "B _ -> B 1 L", "B 1 -> C _ R", "C _ -> C 1 L", "C 1 -> A 1 L"] },
    bb2: { name: "2 states", pad: 2, rules: ["A _ -> B 1 R", "A 1 -> B 1 L", "B _ -> A 1 L", "B 1 -> qa 1 R"] }
  };
  var PRELUDE = ["w 0 -> w _ R", "w _ -> A _ S"];
  SPECS.bb4 = {
    param: { label: "Champion", value: "bb4", options: [["bb4", "4 states"], ["bb3", "3 states"], ["bb2", "2 states"]] },
    start: "w", rules: function (c) { return PRELUDE.concat(CHAMPS[c].rules); }, input: function (c) { return new Array(CHAMPS[c].pad + 1).join("0"); },
    draw: function (cfgs, c, done, total) {
      var pad = CHAMPS[c].pad, st = cfgs.slice(pad + 1), n = Math.max(1, (total || cfgs.length) - pad - 1), width = 1;
      cfgs.forEach(function (g) { width = Math.max(width, g.tape.length, (g.head || 0) + 1); });
      var cw = W / n, ch = H / width, grid = [];
      for (var i = 0; i < width; i++) grid.push(st.map(function (g) { return g.head === i ? ACC : g.tape[i] === "1" ? INK : null; }));
      var ones = st.length ? (st[st.length - 1].tape.match(/1/g) || []).length : 0;
      return { svg: svg(W, H, runs(grid, 0, 0, cw, ch, Math.min(1.5, ch * 0.15))), big: comma(ones) + " ones", sub: comma(Math.max(0, st.length - 1)) + " steps of the champion · time runs left to right, the head in orange" };
    }
  };
  // the sieve: cells 1 .. N on an Ulam spiral
  var SIZES = [[11, 120], [23, 528], [31, 960], [41, 1680]];
  function spiral(side) {
    var N = side * side - 1, pos = [null, [0, 0]], x = 0, y = 0, n = 1, step = 1, di = 0, dir = [[1, 0], [0, -1], [-1, 0], [0, 1]];
    while (n < N) { for (var r = 0; r < 2; r++) { for (var s = 0; s < step && n < N; s++) { x += dir[di][0]; y += dir[di][1]; n++; pos[n] = [x, y]; } di = (di + 1) % 4; } step++; }
    return pos;
  }
  SPECS.sieve = {
    param: { label: "Up to", value: "120", options: SIZES.map(function (s) { return [String(s[1]), comma(s[1])]; }) },
    big: true, start: "s",
    rules: function (N) { return PRI.sieve(+N).rules; }, input: function (N) { return PRI.sieve(+N).tape; },
    draw: function (cfgs, N, done) {
      N = +N; var side = SIZES.filter(function (s) { return s[1] === N; })[0][0], pos = spiral(side), g = cfgs[cfgs.length - 1] || { tape: "", head: 0, state: "s" };
      var m = /^[kmr](\d+)/.exec(g.state || ""), k = m ? +m[1] : 0, c = Math.min(W, H) / side, ox = W / 2 - c / 2, oy = H / 2 - c / 2, b = "", crossed = 0, known = 0;
      for (var n = 1; n <= N; n++) {
        var t = g.tape[n] || "1", col;
        if (n === 1 || t === "0") { col = PALE; if (n > 1) crossed++; }
        else if (t === "_" || n === k) col = ACC;
        else if (done || n <= k) { col = INK; known++; }
        else col = "#8E9DAA";
        b += rect(ox + pos[n][0] * c + 1, oy + pos[n][1] * c + 1, c - 2, c - 2, col);
        if (n === g.head) b += "<rect x='" + f(ox + pos[n][0] * c) + "' y='" + f(oy + pos[n][1] * c) + "' width='" + f(c) + "' height='" + f(c) + "' fill='none' stroke='" + ACC + "' stroke-width='2'/>";
      }
      var say = done ? "done" : /^k/.test(g.state) ? "is " + k + " prime?" : /^m/.test(g.state) ? "crossing out multiples of " + k : /^r/.test(g.state) ? "walking back to " + k : "starting";
      return { svg: svg(W, H, b), big: done ? comma(N - 1 - crossed) + " primes" : comma(crossed) + " crossed out",
               sub: done ? "the cells left standing are the primes up to " + comma(N) + ", on an Ulam spiral" : say + " · dark, prime so far; gray, not yet reached; orange, the number being sieved" };
    }
  };
  // a direct run of the rules: every configuration (state, head, tape without its trailing blanks)
  function direct(start, rules, tape, limit) {
    var m = {}; rules.forEach(function (r) { var a = r.split(" "); m[a[0] + " " + a[1]] = a; });
    var t = tape.split(""), h = 0, q = start, out = [{ state: q, head: h, tape: tape.replace(/_+$/, "") }];
    while (q !== "qa" && q !== "qr" && out.length < (limit || 2e6)) {
      var r = m[q + " " + (t[h] || "_")]; if (!r) { q = "qr"; out.push({ state: q, head: h, tape: t.join("").replace(/_+$/, ""), final: true }); break; }
      t[h] = r[4]; h = Math.max(0, h + (r[5] === "L" ? -1 : r[5] === "R" ? 1 : 0)); q = r[3];
      out.push({ state: q, head: h, tape: t.join("").replace(/_+$/, "") });
    }
    out[out.length - 1].final = true;
    return out;
  }
  // a program's machine and input, for a parameter
  function setup(id, item, param) {
    var s = SPECS[id];
    return { start: s.start || item.src.start, rules: s.rules ? s.rules(param) : item.src.rules, tape: s.input(param) };
  }
  function parse(id, raw) {
    var p = SPECS[id].param;
    if (p.options) return raw;
    if (p.pattern) { if (!new RegExp(p.pattern).test(raw)) throw new Error(p.hint); return raw; }
    var n = Number(raw); if (!(Number.isInteger(n) && n >= p.min && n <= p.max)) throw new Error("a whole number from " + p.min + " to " + p.max);
    return n;
  }
  // a program's input, said briefly: "8 bits", "from n = 27", "to 120"
  var SHOW = { counter: function (k) { return k + " bits"; }, sort: function (s) { return s; }, count: function (n) { return n + " ones"; },
    add: function (s) { return s.replace("+", " + "); }, collatz: function (n) { return "from n = " + n; }, fib: function (k) { return k + " bits"; },
    dragon: function (k) { return k + " bits"; }, koch: function (k) { return k + " bits"; }, fibword: function (k) { return k + " bits"; },
    bb4: function (c) { return CHAMPS[c].name; }, sieve: function (N) { return "to " + comma(+N); } };
  function show(id, param) { return SHOW[id](param); }
  var api = { SPECS: SPECS, direct: direct, setup: setup, parse: parse, show: show, CHAMPS: CHAMPS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TCT_RUNVIZ = api;
})(this);
