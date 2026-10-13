// A network trained in this browser (train.html), kept in localStorage. When the viewer has chosen to use it, this
// script swaps it into window.TCT_DATA before the page builds its engine, and says so at the top of the page.
// Load it right after data.js.
(function (root) {
  "use strict";
  var KEY = "tct-mynet", USE = "tct-use-mine";
  function get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function set(k, v) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); return true; } catch (e) { return false; } }
  function enc(f32) { var u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength), s = ""; for (var i = 0; i < u8.length; i += 8192) s += String.fromCharCode.apply(null, u8.subarray(i, i + 8192)); return btoa(s); }
  function dec(b64) { var s = atob(b64), u8 = new Uint8Array(s.length); for (var i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i); return new Float32Array(u8.buffer); }

  var api = {
    // rec: {meta, off (TrainCore layout offsets), params (Float32Array), vinFull, voutFull, train (window strings)}
    save: function (rec) {
      var stored = { v: 1, meta: rec.meta, off: rec.off, params: enc(rec.params), vinFull: rec.vinFull, voutFull: rec.voutFull, train: rec.train };
      return set(KEY, JSON.stringify(stored));
    },
    load: function () {
      var s = get(KEY); if (!s) return null;
      try { var r = JSON.parse(s); r.params = dec(r.params); return r; } catch (e) { return null; }
    },
    active: function () { return get(USE) === "1" && !!get(KEY); },
    setActive: function (on) { return set(USE, on ? "1" : null); },
    clear: function () { set(USE, null); set(KEY, null); },
    // The weights in data.js's format over the full vocabularies (unused tokens: zero embedding; unused outputs: -1e4).
    weights: function (r, nIn, nOut) {
      var o = r.off, p = r.params, w = {};
      function rows(name) { var t = o[name], out = []; for (var i = 0; i < t.rows; i++) out.push(Array.from(p.subarray(t.at + i * t.cols, t.at + (i + 1) * t.cols))); return out; }
      function zeros(n, m) { var out = []; for (var i = 0; i < n; i++) out.push(new Array(m).fill(0)); return out; }
      var E = zeros(nIn, 64), U = zeros(nOut, 64), bU = new Array(nOut).fill(-1e4), e = rows("E"), u = rows("U"), b = rows("bU")[0];
      r.vinFull.forEach(function (full, i) { E[full] = e[i]; });
      r.voutFull.forEach(function (full, i) { U[full] = u[i]; bU[full] = b[i]; });
      w["E.weight"] = E; w.P = rows("P");
      ["Wq", "Wk", "Wv", "Wo", "W1", "W2"].forEach(function (n) { w[n + ".weight"] = rows(n); });
      w["W1.bias"] = rows("b1"); w["W2.bias"] = rows("b2"); w["U.weight"] = U; w["U.bias"] = [bU];
      return w;
    },
    applied: null
  };
  root.MyNet = api;

  var D = root.TCT_DATA;
  if (D && api.active() && !/train\.html/.test(location.pathname)) {
    var r = api.load();
    if (r) {
      D.weights = api.weights(r, D.vin.length, D.vout.length);
      D.train = r.train;
      D.training = { random_machines: r.meta.machines, windows: r.train.length };
      api.applied = r.meta;
      document.addEventListener("DOMContentLoaded", function () {
        var main = document.querySelector("main"); if (!main) return;
        var bar = document.createElement("div"); bar.className = "mynet-bar"; bar.setAttribute("role", "status");
        bar.innerHTML = "<span>Running <b>your network</b>, trained in this browser on the first " + r.meta.machines.toLocaleString() +
          " random machines (seed " + r.meta.seed + ", " + r.meta.steps.toLocaleString() + " Adam steps), not the committed one.</span>";
        var btn = document.createElement("button"); btn.type = "button"; btn.className = "btn ghost"; btn.textContent = "Use the committed network";
        btn.addEventListener("click", function () { api.setActive(false); location.reload(); });
        bar.appendChild(btn);
        main.insertBefore(bar, main.firstChild);
      });
    }
  }
})(this);
