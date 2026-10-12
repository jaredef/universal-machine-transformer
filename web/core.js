// The turing-complete transformer, in the browser: the universal machine (a port of machines/lag3/derived/utm.py),
// the lagging-window decoder, and lag3's trained network as its step. No DOM here, so it also runs under Node for checks.
(function (root) {
  "use strict";

  var B = "#", SEP = ":";
  var SIM = ["0", "1", "_"];
  var MARK = { "0": "O", "1": "I", "_": "U" }, UNMARK = { O: "0", I: "1", U: "_" };
  var WLET = { "0": "a", "1": "b", "_": "c" }, LETW = { a: "0", b: "1", c: "_" };
  var LOW = { L: "l", R: "r", S: "s" };
  var ALL = ["|", "*", ";", "a", "b", "c", "L", "R", "S", "l", "r", "s", "x", "y", "A", "H", "$",
             "0", "1", "_", "O", "I", "U"];

  // ---------------------------------------------------------------- the universal machine (utm.py)

  function utmRules() {
    var R = [];
    function add(s, r, n, w, m) { R.push([s, r, n, w, m]); }
    function scan(state, move, stop) { ALL.forEach(function (s) { if (stop.indexOf(s) < 0) add(state, s, state, s, move); }); }
    scan("find", "R", ["O", "I", "U"]);
    SIM.forEach(function (s, i) { add("find", MARK[s], "back" + i, MARK[s], "L"); });
    for (var i = 0; i < 3; i++) { scan("back" + i, "L", ["*"]); add("back" + i, "*", "at" + i, "*", "R"); }
    [1, 2].forEach(function (k) { scan("at" + k, "R", [";"]); add("at" + k, ";", "at" + (k - 1), ";", "R"); });
    "abc".split("").forEach(function (w) {
      add("at0", w, "rd_" + w, w, "R");
      "LRS".split("").forEach(function (d) {
        add("rd_" + w, d, "go_" + w + d, LOW[d], "R");
        scan("go_" + w + d, "R", ["O", "I", "U"]);
        SIM.forEach(function (s) {
          var sym = LETW[w];
          if (d === "S") add("go_" + w + d, MARK[s], "ret", MARK[sym], "L");
          else add("go_" + w + d, MARK[s], d === "R" ? "markR" : "markL", sym, d);
        });
      });
    });
    SIM.forEach(function (s) { add("markR", s, "ret", MARK[s], "L"); add("markL", s, "ret", MARK[s], "L"); });
    add("markL", "$", "markR", "$", "R");
    scan("ret", "L", ["l", "r", "s"]);
    "lrs".split("").forEach(function (d) { add("ret", d, "nx", d, "R"); });
    add("nx", "A", "qa", "A", "S"); add("nx", "x", "clr", "x", "L"); add("nx", ";", "clr", ";", "L");
    scan("clr", "L", ["*"]); add("clr", "*", "toL", "|", "L");
    scan("toL", "L", ["_"]); add("toL", "_", "set0", "_", "R");
    add("set0", "|", "cnt", "*", "R");
    scan("cnt", "R", ["l", "r", "s"]);
    "lrs".split("").forEach(function (d) { add("cnt", d, "cx", d, "R"); });
    add("cx", "y", "cx", "y", "R"); add("cx", "x", "adv", "y", "L"); add("cx", ";", "fin", ";", "L");
    scan("adv", "L", ["_"]); add("adv", "_", "adv2", "_", "R");
    scan("adv2", "R", ["*"]); add("adv2", "*", "adv3", "|", "R");
    scan("adv3", "R", ["|"]); add("adv3", "|", "tol2", "*", "L");
    scan("tol2", "L", ["_"]); add("tol2", "_", "cnt", "_", "R");
    add("fin", "y", "fin", "x", "L");
    "LRS".split("").forEach(function (d) { add("fin", LOW[d], "find", d, "R"); });
    return R;
  }
  var UTM = {}, UTM_RULES = utmRules();
  UTM_RULES.forEach(function (r) { UTM[r[0] + "\u0000" + r[1]] = { next: r[2], write: r[3], move: r[4] }; });

  // What the universal machine is doing, in words, by state.
  function phase(state) {
    if (state === "find") return "finding the described machine's head";
    if (/^back/.test(state)) return "carrying the symbol it read back to the current state";
    if (/^at/.test(state)) return "finding the rule for that symbol";
    if (/^rd_/.test(state)) return "reading the rule";
    if (/^go_/.test(state)) return "carrying the write and the move to the head";
    if (/^mark/.test(state)) return "moving the described machine's head";
    if (state === "ret" || state === "nx") return "going back to read the next state";
    if (["clr", "toL", "set0", "cnt", "cx", "adv", "adv2", "adv3", "tol2"].indexOf(state) >= 0) return "moving * to the next state, counting the x's";
    if (state === "fin") return "tidying up the rule it used";
    if (state === "qa") return "the described machine accepted";
    return state;
  }

  // ---------------------------------------------------------------- described machines

  function parseMachine(start, lines) {
    var rules = {}, states = {};
    lines.forEach(function (raw, n) {
      var line = raw.split("#")[0].trim();
      if (!line) return;
      var p = line.split(/\s+/);
      if (p.length !== 6 || p[2] !== "->") throw new Error("line " + (n + 1) + ": write  STATE READ -> NEXT WRITE MOVE");
      if (SIM.indexOf(p[1]) < 0 || SIM.indexOf(p[4]) < 0) throw new Error("line " + (n + 1) + ": symbols must be 0, 1, or _");
      if ("LRS".indexOf(p[5]) < 0 || p[5].length !== 1) throw new Error("line " + (n + 1) + ": the move must be L, R, or S");
      if (/[:#\s]/.test(p[0] + p[3])) throw new Error("line " + (n + 1) + ": state names cannot contain : or #");
      if (rules[p[0] + "\u0000" + p[1]]) throw new Error("line " + (n + 1) + ": a second rule for " + p[0] + " reading " + p[1]);
      rules[p[0] + "\u0000" + p[1]] = { next: p[3], write: p[4], move: p[5], line: n };
      states[p[0]] = 1; states[p[3]] = 1;
    });
    var named = Object.keys(states).filter(function (s) { return s !== start && s !== "qa" && s !== "qr"; }).sort();
    return { start: start, accept: "qa", reject: "qr", rules: rules, order: [start].concat(named) };
  }

  function encode(m, tape) {
    var idx = {};
    m.order.forEach(function (s, i) { idx[s] = i; });
    var entries = m.order.map(function (q) {
      return SIM.map(function (sym) {
        var r = m.rules[q + "\u0000" + sym];
        if (!r) return "H;";
        var target = r.next === "qa" ? "A" : r.next === "qr" ? "H" : "x".repeat(idx[r.next]);
        return WLET[r.write] + r.move + target + ";";
      }).join("");
    });
    var cells = tape.split("");
    if (!cells.length) cells = ["_"];
    cells[0] = MARK[cells[0]];
    return "*" + entries.join("|") + "$" + cells.join("");
  }

  function reference(m, tape, maxSteps) {
    var t = {}, head = 0, state = m.start;
    tape.split("").forEach(function (s, i) { if (s !== "_") t[i] = s; });
    for (var n = 0; n < (maxSteps || 100000); n++) {
      if (state === "qa") return { verdict: "ACCEPT", tape: fmt(t), steps: n };
      if (state === "qr") return { verdict: "REJECT", tape: fmt(t), steps: n };
      var r = m.rules[state + "\u0000" + (t[head] || "_")];
      if (!r) return { verdict: "REJECT", tape: fmt(t), steps: n };
      if (r.write === "_") delete t[head]; else t[head] = r.write;
      state = r.next;
      head = Math.max(0, head + (r.move === "L" ? -1 : r.move === "R" ? 1 : 0));
    }
    return { verdict: "TIMEOUT", tape: fmt(t), steps: n };
  }
  function fmt(t) {
    var keys = Object.keys(t).map(Number);
    if (!keys.length) return "_";
    var hi = Math.max.apply(null, keys), s = "";
    for (var i = 0; i <= hi; i++) s += t[i] || "_";
    return s;
  }

  // ---------------------------------------------------------------- the network (lag3's trained transformer)

  function Network(data) {
    var w = data.weights;
    this.E = w["E.weight"]; this.P = w.P;
    this.Wq = w["Wq.weight"]; this.Wk = w["Wk.weight"]; this.Wv = w["Wv.weight"]; this.Wo = w["Wo.weight"];
    this.W1 = w["W1.weight"]; this.b1 = w["W1.bias"][0]; this.W2 = w["W2.weight"]; this.b2 = w["W2.bias"][0];
    this.U = w["U.weight"]; this.bU = w["U.bias"][0];
    this.d = this.E[0].length; this.h = data.heads;
  }
  function lin(W, x, b) {
    var out = new Float64Array(W.length);
    for (var i = 0; i < W.length; i++) {
      var row = W[i], s = b ? b[i] : 0;
      for (var j = 0; j < row.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }
  // ids: three input token ids. Returns {best, logits, attention: [head][position]}.
  Network.prototype.forward = function (ids) {
    var d = this.d, dh = d / this.h, self = this;
    var X = ids.map(function (id, i) { var e = self.E[id], p = self.P[i], v = new Float64Array(d); for (var c = 0; c < d; c++) v[c] = e[c] + p[c]; return v; });
    var q = lin(this.Wq, X[2]), K = X.map(function (x) { return lin(self.Wk, x); }), V = X.map(function (x) { return lin(self.Wv, x); });
    var y = new Float64Array(d), attention = [];
    for (var hd = 0; hd < this.h; hd++) {
      var lo = hd * dh, sc = [0, 0, 0];
      for (var j = 0; j < 3; j++) { var s = 0; for (var c = lo; c < lo + dh; c++) s += q[c] * K[j][c]; sc[j] = s / Math.sqrt(dh); }
      var m = Math.max(sc[0], sc[1], sc[2]), ex = sc.map(function (v) { return Math.exp(v - m); }), z = ex[0] + ex[1] + ex[2];
      var a = ex.map(function (v) { return v / z; });
      attention.push(a);
      for (var j2 = 0; j2 < 3; j2++) for (var c2 = lo; c2 < lo + dh; c2++) y[c2] += a[j2] * V[j2][c2];
    }
    var o = lin(this.Wo, y), h = new Float64Array(d);
    for (var c3 = 0; c3 < d; c3++) h[c3] = X[2][c3] + o[c3];
    var hid = lin(this.W1, h, this.b1);
    for (var u = 0; u < hid.length; u++) if (hid[u] < 0) hid[u] = 0;
    var o2 = lin(this.W2, hid, this.b2);
    for (var c4 = 0; c4 < d; c4++) h[c4] += o2[c4];
    var logits = lin(this.U, h, this.bU), best = 0;
    for (var k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    return { best: best, logits: logits, attention: attention };
  };

  // ---------------------------------------------------------------- the lagging-window decoder

  function Engine(data) {
    this.data = data;
    this.net = new Network(data);
    this.vin = data.vin; this.vout = data.vout;
    this.id = {}; var self = this;
    this.vin.forEach(function (t, i) { self.id[t] = i; });
    this.K = this.vin.length;
    this.emitIds = this.vout.map(function (n) { return data.emits[n].map(function (t) { return self.id[t]; }); });
    this.memo = new Map();                                 // window key -> vout index (the network's choice)
    this.exactMemo = new Map();                            // window key -> exact output, as a string
    this.trained = new Set(data.train.map(function (w) { return self.key(w.split("\u0001").map(function (t) { return self.id[t]; })); }));
    this.headInfo = this.vin.map(function (t) {
      if (t === B || t.indexOf(SEP) < 0) return null;
      var i = t.indexOf(SEP); return { state: t.slice(0, i), sym: t.slice(i + 1) };
    });
  }
  Engine.prototype.key = function (ids) { return (ids[0] * this.K + ids[1]) * this.K + ids[2]; };
  Engine.prototype.stepOf = function (id) {
    var h = this.headInfo[id];
    if (!h || h.state === "qa" || h.state === "qr") return null;
    return UTM[h.state + "\u0000" + h.sym] || null;
  };
  Engine.prototype.exact = function (ids) {                // lag0.rule for the universal machine
    var l = this.vin[ids[0]], c = this.vin[ids[1]], r = this.vin[ids[2]], t, out;
    if (c === B) { t = this.stepOf(ids[2]); out = t && t.move === "L" ? [B, t.next + SEP + "_"] : [B]; }
    else if (this.headInfo[ids[1]]) {
      t = this.stepOf(ids[1]);
      if (!t) out = [];
      else if (t.move === "S") out = [t.next + SEP + t.write];
      else if (t.move === "R" && r === B) out = [t.write, t.next + SEP + "_"];
      else out = [t.write];
    } else {
      var tl = this.stepOf(ids[0]), tr = this.stepOf(ids[2]);
      out = tl && tl.move === "R" ? [tl.next + SEP + c] : tr && tr.move === "L" ? [tr.next + SEP + c] : [c];
    }
    return out.join(" ");
  };

  // A run of one described machine on one input.
  function Run(engine, machine, tape) {
    this.e = engine; this.machine = machine; this.input = tape;
    this.encoded = encode(machine, tape);
    var cells = this.encoded.split(""), self = this;
    cells[0] = "find" + SEP + cells[0];
    var ring = [B].concat(cells);
    this.cap = 1 << 16; this.T = new Int32Array(this.cap); this.n = 0;
    this.push(engine.id[ring[ring.length - 1]]);
    ring.forEach(function (t) { self.push(engine.id[t]); });
    this.k = 0; this.passes = 0; this.leftGrown = 0;
    this.ringStart = null;                              // where the ring now being written starts
    this.config = this.parse(ring);                    // the latest complete configuration of the universal machine
    this.utmSteps = 0; this.simSteps = 0; this.prevState = "find";
    this.seen = new Set(); this.unseen = 0; this.disagree = 0;
    this.verdict = null; this.last = null;
    this.reference = reference(machine, tape);
    this.history = [{ step: 0, tape: this.config.simTape.slice(), head: this.config.simHead, state: machine.start }];
  }
  Run.prototype.push = function (id) {
    if (this.n === this.cap) { var bigger = new Int32Array(this.cap *= 2); bigger.set(this.T); this.T = bigger; }
    this.T[this.n++] = id;
  };
  Run.prototype.window = function () { return [this.T[this.k], this.T[this.k + 1], this.T[this.k + 2]]; };

  // One forward pass of the network: read the window, append its output, advance.
  Run.prototype.step = function (detail) {
    if (this.verdict) return false;
    var e = this.e, ids = this.window(), key = e.key(ids), best = e.memo.get(key), fwd = null;
    if (best === undefined || detail) {
      fwd = e.net.forward(ids); best = fwd.best; e.memo.set(key, best);
    }
    var outIds = e.emitIds[best];
    if (!this.seen.has(key)) {
      this.seen.add(key);
      if (!e.trained.has(key)) this.unseen++;
      var want = e.exactMemo.get(key);
      if (want === undefined) { want = e.exact(ids); e.exactMemo.set(key, want); }
      if (want !== outIds.map(function (i) { return e.vin[i]; }).join(" ")) this.disagree++;
    }
    this.last = { ids: ids, best: best, out: outIds, forward: fwd };
    this.passes++;
    if (!outIds.length) { this.halt(ids); return false; }
    var mid = e.vin[ids[1]];
    if (mid === B) {
      if (this.ringStart !== null) this.ringComplete(this.ringStart, this.n);
      this.ringStart = this.n;
      if (outIds.length === 2) this.leftGrown++;
    }
    for (var i = 0; i < outIds.length; i++) this.push(outIds[i]);
    this.k++;
    return true;
  };

  Run.prototype.tokens = function (a, b) { var out = []; for (var i = a; i < b; i++) out.push(this.e.vin[this.T[i]]); return out; };

  Run.prototype.ringComplete = function (a, b) {
    var cfg = this.parse(this.tokens(a, b));
    this.utmSteps++;
    if (this.prevState === "fin" && cfg.state === "find") {
      this.simSteps++;
      this.history.push({ step: this.simSteps, tape: cfg.simTape.slice(), head: cfg.simHead, state: cfg.simState });
    }
    this.prevState = cfg.state;
    this.config = cfg;
  };

  Run.prototype.halt = function (ids) {
    var h = this.e.headInfo[ids[1]];
    this.verdict = h && h.state === "qa" ? "ACCEPT" : "REJECT";
    // the last step of the described machine ends at accept (qa) or after its write (nx reading H); a missing rule
    // (at0 reading H) means no step was taken
    if (h && (h.state === "qa" || h.state === "nx")) this.simSteps++;
    var ring = this.tokens(this.k + 1, this.n), i = ring.indexOf(B);
    ring = ring.slice(i).concat(ring.slice(0, i));
    this.config = this.parse(ring);
    var cfg = this.config;
    this.history.push({ step: this.simSteps, tape: cfg.simTape.slice(), head: cfg.simHead,
                        state: this.verdict === "ACCEPT" ? "qa" : "qr", final: true });
    var t = cfg.simTape.join("").replace(/_+$/, "");
    this.result = t || "_";
    this.correct = this.verdict === this.reference.verdict && this.result === this.reference.tape;
  };

  // Read a ring of the transcript as a configuration of the universal machine, and of the machine it is running.
  Run.prototype.parse = function (ring) {
    var cells = [], head = -1, state = null;
    ring.forEach(function (t) {
      if (t === B) return;
      var i = t.indexOf(SEP);
      if (i >= 0) { head = cells.length; state = t.slice(0, i); cells.push(t.slice(i + 1)); } else cells.push(t);
    });
    var dollar = cells.indexOf("$"), simTape = [], simHead = 0, entry = 0, starAt = -1, selected = null;
    var prog = dollar >= 0 ? cells.slice(0, dollar) : cells;
    if (dollar >= 0) cells.slice(dollar + 1).forEach(function (c, j) {
      if (UNMARK[c]) { simHead = j; simTape.push(UNMARK[c]); } else simTape.push(c);
    });
    while (simTape.length > 1 && simTape[simTape.length - 1] === "_" && simTape.length - 1 > simHead) simTape.pop();
    var e = -1, block = 0;
    prog.forEach(function (c, j) {
      if (c === "*" || c === "|") { e++; block = 0; if (c === "*") { starAt = e; } }
      if ("lrs".indexOf(c) >= 0 && c.length === 1) selected = { entry: e, block: block };
      if (c === ";") block++;
    });
    var simState = starAt >= 0 ? this.machine.order[starAt] : "?";
    return { cells: cells, head: head, state: state, dollar: dollar, simTape: simTape, simHead: simHead,
             simState: simState, selected: selected };
  };

  // Run until something happens: n forward passes, the next universal step, the next described step, or the end.
  Run.prototype.advance = function (mode, budget) {
    var start = { u: this.utmSteps, s: this.simSteps }, n = 0;
    while (!this.verdict && n < budget) {
      this.step(false); n++;
      if (mode === "pass" && n >= 1) break;
      if (mode === "utm" && this.utmSteps > start.u) break;
      if (mode === "sim" && (this.simSteps > start.s)) break;
    }
    return n;
  };

  // ---------------------------------------------------------------- a fast run, for long computations

  // The same decoder as Run, in a tight loop over token ids. The transcript is compacted as the window passes, so
  // memory stays bounded by the tape, not the run. onStep(config) is called at every step of the described machine;
  // with stopEachStep set, run() also returns there, so a caller can pace the run one described step at a time.
  function FastRun(engine, machine, tape, onStep) {
    var e = engine, self = this;
    this.e = e; this.machine = machine; this.onStep = onStep;
    this.encoded = encode(machine, tape);
    var cells = this.encoded.split("");
    cells[0] = "find" + SEP + cells[0];
    var ring = [B].concat(cells);
    this.cap = 1 << 20; this.T = new Int32Array(this.cap); this.n = 0;
    this.T[this.n++] = e.id[ring[ring.length - 1]];
    ring.forEach(function (t) { self.T[self.n++] = e.id[t]; });
    this.k = 0; this.ringStart = -1; this.pending = false; this.last = 1;
    this.passes = 0; this.utmSteps = 0; this.simSteps = 0; this.verdict = null;
    this.distinct = 0; this.unseen = 0; this.disagree = 0;
    this.BID = e.id[B];
    if (!e.stateCode) {                                    // 1: find, 2: fin, 3: any other head token, 0: not a head
      e.stateCode = new Int8Array(e.vin.length);
      e.vin.forEach(function (t, i) { var h = e.headInfo[i]; e.stateCode[i] = !h ? 0 : h.state === "find" ? 1 : h.state === "fin" ? 2 : 3; });
      e.emitTyped = e.emitIds.map(function (a) { return Int32Array.from(a); });
      e.fastSeen = new Set();
    }
    this.seen = new Set();
    this.parseOf = Run.prototype.parse;
  }
  FastRun.prototype.miss = function (a, b, c, key) {
    var e = this.e, ids = [a, b, c], best = e.net.forward(ids).best;
    e.memo.set(key, best);
    return best;
  };
  FastRun.prototype.note = function (a, b, c, key, best) {        // per distinct window, once per run
    var e = this.e;
    this.seen.add(key); this.distinct++;
    if (!e.trained.has(key)) this.unseen++;
    var want = e.exactMemo.get(key);
    if (want === undefined) { want = e.exact([a, b, c]); e.exactMemo.set(key, want); }
    if (want !== Array.from(e.emitIds[best], function (i) { return e.vin[i]; }).join(" ")) this.disagree++;
  };
  FastRun.prototype.run = function (budget) {
    var e = this.e, T = this.T, K = e.K, memo = e.memo, emit = e.emitTyped, code = e.stateCode, BID = this.BID;
    var k = this.k, n = this.n, done = 0, stop = false;
    while (done < budget) {
      var a = T[k], b = T[k + 1], c = T[k + 2], key = (a * K + b) * K + c, v = memo.get(key);
      if (v === undefined) v = this.miss(a, b, c, key);
      if (!this.seen.has(key)) this.note(a, b, c, key, v);
      var out = emit[v];
      if (out.length === 0) { this.k = k; this.n = n; this.passes += done; this.halt(b); return done; }
      if (b === BID) {
        if (this.ringStart >= 0) {
          this.utmSteps++;
          if (this.pending) {
            this.pending = false; this.simSteps++; this.T = T; this.k = k; this.n = n; this.at = this.passes + done;
            this.stepDone(this.ringStart, n); stop = this.stopEachStep;
          }
        }
        this.ringStart = n;
      }
      if (n + 2 >= this.cap) {                             // compact: drop what the window has read
        T.copyWithin(0, k, n); n -= k; this.ringStart -= k; k = 0;
        if (n + 2 >= this.cap) { var bigger = new Int32Array(this.cap *= 2); bigger.set(T.subarray(0, n)); T = bigger; }
        this.T = T;
      }
      for (var i = 0; i < out.length; i++) {
        var t = out[i]; T[n++] = t;
        var sc = code[t];
        if (sc) { if (sc === 1 && this.last === 2) this.pending = true; this.last = sc; }
      }
      k++; done++;
      if (stop) break;
    }
    this.k = k; this.n = n; this.passes += done;
    return done;
  };
  FastRun.prototype.tokens = function (a, b) { var out = []; for (var i = a; i < b; i++) out.push(this.e.vin[this.T[i]]); return out; };
  FastRun.prototype.stepDone = function (a, b) {
    var cfg = this.parseOf(this.tokens(a, b));
    if (this.onStep) this.onStep(cfg, this);
  };
  FastRun.prototype.halt = function (b) {                  // as Run.halt: count the last step, read the final ring
    var h = this.e.headInfo[b];
    this.verdict = h && h.state === "qa" ? "ACCEPT" : "REJECT";
    if (h && (h.state === "qa" || h.state === "nx")) this.simSteps++;
    var ring = this.tokens(this.k + 1, this.n), i = ring.indexOf(B);
    this.final = this.parseOf(ring.slice(i).concat(ring.slice(0, i)));
    this.at = this.passes;
    if (this.onHalt) this.onHalt(this.final, this);
  };

  var api = { Engine: Engine, Run: Run, FastRun: FastRun, parseMachine: parseMachine, encode: encode, reference: reference, phase: phase,
              rulesCount: UTM_RULES.length, symbols: ALL };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TCT = api;
})(this);
