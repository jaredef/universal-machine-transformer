// Programs the network never saw, for programs.html: real computations, written as ordinary Turing machines over
// 0 1 _ on a one-way tape (cell 0 is a blank that marks the left end). Numbers are in binary written backwards (least
// significant bit first), so they can grow to the right. Each program has a decoder that reads its tape.
(function (root) {
  "use strict";
  function rev(bits) { return bits.split("").reverse().join(""); }
  function num(lsbFirst) { var s = lsbFirst.replace(/[^01]/g, ""); return s ? parseInt(rev(s), 2) : 0; }
  function bin(n) { return n.toString(2); }
  function cells(tape) { return Array.isArray(tape) ? tape.join("") : tape; }

  // Fibonacci numbers: the pair (a, b) with its bits interleaved, a0 b0 a1 b1 ..., least significant first; one pass,
  // carrying, turns it into (b, a + b)
  var FIB = ["s _ -> A0 _ R"];
  [0, 1].forEach(function (c) {
    [0, 1].forEach(function (x) { FIB.push("A" + c + " " + x + " -> B" + c + x + " " + x + " R"); });            // read a's bit
    [0, 1].forEach(function (x) { [0, 1].forEach(function (y) {
      var t = x + y + c; FIB.push("B" + c + x + " " + y + " -> W" + y + (t >> 1) + " " + (t & 1) + " L");          // read b's bit, write the sum's
    }); });
  });
  [0, 1].forEach(function (y) { [0, 1].forEach(function (c) { [0, 1].forEach(function (z) {
    FIB.push("W" + y + c + " " + z + " -> N" + c + " " + y + " R");                                                // a's bit becomes b's old bit
  }); }); });
  [0, 1].forEach(function (c) { [0, 1].forEach(function (z) { FIB.push("N" + c + " " + z + " -> A" + c + " " + z + " R"); }); });
  FIB.push("A0 _ -> home _ L", "A1 _ -> qa _ S", "home 0 -> home 0 L", "home 1 -> home 1 L", "home _ -> A0 _ R");
  function interleaved(t, which) { var n = 0; for (var i = t.length - 1; i >= 1; i--) if ((i - 1) % 2 === which) n = n * 2 + (t[i] === "1" ? 1 : 0); return n; }

  var P = [
    {
      id: "collatz", title: "Collatz",
      blurb: "Halve n when it is even; otherwise replace it with 3n + 1; stop at 1. From 27 the numbers climb to 9,232 " +
             "before falling to 1, after 111 steps. Whether every starting number reaches 1 is a famous open problem.",
      start: "s", param: { label: "Start from n =", value: 27, min: 1, max: 999 },
      input: function (n) { return "_" + rev(bin(n)); },
      rules: [
        "s _ -> s _ R", "s 0 -> s _ R", "s 1 -> one 0 R",
        "one _ -> fix _ L", "one 0 -> c1 0 R", "one 1 -> c2 1 R", "fix 0 -> qa 1 S",
        "c0 0 -> c0 0 R", "c0 1 -> c1 1 R", "c0 _ -> back _ L",
        "c1 0 -> c0 1 R", "c1 1 -> c2 0 R", "c1 _ -> back 1 L",
        "c2 0 -> c1 0 R", "c2 1 -> c2 1 R", "c2 _ -> c1 0 R",
        "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> s _ R"],
      how: "n is written backwards in binary. An even n loses its last bit (the tape's first digit), which is halving. For " +
           "an odd n, one left-to-right pass multiplies by 3 and adds 1, carrying 0, 1, or 2 (states c0, c1, c2).",
      // the value of n, read when the machine is about to look at n's lowest bit
      value: function (cfg) { return cfg.state === "s" && (cells(cfg.tape)[cfg.head] || "_") !== "_" ? num(cells(cfg.tape)) : null; },
      expect: function (n) { var seq = [n]; while (n !== 1) { n = n % 2 ? 3 * n + 1 : n / 2; seq.push(n); } return seq; },
      show: function (v) { return "n = " + v.toLocaleString(); }, plot: function (v) { return v; },
      result: function (tape) { return "n = " + num(cells(tape)); }
    },
    {
      id: "counter", title: "Binary counter",
      blurb: "A binary counter: add one, again and again, until it runs out of bits (8 bits: 256 additions), each carrying " +
             "as far as it needs to.",
      start: "s", param: { label: "Bits:", value: 8, min: 1, max: 12 },
      input: function (k) { return "_" + "0".repeat(k); },
      rules: ["s _ -> inc _ R", "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> qa _ S",
              "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> inc _ R"],
      how: "The counter is written backwards in binary. inc turns trailing 1s into 0s until it can turn a 0 into a 1; back " +
           "returns to the start. When every bit is 1, the carry runs off the end and the machine stops.",
      value: function (cfg) { return cfg.state === "inc" && cfg.head === 1 ? num(cells(cfg.tape)) : null; },
      expect: function (k) { var seq = []; for (var i = 0; i < Math.pow(2, k); i++) seq.push(i); return seq; },
      show: function (v) { return v.toLocaleString() + "  =  " + bin(v); }, plot: function (v) { return v; },
      result: function (tape) { return "overflowed back to " + num(cells(tape)) + " after counting to " + (Math.pow(2, cells(tape).length - 1) - 1); }
    },
    {
      id: "sort", title: "Sort",
      blurb: "Bubble sort: find a 1 just before a 0, swap them, start over. It stops when every 0 comes before every 1.",
      start: "s", param: { label: "String:", value: "1011001110100110", pattern: "^[01]{1,24}$" },
      input: function (w) { return "_" + w; },
      rules: ["s _ -> r _ R", "r 0 -> r 0 R", "r 1 -> r1 1 R", "r _ -> qa _ S",
              "r1 1 -> r1 1 R", "r1 0 -> sw 1 L", "r1 _ -> qa _ S", "sw 1 -> home 0 L",
              "home 0 -> home 0 L", "home 1 -> home 1 L", "home _ -> r _ R"],
      how: "r scans right; r1 has just passed a 1. Meeting a 0 there, it writes 1, steps back, writes 0 (a swap), and goes " +
           "home to start the scan again.",
      value: function (cfg) { return cfg.state === "r" && cfg.head === 1 ? cells(cfg.tape).slice(1).replace(/_+$/, "") : null; },
      expect: null, show: function (v) { return v; }, plot: function (v) { return v; },
      result: function (tape) { return cells(tape).slice(1).replace(/_+$/, ""); }
    },
    {
      id: "add", title: "Add",
      blurb: "Binary addition by counting: take one from b and add one to a, until b is zero.",
      start: "s", param: { label: "a + b:", value: "1000+99", pattern: "^\\d{1,5}\\+\\d{1,3}$" },
      input: function (s) { var p = s.split("+"); return "_" + rev(bin(+p[1])) + "_" + rev(bin(+p[0])); },
      rules: ["s _ -> dec _ R", "dec 0 -> dec 1 R", "dec 1 -> toA 0 R", "dec _ -> clean _ L",
              "toA 0 -> toA 0 R", "toA 1 -> toA 1 R", "toA _ -> inc _ R",
              "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> back 1 L",
              "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> back2 _ L",
              "back2 0 -> back2 0 L", "back2 1 -> back2 1 L", "back2 _ -> dec _ R",
              "clean 1 -> clean _ L", "clean _ -> qa _ S"],
      how: "The tape holds b, a blank, then a, each written backwards in binary. dec takes one from b, inc adds one to a, " +
           "and back and back2 return to the start. When b is already zero, the borrow runs off b's end: clean erases b.",
      value: function (cfg) {
        if (cfg.state !== "dec" || cfg.head !== 1) return null;
        var t = cells(cfg.tape), i = t.indexOf("_", 1);
        return [num(t.slice(i + 1)), num(t.slice(1, i))];
      },
      expect: null, show: function (v) { return v[0].toLocaleString() + " + " + v[1].toLocaleString(); }, plot: function (v) { return v[0]; },
      result: function (tape) { return "a + b = " + num(cells(tape)); }
    },
    {
      id: "count", title: "Count in binary",
      blurb: "Turn a row of n ones into n written in binary: cross off a one, add one to a binary counter, repeat.",
      start: "s", param: { label: "Ones:", value: 40, min: 1, max: 80 },
      input: function (n) { return "_" + "1".repeat(n); },
      rules: ["s _ -> f _ R", "f 0 -> f 0 R", "f 1 -> g 0 R", "f _ -> qa _ S",
              "g 1 -> g 1 R", "g _ -> inc _ R", "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> back 1 L",
              "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> back2 _ L",
              "back2 0 -> back2 0 L", "back2 1 -> back2 1 L", "back2 _ -> f _ R"],
      how: "f finds the first one not yet crossed off (crossed off is 0) and crosses it off; g walks to the counter past a " +
           "blank; inc adds one (backwards binary); back and back2 return to the start.",
      value: function (cfg) {
        if (cfg.state !== "f" || cfg.head !== 1) return null;
        var t = cells(cfg.tape), i = t.indexOf("_", 1);
        return i < 0 ? 0 : num(t.slice(i + 1));
      },
      expect: null, show: function (v) { return v.toLocaleString() + "  =  " + bin(v); }, plot: function (v) { return v; },
      result: function (tape) { var t = cells(tape), i = t.indexOf("_", 1); return "counted " + num(t.slice(i + 1)) + " (" + rev(t.slice(i + 1).replace(/_+$/, "")) + " in binary)"; }
    }
  ];
  P.push({
    id: "fib", title: "Fibonacci",
    blurb: "0, 1, 1, 2, 3, 5, 8, 13, ...: each number the sum of the two before it, computed in binary, until the next one no " +
           "longer fits. A bigger program than the others (14 states), so each of its steps costs the universal machine more.",
    start: "s", param: { label: "Bits:", value: 8, min: 4, max: 24 },
    input: function (w) { return "_01" + "00".repeat(w - 1); },
    rules: FIB,
    how: "The tape holds a and b with their bits interleaved, least significant first. One left-to-right pass carries 0 or " +
         "1 (A0, A1): it reads a's bit and b's bit (B), writes their sum into b's place, steps back and writes b's old bit " +
         "into a's place (W), and moves on to the next pair (N). So (a, b) becomes (b, a + b). A carry out of the last pair " +
         "means the next number does not fit, and the machine stops.",
    value: function (cfg) { return cfg.state === "A0" && cfg.head === 1 ? interleaved(cells(cfg.tape), 0) : null; },
    expect: function (w) { var seq = [], a = 0, b = 1; while (b < Math.pow(2, w)) { seq.push(a); var c = a + b; a = b; b = c; } return seq; },
    show: function (v) { return v.toLocaleString(); }, plot: function (v) { return v; },
    result: function (tape) { var t = cells(tape); return "the largest that fits: " + interleaved(t, 0).toLocaleString(); }
  });
  var api = { programs: P, num: num, rev: rev };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TCT_PROGRAMS = api;
})(this);
