// Fractal curves computed by Turing machines, for curves.html. Each machine counts n = 1, 2, 3, ... in binary (written
// backwards from cell 1; cell 0 is a blank that marks the left end), and for every n it passes once through an
// announcing state that carries one value. The page draws a line that turns according to those values.
(function (root) {
  "use strict";
  var dragon = ["s _ -> inc _ R",
    "inc 1 -> inc 0 R", "inc 0 -> back 1 L", "inc _ -> qa _ S",
    "back 0 -> back 0 L", "back 1 -> back 1 L", "back _ -> low _ R",
    "low 0 -> low 0 R", "low 1 -> nxt 1 R",
    "nxt 0 -> sayR 0 L", "nxt _ -> sayR _ L", "nxt 1 -> sayL 1 L",
    "sayR 1 -> home 1 L", "sayL 1 -> home 1 L", "home 0 -> home 0 L", "home _ -> inc _ R"];
  var thue = ["s0 _ -> i00 _ R"];
  [0, 1].forEach(function (t) {
    [0, 1].forEach(function (p) {
      thue.push("i" + t + p + " 1 -> i" + t + (1 - p) + " 0 R",              // a carry: count its parity
                "i" + t + p + " 0 -> say" + (p === 0 ? 1 - t : t) + " 1 L",  // the count of 1s changes parity iff the carries were even
                "i" + t + p + " _ -> qa _ S");
    });
  });
  [0, 1].forEach(function (t) { thue.push("say" + t + " 0 -> h" + t + " 0 L", "say" + t + " _ -> i" + t + "0 _ R", "h" + t + " 0 -> h" + t + " 0 L", "h" + t + " _ -> i" + t + "0 _ R"); });

  var zeck = ["s _ -> go _ R",
    "go 0 -> Q 0 R", "go 1 -> S 0 R",                     // add one at the bottom: a 0 is set pending; a 1 becomes 0 with 2 pending one place up
    "S 0 -> Q 0 R", "S _ -> qa _ S",                      // S: on the pending place; Q: looking at the place above it
    "Q 1 -> S 0 R", "Q 0 -> set 0 L", "Q _ -> set _ L",   // a 1 above: F(p) + F(p+1) = F(p+2), carry on; otherwise set the pending place
    "set 0 -> home 1 L",
    "home 0 -> home 0 L", "home 1 -> home 1 L", "home _ -> look _ R",
    "look 0 -> say0 0 S", "look 1 -> say1 1 S", "say0 0 -> go 0 S", "say1 1 -> go 1 S"];
  function fibWord(n) { var a = "0", b = "01"; while (b.length < n) { var c = b + a; a = b; b = c; } return b.slice(0, n).split("").map(Number); }
  function fib(k) { var a = 1, b = 1; for (var i = 2; i < k; i++) { var c = a + b; a = b; b = c; } return b; }

  function binary(tape) { var n = 0; for (var i = tape.length - 1; i >= 1; i--) n = n * 2 + (tape[i] === "1" ? 1 : 0); return n; }
  function zeckendorf(tape) { var n = 0, a = 1, b = 2; for (var i = 1; i < tape.length; i++) { if (tape[i] === "1") n += a; var c = a + b; a = b; b = c; } return n; }

  var CURVES = [
    {
      id: "dragon", title: "Dragon curve",
      blurb: "Fold a strip of paper in half, again and again, then open every fold to a right angle: the strip traces the " +
             "Heighway dragon. The machine computes the folds. For each n it finds n's lowest 1 bit and reads the bit " +
             "just above it: 0 means turn right, 1 means turn left.",
      start: "s", rules: dragon,
      emit: function (state) { return state === "sayR" ? -1 : state === "sayL" ? 1 : null; },
      expect: function (k) { var out = []; for (var n = 1; n < Math.pow(2, k); n++) { var low = n & -n; out.push(n & (low << 1) ? 1 : -1); } return out; },
      // a unit segment, then a quarter turn for each value, then the last segment
      path: function (vals) {
        var x = 0, y = 0, a = 0, pts = [[0, 0]];
        for (var i = 0; i <= vals.length; i++) {
          x += Math.round(Math.cos(a)); y += Math.round(Math.sin(a)); pts.push([x, y]);
          if (i < vals.length) a += vals[i] * Math.PI / 2;
        }
        return pts;
      },
      label: function (v) { return v > 0 ? "turn left" : "turn right"; }, count: binary,
      how: "The counter is written backwards in binary. inc adds one and back returns; low skips n's trailing zeros; nxt " +
           "reads the bit above n's lowest 1 and passes through sayR or sayL; home returns. When every bit is 1 the count " +
           "overflows and the machine stops, after 2^k - 1 folds."
    },
    {
      id: "koch", title: "Koch curve",
      blurb: "The Thue–Morse sequence, 0 1 1 0 1 0 0 1 ..., is the parity of the number of 1s in n. The machine computes it, " +
             "and a line that turns +60° wherever the sequence changes and -120° wherever it repeats is the Koch curve.",
      start: "s0", rules: thue,
      emit: function (state) { return state === "say0" ? 0 : state === "say1" ? 1 : null; },
      expect: function (k) { var out = []; for (var n = 1; n < Math.pow(2, k); n++) { var c = 0, x = n; while (x) { c ^= x & 1; x >>= 1; } out.push(c); } return out; },
      // one segment per value of the sequence (t(0) = 0 first); turn +60 where it changes, -120 where it repeats
      path: function (vals) {
        var seq = [0].concat(vals), x = 0, y = 0, a = 0, pts = [[0, 0]];
        for (var j = 0; j < seq.length; j++) {
          x += Math.cos(a); y -= Math.sin(a); pts.push([x, y]);
          if (j + 1 < seq.length) a += seq[j + 1] !== seq[j] ? Math.PI / 3 : -2 * Math.PI / 3;
        }
        return pts;
      },
      label: function (v) { return "Thue–Morse " + v; }, count: binary,
      how: "The counter is written backwards in binary. Adding one turns the trailing 1s into 0s; the states i00 to i11 " +
           "remember the current Thue–Morse value and whether the number of carries is even, which decides whether the " +
           "count of 1s changes parity. Then say0 or say1 announces the new value, and h returns."
    },
    {
      id: "fibword", title: "Fibonacci word fractal",
      blurb: "The Fibonacci word, 0100101001001..., is what you get by repeatedly appending the previous word to the current " +
             "one. Its n-th digit is the last digit of n written in the Fibonacci number system (Zeckendorf's), so the " +
             "machine counts in that system. A line that draws a segment per digit and, at each 0, turns left at even " +
             "positions and right at odd ones is the Fibonacci word fractal (Monnerot-Dumaine, 2009).",
      start: "s", rules: zeck,
      sizes: [[11, "232 digits (seconds)"], [14, "986 digits (under a minute)"], [16, "2,583 digits (two or three minutes)"]],
      emit: function (state) { return state === "say0" ? 0 : state === "say1" ? 1 : null; },
      expect: function (k) { return fibWord(fib(k + 2)).slice(1); },
      // digit k = 1, 2, ... (the word's first digit, 0, comes first): a segment, then at a 0 a quarter turn, left if k is even
      path: function (vals) {
        var seq = [0].concat(vals), x = 0, y = 0, a = 0, pts = [[0, 0]];
        for (var i = 0; i < seq.length; i++) {
          x += Math.round(Math.cos(a)); y += Math.round(Math.sin(a)); pts.push([x, y]);
          if (seq[i] === 0) a += ((i + 1) % 2 === 0 ? 1 : -1) * Math.PI / 2;
        }
        return pts;
      },
      label: function (v) { return "digit " + v; }, count: zeckendorf,
      how: "n is written in the Fibonacci number system, least significant place first, places worth 1, 2, 3, 5, 8, .... " +
           "Adding one sets a pending place; when the place above it is 1, the two combine into the next place up (F(p) + " +
           "F(p+1) = F(p+2)) and the carry moves on. Then home returns, and look reads n's last digit and announces it " +
           "through say0 or say1. When the count no longer fits, the machine stops."
    }
  ];
  var api = { curves: CURVES };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TCT_CURVES = api;
})(this);
