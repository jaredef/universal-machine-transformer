// The sieve of Eratosthenes as a Turing machine (experiments/big-program/primes.py): cell n of the tape holds 1 while n
// may be prime and 0 once it is crossed out. For k = 2, 3, ...: if cell k is still 1, k is prime; the machine marks it,
// walks right to the end of the tape crossing out every k-th cell, walks back to the mark, restores it, and moves on.
// The countdown to the next multiple is held in the machine's states, one chain of k states per prime stride, which is
// what makes the program big: sieving 0 .. N takes the strides of the primes up to sqrt(N).
(function (root) {
  "use strict";
  function primesUpto(n) {
    var out = [];
    for (var p = 2; p <= n; p++) { var ok = true; for (var d = 2; d * d <= p; d++) if (p % d === 0) { ok = false; break; } if (ok) out.push(p); }
    return out;
  }
  function sieve(N) {
    var top = Math.floor(Math.sqrt(N)), base = primesUpto(top), rules = ["s 0 -> s1 0 R", "s1 0 -> k2 0 R"];
    for (var k = 2; k <= top; k++) {
      var nxt = k < top ? "k" + (k + 1) : "qa";
      rules.push("k" + k + " 0 -> " + nxt + " 0 R");
      if (base.indexOf(k) < 0) continue;
      rules.push("k" + k + " 1 -> m" + k + "_" + (k - 1) + " _ R");
      for (var j = 0; j < k; j++) {
        ["0", "1"].forEach(function (x) { rules.push("m" + k + "_" + j + " " + x + " -> m" + k + "_" + ((j - 1 + k) % k) + " " + (j ? x : "0") + " R"); });
        rules.push("m" + k + "_" + j + " _ -> r" + k + " _ L");
      }
      rules.push("r" + k + " 0 -> r" + k + " 0 L", "r" + k + " 1 -> r" + k + " 1 L", "r" + k + " _ -> " + nxt + " 1 R");
    }
    return { rules: rules, start: "s", tape: "00" + "1".repeat(N - 1), base: base, primes: primesUpto(N) };
  }
  // sizes: N = side^2 - 1, so the Ulam spiral of side x side shows 1 .. N (side^2 itself is left out: it is not prime,
  // but its smallest factor, side, would need a stride the program does not have)
  var sizes = [[11, "1 to 120", "32 states, a second"], [23, "1 to 528", "108 states, about 15 seconds"],
               [31, "1 to 960", "170 states, about 40 seconds"], [41, "1 to 1,680", "250 states, about 2½ minutes"]];
  var api = { sieve: sieve, primesUpto: primesUpto, sizes: sizes };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TCT_PRIMES = api;
})(this);
