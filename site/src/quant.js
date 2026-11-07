// Browser ports of the repo's own code. Each function names its Python source.
// The known-answer check (quant.check.mjs) runs inside `npm run build`.

// level-1-learn/quantize.py round_half_away
export function roundHalfAway(v) {
  return v >= 0 ? Math.floor(v + 0.5) : Math.ceil(v - 0.5);
}

// level-1-learn/quantize.py quantize_symmetric. Python accepts bits 4 and 8 only;
// the playground generalises qmax = 2^(bits-1) - 1 to bits 2..8 with the same rule.
export function quantizeSymmetric(values, bits) {
  const qmax = 2 ** (bits - 1) - 1;
  const maxAbs = Math.max(...values.map((v) => Math.abs(v)));
  const scale = maxAbs > 0 ? maxAbs / qmax : 1.0;
  const codes = values.map((v) => Math.max(-qmax, Math.min(qmax, roundHalfAway(v / scale))));
  return { bits, qmax, scale, codes, dequantized: codes.map((c) => c * scale) };
}

// Python's built-in sum() over floats. CPython 3.12+ uses Neumaier compensated summation,
// which the recorded results depend on in the last bit, so the port mirrors it.
export function pySum(xs) {
  let f = 0, c = 0;
  for (const x of xs) {
    const t = f + x;
    c += Math.abs(f) >= Math.abs(x) ? (f - t) + x : (x - t) + f;
    f = t;
  }
  return c && Number.isFinite(c) ? f + c : f;
}

// level-1-learn/quantize.py error_metrics
export function errorMetrics(original, deq) {
  const errors = original.map((a, i) => Math.abs(a - deq[i]));
  const squared = original.map((a, i) => (a - deq[i]) ** 2);
  return { max_abs_error: Math.max(...errors), mean_abs_error: pySum(errors) / errors.length, mse: pySum(squared) / squared.length, total_abs_error: pySum(errors) };
}

// level-1-learn/quantize.py size_bytes, generalised: payload ceil(count*bits/8) plus one 4-byte FP32 scale.
// At bits 8 and 4 this is exactly the Python formula; bits 32 has no scale.
export function sizeBytes(count, bits) {
  if (bits === 32) return { payload_bytes: count * 4, scale_bytes: 0, total_bytes: count * 4 };
  const payload = Math.floor((count * bits + 7) / 8);
  return { payload_bytes: payload, scale_bytes: 4, total_bytes: payload + 4 };
}

// level-1-learn/quantize.py pack_int4 (two codes per byte, first code low)
export function packInt4(codes) {
  const out = [];
  for (let i = 0; i < codes.length; i += 2) {
    const low = codes[i] & 0xf;
    const high = i + 1 < codes.length ? codes[i + 1] & 0xf : 0;
    out.push(low | (high << 4));
  }
  return out;
}

// level-1-learn/quantize.py reshape, linear_scores, argmax_first, evaluate_task, score_mae
export const reshape = (flat, rows, cols) => Array.from({ length: rows }, (_, r) => flat.slice(r * cols, (r + 1) * cols));
export const linearScores = (matrix, x) => matrix.map((row) => pySum(row.map((w, i) => w * x[i])));
export function argmaxFirst(scores) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  return best;
}
export function evaluateTask(matrix, samples) {
  const predictions = samples.map((s) => argmaxFirst(linearScores(matrix, s.input)));
  const correct = predictions.filter((p, i) => p === samples[i].label).length;
  return { predictions, correct, total: samples.length, accuracy: correct / samples.length };
}
export function scoreMae(a, b, samples) {
  let total = 0, count = 0;
  for (const s of samples) {
    const x = linearScores(a, s.input), y = linearScores(b, s.input);
    x.forEach((v, i) => { total += Math.abs(v - y[i]); count++; });
  }
  return count ? total / count : 0;
}

// level-3-deep/deep_system.py pareto_flags (axes: size_bytes, test_accuracy)
export function paretoFlags(rows) {
  const flags = {};
  for (const row of rows) {
    flags[row.variant] = !rows.some((o) => o !== row
      && o.size_bytes <= row.size_bytes && o.test_accuracy >= row.test_accuracy
      && (o.size_bytes < row.size_bytes || o.test_accuracy > row.test_accuracy));
  }
  return flags;
}

// level-3-deep/deep_system.py recommend_route, including the optional accuracy_tie.
export function recommendRoute(rows, { maxSize = null, minAccuracy = null, maxBatchMs = null, accuracyTie = null } = {}) {
  const survivors = rows.filter((r) => !(maxSize != null && r.size_bytes > maxSize)
    && !(minAccuracy != null && r.test_accuracy < minAccuracy)
    && !(maxBatchMs != null && r.timing_batch_ms_median > maxBatchMs));
  if (!survivors.length) return { winner: null, survivors };
  const cmp = (keys) => (a, b) => { for (const k of keys) { const d = k(a) - k(b); if (d) return d; } return 0; };
  const mse = (r) => r.logit_mse_vs_fp32, size = (r) => r.size_bytes, ms = (r) => r.timing_batch_ms_median;
  if (accuracyTie != null) {
    const best = Math.max(...survivors.map((r) => r.test_accuracy));
    const tied = survivors.filter((r) => best - r.test_accuracy <= accuracyTie + 1e-12);
    return { winner: [...tied].sort(cmp([mse, size, ms]))[0], survivors, tied };
  }
  return { winner: [...survivors].sort(cmp([(r) => -r.test_accuracy, mse, size, ms]))[0], survivors };
}

// level-3-deep/deep_system.py wilson_interval (z 1.96)
export function wilson(correct, total, z = 1.96) {
  const p = correct / total, denom = 1 + (z * z) / total;
  const centre = (p + (z * z) / (2 * total)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / total + (z * z) / (4 * total * total))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}
