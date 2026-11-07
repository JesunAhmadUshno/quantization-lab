// Known-answer self-check, run by `npm run build`: node src/quant.check.mjs
// Asserts the browser ports reproduce the repo's committed result files exactly.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { quantizeSymmetric, errorMetrics, sizeBytes, packInt4, reshape, evaluateTask, scoreMae, paretoFlags, recommendRoute, wilson } from './quant.js';

const read = (p) => JSON.parse(fs.readFileSync(new URL(p, import.meta.url), 'utf8'));
const learn = read('../../level-1-learn/results_learn.json');
const deep = read('../../level-3-deep/results_deep.json');
const seeds = read('../../level-3-deep/results_seeds.json');

// 1. Learn quantizer: codes, scale, dequantized values, errors, sizes, packing, task, score MAE.
const w = learn.weights_flat;
const samples = learn.per_sample.map((s) => ({ input: s.input, label: s.label }));
const fp32 = reshape(w, 4, 3);
for (const bits of [8, 4]) {
  const rec = learn.formats.find((f) => f.bits === bits);
  const q = quantizeSymmetric(w, bits);
  assert.deepEqual(q.codes, rec.codes, `codes ${bits}`);
  assert.equal(q.scale, rec.scale, `scale ${bits}`);
  assert.deepEqual(q.dequantized, rec.dequantized, `dequantized ${bits}`);
  assert.deepEqual(errorMetrics(w, q.dequantized), rec.errors, `errors ${bits}`);
  assert.deepEqual(sizeBytes(w.length, bits), rec.sizes, `sizes ${bits}`);
  if (bits === 4) assert.deepEqual(packInt4(q.codes), rec.packed_bytes, 'int4 packing');
  const tag = bits === 8 ? 'int8' : 'int4';
  const t = evaluateTask(reshape(q.dequantized, 4, 3), samples);
  assert.deepEqual(t.predictions, learn.task[tag].predictions, `predictions ${bits}`);
  assert.equal(t.correct, learn.task[tag].correct, `correct ${bits}`);
  assert.equal(scoreMae(fp32, reshape(q.dequantized, 4, 3), samples), learn.task[tag].score_mae_vs_fp32, `score mae ${bits}`);
}
assert.deepEqual(sizeBytes(w.length, 32), learn.formats[0].sizes, 'fp32 size');
assert.equal(evaluateTask(fp32, samples).correct, learn.task.fp32.correct, 'fp32 task');

// 2. Pareto flags and decision rule on the recorded Deep rows.
const rows = deep.rows;
const flags = paretoFlags(rows);
for (const r of rows) assert.equal(flags[r.variant], r.pareto_optimal, `pareto ${r.variant}`);
const vi = deep.verdict_inputs;
assert.deepEqual(rows.filter((r) => flags[r.variant]).map((r) => r.variant), vi.pareto_routes);
assert.equal(recommendRoute(rows).winner.variant, vi.best_accuracy_route);
assert.equal(recommendRoute(rows, { maxSize: 1200 }).winner.variant, vi.under_1200_bytes_best);
assert.equal(rows.reduce((a, b) => (b.size_bytes < a.size_bytes ? b : a)).variant, vi.smallest_route);
// One-sample tie (accuracy_tie = 1/400): the Python test asserts the same answer.
assert.equal(recommendRoute(rows, { maxSize: 1200, accuracyTie: 1 / 400 }).winner.variant, 'int4_per_channel_packed');

// 3. Wilson intervals equal deep_system.wilson_interval as recorded in results_seeds.json.
for (const r of rows) {
  const rec = seeds.canonical.wilson95[r.variant];
  assert.equal(rec.test_correct, r.test_correct, `seed 0 reproduces ${r.variant}`);
  const [lo, hi] = wilson(r.test_correct, r.test_total);
  assert.ok(Math.abs(lo - rec.low) < 1e-12 && Math.abs(hi - rec.high) < 1e-12, `wilson ${r.variant}`);
}
console.log('quant ok', { int8_max_abs: learn.summary.int8_max_abs_error, int4_max_abs: learn.summary.int4_max_abs_error, pareto: vi.pareto_routes, under1200: vi.under_1200_bytes_best });
