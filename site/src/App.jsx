import { useMemo, useState, useEffect, useId } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import build from '../../level-2-build/results_build.json';
import deep from '../../level-3-deep/results_deep.json';
import seeds from '../../level-3-deep/results_seeds.json';
import { quantizeSymmetric, errorMetrics, sizeBytes, packInt4, reshape, evaluateTask, scoreMae, recommendRoute, wilson } from './quant.js';

const REPO = 'https://github.com/JesunAhmadUshno/quantization-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const f4 = (x) => x.toFixed(4);
const f6 = (x) => x.toFixed(6);

const ROWS = deep.rows;
const BY = Object.fromEntries(ROWS.map((r) => [r.variant, r]));
const FP32 = BY.fp32;
const N_TEST = deep.dataset.test_samples;
const PARAMS = ROWS[0].nonzero_parameters;
// The byte budget the repo's verdict uses, read from its verdict_inputs key (under_<N>_bytes_best).
const BUDGET_KEY = Object.keys(deep.verdict_inputs).find((k) => /^under_\d+_bytes_best$/.test(k));
const BUDGET = +BUDGET_KEY.match(/\d+/)[0];

// Human names generated from the variant ids, so no number is typed by hand.
function routeName(v) {
  return v.split('_int').map((part, i) => {
    const s = i ? `int${part}` : part;
    if (s === 'fp32') return 'FP32';
    if (s.startsWith('pruned_')) return `Pruned ${s.slice(7)}`;
    const m = s.match(/^int(\d)_per_(tensor|channel)(_packed)?$/);
    return m ? `INT${m[1]} per-${m[2]}${m[3] ? ' packed' : ''}` : s;
  }).join(' + ');
}
const short = (v) => routeName(v).replace('per-tensor', 'pt').replace('per-channel packed', 'packed').replace('per-channel', 'pc').replace('Pruned ', 'P');

const SECTIONS = [['top', 'Overview', '◎'], ['quantize', 'Quantizer', '⊞'], ['routes', 'Routes', '◇'], ['decide', 'Decision rule', '⇥'], ['noise', 'Noise', '±'], ['prune', 'Pruning', '✂'], ['levels', 'Levels', '≡'], ['limits', 'Limits', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">Quantization Lab</a>
      <a className="m" href={blob('SPEC.md')}>Spec</a>
      <a className="m" href={blob('level-3-deep/README.md')}>Deep study</a>
      <a className="m" href={blob('docs/research-review.md')}>Research review</a>
      <span className="spacer" />
      <span className="status"><i /> CI reproducible</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

export default function App() {
  useReveal();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <Quantizer />
      <Routes />
      <Decision />
      <Noise />
      <Pruning />
      <Levels />
      <Limits />
      <Dock />
      <footer>
        <p>Code: Apache-2.0 · <a href={REPO}>Repository</a> · <a href={`${REPO}/releases`}>Releases</a></p>
        <p>Reproduce: <code>python level-3-deep/run_deep.py</code> · <code>python level-3-deep/seed_study.py</code> · <code>python -m unittest discover -s tests</code></p>
        <p>Every number on this page is read from <code>level-*/results_*.json</code> at build time; the quantizer, decision rule and intervals are computed in your browser.</p>
      </footer>
    </div>
  );
}

function Hero() {
  const pc = BY.int8_per_channel, i4 = BY.int4_per_channel_packed;
  const tiles = [
    { r: FP32, cap: `FP32 baseline, ${FP32.test_correct} of ${FP32.test_total} correct` },
    { r: pc, cap: `INT8 per-channel, ${pc.test_correct} of ${pc.test_total}, logit MSE ${f6(pc.logit_mse_vs_fp32)}` },
    { r: i4, cap: `INT4 packed, ${i4.test_correct} of ${i4.test_total}, the smallest route` },
  ];
  return (
    <header className="card hero window" id="top" data-title="quantlab — README">
      <div className="head">
        <p className="eyebrow">Quality · size · speed study · {ROWS.length} routes</p>
        <span className="pill good">● Every number from results JSON</span>
      </div>
      <h1>INT8 per-channel keeps <em>full accuracy</em> at less than half the bytes.</h1>
      <p className="sub">
        On a {N_TEST}-sample held-out test, INT8 per-channel scores {pc.test_correct} of {pc.test_total} ({f4(pc.test_accuracy)}), exactly the FP32 result, at {fmt(pc.size_bytes)} bytes
        against FP32&apos;s {fmt(FP32.size_bytes)} ({(pc.size_bytes / FP32.size_bytes * 100).toFixed(0)}% of the size). Pruning alone never shrinks the file: every pruned route
        is still {fmt(BY['pruned_0.5'].size_bytes)} bytes, because zeros are stored densely.
      </p>
      <div className="actions">
        <a className="btn primary" href="#quantize">Try the live quantizer</a>
        <a className="btn ghost" href={blob('level-3-deep/README.md')}>Read the Deep study</a>
        <span className="hint">Scope: {deep.model.architecture}, synthetic data, CPU only, fake quantization.</span>
      </div>
      <div className="tiles tiles3">
        {tiles.map(({ r, cap }) => (
          <div className="tile" key={r.variant}><div className="num">{fmt(r.size_bytes)} B</div><div className="cap">{cap}, accuracy {f4(r.test_accuracy)}</div></div>
        ))}
      </div>
      <ul className="scope">
        <li><b>Data:</b> {deep.dataset.kind}; train {deep.dataset.train_samples}, calibration {deep.dataset.calibration_samples}, test {deep.dataset.test_samples}.</li>
        {deep.substitutions.map((s) => <li key={s}>{s}</li>)}
      </ul>
    </header>
  );
}

/* ---------------------------------------------------------------- live quantizer */

function Quantizer() {
  const recorded = learn.weights_flat;
  const samples = useMemo(() => learn.per_sample.map((s) => ({ input: s.input, label: s.label })), []);
  const [bits, setBits] = useState(4);
  const [w, setW] = useState(recorded);
  const q = quantizeSymmetric(w, bits);
  const err = errorMetrics(w, q.dequantized);
  const size = sizeBytes(w.length, bits), fp32Size = sizeBytes(w.length, 32);
  const fp32 = reshape(w, 4, 3), qm = reshape(q.dequantized, 4, 3);
  const taskF = evaluateTask(fp32, samples), taskQ = evaluateTask(qm, samples);
  const mae = scoreMae(fp32, qm, samples);
  const edited = w.some((v, i) => v !== recorded[i]);
  const rec = learn.formats.find((f) => f.bits === bits);
  const tag = bits === 8 ? 'int8' : 'int4';
  const match = !edited && rec && JSON.stringify(q.codes) === JSON.stringify(rec.codes) && q.scale === rec.scale
    && err.max_abs_error === rec.errors.max_abs_error && err.mse === rec.errors.mse && size.total_bytes === rec.sizes.total_bytes
    && taskQ.correct === learn.task[tag].correct && mae === learn.task[tag].score_mae_vs_fp32;
  const badge = edited ? ['info', 'Weights edited: no recorded value to compare'] : !rec ? ['info', `No recorded run at ${bits} bits (recorded: 8 and 4)`] : match ? ['good', `✓ Matches results_learn.json exactly at ${bits} bits`] : ['bad', 'Mismatch with results_learn.json'];

  const W = 520, H = 220, pad = { l: 34, r: 8, t: 12, b: 22 };
  const lim = Math.max(1, ...w.map(Math.abs));
  const x = (i) => pad.l + (i + 0.5) * ((W - pad.l - pad.r) / w.length);
  const y = (v) => pad.t + (1 - (v + lim) / (2 * lim)) * (H - pad.t - pad.b);
  const levels = Array.from({ length: 2 * q.qmax + 1 }, (_, k) => (k - q.qmax) * q.scale);
  return (
    <section id="quantize" className="card window reveal" data-title="quantize.py — live in your browser">
      <div className="head">
        <div>
          <p className="eyebrow">Live playground · Learn level</p>
          <h2>Squeeze {w.length} weights into {bits} bits and watch the <em>error</em> appear.</h2>
          <p className="sub">
            A line-for-line JavaScript port of <a href={blob('level-1-learn/quantize.py')}>quantize.py</a>: symmetric per-tensor scale = max |w| / qmax, round half away from zero, clamp to ±qmax.
            It runs on the recorded {w.length}-weight, 4-class linear model and its {samples.length} task samples. The Python code accepts 8 and 4 bits; other widths use the same rule.
          </p>
        </div>
        <span className={`pill ${badge[0]}`} role="status">{badge[1]}</span>
      </div>
      <div className="grid2">
        <div>
          <div className="controls" style={{ gridTemplateColumns: '1fr' }}>
            <Range label="Bit width" value={bits} min={2} max={8} step={1} onChange={setBits} show={`${bits}-bit · qmax ${q.qmax}`} />
          </div>
          <div className="weights" role="group" aria-label="Editable weights, 4 rows by 3 columns">
            {w.map((v, i) => (
              <input key={i} type="number" step="0.05" value={v} aria-label={`Weight row ${Math.floor(i / 3) + 1} column ${(i % 3) + 1}`}
                onChange={(e) => { const n = parseFloat(e.target.value); if (Number.isFinite(n)) setW(w.map((o, j) => (j === i ? n : o))); }} />
            ))}
          </div>
          <div className="actions" style={{ marginTop: 10 }}>
            <button className="btn ghost" onClick={() => { setW(recorded); setBits(4); }} disabled={!edited && bits === 4}>Reset to recorded 4-bit</button>
            <button className="btn ghost" onClick={() => setBits(8)} disabled={bits === 8}>Recorded 8-bit</button>
          </div>
          <div className="tiles" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
            <div className="tile"><div className="num">{size.total_bytes} B</div><div className="cap">{size.payload_bytes} payload + {size.scale_bytes} scale, vs {fp32Size.total_bytes} B FP32 ({(fp32Size.total_bytes / size.total_bytes).toFixed(1)}× smaller)</div></div>
            <div className="tile"><div className="num">{taskQ.correct}/{taskQ.total}</div><div className="cap">task accuracy (FP32 of these weights: {taskF.correct}/{taskF.total}); score MAE {mae.toFixed(6)}</div></div>
          </div>
        </div>
        <div>
          <svg className="chart dense" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Original versus ${bits}-bit dequantized weights`}>
            {levels.length <= 31 && levels.map((l) => <line key={l} x1={pad.l} x2={W - pad.r} y1={y(l)} y2={y(l)} stroke="var(--grid)" />)}
            <line className="axis" x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} strokeWidth="1.5" />
            <text x={pad.l - 6} y={y(lim) + 8} textAnchor="end">{lim.toFixed(2)}</text>
            <text x={pad.l - 6} y={y(-lim)} textAnchor="end">{(-lim).toFixed(2)}</text>
            {w.map((v, i) => (
              <g key={i}>
                <line x1={x(i)} x2={x(i)} y1={y(v)} y2={y(q.dequantized[i])} stroke="var(--bad)" strokeWidth="3" />
                <circle cx={x(i)} cy={y(v)} r="5" fill="none" stroke="var(--text)" strokeWidth="1.5" />
                <circle cx={x(i)} cy={y(q.dequantized[i])} r="3.5" fill="var(--primary)" />
              </g>
            ))}
            <text x={pad.l} y={H - 4}>weight 1</text><text x={W - pad.r} y={H - 4} textAnchor="end">weight {w.length}</text>
          </svg>
          <p className="legend"><span><i style={{ background: 'transparent', border: '1.5px solid var(--text)' }} />original</span><span><i style={{ background: 'var(--primary)' }} />dequantized</span><span><i style={{ background: 'var(--bad)' }} />error</span>{levels.length <= 31 && <span>grid lines: the {levels.length} representable levels</span>}</p>
        </div>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Quantity</th><th>Value at {bits} bits</th>{rec && !edited && <th>Recorded</th>}</tr></thead>
          <tbody>
            <tr><td>Scale</td><td className="mono">{q.scale.toPrecision(8)}</td>{rec && !edited && <td className="mono">{rec.scale.toPrecision(8)}</td>}</tr>
            <tr><td>Codes</td><td className="mono">[{q.codes.join(', ')}]</td>{rec && !edited && <td className="mono">[{rec.codes.join(', ')}]</td>}</tr>
            {bits === 4 && <tr><td>Packed bytes (two codes per byte)</td><td className="mono">[{packInt4(q.codes).join(', ')}]</td>{rec && !edited && <td className="mono">[{rec.packed_bytes.join(', ')}]</td>}</tr>}
            <tr><td>Max abs error</td><td className="mono">{f6(err.max_abs_error)}</td>{rec && !edited && <td className="mono">{f6(rec.errors.max_abs_error)}</td>}</tr>
            <tr><td>Mean abs error</td><td className="mono">{f6(err.mean_abs_error)}</td>{rec && !edited && <td className="mono">{f6(rec.errors.mean_abs_error)}</td>}</tr>
            <tr><td>MSE</td><td className="mono">{err.mse.toExponential(4)}</td>{rec && !edited && <td className="mono">{rec.errors.mse.toExponential(4)}</td>}</tr>
            <tr><td>Predictions on the {samples.length} samples</td><td className="mono">[{taskQ.predictions.join(', ')}]</td>{rec && !edited && <td className="mono">[{learn.task[tag].predictions.join(', ')}]</td>}</tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- route explorer */

const AXES = {
  size_bytes: { label: 'Size (QLAB bytes)', fmt: (v) => fmt(v), log: false },
  test_accuracy: { label: 'Test accuracy', fmt: (v) => v.toFixed(4), log: false },
  logit_mse_vs_fp32: { label: 'Logit MSE vs FP32 (log)', fmt: (v) => v.toExponential(2), log: true },
  timing_batch_ms_median: { label: 'Batch ms, median (noisy)', fmt: (v) => v.toFixed(3), log: false },
};
const MSE_FLOOR = Math.min(...ROWS.filter((r) => r.variant !== 'fp32').map((r) => r.logit_mse_vs_fp32)) / 10;

function scaleFor(key, lo, hi) {
  const a = AXES[key];
  const t = (v) => (a.log ? Math.log10(Math.max(MSE_FLOOR, v)) : v);
  const [l, h] = [t(lo), t(hi)], padv = (h - l) * 0.08 || 1;
  return (v, from, to) => from + ((t(v) - (l - padv)) / (h - l + 2 * padv)) * (to - from);
}

function Routes() {
  const [xk, setXk] = useState('size_bytes');
  const [yk, setYk] = useState('test_accuracy');
  const [sel, setSel] = useState('int8_per_channel');
  const W = 720, H = 320, pad = { l: 64, r: 20, t: 16, b: 40 };
  const ext = (k) => [Math.min(...ROWS.map((r) => r[k])), Math.max(...ROWS.map((r) => r[k]))];
  const sx = scaleFor(xk, ...ext(xk)), sy = scaleFor(yk, ...ext(yk));
  const X = (r) => sx(r[xk], pad.l, W - pad.r), Y = (r) => sy(r[yk], H - pad.b, pad.t);
  const sizeAcc = new Set([xk, yk]).size === 2 && [xk, yk].every((k) => k === 'size_bytes' || k === 'test_accuracy');
  const frontier = ROWS.filter((r) => r.pareto_optimal).sort((a, b) => a.size_bytes - b.size_bytes);
  const row = BY[sel];
  const axisSelect = (val, set, label) => {
    const id = `axis-${label}`;
    return (
      <div className="field"><label htmlFor={id}>{label} axis</label>
        <select id={id} value={val} onChange={(e) => set(e.target.value)}>
          {Object.entries(AXES).map(([k, a]) => <option key={k} value={k}>{a.label}</option>)}
        </select>
      </div>
    );
  };
  return (
    <section id="routes" className="card window reveal" data-title="results_deep.json — recorded replay">
      <div className="head">
        <div>
          <p className="eyebrow">Route explorer · recorded</p>
          <h2>Ten routes, four axes. Only <em>two</em> are Pareto optimal.</h2>
          <p className="sub">
            Replay of the recorded study (Muse VM CPU, 2026-11-05). Pareto flags use size and accuracy only, as in <span className="mono">deep_system.pareto_flags</span>; the timing axis
            is the only nondeterministic field and spans {AXES.timing_batch_ms_median.fmt(ext('timing_batch_ms_median')[0])} to {AXES.timing_batch_ms_median.fmt(ext('timing_batch_ms_median')[1])} ms because every route runs the same dequantized forward pass.
          </p>
        </div>
        <span className="pill warn">RECORDED · not rerun</span>
      </div>
      <div className="controls" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
        {axisSelect(xk, setXk, 'Horizontal')}
        {axisSelect(yk, setYk, 'Vertical')}
      </div>
      <svg className="chart dense" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Scatter of ${ROWS.length} recorded routes: ${AXES[yk].label} against ${AXES[xk].label}`}>
        <line className="axis" x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} />
        <line className="axis" x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} />
        {[0.5].map((f) => <g key="mid"><line stroke="var(--grid)" x1={pad.l} x2={W - pad.r} y1={(pad.t + H - pad.b) / 2} y2={(pad.t + H - pad.b) / 2} /><line stroke="var(--grid)" y1={pad.t} y2={H - pad.b} x1={(pad.l + W - pad.r) / 2} x2={(pad.l + W - pad.r) / 2} /></g>)}
        {ext(xk).map((v, i) => <text key={`x${i}`} x={sx(v, pad.l, W - pad.r)} y={H - pad.b + 16} textAnchor="middle">{AXES[xk].fmt(xk === 'logit_mse_vs_fp32' && !i ? Math.max(v, MSE_FLOOR) : v)}</text>)}
        {ext(yk).map((v, i) => <text key={`y${i}`} x={pad.l - 6} y={sy(v, H - pad.b, pad.t) + 4} textAnchor="end">{AXES[yk].fmt(yk === 'logit_mse_vs_fp32' && !i ? Math.max(v, MSE_FLOOR) : v)}</text>)}
        <text x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle">{AXES[xk].label}</text>
        <text x={14} y={(pad.t + H - pad.b) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(pad.t + H - pad.b) / 2})`}>{AXES[yk].label}</text>
        {sizeAcc && <polyline fill="none" stroke="var(--good)" strokeWidth="2" strokeDasharray="6 4" points={frontier.map((r) => `${X(r)},${Y(r)}`).join(' ')} />}
        {ROWS.map((r) => (
          <g key={r.variant} className="pt" tabIndex={0} role="button" aria-label={`Open ${routeName(r.variant)}`} onClick={() => setSel(r.variant)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setSel(r.variant))}>
            <circle cx={X(r)} cy={Y(r)} r={sel === r.variant ? 9 : 6.5} fill={r.pareto_optimal ? 'var(--good)' : r.variant.startsWith('pruned') ? 'var(--bad)' : 'var(--primary)'} stroke="var(--surface-strong)" strokeWidth="2" opacity={sel && sel !== r.variant ? 0.75 : 1} />
            {(sel === r.variant || r.pareto_optimal) && (() => { const top = r.pareto_optimal && r[yk] === Math.max(...frontier.map((f) => f[yk])); const right = X(r) > W * 0.8; const lx = r.pareto_optimal ? X(r) + (right ? -12 : 12) : X(r) + (right ? -14 : 14); const ly = r.pareto_optimal ? (top ? Y(r) - 12 : Y(r) + 24) : Y(r) + 5; return <text x={lx} y={ly} textAnchor={right ? 'end' : 'start'} style={{ fill: 'var(--text)', fontWeight: 600 }}>{short(r.variant)}</text>; })()}
          </g>
        ))}
      </svg>
      <p className="legend"><span><i style={{ background: 'var(--good)' }} />Pareto (recorded flag)</span><span><i style={{ background: 'var(--primary)' }} />quantized / FP32</span><span><i style={{ background: 'var(--bad)' }} />pruned</span>{sizeAcc && <span>dashed: size–accuracy frontier</span>}{AXES[xk].log || AXES[yk].log ? <span>FP32 MSE (≈0) drawn at the floor</span> : null}</p>
      <div className="chips" role="group" aria-label="Choose a route">
        {ROWS.map((r) => <button key={r.variant} aria-pressed={sel === r.variant} onClick={() => setSel(r.variant)}>{short(r.variant)}</button>)}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <caption className="hint" style={{ textAlign: 'left', captionSide: 'top', paddingBottom: 6 }}>Full recorded row: <b>{routeName(row.variant)}</b> ({row.variant})</caption>
          <tbody>
            {Object.entries(row).filter(([k]) => k !== 'variant').map(([k, v]) => (
              <tr key={k}><td className="mono">{k}</td><td className="mono">{typeof v === 'number' ? (Number.isInteger(v) ? v : v.toPrecision(6)) : String(v)}{k.startsWith('timing') ? ' (nondeterministic)' : ''}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- decision rule */

function Decision() {
  const ext = (k) => [Math.min(...ROWS.map((r) => r[k])), Math.max(...ROWS.map((r) => r[k]))];
  const [sMin, sMax] = ext('size_bytes'), [aMin, aMax] = ext('test_accuracy'), [tMin, tMax] = ext('timing_batch_ms_median');
  const step = 1 / N_TEST;
  const [maxSize, setMaxSize] = useState(sMax);
  const [minAcc, setMinAcc] = useState(aMin);
  const [maxMs, setMaxMs] = useState(Math.ceil(tMax * 1000) / 1000);
  const [tie, setTie] = useState(0);
  const res = recommendRoute(ROWS, { maxSize, minAccuracy: minAcc - 1e-12, maxBatchMs: maxMs, accuracyTie: tie ? tie / N_TEST : null });
  const strict = recommendRoute(ROWS, { maxSize, minAccuracy: minAcc - 1e-12, maxBatchMs: maxMs });
  const budgetMode = maxSize <= BUDGET;
  return (
    <section id="decide" className="card window reveal" data-title="recommend_route — live">
      <div className="head">
        <div>
          <p className="eyebrow">Decision rule · computed live</p>
          <h2>Set a budget. The repo&apos;s own rule <em>picks the route</em>.</h2>
          <p className="sub">
            A port of <a href={blob('level-3-deep/deep_system.py')}>deep_system.recommend_route</a>, run here on the recorded rows: keep routes inside every limit, then rank by accuracy, logit MSE, size, batch time.
            The tie slider is the optional <span className="mono">accuracy_tie</span> added in this change: accuracy within that many test samples counts as equal, so logit MSE decides.
          </p>
        </div>
        <div className="seg" role="group" aria-label="Presets">
          <button aria-pressed={maxSize === sMax && tie === 0} onClick={() => { setMaxSize(sMax); setTie(0); setMinAcc(aMin); }}>No limits</button>
          <button aria-pressed={maxSize === BUDGET && tie === 0} onClick={() => { setMaxSize(BUDGET); setTie(0); setMinAcc(aMin); }}>Under {BUDGET} B</button>
          <button aria-pressed={maxSize === BUDGET && tie === 1} onClick={() => { setMaxSize(BUDGET); setTie(1); setMinAcc(aMin); }}>Under {BUDGET} B, 1-sample tie</button>
        </div>
      </div>
      <div className="controls">
        <Range label="Max size" value={maxSize} min={sMin - 1} max={sMax} step={1} onChange={setMaxSize} show={`${fmt(maxSize)} B`} />
        <Range label="Min accuracy" value={minAcc} min={aMin} max={aMax} step={step} onChange={setMinAcc} show={`${Math.round(minAcc * N_TEST)} of ${N_TEST}`} />
        <Range label="Max batch ms (noisy)" value={maxMs} min={Math.floor(tMin * 1000) / 1000} max={Math.ceil(tMax * 1000) / 1000} step={0.001} onChange={setMaxMs} show={`${maxMs.toFixed(3)} ms`} />
        <Range label="Accuracy tie" value={tie} min={0} max={6} step={1} onChange={setTie} show={tie ? `${tie} sample${tie > 1 ? 's' : ''}` : 'off (original rule)'} />
      </div>
      <div className="grid2" style={{ alignItems: 'start' }}>
        <div className="tile verdict">
          <p className="eyebrow">Winner</p>
          {res.winner ? <>
            <div className="num">{routeName(res.winner.variant)}</div>
            <div className="cap">{fmt(res.winner.size_bytes)} B · {res.winner.test_correct} of {res.winner.test_total} · logit MSE {f6(res.winner.logit_mse_vs_fp32)}</div>
          </> : <div className="cap" role="status">No route meets every limit. recommend_route returns None.</div>}
          <p className="hint" style={{ marginTop: 10 }}>Ranking key: {tie ? `accuracy within ${tie}/${N_TEST} of the best is tied, then logit MSE ↑, size ↑, batch ms ↑` : 'accuracy ↓, logit MSE ↑, size ↑, batch ms ↑'}</p>
          {budgetMode && strict.winner && res.winner && (
            <p className={`note ${tie ? 'good' : 'warn'}`}>
              {tie && res.winner.variant !== strict.winner.variant
                ? <>With the tie, the code now agrees with the README verdict (&ldquo;pick plain INT4 per-channel packed&rdquo;). Without it, the code picks {routeName(strict.winner.variant)}.</>
                : <>Recorded disagreement: under {BUDGET} bytes the code picks <b>{routeName(strict.winner.variant)}</b> ({strict.winner.test_correct} of {N_TEST}), while the README verdict picks plain INT4 packed ({BY.int4_per_channel_packed.test_correct} of {N_TEST}) because a one-sample edge with {(strict.winner.logit_mse_vs_fp32 / BY.int4_per_channel_packed.logit_mse_vs_fp32).toFixed(0)}× the logit MSE is noise. Try the 1-sample tie.</>}
            </p>
          )}
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead><tr><th>Survivor ({res.survivors.length})</th><th>Bytes</th><th>Correct</th><th>Logit MSE</th></tr></thead>
            <tbody>
              {res.survivors.map((r) => (
                <tr key={r.variant} className={res.winner?.variant === r.variant ? 'win' : ''}><td>{routeName(r.variant)}{res.tied && !res.tied.includes(r) ? ' (outside tie)' : ''}</td><td>{fmt(r.size_bytes)}</td><td>{r.test_correct}</td><td>{f6(r.logit_mse_vs_fp32)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- uncertainty */

function Noise() {
  const ivs = ROWS.map((r) => ({ r, ci: wilson(r.test_correct, r.test_total) }));
  const fpCi = ivs[0].ci;
  const lo = Math.min(...ivs.map((d) => d.ci[0])), hi = Math.max(...ivs.map((d) => d.ci[1]));
  const W = 720, rowH = 28, l = 190, rgt = 70, H = ivs.length * rowH + 30;
  const x = (v) => l + ((v - lo) / (hi - lo)) * (W - l - rgt);
  const mc = seeds.canonical.mcnemar, sm = seeds.summary, dc = seeds.decision_counts;
  const worse = ivs.filter((d) => d.ci[1] < fpCi[0]);
  return (
    <section id="noise" className="card window reveal" data-title="noise.check — Wilson 95% · seed study">
      <p className="eyebrow">Uncertainty · derived statistics</p>
      <h2>At {N_TEST} test samples, most gaps are <em>inside the noise</em>.</h2>
      <p className="sub">
        Wilson 95% intervals computed in your browser from each row&apos;s recorded test_correct and test_total (a derived statistic, not a rerun). Every route except
        {' '}{worse.map((d) => routeName(d.r.variant)).join(', ')} overlaps the FP32 interval. The exact McNemar p-value against FP32 uses the per-sample predictions recorded by the new seed study, whose seed 0 reproduces the Deep run exactly.
      </p>
      <svg className="chart dense" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Wilson 95 percent intervals for test accuracy by route">
        <rect x={x(fpCi[0])} width={x(fpCi[1]) - x(fpCi[0])} y={0} height={H - 22} fill="var(--primary)" opacity="0.08" />
        {[lo, (lo + hi) / 2, hi].map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle">{t.toFixed(3)}</text>)}
        {ivs.map(({ r, ci }, i) => {
          const out = ci[1] < fpCi[0];
          const c = out ? 'var(--bad)' : r.variant === 'fp32' ? 'var(--primary)' : 'var(--good)';
          return (
            <g key={r.variant} transform={`translate(0 ${i * rowH + 16})`}>
              <text x={0} y={4} style={{ fill: 'var(--text)' }}>{short(r.variant)}</text>
              <line x1={x(ci[0])} x2={x(ci[1])} y1={0} y2={0} stroke={c} strokeWidth="3" strokeLinecap="round" opacity="0.7" />
              <circle cx={x(r.test_accuracy)} cy={0} r="5" fill={c} />
              <text x={W - 4} y={4} textAnchor="end">{r.test_correct}/{r.test_total}</text>
            </g>
          );
        })}
      </svg>
      <p className="hint">Blue band: FP32 interval. Overlapping intervals are a conservative screen, not a formal test; the McNemar column below is the paired test.</p>
      <h3 style={{ marginTop: 20 }}>Seed study: the same {ROWS.length} routes over {seeds.seeds} seed sets</h3>
      <p className="hint">Training init and the train, calibration and test draws vary; class centers are fixed. Recorded in <a href={blob('level-3-deep/results_seeds.json')}>results_seeds.json</a> (NumPy {seeds.environment.numpy}, no timing).</p>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Route</th><th>Wilson 95% (seed 0)</th><th>McNemar p vs FP32 (seed 0)</th><th>Accuracy mean ± sd ({seeds.seeds} seeds)</th><th>Δ correct vs FP32, range</th><th>Seeds with p &lt; 0.05</th></tr></thead>
          <tbody>
            {ivs.map(({ r, ci }) => (
              <tr key={r.variant}>
                <td>{routeName(r.variant)}</td><td>{ci[0].toFixed(3)} to {ci[1].toFixed(3)}</td>
                <td className={mc[r.variant].mcnemar_p < 0.05 ? 'no' : ''}>{r.variant === 'fp32' ? 'n/a' : `${mc[r.variant].mcnemar_p.toPrecision(3)} (${mc[r.variant].mcnemar_b} lost, ${mc[r.variant].mcnemar_c} gained)`}</td>
                <td>{f4(sm[r.variant].accuracy_mean)} ± {f4(sm[r.variant].accuracy_sd)}</td>
                <td>{sm[r.variant].delta_correct_vs_fp32_min} to {sm[r.variant].delta_correct_vs_fp32_max}</td>
                <td>{sm[r.variant].seeds_mcnemar_p_below_0_05} of {seeds.seeds}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid2" style={{ marginTop: 16 }}>
        {[['under_1200_strict', 'Original rule'], ['under_1200_one_sample_tie', 'With a 1-sample tie']].map(([k, t]) => (
          <div className="tile" key={k}>
            <h3>{t}: winner under {BUDGET} B across seeds</h3>
            {Object.entries(dc[k]).map(([v, n]) => <p className="hint" style={{ margin: '4px 0 0' }} key={v}><b>{n} of {seeds.seeds}</b> · {routeName(v)}</p>)}
          </div>
        ))}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- pruning */

function Pruning() {
  const rows = ROWS.filter((r) => r.variant === 'fp32' || /^pruned_[\d.]+$/.test(r.variant)).map((r) => ({ r, s: r.sparsity ?? 0 }));
  const maxMse = Math.max(...rows.map((d) => d.r.logit_mse_vs_fp32));
  return (
    <section id="prune" className="card window reveal" data-title="prune.log — recorded">
      <p className="eyebrow">Pruning reality · recorded</p>
      <h2>Accuracy hides the damage. The <em>logits</em> don&apos;t.</h2>
      <p className="sub">
        Magnitude pruning with no fine-tuning and dense FP32 storage (NumPy equivalent track). The file never shrinks, accuracy moves by a few samples until high sparsity,
        and logit MSE against FP32 climbs fast. These rows are a lower bound on pruned quality, not a fair test of pruning with recovery.
      </p>
      <div className="bars" style={{ marginTop: 16 }}>
        {rows.map(({ r, s }) => (
          <div className="prune-row" key={r.variant}>
            <div className="lab"><span>Sparsity {s}</span><b>{r.test_correct} of {r.test_total}</b></div>
            <div className="pair">
              <div><div className="hint">Size {fmt(r.size_bytes)} B</div><div className="track"><div className="fill" style={{ width: `${(r.size_bytes / FP32.size_bytes) * 100}%`, background: 'var(--primary)' }} /></div></div>
              <div><div className="hint">Logit MSE {f6(r.logit_mse_vs_fp32)}</div><div className="track"><div className="fill" style={{ width: `${Math.max(0.5, (r.logit_mse_vs_fp32 / maxMse) * 100)}%`, background: 'var(--bad)' }} /></div></div>
            </div>
          </div>
        ))}
      </div>
      <p className="hint" style={{ marginTop: 12 }}>Size bars are relative to FP32 ({fmt(FP32.size_bytes)} B); MSE bars are relative to the largest recorded value ({f6(maxMse)}).</p>
    </section>
  );
}

/* ---------------------------------------------------------------- levels */

function Levels() {
  const s = learn.summary;
  const brow = Object.fromEntries(build.rows.map((r) => [r.variant, r]));
  const rebuilt = ROWS.filter((r) => brow[r.variant]);
  const same = (r) => { const b = brow[r.variant]; return b.size_bytes === r.size_bytes && b.test_correct === r.test_correct && b.logit_mse_vs_fp32 === r.logit_mse_vs_fp32; };
  const allSame = rebuilt.every(same);
  const steps = [
    ['Learn', `${s.weight_count} weights, stdlib only`, `FP32 ${s.fp32_total_bytes} B → INT8 ${s.int8_total_bytes} B → INT4 ${s.int4_total_bytes} B; all ${s.task_samples} task samples correct at every width.`, 'level-1-learn/README.md'],
    ['Build', build.model.architecture, `${build.rows.length} variants in the QLAB container; INT8 per-channel ${brow.int8_per_channel.test_correct} of ${brow.int8_per_channel.test_total} at ${fmt(brow.int8_per_channel.size_bytes)} B.`, 'level-2-build/README.md'],
    ['Deep', `${ROWS.length} routes, Pareto + decision rule`, `${deep.verdict_inputs.pareto_routes.length} Pareto routes; smallest is ${routeName(deep.verdict_inputs.smallest_route)} at ${fmt(BY[deep.verdict_inputs.smallest_route].size_bytes)} B.`, 'level-3-deep/README.md'],
  ];
  return (
    <section id="levels" className="card window reveal" data-title="levels/ — Learn · Build · Deep">
      <p className="eyebrow">Build ladder</p>
      <h2>Three levels, each one <em>reusing</em> the last by import.</h2>
      <ol className="steps" style={{ marginTop: 16 }}>
        {steps.map(([t, k, d, p], i) => (
          <li key={t}><span className="n">{i + 1}</span><div><b>{t}</b> · <span className="hint">{k}</span><div className="hint">{d} <a href={blob(p)}>README</a></div></div></li>
        ))}
      </ol>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>Deep rebuilds the {rebuilt.length} Build rows</h3><span className={`pill ${allSame ? 'good' : 'bad'}`}>{allSame ? '✓ exact reproduction' : 'differs'}</span></div>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Route</th><th>Bytes</th><th>Correct</th><th>Logit MSE</th><th>Match</th></tr></thead>
              <tbody>{rebuilt.map((r) => <tr key={r.variant}><td>{short(r.variant)}</td><td>{fmt(r.size_bytes)}</td><td>{r.test_correct}</td><td>{f6(r.logit_mse_vs_fp32)}</td><td className={same(r) ? 'ok' : 'no'}>{same(r) ? 'same' : 'differs'}</td></tr>)}</tbody>
            </table>
          </div>
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Calibration: clip percentile chosen per matrix</h3>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Route</th><th>W1</th><th>W2</th></tr></thead>
              <tbody>{Object.entries(deep.calibration).map(([k, c]) => <tr key={k}><td>{short(k)}</td><td>{c.W1.clip_percentile}</td><td>{c.W2.clip_percentile}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="hint">Most matrices keep percentile 100 (no clipping), so &ldquo;calibrated&rdquo; changes little at this scale.</p>
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- limits */

function Limits() {
  const backlog = [
    ['Real pretrained model', 'Quantize a small open model so the study sees real weight distributions and outlier channels.'],
    ['GGUF and llama.cpp', 'Produce and validate a GGUF file; QLAB stays the teaching format.'],
    ['GPU window', 'Re-time the routes on the GPU node assigned to this project.'],
    ['Real quantized kernel', 'Time an actual INT8 or INT4 execution path so the speed axis can separate routes.'],
    ['TensorFlow row', 'Replace the NumPy equivalent pruning rows with a TensorFlow Model Optimization result.'],
    ['Pruning with recovery', 'Sparse storage and fine-tuning after pruning, then re-measure.'],
    ['Scale sweep', 'Repeat across model sizes and a harder task.'],
    ['Activation and asymmetric quantization; GPTQ and AWQ rows', 'Weight-only symmetric round-to-nearest is all that is measured here.'],
  ];
  return (
    <section id="limits" className="card window reveal" data-title="ROADMAP.md — limits and v1.1 backlog">
      <p className="eyebrow">Limits and roadmap</p>
      <h2>What this study <em>cannot</em> tell you.</h2>
      <p className="sub">One {PARAMS}-parameter MLP on synthetic Gaussian-mixture data, fake quantization on CPU, weights only. It shows the shape of a quantization study, not numbers that transfer to LLMs.</p>
      <ul className="scope">{deep.substitutions.map((s) => <li key={s}>{s}</li>)}</ul>
      <h3 style={{ marginTop: 20 }}>v1.1 backlog: not built, not measured</h3>
      <div className="docs" style={{ marginTop: 12 }}>
        {backlog.map(([t, d]) => <a key={t} href={blob('ROADMAP.md')}><b>{t}</b><small>{d}</small></a>)}
      </div>
      <h3 style={{ marginTop: 24 }}>Reproduce</h3>
      <ol className="steps" style={{ marginTop: 12 }} id="reproduce">
        <li><span className="n">1</span><div><b>Clone and install</b><div className="hint"><code>git clone {REPO}.git && pip install -r requirements.txt</code></div></div></li>
        <li><span className="n">2</span><div><b>Run the study</b><div className="hint"><code>python level-3-deep/run_deep.py</code> (timing fields change per run) and <code>python level-3-deep/seed_study.py</code></div></div></li>
        <li><span className="n">3</span><div><b>Run the tests</b><div className="hint"><code>python -m unittest discover -s tests</code></div></div></li>
      </ol>
      <div className="docs" style={{ marginTop: 20 }}>
        <a href={blob('docs/research-review.md')}><b>Research review</b><small>Rubric grades, fixes and remaining gaps</small></a>
        <a href={blob('level-3-deep/README.md')}><b>Deep study</b><small>Verdict, failure modes, grade</small></a>
        <a href={blob('ROADMAP.md')}><b>Roadmap</b><small>The v1.1 backlog</small></a>
      </div>
    </section>
  );
}
