"""
Generates the static training dashboard written into a training run's
output directory. The dashboard polls two sibling files over HTTP:

  - run_meta.json        (written once, at training start)
  - training_log.jsonl   (appended to throughout training)

and renders:
  - An interactive architecture diagram built once from run_meta.json's cfg
    (token embedding -> N transformer blocks -> tied output projection),
    with parameter counts computed client-side.
  - Live-updating loss / perplexity / aux-loss / learning-rate charts,
    redrawn on each poll as new lines are appended to the log.
  - A hand-draggable slider that scrubs through logged expert-utilization
    snapshots, with an autoplay button, so routing balance over training
    can be replayed rather than only watched live.

This file has no dependency on mlx/numpy/tokenizers -- it only writes
static text -- so it's safe to import from train.py without pulling in
anything heavy beyond what train.py already imports.
"""
import json
import os


def write_run_meta(out_dir, cfg, args_dict, total_params):
    """Write the one-time architecture/run metadata file the dashboard reads."""
    meta = {
        "cfg": cfg.__dict__,
        "args": args_dict,
        "total_params": total_params,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_dashboard(out_dir):
    """Write dashboard.html into out_dir, alongside where the log will live."""
    with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(DASHBOARD_HTML)


def append_log(out_dir, entry):
    """Append one JSON line to training_log.jsonl."""
    with open(os.path.join(out_dir, "training_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# The HTML template below is markup/CSS/JS, not Python logic -- line-length
# and docstring conventions for Python don't meaningfully apply to it.
# pylint: disable=line-too-long
DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Training Dashboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --border: #2a2e37;
    --text: #e6e8eb; --muted: #8b92a3; --accent: #5aa9e6; --good: #7ce65a;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 24px;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 20px;
  }
  .status-row { display: flex; gap: 24px; flex-wrap: wrap; align-items: center; margin-bottom: 20px; }
  .status-item { }
  .status-value { font-size: 22px; font-weight: 700; }
  .status-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
  .live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--good); margin-right: 6px;
  }
  .live-dot.stale { background: var(--muted); }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-title { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .chart-svg { width: 100%; height: 140px; display: block; }
  .arch-flow { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .arch-box {
    background: #0f1115; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 16px; text-align: center; min-width: 220px; cursor: default;
  }
  .arch-box:hover { border-color: var(--accent); }
  .arch-box-title { font-size: 13px; font-weight: 600; }
  .arch-box-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .arch-arrow { color: var(--muted); font-size: 14px; }
  .arch-block-inner {
    background: #12141a; border: 1px dashed var(--border); border-radius: 8px;
    padding: 12px; margin-top: 6px; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
  }
  .arch-subbox { background: #0f1115; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; min-width: 150px; }
  .arch-subbox-title { font-size: 11px; font-weight: 600; color: var(--text); }
  .arch-subbox-sub { font-size: 10px; color: var(--muted); margin-top: 3px; line-height: 1.4; }
  .expert-boxes { display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; justify-content: center; }
  .expert-box {
    width: 26px; height: 22px; border-radius: 4px; font-size: 9px;
    display: flex; align-items: center; justify-content: center; color: #0f1115; font-weight: 700;
  }
  .arch-tie-note { font-size: 11px; color: var(--accent); margin-top: 6px; text-align: center; }
  .arch-total { font-size: 12px; color: var(--muted); margin-top: 10px; text-align: center; }
  .scrubber-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .scrubber-row input[type=range] { flex: 1; }
  .scrubber-step-label { font-size: 12px; color: var(--muted); min-width: 90px; }
  .play-btn {
    background: var(--panel); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px;
  }
  .play-btn:hover { border-color: var(--accent); }
  .util-layer-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .util-layer-label { font-size: 11px; color: var(--muted); width: 56px; flex-shrink: 0; }
  .util-bars-inline { display: flex; gap: 3px; flex: 1; height: 20px; align-items: flex-end; }
  .util-bar-inline { flex: 1; border-radius: 2px 2px 0 0; min-height: 2px; }
  .empty-note { font-size: 13px; color: var(--muted); text-align: center; padding: 20px; }
</style>
</head>
<body>

<h1>Training Dashboard</h1>
<div class="subtitle" id="subtitle">Waiting for run_meta.json...</div>

<div class="panel">
  <div class="status-row" id="statusRow"></div>
</div>

<div class="panel">
  <h2>Architecture</h2>
  <div id="archContainer" class="empty-note">Loading...</div>
</div>

<div class="panel">
  <h2>Live training curves</h2>
  <div class="charts-grid" id="chartsGrid"></div>
</div>

<div class="panel">
  <h2>Expert utilization over training (scrub or autoplay)</h2>
  <div id="scrubberContainer" class="empty-note">No diagnostic snapshots logged yet.</div>
</div>

<script>
let runMeta = null;
let trainEntries = [];
let evalEntries = [];
let diagEntries = [];
let lastLogLength = 0;
let scrubIndex = 0;
let autoplayTimer = null;

const expertColors = [
  "#5aa9e6", "#e6785a", "#7ce65a", "#e6c15a", "#b05ae6",
  "#5ae6c1", "#e65a9e", "#a3e65a", "#5a78e6", "#e6a35a"
];
function colorForExpert(idx) { return expertColors[idx % expertColors.length]; }

// ---------- data loading ----------

async function pollOnce() {
  try {
    if (!runMeta) {
      const r = await fetch("run_meta.json", { cache: "no-store" });
      if (r.ok) {
        runMeta = await r.json();
        renderArchitecture();
        document.getElementById("subtitle").textContent =
          "Model with " + runMeta.total_params.toLocaleString() + " parameters -- polling training_log.jsonl every 2s";
      }
    }

    const logResp = await fetch("training_log.jsonl", { cache: "no-store" });
    if (logResp.ok) {
      const text = await logResp.text();
      const lines = text.split("\\n").filter((l) => l.trim().length > 0);
      const isNew = lines.length !== lastLogLength;
      lastLogLength = lines.length;

      trainEntries = []; evalEntries = []; diagEntries = [];
      for (const line of lines) {
        try {
          const entry = JSON.parse(line);
          if (entry.type === "train") trainEntries.push(entry);
          else if (entry.type === "eval") evalEntries.push(entry);
          else if (entry.type === "diag") diagEntries.push(entry);
        } catch (e) { /* tolerate a partially-written last line */ }
      }

      renderStatus(isNew);
      renderCharts();
      renderScrubber();
    }
  } catch (e) {
    renderStatus(false, true);
  }
}

// ---------- status ----------

function renderStatus(isNew, errored) {
  const container = document.getElementById("statusRow");
  const last = trainEntries[trainEntries.length - 1];
  const lastEval = evalEntries[evalEntries.length - 1];
  container.innerHTML = "";

  const items = [
    { label: "status", value: errored ? "Error polling log" : (isNew ? "Live" : "No new data"), dot: true, stale: !isNew || errored },
    { label: "step", value: last ? last.step.toLocaleString() : "--" },
    { label: "train ce", value: last ? last.ce.toFixed(4) : "--" },
    { label: "val ce", value: lastEval ? lastEval.val_ce.toFixed(4) : "--" },
    { label: "aux loss", value: last ? last.aux.toFixed(4) : "--" },
    { label: "lr", value: last ? last.lr.toExponential(2) : "--" },
    { label: "tok/s", value: last && last.tok_per_sec ? Math.round(last.tok_per_sec).toLocaleString() : "--" },
  ];

  for (const item of items) {
    const div = document.createElement("div");
    div.className = "status-item";
    const valueDiv = document.createElement("div");
    valueDiv.className = "status-value";
    if (item.dot) {
      const dot = document.createElement("span");
      dot.className = "live-dot" + (item.stale ? " stale" : "");
      valueDiv.appendChild(dot);
      valueDiv.appendChild(document.createTextNode(item.value));
    } else {
      valueDiv.textContent = item.value;
    }
    const labelDiv = document.createElement("div");
    labelDiv.className = "status-label";
    labelDiv.textContent = item.label;
    div.appendChild(valueDiv);
    div.appendChild(labelDiv);
    container.appendChild(div);
  }
}

// ---------- architecture diagram ----------

function estimateParams(cfg) {
  const headDim = Math.floor(cfg.d_model / cfg.n_heads);
  const attnParams = 4 * cfg.d_model * (cfg.n_heads * headDim); // wq,wk,wv,wo
  const hidden = Math.floor(cfg.d_model * cfg.expert_hidden_mult);
  const expertParams = 3 * cfg.d_model * hidden; // w1, w2, w3
  const moeParams = cfg.n_experts * expertParams + cfg.d_model * cfg.n_experts; // experts + gate
  const normParams = 2 * cfg.d_model; // attn_norm + moe_norm
  const blockParams = attnParams + moeParams + normParams;
  const embParams = cfg.vocab_size * cfg.d_model; // tied: counted once
  const finalNormParams = cfg.d_model;
  return {
    embParams, blockParams, finalNormParams,
    total: embParams + cfg.n_layers * blockParams + finalNormParams,
    attnParams, moeParams, expertParams, hidden, headDim,
  };
}

function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function renderArchitecture() {
  const cfg = runMeta.cfg;
  const p = estimateParams(cfg);
  const container = document.getElementById("archContainer");
  container.className = "";
  container.innerHTML = "";

  const flow = document.createElement("div");
  flow.className = "arch-flow";

  function addBox(title, sub) {
    const box = document.createElement("div");
    box.className = "arch-box";
    const t = document.createElement("div");
    t.className = "arch-box-title";
    t.textContent = title;
    const s = document.createElement("div");
    s.className = "arch-box-sub";
    s.textContent = sub;
    box.appendChild(t);
    box.appendChild(s);
    flow.appendChild(box);
  }

  function addArrow() {
    const a = document.createElement("div");
    a.className = "arch-arrow";
    a.textContent = "\\u2193";
    flow.appendChild(a);
  }

  addBox("Token Embedding", cfg.vocab_size.toLocaleString() + " x " + cfg.d_model + "  (" + fmt(p.embParams) + " params)");
  addArrow();

  const blockWrap = document.createElement("div");
  blockWrap.className = "arch-box";
  blockWrap.style.minWidth = "460px";
  const blockTitle = document.createElement("div");
  blockTitle.className = "arch-box-title";
  blockTitle.textContent = "Transformer Block \u00d7 " + cfg.n_layers;
  const blockSub = document.createElement("div");
  blockSub.className = "arch-box-sub";
  blockSub.textContent = fmt(p.blockParams) + " params/block \u00d7 " + cfg.n_layers + " = " + fmt(p.blockParams * cfg.n_layers);
  blockWrap.appendChild(blockTitle);
  blockWrap.appendChild(blockSub);

  const inner = document.createElement("div");
  inner.className = "arch-block-inner";

  const attnBox = document.createElement("div");
  attnBox.className = "arch-subbox";
  attnBox.innerHTML = "<div class=\\"arch-subbox-title\\">Self-Attention (causal + RoPE)</div>" +
    "<div class=\\"arch-subbox-sub\\">" + cfg.n_heads + " heads \u00d7 " + p.headDim + " dim<br>" +
    fmt(p.attnParams) + " params</div>";

  const moeBox = document.createElement("div");
  moeBox.className = "arch-subbox";
  moeBox.style.minWidth = "220px";
  const moeTitle = document.createElement("div");
  moeTitle.className = "arch-subbox-title";
  moeTitle.textContent = "MoE FFN (top-" + cfg.top_k + " of " + cfg.n_experts + ")";
  const moeSub = document.createElement("div");
  moeSub.className = "arch-subbox-sub";
  moeSub.textContent = fmt(p.expertParams) + " params/expert \u00d7 " + cfg.n_experts + " = " + fmt(p.moeParams);
  const expertBoxes = document.createElement("div");
  expertBoxes.className = "expert-boxes";
  for (let e = 0; e < cfg.n_experts; e++) {
    const eb = document.createElement("div");
    eb.className = "expert-box";
    eb.style.background = colorForExpert(e);
    eb.textContent = "E" + e;
    expertBoxes.appendChild(eb);
  }
  moeBox.appendChild(moeTitle);
  moeBox.appendChild(moeSub);
  moeBox.appendChild(expertBoxes);

  inner.appendChild(attnBox);
  inner.appendChild(moeBox);
  blockWrap.appendChild(inner);
  flow.appendChild(blockWrap);
  addArrow();

  addBox("Final Norm", fmt(p.finalNormParams) + " params");
  addArrow();

  addBox("Output Projection", cfg.d_model + " x " + cfg.vocab_size.toLocaleString());
  const tieNote = document.createElement("div");
  tieNote.className = "arch-tie-note";
  tieNote.textContent = "\\u2191 tied to Token Embedding above -- no separate parameters, reuses the same matrix";
  flow.appendChild(tieNote);

  const totalNote = document.createElement("div");
  totalNote.className = "arch-total";
  totalNote.textContent = "Total: ~" + fmt(p.total) + " parameters (approximate, computed client-side from config)";
  flow.appendChild(totalNote);

  container.appendChild(flow);
}

// ---------- charts ----------

function drawLineChart(containerId, series, opts) {
  opts = opts || {};
  const width = 480, height = 140, padL = 44, padB = 20, padT = 10, padR = 10;
  const container = document.getElementById(containerId);

  const allPoints = series.flatMap((s) => s.points);
  if (allPoints.length === 0) {
    container.innerHTML = "<div class=\\"empty-note\\">No data yet</div>";
    return;
  }

  const xs = allPoints.map((p) => p[0]);
  const ys = allPoints.map((p) => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const yRange = (yMax - yMin) || 1;
  const xRange = (xMax - xMin) || 1;

  function sx(x) { return padL + ((x - xMin) / xRange) * (width - padL - padR); }
  function sy(y) { return height - padB - ((y - yMin) / yRange) * (height - padT - padB); }

  let svg = "<svg class=\\"chart-svg\\" viewBox=\\"0 0 " + width + " " + height + "\\">";
  svg += "<text x=\\"4\\" y=\\"" + (padT + 4) + "\\" fill=\\"#8b92a3\\" font-size=\\"9\\">" + yMax.toFixed(opts.decimals || 3) + "</text>";
  svg += "<text x=\\"4\\" y=\\"" + (height - padB) + "\\" fill=\\"#8b92a3\\" font-size=\\"9\\">" + yMin.toFixed(opts.decimals || 3) + "</text>";
  svg += "<text x=\\"" + padL + "\\" y=\\"" + (height - 4) + "\\" fill=\\"#8b92a3\\" font-size=\\"9\\">step " + Math.round(xMin) + "</text>";
  svg += "<text x=\\"" + (width - padR - 40) + "\\" y=\\"" + (height - 4) + "\\" fill=\\"#8b92a3\\" font-size=\\"9\\">step " + Math.round(xMax) + "</text>";

  for (const s of series) {
    if (s.points.length === 0) continue;
    if (s.style === "dots") {
      for (const [x, y] of s.points) {
        svg += "<circle cx=\\"" + sx(x) + "\\" cy=\\"" + sy(y) + "\\" r=\\"2.5\\" fill=\\"" + s.color + "\\" />";
      }
    } else {
      const pts = s.points.map(([x, y]) => sx(x) + "," + sy(y)).join(" ");
      svg += "<polyline points=\\"" + pts + "\\" fill=\\"none\\" stroke=\\"" + s.color + "\\" stroke-width=\\"1.5\\" />";
    }
  }
  svg += "</svg>";
  container.innerHTML = svg;
}

function renderCharts() {
  const grid = document.getElementById("chartsGrid");
  if (grid.children.length === 0) {
    ["ce", "ppl", "aux", "lr"].forEach((id) => {
      const wrap = document.createElement("div");
      const title = document.createElement("div");
      title.className = "chart-title";
      title.textContent = { ce: "Cross-entropy (train + val)", ppl: "Perplexity (train + val)", aux: "MoE aux (load-balancing) loss", lr: "Learning rate" }[id];
      const chartDiv = document.createElement("div");
      chartDiv.id = "chart-" + id;
      wrap.appendChild(title);
      wrap.appendChild(chartDiv);
      grid.appendChild(wrap);
    });
  }

  const trainCe = trainEntries.map((e) => [e.step, e.ce]);
  const valCe = evalEntries.map((e) => [e.step, e.val_ce]);
  drawLineChart("chart-ce", [
    { points: trainCe, color: "#5aa9e6" },
    { points: valCe, color: "#7ce65a", style: "dots" },
  ]);

  const trainPpl = trainEntries.map((e) => [e.step, e.ppl]);
  const valPpl = evalEntries.map((e) => [e.step, e.val_ppl]);
  drawLineChart("chart-ppl", [
    { points: trainPpl, color: "#5aa9e6" },
    { points: valPpl, color: "#7ce65a", style: "dots" },
  ], { decimals: 1 });

  const aux = trainEntries.map((e) => [e.step, e.aux]);
  drawLineChart("chart-aux", [{ points: aux, color: "#e6c15a" }]);

  const lr = trainEntries.map((e) => [e.step, e.lr]);
  drawLineChart("chart-lr", [{ points: lr, color: "#b05ae6" }], { decimals: 6 });
}

// ---------- expert utilization scrubber ----------

function renderScrubber() {
  const container = document.getElementById("scrubberContainer");
  if (diagEntries.length === 0) {
    container.className = "empty-note";
    container.innerHTML = "No diagnostic snapshots logged yet.";
    return;
  }
  container.className = "";
  if (scrubIndex >= diagEntries.length) scrubIndex = diagEntries.length - 1;

  container.innerHTML = "";

  const row = document.createElement("div");
  row.className = "scrubber-row";

  const playBtn = document.createElement("button");
  playBtn.className = "play-btn";
  playBtn.textContent = autoplayTimer ? "\\u23f8 Pause" : "\\u25b6 Autoplay";
  playBtn.onclick = toggleAutoplay;

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = 0;
  slider.max = diagEntries.length - 1;
  slider.value = scrubIndex;
  slider.oninput = (e) => { scrubIndex = parseInt(e.target.value); renderScrubber(); };

  const stepLabel = document.createElement("div");
  stepLabel.className = "scrubber-step-label";
  stepLabel.textContent = "step " + diagEntries[scrubIndex].step.toLocaleString();

  row.appendChild(playBtn);
  row.appendChild(slider);
  row.appendChild(stepLabel);
  container.appendChild(row);

  const snapshot = diagEntries[scrubIndex];
  snapshot.per_layer_utilization.forEach((counts, layerIdx) => {
    const layerRow = document.createElement("div");
    layerRow.className = "util-layer-row";
    const label = document.createElement("div");
    label.className = "util-layer-label";
    label.textContent = "Layer " + layerIdx;
    const bars = document.createElement("div");
    bars.className = "util-bars-inline";
    const maxCount = Math.max(1, ...counts);
    counts.forEach((count, e) => {
      const bar = document.createElement("div");
      bar.className = "util-bar-inline";
      bar.style.height = Math.max(2, (count / maxCount) * 100) + "%";
      bar.style.background = colorForExpert(e);
      bar.title = "Expert " + e + ": " + count;
      bars.appendChild(bar);
    });
    layerRow.appendChild(label);
    layerRow.appendChild(bars);
    container.appendChild(layerRow);
  });
}

function toggleAutoplay() {
  if (autoplayTimer) {
    clearInterval(autoplayTimer);
    autoplayTimer = null;
  } else {
    autoplayTimer = setInterval(() => {
      scrubIndex = (scrubIndex + 1) % diagEntries.length;
      renderScrubber();
    }, 600);
  }
  renderScrubber();
}

pollOnce();
setInterval(pollOnce, 2000);
</script>

</body>
</html>
"""
# pylint: enable=line-too-long