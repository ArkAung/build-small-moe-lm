"""
Teaching tool: run one prompt through a trained checkpoint and generate a
single self-contained HTML file that visualizes, per layer:

  - Attention weights (heatmap, selectable head)
  - Router decisions (which expert each token was routed to, top-1 and top-2)
  - Expert utilization (how many tokens in this prompt went to each expert)

Designed to be opened directly in a browser -- no server needed, all data
is embedded inline as JSON.

Usage:
    python inspect_model.py \
        --checkpoint checkpoints/final.safetensors \
        --config checkpoints/config.json \
        --tokenizer data/tokenizer.json \
        --prompt "Once upon a time there was a" \
        --out inspection.html
"""
import argparse
import json

import numpy as np
import mlx.core as mx

from common import load_model_and_tokenizer


def to_list(x):
    """Convert an mx.array to nested python lists for JSON serialization."""
    return np.array(x).astype(np.float32).tolist()


def display_token(tok, token_id):
    """Turn a byte-level-BPE token string into something readable for display."""
    piece = tok.id_to_token(token_id)
    if piece is None:
        return "<unk>"
    # byte-level BPE marks a leading space with 'Ġ'
    piece = piece.replace("\u0120", " ").replace("\u010a", "\\n")
    return piece if piece.strip() != "" else piece.replace(" ", "\u2423")


def main():
    """Parse args, run one forward pass with capture=True, and write the HTML view."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a")
    parser.add_argument("--out", type=str, default="inspection.html")
    args = parser.parse_args()

    model, cfg, tok = load_model_and_tokenizer(args.checkpoint, args.config, args.tokenizer)

    bos_id = tok.token_to_id("<bos>")
    ids = [bos_id] + tok.encode(args.prompt).ids
    tokens = mx.array([ids])

    logits, aux_loss, _, captures = model(tokens, capture=True)
    mx.eval(logits, captures)

    n_tokens = len(ids)
    token_labels = [display_token(tok, i) for i in ids]

    layers_data = []
    for layer_idx, cap in enumerate(captures):
        attn_weights = cap["attn"]["attn_weights"]        # (1, n_heads, T, T)
        n_heads = attn_weights.shape[1]
        attn_per_head = [to_list(attn_weights[0, h]) for h in range(n_heads)]

        topk_idx = cap["moe"]["topk_idx"][0]               # (T, top_k)
        topk_probs = cap["moe"]["topk_probs"][0]           # (T, top_k)
        router_probs = cap["moe"]["router_probs"][0]       # (T, n_experts)

        topk_idx_list = np.array(topk_idx).astype(int).tolist()
        topk_probs_list = to_list(topk_probs)
        router_probs_list = to_list(router_probs)

        # expert utilization: count of tokens whose top-1 choice is each expert
        top1_choices = [row[0] for row in topk_idx_list]
        utilization = [top1_choices.count(e) for e in range(cfg.n_experts)]

        layers_data.append({
            "layer": layer_idx,
            "n_heads": n_heads,
            "attn_per_head": attn_per_head,
            "topk_idx": topk_idx_list,
            "topk_probs": topk_probs_list,
            "router_probs": router_probs_list,
            "utilization": utilization,
        })

    data = {
        "prompt": args.prompt,
        "tokens": token_labels,
        "n_layers": cfg.n_layers,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "aux_loss": float(aux_loss.item()),
        "layers": layers_data,
    }

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote inspection view -> {args.out}")
    print(f"Prompt tokenized to {n_tokens} tokens. Open {args.out} in a browser.")


# The HTML template below is markup/CSS/JS, not Python logic -- line-length
# and docstring conventions for Python don't meaningfully apply to it.
# pylint: disable=line-too-long
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MoE Model Inspector</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --border: #2a2e37;
    --text: #e6e8eb; --muted: #8b92a3; --accent: #5aa9e6;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 24px;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .prompt-box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 14px;
  }
  .intro-panel { font-size: 13px; line-height: 1.6; margin-bottom: 20px; }
  .intro-panel ul { margin: 8px 0 0 18px; padding: 0; }
  .intro-panel li { margin-bottom: 6px; }
  .layer-note { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
  .layer-tabs { display: flex; gap: 6px; margin-bottom: 20px; flex-wrap: wrap; }
  .layer-tab {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--panel); color: var(--muted); cursor: pointer; font-size: 13px;
  }
  .layer-tab.active { background: var(--accent); color: #0f1115; border-color: var(--accent); font-weight: 600; }
  .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .panel h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  #heatmap { overflow-x: auto; }
  .heat-table { border-collapse: collapse; }
  .heat-table th, .heat-table td { padding: 0; }
  .heat-col-label {
    writing-mode: vertical-rl; transform: rotate(180deg);
    font-size: 10px; color: var(--muted); font-family: ui-monospace, monospace;
    cursor: pointer; padding: 2px 3px; max-height: 90px;
  }
  .heat-row-label {
    font-size: 10px; color: var(--muted); font-family: ui-monospace, monospace;
    cursor: pointer; padding: 0 6px; text-align: right; white-space: nowrap;
  }
  .heat-col-label.selected-label, .heat-row-label.selected-label { color: var(--text); font-weight: 700; }
  .heat-cell-td { cursor: pointer; }
  .heat-cell { width: 18px; height: 18px; border-radius: 2px; }
  .heat-cell.on-selected { outline: 1px solid rgba(255,255,255,0.35); outline-offset: -1px; }
  .note { font-size: 12px; color: var(--muted); line-height: 1.5; margin: 4px 0 14px; }
  .head-grid { display: flex; flex-wrap: wrap; gap: 14px; }
  .head-thumb-wrap {
    width: 108px; text-align: center; cursor: pointer; padding: 8px;
    border-radius: 8px; border: 1px solid var(--border);
  }
  .head-thumb-wrap:hover { border-color: var(--muted); }
  .head-thumb-wrap.active { border-color: var(--accent); background: rgba(90, 169, 230, 0.1); }
  .head-thumb-wrap canvas {
    width: 90px; height: 90px; image-rendering: pixelated;
    border-radius: 4px; display: block; margin: 0 auto 6px;
  }
  .head-thumb-title { font-size: 11px; color: var(--text); font-weight: 600; margin-bottom: 3px; }
  .head-thumb-label { font-size: 10px; color: var(--muted); line-height: 1.3; }
  .legend-row { display: flex; flex-wrap: wrap; gap: 10px 14px; margin-bottom: 14px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
  .legend-swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; flex-shrink: 0; }
  .token-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 16px; }
  .token-chip {
    padding: 4px 8px; border-radius: 5px; font-size: 12px; color: var(--text);
    font-family: ui-monospace, monospace; cursor: pointer;
    border: 1px solid transparent;
  }
  .token-chip:hover { border-color: var(--text); }
  .token-chip.selected { outline: 2px solid var(--text); outline-offset: 1px; }
  .util-bars { display: flex; align-items: flex-end; gap: 8px; height: 120px; margin-top: 8px; }
  .util-bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }
  .util-bar { width: 100%; background: var(--accent); border-radius: 3px 3px 0 0; min-height: 2px; }
  .util-label { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .tooltip {
    position: fixed; background: #000; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 10;
    display: none; max-width: 260px; line-height: 1.4; white-space: pre-line;
  }
  .legend { font-size: 12px; color: var(--muted); margin-top: 8px; }
  .aux-note { font-size: 12px; color: var(--muted); margin-top: 16px; }
</style>
</head>
<body>

<h1>MoE Model Inspector</h1>
<div class="subtitle">Attention + routing internals for one forward pass</div>

<div class="prompt-box" id="promptBox"></div>

<div class="panel intro-panel">
  <h2>How to read this</h2>
  <ul>
    <li><b>All heads at a glance:</b> a small snapshot of every attention head in the current layer, side by side, so you can compare them directly instead of flipping through one at a time. Click a tile to load it into the detailed view below.</li>
    <li><b>Head detail:</b> each row is the token currently being computed (the "query"); each column is a token it's looking back at (the "key"). Brighter cell = more attention weight. This model is causal (decoder-only), so a token can only attend to itself and earlier tokens -- the upper-right triangle is always exactly zero. Click a token label (row or column) to highlight it.</li>
    <li><b>Router decisions:</b> each chip is split into colored segments showing its top-2 expert assignment -- the width of each segment is that expert's share of the routing weight, so a chip that's almost entirely one color means the router is confident; an even split means it's torn between two experts. Click a chip to highlight that same token in the attention heatmap.</li>
    <li><b>Expert utilization:</b> how many tokens in this one prompt went to each expert (top-1 choice). Flat bars = balanced routing. One or two tall bars = the router has collapsed onto a handful of favorites.</li>
  </ul>
</div>

<div class="layer-note">Layer 0 is closest to the input embeddings; the highest-numbered layer feeds directly into the output prediction.</div>
<div class="layer-tabs" id="layerTabs"></div>

<div class="legend-row" id="expertLegend"></div>

<div class="panel">
  <h2>All heads at a glance</h2>
  <div class="note" style="margin-top:0;">
    Each transformer layer runs several attention heads in parallel, and they
    often specialize differently -- some track the immediately previous
    token, some anchor heavily on the first token (a common pattern called an
    "attention sink"), some spread attention broadly across everything so
    far. Comparing them side by side is the only way to actually see that
    specialization; flipping through them one at a time hides it. Click a
    tile to inspect that head in detail below.
    <br><i>(The one-line labels are rough heuristics to give you a starting
    vocabulary, not precise categories -- use your own eyes too.)</i>
  </div>
  <div class="head-grid" id="headGrid"></div>
</div>

<div class="panel">
  <h2>Head detail</h2>
  <div class="note" style="margin-top:0;" id="headDetailNote"></div>
  <div id="heatmap"></div>
</div>

<div class="grid">
  <div class="panel">
    <h2>Router decisions</h2>
    <div class="token-row" id="routingTokens"></div>
    <h2 style="margin-top:20px;">Expert utilization (top-1, this prompt)</h2>
    <div class="util-bars" id="utilBars"></div>
  </div>
</div>

<div class="aux-note" id="auxNote"></div>
<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;

let currentLayer = 0;
let currentHead = 0;
let selectedToken = null;

const expertColors = [
  "#5aa9e6", "#e6785a", "#7ce65a", "#e6c15a", "#b05ae6",
  "#5ae6c1", "#e65a9e", "#a3e65a", "#5a78e6", "#e6a35a"
];

function colorForExpert(idx) {
  return expertColors[idx % expertColors.length];
}

function toggleSelected(i) {
  selectedToken = (selectedToken === i ? null : i);
  renderHeatmap();
  renderRoutingTokens();
}

function renderPromptBox() {
  document.getElementById("promptBox").textContent = 'Prompt: "' + DATA.prompt + '"  \u00b7  ' +
    DATA.tokens.length + ' tokens  \u00b7  ' + DATA.n_layers + ' layers  \u00b7  ' +
    DATA.n_experts + ' experts (top-' + DATA.top_k + ' routing)';
}

function renderLegend() {
  const container = document.getElementById("expertLegend");
  container.innerHTML = "";
  for (let e = 0; e < DATA.n_experts; e++) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("div");
    swatch.className = "legend-swatch";
    swatch.style.background = colorForExpert(e);
    item.appendChild(swatch);
    item.appendChild(document.createTextNode("Expert " + e));
    container.appendChild(item);
  }
}

function headStats(weights) {
  // Rough summary of one head's attention pattern over this one prompt.
  const T = weights.length;
  let selfSum = 0, prevSum = 0, firstSum = 0, distSum = 0, prevCount = 0;
  for (let i = 0; i < T; i++) {
    selfSum += weights[i][i];
    if (i > 0) {
      prevSum += weights[i][i - 1];
      firstSum += weights[i][0];
      prevCount += 1;
    }
    let rowDist = 0;
    for (let j = 0; j <= i; j++) rowDist += weights[i][j] * (i - j);
    distSum += rowDist;
  }
  return {
    avgSelf: selfSum / T,
    avgPrev: prevCount > 0 ? prevSum / prevCount : 0,
    avgFirst: prevCount > 0 ? firstSum / prevCount : 0,
    avgDist: distSum / T,
  };
}

function headLabel(stats) {
  if (stats.avgFirst > 0.3) return "Attention sink -- anchors heavily on the first token";
  if (stats.avgPrev > 0.35) return "Previous-token head -- mostly looks one step back";
  if (stats.avgSelf > 0.35) return "Self-focused -- mostly attends to itself";
  if (stats.avgDist > 3) return "Long-range -- looks far back on average";
  return "Diffuse -- spreads attention broadly";
}

function drawHeatmapCanvas(canvas, weights) {
  const T = weights.length;
  canvas.width = T;
  canvas.height = T;
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(T, T);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const v = weights[i][j];
      const alpha = Math.min(1, v * 3);
      const r = Math.round(15 + (90 - 15) * alpha);
      const g = Math.round(17 + (169 - 17) * alpha);
      const b = Math.round(21 + (230 - 21) * alpha);
      const idx = (i * T + j) * 4;
      imgData.data[idx] = r;
      imgData.data[idx + 1] = g;
      imgData.data[idx + 2] = b;
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function renderHeadGrid() {
  const layer = DATA.layers[currentLayer];
  const container = document.getElementById("headGrid");
  container.innerHTML = "";

  for (let h = 0; h < layer.n_heads; h++) {
    const weights = layer.attn_per_head[h];
    const stats = headStats(weights);
    const label = headLabel(stats);

    const wrap = document.createElement("div");
    wrap.className = "head-thumb-wrap" + (h === currentHead ? " active" : "");
    wrap.onclick = () => { currentHead = h; renderHeadGrid(); renderHeatmap(); };

    const canvas = document.createElement("canvas");
    drawHeatmapCanvas(canvas, weights);

    const title = document.createElement("div");
    title.className = "head-thumb-title";
    title.textContent = "Head " + h;

    const labelDiv = document.createElement("div");
    labelDiv.className = "head-thumb-label";
    labelDiv.textContent = label;

    wrap.appendChild(canvas);
    wrap.appendChild(title);
    wrap.appendChild(labelDiv);
    container.appendChild(wrap);
  }
}

function renderTabs() {
  const container = document.getElementById("layerTabs");
  container.innerHTML = "";
  for (let i = 0; i < DATA.n_layers; i++) {
    const tab = document.createElement("div");
    tab.className = "layer-tab" + (i === currentLayer ? " active" : "");
    tab.textContent = "Layer " + i;
    tab.onclick = () => { currentLayer = i; currentHead = 0; renderAll(); };
    container.appendChild(tab);
  }
}

function renderHeatmap() {
  const layer = DATA.layers[currentLayer];
  const weights = layer.attn_per_head[currentHead];
  const stats = headStats(weights);
  document.getElementById("headDetailNote").textContent =
    "Showing Head " + currentHead + " -- " + headLabel(stats) +
    ". Click a tile above to switch heads.";
  const T = weights.length;
  const container = document.getElementById("heatmap");
  container.innerHTML = "";
  const tooltip = document.getElementById("tooltip");

  const table = document.createElement("table");
  table.className = "heat-table";

  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th"));
  for (let j = 0; j < T; j++) {
    const th = document.createElement("th");
    const label = document.createElement("div");
    label.className = "heat-col-label" + (selectedToken === j ? " selected-label" : "");
    label.textContent = DATA.tokens[j];
    label.title = DATA.tokens[j];
    label.onclick = () => toggleSelected(j);
    th.appendChild(label);
    headRow.appendChild(th);
  }
  table.appendChild(headRow);

  for (let i = 0; i < T; i++) {
    const row = document.createElement("tr");
    const th = document.createElement("th");
    const rlabel = document.createElement("div");
    rlabel.className = "heat-row-label" + (selectedToken === i ? " selected-label" : "");
    rlabel.textContent = DATA.tokens[i];
    rlabel.title = DATA.tokens[i];
    rlabel.onclick = () => toggleSelected(i);
    th.appendChild(rlabel);
    row.appendChild(th);

    for (let j = 0; j < T; j++) {
      const v = weights[i][j];
      const td = document.createElement("td");
      td.className = "heat-cell-td";
      const cell = document.createElement("div");
      cell.className = "heat-cell" + (selectedToken !== null && (i === selectedToken || j === selectedToken) ? " on-selected" : "");
      const alpha = Math.min(1, v * 3);
      cell.style.background = "rgba(90, 169, 230, " + alpha.toFixed(3) + ")";
      td.appendChild(cell);
      td.onmouseenter = (e) => {
        tooltip.style.display = "block";
        tooltip.textContent = DATA.tokens[i] + " -> " + DATA.tokens[j] + ": " + v.toFixed(4);
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY + 12) + "px";
      };
      td.onmouseleave = () => { tooltip.style.display = "none"; };
      row.appendChild(td);
    }
    table.appendChild(row);
  }

  container.appendChild(table);
}

function renderRoutingTokens() {
  const layer = DATA.layers[currentLayer];
  const container = document.getElementById("routingTokens");
  container.innerHTML = "";
  const tooltip = document.getElementById("tooltip");

  DATA.tokens.forEach((tok, i) => {
    const idx = layer.topk_idx[i];
    const probs = layer.topk_probs[i];
    const chip = document.createElement("div");
    chip.className = "token-chip" + (selectedToken === i ? " selected" : "");
    chip.textContent = tok;

    const c1 = colorForExpert(idx[0]);
    if (idx.length >= 2) {
      const c2 = colorForExpert(idx[1]);
      const p1pct = (probs[0] * 100).toFixed(1);
      chip.style.background = "linear-gradient(to right, " + c1 + "77 0%, " + c1 + "77 " + p1pct + "%, " + c2 + "77 " + p1pct + "%, " + c2 + "77 100%)";
    } else {
      chip.style.background = c1 + "55";
    }
    chip.style.borderColor = c1;

    chip.onclick = () => toggleSelected(i);

    chip.onmouseenter = (e) => {
      let lines = ["Token: " + tok];
      for (let k = 0; k < DATA.top_k; k++) {
        lines.push("  expert " + idx[k] + ": " + (probs[k] * 100).toFixed(1) + "%");
      }
      tooltip.style.display = "block";
      tooltip.textContent = lines.join("\\n");
      tooltip.style.left = (e.clientX + 12) + "px";
      tooltip.style.top = (e.clientY + 12) + "px";
    };
    chip.onmouseleave = () => { tooltip.style.display = "none"; };

    container.appendChild(chip);
  });
}

function renderUtilBars() {
  const layer = DATA.layers[currentLayer];
  const container = document.getElementById("utilBars");
  container.innerHTML = "";
  const maxCount = Math.max(1, ...layer.utilization);

  layer.utilization.forEach((count, e) => {
    const wrap = document.createElement("div");
    wrap.className = "util-bar-wrap";
    const bar = document.createElement("div");
    bar.className = "util-bar";
    bar.style.height = Math.max(2, (count / maxCount) * 100) + "%";
    bar.style.background = colorForExpert(e);
    const label = document.createElement("div");
    label.className = "util-label";
    label.textContent = "E" + e + " (" + count + ")";
    wrap.appendChild(bar);
    wrap.appendChild(label);
    container.appendChild(wrap);
  });
}

function renderAuxNote() {
  document.getElementById("auxNote").textContent =
    "Load-balancing aux loss for this forward pass: " + DATA.aux_loss.toFixed(4) +
    " (near-uniform expert utilization above -> healthy routing; " +
    "utilization concentrated on 1-2 experts -> router collapse).";
}

function renderAll() {
  renderTabs();
  renderHeadGrid();
  renderHeatmap();
  renderRoutingTokens();
  renderUtilBars();
  renderAuxNote();
}

renderPromptBox();
renderLegend();
renderAll();
</script>

</body>
</html>
"""
# pylint: enable=line-too-long

if __name__ == "__main__":
    main()