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
from tokenizers import Tokenizer

from model import MoETransformer, ModelConfig


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a")
    parser.add_argument("--out", type=str, default="inspection.html")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig(**cfg_dict)

    model = MoETransformer(cfg)
    model.load_weights(args.checkpoint)
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    bos_id = tok.token_to_id("<bos>")
    ids = [bos_id] + tok.encode(args.prompt).ids
    tokens = mx.array([ids])

    logits, aux_loss, _, captures = model(tokens, capture=True)
    mx.eval(logits, captures)

    T = len(ids)
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
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote inspection view -> {args.out}")
    print(f"Prompt tokenized to {T} tokens. Open {args.out} in a browser.")


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
    border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;
    font-size: 14px;
  }
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
  .head-select { margin-bottom: 10px; }
  .head-select select {
    background: #0f1115; color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; font-size: 13px;
  }
  #heatmap { display: grid; gap: 2px; }
  .heat-cell { aspect-ratio: 1; border-radius: 2px; position: relative; }
  .token-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 16px; }
  .token-chip {
    padding: 4px 8px; border-radius: 5px; font-size: 12px;
    font-family: ui-monospace, monospace; cursor: default;
    border: 1px solid transparent;
  }
  .token-chip:hover { border-color: var(--accent); }
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

<div class="layer-tabs" id="layerTabs"></div>

<div class="grid">
  <div class="panel">
    <h2>Attention</h2>
    <div class="head-select">Head:
      <select id="headSelect"></select>
    </div>
    <div id="heatmap"></div>
    <div class="legend">Rows = query token, columns = key token. Brighter = more attention weight.</div>
  </div>

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

const expertColors = [
  "#5aa9e6", "#e6785a", "#7ce65a", "#e6c15a", "#b05ae6",
  "#5ae6c1", "#e65a9e", "#a3e65a", "#5a78e6", "#e6a35a"
];

function colorForExpert(idx) {
  return expertColors[idx % expertColors.length];
}

function renderPromptBox() {
  document.getElementById("promptBox").textContent = 'Prompt: "' + DATA.prompt + '"  \u00b7  ' +
    DATA.tokens.length + ' tokens  \u00b7  ' + DATA.n_layers + ' layers  \u00b7  ' +
    DATA.n_experts + ' experts (top-' + DATA.top_k + ' routing)';
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

function renderHeadSelect() {
  const sel = document.getElementById("headSelect");
  const layer = DATA.layers[currentLayer];
  sel.innerHTML = "";
  for (let h = 0; h < layer.n_heads; h++) {
    const opt = document.createElement("option");
    opt.value = h;
    opt.textContent = "Head " + h;
    sel.appendChild(opt);
  }
  sel.value = currentHead;
  sel.onchange = (e) => { currentHead = parseInt(e.target.value); renderHeatmap(); };
}

function renderHeatmap() {
  const layer = DATA.layers[currentLayer];
  const weights = layer.attn_per_head[currentHead];
  const T = weights.length;
  const container = document.getElementById("heatmap");
  container.innerHTML = "";
  container.style.gridTemplateColumns = "repeat(" + T + ", 1fr)";

  const tooltip = document.getElementById("tooltip");

  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const v = weights[i][j];
      const cell = document.createElement("div");
      cell.className = "heat-cell";
      cell.style.gridColumn = j + 1;
      cell.style.gridRow = i + 1;
      const alpha = Math.min(1, v * 3);
      cell.style.background = "rgba(90, 169, 230, " + alpha.toFixed(3) + ")";
      cell.onmouseenter = (e) => {
        tooltip.style.display = "block";
        tooltip.textContent = DATA.tokens[i] + " -> " + DATA.tokens[j] + ": " + v.toFixed(4);
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY + 12) + "px";
      };
      cell.onmouseleave = () => { tooltip.style.display = "none"; };
      container.appendChild(cell);
    }
  }
}

function renderRoutingTokens() {
  const layer = DATA.layers[currentLayer];
  const container = document.getElementById("routingTokens");
  container.innerHTML = "";
  const tooltip = document.getElementById("tooltip");

  DATA.tokens.forEach((tok, i) => {
    const top1 = layer.topk_idx[i][0];
    const chip = document.createElement("div");
    chip.className = "token-chip";
    chip.textContent = tok;
    chip.style.background = colorForExpert(top1) + "33";
    chip.style.color = colorForExpert(top1);
    chip.style.borderColor = colorForExpert(top1);

    chip.onmouseenter = (e) => {
      let lines = ["Token: " + tok];
      for (let k = 0; k < DATA.top_k; k++) {
        lines.push("  expert " + layer.topk_idx[i][k] + ": " + (layer.topk_probs[i][k] * 100).toFixed(1) + "%");
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
  renderHeadSelect();
  renderHeatmap();
  renderRoutingTokens();
  renderUtilBars();
  renderAuxNote();
}

renderPromptBox();
renderAll();
</script>

</body>
</html>
"""

if __name__ == "__main__":
    main()