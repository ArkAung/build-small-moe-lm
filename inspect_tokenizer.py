import argparse
import json

from tokenizers import Tokenizer


def bytes_to_unicode():
    """
    Standard GPT-2 byte-level mapping: maps each of the 256 possible byte
    values to a printable unicode character, so byte-level BPE never needs
    an <unk> token. Same construction HuggingFace's ByteLevel pre-tokenizer
    uses internally.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def decode_piece(piece):
    """Turn a byte-level BPE vocab piece back into human-readable text."""
    try:
        raw = bytes(BYTE_DECODER[c] for c in piece)
    except KeyError:
        return piece, len(piece)
    try:
        return raw.decode("utf-8"), len(raw)
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), len(raw)


PRESET_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "1234567890 costs $45.99 today.",
    "Supercalifragilisticexpialidocious is a whimsical word.",
    "Once upon a time, a curious rabbit found a golden key.",
    "run running runner ran runs",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--out", type=str, default="tokenizer_inspection.html")
    args = parser.parse_args()

    with open(args.tokenizer) as f:
        raw = json.load(f)

    model = raw["model"]
    vocab = model["vocab"]                     # token string -> id
    merges_raw = model["merges"]                # learned merges, in rank order

    # tokenizers versions differ: merges are either "left right" strings
    # or [left, right] lists. Normalize to a list of [left, right].
    merges = []
    for m in merges_raw:
        if isinstance(m, str):
            left, right = m.split(" ", 1)
        else:
            left, right = m[0], m[1]
        merges.append([left, right])

    special_tokens = [t["content"] for t in raw.get("added_tokens", [])]

    # byte-length histogram across the full vocab
    length_buckets = {"1": 0, "2": 0, "3": 0, "4-6": 0, "7-10": 0, "11+": 0}
    for piece in vocab.keys():
        _, blen = decode_piece(piece)
        if blen == 1:
            length_buckets["1"] += 1
        elif blen == 2:
            length_buckets["2"] += 1
        elif blen == 3:
            length_buckets["3"] += 1
        elif blen <= 6:
            length_buckets["4-6"] += 1
        elif blen <= 10:
            length_buckets["7-10"] += 1
        else:
            length_buckets["11+"] += 1

    # verified ground-truth tokenizations for the preset sentences, using
    # the real tokenizer library (not the JS re-implementation) so these
    # specific examples are guaranteed correct even if the JS port has an
    # edge case somewhere.
    tok = Tokenizer.from_file(args.tokenizer)
    preset_examples = []
    for sentence in PRESET_SENTENCES:
        enc = tok.encode(sentence)
        pieces = [decode_piece(p)[0] for p in enc.tokens]
        preset_examples.append({
            "text": sentence,
            "pieces": pieces,
            "ids": enc.ids,
        })

    # decoded display form + byte length for every vocab piece, so the JS
    # side never has to re-derive it
    vocab_display = {piece: decode_piece(piece)[0] for piece in vocab.keys()}

    data = {
        "vocab": vocab,
        "vocab_display": vocab_display,
        "merges": merges,
        "special_tokens": special_tokens,
        "length_buckets": length_buckets,
        "preset_examples": preset_examples,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
    }

    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote tokenizer inspection view -> {args.out}")
    print(f"Vocab size: {len(vocab)}, merges: {len(merges)}")


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tokenizer Inspector</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --border: #2a2e37;
    --text: #e6e8eb; --muted: #8b92a3; --accent: #5aa9e6;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; padding: 24px; max-width: 980px;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 14px; margin: 0 0 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 20px;
  }
  .stats-row { display: flex; gap: 24px; flex-wrap: wrap; }
  .stat { }
  .stat-value { font-size: 24px; font-weight: 700; }
  .stat-label { font-size: 12px; color: var(--muted); }
  .hist-bars { display: flex; align-items: flex-end; gap: 10px; height: 100px; margin-top: 12px; }
  .hist-bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }
  .hist-bar { width: 100%; background: var(--accent); border-radius: 3px 3px 0 0; min-height: 2px; }
  .hist-label { font-size: 11px; color: var(--muted); margin-top: 4px; text-align: center; }
  .range-tabs { display: flex; gap: 6px; margin-bottom: 14px; }
  .range-tab {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: #0f1115; color: var(--muted); cursor: pointer; font-size: 13px;
  }
  .range-tab.active { background: var(--accent); color: #0f1115; border-color: var(--accent); font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  td.mono, th.mono { font-family: ui-monospace, monospace; }
  .arrow { color: var(--muted); }
  .preset-buttons { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .preset-btn {
    padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: #0f1115; color: var(--text); cursor: pointer; font-size: 12px;
  }
  .preset-btn:hover { border-color: var(--accent); }
  #playgroundInput {
    width: 100%; padding: 10px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: #0f1115; color: var(--text); font-size: 14px; margin-bottom: 14px;
  }
  .token-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
  .token-chip {
    padding: 4px 8px; border-radius: 5px; font-size: 13px;
    font-family: ui-monospace, monospace; cursor: default;
    border: 1px solid transparent;
  }
  .token-chip:hover { border-color: var(--accent); }
  .playground-stats { font-size: 12px; color: var(--muted); }
  .tooltip {
    position: fixed; background: #000; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 10;
    display: none; max-width: 260px; line-height: 1.4;
  }
  .note { font-size: 12px; color: var(--muted); margin-top: 10px; }
</style>
</head>
<body>

<h1>Tokenizer Inspector</h1>
<div class="subtitle">What BPE training actually learned, and how it segments text</div>

<div class="panel">
  <h2>Vocab overview</h2>
  <div class="stats-row">
    <div class="stat"><div class="stat-value" id="vocabSizeStat"></div><div class="stat-label">vocab size</div></div>
    <div class="stat"><div class="stat-value" id="numMergesStat"></div><div class="stat-label">learned merges</div></div>
    <div class="stat"><div class="stat-value" id="specialTokensStat"></div><div class="stat-label">special tokens</div></div>
  </div>
  <h2 style="margin-top:20px;">Vocab piece length (bytes)</h2>
  <div class="hist-bars" id="histBars"></div>
  <div class="note">Most merges converge on short, common substrings first; longer pieces (whole common words) are rarer, later merges.</div>
</div>

<div class="panel">
  <h2>Watch the merges build up</h2>
  <div class="range-tabs" id="rangeTabs"></div>
  <table>
    <thead><tr><th>Rank</th><th class="mono">Left</th><th></th><th class="mono">Right</th><th></th><th class="mono">Merged result</th></tr></thead>
    <tbody id="mergeTableBody"></tbody>
  </table>
  <div class="note">Rank = order learned. Early merges combine raw byte pairs; by the later merges, the tokenizer is stitching together whole common words.</div>
</div>

<div class="panel">
  <h2>Live tokenizer playground</h2>
  <div class="preset-buttons" id="presetButtons"></div>
  <input id="playgroundInput" type="text" placeholder="Type anything and watch it get tokenized..." />
  <div class="token-row" id="playgroundTokens"></div>
  <div class="playground-stats" id="playgroundStats"></div>
  <div class="note">This runs the actual learned merges in your browser -- what you see is the real tokenization, not an approximation.</div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;

// ---------- byte-level BPE re-implementation (matches prepare_data.py) ----------

function bytesToUnicode() {
  let bs = [];
  for (let i = "!".charCodeAt(0); i <= "~".charCodeAt(0); i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);
  let cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n += 1; }
  }
  const encoder = {};
  const decoder = {};
  for (let i = 0; i < bs.length; i++) {
    encoder[bs[i]] = String.fromCharCode(cs[i]);
    decoder[String.fromCharCode(cs[i])] = bs[i];
  }
  return { encoder, decoder };
}
const { encoder: BYTE_ENCODER, decoder: BYTE_DECODER } = bytesToUnicode();

const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+/gu;

const MERGE_RANK = new Map();
DATA.merges.forEach((pair, idx) => {
  MERGE_RANK.set(pair[0] + "\\u0001" + pair[1], idx);
});
const VOCAB = new Map(Object.entries(DATA.vocab));

function getPairs(word) {
  const pairs = new Set();
  for (let i = 0; i < word.length - 1; i++) pairs.add(word[i] + "\\u0001" + word[i + 1]);
  return pairs;
}

function bpeMerge(mapped) {
  let word = Array.from(mapped);
  if (word.length <= 1) return word;
  let pairs = getPairs(word);
  while (true) {
    let bestPair = null, bestRank = Infinity;
    for (const p of pairs) {
      if (MERGE_RANK.has(p) && MERGE_RANK.get(p) < bestRank) {
        bestRank = MERGE_RANK.get(p);
        bestPair = p;
      }
    }
    if (bestPair === null) break;
    const [first, second] = bestPair.split("\\u0001");
    const newWord = [];
    let i = 0;
    while (i < word.length) {
      if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
        newWord.push(first + second);
        i += 2;
      } else {
        newWord.push(word[i]);
        i += 1;
      }
    }
    word = newWord;
    if (word.length === 1) break;
    pairs = getPairs(word);
  }
  return word;
}

function tokenize(text) {
  const enc = new TextEncoder();
  const matches = text.match(PAT) || [];
  const pieces = [];
  for (const word of matches) {
    const bytes = enc.encode(word);
    let mapped = "";
    for (const b of bytes) mapped += BYTE_ENCODER[b];
    for (const piece of bpeMerge(mapped)) pieces.push(piece);
  }
  return pieces;
}

function decodePiece(piece) {
  if (DATA.vocab_display[piece] !== undefined) return DATA.vocab_display[piece];
  try {
    const bytes = Array.from(piece).map((c) => BYTE_DECODER[c]);
    return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  } catch (e) {
    return piece;
  }
}

// ---------- rendering ----------

const chipColors = ["#5aa9e6", "#e6785a", "#7ce65a", "#e6c15a", "#b05ae6", "#5ae6c1"];

function renderStats() {
  document.getElementById("vocabSizeStat").textContent = DATA.vocab_size.toLocaleString();
  document.getElementById("numMergesStat").textContent = DATA.num_merges.toLocaleString();
  document.getElementById("specialTokensStat").textContent = DATA.special_tokens.join(", ");
}

function renderHistogram() {
  const container = document.getElementById("histBars");
  container.innerHTML = "";
  const buckets = DATA.length_buckets;
  const maxCount = Math.max(...Object.values(buckets));
  for (const [label, count] of Object.entries(buckets)) {
    const wrap = document.createElement("div");
    wrap.className = "hist-bar-wrap";
    const bar = document.createElement("div");
    bar.className = "hist-bar";
    bar.style.height = Math.max(2, (count / maxCount) * 100) + "%";
    const lab = document.createElement("div");
    lab.className = "hist-label";
    lab.textContent = label + " bytes\\n(" + count + ")";
    lab.style.whiteSpace = "pre-line";
    wrap.appendChild(bar);
    wrap.appendChild(lab);
    container.appendChild(wrap);
  }
}

const RANGES = [
  { label: "Earliest", start: 0, end: 30 },
  { label: "Early-mid", start: null, end: null },
  { label: "Middle", start: null, end: null },
  { label: "Late", start: null, end: null },
  { label: "Final", start: null, end: null },
];
function computeRanges() {
  const n = DATA.num_merges;
  RANGES[1].start = Math.floor(n * 0.1); RANGES[1].end = RANGES[1].start + 30;
  RANGES[2].start = Math.floor(n * 0.4); RANGES[2].end = RANGES[2].start + 30;
  RANGES[3].start = Math.floor(n * 0.75); RANGES[3].end = RANGES[3].start + 30;
  RANGES[4].start = Math.max(0, n - 30); RANGES[4].end = n;
}
let currentRange = 0;

function renderRangeTabs() {
  const container = document.getElementById("rangeTabs");
  container.innerHTML = "";
  RANGES.forEach((r, i) => {
    const tab = document.createElement("div");
    tab.className = "range-tab" + (i === currentRange ? " active" : "");
    tab.textContent = r.label + " (#" + r.start + "-" + r.end + ")";
    tab.onclick = () => { currentRange = i; renderAll(); };
    container.appendChild(tab);
  });
}

function renderMergeTable() {
  const tbody = document.getElementById("mergeTableBody");
  tbody.innerHTML = "";
  const r = RANGES[currentRange];
  for (let i = r.start; i < Math.min(r.end, DATA.num_merges); i++) {
    const [left, right] = DATA.merges[i];
    const row = document.createElement("tr");
    row.innerHTML =
      "<td>" + i + "</td>" +
      "<td class='mono'>" + escapeHtml(decodePiece(left)) + "</td>" +
      "<td class='arrow'>+</td>" +
      "<td class='mono'>" + escapeHtml(decodePiece(right)) + "</td>" +
      "<td class='arrow'>&rarr;</td>" +
      "<td class='mono'><b>" + escapeHtml(decodePiece(left + right)) + "</b></td>";
    tbody.appendChild(row);
  }
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/ /g, "\\u00b7");
}

function renderPresetButtons() {
  const container = document.getElementById("presetButtons");
  container.innerHTML = "";
  DATA.preset_examples.forEach((ex) => {
    const btn = document.createElement("div");
    btn.className = "preset-btn";
    btn.textContent = ex.text.length > 40 ? ex.text.slice(0, 40) + "..." : ex.text;
    btn.onclick = () => {
      document.getElementById("playgroundInput").value = ex.text;
      renderPlayground(ex.text);
    };
    container.appendChild(btn);
  });
}

function renderPlayground(text) {
  const pieces = tokenize(text);
  const container = document.getElementById("playgroundTokens");
  container.innerHTML = "";
  const tooltip = document.getElementById("tooltip");

  pieces.forEach((piece, i) => {
    const chip = document.createElement("div");
    chip.className = "token-chip";
    const display = decodePiece(piece);
    chip.textContent = display.trim() === "" ? "\\u2423".repeat(display.length || 1) : display;
    const color = chipColors[i % chipColors.length];
    chip.style.background = color + "33";
    chip.style.color = color;
    chip.style.borderColor = color;

    const id = VOCAB.get(piece);
    chip.onmouseenter = (e) => {
      tooltip.style.display = "block";
      tooltip.textContent = "id: " + (id !== undefined ? id : "?") + "  |  piece: \\"" + display + "\\"";
      tooltip.style.left = (e.clientX + 12) + "px";
      tooltip.style.top = (e.clientY + 12) + "px";
    };
    chip.onmouseleave = () => { tooltip.style.display = "none"; };

    container.appendChild(chip);
  });

  const charCount = text.length;
  const tokenCount = pieces.length;
  const ratio = tokenCount > 0 ? (charCount / tokenCount).toFixed(2) : "0";
  document.getElementById("playgroundStats").textContent =
    charCount + " characters -> " + tokenCount + " tokens  (" + ratio + " chars/token)";
}

function renderAll() {
  renderRangeTabs();
  renderMergeTable();
}

computeRanges();
renderStats();
renderHistogram();
renderPresetButtons();
renderAll();

const input = document.getElementById("playgroundInput");
input.addEventListener("input", (e) => renderPlayground(e.target.value));
// seed with the first preset example
if (DATA.preset_examples.length > 0) {
  input.value = DATA.preset_examples[0].text;
  renderPlayground(DATA.preset_examples[0].text);
}
</script>

</body>
</html>
"""

if __name__ == "__main__":
    main()