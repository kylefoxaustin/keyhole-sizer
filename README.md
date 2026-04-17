# keyhole-sizer

Interactive NPU sizing sandbox for the
[Keyhole](https://github.com/kylefoxaustin/keyhole) edge-AI bake-off findings.

A Streamlit app that wraps the measured bake-off data in tunable sliders:
pick an NPU tier (or build a custom one), a vision pipeline, concurrent
stream count, and whether an LLM co-exists on the same silicon — then
watch live FPS / tok/s / VRAM-fit / duty-cycle projections.

If you've read the Keyhole deck and want to answer **"what if my NPU had
a 96-bit bus instead of 128?"** or **"how many 480p streams can I run
with a 2 Hz LLM query rate?"**, this is the tool.

## Install & run

```bash
# 1. Clone
git clone https://github.com/kylefoxaustin/keyhole-sizer.git
cd keyhole-sizer

# 2. Python venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Browser opens to `http://localhost:8501`. No GPU needed — this app is
pure projection math on top of already-measured numbers.

## What you can tune

**Hardware** (sidebar):
- **Tier preset:** `NPU Low` (64-bit LPDDR4) / `NPU Mid` (128-bit
  LPDDR5X — Keyhole shipping target) / `NPU High` (high-bin LPDDR5X) /
  `Custom` (roll your own)
- **Custom mode:** bus width, memory type, data rate, bandwidth
  efficiency, peak BF16 TOPS, peak FP8 TOPS, compute efficiency, DRAM
  capacity, TDP

**Vision workload:**
- Pipeline: SAM 3 baseline, EfficientSAM-Small FP8, Hybrid V2 variants,
  TRT FP8 shipping stack, YOLO-only
- Resolution: 720p / 1080p / 4K
- Concurrent streams: 1..16 (YOLO batching applied automatically)

**LLM co-exist:**
- Toggle on/off
- Qwen3-30B-A3B quantization: Q4_K_M / Q5_K_M / Q8_0
- Queries per minute (slider)
- Answer length: short (~200 tok) or RAG (8K prompt + 2K response)

## What you see

- **Vision FPS per stream** (under concurrency + LLM duty cycle)
- **Total system FPS** across all streams
- **Memory fit** — does the current config spill your DRAM?
- **LLM decode tok/s** + TTFT at the chosen quant
- **Pipeline timing breakdown** — how much is YOLO vs CLIP
- **Per-tier comparison** — same workload on Low/Mid/High
- **Stream scaling curve** — per-stream FPS & total FPS as N changes
- **Duty-cycle curve** — vision FPS vs LLM queries/min, two answer
  styles

## Where the numbers come from

Every baseline constant traces back to a specific bake-off script in the
Keyhole project. See
[`keyhole/REPRODUCE.md`](https://github.com/kylefoxaustin/keyhole/blob/main/REPRODUCE.md)
for how to regenerate them on your own RTX 5090.

| Baseline | Source |
|----------|--------|
| YOLO-seg FP8 edge ms @ resolution | `bakeoff_trt_yolo.py` |
| CLIP FP8 edge ms | `bakeoff_trt_clip.py` |
| YOLO batching curve (B=1..16) | `bakeoff_concurrency.py` |
| SAM 3 / EfficientSAM / MobileSAM edge ms | `bakeoff_sam_variants.py`, `bakeoff_fp8.py` |
| LLM NPU tier actuals (TTFT, decode) | Vendor benchmarks, folded into `bakeoff_llm.py` |
| Qwen3-30B-A3B GGUF sizes + bytes/param | `bakeoff_llm.py` |

Vision pipelines are **bandwidth-bound** on the NPUs we're modeling, so
edge ms scales inversely with effective bandwidth. LLM decode is
bandwidth-bound on active-params × bytes-per-param (MoE wins — only 3B
of the 30B total are loaded per token).

## Limitations

- **Synthetic projection, not simulation.** If you push the custom
  sliders way out of range (e.g. 2048-bit bus @ 100 GT/s), the
  projection still linearly scales — the math doesn't know about
  cache hierarchies, tiling, or NoC topology.
- **Pipeline list is fixed** to what Keyhole measured. Adding a new
  pipeline requires editing `sizer/npu_model.py::PIPELINES`.
- **LLM model is fixed** to Qwen3-30B-A3B. Other models would need
  their own per-quant bytes-per-param + active-param counts.

## Related projects

- **[Keyhole](https://github.com/kylefoxaustin/keyhole)** — the edge-AI
  video intelligence project that produced all the bake-off data this
  sandbox wraps. 49-slide deck of results + the raw measurement
  scripts.
- **[keyhole-UI](https://github.com/kylefoxaustin/keyhole-UI)** — a
  Next.js app that demos the Keyhole pipeline on real videos. Upload a
  clip, see object detection + CLIP concept tags + semantic search.
  Different purpose than this sizer: a user-facing product demo vs. a
  hardware-sizing tool.

## License

Same as Keyhole (see parent repo).
