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

# 2. Option A — reuse an existing venv (if you already have the Keyhole
#    project's venv, it has streamlit+plotly already):
source ~/.virtualenvs/keyhole/bin/activate

# 2. Option B — fresh venv with plain stdlib venv:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Option C — virtualenvwrapper:
mkvirtualenv keyhole-sizer --python=/usr/bin/python3.10
pip install -r requirements.txt
#    Note: on some systems virtualenvwrapper + virtualenv 20.x fails to
#    create the bin/ dir. If `workon keyhole-sizer` gives "no activate
#    script", remove the half-built env and use Option B instead:
#       rm -rf ~/.virtualenvs/keyhole-sizer && python3 -m venv .venv

# 3. Launch
streamlit run app.py
```

Browser opens to `http://localhost:8501`. No GPU needed — this app is
pure projection math on top of already-measured numbers.

## Deployment (Streamlit Community Cloud)

The app is deployed at **share.streamlit.io** with a shared-password gate.
Auto-redeploys on every push to `main`.

To set up your own deployment:

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   grant access to this repo.
2. **New app** → pick `kylefoxaustin/keyhole-sizer`, branch `main`,
   main file `app.py`.
3. Click **Advanced settings → Secrets** and add:
   ```toml
   PASSWORD = "your-shared-password-here"
   ```
4. Deploy. The password gate is active whenever the `PASSWORD` secret is set;
   when absent (e.g. local dev via `streamlit run app.py`), the gate is
   bypassed so you don't have to type it during development.

## Platform-budget CSV export

The sizer can emit **additive platform-budget CSV rows** for feeding into an
SoC-level spreadsheet (total NPU duty cycle, DDR GB/s consumed, MB resident,
power). Three ways to get the data:

1. **UI button** — every rendered config has a **💾 Download platform budget
   CSV (current config)** button that emits a single vision row + (if
   enabled) a single LLM row for the currently-selected config.
2. **CLI** — `scripts/export_platform_budget.py` emits a row for any
   combination of pipeline × HW × resolution × streams × optional LLM.
   Run `python scripts/export_platform_budget.py --list` for valid keys.
3. **Full matrix** — `scripts/export_platform_matrix.py` iterates every
   preset HW tier × pipeline × resolution × stream count (1/2/4/8/16) +
   every LLM quant × workload, writing `data/platform_budget_matrix.csv`
   (~585 rows). Custom HW is skipped (use the UI download for custom).

**Schema** (per row = one workload slot):
- `ss_*` columns (duty cycle, DDR GB/s, TOPS, MB resident, watts,
  throughput) are **additive** across rows at the platform level.
- `peak_*` columns (per-frame ms, peak GB/s, peak TOPS) are NOT additive —
  they're per-workload ceilings.
- `hw_*` columns are duplicated on every row so each row is self-contained.
- `sizer_commit_sha` + `export_timestamp_iso` let you trace a row back to
  the sizer revision that emitted it.

**Caveats baked into the CSV header comments** (read before using for procurement):
- Power is TDP × duty-cycle approximation, NOT measured per-workload.
- NPU Low-LP5 / Low-LP5X / Mid / Mid-INT8 / High numbers are
  bandwidth-scaled from RTX 5090 measurements, NOT measured on actual
  NPU silicon.

**Consume in pandas:** `pd.read_csv(path, comment='#')`.

## What you can tune

**Hardware** (sidebar):
- **Tier preset:** `NPU Low-LP5` (64-bit LPDDR5 @ 6.4 GT/s, 51.2 GB/s;
  dense INT8-only silicon class) / `NPU Low-LP5X` (same 64-bit bus,
  LPDDR5X @ 8.4 GT/s, 67.2 GB/s) / `NPU Mid` (128-bit LPDDR5X @ 8.4
  GT/s, 134.4 GB/s — Keyhole shipping target, BF16/FP8-capable) /
  `NPU Mid-INT8` (same BW as Mid, INT8-only silicon) / `NPU High`
  (128-bit LPDDR5X @ 11.2 GT/s, high-bin) / `Custom` (roll your own).
  All presets assume 70% bandwidth efficiency.
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
