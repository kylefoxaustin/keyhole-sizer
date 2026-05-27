# keyhole-sizer

[![version](https://img.shields.io/badge/version-v1.1.0-blue)](https://github.com/kylefoxaustin/keyhole-sizer/releases/tag/v1.1.0)
[![streamlit](https://img.shields.io/badge/streamlit-live-FF4B4B)](https://keyhole-sizer.streamlit.app)
[![engine](https://img.shields.io/badge/engine-ratchet%20v0.2.4-green)](https://github.com/kylefoxaustin/ratchet)

Interactive NPU sizing sandbox for the
[Keyhole](https://github.com/kylefoxaustin/keyhole) edge-AI bake-off findings.

A Streamlit app that wraps the measured bake-off data in tunable controls:
pick an NPU tier (or build a custom one), a vision pipeline, concurrent
stream count, and whether an LLM co-exists on the same silicon — then
watch live FPS / tok/s / VRAM-fit / duty-cycle projections.

**Companion app:**
[personal-ai-assistant-sizer](https://github.com/kylefoxaustin/personal-ai-assistant-sizer)
(LLM-only sizing for the Skippy assistant deployment). Both sizers share
schema, source taxonomy, anchor-secrets mechanism, tab structure, and — as
of v1.1.0 — the same shared engine. Pick the one that matches your
workload.

**Engine:** As of v1.1.0 the canonical NPU tier registry, capability
taxonomy, `hw_with_memory` memory-upgrade clones, anchor loader, and
Hardware dataclass all live in the shared [`ratchet`](https://github.com/kylefoxaustin/ratchet)
package (pinned to v0.2.4). Surface-side keyhole-sizer keeps its UI,
projection layer, vision pipelines, LLM catalog, platform-budget, and
KPI breakdown. The retrofit cut –496 lines net while keeping behavior
parity on the canonical tiers + fixing the memory-upgrade anchor
discontinuity (Mid + LPDDR6-14 now climbs from 37.85 → 63.08 tok/s
instead of dropping to cross-class projection). See the
[v1.1.0 release notes](https://github.com/kylefoxaustin/keyhole-sizer/releases/tag/v1.1.0)
for the full diff.

If you've read the Keyhole deck and want to answer **"what if my NPU had
a 96-bit bus instead of 128?"** or **"how many 480p streams can I run
with a 2 Hz LLM query rate?"**, this is the tool.

## Install & run

```bash
git clone https://github.com/kylefoxaustin/keyhole-sizer.git
cd keyhole-sizer

# Option A — fresh venv with plain stdlib venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B — virtualenvwrapper
mkvirtualenv keyhole-sizer --python=/usr/bin/python3.10
pip install -r requirements.txt

# Option C — reuse the Keyhole project's venv (has streamlit+plotly already)
source ~/.virtualenvs/keyhole/bin/activate

# Launch
streamlit run app.py
```

Browser opens to `http://localhost:8501`. No GPU needed — this app is
pure projection math on top of already-measured numbers.

**Note on the ratchet engine dependency:** `requirements.txt` pulls
ratchet from its public GitHub source repo (not PyPI — the name
`ratchet` on PyPI is an unrelated project). The pin is `ratchet @
git+https://github.com/kylefoxaustin/ratchet.git@v0.2.4`. Bumping
ratchet means editing that tag in the URL and pushing. If you're
running locally with ratchet as an editable install (`pip install -e
~/path/to/ratchet`), pip prefers your editable over the git pin —
useful for engine co-development.

## Deployment (Streamlit Community Cloud)

The app is deployed at **https://keyhole-sizer.streamlit.app** with a
shared-password gate. Auto-redeploys on every push to `main`.

To set up your own deployment:

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   grant access to this repo.
2. **New app** → pick `kylefoxaustin/keyhole-sizer`, branch `main`,
   main file `app.py`.
3. **Advanced settings → Secrets**:
   ```toml
   PASSWORD = "your-shared-password-here"
   ```
   (Optional — when absent, the password gate is bypassed so local dev
   doesn't require typing it.)
4. Deploy.

**Note:** changes to `app.py` auto-reload; changes to files under `sizer/`
require a manual reboot from the Streamlit Cloud console. (See
[`memory/project_streamlit_deploy.md`](https://github.com/kylefoxaustin/personal-ai-framework)
in the framework repo for the inotify-stuck-reload trap.)

## What you can tune

### Hardware

| Tier preset | Memory | Effective BW | INT8 / FP path |
|---|---|---:|---|
| **i.MX 95** (ground truth) | 32-bit LPDDR5 @ 6.4 GT/s | 25.6 GB/s × 0.70 | INT8-only (Neutron) |
| Low-LP5-64bit | 64-bit LPDDR5 @ 6.4 GT/s | 51.2 GB/s × 0.70 | INT8-only |
| Low-LP5X | 64-bit LPDDR5X @ 8.4 GT/s | 67.2 GB/s × 0.70 | INT8-only |
| NPU Mid | 128-bit LPDDR5X @ 8.4 GT/s | 134.4 GB/s × 0.70 | INT8-only |
| **NPU High** *(default)* | 128-bit LPDDR5X @ 8.4 GT/s | 134.4 GB/s × 0.70 | FP-capable (FP16 / FP8 / BF16) |
| RTX 5090 | GDDR7 @ 28 GT/s | reference cell | full FP + INT8 + INT4 |
| Custom | configurable | computed | configurable |

Mid + High share memory bandwidth (Class 3b `LP5X-8.4-128b`); High
differentiates on **compute** (1.375× TOPS), **capacity** (1.33× DRAM),
and **TDP** (1.6×), NOT memory BW. FP recipes pin to High. Default tier
is High to keep the default LLM (fp16 dense) compatible on first load.

**Optional memory upgrade overlays** on Mid + High: LPDDR5T @ 11.2 GT/s
(179 GB/s), LPDDR6 @ 12 GT/s (192 GB/s), LPDDR6 @ 14 GT/s (224 GB/s).
BW-projected — flips the source badge to 🟡 same-class.

**Custom mode:** bus width, memory type, data rate, BW efficiency, peak
BF16/FP8 TOPS, compute efficiency, DRAM capacity, TDP.

**NPU share** (sidebar): 100% / 75% / 50% / 25%. Models SoC bus contention
from display / camera / audio paths. Default 75% for NPU tiers, 100% for
5090 (dedicated VRAM). Scales BW-bound paths only.

### Vision workload

- **Pipelines (23 entries):** SAM 3 baseline, EfficientSAM-Small, Hybrid V2
  variants, TRT FP8 shipping stack, YOLO-only, ResNet-50 vendor-compare
  pipelines (INT8 + INT4), CLIP, ViT alternatives, EfficientSAM3,
  YOLOE-26 — organized into 5 narrative tracks via the sidebar
  pipeline picker.
- **Resolution:** 720p / 1080p / 4K.
- **Concurrent streams:** 1..16 (YOLO batching applied automatically).
- **Compiler quality:** slider for projected pipelines (non-applicable
  when measured-silicon anchor fires).

### LLM workload

- **17-model catalog** with role icons in the dropdown:
  🚀 production (1) · 🔬 fine-tune (7) · 📚 stock base (6) · ⚙️ perf
  reference (3). Default: 🚀 **Skippy 7B v4** (Qwen2.5-7B FT, current
  production). Models incompatible with the selected NPU tier sink to
  the bottom of the dropdown with a 🔴 prefix.
- **Workload pattern:** plain_chat (reference) · long_form_reasoning ·
  tool_use · rag_long_context · cold_start. Scales decode tok/s + TTFT
  by reference multipliers (26× spread across categories). Disabled
  when dtype-mismatch is in play (model won't actually run).
- **Precision (quant):** Q4_K_M / Q5_K_M / Q8_0 / FP8 / FP16 — picker
  shape depends on the model's `compute_dtype` and the NPU tier's
  capability levels.
- **Queries per minute** (slider) + **answer length** (short ~200 tok or
  RAG 8K prompt + 2K response).

## What you see

The app uses a tab strip below the headline metric row. Tab order adapts
to which workloads are enabled:

| Tab | Always | When |
|---|---|---|
| Overview | ✓ | metric tiles + headline cards |
| Accuracy | | LLM on (Finding 4 cross-model + per-category) |
| Precision | | LLM on (precision-axis ladder + capability levels) |
| Performance | | LLM on (cross-tier comparison) |
| Stream scaling | | Vision on (per-stream / total FPS curves) |
| Duty-cycle | | Vision + LLM on (FPS vs queries/min) |
| **KPIs** | ✓ | CSV exports + per-pipeline KPI spreadsheet preview |
| Detail | ✓ | raw projection dicts, debug surface |

Headline metrics: Vision FPS per stream · Total system FPS · LLM decode
tok/s · TTFT · End-to-end latency · Memory fit. Each tile carries a
source badge (🟢 measured / 🟢 measured_anchor / 🟡 same-class /
🟠 cross-class projected) so users can see data pedigree at a glance.

## Where the numbers come from

The sizer composes **three layers**, in priority order:

1. **Measured-silicon anchor overlay** (🟢) — when a real-silicon
   measurement exists for the selected (tier, precision, model) cell, it
   hot-swaps the BW-projected baseline. NPU silicon measurements live in
   `.streamlit/secrets.toml` (gitignored); see
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)
   for the schema. RTX 5090 measurements are in-tree (public bake-off
   data).
2. **Phase 1 measured override** (🟢) — `Hardware.measured_edge_ms` maps
   `(pipeline, resolution) → ms` for vision; `Hardware.measured_llm` maps
   `(model, quant) → {decode_tok_s, prefill_tok_s}` for LLMs. When a
   tier has the cell, `project_*()` returns measured ms verbatim.
3. **Phase 2 projection** — `max(bw_floor, compute_floor) + overhead` with
   tier-specific calibration constants. Source badge `cross_class` 🟠
   when no anchor in this tier_family; `same_class_anchor` 🟡 when
   BW-scaled within family from a measured anchor.

Bake-off data origins:

| Baseline | Source |
|----------|--------|
| YOLO-seg FP8 edge ms @ resolution | `keyhole/bakeoff_trt_yolo.py` |
| CLIP FP8 edge ms | `keyhole/bakeoff_trt_clip.py` |
| YOLO batching curve (B=1..16) | `keyhole/bakeoff_concurrency.py` |
| ResNet-50 INT8/INT4 ncu DRAM | `keyhole/scripts/export_ncu_for_sizer.py` |
| SAM 3 / EfficientSAM / MobileSAM edge ms | `keyhole/bakeoff_sam_variants.py`, `bakeoff_fp8.py` |
| LLM 5090 perf cells (TTFT, decode) | `keyhole/bakeoff_llm.py` |
| Qwen3-30B-A3B + Qwen2.5 dense GGUF sizes | `keyhole/bakeoff_llm.py` |
| LLM accuracy (Finding 4 per-category) | `personal-ai-framework` eval harness |
| NPU silicon anchors (Mid / High / Low-LP5X) | `.streamlit/secrets.toml` (gitignored) |

Vision pipelines on NPUs are **bandwidth-bound** in the common case, so
edge ms scales inversely with effective bandwidth on un-anchored cells.
LLM decode is bandwidth-bound on `active_params × bytes_per_param` (MoE
wins — Qwen3-30B-A3B Q4_K_M streams only ~1.65 GB per token despite the
30B total params). LLM prefill (TTFT) is compute-bound — bigger TOPS
helps TTFT, more BW does not.

## Platform-budget CSV export

The **KPIs tab** carries two CSV download buttons + two KPI-spreadsheet
preview buttons:

- **💾 This config** — single platform-budget row for the current
  (HW × pipeline × resolution × streams × LLM) configuration.
- **📦 All configs** — full matrix (~585 rows) of every preset HW tier ×
  pipeline × resolution × stream count × LLM (quant × workload ×
  answer_kind). Custom HW is skipped — use "This config" for custom.
- **📊 KPI spreadsheet (all models / this model)** — per-pipeline KPI
  preview table + ⬇ Download XLSX. Vision-only (per-pipeline rows make
  no sense LLM-only).

CLI alternatives in `scripts/`:
- `export_platform_budget.py` — one row for any specific config.
- `export_platform_matrix.py` — full matrix to `data/platform_budget_matrix.csv`.

**Schema:**
- `ss_*` columns (duty cycle, DDR GB/s, TOPS, MB resident, watts,
  throughput) are **additive** across rows at the platform level.
- `peak_*` columns (per-frame ms, peak GB/s, peak TOPS) are NOT
  additive — per-workload ceilings.
- `hw_*` columns duplicated per row so each row is self-contained.
- `sizer_commit_sha` + `export_timestamp_iso` trace a row back to the
  sizer revision that emitted it.

**Caveats baked into the CSV header `#` comments** (read before using
for procurement):
- Power is `TDP × duty-cycle` approximation, NOT measured per-workload.
- NPU Low/Mid/High vision numbers without anchors are BW-scaled from
  RTX 5090 measurements, NOT measured on actual NPU silicon.

**Consume in pandas:** `pd.read_csv(path, comment='#')`.

## Anchor-secrets system (private silicon measurements)

Real NPU silicon measurements for the LLM + CNN cells in the spec live
in `.streamlit/secrets.toml` (gitignored). The loader at
[`sizer/npu_anchors.py`](sizer/npu_anchors.py) reads them at runtime and
hot-swaps them into projection results.

**Discipline rule:** measurement values are NEVER committed, logged, or
quoted in source / bus traffic / commit messages — refer by **KEY**
(`npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe.tokps`) not VALUE. Treat
like credentials.

For your own deployment, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml`, fill in your measurements, and the
appropriate cells will flip from 🟠 projected to 🟢 measured. Cells
without an anchor entry fall through to the projection path
transparently.

Coverage at v1.1.0: **9/9 LLM + 6/6 CNN spec cells** reachable from the
headline tiles (unchanged from v1.0.0 — the retrofit kept anchor
reachability identical).

## Limitations

- **Synthetic projection, not simulation** outside anchored cells. If
  you push custom sliders way out of range (e.g. 2048-bit bus @ 100 GT/s),
  the projection still linearly scales — the math doesn't model cache
  hierarchies, tiling, or NoC topology.
- **Pipeline list is fixed** to what Keyhole measured. Adding a new
  pipeline requires three registrations (`PIPELINES` + `PIPELINE_TRACKS`
  + `PIPELINE_STAGES`); startup assertion fires if any are missing.
- **LLM catalog is fixed** to the 17 entries in `sizer/llm_models.py`.
  Adding a new model requires a `LLMModel` entry + (optionally) a
  `measurement_alias` pointing perf-anchor lookup at an existing entry,
  + (optionally) a `.streamlit/secrets.toml` cell for the
  measured-silicon overlay.
- **NPU silicon vision anchors are sparse outside the i.MX 95 + Low-LP5X
  measured cells.** Mid / High vision cells fall through to BW-scaling
  from 5090 unless you populate `.streamlit/secrets.toml`. LLM cells
  have broader coverage.

## Related projects

- **[personal-ai-assistant-sizer](https://github.com/kylefoxaustin/personal-ai-assistant-sizer)** —
  LLM-only sizing for the Skippy assistant deployment. Same anchor-secrets
  mechanism, same tab structure, same source taxonomy. Pick this one if
  you don't care about the vision workload.
- **[Keyhole](https://github.com/kylefoxaustin/keyhole)** — the edge-AI
  video intelligence project that produced the bake-off data this sandbox
  wraps. 63-slide deck of results + the raw measurement scripts.
- **[personal-ai-framework](https://github.com/kylefoxaustin/personal-ai-framework)** —
  the Skippy fine-tuning / evaluation framework. Source of the Finding 4
  per-category accuracy data surfaced in the Accuracy tab.
- **[keyhole-UI](https://github.com/kylefoxaustin/keyhole-UI)** — a
  Next.js app that demos the Keyhole pipeline on real videos. Different
  purpose: user-facing product demo vs. this hardware-sizing tool.

## Version history

| Version | Date | Highlights |
|---|---|---|
| **v1.1.0** | 2026-05-23 | Retrofit onto shared [`ratchet`](https://github.com/kylefoxaustin/ratchet) engine v0.2.4 (phase 3 of cross-repo engine consolidation). Adopted ratchet for `Hardware` / `TIERS` / `hw_with_memory` / `MEMORY_UPGRADE_OPTIONS` / anchor loader / capability tables. Net –496 lines. Surface-side adapters bridge keyhole's UI conventions to ratchet's typed data. Per-tier anchors re-attached at import via `measured.py::attach_keyhole_anchors_to_ratchet_tiers()`. Two intended behavior diffs vs v1.0.0: (1) NPU Low-LP5-64bit TDP 10 → 20 W (display only, no projection impact, ratchet's Amendment-4 TDP ladder); (2) memory-upgrade LLM anchor now BW-scales (Amendment 5 bug fix — Mid + LPDDR6-14 climbs from 37.85 → 63.08 tok/s instead of dropping to cross-class). `requirements.txt` pins ratchet via `git+https` URL (PyPI's "ratchet" is an unrelated package). |
| **v1.0.0** | 2026-05-18 | First tagged release. Recovery point ahead of cross-repo engine-extraction work. Captures: 17-entry LLM catalog with role icons, 23-pipeline coverage (incl. 4-bit-weight CNN variants), anchor-secrets system (9/9 LLM + 6/6 CNN reachable), 8-tab UX mirroring PAI sizer (Overview · Accuracy · Precision · Performance · Stream · Duty · KPIs · Detail), 4-state source taxonomy + NPU_share third factor + Phase 2 compute-clamp, 5-state workload-pattern multipliers, capability-levels taxonomy, measurement_alias mechanism, category_deltas dict-of-dicts schema. |

## License

Same as Keyhole (see parent repo).
