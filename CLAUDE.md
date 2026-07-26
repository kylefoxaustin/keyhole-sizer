# CLAUDE.md — keyhole-sizer (video NPU sizer)

Streamlit sizer for the Keyhole bake-off findings — vision pipelines + LLM +
platform budget. As of **v1.1.0** it is retrofitted onto **ratchet** (the shared
SoC sizing engine) — phase 3 of the engine consolidation.

## ratchet retrofit (v1.1.0, Option C analog)

keyhole depends on `ratchet>=0.2.4,<0.3.0` (the `<0.3.0` upper bound is
deliberate — surfaces bump their pin intentionally; they don't auto-upgrade).

**Adopted from ratchet** (local defs deleted): the `Hardware` dataclass + the
tier instances + the capability tables + `hw_with_memory` +
`MEMORY_UPGRADE_OPTIONS`; the anchor loader (`sizer/npu_anchors.py` → deleted,
now `ratchet.anchors`, byte-identical). `sizer/npu_model.py`'s `TIERS` is
keyhole's **video ladder** composed from ratchet's registry (6 tiers, including
the vision-only i.MX 95 ground-truth tier; omits LP4 / LP5-32bit).

**Kept surface-side** (Option C — keyhole keeps projection + vision): its own
`project_llm` (dict result) + what-if subsystem, `project_vision` + the vision
`PIPELINES`, `platform_budget`, `kpi_breakdown`, and the LLM catalog
(`llm_models.py`). Projection consolidation onto ratchet's `project_llm` is a
deliberate later pass.

### Surface-side adapter pattern (read this before adding dtype/vision code)

ratchet's `Hardware` is a typed engine object; it does **not** carry keyhole's
old per-instance method/field conventions. Three surface-side helpers in
`sizer/npu_model.py` bridge keyhole's UI conventions to ratchet's typed data —
**this is the canonical place to extend when you add a new dtype or
vision-overlay path:**

- **`capability_level(hw, dtype)`** — keyhole queries `int8/fp8/bf16/fp16` and
  expects a string; ratchet's `capability_levels` is `dict[str, CapabilityInfo]`
  keyed `int8/fp8/'bf16/fp16'/q4_km` with a `CapabilityLevel` *enum*. The helper
  maps the dtype key (bf16/fp16 → 'bf16/fp16') and returns the enum `.value`
  (ratchet's enum values match keyhole's string taxonomy). Falls back to a
  peak-TOPS heuristic. (Call it as `capability_level(hw, dtype)`, **not**
  `hw.capability_level(...)`.)
- **`_measured_edge_ms(hw, pipeline, res)`** — keyhole's vision code wants a
  float ms; ratchet stores `measured_vision_overrides` as
  `{pipeline: {res: {ms_per_inference, fps}}}`. The helper unwraps the leaf.
- **`_get_measured(hw, model_key, quant)`** — per-cell LLM lookup in keyhole's
  `model → quant` shape (ratchet's `get_measured_llm_cell` is workload-keyed and
  unused here).

### Anchor / vision re-attach

keyhole's per-tier measurements (LLM anchors + vision edge-ms) are re-attached
to ratchet's shared tier instances at import by
`sizer/measured.py:attach_keyhole_anchors_to_ratchet_tiers()` (the PAI
5090-attach pattern). Tier-level Skippy-MoE-Q4 anchors live in
`measured_decode_overrides`/`measured_prefill_overrides` (keyed `skippy_finetune`);
the 5090 per-(model, quant) bundle lives in `measured_llm`; vision lives in
`measured_vision_overrides` (single canonical location per measurement).

## Intended diffs vs v1.0.0 (see PHASE3_PARITY_REPORT.md)

keyhole keeps its own projection math, so projections are otherwise identical.
Two deliberate differences:

1. **NPU Low-LP5-64bit TDP: 10 → 20 W** (display only). keyhole inherits
   ratchet's Amendment-4 TDP ladder (10/15/20 for LP4/LP5-32/LP5-64). TDP is
   informational — not consumed by projection.
2. **Memory-upgrade LLM anchor now BW-scales (Amendment 5).** A deliberate bug
   fix, not parity-preserving: Mid/High + MoE under a memory upgrade now climb
   monotonically from the stock measurement (e.g. Mid+LPDDR6-14 = 63.08 tok/s)
   instead of dropping to cross-class. keyhole's `_maybe_anchor_overlay_llm`
   BW-scales by `mem_bandwidth_gbs / stock_mem_bandwidth_gbs` for `bw_projected`
   clones (mirrors ratchet v0.2.3 / ADR 011 Amendment 5).

**No AMENDMENT-1 cells flip** (keyhole keeps its `compute_dtype` gate via the
`capability_level` adapter, identical results). **No capability badge diffs**
(keyhole queries int8/fp8/bf16/fp16; never `q4_km`). **i.MX 95 measurement match
confirmed** — 32.0 ms matches between keyhole and ratchet (same NXP eIQ source;
keyhole reads its `1080p` resolution key, ratchet's `1920x1080` entry is
preserved by the merge).

## Vision Amendment 5 — CLOSED in the engine (v2.0.1). Read this before touching anchors.

**Status: fixed, in `sizer/npu_model.py`.** A memory-upgrade clone BW-scales its
measured *vision* anchor (`_anchor_bw_scale()`, applied at the
`measured_override_ms` lookup in `project_vision`), and the badge degrades
🟢 `measured` → 🟡 `same_class_anchor` on `bw_projected` clones — matching what
`project_llm` already did. Invariant now enforced across all 15 vision
tier×pipeline×resolution×upgrade cells: `fps_ratio == bw_ratio` exactly.

**This section used to say the opposite, and the history is the lesson — it is
why the fix now lives in the engine and not in `app.py`:**

1. v1.1.0 (05-23) deferred the vision mirror of Amendment 5. *This file was
   written that day and said so.*
2. `7bee0fc` (05-27, tagged **v1.1.1**) actually fixed it — but wrote the fix
   into **`app.py`**'s `_maybe_anchor_overlay_cnn()`. This file was never
   updated, so its "deferred" note was wrong 3d 19h after being written.
3. `49e6a63` (06-10, **v2.0.0**) replaced `app.py` wholesale to promote the
   horizontal layout. **The fix went with it.** It survived only in
   `app_vertical_legacy.py`, which nothing imports — so the shipped product
   carried stock vision anchors verbatim onto upgraded parts, badged
   🟢 `measured`, for **46 days** (up to 54% understated fps). Nobody reverted
   anything; a UI refactor silently un-shipped a tagged release's fix.
4. v2.0.1 (this fix) re-applies it **in the engine**, where a surface rewrite
   cannot reach it.

**Rules this buys — follow them:**

- **Anchor-resolution and provenance logic goes in `sizer/`, never in `app.py`.**
  `app.py` is replaceable (it has been replaced once, wholesale). A correction
  written into the surface has a refactor-shaped expiry date.
- **Any `bw_projected` clone must degrade its badge.** It is a part that was
  never built and never measured; 🟢 `measured` on one is a DERIVED number
  wearing a MEASURED tag (Fleet Law 1). Check both workloads — LLM and vision
  resolve anchors on separate paths and have drifted apart before.
- **A conservative error is not a safe error, it is a durable one.** This bug
  *understated* the hardware, which is why it lived 46 days. `pai-sizer`'s
  analogous bug flattered by 14% and was caught much sooner.
- Written up as a primary source for the fleet's IEEE paper:
  `claude-connect/docs/paper/cases_sizer.md`, Cases 1–2.

## Running

`streamlit run app.py` (password gate before the main UI). Measured silicon
anchors load at runtime from gitignored `.streamlit/secrets.toml` (KEY-not-VALUE
discipline — values are credentials, never committed).
