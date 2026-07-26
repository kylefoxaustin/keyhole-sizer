# Phase 3 — keyhole-sizer retrofit onto ratchet: parity report

**Date:** 2026-05-21
**Repo:** keyhole-sizer → **v1.1.0** (pending tag)
**Engine:** ratchet **v0.2.4**
**Scope:** Option **C analog**, D5 option (i) — full adoption of ratchet's
`Hardware`/`TIERS` + surface-side adapters.
**Status:** READY TO TAG pending reviewer sign-off on this report + a visual smoke.

---

## 1. What changed

keyhole keeps its own `project_llm`, `project_vision`, what-if subsystem, vision
pipelines, `platform_budget`, `kpi_breakdown`, and LLM catalog. It adopts ratchet
for `Hardware`, the tier instances/ladder, `hw_with_memory`,
`MEMORY_UPGRADE_OPTIONS`, the anchor loader, and the capability tables.

Net **−496 lines** (`npu_model.py` −723, `npu_anchors.py` deleted (−136),
`measured.py` +132 for the re-attach, `app.py` ±27).

**Adopted from ratchet (local defs deleted):** the `Hardware` dataclass + tier
instances + capability tables (`_NEUTRON_INT8_ONLY` / `_NPU_FULL_DTYPE` /
`_SM120_BLACKWELL`) + `hw_with_memory` + `MEMORY_UPGRADE_OPTIONS`;
`sizer/npu_anchors.py` → `ratchet.anchors` (byte-identical loader). `TIERS` is
keyhole's video ladder (6 tiers incl. i.MX 95) composed from ratchet's registry.

**Surface-side adapters** (ratchet's `Hardware` doesn't carry these):
- `capability_level(hw, dtype)` — maps keyhole's int8/fp8/bf16/fp16 onto
  ratchet's `capability_levels` (CapabilityInfo, keyed int8/fp8/'bf16/fp16'/q4_km);
  ratchet's enum `.value` matches keyhole's string taxonomy. (5 app.py call
  sites: `hw.capability_level(x)` → `capability_level(hw, x)`.)
- `_measured_edge_ms(hw, pipeline, res)` — reads ratchet's
  `measured_vision_overrides` (`{ms_per_inference, fps}`) and returns the float
  ms keyhole's vision code expects. (Replaces `hw.measured_edge_ms` at the
  vision read sites.)
- `_get_measured` / `_MOE_KEY` helpers for the LLM path.

**Anchor migration** (`sizer/measured.py`, `attach_keyhole_anchors_to_ratchet_tiers`,
runs at import — the PAI 5090-attach pattern):
- Tier-level Skippy-MoE-Q4 anchors (legacy flat fields) → ratchet's
  `measured_decode_overrides`/`measured_prefill_overrides` keyed `skippy_finetune`
  (LP5-64 29.27/613.2, Mid 37.85/2917.4, High 37.85/5835.3, 5090 249.8/6228).
- 5090 per-(model, quant) bundle cells → `measured_llm` (qwen 7b/32b/14b, llama,
  mistral). The MoE skippy_finetune cell is dropped here (single canonical
  location = `measured_decode_overrides`, per D2).
- Vision `measured_edge_ms` → `measured_vision_overrides` via wrap-the-leaf
  (`ms` → `{ms_per_inference, fps}`), MERGED into ratchet's existing entries.

---

## 2. Verification

**LLM anchor parity — the four migrated tier anchors reproduce v1.0.0** (decode @
100% share, all `measured_anchor`):

| tier | decode tok/s | source |
|---|---|---|
| NPU Low-LP5-64bit | 29.27 | measured_anchor |
| NPU Mid | 37.85 | measured_anchor |
| NPU High | 37.85 | measured_anchor |
| RTX 5090 (reference) | 249.8 | measured_anchor |
| 5090 qwen2.5-7B Q4 (bundle) | 183.9 | measured |

**AMENDMENT 5 fix — verified.** keyhole's `_maybe_anchor_overlay_llm` had the
same memory-upgrade guard; replaced with BW-scaling. Mid + MoE across the memory
dropdown now climbs monotonically (was: dropped to cross-class):

| Mid memory upgrade | decode tok/s | source |
|---|---|---|
| stock | 37.85 | measured_anchor |
| LPDDR5T @ 11.2 | 50.47 | same_class_anchor |
| LPDDR6 @ 12 | 54.07 | same_class_anchor |
| LPDDR6 @ 14 | 63.08 | same_class_anchor |

**AMENDMENT 1 — absence confirmed.** keyhole keeps its own `project_llm` and
gates model compatibility via `capability_level(hw, compute_dtype)` (the
adapter), which returns identical results to keyhole's former peak-TOPS
heuristic on the canonical tiers. No quant-scheme gating change; no cells flip.

**AMENDMENT 6 — carries.** keyhole inherits ratchet's i.MX 95 TDP = 10 W —
which Amendment 6 corrected *to* keyhole's own production value. So i.MX 95 TDP
is unchanged for keyhole (10 → 10).

**i.MX 95 measurement match check.** keyhole's i.MX 95 anchor (yolov8n INT8
@ 1080p = **32.0 ms**) **matches** ratchet's canonical entry (32.0 ms / 31.25
fps — same NXP eIQ source). No value conflict. *Bounded-but-unexpected:* the
resolution-key convention differs (keyhole `"1080p"` vs ratchet `"1920x1080"`);
the merge preserves both, keyhole reads its own `1080p` key, ratchet's
`1920x1080` entry is preserved-but-unused by keyhole (harmless).

**Vision projection magnitude sanity.** Measured anchors retrieved correctly:
i.MX 95 yolov8n @1080p = 32.0 ms (shown 42.67 ms at the default 75% NPU share —
keyhole's existing `measured_ms / share` contention model, unchanged), LP5X
yolov8n = 2.0 ms (2.67 @ 75%), 5090 sam3 = 95.0 ms, all `src=measured`. The
vision projection *logic* is untouched and every tier's silicon specs matched
ratchet exactly, so non-measured vision cells (BW-projected) are byte-identical
to v1.0.0.

**Capability badge diffs — NONE.** keyhole queries int8/fp8/bf16/fp16; ratchet's
canonical tables return identical levels for all six tiers (i.MX 95 / LP5-64 /
Mid INT8-only; LP5X / High full; 5090 int8 `tensor_compat`). keyhole never
queries `q4_km`, so PAI's badge flip does not recur.

**Boot:** `py_compile` clean; `AppTest` runs the full script with no uncaught
exception. (A behind-gate visual smoke is still recommended — see §4.)

---

## 3. Intended diffs vs v1.0.0

1. **NPU Low-LP5-64bit TDP: 10 → 20 W** (display only). keyhole inherits
   ratchet's Amendment-4 TDP ladder (10/15/20 for LP4/LP5-32/LP5-64). TDP is
   informational; not consumed by projection. All other tier specs identical.
2. **Memory-upgrade LLM anchor now BW-scales (Amendment 5).** A deliberate bug
   fix, not parity-preserving — Mid/High + MoE under a memory upgrade now climb
   monotonically from the stock measurement instead of dropping to cross-class.

No other projection diffs: keyhole keeps its own projection math, every silicon
spec matched ratchet, and capability badges are identical.

---

## 4. Flags / what I could not verify

- **CNN/vision overlay (`_maybe_anchor_overlay_cnn`) still has the
  memory-upgrade guard** (app.py ~line 911). It's the same *class* of issue as
  Amendment 5 but for vision, and was **out of Amendment 5's LLM scope** — left
  untouched. If you want vision measured-anchors to BW-scale under memory
  upgrades too, that's a follow-up decision (vision is BW-bound, so arguably
  yes, but it wasn't the reported bug).

  > ⚠ **CORRECTION appended 2026-07-26 — the bullet above was accurate on
  > 2026-05-23 and has been stale since 2026-05-27.** Do not act on it. It was
  > fixed 4 days after this report was written (`7bee0fc`, tagged v1.1.1) —
  > *in `app.py`* — then silently un-shipped when `49e6a63` (v2.0.0) replaced
  > `app.py` wholesale, leaving the fix stranded in the never-imported
  > `app_vertical_legacy.py`. The shipped product carried stock vision anchors
  > verbatim onto memory-upgrade clones, badged 🟢 `measured`, for 46 days.
  > **Now closed in the engine** (`sizer/npu_model.py::_anchor_bw_scale`,
  > v2.0.1) where a surface rewrite can't reach it. The `app.py ~line 911`
  > reference points at nothing; the symbol is gone from the live app. See
  > `CLAUDE.md` § "Vision Amendment 5 — CLOSED in the engine" for the full
  > chronology and the rules it produced.
- **Behind-the-gate visual walkthrough** — validated all projection / vision /
  capability / anchor paths programmatically, but recommend a quick visual smoke
  (tier × pipeline grid, the LLM tab incl. memory upgrades, the measured-anchors
  expander) before/after tagging.

---

## 5. Decisions honored
D1 (surface-side re-attach via measured.py), D2 (single canonical location:
MoE → measured_decode_overrides, 5090 bundle → measured_llm, duplicates
dropped), D3/Amendment 6 (i.MX 95 TDP corrected in ratchet v0.2.4), D4
(augmented dict — Mid carries both `qwen3_30b_a3b_moe` and `skippy_finetune`),
D5(i) (full Hardware adoption + adapters). Refinements: helper functions (not
direct repoints), wrap-the-leaf vision transform in measured.py, i.MX 95 merge
with match-check.

**Recommendation: tag keyhole v1.1.0** after a visual smoke. Parity holds except
the two intended diffs (one TDP display correction + the Amendment-5 fix).
