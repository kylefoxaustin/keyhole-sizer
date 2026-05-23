# Phase 3 — keyhole-sizer retrofit onto ratchet (recon + migration plan)

**Date:** 2026-05-21
**Repo:** keyhole-sizer (currently `v1.0.0`, production)
**Engine:** ratchet **v0.2.3**
**Target:** keyhole-sizer **v1.1.0**
**Scope:** Option **C analog** (confirmed) — adopt ratchet for the canonical
engine pieces; keep keyhole's projection, what-if, vision, platform, catalog.
**Status:** RECON COMPLETE — read-only. Holding for sign-off before destructive
edits.

This doc is the **contract for the phase-3 retrofit session** (same role as
PHASE2_RETROFIT_PLAN.md for PAI). If execution diverges, stop and surface before
destructive edits.

---

## 0. Headline

keyhole-sizer is a **video** sizer (~7,336 lines): vision pipelines + LLM +
platform budget. The vision half has **no ratchet analog** (design §1 non-scope)
and stays entirely surface-side. The retrofit adopts ratchet only for
`Hardware`/`TIERS`/`hw_with_memory`/`MEMORY_UPGRADE_OPTIONS`/the anchor
loader/capability tables.

The retrofit is **bigger than PAI's in exactly one place**: keyhole carries
tier-level **MoE anchors on four tiers** (LP5-64, Mid, High, 5090-ref) in *two*
representations — legacy flat fields **and** quant-keyed `measured_llm` dicts —
that ratchet's registry doesn't have. Migrating those onto ratchet's schema
(re-attaching to ratchet's tier instances) is the central lift, and it raises
the real decision points (§5).

---

## 1. Verification results (the three reviewer asks)

### 1a. Tier-spec comparison vs ratchet v0.2.2 (Amendment-4 confirmation)
keyhole's ladder = i.MX 95, Low-LP5-64bit, Low-LP5X, Mid, High, 5090-ref
(omits LP4 / LP5-32bit — its own ladder). Compared field-by-field against
ratchet:

- **All silicon-fact fields MATCH** for every shared tier — peak TOPS
  (`0/200/0` Mid, `200/400/400` High, `209/419/419` 5090, `0/2/0` i.MX95,
  `50/100/100` LP5X), mem BW, capacity, bus width, mem type, data rate. This
  **empirically confirms Amendment 4's load-bearing assumption** — keyhole and
  PAI/ratchet agree on the silicon facts (notably NPU High = 40 W, which keyhole
  independently has).
- **Two TDP-only disagreements** (informational; TDP is not consumed by
  projection):
  - NPU Low-LP5-64bit: keyhole **10 W** vs ratchet **20 W**.
  - NPU i.MX 95: keyhole **10 W** vs ratchet **8 W**.
  → See Decision **D3**.

### 1b. Anchor consumption pattern
Confirmed keyhole uses **both** patterns, like PAI:
- **Overlay:** `_maybe_anchor_overlay_llm(r, model_key, hw, tier_name, npu_share,
  workload)` (app.py:840) — hot-swaps a projection result, with workload
  multipliers (`decode_p50_mult`, `ttft_p50_mult`). Also a CNN overlay
  (`_maybe_anchor_overlay_cnn`) for vision.
- **Direct-loader:** `_anchor_llm.bytes_per_token(...)`, `.badge`, `.tokps` in a
  measured-anchors view (app.py:~1970).

`sizer/npu_anchors.py` is **byte-identical** to PAI's (and to ratchet's lifted
loader). The loader swap (`from ratchet.anchors import load_llm_anchor,
load_cnn_anchor`) + `git rm sizer/npu_anchors.py` is a **no-op behaviorally**.

### 1c. Legacy flat-field enumeration + MoE catalog key
**MoE catalog key in keyhole:** `skippy_finetune` (sizer/llm_models.py:238;
`instruct_moe_stock` carries `measurement_alias="skippy_finetune"`).

**Flat fields** (`measured_llm_q4_decode_tok_s`, `measured_llm_ttft_1k_sec`):
- **Defined:** Hardware fields, npu_model.py:43-44.
- **Written (tier defs):** LP5-64 `29.27 / 1.67` (256-257), Mid `37.85 / 0.351`
  (318-319), High `37.85 / 0.1755` (364-365), 5090-ref `249.8 / 0.165`
  (415-416). All four tiers **also** carry a `measured_llm` per-cell dict with
  the same values (`{"skippy_finetune": {"Q4_K_M": {decode, prefill}}}`), so the
  flat fields are **legacy duplicates** of the per-cell data.
- **Read:** `project_llm` legacy path (npu_model.py:1799, 1805, 1822-1823) and
  `hw_with_memory` BW-scaling (597-599, 623).
- **No surface-side reads** in app.py / other modules — all reads are inside
  npu_model.py's projection + clone helpers.

Coverage gap vs ratchet: ratchet's registry carries the MoE anchor on **Mid
only** (`measured_decode_overrides={"qwen3_30b_a3b_moe": 37.85}`,
`measured_prefill_overrides={...: 2849.0}`). keyhole's LP5-64 (29.27, a *vendor*
anchor — "NOT Skippy-specific" per the tier comment), High (37.85/5835), and
5090-ref (249.8/6228) anchors are **not** in ratchet.

---

## 2. File-level inventory

`app.py` imports from 8 sizer modules. Retrofit touch points:

| Import / module | Disposition |
|---|---|
| `from sizer.npu_anchors import load_llm_anchor, load_cnn_anchor` | → `from ratchet.anchors import ...`; `git rm sizer/npu_anchors.py` |
| `sizer.npu_model`: `Hardware`, `TIERS`, `NPU_MID` | → source from ratchet; `TIERS` = keyhole's ladder composed from ratchet (incl. `IMX95_MEASURED`) |
| `sizer.npu_model`: `MEMORY_TYPES`, `PIPELINES`, `WORKLOAD_CATEGORIES`, `BYTES_PER_PARAM`, `describe_hw`, `project_vision`, `project_llm`, `theoretical_bandwidth`, `vision_fps_under_llm_load`, `workload_distribution_on_hw`, `workload_multiplier` | **stay surface-side** (vision + keyhole projection + keyhole tables) |
| `hw_with_memory` / `MEMORY_UPGRADE_OPTIONS` | → source from ratchet (keyhole's `MEMORY_TYPES` UI maps onto ratchet's `hw_with_memory`) |
| `sizer.precision`: `CAPABILITY_LABELS`, `CAPABILITY_DESCRIPTIONS` | keyhole-specific UI labels — **stay**; the inline `capability_levels` tables (`_NEUTRON_INT8_ONLY_CAPABILITY` etc.) → ratchet's canonical tables |
| `sizer.llm_models` (catalog, 1313 lines), `sizer.llm_quant_levels`, `sizer.platform_budget`, `sizer.kpi_breakdown`, `sizer.measured` | **stay surface-side** |

Deleted from `npu_model.py`: `Hardware` class, the tier instances, `TIERS`
literal, `hw_with_memory`, `MEMORY_UPGRADE_OPTIONS`, the inline
`_NEUTRON_INT8_ONLY_CAPABILITY` / `_NPU_FULL_DTYPE_CAPABILITY` /
`_SM120_BLACKWELL_CAPABILITY` tables (→ ratchet), and the legacy flat-field
projection path (see §3). Kept + rewired: `project_llm`, `project_vision`, the
what-if/vision helpers, `_find_same_family_anchor`, `BYTES_PER_PARAM`,
`PIPELINES`, `MEMORY_TYPES`, `WORKLOAD_CATEGORIES`.

**Adapters (mirror PAI):** `_get_measured` (ratchet's `get_measured_llm_cell` +
keyhole's `measurement_alias` fallback) and a key adapter if keyhole reads
ratchet's canonical-keyed Mid anchor (see D4).

---

## 3. Legacy-field migration mapping (the central lift)

keyhole's tier-level MoE anchors must move onto ratchet's tier instances, in
ratchet's schema, at import (the way `measured.py` already attaches 5090
measurements). Proposed mapping (flat → `measured_decode_overrides` +
`measured_prefill_overrides`, keyed `skippy_finetune`; prefill_tok_s =
1024 / ttft_1k_sec):

| Tier (ratchet instance) | decode_overrides[`skippy_finetune`] | prefill_overrides[`skippy_finetune`] | In ratchet today? |
|---|---|---|---|
| `NPU_LOW_LP5_64BIT` | 29.27 | 613.2 | ✗ (keyhole vendor anchor) |
| `NPU_MID` | 37.85 | 2849.0 | ✓ but keyed `qwen3_30b_a3b_moe` |
| `NPU_HIGH` | 37.85 | 5835.3 | ✗ |
| `RTX_5090_REFERENCE` | 249.8 | 6228.0 | ✗ (also has per-cell bundle) |

Then **delete the flat-field path** from keyhole's `project_llm` (its data now
lives in `measured_decode_overrides`/`measured_prefill_overrides`, which ratchet's
`hw_with_memory` BW-scales correctly — and which the Amendment-5 overlay also
BW-scales). keyhole's per-cell `measured_llm` dicts (quant-keyed) are a separate
representation — see D2.

---

## 4. Amendment 5 patch (in scope, lands in keyhole's overlay)

**AMENDMENT 5 (2026-05-21):** memory-upgrade clones BW-scale the private anchor's
decode instead of dropping it (already shipped in ratchet v0.2.3 / ADR 011; PAI
v1.1.0 carries the mirror). keyhole's `_maybe_anchor_overlay_llm` (app.py:862-864)
has the same `mem_data_rate_gtps != 8.4 → return r` guard. Patch:

```python
# REMOVE the guard:
#   if abs(getattr(hw, "mem_data_rate_gtps", 0.0) - 8.4) > 0.05:
#       return r
# ...and after loading the anchor, BW-scale decode for memory-upgrade clones
# (decode is BW-bound; TTFT held at stock). Stock tiers keep ratio 1.0:
    bw_ratio = 1.0
    if getattr(hw, "bw_projected", False) and hw.stock_mem_bandwidth_gbs:
        bw_ratio = hw.mem_bandwidth_gbs / hw.stock_mem_bandwidth_gbs
    r2["decode_tok_s"] = anchor.tokps * decode_mult * bw_ratio
    # ttft_1k_sec handling unchanged (held at stock unless ttft_mult applies)
```

---

## 5. Decision points

**D1 — Where do keyhole's extra tier anchors (LP5-64, High, 5090-ref) live?**
ratchet's registry only has the Mid MoE anchor. keyhole's LP5-64 (29.27, an
explicit *vendor* anchor "NOT Skippy-specific"), High (37.85/5835), and 5090-ref
(249.8/6228) are keyhole-specific. **Recommend: re-attach surface-side** to
ratchet's tier instances at import (in keyhole's `measured.py`), as
`measured_decode_overrides`/`measured_prefill_overrides` keyed `skippy_finetune`
— *not* added to ratchet's canonical registry (they're not cross-surface; PAI
doesn't carry them). This mirrors the existing 5090-`measured_llm` attach
pattern and keeps ratchet canonical.

**D2 — keyhole's quant-keyed `measured_llm` dicts.** keyhole's `measured_llm` is
shaped `{model_key: {quant: cell}}` (e.g. `{"skippy_finetune": {"Q4_K_M": {...}}}`),
whereas ratchet's is `{model_key: {workload_id: cell}}`. Under Option C keyhole
keeps its own `project_llm` (which reads `measured_llm[model][quant]`), so the
field is a generic dict on ratchet's `Hardware` and keyhole's access still works;
ratchet's `get_measured_llm_cell` simply goes unused by keyhole. **Recommend:
keep keyhole's `measured_llm` shape as-is** (re-attached to ratchet instances),
note the divergence, and reconcile to ratchet's workload-keyed shape only if/when
projection consolidates. *Open question:* the flat-field anchors and the
`measured_llm` dicts duplicate the same numbers — after D1 migrates the flat
fields to `measured_decode_overrides`, do we **also** keep the duplicate
`measured_llm` entries, or drop them and let `project_llm` resolve from
`measured_decode_overrides`? (Recommend dropping the duplicates and standardizing
on `measured_decode_overrides` for tier-level MoE anchors; keep `measured_llm`
only for genuine per-(model, workload) bundle cells on 5090-ref.)

**D3 — TDP-only spec diffs.** keyhole LP5-64 10 W vs ratchet 20 W; keyhole
i.MX 95 10 W vs ratchet 8 W. TDP is informational (not projected). Under Option C
keyhole inherits ratchet's values (display diff vs v1.0.0, like PAI's TDP diffs).
**But** i.MX 95 is a tier *only keyhole* carries (it's keyhole's ground-truth
measurement), and keyhole says 10 W while ratchet (from the design doc) says 8 W
— by the Amendment-4 rule ("the surface wins on invariant facts; keyhole is the
sole authority here"), ratchet's i.MX 95 TDP arguably should be **corrected to
10 W** (a tiny ratchet v0.2.4). LP5-64's 20 W was the reviewer's deliberate
Amendment-4 choice (both surfaces had 10 W; reviewer imposed 10/15/20). **Decision
needed:** (a) correct ratchet i.MX 95 → 10 W (+ keep LP5-64 = 20 W as imposed), or
(b) accept both as display diffs and move on (TDP doesn't matter). I lean (a) for
i.MX 95 (consistency with the canonical rule), reviewer's call on whether it's
worth a v0.2.4.

**D4 — Mid anchor key reconciliation.** ratchet's `NPU_MID` already has
`measured_decode_overrides={"qwen3_30b_a3b_moe": 37.85}`. keyhole keys its MoE as
`skippy_finetune`. If keyhole re-attaches `skippy_finetune: 37.85` to Mid, the
dict carries both keys (harmless); keyhole's `project_llm` resolves
`skippy_finetune`. **Recommend:** keyhole attaches its own `skippy_finetune`-keyed
entries (don't rely on ratchet's canonical key); no key adapter needed beyond
`measurement_alias`. (Simpler than PAI's `_canonical_anchor_keys`, since keyhole
overwrites/augments with its own keys rather than reading ratchet's.)

---

## 6. Ready-for-execute checklist (pending sign-off)

1. Pin `ratchet>=0.2.3,<0.3.0`; `pip install -e ../ratchet` in keyhole's env.
2. Loader swap (§1b) + `git rm sizer/npu_anchors.py`.
3. `npu_model.py`: delete `Hardware`, tier instances, `TIERS` literal,
   `hw_with_memory`, `MEMORY_UPGRADE_OPTIONS`, inline capability tables → import
   from ratchet; compose keyhole's ladder `TIERS` from ratchet (incl.
   `IMX95_MEASURED`).
4. Migrate tier-level MoE anchors per §3 + D1 (re-attach to ratchet instances in
   `measured.py` as `measured_decode_overrides`/`measured_prefill_overrides`);
   delete the flat-field projection path; resolve D2's duplicate question.
5. `precision.py`: inline capability tables → ratchet's canonical tables (keep
   `CAPABILITY_LABELS`/`CAPABILITY_DESCRIPTIONS` UI strings).
6. Adapters: `_get_measured` (+ alias); confirm `describe_hw`/projection read
   only fields ratchet's `Hardware` provides.
7. Amendment 5 patch in `_maybe_anchor_overlay_llm` (§4).
8. Apply D3 (TDP) + D4 (key) per sign-off.
9. Vision path untouched — verify `project_vision`/pipelines still import + run.
10. Parity report (incl. the MoE anchor cells across tiers + memory upgrades,
    and the Amendment-5 fix), magnitude sanity-checks, intended-diff list.
    Visual smoke (password gate?) before tag.
11. Tag `v1.1.0`, push (after sign-off + smoke).

---

## 7. What I will NOT do without sign-off
- No `git rm`, no `npu_model.py` deletions, no anchor migration until approved.
- If anything diverges from this plan mid-execution, stop and surface before
  destructive edits (per the phase-2 lesson — including scope *reductions*).

**Decisions needed:** D1 (re-attach approach), D2 (measured_llm shape + drop
duplicates?), D3 (correct ratchet i.MX 95 TDP → 10 W as v0.2.4, or accept), D4
(key reconciliation). Once you rule, keyhole executes through to v1.1.0
tagged+pushed unless something new surfaces.
