"""LLM model catalog for the sizer.

Three selectable models for the LLM workload comparison surface:
- Skippy MoE domain fine-tune (default, current production)
- Skippy dense domain fine-tune (prior production, near-parity at the
  pass-rate level — see "base-identity caveat" below)
- Qwen3-30B-A3B-Thinking-2507 (stock public reasoning baseline)

Two of these are Q4_K_M GGUF MoE 30B/3B-active (Skippy MoE FT,
Thinking stock) — same decode-tok/s ceiling on every Hardware tier
since BW-bound physics doesn't care which weights are in the file.
The third (Skippy dense FT, Qwen2.5-14B) has a DIFFERENT architecture:
14B dense traverses the full weight set per token, so it's roughly
3-4× slower than MoE 3B-active at the same BW. The sizer's perf path
is currently hardcoded to MoE numbers (`measured_llm_q4_decode_tok_s`
on Hardware tiers was anchored to Qwen3-30B-A3B), so when the dense
fine-tune is selected, the headline tok/s shown in the metric tile is
optimistic. The model expander surfaces this caveat in-line; future
work could split `Hardware.measured_llm_*` by model architecture.

## Base-identity caveat (per [docs] 2026-05-05 09:45)

Skippy MoE FT was trained on **Qwen3-30B-A3B-Instruct-2507** base
(verified from commit 704a2fb + on-disk adapter_config.json), NOT the
Thinking-2507 base. Thinking stock here is the Thinking-2507 sister
model — a DIFFERENT base than Skippy MoE FT. So the apparent +5.3pp
"domain fine-tune win" (0.689 - 0.636) is base-confounded by the
+7.6pp Instruct-vs-Thinking sister-model gap. Apples-to-apples (Skippy
MoE FT vs Instruct-2507 stock at 0.712) is **−2.3pp** — the MoE
fine-tune slightly regressed vs its own base.

The apples-to-apples-validated fine-tunes live in [docs]'s recipe-
taxonomy: Qwen2.5-7B v4 (+3.1pp vs 7B Instruct base, pass_rate 0.705)
and Qwen2.5-14B v4 (+5.3pp vs 14B Instruct base, pass_rate 0.727).
NEITHER of those is the SKIPPY_DENSE_FINETUNE row in this catalog —
that row (pass_rate 0.682) is an older pre-v4 dense LoRA, kept for
the dense-vs-MoE cost story but NOT the carrier of the +5.3pp claim.
The MoE recipe (attention-only LoRA on Qwen3-30B-A3B) is the failure
mode — pending an MoE-aware LoRA target test (router + experts) to
confirm whether MoE-base fine-tuning is recoverable. Production
reverted to Skippy 7B v4 dense in the meantime.

## The two stories the catalog answers

1. Model-architecture trade: "would a stock public reasoning model
   replace the fine-tune?" — comparison is base-confounded for the
   MoE row; deck_bullet now frames the dense gains as the validated
   case and MoE as pending-validation.
2. Cost trade: "would the older dense fine-tune work just as well?" →
   the two FTs land at similar pass-rates (Δ +0.7pp), but they got
   there via different paths: dense recovered capability, MoE
   regressed-but-arrived-near-parity. Cost story still holds: 4-5×
   fewer weights to read per token via MoE 3B-active routing.

## Convention: category_deltas

All `category_deltas` are **vs the production reference**
(`PRODUCTION_REFERENCE_KEY` = Skippy MoE FT). Sign convention:
positive = this model wins vs production. The production reference
itself has an empty `category_deltas` dict (it can't differ from
itself). UI surfaces these as "Per-category Δ vs Skippy MoE
fine-tune (production)" in the accuracy expander. Note that
THINKING_MOE_STOCK's category_deltas mix BASE-GAP (Instruct vs
Thinking baselines) with RECIPE EFFECT (LoRA fine-tune); they are
NOT clean fine-tune-recipe deltas.

Accuracy from Skippy v2 prompt set: 44 prompts × 3 samples = 132,
RAG enabled (8 chunks via hybrid retrieval). Diffs:
- `eval/results/acc_diff_skippy_fine_tune_vs_thinking.md`
- `eval/results/acc_diff_dense_q4km_vs_moe_q4km_v2_rag.md`
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModel:
    """One selectable LLM with optional accuracy stats + 5090 perf anchors."""
    key: str
    label: str
    family: str
    base: str
    total_params_b: float
    active_params_b: float
    quant: str
    size_gb: float
    description: str
    deck_bullet: str          # the one-line framing for the deck/UI

    # Accuracy fields are optional — Skippy v2 prompt-set evaluation only
    # exists for production candidates. Performance-comparison entries
    # (e.g., Qwen 2.5 7B / 32B dense, added 2026-05-01 for cross-model
    # bandwidth-vs-compute calibration) leave these None.
    fine_tune: str = ""
    pass_rate: float | None = None
    pass_n_passes: int | None = None
    pass_n_total: int | None = None
    # Per-category Δ vs the production reference (sign convention:
    # positive = THIS wins). Keyed by Skippy v2 prompt category.
    # Empty dict for production reference itself; None for entries
    # without v2 evaluation.
    category_deltas: dict[str, int] | None = None

    # Compute dtype the model executes on dedicated NPU silicon. Q4_K_M
    # weight-only MoE models run as INT8 dequant + INT8 matmul on
    # purpose-built INT8 NPUs (Skippy MoE Q4 / Thinking-2507). Dense Q4
    # models like Qwen 2.5 14B / 7B / 32B Q4 use llama-cpp's fp16
    # internal path — weights are dequantized to fp16 for the matmul,
    # requiring native FP16 tensor support. Per [pai-sizer] e69237b +
    # [backend] 15:46 parity ask. Used by app.py to gate dtype-mismatch
    # projections against `Hardware.capability_levels` and surface a 🔴
    # 'dtype_mismatch' banner instead of silently projecting fp16
    # numbers on INT8-only silicon.
    compute_dtype: str = "int8"

    # Per-token compute cost (GFLOPs/token) for the prefill compute floor.
    # First-order: 2 × active_params_b for matmul-bound forward (GPT-style
    # transformer FLOP estimate). Per [backend] 2026-04-29 12:38 + 13:17
    # design doc. 0.0 = unknown / fall back to 2 × active_params_b at
    # use site.
    gops_per_token: float = 0.0

    @property
    def decode_bw_per_token_gb(self) -> float:
        """Approximate DRAM bytes read per decode token, in GB.

        First-order projection used to scale Hardware tier perf
        anchors across model architectures:
        - MoE: size_gb × (active/total) — only the routed experts
          are read per token, so BW scales with active params.
        - Dense: size_gb — every weight is read every token.

        Ignores shared layers (embedding lookup, attention QKV
        projections common across experts) and KV-cache traffic
        (small at our 1K-prompt anchor). Refining either of those
        would tighten the projection but doesn't change the order
        of magnitude for the dense-vs-MoE comparison the sizer
        cares about.
        """
        if self.total_params_b <= 0:
            return self.size_gb
        active_fraction = self.active_params_b / self.total_params_b
        return self.size_gb * active_fraction


SKIPPY_MOE_FINETUNE = LLMModel(
    key="skippy_finetune",
    label="Skippy MoE fine-tune (DEFAULT)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B-Instruct-2507 (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="attention-only QLoRA r=64 — NXP datasheets, transcripts, Skippy prod corpus",
    pass_rate=0.689,
    pass_n_passes=91,
    pass_n_total=132,
    description=(
        "Kyle's domain fine-tune of Qwen3-30B-A3B-Instruct-2507 — the model "
        "in Skippy production at the time of the v2-RAG eval. Attention-only "
        "QLoRA r=64 adapted on internal corpora (NXP datasheets, meeting "
        "transcripts, source code). Default selection and the **production "
        "reference** for per-category Δ comparisons across the catalog. NOTE: "
        "production has since reverted to Skippy 7B v4 dense per [docs] "
        "2026-05-05; MoE-base fine-tune validation pending an MoE-aware LoRA "
        "target test (router + experts)."
    ),
    deck_bullet=(
        "Fine-tuning gains are validated on dense Qwen2.5 (7B v4 +3.1pp, "
        "14B v4 +5.3pp vs respective Instruct bases — apples-to-apples). "
        "MoE-base validation is pending — the current attention-only LoRA "
        "recipe does not transfer capability to Qwen3-MoE. MoE-aware LoRA "
        "target test (router + experts) on RunPod is the next milestone."
    ),
    # Production reference: vs itself = no Δs. Other entries' deltas
    # are computed against this row.
    category_deltas={},
    # MoE Q4 weight-only on dedicated INT8 NPU runs as INT8 dequant +
    # INT8 matmul (the Mid silicon's native path). Validates against
    # Mid's INT8-only capability_levels post 548bc41.
    compute_dtype="int8",
    # 5090 anchor lives on RTX_5090_REFERENCE.measured_llm (per 66edfa2
    # cross-app schema parity with PAI sizer e69237b); single source of
    # truth, consumed by project_llm.
    gops_per_token=6.0,  # 2 × 3B active (matmul-bound MoE forward)
)


SKIPPY_DENSE_FINETUNE = LLMModel(
    key="skippy_dense_finetune",
    label="Skippy dense fine-tune (Qwen2.5-14B)",
    family="Dense — 14B params (no expert sparsity)",
    base="Qwen2.5-14B-Instruct (Alibaba)",
    total_params_b=14.0,
    active_params_b=14.0,
    quant="Q4_K_M GGUF",
    size_gb=9.2,
    fine_tune="domain LoRA — same NXP/Skippy corpus as the MoE entry",
    pass_rate=0.682,
    pass_n_passes=90,
    pass_n_total=132,
    description=(
        "Kyle's prior-production dense fine-tune (Qwen2.5-14B-Instruct base, "
        "older domain-LoRA recipe — pre-v4). Pass rate 0.682 here is from "
        "this earlier variant; it is NOT the same model as the apples-to-"
        "apples-validated 14B v4 (0.727, +5.3pp vs 14B Instruct stock — "
        "per [docs] 2026-05-05). Kept in catalog for the dense-vs-MoE "
        "cost story; the validated +5.3pp gain anchor lives in [docs]'s "
        "recipe-taxonomy, not on this row."
    ),
    deck_bullet=(
        "Older dense fine-tune entry (pre-v4, 0.682). Lands near the MoE "
        "FT row at the pass-rate level (Δ +0.7pp), but the apples-to-apples "
        "validated dense fine-tune is **14B v4 (0.727, +5.3pp)** — a "
        "different model not currently in this catalog. Cost story still "
        "holds for either dense entry: MoE 3B-active reads 4-5× fewer "
        "weights per token than any 14B dense forward."
    ),
    # Δ vs Skippy MoE FT (production reference). MoE wins rag_datasheet
    # +3 → dense -3 here; MoE loses refusal -2 → dense +2 here.
    category_deltas={
        "rag_datasheet": -3,
        "refusal":       +2,
    },
    # Dense Qwen 2.5 14B Q4 uses llama-cpp's fp16 internal path —
    # weights dequantized to fp16 for the matmul, requires native FP16
    # tensor support. Per [pai-sizer] e69237b: "Dense 14B Q4 stays fp16."
    # Selecting this model on Mid (INT8-only post 548bc41) triggers
    # 🔴 dtype_mismatch in the app.py UI gate.
    compute_dtype="fp16",
    gops_per_token=28.0,  # 2 × 14B (full dense forward)
)


INSTRUCT_MOE_STOCK = LLMModel(
    key="instruct_moe_stock",
    label="Qwen3-30B-A3B-Instruct-2507 (stock)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B-Instruct-2507 (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="stock (no fine-tune)",
    pass_rate=0.712,
    pass_n_passes=94,
    pass_n_total=132,
    description=(
        "Qwen3-30B-A3B-Instruct-2507 stock — the **true base** of Skippy MoE FT "
        "and therefore the apples-to-apples reference for the MoE fine-tune row. "
        "Eval source: `eval/results/acc_baseline-qwen3-30b-a3b-instruct-2507-"
        "v2-rag_20260504-170335.json` (132-sample v2-RAG, 5090 host, Q4_K_M, "
        "deterministic temp=0, RAG on, agent loop off). Same eval harness as "
        "the other catalog rows. Per [docs] 2026-05-05."
    ),
    deck_bullet=(
        "True base of Skippy MoE FT. Apples-to-apples reference: SKIPPY_MOE_"
        "FINETUNE (0.689) vs this (0.712) = **−2.3pp** — fine-tune slightly "
        "regressed vs its own base. The +5.3pp 'win' vs Thinking sibling was "
        "sister-model gap, not recipe gain."
    ),
    # Per-category delta breakdown vs Skippy MoE FT pending — overall is
    # +2.3pp; per-category not yet derived from the eval JSON. Empty dict
    # for now keeps the UI rendering clean (no spurious per-category rows).
    category_deltas={},
    # Same Qwen3-30B-A3B Q4 MoE architecture as Skippy MoE FT — same
    # INT8 dequant + INT8 matmul path on dedicated INT8 NPU silicon.
    compute_dtype="int8",
    gops_per_token=6.0,  # 2 × 3B active — same architecture as Skippy MoE
)


THINKING_MOE_STOCK = LLMModel(
    key="thinking_stock",
    label="Qwen3-30B-A3B-Thinking-2507 (stock public)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B-Thinking-2507 (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="public reasoning-tune (no domain corpus)",
    pass_rate=0.636,
    pass_n_passes=84,
    pass_n_total=132,
    description=(
        "Public Qwen3-30B-A3B-Thinking-2507 — Alibaba's reasoning-tuned "
        "sister model in the Qwen3-30B-A3B family. CAVEAT: this is the "
        "Thinking-2507 variant; Skippy MoE FT is on the Instruct-2507 "
        "variant. The two share an architecture but NOT a base — the "
        "Instruct-vs-Thinking baseline gap on v2-RAG is +7.6pp (per [docs] "
        "2026-05-05). Apples-to-apples comparisons against Skippy MoE FT "
        "should account for this base-sister gap."
    ),
    deck_bullet=(
        "Public reasoning-tuned sister to Skippy MoE FT's base. NOT a "
        "clean fine-tune-vs-stock comparison — the +5.3pp pass-rate "
        "delta seen in the catalog mixes a +7.6pp base-sister gap "
        "(Instruct vs Thinking) with the recipe effect. Use this row "
        "for 'how does public reasoning rank' framing, not for fine-tune "
        "validation."
    ),
    # CAVEAT: these deltas are vs Skippy MoE FT (Instruct-2507 base),
    # but THIS row is Thinking-2507 base — so the deltas mix BASE-GAP
    # (Instruct vs Thinking sister-models, +7.6pp Instruct advantage)
    # with RECIPE EFFECT (LoRA fine-tune). They are NOT clean
    # fine-tune-recipe deltas. Per [docs] 2026-05-05, the apples-to-apples
    # MoE fine-tune delta vs Instruct stock is −2.3pp (regressed). The
    # apparent "FT wins rag_datasheet" pattern below is plausibly the
    # Instruct-base advantage, not the LoRA's contribution.
    category_deltas={
        "rag_datasheet": -8,
        "rag_email":     -3,
        "numerical_precision": +3,
        "refusal":       +2,
    },
    # Same Qwen3-30B-A3B Q4 MoE architecture as Skippy MoE FT — runs the
    # same INT8 dequant + INT8 matmul path on dedicated INT8 NPU silicon.
    compute_dtype="int8",
    gops_per_token=6.0,  # 2 × 3B active — same architecture as Skippy MoE
)


# ─────────── Performance-comparison dense models (added 2026-05-01) ────────
# Per [backend] 20:08 weekend bake-off campaign. These are NOT Skippy
# domain fine-tunes — no Skippy v2 prompt-set evaluation. Sole purpose
# is to anchor the dense-vs-MoE bandwidth-vs-compute comparison on the
# 5090 across multiple quants. Surface in the dropdown as comparison
# entries; production deployment would use Skippy MoE / dense FT.
#
# Headline narrative this enables:
# - Skippy MoE 30B-A3B Q4: ~250 tok/s on 5090 (3B active streams 1.65 GB/tok)
# - Qwen 2.5 32B dense Q4:  52.7 tok/s on 5090 (full 32B streams 17.88 GB/tok)
# - 4.7× MoE-vs-dense BW advantage at the 30B-class size — exactly because
#   MoE only reads active experts per token.

QWEN25_7B_DENSE_INSTRUCT = LLMModel(
    key="qwen25_7b_dense",
    label="Qwen 2.5 7B Instruct (dense — perf reference)",
    family="Dense — 7B params (no expert sparsity)",
    base="Qwen2.5-7B-Instruct (Alibaba)",
    total_params_b=7.6,
    active_params_b=7.6,
    quant="Q4_K_M GGUF (also Q5_K_M, Q8_0 measured)",
    size_gb=4.18,  # Q4 footprint; sizer's per-quant scaling applies via decode_bw_per_token_gb
    description=(
        "Stock Qwen 2.5 7B dense — added as a perf-comparison anchor for the "
        "dense-vs-MoE narrative. NOT a Skippy domain fine-tune; no v2 prompt-set "
        "evaluation. Decode tok/s on 5090 measured across Q4/Q5/Q8 quants for "
        "cross-quant BW realization calibration."
    ),
    deck_bullet=(
        "7B dense Q4 = 184 tok/s on 5090 vs Skippy MoE 30B-A3B Q4 = 250 tok/s — "
        "MoE wins on bandwidth despite 4× more total params, because only 3B "
        "active streams per token."
    ),
    # Dense Q4 → fp16 internal path on llama-cpp (gates 🔴 dtype_mismatch
    # on Mid INT8-only via the f851d24 app.py UI gate; valid on High +
    # 5090 which retain FP support).
    compute_dtype="fp16",
    # 5090 anchors (Q4/Q5/Q8 decode + prefill) live on
    # RTX_5090_REFERENCE.measured_llm — single source of truth.
    gops_per_token=15.2,  # 2 × 7.6B
)


QWEN25_32B_DENSE_INSTRUCT = LLMModel(
    key="qwen25_32b_dense",
    label="Qwen 2.5 32B Instruct (dense — perf reference)",
    family="Dense — 32B params (no expert sparsity)",
    base="Qwen2.5-32B-Instruct (Alibaba)",
    total_params_b=32.5,
    active_params_b=32.5,
    quant="Q4_K_M GGUF (also Q5_K_M measured; Q8 won't fit on 5090)",
    size_gb=17.88,  # Q4 footprint
    description=(
        "Stock Qwen 2.5 32B dense — perf-comparison anchor at the 30B-class "
        "size to slope-test MoE vs dense at the SAME total-param magnitude. "
        "Same architecture family as Skippy MoE (Qwen3-30B-A3B = 30B total) "
        "but everything streams every token. NOT evaluated on Skippy v2 "
        "prompt set."
    ),
    deck_bullet=(
        "Qwen 2.5 32B dense Q4 = 52.7 tok/s on 5090. Same total-param "
        "magnitude as Skippy MoE 30B-A3B (250 tok/s), but full 17.88 GB/tok "
        "vs MoE's 1.65 GB/tok = 4.7× BW advantage for sparse MoE."
    ),
    compute_dtype="fp16",
    # 5090 anchors (Q4/Q5 decode + prefill) live on
    # RTX_5090_REFERENCE.measured_llm — single source of truth. No Q8
    # entry — won't fit on 5090's 32 GB VRAM.
    gops_per_token=65.0,  # 2 × 32.5B
)


LLM_MODELS: dict[str, LLMModel] = {
    # Order matters — drives selectbox display order. Convention:
    # production first (default), then prior-production (cost-different
    # but quality-parity), then external Skippy-eval'd baselines, then
    # perf-comparison-only references (no Skippy v2 evaluation).
    SKIPPY_MOE_FINETUNE.key:        SKIPPY_MOE_FINETUNE,
    SKIPPY_DENSE_FINETUNE.key:      SKIPPY_DENSE_FINETUNE,
    INSTRUCT_MOE_STOCK.key:         INSTRUCT_MOE_STOCK,
    THINKING_MOE_STOCK.key:         THINKING_MOE_STOCK,
    QWEN25_7B_DENSE_INSTRUCT.key:   QWEN25_7B_DENSE_INSTRUCT,
    QWEN25_32B_DENSE_INSTRUCT.key:  QWEN25_32B_DENSE_INSTRUCT,
}

DEFAULT_LLM_MODEL_KEY = SKIPPY_MOE_FINETUNE.key

# Reference for all per-category-Δ rendering. UI labels comparisons as
# "vs Skippy MoE fine-tune (production)" and the production model
# itself shows no per-category breakdown (it would be 0 across the board).
PRODUCTION_REFERENCE_KEY = SKIPPY_MOE_FINETUNE.key


def accuracy_delta_pp(a: LLMModel, b: LLMModel) -> float | None:
    """Return (a.pass_rate - b.pass_rate) in percentage points.

    Returns None when either side lacks a pass_rate (perf-comparison-only
    entries that weren't evaluated on the Skippy v2 prompt set).
    """
    if a.pass_rate is None or b.pass_rate is None:
        return None
    return (a.pass_rate - b.pass_rate) * 100.0


# Reference for the perf-axis scaling. Hardware tiers' measured_llm_*
# values were anchored to this model's architecture (Qwen3-30B-A3B
# Q4_K_M, MoE 3B-active). Other models scale relative to it.
PERF_REFERENCE_MODEL_KEY = SKIPPY_MOE_FINETUNE.key


def perf_scale_factor(model: LLMModel) -> float:
    """Multiplicative factor to scale tier-anchored decode tok/s for `model`.

    factor = 1.0 for the perf-reference model itself (and any model with
    the same architecture, e.g. Thinking-stock = same Qwen3-30B-A3B base).
    factor < 1.0 means this model is slower (more BW per token). For
    dense 14B vs MoE 3B-active reference: factor ≈ 0.20.
    """
    reference = LLM_MODELS[PERF_REFERENCE_MODEL_KEY]
    return reference.decode_bw_per_token_gb / model.decode_bw_per_token_gb


def scale_llm_projection(
    llm_proj: dict,
    model: LLMModel,
    hw_mem_capacity_gb: float | None = None,
) -> dict:
    """Apply per-model BW scaling to a `project_llm()` result dict.

    Returns a new dict — the input is not mutated. tok/s scales by
    `perf_scale_factor(model)`; per-token / per-answer time fields
    scale by 1/factor (slower model = longer time-per-X). Also
    overrides `gguf_size_gb` to the selected model's actual file
    size and recomputes `fits_in_memory` if `hw_mem_capacity_gb`
    is supplied (the reference model's size baked into project_llm
    won't be accurate for the dense entry).

    **Guard added 2026-05-01:** when project_llm fired the per-cell
    anchor path (`llm_source == 'measured'` — e.g. RTX 5090 + Qwen 2.5
    7B Q4 reads 183.9 directly from `hw.measured_llm[model_key][quant]`),
    the returned values are already model-specific. Re-scaling would
    double-apply the per-model BW ratio and corrupt the measurement.
    Skip the scaling step in that case but still set gguf_size /
    fits_in_memory from the model dict.
    """
    scaled = dict(llm_proj)
    scaled["gguf_size_gb"] = model.size_gb
    if hw_mem_capacity_gb is not None:
        scaled["fits_in_memory"] = model.size_gb <= hw_mem_capacity_gb

    # Per-cell measured anchor fired → values are already model-specific.
    # Skip the per-model BW scaling.
    if scaled.get("llm_source") == "measured":
        return scaled

    factor = perf_scale_factor(model)
    if factor == 1.0:
        return scaled

    if "decode_tok_s" in scaled:
        scaled["decode_tok_s"] = scaled["decode_tok_s"] * factor
    for time_key in ("ttft_1k_sec", "short_answer_sec",
                     "rag_total_sec", "rag_prefill_sec", "rag_decode_sec"):
        if time_key in scaled:
            scaled[time_key] = scaled[time_key] / factor
    return scaled


# Human-readable category labels for the per-category breakdown UI.
CATEGORY_LABELS: dict[str, str] = {
    "rag_datasheet":       "RAG · datasheet retrieval",
    "rag_email":           "RAG · email retrieval",
    "numerical_precision": "Numerical reasoning",
    "refusal":             "Refusal / scope control",
    "coding":              "Coding",
    "reasoning":           "General reasoning",
    "multihop":            "Multi-hop",
    "general":             "General Q&A",
    "persona":             "Persona / style",
    "rag_blog":            "RAG · blog retrieval",
}
