"""LLM model catalog for the sizer.

Selectable models for the LLM workload comparison surface (14 entries
post 2026-05-08 Mistral v4 FT integration):

Production + validated FTs:
- Skippy 7B v4 (Qwen2.5-7B FT — DEFAULT, current production, +3.1pp clean)
- Skippy 14B v4 (Qwen2.5-14B FT — best dense headline, +5.3pp BUT fabricates)
- Skippy 32B v4 (Qwen2.5-32B FT — recipe trade −4.6pp, NOT recommended)
- Skippy Mistral v4 (Mistral-7B FT — cross-family regression −3.8pp, NOT recommended)
- Skippy MoE-router v1 (Qwen3-30B-A3B FT — recommended MoE recipe, −3.8pp)

Historical + cautionary FTs:
- Skippy MoE FT v1 (original MoE attention-only LoRA — pre-router-v1)
- Skippy MoE-full v1 (over-fit cautionary; expert-FFN LoRA broke rag_blog)
- Skippy dense fine-tune (older Qwen2.5-14B pre-v4 dense LoRA)

Apples-to-apples baselines:
- Qwen3-30B-A3B-Instruct-2507 (true base for MoE FT — −2.3pp vs base)
- Qwen3-30B-A3B-Thinking-2507 (sister-model context, NOT the MoE FT base)
- Qwen 2.5 7B Instruct (apples-to-apples 7B v4 baseline)
- Qwen 2.5 32B Instruct (apples-to-apples 32B v4 baseline)

Cross-family baselines (Tier 3 #1):
- Meta Llama-3.1 8B Instruct (cross-family — −10.6pp vs Qwen 7B at same size)
- Mistral 7B v0.3 Instruct (cross-family — −6.8pp vs Qwen 7B; reasoning 0/6)

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

## Convention: category_deltas (dict-of-dicts, SEMANTIC GRADE post-2026-05-11)

Each entry's `category_deltas` holds **raw per-category rates**:

    {category: {"pass": int, "n": int, "rate": float}}

The 9 v2-RAG categories are: coding, general, multihop,
numerical_precision, rag_blog, rag_datasheet, rag_email, reasoning,
refusal. Per-category sums reconcile to `pass_n_passes`.

Δ vs the production reference (`PRODUCTION_REFERENCE_KEY` =
Skippy 7B v4) is computed at render time from raw counts.

## ⚠ Grade type — SEMANTIC headline post-2026-05-11

All entries carry **GPT-4o semantic-graded** pass_rates as of
[docs] 2026-05-11 white paper Finding 4 + reviewer closure. Two
entries (`skippy_finetune` MoE FT v1, `skippy_dense_finetune` pre-v4
dense) lack a `_semantic.json` regrade payload and carry SUBSTRING
values with explicit caveats in their category_deltas comment block.

The headline shift: production Skippy 7B v4 was substring 0.705 →
semantic 0.606. The original substring +3.1pp lift vs the Qwen 2.5
7B Instruct base reverses to a semantic −4.6pp regression. Per
[docs]'s reviewer (verbatim, 2026-05-11 09:31):

  "The Qwen-family format bias finding is the single most valuable
  methodology output of this entire campaign — more valuable than
  gotcha #7, more valuable than the two-factor model. The recipe's
  value is voice transfer and safety calibration, not capability
  lift; the substring-headline-capability gain on this corpus was
  a format-fidelity artifact specific to Qwen-shaped phrasings."

Production decision UNAFFECTED — the three-gate framework
(capability + voice + safety) was designed exactly for this.
Substring failed silently; voice + safety carried the real signal.

Migration history:
  - 2026-05-07 (3635622): substring dict-of-dicts schema landed
    (was signed-int deltas). Cross-app parity with PAI 8d20beb.
  - 2026-05-11 (d7f082c): semantic-graded headline + per-cat
    swap. Mirrors PAI e416ee0. Substring values archived in
    PAI sizer's npu_model.py / `_semantic.json` files on Drive.

`METHODOLOGY_VERSION` (module constant below) labels the eval-
methodology cycle these numbers come from. Bump when something
changes that affects how a customer should read the numbers:
new substring/judge/semantic framing, new RAG protocol, new
grader, new eval set, etc. Cross-app lockstep with PAI sizer's
`sizer_bundle.json __meta__.methodology_version` and Skippy
side's `eval/build_sizer_bundle.py` output (per [pai-sizer]
2026-05-11 12:13 + [docs] 12:15 + [pai-sizer] 12:21 / 12:22).

Accuracy from Skippy v2 prompt set: 44 prompts × 3 samples = 132,
RAG enabled (8 chunks via hybrid retrieval). Diffs:
- `eval/results/acc_diff_skippy_fine_tune_vs_thinking.md`
- `eval/results/acc_diff_dense_q4km_vs_moe_q4km_v2_rag.md`
"""
from __future__ import annotations

from dataclasses import dataclass

# Methodology-version label for the LLM-catalog eval cycle these
# pass_rate + per-category numbers come from. Cross-app lockstep:
#   - PAI sizer    sizer_bundle.json __meta__.methodology_version
#   - Skippy side  eval/build_sizer_bundle.py
#   - keyhole-sizer  this constant
# Bump when something affects how a customer should READ the numbers:
# new substring/judge/semantic framing, new RAG protocol, new grader,
# new eval set composition, etc. (Distinct from `bundle_version` in
# sizer_bundle.json which versions the perf/DRAM measurement schema.)
METHODOLOGY_VERSION = "2026-05-11-semantic-regrade-shipped"


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
    # Per-category raw rates (dict-of-dicts shape per [pai-sizer]
    # 2026-05-07 8d20beb migration). Each category maps to:
    #   {"pass": int, "n": int, "rate": float}
    # Δ vs production reference is computed at render time from raw
    # counts (pass_diff = this.pass - production.pass for matching
    # category). Empty dict for entries without per-category data;
    # None for entries without v2 evaluation. Migration from the old
    # signed-int delta shape landed 2026-05-07; the data is canonical
    # now (pass/n totals reconcile to pass_rate × pass_n_total).
    category_deltas: dict[str, dict[str, int | float]] | None = None

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

    # Optional perf-anchor alias — when set, use this OTHER model's key
    # to look up `Hardware.measured_llm[...][quant]` for performance
    # projections. Used when a model shares the exact perf characteristics
    # of another catalog entry (same architecture, same quant) so existing
    # 5090 measurements transfer verbatim. Examples:
    #   - SKIPPY_7B_V4 (Qwen2.5-7B fine-tune) ↔ qwen25_7b_dense (Qwen2.5-7B
    #     stock): same arch, same Q4 path; 5090 measurement is the same.
    #   - INSTRUCT_MOE_STOCK (no fine-tune) ↔ skippy_finetune (LoRA-adapted):
    #     same Qwen3-30B-A3B MoE arch; perf identical, accuracy differs.
    # Mirrors PAI sizer's measurement_alias (commit 3cb533a). Resolved in
    # app.py before calling project_llm, so npu_model.py stays decoupled
    # from the LLM catalog.
    measurement_alias: str | None = None

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
    # ⚠ SUBSTRING-GRADED — no _semantic.json available for this row in
    # [docs] 2026-05-11 bulk regrade. Per the Qwen-family-bias family
    # pattern (per semantic_regrade_catalog.md), semantic would likely
    # land ~−3.2pp lower (~65.7% on 132-basis), preserving the apples-
    # to-apples 'regressed vs Instruct-2507 base' direction. Sum
    # reconciles to pass_n_passes=91.
    # Source: eval/results/acc_reference-moe-q4km-v2-rag_20260423-091231.json
    # per [docs] 2026-05-07 17:26.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  3, "n":  3, "rate": 1.000},
        "multihop":            {"pass":  6, "n":  9, "rate": 0.667},
        "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
        "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
        "rag_datasheet":       {"pass": 54, "n": 78, "rate": 0.692},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
        "refusal":             {"pass":  7, "n":  9, "rate": 0.778},
    },
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
    # ⚠ SUBSTRING-GRADED — no _semantic.json available for this row in
    # [docs] 2026-05-11 bulk regrade. Per the Qwen-family-bias family
    # pattern, semantic would likely land ~−3.2pp lower. Sum reconciles
    # to pass_n_passes=90.
    # Source: eval/results/acc_reference-dense-q4km-v2-rag_20260423-091847.json
    # per [docs] 2026-05-07 17:26. Schema migration from old signed-int
    # deltas (vs MoE FT v1) landed 2026-05-07.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  3, "n":  3, "rate": 1.000},
        "multihop":            {"pass":  6, "n":  9, "rate": 0.667},
        "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
        "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
        "rag_datasheet":       {"pass": 51, "n": 78, "rate": 0.654},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
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
    pass_rate=0.659,
    pass_n_passes=87,
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
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Substring read 0.712; semantic 0.659 (−5.5pp). Sum reconciles to
    # pass_n_passes=87.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  3, "n":  6, "rate": 0.500},
        "multihop":            {"pass":  4, "n":  9, "rate": 0.444},
        "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
        "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
        "rag_datasheet":       {"pass": 50, "n": 78, "rate": 0.641},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
    # Same Qwen3-30B-A3B Q4 MoE architecture as Skippy MoE FT — same
    # INT8 dequant + INT8 matmul path on dedicated INT8 NPU silicon.
    compute_dtype="int8",
    gops_per_token=6.0,  # 2 × 3B active — same architecture as Skippy MoE
    # Same MoE arch as Skippy MoE FT — perf cells transfer verbatim.
    measurement_alias="skippy_finetune",
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
    pass_rate=0.561,
    pass_n_passes=74,
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
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Substring read 0.636; semantic 0.561 (−12.7pp — Thinking-2507 is
    # the largest single-cell drop in the catalog, reinforcing the
    # Qwen-family-format-bias finding). Sum reconciles to pass_n_passes=74.
    category_deltas={
        "coding":              {"pass":  5, "n":  6, "rate": 0.833},
        "general":             {"pass":  6, "n":  6, "rate": 1.000},
        "multihop":            {"pass":  1, "n":  9, "rate": 0.111},
        "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
        "rag_blog":            {"pass":  0, "n":  3, "rate": 0.000},
        "rag_datasheet":       {"pass": 46, "n": 78, "rate": 0.590},
        "rag_email":           {"pass":  0, "n":  3, "rate": 0.000},
        "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
        "refusal":             {"pass":  7, "n":  9, "rate": 0.778},
    },
    # Same Qwen3-30B-A3B Q4 MoE architecture as Skippy MoE FT — runs the
    # same INT8 dequant + INT8 matmul path on dedicated INT8 NPU silicon.
    compute_dtype="int8",
    gops_per_token=6.0,  # 2 × 3B active — same architecture as Skippy MoE
)


# ─────────── Validated dense fine-tunes (added 2026-05-06 per [docs]) ─────
# These are the apples-to-apples-validated fine-tune anchors per [docs]
# 2026-05-05 09:45 + 2026-05-06 09:19. Both trained with the v4 recipe
# (SFTTrainer + assistant_only_loss, 100 refusal exemplars, 2 epochs)
# on Qwen2.5 Instruct bases. Cross-app catalog convergence with PAI
# sizer (per [docs] 09:19): both sizers carry { 7B v4, 14B v4, MoE FT
# (v1), Instruct-2507 stock, Thinking-2507 stock, plus dense bases }.

SKIPPY_7B_V4 = LLMModel(
    key="skippy_7b_v4",
    label="Skippy 7B v4 (Qwen2.5-7B FT — current production)",
    family="Dense — 7B params (no expert sparsity)",
    base="Qwen2.5-7B-Instruct (Alibaba)",
    total_params_b=7.6,
    active_params_b=7.6,
    quant="Q4_K_M GGUF",
    size_gb=4.18,
    fine_tune="v4 SFT — assistant_only_loss, 100 refusal exemplars, 2 epochs",
    pass_rate=0.606,
    pass_n_passes=80,
    pass_n_total=132,
    description=(
        "**Current production model** per [docs] 2026-05-04 22:20. Ships via "
        "the **three-gate framework** (capability + voice + safety), NOT a "
        "capability headline. The original substring +3.1pp lift vs Qwen 2.5 "
        "7B Instruct base **eroded across five successive cross-checks** "
        "(LLM-judge, temperature sensitivity, cross-judge, semantic regrade) "
        "→ **semantic regrade reverses to −4.6pp** (0.606 vs base 0.652). "
        "Per [docs] 2026-05-11 white paper Finding 4: the recipe's value is "
        "**voice transfer + safety calibration**, not capability lift. The "
        "substring lift was format-fidelity matching trained Qwen phrasings. "
        "Production decision unaffected — voice ✓ (152 char vs base's 324), "
        "safety ✓ (refusal 9/9), capability passes three-gate floor. v4 "
        "recipe = SFTTrainer + assistant_only_loss, 100 refusal exemplars, "
        "2 epochs. Trained locally on 5090 in ~46 min, $0."
    ),
    deck_bullet=(
        "Skippy 7B v4 = production via three-gate framework, NOT capability "
        "headline. Substring +3.1pp lift eroded to semantic −4.6pp across "
        "five cross-checks — recipe value is voice + safety transfer, not "
        "capability gain. The three-gate framework caught substring failing "
        "silently. Reviewer-final framing: 'Qwen-family format bias is the "
        "single most valuable methodology output of this campaign.'"
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11 white
    # paper Finding 4 + reviewer closure. Substring values mirrored from
    # PAI sizer's commit e416ee0 (single source of truth). Sum reconciles
    # to pass_n_passes=80. Headline finding: substring +3.1pp lift reverses
    # to semantic −4.6pp regression — the original capability-lift claim
    # was format-fidelity to trained Qwen phrasings, not real gain.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  2, "n":  6, "rate": 0.333},
        "multihop":            {"pass":  3, "n":  9, "rate": 0.333},
        "numerical_precision": {"pass":  1, "n":  6, "rate": 0.167},
        "rag_blog":            {"pass":  0, "n":  3, "rate": 0.000},
        "rag_datasheet":       {"pass": 53, "n": 78, "rate": 0.679},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  3, "n":  6, "rate": 0.500},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
    # Dense Q4 → fp16 internal path on llama.cpp. Selecting on Mid (INT8-only)
    # triggers 🔴 dtype_mismatch in the app.py UI gate.
    compute_dtype="fp16",
    gops_per_token=15.2,  # 2 × 7.6B (full dense forward)
    # Same Qwen2.5-7B dense arch as qwen25_7b_dense — 5090 measurements
    # (Q4/Q5/Q8) transfer verbatim via this alias. Without it, project_llm
    # falls through to the flat-field MoE Q4 anchor (~250 tok/s) which is
    # wildly wrong for dense 7B (real ~184 tok/s).
    measurement_alias="qwen25_7b_dense",
)


SKIPPY_14B_V4 = LLMModel(
    key="skippy_14b_v4",
    label="Skippy 14B v4 (Qwen2.5-14B FT — best headline, NOT recommended for production)",
    family="Dense — 14B params (no expert sparsity)",
    base="Qwen2.5-14B-Instruct (Alibaba)",
    total_params_b=14.0,
    active_params_b=14.0,
    quant="Q4_K_M GGUF",
    size_gb=9.2,
    fine_tune="v4 QLoRA 4-bit — same recipe shape as 7B v4, larger base",
    pass_rate=0.697,
    pass_n_passes=92,
    pass_n_total=132,
    description=(
        "Apples-to-apples-validated dense fine-tune: +5.3pp vs Qwen 2.5 14B "
        "Instruct base — best headline of the v4 campaign. ⚠️ **NOT "
        "recommended for production** despite the headline. Customer-template "
        "verdict (per [docs] 2026-05-07 13:17 wrap-up): 'Validated with "
        "caveat — add adversarial refusal data before shipping.' Per-category "
        "asymmetry: 🟢 numerical_precision +3 (perfect 6/6), rag_datasheet "
        "substantial gain; 🔴 rag_email −3 (3/3 → 0/3), made_up_peripheral "
        "fabrication 0/3 — confidently invents 'QuantumFlow Engine' specs "
        "for fictional peripherals. Note that 32B Instruct STOCK has the "
        "same fabrication problem (3/3 wrong on those prompts), so this is "
        "partly a Qwen base-model behavior the v4 recipe inherits + amplifies. "
        "Trained on 5090 (QLoRA 4-bit) in ~46 min."
    ),
    deck_bullet=(
        "Skippy 14B v4 = +5.3pp headline win on its Instruct base — the "
        "biggest validated fine-tune gain in the v4 campaign. But "
        "fabrication on made-up peripherals (0/3) and rag_email regression "
        "(0/3) make it unsuitable for production despite the headline. "
        "Cautionary entry — bigger isn't strictly better on this recipe."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11. One of
    # only two cross-family v4 cells that lift under semantic (along with
    # Gemma 9B v4). Apples-to-apples vs Qwen 2.5 14B Instruct base: substring
    # +5.3pp; semantic +4.8–5.5pp (still lifts; direction survives). Sum
    # reconciles to pass_n_passes=92.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  6, "n":  6, "rate": 1.000},
        "multihop":            {"pass":  6, "n":  9, "rate": 0.667},
        "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
        "rag_blog":            {"pass":  2, "n":  3, "rate": 0.667},
        "rag_datasheet":       {"pass": 60, "n": 78, "rate": 0.769},
        "rag_email":           {"pass":  0, "n":  3, "rate": 0.000},  # ⚠ regressed
        "reasoning":           {"pass":  3, "n":  6, "rate": 0.500},
        "refusal":             {"pass":  6, "n":  9, "rate": 0.667},  # ⚠ made_up_peripheral
    },
    compute_dtype="fp16",
    gops_per_token=28.0,  # 2 × 14B (full dense forward)
    # 14B Q4 dense perf cell added to RTX_5090_REFERENCE.measured_llm
    # 2026-05-07 per [docs] 10:18 measurement (decode 125.7 tok/s,
    # prefill 5117 tok/s, median over n=132). Same arch / quant as the
    # underlying 14B base — fine-tune doesn't change perf, only weights.
    measurement_alias="qwen25_14b_dense",
)


SKIPPY_QWEN25_32B_V4 = LLMModel(
    key="skippy_qwen25_32b_v4",
    label="Skippy 32B v4 (Qwen2.5-32B FT — recipe trade, NOT recommended)",
    family="Dense — 32B params (no expert sparsity)",
    base="Qwen2.5-32B-Instruct (Alibaba)",
    total_params_b=32.5,
    active_params_b=32.5,
    quant="Q4_K_M GGUF",
    size_gb=17.88,
    fine_tune="v4 — 2 ep + assistant_only_loss + messages format (recipe-clean)",
    pass_rate=0.644,
    pass_n_passes=85,
    pass_n_total=132,
    description=(
        "Recipe-CLEAN 32B v4 fine-tune. 2 epochs + assistant_only_loss + "
        "messages format (full v4 recipe, not the earlier 3-epoch confound "
        "run). Per [docs] 2026-05-07 12:44: regresses **−4.6pp** vs Qwen "
        "2.5 32B Instruct stock (0.682). The recipe doesn't plateau — it "
        "actively trades capability for safety: +3 refusal calibration "
        "(fixes 32B base's made_up_peripheral fabrication 3/3 → 0/3) "
        "BUT −3 numerical_precision, −3 rag_datasheet (over-fit), −3 "
        "multihop (under-trained). Net regression. Voice transfers cleanly "
        "(152 char, matches 14B v4 sweet spot). **Customer rule:** don't "
        "apply v4 recipe at 32B with 6.5K-example corpus unless you weight "
        "refusal-calibration ≥ 3× capability-headline. Cloud spend ~$25 H100."
    ),
    deck_bullet=(
        "32B v4 trades 9 capability points for 3 safety points — net "
        "−4.6pp vs Qwen 2.5 32B Instruct. The recipe doesn't extend "
        "cleanly past 14B on a 6.5K-example corpus. Informative-failure "
        "data: corpus-size-vs-param-count mismatch is a real customer "
        "constraint, not a plateau."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11. Apples-
    # to-apples vs Qwen 2.5 32B Instruct stock: substring −4.6pp; semantic
    # narrows to −3.0pp (still regresses; trade narrows but holds). Sum
    # reconciles to pass_n_passes=85.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  6, "n":  6, "rate": 1.000},
        "multihop":            {"pass":  1, "n":  9, "rate": 0.111},  # under-trained
        "numerical_precision": {"pass":  0, "n":  6, "rate": 0.000},
        "rag_blog":            {"pass":  0, "n":  3, "rate": 0.000},
        "rag_datasheet":       {"pass": 54, "n": 78, "rate": 0.692},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},  # base fabrication fixed
    },
    compute_dtype="fp16",
    gops_per_token=65.0,  # 2 × 32.5B (full dense forward)
    # Same arch / quant as Qwen 2.5 32B Instruct stock — perf cells transfer.
    measurement_alias="qwen25_32b_dense",
)


SKIPPY_MISTRAL_V4 = LLMModel(
    key="skippy_mistral_v4",
    label="Skippy Mistral 7B v0.3 v4 (recipe regressed, NOT recommended)",
    family="Dense — 7B params (no expert sparsity)",
    base="Mistral 7B v0.3 Instruct (Mistral AI)",
    total_params_b=7.25,
    active_params_b=7.25,
    quant="Q4_K_M GGUF",
    size_gb=4.37,
    fine_tune="v4 SFT — same recipe + corpus as Skippy 7B v4, applied to Mistral base",
    pass_rate=0.568,
    pass_n_passes=75,
    pass_n_total=132,
    description=(
        "Mistral 7B v0.3 + v4 recipe (same SFTTrainer + assistant_only_loss "
        "+ 100 refusal exemplars + 2 epochs as Skippy 7B v4) — Tier 3 #1 "
        "cross-family fine-tune attempt per [docs] 2026-05-08 09:56. "
        "**REGRESSED −3.8pp** vs stock Mistral 7B v0.3 (0.568 vs 0.606). "
        "For comparison: same recipe + same corpus on Qwen 2.5 7B base = "
        "+3.1pp (Skippy 7B v4 production). The recipe transfer is NOT "
        "just architecture-coupled (dense ≠ MoE) — it's also **base-"
        "family-coupled**. Qualitative gains transfer (refusal +3, "
        "rag_email +3, numerical_precision +3) but capability cost varies "
        "by base: on Mistral the recipe damages retrieval (rag_datasheet "
        "−8, rag_blog −3) and breaks coding (−3). Hypothesis (untested): "
        "Mistral's chat template + assistant_only_loss reweights away "
        "from RAG-following more than Qwen2.5. Cautionary cell — DO NOT "
        "ship; informative-failure data for the recipe-taxonomy story."
    ),
    deck_bullet=(
        "Skippy v4 recipe ON Mistral 7B v0.3 = 0.568 — REGRESSED 3.8pp "
        "from stock (vs Qwen 7B's +3.1pp lift on the SAME recipe + same "
        "corpus). Voice ✅, Safety ✅ (refusal 9/9), Capability ❌. "
        "Recipe transfer is base-family-coupled, not just dense-vs-MoE-"
        "coupled. Customer rule: re-validate the recipe per base family "
        "before extending."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11
    # white paper Finding 4. Substring originally read −3.8pp vs stock
    # Mistral; semantic widens to −6.1pp because stock Mistral lifted
    # +2.4pp under semantic but v4 stayed flat (±0pp). Sum reconciles
    # to pass_n_passes=75.
    category_deltas={
        "coding":              {"pass":  3, "n":  6, "rate": 0.500},
        "general":             {"pass":  4, "n":  6, "rate": 0.667},
        "multihop":            {"pass":  3, "n":  9, "rate": 0.333},
        "numerical_precision": {"pass":  4, "n":  6, "rate": 0.667},
        "rag_blog":            {"pass":  0, "n":  3, "rate": 0.000},
        "rag_datasheet":       {"pass": 49, "n": 78, "rate": 0.628},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  0, "n":  6, "rate": 0.000},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
    compute_dtype="fp16",
    gops_per_token=14.5,  # 2 × 7.25B
    # Same Mistral 7B v0.3 base arch + GGUF size + compute graph as the
    # stock entry — perf transfers verbatim (verified: GGUF is 4.37 GB,
    # identical to stock). [backend] 23:08 measurement.
    measurement_alias="mistral_7b_v03_dense",
)


# ─────────── Validated MoE recipes (added 2026-05-07 per [docs] Tier 2.x) ──
# Two new MoE-base entries from the [docs] 2026-05-06/07 RunPod campaign:
# the "router-v1" recipe (recommended) and the "full-v1" expert-FFN recipe
# (cautionary over-fit). Both extend the original Skippy MoE FT v1 by
# adding LoRA targets beyond attention. Per [docs] 13:17 wrap-up: router-v1
# is the recommended MoE recipe; full-v1 is informative-failure data.

SKIPPY_MOE_ROUTER_V1 = LLMModel(
    key="skippy_moe_router_v1",
    label="Skippy MoE-router v1 (recommended MoE recipe)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B-Instruct-2507 (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="(attention + router) LoRA — adds gate.weight via target_parameters",
    pass_rate=0.644,
    pass_n_passes=85,
    pass_n_total=132,
    description=(
        "MoE fine-tune with attention + router LoRA targets. Per [docs] "
        "2026-05-06 19:49: validates that adding the router (via peft "
        "target_parameters=['gate.weight']) recovers the reasoning "
        "regression that attention-only LoRA causes on Qwen3-A3B "
        "(multihop 0/9 → 6/9 vs MoE v4). Headline still −3.8pp vs "
        "Instruct-2507 base (67.4% vs 71.2%) — domain-knowledge gap "
        "(rag_datasheet 51/78 vs base 55/78) suggests expert FFNs "
        "would need targeting too… BUT [docs] 13:17 confirmed the "
        "expert-FFN extension OVER-FITS at 6.5K-example corpus size "
        "(see SKIPPY_MOE_FULL_V1). **Customer rule:** include the router "
        "in MoE LoRA targets, do NOT add expert FFN LoRA at this corpus "
        "size. ~$15 H100 cost."
    ),
    deck_bullet=(
        "MoE-router v1 is the recommended MoE-base recipe per [docs] "
        "Tier 2.x: include gate.weight in LoRA targets, recovers "
        "reasoning capability that attention-only loses. Still −3.8pp "
        "vs Instruct base — domain-knowledge gap remains, expert-FFN "
        "extension fails (over-fits). The recipe's diagnostic value: "
        "MoE bases need router targeting, but corpus-too-small for "
        "expert-FFN targeting at 6.5K examples."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Apples-to-apples vs Instruct-2507 base: substring read −3.8pp;
    # semantic narrows to −1.5pp (0.644 vs base 0.659). Sum reconciles
    # to pass_n_passes=85.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  0, "n":  6, "rate": 0.000},
        "multihop":            {"pass":  6, "n":  9, "rate": 0.667},  # recovered from 0/9
        "numerical_precision": {"pass":  0, "n":  6, "rate": 0.000},
        "rag_blog":            {"pass":  0, "n":  3, "rate": 0.000},
        "rag_datasheet":       {"pass": 57, "n": 78, "rate": 0.731},
        "rag_email":           {"pass":  2, "n":  3, "rate": 0.667},
        "reasoning":           {"pass":  5, "n":  6, "rate": 0.833},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
    compute_dtype="int8",
    gops_per_token=6.0,  # 2 × 3B active — same MoE architecture
    # Same Qwen3-30B-A3B Q4 MoE arch as Skippy MoE FT — perf transfers.
    measurement_alias="skippy_finetune",
)


SKIPPY_MOE_FULL_V1 = LLMModel(
    key="skippy_moe_full_v1",
    label="Skippy MoE-full v1 (over-fit cautionary entry)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B-Instruct-2507 (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="(attention + router + packed-expert FFN) LoRA at r=8, ~470M trainable",
    pass_rate=0.621,
    pass_n_passes=82,
    pass_n_total=132,
    description=(
        "MoE fine-tune with attention + router + packed-expert FFNs (r=8 "
        "via peft target_parameters on packed [128, ...] tensors per [docs] "
        "13:17 patch: experts.gate_up_proj [128, 1536, 2048] + "
        "experts.down_proj [128, 2048, 768]). Per [docs] 2026-05-07 10:21: "
        "tests whether expert-level LoRA recovers the domain-knowledge gap "
        "that router-v1 left broken. **Hypothesis FALSIFIED.** 374M "
        "trainable params (1.21% of model) over-fit the 6,517-example "
        "corpus, BROKE rag_blog (3/3 → 0/3) and worsened rag_datasheet "
        "(51 → 47/78) vs router-v1. Voice clipped to 104 char avg "
        "(vs 141 router-v1) — model became too terse for long-form "
        "retrieval. **Customer rule for MoE bases:** stop at router LoRA; "
        "expert FFN LoRA over-fits at this corpus size. Cautionary "
        "informative-failure entry — DO NOT ship."
    ),
    deck_bullet=(
        "MoE-full v1 = cautionary over-fit data. Adding expert-FFN LoRA "
        "(beyond router-v1) at 6.5K-example corpus size dropped headline "
        "67.4% → 62.9%. Voice clipped to 104 char (over-terse), broke "
        "rag_blog 3/3 → 0/3. The extra LoRA capacity lacks training "
        "signal at this corpus size — more isn't always better."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Sum reconciles to pass_n_passes=82.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  3, "n":  6, "rate": 0.500},
        "multihop":            {"pass":  3, "n":  9, "rate": 0.333},
        "numerical_precision": {"pass":  1, "n":  6, "rate": 0.167},
        "rag_blog":            {"pass":  1, "n":  3, "rate": 0.333},
        "rag_datasheet":       {"pass": 53, "n": 78, "rate": 0.679},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  3, "n":  6, "rate": 0.500},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
    compute_dtype="int8",
    gops_per_token=6.0,  # 2 × 3B active — same MoE arch
    measurement_alias="skippy_finetune",
)


# ─────────── Performance-comparison dense models (added 2026-05-01) ────────
# Per [backend] 20:08 weekend bake-off campaign. Two of these (7B Instruct,
# 32B Instruct) anchor the dense-vs-MoE bandwidth-vs-compute comparison on
# the 5090 across multiple quants. The 7B entry now also serves as the
# apples-to-apples baseline for SKIPPY_7B_V4 (per [docs] 2026-05-06 09:19).
#
# Headline narrative this enables:
# - Skippy MoE 30B-A3B Q4: ~250 tok/s on 5090 (3B active streams 1.65 GB/tok)
# - Qwen 2.5 32B dense Q4:  52.7 tok/s on 5090 (full 32B streams 17.88 GB/tok)
# - 4.7× MoE-vs-dense BW advantage at the 30B-class size — exactly because
#   MoE only reads active experts per token.

QWEN25_7B_DENSE_INSTRUCT = LLMModel(
    key="qwen25_7b_dense",
    label="Qwen 2.5 7B Instruct (dense — apples-to-apples 7B v4 baseline)",
    family="Dense — 7B params (no expert sparsity)",
    base="Qwen2.5-7B-Instruct (Alibaba)",
    total_params_b=7.6,
    active_params_b=7.6,
    quant="Q4_K_M GGUF (also Q5_K_M, Q8_0 measured)",
    size_gb=4.18,  # Q4 footprint; sizer's per-quant scaling applies via decode_bw_per_token_gb
    pass_rate=0.652,
    pass_n_passes=86,
    pass_n_total=132,
    description=(
        "Stock Qwen 2.5 7B dense Instruct — serves dual duty: (a) perf-"
        "comparison anchor for the dense-vs-MoE narrative (decode tok/s "
        "measured across Q4/Q5/Q8 quants on 5090), and (b) **apples-to-"
        "apples baseline for Skippy 7B v4** (the +3.1pp gain claim is "
        "0.705 - 0.674 vs this row, per [docs] 2026-05-06 09:19). "
        "Per-category breakdown: coding 6/6, general 3/3, multihop 5/9, "
        "num_precision 3/6, rag_blog 3/3, rag_datasheet 54/78 (0.692), "
        "rag_email 0/3, reasoning 6/6, refusal 9/9."
    ),
    deck_bullet=(
        "7B dense Q4 = 184 tok/s on 5090 vs Skippy MoE 30B-A3B Q4 = 250 tok/s — "
        "MoE wins on bandwidth despite 4× more total params, because only 3B "
        "active streams per token. Also the apples-to-apples baseline for "
        "Skippy 7B v4's +3.1pp validated fine-tune gain."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Substring read 0.674; semantic 0.652 (−2.2pp — small shift; Qwen-
    # family bases are the baseline against which the substring format
    # bonus is measured). Sum reconciles to pass_n_passes=86.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  6, "n":  6, "rate": 1.000},
        "multihop":            {"pass":  1, "n":  9, "rate": 0.111},
        "numerical_precision": {"pass":  0, "n":  6, "rate": 0.000},
        "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
        "rag_datasheet":       {"pass": 57, "n": 78, "rate": 0.731},
        "rag_email":           {"pass":  0, "n":  3, "rate": 0.000},
        "reasoning":           {"pass":  4, "n":  6, "rate": 0.667},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
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
    label="Qwen 2.5 32B Instruct (dense — apples-to-apples 32B v4 baseline)",
    family="Dense — 32B params (no expert sparsity)",
    base="Qwen2.5-32B-Instruct (Alibaba)",
    total_params_b=32.5,
    active_params_b=32.5,
    quant="Q4_K_M GGUF (also Q5_K_M measured; Q8 won't fit on 5090)",
    size_gb=17.88,  # Q4 footprint
    pass_rate=0.674,
    pass_n_passes=89,
    pass_n_total=132,
    description=(
        "Stock Qwen 2.5 32B Instruct — serves dual duty: (a) perf-comparison "
        "anchor at the 30B-class size for the dense-vs-MoE slope test, and "
        "(b) **apples-to-apples baseline for Skippy 32B v4** (per [docs] "
        "2026-05-07 12:44). Notable per-category: numerical_precision 6/6 "
        "(perfect — base capability the FT recipe destroyed), but refusal "
        "6/9 (made_up_peripheral 3/3 wrong — confidently fabricates "
        "fictional features). The fabrication problem is a Qwen2.5 base-"
        "model behavior, not something fine-tuning created — both 32B "
        "v4 variants RECOVER it to 9/9 at the cost of capability points "
        "elsewhere (net −4.6pp). Per-category breakdown: coding 6/6, "
        "general 3/3, multihop 6/9, num_precision 6/6, rag_blog 3/3, "
        "rag_datasheet 51/78 (0.654), rag_email 3/3, reasoning 6/6, "
        "refusal 6/9 (made_up_peripheral 3/3 fabricated)."
    ),
    deck_bullet=(
        "Qwen 2.5 32B dense Q4 = 52.7 tok/s on 5090 vs Skippy MoE 30B-A3B "
        "Q4 = 250 tok/s (4.7× MoE bandwidth advantage). Also the apples-"
        "to-apples baseline for Skippy 32B v4 — at 0.682 pass-rate, the "
        "FT regresses 4.6pp from this row. The recipe trade is real."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Substring read 0.682; semantic 0.674 (−0.8pp — small shift,
    # Qwen-family base). Notable: refusal 7/9 — made_up_peripheral
    # fabrication is partial under semantic (substring read 6/9). Sum
    # reconciles to pass_n_passes=89.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  6, "n":  6, "rate": 1.000},
        "multihop":            {"pass":  6, "n":  9, "rate": 0.667},
        "numerical_precision": {"pass":  1, "n":  6, "rate": 0.167},
        "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
        "rag_datasheet":       {"pass": 54, "n": 78, "rate": 0.692},
        "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
        "reasoning":           {"pass":  3, "n":  6, "rate": 0.500},
        "refusal":             {"pass":  7, "n":  9, "rate": 0.778},  # ⚠ fabrication
    },
    compute_dtype="fp16",
    # 5090 anchors (Q4/Q5 decode + prefill) live on
    # RTX_5090_REFERENCE.measured_llm — single source of truth. No Q8
    # entry — won't fit on 5090's 32 GB VRAM.
    gops_per_token=65.0,  # 2 × 32.5B
)


# ─────────── Cross-family baseline (Tier 3 #1, added 2026-05-07) ─────────
# Per [docs] 2026-05-07 22:28: Llama-3.1 8B Instruct stock as a non-Qwen
# baseline. Tests whether the Skippy v4 fine-tune story transfers across
# base architecture families. Spoiler: −10.6pp vs Qwen2.5-7B-Instruct at
# similar size, with reasoning catastrophic at 1/6 (vs Qwen 6/6) and an
# 18× higher trailing-question rate ('what else can I help you with?' as
# default voice).
#
# Customer-template implication: v4 recipe gains validated only on the
# Qwen2.5 family. Don't assume +3-5pp transfers to non-Qwen bases without
# a fresh apples-to-apples baseline. Llama v4 fine-tune was the planned
# follow-up but Meta-Llama is HF-gated; [docs] is pivoting the FT to
# Mistral-7B-Instruct-v0.3 (ungated, similar size, different family).

LLAMA_3_1_8B_INSTRUCT_STOCK = LLMModel(
    key="llama_3_1_8b_instruct_stock",
    label="Meta Llama-3.1 8B Instruct (stock — cross-family baseline)",
    family="Dense — 8B params (no expert sparsity)",
    base="Meta Llama-3.1 8B Instruct (Meta)",
    total_params_b=8.03,
    active_params_b=8.03,
    quant="Q4_K_M GGUF (bartowski mirror)",
    size_gb=4.92,  # Q4_K_M footprint for 8B class
    fine_tune="stock (no fine-tune)",
    pass_rate=0.583,
    pass_n_passes=77,
    pass_n_total=132,
    description=(
        "Cross-family baseline per [docs] 2026-05-07 22:28. Materially "
        "weaker than Qwen-family bases at similar size: −10.6pp vs Qwen "
        "2.5 7B Instruct (0.568 vs 0.674). Reasoning catastrophic at "
        "1/6 (vs Qwen 6/6) — 5/6 of Skippy's reasoning prompts fail at "
        "the Llama base level. rag_datasheet also weaker (45/78 vs "
        "Qwen 54/78). Same made_up_peripheral fabrication issue as Qwen "
        "2.5 32B Instruct (refusal 6/9). Voice profile differs sharply: "
        "18× the trailing-question rate ('what else can I help you with?') "
        "— Kyle voice gate would treat this as immediate stylistic "
        "regression. **Customer rule:** don't assume v4 recipe gains "
        "transfer to non-Qwen bases without a fresh baseline + voice gate "
        "retune for the target family's natural cadence."
    ),
    deck_bullet=(
        "Cross-family baseline: Llama-3.1 8B Instruct stock = 0.568 on "
        "v2-RAG, **−10.6pp vs Qwen 2.5 7B Instruct** at similar size. "
        "Per-vendor capability variance is real and large at this param "
        "count. v4 recipe gains validated on Qwen2.5 family — won't "
        "necessarily transfer to other vendors without per-base "
        "calibration."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Substring read 0.568; semantic 0.583 (+1.6pp — substring was being
    # unfairly harsh on non-Qwen bases; cross-family gap to Qwen narrows
    # from −10.6pp to −6.9pp). Sum reconciles to pass_n_passes=77.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  6, "n":  6, "rate": 1.000},
        "multihop":            {"pass":  5, "n":  9, "rate": 0.556},
        "numerical_precision": {"pass":  0, "n":  6, "rate": 0.000},
        "rag_blog":            {"pass":  2, "n":  3, "rate": 0.667},
        "rag_datasheet":       {"pass": 48, "n": 78, "rate": 0.615},
        "rag_email":           {"pass":  1, "n":  3, "rate": 0.333},
        "reasoning":           {"pass":  0, "n":  6, "rate": 0.000},
        "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
    },
    # Dense Q4 → fp16 internal path on llama.cpp. Selecting on Mid (INT8-
    # only) triggers 🔴 dtype_mismatch in the app.py UI gate.
    compute_dtype="fp16",
    gops_per_token=16.0,  # 2 × 8B (full dense forward)
    # 5090 perf cell wired 2026-05-08 from [backend] 2026-05-07 23:08
    # bake-off (decode 171.0 tok/s RAG 8K+2K, prefill@2K 10162 tok/s).
    # Replaces the 🟠 cross_class fallback (332.79 tok/s, 1.95× over-
    # projection per [backend] calibration check) with 🟢 measured.
    # GGUF arch + size matches across stock + any future Llama FT, so
    # if a Llama v4 FT lands (currently blocked — Meta-Llama HF-gated;
    # Mistral pivot in flight), the same alias would carry over.
    measurement_alias="llama_3_1_8b_dense",
)


MISTRAL_7B_V03_INSTRUCT_STOCK = LLMModel(
    key="mistral_7b_v03_instruct_stock",
    label="Mistral 7B v0.3 Instruct (stock — cross-family baseline)",
    family="Dense — 7B params (no expert sparsity)",
    base="Mistral 7B v0.3 Instruct (Mistral AI)",
    total_params_b=7.25,
    active_params_b=7.25,
    quant="Q4_K_M GGUF (bartowski mirror)",
    size_gb=4.37,
    fine_tune="stock (no fine-tune)",
    pass_rate=0.629,
    pass_n_passes=83,
    pass_n_total=132,
    description=(
        "Cross-family baseline per [docs] 2026-05-08 09:01. Sits between "
        "Qwen 2.5 7B Instruct (0.674) and Llama-3.1 8B Instruct (0.568) "
        "on the cross-family ladder: −6.8pp vs Qwen 7B base. Same "
        "made_up_peripheral fabrication issue (refusal 6/9) that Llama "
        "and Qwen-32B share at the base level. **Reasoning catastrophic "
        "at 0/6** — Mistral fails ALL 6 of Skippy's reasoning prompts "
        "at the base level (Qwen 6/6, Llama 1/6). Customer rule: "
        "reasoning capability varies wildly between vendor bases at "
        "similar param count. Mistral v4 FT is the planned Tier 3 "
        "follow-up (training completed 2026-05-07 23:55 per [docs] 09:41; "
        "merge → GGUF → eval ETA ~15-20 min from there)."
    ),
    deck_bullet=(
        "Cross-family baseline: Mistral 7B v0.3 Instruct stock = 0.606 "
        "on v2-RAG, between Qwen (0.674) and Llama (0.568). Reasoning "
        "category is the cross-family delta: 0/6 here vs Qwen's 6/6 "
        "— vendor chain-of-thought training shows up directly in pass "
        "rate. Same hardware budget as Qwen 7B (within ~7% on 5090 "
        "perf), different quality outcome."
    ),
    # Per-category raw rates — SEMANTIC GRADE per [docs] 2026-05-11.
    # Substring read 0.606; semantic 0.629 (+2.4pp — substring was unfairly
    # harsh on Mistral). Cross-family ladder semantic: Qwen 65.2 > Mistral
    # 62.9 > Llama 58.3. Sum reconciles to pass_n_passes=83.
    category_deltas={
        "coding":              {"pass":  6, "n":  6, "rate": 1.000},
        "general":             {"pass":  3, "n":  6, "rate": 0.500},
        "multihop":            {"pass":  7, "n":  9, "rate": 0.778},
        "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
        "rag_blog":            {"pass":  2, "n":  3, "rate": 0.667},
        "rag_datasheet":       {"pass": 56, "n": 78, "rate": 0.718},
        "rag_email":           {"pass":  0, "n":  3, "rate": 0.000},
        "reasoning":           {"pass":  0, "n":  6, "rate": 0.000},
        "refusal":             {"pass":  6, "n":  9, "rate": 0.667},
    },
    # Dense Q4 → fp16 internal path on llama.cpp. Selecting on Mid (INT8-
    # only) triggers 🔴 dtype_mismatch in the app.py UI gate.
    compute_dtype="fp16",
    gops_per_token=14.5,  # 2 × 7.25B (full dense forward)
    # Wired to mistral_7b_v03_dense cell in RTX_5090_REFERENCE.measured_llm
    # ([backend] 23:08 bake-off: decode 182.7 tok/s RAG, prefill@2K 10217).
    # Same alias would carry over to a future Mistral v4 FT (FT preserves
    # base arch + GGUF size + compute graph; only weight values change).
    measurement_alias="mistral_7b_v03_dense",
)


LLM_MODELS: dict[str, LLMModel] = {
    # Order matters — drives selectbox display order. Convention:
    # production first (default), then prior-production (cost-different
    # but quality-parity), then external Skippy-eval'd baselines, then
    # perf-comparison-only references (no Skippy v2 evaluation).
    # Production-first ordering: current production model leads, then
    # validated dense FTs by size, then validated MoE recipes, then
    # historical / cautionary FTs, then apples-to-apples baselines, then
    # perf-only references.
    SKIPPY_7B_V4.key:               SKIPPY_7B_V4,                 # current production
    SKIPPY_14B_V4.key:              SKIPPY_14B_V4,                # best dense headline
    SKIPPY_QWEN25_32B_V4.key:       SKIPPY_QWEN25_32B_V4,         # 32B trade — NOT recommended
    SKIPPY_MISTRAL_V4.key:          SKIPPY_MISTRAL_V4,            # cross-family FT regression — NOT recommended
    SKIPPY_MOE_FINETUNE.key:        SKIPPY_MOE_FINETUNE,          # MoE FT v1 (historical)
    SKIPPY_MOE_ROUTER_V1.key:       SKIPPY_MOE_ROUTER_V1,         # recommended MoE recipe
    SKIPPY_MOE_FULL_V1.key:         SKIPPY_MOE_FULL_V1,           # MoE over-fit (cautionary)
    SKIPPY_DENSE_FINETUNE.key:      SKIPPY_DENSE_FINETUNE,        # pre-v4 dense (historical)
    INSTRUCT_MOE_STOCK.key:         INSTRUCT_MOE_STOCK,           # MoE apples-to-apples base
    THINKING_MOE_STOCK.key:         THINKING_MOE_STOCK,           # sister-model context
    QWEN25_7B_DENSE_INSTRUCT.key:   QWEN25_7B_DENSE_INSTRUCT,     # 7B v4 baseline
    QWEN25_32B_DENSE_INSTRUCT.key:  QWEN25_32B_DENSE_INSTRUCT,    # 32B v4 baseline
    LLAMA_3_1_8B_INSTRUCT_STOCK.key: LLAMA_3_1_8B_INSTRUCT_STOCK, # cross-family baseline (Tier 3)
    MISTRAL_7B_V03_INSTRUCT_STOCK.key: MISTRAL_7B_V03_INSTRUCT_STOCK,  # cross-family baseline (Tier 3)
}

DEFAULT_LLM_MODEL_KEY = SKIPPY_7B_V4.key

# Reference for all per-category-Δ rendering. UI labels comparisons as
# "vs <production model> (production)" and the production model itself
# shows no per-category breakdown (it would be 0 against itself).
#
# Shifted from SKIPPY_MOE_FINETUNE.key → SKIPPY_7B_V4.key per [docs]
# 2026-05-06 09:51: 7B v4 IS production as of 2026-05-04 17:30 (revert
# from the brief Instruct-2507 stock swap). The "Δ vs production" column
# should answer "how does this compare to what Skippy actually ships
# today?" — anchoring against the old MoE FT row was misleading.
PRODUCTION_REFERENCE_KEY = SKIPPY_7B_V4.key


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
