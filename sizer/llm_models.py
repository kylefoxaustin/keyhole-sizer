"""LLM model catalog for the sizer.

Three selectable models for the LLM workload comparison surface:
- Skippy MoE domain fine-tune (default, current production)
- Skippy dense domain fine-tune (prior production, near-parity)
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

The two stories the catalog answers:
1. Model-architecture trade: "would a stock public reasoning model
   just replace the domain fine-tune?" → not on your domain.
2. Cost trade: "would the older dense fine-tune work just as well?" →
   yes on quality (Δ +0.7pp), no on cost (4-5× more weights to read
   per token).

## Convention: category_deltas

All `category_deltas` are **vs the production reference**
(`PRODUCTION_REFERENCE_KEY` = Skippy MoE FT). Sign convention:
positive = this model wins vs production. The production reference
itself has an empty `category_deltas` dict (it can't differ from
itself). UI surfaces these as "Per-category Δ vs Skippy MoE
fine-tune (production)" in the accuracy expander.

Accuracy from Skippy v2 prompt set: 44 prompts × 3 samples = 132,
RAG enabled (8 chunks via hybrid retrieval). Diffs:
- `eval/results/acc_diff_skippy_fine_tune_vs_thinking.md`
- `eval/results/acc_diff_dense_q4km_vs_moe_q4km_v2_rag.md`
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMModel:
    """One selectable LLM with accuracy stats vs the v2 prompt set."""
    key: str
    label: str
    family: str
    base: str
    total_params_b: float
    active_params_b: float
    quant: str
    size_gb: float
    fine_tune: str            # 'domain LoRA (NXP datasheets)' / 'public reasoning-tune' / etc.
    pass_rate: float          # overall, RAG on
    pass_n_passes: int
    pass_n_total: int
    description: str
    deck_bullet: str          # the one-line framing for the deck/UI
    # Per-category Δ vs the OTHER model (sign convention: positive = THIS wins).
    # Keyed by Skippy v2 prompt category. Categories not present = flat.
    category_deltas: dict[str, int]


SKIPPY_MOE_FINETUNE = LLMModel(
    key="skippy_finetune",
    label="Skippy MoE fine-tune (DEFAULT)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="domain LoRA — NXP datasheets, transcripts, Skippy prod corpus",
    pass_rate=0.689,
    pass_n_passes=91,
    pass_n_total=132,
    description=(
        "Kyle's domain fine-tune of Qwen3-30B-A3B — the model shipping in "
        "Skippy production. LoRA-adapted on internal corpora (NXP datasheets, "
        "meeting transcripts, source code). Default selection and the "
        "**production reference** for per-category Δ comparisons across the "
        "catalog."
    ),
    deck_bullet=(
        "Domain fine-tuning buys +5.3pp on Kyle's RAG eval vs stock public "
        "reasoning models — not by making the model 'smarter' in general, "
        "but by teaching it the retrieval vocabulary of the domain."
    ),
    # Production reference: vs itself = no Δs. Other entries' deltas
    # are computed against this row.
    category_deltas={},
)


SKIPPY_DENSE_FINETUNE = LLMModel(
    key="skippy_dense_finetune",
    label="Skippy dense fine-tune (Qwen2.5-14B)",
    family="Dense — 14B params (no expert sparsity)",
    base="Qwen2.5-14B (Alibaba)",
    total_params_b=14.0,
    active_params_b=14.0,
    quant="Q4_K_M GGUF",
    size_gb=9.2,
    fine_tune="domain LoRA — same NXP/Skippy corpus as the MoE entry",
    pass_rate=0.682,
    pass_n_passes=90,
    pass_n_total=132,
    description=(
        "Kyle's prior-production dense fine-tune. Same LoRA recipe as the "
        "MoE entry, applied to the dense Qwen2.5-14B base. Maintained for "
        "the dense-vs-MoE quality comparison — kept in catalog because the "
        "answer to 'is MoE actually better?' isn't 'on quality' (it's a "
        "wash), it's 'on cost-per-token.'"
    ),
    deck_bullet=(
        "Dense and MoE fine-tunes hit near-parity on quality (Δ +0.7pp) — "
        "MoE wins on per-token cost (3B active << 14B dense), NOT accuracy. "
        "Choosing MoE is a cost decision, not a capability one."
    ),
    # Δ vs Skippy MoE FT (production reference). MoE wins rag_datasheet
    # +3 → dense -3 here; MoE loses refusal -2 → dense +2 here.
    category_deltas={
        "rag_datasheet": -3,
        "refusal":       +2,
    },
)


THINKING_MOE_STOCK = LLMModel(
    key="thinking_stock",
    label="Qwen3-30B-A3B-Thinking-2507 (stock public)",
    family="Sparse MoE — 128 experts × 8 used per token",
    base="Qwen3-30B-A3B (Alibaba)",
    total_params_b=30.0,
    active_params_b=3.0,
    quant="Q4_K_M GGUF",
    size_gb=18.0,
    fine_tune="public reasoning-tune (no domain corpus)",
    pass_rate=0.636,
    pass_n_passes=84,
    pass_n_total=132,
    description=(
        "Public Qwen3-30B-A3B-Thinking-2507 — Alibaba's reasoning-tuned variant "
        "of the same base. No exposure to Kyle's domain corpus. Used as the "
        "'would a free public model just replace the fine-tune?' comparison."
    ),
    deck_bullet=(
        "Public reasoning models are stronger in general, but lose to the "
        "domain fine-tune on retrieval-grounded queries — domain knowledge "
        "doesn't fall out of larger general capability."
    ),
    category_deltas={
        "rag_datasheet": -8,
        "rag_email":     -3,
        "numerical_precision": +3,
        "refusal":       +2,
    },
)


LLM_MODELS: dict[str, LLMModel] = {
    # Order matters — drives selectbox display order. Convention:
    # production first (default), then prior-production (cost-different
    # but quality-parity), then external baselines.
    SKIPPY_MOE_FINETUNE.key:   SKIPPY_MOE_FINETUNE,
    SKIPPY_DENSE_FINETUNE.key: SKIPPY_DENSE_FINETUNE,
    THINKING_MOE_STOCK.key:    THINKING_MOE_STOCK,
}

DEFAULT_LLM_MODEL_KEY = SKIPPY_MOE_FINETUNE.key

# Reference for all per-category-Δ rendering. UI labels comparisons as
# "vs Skippy MoE fine-tune (production)" and the production model
# itself shows no per-category breakdown (it would be 0 across the board).
PRODUCTION_REFERENCE_KEY = SKIPPY_MOE_FINETUNE.key


def accuracy_delta_pp(a: LLMModel, b: LLMModel) -> float:
    """Return (a.pass_rate - b.pass_rate) in percentage points."""
    return (a.pass_rate - b.pass_rate) * 100.0


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
