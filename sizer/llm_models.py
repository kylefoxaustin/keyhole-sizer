"""LLM model catalog for the sizer.

Two selectable models for the LLM workload comparison surface:
- Skippy's domain fine-tune (default, shipping in production)
- Qwen3-30B-A3B-Thinking-2507 (stock public reasoning baseline)

Both are Q4_K_M GGUF MoE 30B/3B-active, so decode-tok/s projections
on the Hardware tiers are identical across models — bandwidth-bound
physics doesn't care which weights are in the file. **Accuracy is the
only thing that differs.** The comparison answers a specific question
Kyle's silicon-architecture audience asks: "would a stock public
reasoning model just replace the domain fine-tune?" — answer: not on
your domain.

Accuracy from Skippy v2 prompt set: 44 prompts × 3 samples = 132,
RAG enabled (8 chunks via hybrid retrieval), measured by [backend]
session 2026-04-24, results in
`eval/results/acc_diff_skippy_fine_tune_vs_thinking.md` on the
personal-ai-framework side.
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
        "meeting transcripts, source code). Default selection."
    ),
    deck_bullet=(
        "Domain fine-tuning buys +5.3pp on Kyle's RAG eval — not by making "
        "the model 'smarter' in general, but by teaching it the retrieval "
        "vocabulary of the domain."
    ),
    category_deltas={
        "rag_datasheet": +8,    # 26 prompts; fine-tune lands the domain vocabulary
        "rag_email":     +3,    # 1 prompt × 3 samples; stock failed all three
        "numerical_precision": -3,  # general reasoning Thinking trains for harder
        "refusal":       -2,    # scope-limiting tuned harder in Thinking
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
    SKIPPY_MOE_FINETUNE.key: SKIPPY_MOE_FINETUNE,
    THINKING_MOE_STOCK.key:  THINKING_MOE_STOCK,
}

DEFAULT_LLM_MODEL_KEY = SKIPPY_MOE_FINETUNE.key


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
