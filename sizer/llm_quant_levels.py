"""Precision-axis quality reference for the LLM workload.

Captures the measured accuracy cost of running the LLM at different
quantization recipes — fp16 → FP8 → W8A8 INT8 — plus no-RAG and
fine-tune anchors so the full ladder reads cleanly. All measured by
[docs] against the v2 prompt set (44 prompts × 3 samples = 132).

Distinct from `sizer/llm_models.py`, which captures the **model axis**
(fine-tune vs stock public reasoning model). This module is the
**precision axis** (quality cost of the quantization recipe on the
base model). Kyle's deck wants both stories surfaced:

  - Model axis:     dense fine-tunes (Qwen2.5-7B v4 +3.1pp, 14B v4
                    +5.3pp vs respective Instruct bases) post real
                    apples-to-apples gains. MoE-base validation pending
                    per [docs] 2026-05-05 — the previously cited +5pp
                    "domain headroom" claim was confounded by the
                    +7.6pp Instruct-vs-Thinking sister-model gap. See
                    `sizer/llm_models.py` module docstring for the
                    base-identity caveat.
  - Precision axis: W8A8 INT8 costs ~3.8pp vs fp16, concentrated in
                    retrieval-grounded wording; structured output
                    (coding, reasoning) is byte-identical (Jaccard 1.0)

The precision-axis story composes with the 5090 capability caption:
W8A8 vLLM on consumer Blackwell SM120 throws
`RuntimeError: Int8 not supported on SM120 …` from
`torch.ops._C.cutlass_scaled_mm` because CUTLASS compiles INT8 kernels
fresh for the target arch and SM120 lacks native INT8 tensor-core
instructions. UI annotates W8A8 rows on `tensor_compat` tiers as n/a.

DO NOT reintroduce a "refusal-specificity" framing. The 2026-04-23
H100 reference run had refusals -2, but the 2026-04-24 reproducer had
refusals +0 — that earlier localization didn't hold. Stable claim is
"headline -3.8pp, retrieval-grounded wording drifts most, structured
output untouched." See `acc_diff_fp16_vs_int8_v2_rag_pure.md` on
[docs]'s Drive folder for the artifact.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMQuantConfig:
    """One quantization-recipe / RAG configuration with measured pass rate."""
    key: str
    label: str
    base_model: str       # 'Qwen2.5-14B base' / 'Qwen3-30B-A3B fine-tune' / etc.
    precision: str        # 'fp16' / 'FP8' / 'W8A8 INT8' / 'Q4_K_M GGUF'
    rag_enabled: bool
    runtime: str          # 'vLLM' / 'llama.cpp (GGUF)' / etc.
    measurement_host: str
    pass_rate: float
    pass_n_passes: int
    pass_n_total: int = 132
    note: str = ""


# Reference: dense fp16 + RAG. The Δpp anchor for "quality cost of
# quantization" claims. Measured on a non-Blackwell GPU because consumer
# 5090 SM120 refuses cutlass_scaled_mm for INT8 path comparisons.
QWEN_FP16_RAG = LLMQuantConfig(
    key="qwen14b_fp16_rag",
    label="Dense fp16 + RAG",
    base_model="Qwen2.5-14B (base, no fine-tune)",
    precision="fp16",
    rag_enabled=True,
    runtime="vLLM",
    measurement_host="H100 80GB",
    pass_rate=0.652,
    pass_n_passes=86,
    note="Anchor for precision-axis Δpp.",
)

QWEN_FP8_RAG = LLMQuantConfig(
    key="qwen14b_fp8_rag",
    label="Dense FP8 + RAG",
    base_model="Qwen2.5-14B (base, no fine-tune)",
    precision="FP8",
    rag_enabled=True,
    runtime="vLLM",
    measurement_host="RTX 5090",
    pass_rate=0.636,
    pass_n_passes=84,
    note="-1.6pp vs fp16. FP8 tensor cores native on SM120 — no compat issue.",
)

QWEN_W8A8_RAG = LLMQuantConfig(
    key="qwen14b_w8a8_rag",
    label="Dense W8A8 INT8 + RAG",
    base_model="Qwen2.5-14B (base, no fine-tune)",
    precision="W8A8 INT8 (SmoothQuant + GPTQ)",
    rag_enabled=True,
    runtime="vLLM",
    measurement_host="H100 80GB (5090 SM120 cutlass_scaled_mm blocked)",
    pass_rate=0.614,
    pass_n_passes=81,
    note=(
        "-3.8pp vs fp16, reproduced across two H100 runs. Regression "
        "concentrated in retrieval-grounded wording; structured output "
        "byte-identical."
    ),
)

QWEN_FP16_NO_RAG = LLMQuantConfig(
    key="qwen14b_fp16_no_rag",
    label="Dense fp16, no RAG",
    base_model="Qwen2.5-14B (base, no fine-tune)",
    precision="fp16",
    rag_enabled=False,
    runtime="vLLM",
    measurement_host="H100 80GB",
    pass_rate=0.303,
    pass_n_passes=40,
    note="Why RAG is load-bearing: -34.9pp without it.",
)

QWEN_FP8_NO_RAG = LLMQuantConfig(
    key="qwen14b_fp8_no_rag",
    label="Dense FP8, no RAG",
    base_model="Qwen2.5-14B (base, no fine-tune)",
    precision="FP8",
    rag_enabled=False,
    runtime="vLLM",
    measurement_host="RTX 5090",
    pass_rate=0.318,
    pass_n_passes=42,
)

QWEN_Q4KM_FT_RAG = LLMQuantConfig(
    key="qwen14b_q4km_finetune_rag",
    label="Dense Q4_K_M fine-tune + RAG",
    base_model="Qwen2.5-14B (Skippy domain LoRA)",
    precision="Q4_K_M GGUF",
    rag_enabled=True,
    runtime="llama.cpp",
    measurement_host="RTX 5090",
    pass_rate=0.682,
    pass_n_passes=90,
    note="Fine-tune adds +3.0pp over fp16 base despite the Q4 quant haircut.",
)

SKIPPY_MOE_Q4KM_RAG = LLMQuantConfig(
    key="qwen3_a3b_q4km_finetune_rag",
    label="MoE Q4_K_M fine-tune + RAG (production)",
    base_model="Qwen3-30B-A3B (Skippy domain LoRA)",
    precision="Q4_K_M GGUF",
    rag_enabled=True,
    runtime="llama.cpp",
    measurement_host="RTX 5090",
    pass_rate=0.689,
    pass_n_passes=91,
    note="The model actually shipping. +3.7pp over fp16 dense base.",
)


# Ordered weakest → strongest so the ladder reads as a quality progression.
LLM_QUANT_LADDER: tuple[LLMQuantConfig, ...] = (
    QWEN_FP16_NO_RAG,
    QWEN_FP8_NO_RAG,
    QWEN_W8A8_RAG,
    QWEN_FP8_RAG,
    QWEN_FP16_RAG,
    QWEN_Q4KM_FT_RAG,
    SKIPPY_MOE_Q4KM_RAG,
)

# Anchor used as fp16 reference for Δpp computations.
FP16_REFERENCE = QWEN_FP16_RAG


# Per-category Δ for W8A8 vs fp16 reference (both runs RAG on, v2 prompts).
# Sourced from `acc_diff_fp16_vs_int8_v2_rag_pure.md` (Drive,
# personal-ai-assistant/), 2026-04-24 reproducer run. Sign convention:
# positive = W8A8 wins. All categories present in the diff are listed
# even when flat, so the table makes "structured output untouched" visible.
W8A8_VS_FP16_CATEGORY_DELTAS: dict[str, int] = {
    "rag_datasheet": -2,
    "rag_email":     -3,
    "coding":         0,    # byte-identical, Jaccard 1.0
    "reasoning":      0,    # byte-identical, Jaccard 1.0
    "numerical_precision": 0,
    "multihop":       0,
    "general":        0,
    "persona":        0,
    "rag_blog":       0,
    "refusal":        0,    # see module docstring — DO NOT cite as -2
}


def delta_pp_vs_fp16(cfg: LLMQuantConfig) -> float:
    """Return (cfg.pass_rate - FP16_REFERENCE.pass_rate) in percentage points."""
    return (cfg.pass_rate - FP16_REFERENCE.pass_rate) * 100.0
