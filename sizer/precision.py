"""Hardware capability taxonomy — how silicon can execute a given dtype.

Each Hardware tier declares a `capability_levels` mapping that explains,
per dtype, what kernel path the silicon is capable of taking. The
taxonomy exists to surface 'why' behind measured outcomes — especially
why consumer Blackwell (SM120) runs INT8 TRT engines successfully via
pre-compiled Ampere IMMA kernels, but vLLM CUTLASS fresh-compile fails
for the same dtype. See STORY_DRAFT.md Tier 2 for the narrative.

Levels (decreasing order of expected efficiency):
  tensor_native  — silicon has tensor-core instructions for this dtype
                   compiled natively for its arch. Optimal path.
  tensor_compat  — tensor-core execution via CUDA binary compatibility
                   with an older arch. Kernels still run on tensor cores
                   but were compiled for a different SM target. The 5090
                   INT8 case: SM120 dropped new IMMA instructions, but
                   sm80 IMMA kernels (pre-compiled in TRT 10.16) run
                   correctly via CUDA binary compatibility.
  cuda_core      — no tensor-core path for this dtype. Execution falls
                   back to generic CUDA / shader cores. Significantly
                   slower than either tensor path.
  unsupported    — silicon cannot execute this dtype at all. LLM loaders
                   typically error out; vision pipelines fail to build.

Build-time / toolchain concerns (e.g. TRT 10.16 silently falling back
from FP8 to FP16 when the engine lacks explicit QDQ nodes) are NOT
captured here — those are engine-specific, not a hardware capability.
Track them per-pipeline via an `effective_precision` field if/when
enough engines need the distinction to justify the extra schema.
"""
from __future__ import annotations

from typing import Literal

CapabilityLevel = Literal["tensor_native", "tensor_compat", "cuda_core", "unsupported"]

CAPABILITY_LABELS: dict[CapabilityLevel, str] = {
    "tensor_native": "Tensor core (native)",
    "tensor_compat": "Tensor core (via binary compat)",
    "cuda_core":     "CUDA core fallback",
    "unsupported":   "Unsupported",
}

CAPABILITY_DESCRIPTIONS: dict[CapabilityLevel, str] = {
    "tensor_native": "Silicon has native tensor-core instructions for this dtype compiled for its arch.",
    "tensor_compat": (
        "Tensor-core execution via pre-compiled kernels from an older arch "
        "(CUDA binary compatibility). Fresh-compile paths (vLLM CUTLASS) "
        "hit `RuntimeError: Int8 not supported on SM120. Use FP8 "
        "quantization instead, or run on older arch (SM < 100).` — "
        "TRT works because it ships sm80 IMMA kernels in its binary "
        "library; CUTLASS fails because it tries to emit SM120-native "
        "INT8 instructions that don't exist."
    ),
    "cuda_core":     "No tensor-core path for this dtype; execution falls through to generic CUDA cores.",
    "unsupported":   "Silicon cannot execute this dtype — the stack will fail to load or refuse the op.",
}

# Canonical dtype ordering for display / docstrings.
DTYPE_ORDER: tuple[str, ...] = ("int8", "fp8", "bf16", "fp16")
