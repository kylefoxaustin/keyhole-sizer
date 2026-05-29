"""VLA (Vision-Language-Action) model catalog for the sizer.

Selectable VLAs for the v1.2.0 VLA workload surface — six entries
covering the architectural quadrants the deck needs (autoregressive
vs. dual-loop, integer-friendly vs. FP-required):

| Key                   | Architecture        | dtype_default | Notes                                    |
|-----------------------|---------------------|---------------|------------------------------------------|
| openvla_7b_single     | single_loop         | int8          | Native autoregressive, 7-DOF discretized |
| openvla_7b_cached     | dual_loop_capable   | int8          | Synthetic projection (no published artifact) |
| nora_3b               | single_loop         | int8          | Compute-friendly baseline; FAST+ tokenizer |
| nora_1p5              | dual_loop_native    | int8+fp8      | Flow-matching action expert (FP-required head) |
| pi_0p5                | dual_loop_native    | int8+bf16     | PaliGemma + Gemma-300M flow-matching expert |
| bitvla                | single_loop         | int_only      | Ternary backbone — fully integer execution |

## Component schema

Each VLA is modeled as a composition of three components — vision
encoder + LLM backbone + action head — each carrying its own arithmetic
intensity, parameter count, and per-call FLOP estimate. Component
dtype_required encodes the QuantVLA finding that flow-matching action
expert heads need FP (BF16 or FP8) — INT8 quantization on the diffusion
head measurably breaks task success.

## Data source

`VLA_MODEL_DATA.csv` (May 2026) is the canonical source for:
  - per-model display_name, architecture, dtype paths, sliders bounds
  - component param counts (vlm_params_b + action_params_m split)
  - DRAM footprint at BF16/INT8/INT4
  - measured_5090_ms_per_action (5 of 6 cells; openvla_7b_cached is None)
  - LIBERO success rate (4 of 6 cells; pi_0.5 + openvla_cached have None)
  - arxiv_id + citation_year + source_paper

Schema fields NOT in the CSV (provided here as first-order estimates):
  - VLAComponent.flops_per_call_g — derived from 2 × params_b for matmul-
    dominated forwards; vision encoder estimates from typical SigLIP /
    DINOv2 / Qwen2.5-VL ViT GFLOP figures at 224×224 to 336×336 input.
    REFINE WHEN: Phase 3 projection lands and we calibrate against the
    measured RTX 5090 anchors.
  - VLAComponent.arithmetic_intensity — from the plan's published
    ranges (conv-heavy 50-200, transformer decode 2-5, diffusion DiT
    10-30); using the median of each range here.

The vision encoder params_b split is from architecture papers
(SigLIP-large 400M + DINOv2-large 300M ≈ 600M fused on OpenVLA;
Qwen2.5-VL ViT ≈ 500M; SigLIP-base ≈ 400M; PaliGemma-3B ViT ≈ 400M).
The LLM backbone params_b is vlm_params_b minus the vision share.

## Slider locking discipline

Per the plan's discipline table + the VLA_MODEL_DATA.csv slider bounds:

| Model              | vlm_hz_min == max == default? | Architecturally     |
|--------------------|-------------------------------|---------------------|
| openvla_7b_single  | yes (10/10/10/10)             | native autoregressive |
| openvla_7b_cached  | NO (1-10 / 15-60)             | synthetic dual-loop |
| nora_3b            | yes (30/30/30/30)             | native autoregressive |
| nora_1p5           | NO (1-5 / 20-60)              | native dual-loop    |
| pi_0p5             | NO (1-3 / 30-50)              | native dual-loop    |
| bitvla             | yes (30/30/30/30)             | native autoregressive |

UI surface (Phase 5) reads vlm_hz_min/max/default + action_hz_min/max/default
and disables the sliders when min == max == default. Source taxonomy
(Phase 4) flips badge to 🟠 when the user moves any slider off the
default — that part lives in app.py, not here.

## Tracks (narrative grouping for the sidebar picker)

`VLA_TRACKS` mirrors `PIPELINE_TRACKS` in app.py: a dict[track_key,
list[model_keys]] that groups the catalog by narrative. The 4 tracks
overlap intentionally (e.g. openvla_7b_single is in both
`autoregressive` and `integer_friendly`) — picker UI shows the model in
whichever track the user navigated through.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLAComponent:
    """One stage of a VLA inference pipeline.

    Three component types appear in the catalog:
      - 'vision_encoder' — typically conv-heavy (SigLIP / DINOv2 / ViT),
        high arithmetic intensity, one forward per VLM call
      - 'llm_backbone'   — transformer decode regime, low arithmetic
        intensity, per-call FLOPS scales with prompt + generated tokens
      - 'action_head'    — either a discrete-token classifier (reuses
        LLM lm_head, near-zero standalone cost) or a flow-matching /
        diffusion DiT (K denoising steps per action chunk, FP-required)
    """
    name: str                       # 'vision_encoder' | 'llm_backbone' | 'action_head'
    params_b: float                 # billions of parameters in this component
    flops_per_call_g: float         # GFLOPS per forward call — vision: per frame;
                                    # LLM: per VLM call (prompt + ~7 action tokens);
                                    # action: per action chunk (K denoising steps for FM)
    dtype_required: tuple[str, ...] # acceptable execution precisions; ('int8','fp8','bf16')
                                    # means EITHER works; ('bf16',) means FP-only
    arithmetic_intensity: float     # ops/byte — used by Phase 3 roofline math
                                    # to decide bw_bound vs compute_bound regime


@dataclass(frozen=True)
class VLAModel:
    """A full VLA pipeline — composition of vision + LLM + action components.

    Required fields cover the data the plan's Phase 3 projection function
    consumes. Optional fields carry catalog metadata (DRAM footprint,
    accuracy proxy, citation) for the UI tab + KPI export.
    """
    key: str
    display_name: str
    components: dict[str, VLAComponent]   # keyed by component name

    # Loop architecture per the plan's locked decisions
    architecture: str               # 'single_loop' | 'dual_loop_capable' | 'dual_loop_native'

    # Frequency-slider defaults + bounds. For inherently single-loop
    # models, vlm_hz_min == vlm_hz_max == default_vlm_hz (UI greys the
    # slider per Phase 5 discipline).
    default_vlm_hz: float
    default_action_hz: float
    vlm_hz_min: float
    vlm_hz_max: float
    action_hz_min: float
    action_hz_max: float

    # Provenance
    source_paper: str
    measured_5090_ms_per_action: float | None   # None = projection-only entry
                                                # When measured_5090_calibrated,
                                                # this is the END-TO-END p50 (prefill
                                                # + action-token decode + EOS) under
                                                # stock HF generate() — the honest
                                                # per-action latency, not the paper's
                                                # idealized prefill anchor.
    notes: str

    # ── Optional catalog metadata ──────────────────────────────────────
    # LIBERO success rate (placeholder accuracy proxy until a VLA-specific
    # benchmark suite lands — see Open issue #1 in the plan). None for
    # entries without a published LIBERO number.
    libero_success_pct: float | None = None

    # Inference DRAM footprint at the three weight-precision levels.
    # Source: VLA_MODEL_DATA.csv. None on entries without a published
    # number for that precision (e.g. bitvla is identical across widths
    # since it's already maximally compressed at ternary).
    inference_dram_gb_bf16: float | None = None
    inference_dram_gb_int8: float | None = None
    inference_dram_gb_int4: float | None = None

    # Citation
    arxiv_id: str = ""
    citation_year: int = 0

    # Default dtype path string (matches CSV's dtype_path_default).
    # Encoding: "int8" means uniform int8; "int8+fp8" means LLM int8 +
    # action head FP8; "int8+bf16" means LLM int8 + action head BF16;
    # "int_only" means strictly integer (ternary + INT8 acts).
    dtype_path_default: str = ""
    dtype_path_alt: str = ""

    # HuggingFace repo path for weight loading (consumed by
    # keyhole-backend's bakeoff_vla.py harness). Empty string means
    # "needs verification before running the bake-off" — keeps catalog-
    # driven data flow working even when the path hasn't been confirmed.
    # Populated where the path is publicly documented (OpenVLA); left
    # empty where the brief flagged verification still needed (NORA's
    # default of "declare-lab/nora" is set per the next-session brief
    # but treated as "verify on load" by the harness; π0.5, NORA-1.5,
    # BitVLA need backend lookup before populating).
    hf_repo: str = ""

    # True when measured RTX 5090 anchor has been calibrated against
    # this catalog entry's component-level FLOP estimates (i.e. the
    # flops_per_call_g / arithmetic_intensity values in the component
    # dataclasses have been refined from first-order estimates to
    # measurement-anchored values). Default False until backend's
    # bakeoff_vla.py harness lands the measurement and the sizer side
    # absorbs the calibration. Flipping this to True is the credibility
    # upgrade per the next-session brief — it gates whether the source
    # taxonomy shows 🟡 same_class (calibrated) or 🟠 cross_class
    # (uncalibrated projection).
    measured_5090_calibrated: bool = False

    # Measured RTX 5090 VLM-forward (prefill) p50 in ms — the calibrated
    # PREFILL component, distinct from measured_5090_ms_per_action (which is
    # the full end-to-end per-action latency when calibrated). This is the
    # number that reproduces a paper's published forward-pass anchor (e.g.
    # NORA's "33 ms / 30 Hz" matches the VLM forward, NOT the e2e). The gap
    # between prefill and e2e is action-token decode overhead under stock HF
    # generate() — optimization headroom (no CUDA graphs / static KV cache /
    # torch.compile), not measurement error. None until a 5090 bake-off lands.
    measured_5090_prefill_ms: float | None = None


# ───────────────────────────────────────────────────────────────────────
# Module-level constant per VLA entry. Convention mirrors llm_models.py:
# named constant → collected dict at end. Constants are easy to import
# individually if a script needs a specific entry.
#
# FLOP-per-call estimates are first-order (see module docstring). Vision
# encoder FLOPS use typical-resolution forward-pass figures from the
# respective architecture papers; LLM backbone FLOPS use 2 × params_b
# per token × ~7 action tokens per call (VLA action tokenization is
# typically 6-8 tokens for 6-DOF/7-DOF actions); action-head FLOPS for
# flow-matching use K=10 denoising steps × per-step matmul cost.
# ───────────────────────────────────────────────────────────────────────


# OpenVLA 7B — native autoregressive (Kim et al RSS 2024).
# Total ~7.5B = ~0.6B vision (SigLIP-large + DINOv2-large fused)
# + ~6.4B LLM (Llama-2 7B base) + 0 action expert (action tokens
# go through the LLM lm_head as discretized 7-DOF bins).
OPENVLA_7B_SINGLE = VLAModel(
    key="openvla_7b_single",
    display_name="OpenVLA 7B (single-loop autoregressive)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.6,                           # SigLIP-L 400M + DINOv2-L 300M fused
            flops_per_call_g=80.0,                  # fused conv-heavy forward at 224×224
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,             # conv-heavy median
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=6.4,                           # Llama-2 7B minus vision projection adapter
            flops_per_call_g=90.0,                  # ~7 action tokens × 2 × 6.4B
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,               # transformer decode median
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.0,                           # discrete token via LLM lm_head — no separate head
            flops_per_call_g=0.0,                   # cost is in llm_backbone
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,               # nominal — degenerate
        ),
    },
    architecture="single_loop",
    default_vlm_hz=10.0, default_action_hz=10.0,
    vlm_hz_min=10.0, vlm_hz_max=10.0,
    action_hz_min=10.0, action_hz_max=10.0,
    source_paper="Kim et al RSS 2024",
    measured_5090_ms_per_action=73.0,               # IndexBox 4090 measurement; 5090 estimate ~50 ms
    notes=(
        "Native architecture is autoregressive. Discretized 7-DOF actions "
        "through Llama 2 tokenizer (256 bins per dim). Each forward = vision + "
        "LLM decode for action tokens. No cached intent. Measured 73 ms/action "
        "on RTX 4090 per IndexBox; 5090 estimate ~50 ms."
    ),
    libero_success_pct=76.5,
    inference_dram_gb_bf16=15.0,
    inference_dram_gb_int8=7.5,
    inference_dram_gb_int4=3.75,
    arxiv_id="2406.09246",
    citation_year=2024,
    dtype_path_default="int8",
    dtype_path_alt="fp8",
    hf_repo="openvla/openvla-7b",                   # publicly documented
)


# OpenVLA 7B (cached dual-loop) — SYNTHETIC PROJECTION. No published
# artifact; sliders exposed for what-if exploration; source taxonomy
# always cross_class (🟠) for any cell from this entry per Phase 4.
OPENVLA_7B_CACHED = VLAModel(
    key="openvla_7b_cached",
    display_name="OpenVLA 7B (cached dual-loop projection)",
    components={
        # Same physical components as the single-loop variant — the
        # difference is purely scheduling (VLM at low Hz, replay cached
        # semantic embedding for action token generation at higher Hz).
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.6,
            flops_per_call_g=80.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=6.4,
            flops_per_call_g=90.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.0,
            flops_per_call_g=0.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
    },
    architecture="dual_loop_capable",
    default_vlm_hz=2.0, default_action_hz=30.0,
    vlm_hz_min=1.0, vlm_hz_max=10.0,
    action_hz_min=15.0, action_hz_max=60.0,
    source_paper="projection variant; no published artifact",
    measured_5090_ms_per_action=None,               # projection only
    notes=(
        "Synthetic projection: OpenVLA with hypothetical cache wrapper "
        "running VLM at 2 Hz and re-using semantic embedding for action "
        "token generation at 30 Hz. NOT a measured model — projection only. "
        "Source taxonomy always cross_class for this entry. Useful for "
        "what-if exploration of caching benefit."
    ),
    libero_success_pct=None,                        # no published number — projection-only
    inference_dram_gb_bf16=15.0,
    inference_dram_gb_int8=7.5,
    inference_dram_gb_int4=3.75,
    arxiv_id="2406.09246",
    citation_year=2026,                             # year of THIS projection variant
    dtype_path_default="int8",
    dtype_path_alt="fp8",
    hf_repo="openvla/openvla-7b",                   # same underlying weights as the single-loop variant
)


# NORA 3B — single-loop, edge-tuned (Hung et al arxiv Apr 2025).
# Compute-friendly baseline; 8.3 GB inference per paper. Total 3.0B =
# ~0.5B vision (Qwen2.5-VL ViT) + ~2.0B LLM + 0.5B action head.
NORA_3B = VLAModel(
    key="nora_3b",
    display_name="NORA 3B (single-loop)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.5,                           # Qwen2.5-VL ViT
            flops_per_call_g=50.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=2.0,                           # Qwen2.5-VL 3B base minus ViT
            flops_per_call_g=28.0,                  # ~7 action tokens × 2 × 2.0B
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.5,                           # FAST+ tokenizer head
            flops_per_call_g=30.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
    },
    architecture="single_loop",
    default_vlm_hz=30.0, default_action_hz=30.0,
    vlm_hz_min=30.0, vlm_hz_max=30.0,
    action_hz_min=30.0, action_hz_max=30.0,
    source_paper="Hung et al arxiv Apr 2025",
    # MEASURED on RTX 5090 (bf16 / sdpa, n=20, warmup=5, torch 2.11.0+cu130,
    # transformers 5.5.4) by [backend] 2026-05-29, harness on origin/main at
    # 27342e8. End-to-end p50 = 79.22 ms (p95 81.37) = prefill + 5 FAST+
    # action tokens + EOS @ 12.6 Hz. The paper's "33 ms / 30 Hz" reproduces
    # our VLM-FORWARD (prefill) p50 of 30.63 ms — see measured_5090_prefill_ms.
    # The prefill→e2e gap (~9.72 ms/token decode over 5 tokens, ~3× the bf16
    # BW floor) is optimization headroom under stock HF generate() (no CUDA
    # graphs / static KV cache / torch.compile), NOT measurement error.
    measured_5090_ms_per_action=79.22,
    measured_5090_prefill_ms=30.63,
    measured_5090_calibrated=True,
    notes=(
        "3B-parameter VLA on Qwen2.5-VL-3B backbone. Designed for real-time "
        "edge deployment. FAST+ tokenizer for action sequences. Single-loop "
        "autoregressive — no cached intent in the published model. Used as the "
        "compute-friendly baseline. MEASURED on RTX 5090: end-to-end 79.22 ms/"
        "action (p50) under stock HF generate(); VLM-forward prefill 30.63 ms "
        "reproduces the paper's 33 ms anchor (calibration confirmed at the "
        "prefill). Peak VRAM 7.13 GB (weights 7.11 GB) — implies ~3.56B real "
        "loaded params incl. the Qwen2.5-VL vision tower, above the 3.0B "
        "backbone figure; the paper's 8.3 GB likely includes context overhead."
    ),
    libero_success_pct=72.1,
    inference_dram_gb_bf16=6.0,
    inference_dram_gb_int8=3.0,
    inference_dram_gb_int4=1.5,
    arxiv_id="2504.19854",
    citation_year=2025,
    dtype_path_default="int8",
    dtype_path_alt="fp8",
    hf_repo="declare-lab/nora",                     # verified firsthand by [backend] 2026-05-29
)


# NORA-1.5 — dual-loop native, flow-matching action expert.
# Action expert REQUIRES FP per QuantVLA findings (INT8 on diffusion
# head breaks task success). Total 3.3B = 0.5B vision + 2.0B LLM + 0.8B
# flow-matching expert.
NORA_1P5 = VLAModel(
    key="nora_1p5",
    display_name="NORA-1.5 (flow-matching dual-loop)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.5,
            flops_per_call_g=50.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=2.0,
            flops_per_call_g=28.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.8,                           # flow-matching expert
            # 10 denoising steps × 2 × 0.8B per step = 16 GFLOPS chunk
            # cost; conservative 50 GFLOPS factoring per-step overhead.
            flops_per_call_g=50.0,
            # FP-only — INT8 quantization of diffusion head breaks task
            # success per QuantVLA findings (CSV notes).
            dtype_required=("fp8", "bf16"),
            arithmetic_intensity=20.0,              # diffusion DiT median
        ),
    },
    architecture="dual_loop_native",
    default_vlm_hz=3.0, default_action_hz=40.0,
    vlm_hz_min=1.0, vlm_hz_max=5.0,
    action_hz_min=20.0, action_hz_max=60.0,
    source_paper="same group arxiv Nov 2025",
    measured_5090_ms_per_action=None,
    notes=(
        "NORA + flow-matching action expert coupled via layer-wise "
        "self-attention. Action expert ~800M params. Dual-loop native: "
        "VLM at ~3 Hz, flow-matching action expert at 40 Hz. Action expert "
        "REQUIRES FP (BF16 or FP8) — INT8 quantization of diffusion head "
        "breaks task success per QuantVLA findings. INT8 path for VLM "
        "stages only."
    ),
    libero_success_pct=79.4,
    inference_dram_gb_bf16=7.5,
    inference_dram_gb_int8=3.75,
    inference_dram_gb_int4=1.85,
    arxiv_id="2511.14659",
    citation_year=2025,
    dtype_path_default="int8+fp8",
    dtype_path_alt="int8+bf16",
    hf_repo="declare-lab/nora-1.5",                 # verified by [backend] at 6577d99
)


# π0.5 — Physical Intelligence; dual-system PaliGemma + Gemma-300M
# flow-matching expert. Native bf16 per HF config. Action head requires FP.
PI_0P5 = VLAModel(
    key="pi_0p5",
    display_name="π0.5 (PaliGemma + Gemma action expert)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.4,                           # SigLIP-base inside PaliGemma
            flops_per_call_g=40.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=2.6,                           # Gemma-2B + PaliGemma overhead
            flops_per_call_g=36.0,                  # ~7 action tokens × 2 × 2.6B
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.3,                           # Gemma-300M flow-matching
            # 10 denoising steps × 2 × 0.3B per step ≈ 6 GFLOPS; conservative
            # 30 GFLOPS factoring per-step overhead + 10-action chunk.
            flops_per_call_g=30.0,
            dtype_required=("fp8", "bf16"),         # FP-required per HF config
            arithmetic_intensity=20.0,
        ),
    },
    architecture="dual_loop_native",
    default_vlm_hz=2.0, default_action_hz=50.0,
    vlm_hz_min=1.0, vlm_hz_max=3.0,
    action_hz_min=30.0, action_hz_max=50.0,
    source_paper="Physical Intelligence Apr 2025",
    measured_5090_ms_per_action=None,
    notes=(
        "VLM=PaliGemma-3B (frozen during inference), action expert=Gemma-300M "
        "flow-matching. 10-step denoising per action, 10 actions per chunk. "
        "Native bfloat16 per HF config (action_expert_variant=gemma_300m, "
        "dtype=bfloat16). Max 3 cameras; produces 50 Hz control output. "
        "Strong open-world generalization. Flow-matching head requires FP."
    ),
    libero_success_pct=None,                        # paper doesn't report LIBERO
    inference_dram_gb_bf16=6.6,
    inference_dram_gb_int8=3.3,
    inference_dram_gb_int4=1.65,
    arxiv_id="2504.16054",
    citation_year=2025,
    dtype_path_default="int8+bf16",
    dtype_path_alt="fp8+bf16",
    hf_repo="lerobot/pi0_5",                        # verified by [backend] at 6577d99 (LeRobot hosts, not physical-intelligence/)
)


# BitVLA — fully integer (1.58-bit ternary backbone + INT8 activations).
# Single-loop, no FP anywhere. Memory footprint identical across bit
# widths (already maximally compressed).
BITVLA = VLAModel(
    key="bitvla",
    display_name="BitVLA (ternary + INT8)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.4,                           # SigLIP
            flops_per_call_g=40.0,
            dtype_required=("int8",),               # SigLIP runs INT8 fine
            arithmetic_intensity=100.0,
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=2.0,                           # custom ternary backbone
            flops_per_call_g=28.0,
            dtype_required=("int8",),               # ternary weights, INT8 acts
            # Lower AI than standard transformer — ternary weights are
            # 0.2 bytes/param vs INT8's 1.0, so the same FLOPs move
            # fewer bytes through DRAM. Conservative estimate.
            arithmetic_intensity=5.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.5,
            flops_per_call_g=30.0,
            dtype_required=("int8",),
            arithmetic_intensity=3.0,
        ),
    },
    architecture="single_loop",
    default_vlm_hz=30.0, default_action_hz=30.0,
    vlm_hz_min=30.0, vlm_hz_max=30.0,
    action_hz_min=30.0, action_hz_max=30.0,
    source_paper="arxiv Mar 2026",
    measured_5090_ms_per_action=12.0,
    notes=(
        "Fully native 1.58-bit (ternary weights {-1,0,+1}) plus INT8 "
        "activations. NO FP required anywhere. Architecturally single-loop "
        "— no cache layer. Paper reports 4.4x speedup and 11x memory "
        "reduction vs full-precision. Memory footprint identical across "
        "bit widths (already maximally compressed). Closest to "
        "integer-only VLA published."
    ),
    libero_success_pct=68.0,
    inference_dram_gb_bf16=1.5,
    inference_dram_gb_int8=1.5,
    inference_dram_gb_int4=1.5,                     # identical across widths
    arxiv_id="2603.xxxx",                           # placeholder per CSV
    citation_year=2026,
    dtype_path_default="int_only",
    dtype_path_alt="int_only",
)


# ───────────────────────────────────────────────────────────────────────
# Collected catalog — order matters for sidebar fallback display when
# track is not selected. Single-loop measured models first (the
# headline-anchored entries), then dual-loop, with the synthetic-
# projection entry (openvla_7b_cached) sinking to the projection-only
# group.
# ───────────────────────────────────────────────────────────────────────
VLA_MODELS: dict[str, VLAModel] = {
    OPENVLA_7B_SINGLE.key: OPENVLA_7B_SINGLE,
    NORA_3B.key:           NORA_3B,
    BITVLA.key:            BITVLA,
    NORA_1P5.key:          NORA_1P5,
    PI_0P5.key:            PI_0P5,
    OPENVLA_7B_CACHED.key: OPENVLA_7B_CACHED,      # projection-only — last
}


# Narrative tracks for the sidebar picker. Tracks intentionally overlap
# (e.g. openvla_7b_single appears in both autoregressive and
# integer_friendly) — UI shows the model in whichever track the user
# navigated through. Mirrors PIPELINE_TRACKS convention in app.py.
VLA_TRACKS: dict[str, list[str]] = {
    "autoregressive":   [OPENVLA_7B_SINGLE.key, BITVLA.key],
    "dual_loop":        [OPENVLA_7B_CACHED.key, NORA_3B.key, NORA_1P5.key, PI_0P5.key],
    "integer_friendly": [BITVLA.key, OPENVLA_7B_SINGLE.key],
    "fp_required":      [NORA_1P5.key, PI_0P5.key],
}


# Startup invariant — every track-listed key must be a real VLA_MODELS
# entry. Loud-fail at import time (same discipline as PIPELINES /
# PIPELINE_TRACKS in app.py) so misregistration surfaces immediately
# instead of producing silent dropdown gaps.
_orphan_track_keys = {
    k for keys in VLA_TRACKS.values() for k in keys
} - set(VLA_MODELS.keys())
assert not _orphan_track_keys, (
    f"VLA_TRACKS references keys not in VLA_MODELS: {sorted(_orphan_track_keys)}"
)
