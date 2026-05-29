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

    # Measured RTX 5090 VLM-forward (prefill) p50 in ms — vision + LLM prefill
    # combined, distinct from measured_5090_ms_per_action (the full end-to-end
    # per-action latency when calibrated). For some models this reproduces a
    # published forward-pass anchor (e.g. NORA's "33 ms / 30 Hz" matches the
    # VLM forward, NOT the e2e). The gap between VLM-forward and e2e is
    # action-token decode under stock HF generate() — optimization headroom
    # (no CUDA graphs / static KV cache / torch.compile), not measurement
    # error. None until a 5090 bake-off lands.
    measured_5090_prefill_ms: float | None = None

    # Number of autoregressive action tokens decoded per action (single-loop
    # models). Drives e2e = vlm_forward + n_action_tokens × decode_ms/token.
    # NORA=5 (FAST+, measured), OpenVLA=7 (7-DOF discrete, measured), BitVLA=8
    # (pending verification). For dual-loop models this field is NOT the
    # binding metric (flow-matching emits action chunks via K denoising steps,
    # not AR tokens) — Phase 3b uses denoising-step semantics instead.
    n_action_tokens: int = 7

    # Measured RTX 5090 per-component latency split (when measured_5090_calibrated).
    # Keys: vision_ms, llm_prefill_ms, decode_ms_per_token. The two forward
    # components are compute-bound (vision scales by compute_util_factor,
    # llm_prefill by llm_prefill_util_factor — different ratios across silicon,
    # so they MUST be stored separately); decode is BW-bound. project_vla's
    # 🔵 calibrated path scales these measured LATENCIES (not FLOPs) by the
    # per-component compute/BW ratio to a target tier — footgun-free, since
    # the 5090 util cancels in the ratio. None until a 5090 bake-off lands.
    measured_5090_components: dict | None = None

    # PHYSICAL per-stage GFLOP (hardware-independent, 2·P·T matmul rule, real
    # params counted off the loaded model by [backend]). Keys: vision, prefill,
    # decode_per_tok, total. This is the un-corrupted FLOP truth that retires
    # the effective-FLOP-at-5090-util footgun — distinct from the per-component
    # flops_per_call_g (which conflate stages and were first-order/back-solved).
    # A future from-physics projection should consume THIS + measured_5090_util.
    physical_flops_g: dict | None = None

    # Measured RTX 5090 achieved util per stage = physical_flops / measured_p50
    # / 209 TFLOPS bf16 peak (the sizer's roofline value). Keys: vision, prefill,
    # decode. The real fraction-of-peak each stage hits — calibration gold:
    # batch-1 ViTs are NOT compute-saturated (~0.11-0.29, not the CNN-calibrated
    # 0.85), and decode ≈ 0.003-0.005 IS the bandwidth wall quantified (decode
    # does ~0.3% of peak compute — pure weight-streaming).
    measured_5090_util: dict | None = None


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
            params_b=0.731,                         # SigLIP+DINOv2 fused (real count, dd46b14)
            flops_per_call_g=374.0,                 # physical (2·P·T); was 80 first-order
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,             # compute-bound — AI non-binding
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=6.74,                          # Llama-2 7B body (real count, dd46b14)
            # physical prefill 3811 GF over the TRUE 280-token seqlen (24 text +
            # 256 vision injected as embeddings) — naive 2·P·T on the ~24 text
            # input_ids undercounts ~10×; [backend] hooks the real seqlen.
            flops_per_call_g=3811.0,                # physical prefill (decode per-tok in physical_flops_g)
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=1.0,               # bf16 decode ≈ 1.0 ops/byte; decode 13.74 GF/tok @ 0.5% util
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.131,                         # lm_head (real count); + 71M projector
            # NO standalone action FLOP: 7 discrete 256-bin tokens decode THROUGH
            # the LLM body + lm_head (counted in decode), not a separate head.
            flops_per_call_g=0.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=1.0,
        ),
    },
    architecture="single_loop",
    default_vlm_hz=10.0, default_action_hz=10.0,
    vlm_hz_min=10.0, vlm_hz_max=10.0,
    action_hz_min=10.0, action_hz_max=10.0,
    source_paper="Kim et al RSS 2024",
    # MEASURED on RTX 5090 (bf16/sdpa, n=20, transformers 4.40.1) by [backend]
    # 2026-05-29, keyhole b2e7397. End-to-end p50 = 126.50 ms (7.9 Hz); VLM-
    # forward 46.60 ms = vision 6.23 (13%) + LLM prefill 40.38 (87%); decode
    # 13.32 ms/token. Was 73.0 (an RTX 4090 IndexBox bench, not 5090).
    measured_5090_ms_per_action=126.50,
    measured_5090_prefill_ms=46.60,                 # VLM-forward (vision + LLM prefill)
    measured_5090_calibrated=True,
    n_action_tokens=7,                              # 7-DOF discrete, measured (reviewer's "8" superseded)
    measured_5090_components={
        "vision_ms": 6.23,
        "llm_prefill_ms": 40.38,
        "decode_ms_per_token": 13.32,
    },
    physical_flops_g={                              # [backend] dd46b14, 2·P·T, true 280-tok prefill seqlen
        "vision": 374.0,
        "prefill": 3811.0,
        "decode_per_tok": 13.74,
        "total": 4281.0,
    },
    measured_5090_util={                            # achieved fraction-of-peak on 5090 (209 TF bf16)
        "vision": 0.29,
        "prefill": 0.43,                            # 7B prefill over 280 tokens — most compute-bound stage
        "decode": 0.005,                            # bandwidth wall
    },
    notes=(
        "Native architecture is autoregressive. Discretized 7-DOF actions "
        "through Llama 2 tokenizer (256 bins per dim). Each forward = vision + "
        "LLM decode for action tokens. No cached intent. MEASURED on RTX 5090 "
        "stock HF: e2e 126.50 ms/action (7.9 Hz), VLM-forward 46.60 ms (vision "
        "6.23 + prefill 40.38), peak VRAM 14.41 GB (weights 14.09 = 7.5B "
        "confirmed). Note the cited 73 ms is an OPTIMIZED RTX 4090 IndexBox "
        "bench; our higher stock-HF 5090 number is optimization headroom + "
        "stack/GPU difference, not a slower GPU. ENV: OpenVLA runs correctly "
        "ONLY under transformers==4.40.1 — under >=4.57 it silently drops "
        "pixel_values (vision fires 0x, action image-invariant); requires a "
        "pinned venv. Component FLOPs left at first-order pending backend's "
        "analytical physical-FLOP attribution (the calibrated path uses the "
        "measured latencies above, not these FLOPs)."
    ),
    libero_success_pct=76.5,
    inference_dram_gb_bf16=15.0,
    inference_dram_gb_int8=7.5,
    inference_dram_gb_int4=3.75,
    arxiv_id="2406.09246",
    citation_year=2024,
    dtype_path_default="int8",
    dtype_path_alt="fp8",
    hf_repo="openvla/openvla-7b",                   # publicly documented; verified firsthand by [backend]
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
    n_action_tokens=7,                              # same weights as single-loop variant
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
# Compute-friendly baseline. Real params (counted off the loaded model by
# [backend]): vision 669M + llm_body 3.09B + lm_head 315M = 3.76B total
# (above the paper's 3.0B "backbone" figure).
#
# ── 5090 CALIBRATION (keyhole 49068ec latency split + dd46b14 physical FLOP) ──
# Per-stage wall-clock (forward-hook CUDA events, bf16/sdpa, n=20):
#   vision  13.83 ms | llm_prefill 16.11 ms | decode 9.72 ms/token (BW-bound)
# PHYSICAL GFLOP (2·P·T, hardware-independent) + measured 5090 achieved util:
#   vision  342 GF @ 11% util | prefill 581 GF @ 16% | decode 6.81 GF/tok @ 0.3%
# These live structurally in physical_flops_g + measured_5090_util below.
#
# FOOTGUN RESOLVED: an earlier back-solve stored EFFECTIVE GFLOP at an assumed
# 0.85 vision util (vision ~2458 GF) — an artifact, since a batch-1 ViT really
# hits ~11% util, so physical is 342 GF (~7× lower). The component
# flops_per_call_g below are now the PHYSICAL per-stage figures; treat
# physical_flops_g as authoritative. (These are not consumed for this model —
# it projects via the 🔵 calibrated latency-anchor path — but are now honest.)
NORA_3B = VLAModel(
    key="nora_3b",
    display_name="NORA 3B (single-loop)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.669,                         # Qwen2.5-VL ViT (real count, dd46b14)
            flops_per_call_g=342.0,                 # physical (was 2458 effective-@-0.85-util artifact)
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,             # compute-bound — AI non-binding
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=3.09,                          # Qwen2.5-VL 3B body (real count, dd46b14)
            flops_per_call_g=581.0,                 # physical prefill (decode is per-tok in physical_flops_g)
            dtype_required=("int8", "fp8", "bf16"),
            # Physical bf16 decode AI = 2·P FLOP/tok ÷ 2·P bytes/tok = 1.0 (any
            # bf16 decode is ~1.0 ops/byte; halves to 0.5 at int8). decode = 6.81
            # GF/tok @ 0.3% util — the BW wall (pure weight-streaming).
            arithmetic_intensity=1.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.315,                         # lm_head (tied to embeddings); real count
            # NO standalone action FLOP: NORA is single-loop autoregressive — FAST+
            # action tokens decode THROUGH the LLM body + lm_head (counted in the
            # decode term), not a separate head. Per [backend] dd46b14 (no double-count).
            flops_per_call_g=0.0,
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=1.0,
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
    n_action_tokens=5,                              # FAST+, measured (5 tokens + EOS)
    # Per-component split from keyhole 49068ec (run-2 VLM-forward 29.85 ms;
    # decode/token from the run-1 e2e that set measured_5090_ms_per_action).
    # ~1% run-to-run vs the 79.22 e2e; well within noise.
    measured_5090_components={
        "vision_ms": 13.83,
        "llm_prefill_ms": 16.11,
        "decode_ms_per_token": 9.72,
    },
    physical_flops_g={                              # [backend] dd46b14, 2·P·T, hardware-independent
        "vision": 342.0,
        "prefill": 581.0,
        "decode_per_tok": 6.81,
        "total": 965.0,                             # vision + prefill + 6 decode tokens
    },
    measured_5090_util={                            # achieved fraction-of-peak on 5090 (209 TF bf16)
        "vision": 0.11,
        "prefill": 0.16,
        "decode": 0.003,                            # the bandwidth wall, quantified
    },
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
    n_action_tokens=5,                              # placeholder — dual-loop emits chunks via denoising; Phase 3b semantics differ
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
    n_action_tokens=10,                             # placeholder — 10 actions/chunk via denoising; Phase 3b semantics differ
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
    measured_5090_ms_per_action=12.0,               # paper figure (not a firsthand 5090 measurement)
    n_action_tokens=8,                              # pending verification (reviewer estimate)
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
