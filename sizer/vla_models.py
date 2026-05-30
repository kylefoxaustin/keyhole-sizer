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
    #
    # For DUAL-LOOP models this also carries the fast-loop diagnosis:
    #   denoise_step (≈0.0007 = 0.07% peak FLOP), denoise_step_bw_util
    #   (≈0.016 = 1.6% peak BW), and denoise_bottleneck = "launch/overhead-bound".
    #   The denoise step is NEITHER compute- NOR bandwidth-bound — it is
    #   kernel-dispatch-bound in stock eager HF. This is WHY project_vla must NOT
    #   bandwidth-scale the measured fast-loop latency to edge (see project_vla's
    #   dual-loop branch): unlike single-loop AR decode (a hard BW-wall), the
    #   fast loop is silicon-independent in eager mode and has large optimization
    #   headroom (CUDA graphs / fusion / compile).
    measured_5090_util: dict | None = None

    # Measured RTX 5090 DUAL-LOOP topology anchor (when architecture is
    # dual_loop_* AND measured_5090_calibrated). The slow VLM backbone runs ONCE
    # per action chunk; a separate flow-matching action expert runs
    # `num_denoise_steps` denoise steps to emit an `action_chunk_length`-action
    # chunk. project_vla's dual-loop branch consumes this. Keys:
    #   vlm_backbone_ms          — slow-loop p50 (= measured_5090_components
    #                              vision_ms + llm_prefill_ms)
    #   denoise_step_ms          — fast-loop per-step p50 (LAUNCH-BOUND — see below)
    #   num_denoise_steps        — N denoise steps per chunk
    #   action_chunk_length      — H actions emitted per chunk
    #   chunk_latency_ms         — vlm_backbone + N×denoise_step
    #   amortized_ms_per_action  — chunk_latency / H (the honest control latency)
    #   control_hz_amortized     — 1000 / amortized_ms_per_action
    #   fast_loop_only_hz        — H / denoise_loop (VLM reused/pipelined — the
    #                              published "~40 Hz action expert" regime)
    #   denoise_bottleneck       — "launch/overhead-bound" (carried to the result
    #                              so the UI/deck can render the headroom caveat)
    # CRITICAL: denoise_step_ms is launch/dispatch-bound, NOT silicon-bound. The
    # dual-loop projection carries it UNCHANGED to edge as the eager ceiling
    # (launch-bound ⇒ silicon-independent), and computes an optimized physics
    # floor separately — it does NOT bandwidth-scale it (that footgun would
    # produce ~340 ms/step nonsense on edge). None until a dual-loop bake-off lands.
    measured_5090_dual_loop: dict | None = None

    # Measured RTX 5090 OFT PARALLEL-CHUNK anchor (when architecture is
    # "oft_parallel_chunk" AND measured_5090_calibrated). OpenVLA-OFT topology:
    # ONE VLM forward over [image + prompt + H action-placeholder tokens +
    # proprio] → an L1-regression head reads the action-position hidden states
    # in PARALLEL → H actions from a single forward. NO autoregressive token
    # loop, NO decode-per-token term, NO AR-decode BW-wall — the forward is
    # PREFILL-shaped (compute-bound). This is WHY OFT is fast (BitVLA 65 Hz vs
    # OpenVLA-7B AR 7.9 Hz). project_vla's oft branch consumes this. Keys:
    #   forward_ms          — full parallel forward p50 (= vision_ms + llm_forward_ms)
    #   vision_ms           — vision tower p50 (compute-bound, may be ×n_cameras)
    #   llm_forward_ms      — the single parallel LLM forward p50 (prefill-shaped)
    #   action_chunk_length — H actions emitted per forward
    #   ms_per_action       — forward_ms / H (the amortized control latency)
    #   action_hz           — 1000 / ms_per_action
    # The oft projection scales vision_ms + llm_forward_ms (both compute-bound,
    # latency-anchor scaled like single-loop's vision+prefill) and amortizes over
    # H — there is no decode term and no FP gate. None until an OFT bake-off lands.
    measured_5090_oft: dict | None = None

    # ── Phase 3c: multi-camera + fleet modeling ─────────────────────────────
    # How many camera feeds this VLA's architecture natively supports. π0.5 = 3,
    # BitVLA = 2 (LIBERO agentview + wrist), the rest = 1. project_vla rejects
    # n_cameras > this with runs:False (use fleet_size replication instead — a
    # 3-camera-stitched panorama on a 1-camera model is out-of-distribution).
    max_cameras_native: int = 1

    # The camera count the measured 5090 anchor was captured at — the BASELINE
    # for per-camera vision scaling. CRITICAL: for π0.5 (3) and BitVLA (2) the
    # stored vision_ms is ALREADY a multi-camera measurement, so per-camera cost
    # = measured vision_ms / measured_n_cameras (NOT / n_cameras). project_vla
    # defaults n_cameras to THIS value, so the default projection reproduces the
    # measured headline; other camera counts scale linearly and drop to 🟠.
    measured_n_cameras: int = 1

    # True when the LLM/action cost does NOT scale with camera count (only the
    # vision encoder fires once per camera; the LLM consumes the fused tokens
    # once). True for all natively-multi-camera VLAs measured so far; vacuous for
    # single-camera models (n always = 1). project_vla scales ONLY vision by
    # n_cameras when this holds.
    llm_backbone_invariant_to_n_cameras: bool = True


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
# head breaks task success).
#
# ── 5090 CALIBRATION (keyhole 4caf501 dual-loop measurement + physical FLOP) ──
# Same Qwen2.5-VL-3B backbone as NORA-3B + a separate flow-matching expert.
# Real params (counted off the loaded model by [backend], measured total 3.99B
# — the CSV's 3.3B and 800M action-expert figure both OVER/under-count):
#   vision 668.7M | llm_body 3089.6M | lm_head 314.8M | action_expert 228.4M
# Two corrections vs the original first-order entry: action expert 800M→228.4M
# (echoes the NORA-3B vision 2458→342 over-count footgun); total 3.3B→3.99B.
#
# DUAL-LOOP topology (vla_summary_nora_1p5.json): the VLM backbone runs ONCE per
# chunk (26.5 ms = vision 13.34 / prefill 13.22) → frozen KV cache; the action
# expert then runs N=10 flow-matching denoise steps (15.78 ms/step → 156.2 ms
# loop) to emit a 5-action chunk. chunk 182.8 ms → amortized 36.6 ms/action =
# 27.4 Hz; fast-loop-only 32.0 Hz (the published "~40 Hz expert" regime).
#
# ⚠️ FAST LOOP IS NOT BW-WALLED (the categorical difference from single-loop):
# the denoise step is neither compute-bound (0.07% peak FLOP) NOR bandwidth-bound
# (1.6% peak BW, eff 29 GB/s) — it is KERNEL-LAUNCH/DISPATCH-bound in stock eager
# HF (36 tiny expert layers, custom-Python MoT attention over 5 tokens). It has
# large optimization headroom (CUDA graphs / fusion / compile). project_vla's
# dual-loop branch carries it UNCHANGED to edge (launch-bound ⇒ silicon-
# independent) as the eager ceiling and computes a separate physics floor — it
# does NOT bandwidth-scale it. See measured_5090_dual_loop + measured_5090_util.
NORA_1P5 = VLAModel(
    key="nora_1p5",
    display_name="NORA-1.5 (flow-matching dual-loop)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.669,                         # Qwen2.5-VL ViT (real count, 4caf501)
            flops_per_call_g=342.37,                # physical (2·P·T)
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,             # compute-bound — AI non-binding
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=3.09,                          # Qwen2.5-VL 3B body (real count, 4caf501)
            flops_per_call_g=640.02,                # physical prefill over 94-tok seqlen
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.2284,                        # flow-matching expert (real 228.4M; was 800M over-count)
            # 10 denoise steps × 2.28 GF/step = 22.84 GF/chunk (physical, 2·P·T
            # matmul lower bound — cross-attention over the VLM KV omitted).
            flops_per_call_g=22.84,
            # FP-only — INT8 quantization of diffusion head breaks task
            # success per QuantVLA findings. A HARD GATE: eliminates INT8-only
            # silicon (e.g. NPU Mid) entirely.
            dtype_required=("fp8", "bf16"),
            arithmetic_intensity=20.0,              # diffusion DiT median
        ),
    },
    architecture="dual_loop_native",
    default_vlm_hz=3.0, default_action_hz=40.0,
    vlm_hz_min=1.0, vlm_hz_max=5.0,
    action_hz_min=20.0, action_hz_max=60.0,
    source_paper="same group arxiv Nov 2025",
    # Dual-loop has no single AR "per-action" measurement; the binding headline
    # is the amortized chunk metric in measured_5090_dual_loop. Kept None here
    # (this field is the single-loop AR e2e); set measured_5090_calibrated so the
    # dual-loop projection path activates.
    measured_5090_ms_per_action=None,
    measured_5090_calibrated=True,
    n_action_tokens=5,                              # = action_chunk_length H; dual-loop emits chunks via denoising, not AR tokens
    # VLM-backbone per-stage split (slow loop) — consumed by the dual-loop 🔵
    # calibrated path exactly like single-loop's vision+prefill (compute-bound,
    # latency-anchor scaled). No decode_ms_per_token: dual-loop has no AR decode.
    measured_5090_components={
        "vision_ms": 13.339,
        "llm_prefill_ms": 13.217,
    },
    measured_5090_dual_loop={                       # [backend] 4caf501, RTX 5090 bf16 n=20
        "vlm_backbone_ms": 26.513,
        "denoise_step_ms": 15.775,                  # LAUNCH-BOUND — carried unchanged to edge, never BW-scaled
        "num_denoise_steps": 10,
        "action_chunk_length": 5,
        "chunk_latency_ms": 182.757,
        "amortized_ms_per_action": 36.551,
        "control_hz_amortized": 27.36,
        "fast_loop_only_hz": 32.0,
        "denoise_bottleneck": "launch/overhead-bound",
        "denoise_step_effective_bw_gbs": 29.0,      # 1.6% peak BW — fast loop is mostly launch, tiny BW fraction
    },
    physical_flops_g={                              # [backend] 4caf501, 2·P·T, hardware-independent
        "vision": 342.37,
        "prefill": 640.02,
        "denoise_step": 2.2838,
        "denoise_total": 22.838,                    # N=10 steps
        "action_chunk_total": 1005.23,             # vlm backbone + denoise loop
        "per_action": 201.046,                      # action_chunk_total / H=5
    },
    measured_5090_util={                            # achieved fraction-of-peak on 5090 (209 TF bf16 / 1792 GB/s)
        "vision": 0.1228,
        "prefill": 0.2317,
        "denoise_step": 0.0007,                     # 0.07% peak FLOP — NOT compute-bound
        "denoise_step_bw_util": 0.0162,             # 1.6% peak BW (eff 29 GB/s) — NOT bandwidth-bound
        "denoise_bottleneck": "launch/overhead-bound",
    },
    notes=(
        "NORA + flow-matching action expert coupled via layer-wise "
        "self-attention. Same Qwen2.5-VL-3B backbone as NORA-3B + a separate "
        "228M flow-matching expert (CSV's 800M over-counts). Dual-loop native: "
        "VLM backbone once/chunk (26.5 ms), action expert 10 denoise steps "
        "(15.8 ms/step) → 5-action chunk. MEASURED on RTX 5090 (bf16, n=20): "
        "amortized 36.6 ms/action = 27.4 Hz; fast-loop-only 32.0 Hz (the "
        "published ~40 Hz expert regime). Peak VRAM 7.62 GB. Action expert "
        "REQUIRES FP (BF16 or FP8) — INT8 of the diffusion head breaks task "
        "success per QuantVLA findings; a HARD GATE that eliminates INT8-only "
        "silicon (NPU Mid won't run it). CRITICAL: the denoise step is "
        "launch/dispatch-bound (0.07% FLOP, 1.6% BW), NOT bandwidth-walled like "
        "single-loop AR decode — large optimization headroom, so edge projection "
        "carries it unchanged (eager ceiling) + a physics floor, never BW-scaled. "
        "ENV: runs under transformers==4.54.1 (pinned venv — the MoT attention "
        "reads the legacy tuple KV cache newer transformers broke)."
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


# π0.5 — Physical Intelligence; dual-system PaliGemma (gemma_2b) + Gemma expert
# flow-matching. The AMORTIZATION EXTREME of the catalog: one VLM forward feeds a
# 50-action chunk → 367 Hz amortized (vs NORA-1.5's chunk of 5 → 27 Hz). Action
# head requires FP. Native bf16-AMP (float32 master weights) per the lerobot path.
#
# ── 5090 CALIBRATION (keyhole 0fa4474 dual-loop measurement + physical FLOP) ──
# Real params (counted off the loaded model by [backend]): vision 412.4M +
# llm_body 2508.5M + action_expert 430.4M = 4143.4M total (CSV's expert 300M
# under-counts; measured ~430M). Same dual-loop topology as NORA-1.5: PaliGemma
# VLM prefix runs ONCE over 915 tokens (3 cameras + long prompt) → KV cache, then
# the Gemma expert runs N=10 denoise steps → 50-action chunk.
#
# 5090 (bf16-AMP, n=20): VLM backbone 52.8 ms (SigLIP vision 14.6 ×3 cameras +
# gemma_2b prefill 38.2) + denoise 7.2 ms/step ×10 → 73.3 ms loop → 136 ms chunk
# → amortized 2.72 ms/action = 367 Hz; fast-loop-only 682 Hz. Peak VRAM 20.9 GB.
#
# ⚠️ DENOISE BOTTLENECK DIFFERS FROM NORA-1.5 — data-driven, do NOT reuse the
# launch-bound label: π0.5's denoise is "mixed (partial-BW + launch overhead)"
# — compute 2.9% / BW 13.4% (eff 238 GB/s). The 430M expert over 50 tokens
# streams REAL weight bytes, so on a lower-BW edge part π0.5's fast loop degrades
# MORE than NORA-1.5's. project_vla's launch+BW decomposition handles this from
# the per-model denoise_step_effective_bw_gbs (no string-branch on the label).
# 🔸 bf16-AMP keeps float32 (4-byte) master weights → the 13.4% BW util is an
# UPPER bound; a true bf16-weight deploy ~halves it (→ more launch-leaning). AMP
# is used because π0.5's expert hardcodes float32 time-embedding internals.
PI_0P5 = VLAModel(
    key="pi_0p5",
    display_name="π0.5 (PaliGemma + Gemma action expert)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.4124,                        # SigLIP inside PaliGemma (real count, 0fa4474)
            flops_per_call_g=633.51,                # physical, ×3 cameras
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=100.0,
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=2.5085,                        # gemma_2b body (real count, 0fa4474)
            flops_per_call_g=4590.61,               # physical prefill over 915-token seqlen (most compute-bound stage)
            dtype_required=("int8", "fp8", "bf16"),
            arithmetic_intensity=3.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.4304,                        # Gemma flow-matching expert (real 430M; CSV's 300M under-counts)
            flops_per_call_g=430.36,                # physical denoise total (N=10 steps over 50-action chunk)
            dtype_required=("fp8", "bf16"),         # FP-required — hard gate on INT8-only silicon
            arithmetic_intensity=20.0,
        ),
    },
    architecture="dual_loop_native",
    default_vlm_hz=2.0, default_action_hz=50.0,
    vlm_hz_min=1.0, vlm_hz_max=3.0,
    action_hz_min=30.0, action_hz_max=50.0,
    source_paper="Physical Intelligence Apr 2025",
    measured_5090_ms_per_action=None,               # dual-loop headline is the amortized chunk metric
    measured_5090_calibrated=True,
    n_action_tokens=50,                             # = action_chunk_length H (50 actions/chunk via denoising)
    measured_5090_components={                      # VLM-backbone split (slow loop)
        "vision_ms": 14.563,                        # SigLIP ×3 cameras
        "llm_prefill_ms": 38.244,                   # gemma_2b over 915 tokens
    },
    measured_5090_dual_loop={                       # [backend] 0fa4474, RTX 5090 bf16-AMP n=20
        "vlm_backbone_ms": 52.807,
        "denoise_step_ms": 7.217,
        "num_denoise_steps": 10,
        "action_chunk_length": 50,
        "chunk_latency_ms": 136.161,
        "amortized_ms_per_action": 2.723,
        "control_hz_amortized": 367.21,
        "fast_loop_only_hz": 681.73,
        "denoise_bottleneck": "mixed (partial-BW + launch overhead)",
        "denoise_step_effective_bw_gbs": 238.5,     # 13.4% peak BW — real weight traffic, scales on edge
    },
    physical_flops_g={                              # [backend] 0fa4474, 2·P·T, hardware-independent
        "vision": 633.51,
        "prefill": 4590.61,
        "denoise_step": 43.0361,
        "denoise_total": 430.361,                   # N=10 steps
        "action_chunk_total": 5654.49,
        "per_action": 113.09,                       # action_chunk_total / H=50
    },
    measured_5090_util={                            # achieved fraction-of-peak on 5090 (209 TF bf16 / 1792 GB/s)
        "vision": 0.2081,
        "prefill": 0.5743,                          # gemma_2b over 915 tokens — MOST compute-bound stage in the bake-off
        "denoise_step": 0.0285,                     # 2.9% peak FLOP
        "denoise_step_bw_util": 0.1331,             # 13.4% peak BW (eff 238 GB/s) — partial-BW, NOT pure launch
        "denoise_bottleneck": "mixed (partial-BW + launch overhead)",
        "weight_bytes_per_param": 4.0,              # bf16-AMP float32 master weights → BW util is an UPPER bound
    },
    notes=(
        "VLM=PaliGemma (gemma_2b, frozen during inference), action expert=Gemma "
        "flow-matching (~430M). 10-step denoising, 50 actions per chunk, 3 "
        "cameras, action_dim 32. THE AMORTIZATION EXTREME: one VLM forward over "
        "a 50-action chunk → MEASURED 367 Hz amortized on RTX 5090 (chunk 136 ms; "
        "fast-loop-only 682 Hz) — vs NORA-1.5's chunk of 5 → 27 Hz. Same chunk "
        "latency class (~135 vs 183 ms) but 10× the actions per VLM forward; chunk "
        "size is the amortization knob. Flow-matching head REQUIRES FP (bf16/fp8) "
        "— hard gate eliminates INT8-only silicon. Denoise is 'mixed (partial-BW "
        "+ launch overhead)' (13.4% BW), NOT pure launch-bound like NORA-1.5 — so "
        "it degrades more on low-BW edge; the projection scales the BW fraction "
        "per-model. bf16-AMP (float32 master weights) → measured BW util is an "
        "upper bound. ENV: runs ONLY in a pinned venv (Python 3.12, lerobot 0.5.2 "
        "from git main — PyPI 0.4.4 hard-gates pi05 on unshipped patched-transformers)."
    ),
    libero_success_pct=None,                        # paper doesn't report LIBERO
    # Footprint estimates from real 4.14B params: bf16 ≈ 2 B/param. Measured peak
    # VRAM 20.9 GB (weights 15.45 GB) is the bf16-AMP float32-master figure — an
    # upper bound; a true bf16-weight deploy is ~half.
    inference_dram_gb_bf16=8.3,
    inference_dram_gb_int8=4.15,
    inference_dram_gb_int4=2.07,
    arxiv_id="2504.16054",
    citation_year=2025,
    dtype_path_default="int8+bf16",
    dtype_path_alt="fp8+bf16",
    hf_repo="lerobot/pi05_base",                    # [backend] 0fa4474 fixed lerobot/pi0_5 (404) → lerobot/pi05_base
    # Multi-camera (Phase 3c): π0.5 natively supports up to 3 cameras; the 5090
    # measurement was captured at 3 (vision_ms 14.563 = SigLIP ×3 cams). So 3 is
    # the calibrated default; n_cameras 1/2 are linear down-scales (🟠).
    max_cameras_native=3,
    measured_n_cameras=3,
)


# BitVLA — ternary backbone, OFT PARALLEL-CHUNK (NOT single-loop AR).
#
# ── 5090 CALIBRATION (keyhole 3317776) — TWO corrections to the original entry ──
# 1. ARCHITECTURE: the CSV said single_loop; the actual LIBERO checkpoint is
#    OpenVLA-OFT. ONE VLM forward over [image×2 + prompt + 8 action-placeholder
#    tokens + proprio] → an L1-regression head reads the action-position hidden
#    states in PARALLEL → 8 actions from a single forward. No AR token loop, no
#    decode-per-token, no AR-decode BW-wall — the forward is prefill-shaped
#    (compute-bound, util 9-14%). This is WHY it's fast: 65 Hz vs OpenVLA-7B 7.9.
# 2. TERNARY IS A MEMORY WIN ONLY (in this measured path): the HF "bf16"
#    checkpoint runs ternary BitLinear as DENSE bf16 matmuls (weights stored
#    bf16, 5.47 GB — NOT packed 1.58-bit/0.2-byte). Ternary buys nothing on
#    compute/latency here; only the ~6 GB VRAM is the realized win. The paper's
#    4.4× speedup / 0.2-byte decode-BW story REQUIRES bitblas/LUT ternary kernels
#    NOT used here — treat that as a SEPARATE optimistic kernel-dependent floor
#    (it assumes kernels we haven't measured), never the measured headline.
#
# Real params (counted off the loaded model, 3317776): vision 397.2M + ternary-
# LLM 2412.8M = 2819.5M total (above the paper's nominal figure). bf16, n=20.
BITVLA = VLAModel(
    key="bitvla",
    display_name="BitVLA (ternary, OFT parallel-chunk)",
    components={
        "vision_encoder": VLAComponent(
            name="vision_encoder",
            params_b=0.3972,                        # bitSigLIP-L (real count, 3317776)
            flops_per_call_g=406.73,                # physical, ×2 cameras
            dtype_required=("int8",),               # SigLIP runs INT8 fine
            arithmetic_intensity=100.0,             # compute-bound — AI non-binding
        ),
        "llm_backbone": VLAComponent(
            name="llm_backbone",
            params_b=2.4128,                         # ternary backbone (real count, 3317776)
            flops_per_call_g=2895.41,               # physical parallel forward (prefill-shaped, 600-tok prefix)
            dtype_required=("int8",),               # ternary weights, INT8 acts
            # Ternary weights are nominally 0.2 byte/param vs INT8's 1.0 — but
            # the MEASURED bf16-dense path does NOT realize this (weights stored
            # bf16). The 0.2-byte / low-AI story is the optimistic kernel-dependent
            # floor (bitblas/LUT), not the measured number. AI here is non-binding
            # anyway: OFT forward is compute-bound, not decode-BW-bound.
            arithmetic_intensity=5.0,
        ),
        "action_head": VLAComponent(
            name="action_head",
            params_b=0.0,                           # L1-regression head reads action-position
            flops_per_call_g=0.0,                   # hidden states (in the forward) — no standalone cost
            dtype_required=("int8",),
            arithmetic_intensity=3.0,
        ),
    },
    architecture="oft_parallel_chunk",
    default_vlm_hz=65.0, default_action_hz=65.0,
    vlm_hz_min=65.0, vlm_hz_max=65.0,
    action_hz_min=65.0, action_hz_max=65.0,
    source_paper="arxiv Mar 2026",
    measured_5090_ms_per_action=15.384,             # = forward 123.071 / 8 actions (amortized)
    measured_5090_prefill_ms=123.071,               # the full OFT parallel forward (vision + LLM)
    measured_5090_calibrated=True,
    n_action_tokens=8,                              # = action_chunk_length H (8 actions / parallel forward)
    measured_5090_components={                      # OFT forward split — NO decode_ms_per_token (no AR loop)
        "vision_ms": 22.558,                        # bitSigLIP ×2 cameras
        "llm_forward_ms": 100.513,                  # single parallel forward, prefill-shaped
    },
    measured_5090_oft={                             # [backend] 3317776, RTX 5090 bf16 n=20
        "forward_ms": 123.071,
        "vision_ms": 22.558,
        "llm_forward_ms": 100.513,
        "action_chunk_length": 8,
        "ms_per_action": 15.384,
        "action_hz": 65.0,
    },
    physical_flops_g={                              # [backend] 3317776, 2·P·T, hardware-independent
        "vision": 406.73,
        "llm_forward": 2895.41,                     # parallel forward (no decode_per_tok — no AR loop)
        "action_forward_total": 3302.14,
        "per_action": 412.77,                       # action_forward_total / H=8
    },
    measured_5090_util={                            # achieved fraction-of-peak on 5090 (209 TF bf16)
        "vision": 0.0863,
        "llm_forward": 0.1378,                      # prefill-shaped, compute-bound — NOT a decode BW-wall
    },
    notes=(
        "Ternary backbone (1.58-bit weights {-1,0,+1}), bitSigLIP-L vision. "
        "OFT PARALLEL-CHUNK (OpenVLA-OFT), NOT single-loop AR: ONE VLM forward "
        "over [image×2 + prompt + 8 action tokens + proprio] → L1-regression head "
        "reads action-position hiddens in PARALLEL → 8 actions/forward. No AR "
        "loop, no decode-per-token, no BW-wall — prefill-shaped compute-bound "
        "forward (util 9-14%). MEASURED on RTX 5090 (bf16, n=20): forward 123 ms "
        "→ 15.4 ms/action = 65 Hz (vs OpenVLA-7B AR 7.9 Hz — the OFT speed story). "
        "Peak VRAM 6.07 GB (weights 5.47) vs OpenVLA-7B 14.4 GB. ⚠️ TERNARY = "
        "MEMORY WIN ONLY here: the HF bf16 checkpoint runs ternary BitLinear as "
        "DENSE bf16 matmuls (weights stored bf16, NOT packed 0.2-byte), so the "
        "4.4× speedup / 0.2-byte decode-BW story is NOT realized — it requires "
        "bitblas/LUT kernels not used in this path. Treat that as a SEPARATE "
        "optimistic kernel-dependent floor, not this measured number. The only "
        "int_only catalog entry → runs on INT8-only silicon (NPU Mid) with NO FP "
        "gate, unlike the dual-loop FP-required heads."
    ),
    libero_success_pct=68.0,
    # bf16-stored weights 5.47 GB / peak VRAM 6.07 GB (measured). NOT the packed-
    # ternary ~1.4 GB figure (that needs bitblas kernels, not this path).
    inference_dram_gb_bf16=6.07,
    inference_dram_gb_int8=3.0,                     # if int8-stored; packed-ternary would be ~1.4 (kernel-dependent)
    inference_dram_gb_int4=1.5,
    arxiv_id="2603.xxxx",                           # placeholder per CSV
    citation_year=2026,
    dtype_path_default="int_only",
    dtype_path_alt="int_only",
    hf_repo="hongyuw/ft-bitvla-bitsiglipL-224px-libero_goal-bf16",  # LIBERO-goal FT checkpoint ([backend] 3317776)
    # Multi-camera (Phase 3c): the LIBERO checkpoint takes 2 cameras (agentview +
    # wrist); the 5090 measurement was captured at 2 (vision_ms 22.558 = ×2 cams).
    # So 2 is the calibrated default; n_cameras=1 is a linear down-scale (🟠).
    max_cameras_native=2,
    measured_n_cameras=2,
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
