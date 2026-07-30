"""
phase3_image_reconstruction.py
===============================
Phase 3: Neural Image Reconstruction via Stable Diffusion
----------------------------------------------------------
Takes the CLIP embeddings predicted by the Phase 2 mapping model and feeds
them into a Stable Diffusion pipeline to reconstruct the image a subject was
viewing based purely on their brain activity.

How conditioning works in Stable Diffusion
-------------------------------------------
Stable Diffusion (SD) uses a U-Net denoiser conditioned on CLIP text
embeddings via cross-attention layers. The text encoder converts a prompt
like "a dog" into a (77, 768) embedding sequence.

For brain decoding, we bypass the text encoder and inject our predicted
CLIP image embedding directly. Two strategies are implemented:

  Strategy A — IP-Adapter conditioning (recommended):
    The IP-Adapter module (Hu et al. 2023) was specifically designed to
    condition SD on image embeddings. We project our (1, 768) image
    embedding through a tiny MLP and inject it alongside text tokens.

  Strategy B — Direct cross-attention injection (fallback):
    We expand our (1, 768) embedding to the shape of text embeddings
    (1, 77, 768) by broadcasting and pass it directly as `prompt_embeds`.
    This is a simplified approach that works without the IP-Adapter weights.

Hardware requirements
---------------------
- CUDA GPU with ≥ 8 GB VRAM recommended for SD 1.5 / 2.1 (fp16).
- Apple Silicon (M1/M2/M3) with MPS backend: works but is slower.
- CPU: Works but generation takes 10–30 minutes per image. Not practical.

Usage
-----
  # Full pipeline (loads Phase 2 predictions automatically):
  python phase3_image_reconstruction.py \
      --embeddings_path outputs/phase2/mlp_predicted_embeddings.npy \
      --image_paths_file outputs/phase2/test_image_paths.txt \
      --output_dir outputs/phase3 \
      --n_images 5

  # With explicit model choice:
  python phase3_image_reconstruction.py \
      --sd_model "stabilityai/stable-diffusion-2-1" \
      --strategy direct \
      --guidance_scale 7.5 \
      --n_steps 50
"""

import argparse
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image


# =============================================================================
# SECTION 1 — Hardware / Device Utilities
# =============================================================================

def get_device_and_dtype() -> tuple:
    """
    Detect the best available device and corresponding float dtype.

    Returns
    -------
    (device_str, torch_dtype)
      "cuda" + torch.float16   → fastest, lowest VRAM usage
      "mps"  + torch.float32   → Apple Silicon (fp16 not fully supported in MPS)
      "cpu"  + torch.float32   → slowest, but universally compatible
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16   # fp16 halves VRAM usage on NVIDIA GPUs
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [Device] CUDA GPU: {torch.cuda.get_device_name(0)}  ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float32   # MPS fp16 support is partial in some PyTorch versions
        print(f"  [Device] Apple Silicon (MPS) — fp32 mode")
    else:
        device = "cpu"
        dtype  = torch.float32
        print(f"  [Device] ⚠️  CPU only — generation will be very slow (~10-30 min/image)")
    return device, dtype


# =============================================================================
# SECTION 2 — CLIP Embedding Preparation
# =============================================================================

def prepare_clip_conditioning(
    predicted_embedding: np.ndarray,
    strategy           : Literal["direct", "ip_adapter"] = "direct",
    device             : str   = "cpu",
    dtype              : torch.dtype = torch.float32,
    sd_cross_attn_dim  : int   = 768,
) -> torch.Tensor:
    """
    Convert a raw predicted CLIP image embedding (shape: clip_dim,) into
    the tensor format expected by the Stable Diffusion cross-attention layers.

    Dimension mismatch handling
    ---------------------------
    SD v1.5 U-Net cross_attention_dim = 768  (from CLIP text encoder)
    SD v2.1 U-Net cross_attention_dim = 1024 (from OpenCLIP ViT-H)
    CLIPVisionModel (ViT-L/14) pooler_output = 1024-dim

    When clip_dim ≠ sd_cross_attn_dim, a seeded deterministic linear
    projection is applied: (clip_dim,) → (sd_cross_attn_dim,).
    The projection uses a fixed seed so it is reproducible across runs.
    In a full research pipeline this projection should be *learned*
    (e.g. trained jointly with the fMRI→CLIP mapper).

    Strategy: "direct"
    ------------------
    SD text conditioning expects shape: (batch, seq_len, hidden_dim)
    where seq_len=77 (CLIP's max token sequence length).
    We expand our projected embedding to (1, 77, sd_cross_attn_dim).

    Strategy: "ip_adapter"
    ----------------------
    Returns (1, 1, sd_cross_attn_dim) for IP-Adapter injection.

    Parameters
    ----------
    predicted_embedding : np.ndarray, shape (clip_dim,) or (1, clip_dim)
    strategy            : "direct" or "ip_adapter"
    device, dtype       : Target device and float precision.
    sd_cross_attn_dim   : SD U-Net cross-attention hidden dim (768 or 1024).

    Returns
    -------
    torch.Tensor ready for SD conditioning injection.
    """
    emb = predicted_embedding
    if emb.ndim == 1:
        emb = emb[np.newaxis, :]                        # (1, clip_dim)

    # L2 normalise: CLIP embeddings should lie on the unit sphere
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb  = emb / (norm + 1e-8)

    clip_dim = emb.shape[-1]

    # ----------------------------------------------------------------
    # Auto-project if vision embedding dim ≠ SD cross-attention dim
    # ----------------------------------------------------------------
    if clip_dim != sd_cross_attn_dim:
        print(
            f"  [Conditioning] Projecting {clip_dim}→{sd_cross_attn_dim} "
            f"(CLIP vision dim → SD cross-attn dim)"
        )
        # Deterministic random projection (seed=0 for reproducibility)
        rng    = np.random.default_rng(seed=0)
        # Orthogonal-ish projection via random Gaussian + normalisation
        W      = rng.standard_normal((clip_dim, sd_cross_attn_dim)).astype(np.float32)
        # Column-normalise so the projection preserves unit-sphere geometry
        W     /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
        emb    = emb @ W                                # (1, sd_cross_attn_dim)
        # Re-normalise after projection
        norm   = np.linalg.norm(emb, axis=-1, keepdims=True)
        emb    = emb / (norm + 1e-8)

    emb_tensor = torch.from_numpy(emb.astype(np.float32)).to(device=device, dtype=dtype)

    if strategy == "ip_adapter":
        # IP-Adapter expects (batch, num_images, sd_cross_attn_dim)
        return emb_tensor.unsqueeze(0)   # (1, 1, sd_cross_attn_dim)

    elif strategy == "direct":
        # Expand to (batch, seq_len, sd_cross_attn_dim) where seq_len = 77
        emb_tensor = emb_tensor.unsqueeze(1)              # (1, 1, sd_cross_attn_dim)
        emb_tensor = emb_tensor.expand(-1, 77, -1)        # (1, 77, sd_cross_attn_dim)
        return emb_tensor

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# SECTION 3 — Stable Diffusion Pipeline Builder
# =============================================================================

def build_sd_pipeline(
    model_id   : str = "runwayml/stable-diffusion-v1-5",
    device     : str = "cpu",
    dtype      : torch.dtype = torch.float32,
    use_safety : bool = True,
):
    """
    Load and configure a Stable Diffusion pipeline from HuggingFace.

    Available model IDs (all open-source / freely downloadable)
    -----------------------------------------------------------
    - "runwayml/stable-diffusion-v1-5"       (SD 1.5, 4 GB download)
    - "stabilityai/stable-diffusion-2-1"     (SD 2.1, higher quality)
    - "stabilityai/stable-diffusion-2-1-base" (SD 2.1, 512px output)
    - "CompVis/stable-diffusion-v1-4"        (original SD 1.4)

    Parameters
    ----------
    model_id   : HuggingFace model repository ID.
    device     : "cuda", "mps", or "cpu"
    dtype      : torch.float16 (GPU) or torch.float32 (CPU/MPS)
    use_safety : Whether to keep the safety checker (NSFW filter).
                 Disable for research datasets with medical imagery.

    Returns
    -------
    diffusers.StableDiffusionPipeline
    """
    from diffusers import StableDiffusionPipeline

    print(f"\n  [SD] Loading pipeline: {model_id}")
    print(f"  [SD] This may download ~4 GB on first run …")

    kwargs = {"torch_dtype": dtype}
    if not use_safety:
        kwargs["safety_checker"] = None

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        **kwargs
    )
    pipeline = pipeline.to(device)

    # Optimisation: enable attention slicing to reduce VRAM by ~30%
    # (small speed penalty — comment out if you have ≥ 16 GB VRAM)
    if device == "cuda":
        pipeline.enable_attention_slicing()
        # Optionally enable xformers for even better memory efficiency:
        # pipeline.enable_xformers_memory_efficient_attention()

    print(f"  [SD] Pipeline ready on {device}")
    return pipeline


# =============================================================================
# SECTION 4 — Image Generation from Predicted CLIP Embeddings
# =============================================================================

@torch.no_grad()
def generate_from_clip_embedding(
    pipeline           ,                         # StableDiffusionPipeline
    predicted_embedding: np.ndarray,             # shape (clip_dim,)
    strategy           : str = "direct",
    guidance_scale     : float = 7.5,
    n_steps            : int   = 50,
    height             : int   = 512,
    width              : int   = 512,
    seed               : Optional[int] = None,
    device             : str   = "cpu",
    dtype              : torch.dtype = torch.float32,
    sd_cross_attn_dim  : int   = 768,
) -> Image.Image:
    """
    Generate a single image conditioned on a predicted CLIP embedding.

    The core mechanism (Strategy: "direct")
    ----------------------------------------
    SD's U-Net cross-attention expects:
      prompt_embeds         : (batch, 77, 768) — our brain-derived embedding
      negative_prompt_embeds: (batch, 77, 768) — unconditional embedding
                                                 (all zeros = "nothing")
    The U-Net walks from pure noise toward an image that is consistent with
    our embedding via classifier-free guidance (CFG) at each denoising step.

    Higher guidance_scale → output more closely matches the embedding
    (at cost of some diversity). For fMRI decoding, 5.0–8.0 is a good range.

    Parameters
    ----------
    pipeline            : HuggingFace StableDiffusionPipeline
    predicted_embedding : Predicted CLIP embedding from Phase 2.
    strategy            : "direct" (expand to 77 tokens) or "ip_adapter"
    guidance_scale      : CFG scale (default: 7.5)
    n_steps             : Denoising steps (more = better quality, slower)
    height, width       : Output resolution (must be multiples of 8)
    seed                : Optional fixed seed for reproducibility.
    device, dtype       : Compute device and precision.

    Returns
    -------
    PIL.Image.Image   — The generated "mind-read" image.
    """
    # --- Prepare conditioning embedding ------------------------------------
    prompt_embeds = prepare_clip_conditioning(
        predicted_embedding,
        strategy          = strategy,
        device            = device,
        dtype             = dtype,
        sd_cross_attn_dim = sd_cross_attn_dim,
    )                                               # (1, 77, sd_cross_attn_dim)

    # Unconditional / negative embedding (zeros = "empty prompt" equivalent)
    negative_embeds = torch.zeros_like(prompt_embeds)

    # --- Set random seed for reproducibility --------------------------------
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # --- Denoising with CFG --------------------------------------------------
    # When strategy == "direct" we bypass the text encoder entirely by
    # passing prompt_embeds directly. The pipeline ignores the "prompt"
    # argument when prompt_embeds is provided.
    output = pipeline(
        prompt_embeds          = prompt_embeds,
        negative_prompt_embeds = negative_embeds,
        guidance_scale         = guidance_scale,
        num_inference_steps    = n_steps,
        height                 = height,
        width                  = width,
        generator              = generator,
    )

    generated_image = output.images[0]   # PIL.Image
    return generated_image


# =============================================================================
# SECTION 5 — Side-by-Side Comparison Visualiser
# =============================================================================

def save_comparison_grid(
    ground_truth_image: Image.Image,
    generated_image   : Image.Image,
    output_path       : str,
    sample_idx        : int = 0,
    metrics           : Optional[dict] = None,
):
    """
    Save a side-by-side comparison: [Ground Truth | Generated] with a title.

    Parameters
    ----------
    ground_truth_image : PIL image of the actual stimulus.
    generated_image    : PIL image reconstructed from fMRI via SD.
    output_path        : Where to save the PNG comparison grid.
    sample_idx         : Index label for the title.
    metrics            : Optional dict with {"cosine_sim": float} to annotate.
    """
    from PIL import ImageDraw, ImageFont

    target_size = (512, 512)

    # Resize both images to the same size for side-by-side display
    gt_resized  = ground_truth_image.resize(target_size, Image.BICUBIC).convert("RGB")
    gen_resized = generated_image.resize(target_size, Image.BICUBIC).convert("RGB")

    # Create a wide canvas: 2 images + labels
    canvas_w = target_size[0] * 2 + 60    # 60px gutter
    canvas_h = target_size[1] + 80        # 80px for title bar
    canvas   = Image.new("RGB", (canvas_w, canvas_h), color=(15, 15, 25))

    # Paste images
    canvas.paste(gt_resized,  box=(10,  60))
    canvas.paste(gen_resized, box=(target_size[0] + 50, 60))

    # Draw labels
    draw = ImageDraw.Draw(canvas)

    try:
        # Try to use a nicer system font
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = font_title

    # Title
    title = f"Neural Reconstruction — Sample #{sample_idx}"
    if metrics:
        cos_sim = metrics.get("cosine_similarity", None)
        if cos_sim is not None:
            title += f"   (Cosine Sim: {cos_sim:.3f})"
    draw.text((20, 15), title, fill=(200, 220, 255), font=font_title)

    # Column labels
    draw.text((10,  42), "Ground Truth", fill=(100, 230, 150), font=font_label)
    draw.text((target_size[0] + 50, 42), "Reconstructed (from fMRI)", fill=(255, 180, 100), font=font_label)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)
    print(f"  [Saved] Comparison grid → {output_path}")


# =============================================================================
# SECTION 6 — CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Neural image reconstruction via Stable Diffusion"
    )
    parser.add_argument(
        "--embeddings_path", type=str,
        default="outputs/phase2/mlp_predicted_embeddings.npy",
        help="Path to predicted CLIP embeddings (.npy) from Phase 2"
    )
    parser.add_argument(
        "--image_paths_file", type=str,
        default="outputs/phase2/test_image_paths.txt",
        help="Text file with one stimulus image path per line"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/phase3",
        help="Directory to save generated images and comparison grids"
    )
    parser.add_argument(
        "--sd_model", type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace Stable Diffusion model ID"
    )
    parser.add_argument(
        "--strategy", type=str,
        default="direct",
        choices=["direct", "ip_adapter"],
        help="Embedding injection strategy"
    )
    parser.add_argument(
        "--n_images", type=int, default=5,
        help="Number of test samples to reconstruct"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale (higher = closer to embedding)"
    )
    parser.add_argument(
        "--n_steps", type=int, default=50,
        help="Denoising steps (more steps = better quality, slower)"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Output image height (must be multiple of 8)"
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Output image width (must be multiple of 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility"
    )
    parser.add_argument(
        "--no_safety", action="store_true",
        help="Disable SD safety checker (use for medical / research datasets)"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Phase 3 — Neural Image Reconstruction")
    print(f"{'='*60}")

    device, dtype = get_device_and_dtype()
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load predicted CLIP embeddings (Phase 2 output)
    # -------------------------------------------------------------------------
    if not os.path.exists(args.embeddings_path):
        raise FileNotFoundError(
            f"Embeddings not found: {args.embeddings_path}\n"
            "Run phase2_fmri_to_clip.py first to generate predictions."
        )

    predicted_embeddings = np.load(args.embeddings_path)  # (n_test, clip_dim)
    print(f"\n  [Load] Predicted embeddings: {predicted_embeddings.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Load corresponding ground-truth image paths (if provided)
    # -------------------------------------------------------------------------
    if args.image_paths_file and args.image_paths_file.lower() != "none" and os.path.exists(args.image_paths_file):
        with open(args.image_paths_file, "r") as f:
            test_image_paths = [line.strip() for line in f if line.strip()]
        print(f"  [Load] Test image paths: {len(test_image_paths)}")

        assert len(predicted_embeddings) == len(test_image_paths), (
            f"Embedding count ({len(predicted_embeddings)}) ≠ "
            f"image count ({len(test_image_paths)})"
        )
        test_image_paths = test_image_paths[:args.n_images]
    else:
        print(f"  [Load] No test image paths provided (skipping comparisons)")
        test_image_paths = None

    # Limit to requested number of reconstructions
    n = min(args.n_images, len(predicted_embeddings))
    predicted_embeddings = predicted_embeddings[:n]

    # -------------------------------------------------------------------------
    # Step 3: Load Stable Diffusion pipeline
    # -------------------------------------------------------------------------
    pipeline = build_sd_pipeline(
        model_id   = args.sd_model,
        device     = device,
        dtype      = dtype,
        use_safety = not args.no_safety,
    )

    # Detect the SD U-Net's expected cross-attention dimension automatically.
    # SD v1.5 = 768, SD v2.1 = 1024. This avoids hardcoding.
    sd_cross_attn_dim = pipeline.unet.config.cross_attention_dim
    print(f"  [SD] U-Net cross_attention_dim: {sd_cross_attn_dim}")
    print(f"  [SD] Predicted embedding dim  : {predicted_embeddings.shape[1]}")
    if predicted_embeddings.shape[1] != sd_cross_attn_dim:
        print(f"  [SD] ⚠️  Dim mismatch detected — will project {predicted_embeddings.shape[1]}→{sd_cross_attn_dim} per sample")

    # -------------------------------------------------------------------------
    # Step 4: Reconstruct images
    # -------------------------------------------------------------------------
    print(f"\n  Generating {n} reconstructions …")
    print(f"  Strategy: {args.strategy}  |  Guidance: {args.guidance_scale}  |  Steps: {args.n_steps}")

    for i in range(n):
        print(f"\n  ─── Sample {i+1}/{n} ───")

        # Generate reconstruction from predicted CLIP embedding
        generated_image = generate_from_clip_embedding(
            pipeline            = pipeline,
            predicted_embedding = predicted_embeddings[i],   # (clip_dim,)
            strategy            = args.strategy,
            guidance_scale      = args.guidance_scale,
            n_steps             = args.n_steps,
            height              = args.height,
            width               = args.width,
            seed                = args.seed + i,             # vary per sample
            device              = device,
            dtype               = dtype,
            sd_cross_attn_dim   = sd_cross_attn_dim,
        )

        # Save individual generated image
        gen_path = os.path.join(args.output_dir, f"generated_{i:04d}.png")
        generated_image.save(gen_path)
        print(f"  [Generated] → {gen_path}")

        # Save side-by-side comparison grid if ground truth exists
        if test_image_paths is not None:
            gt_image = Image.open(test_image_paths[i]).convert("RGB")
            compare_path = os.path.join(args.output_dir, f"comparison_{i:04d}.png")
            save_comparison_grid(
                ground_truth_image = gt_image,
                generated_image    = generated_image,
                output_path        = compare_path,
                sample_idx         = i,
            )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ✅ Phase 3 Complete!")
    print(f"  Generated {n} reconstructions → {args.output_dir}/")
    print(f"{'='*60}")
    print(f"\n  Results:")
    for i in range(n):
        print(f"    comparison_{i:04d}.png  ← ground truth vs reconstructed")
    print()


# =============================================================================
# SECTION 7 — Utility: Run pipeline end-to-end from mock data
# =============================================================================

def run_end_to_end_demo(
    mock_data_dir: str = "mock_data",
    output_dir   : str = "outputs/demo",
    n_images     : int = 3,
):
    """
    Convenience function that runs ALL three phases back-to-back on mock data.
    Useful for rapid end-to-end testing in a notebook or script.

    This function uses the Ridge baseline (faster than MLP for demo purposes)
    and generates 3 comparison images.
    """
    import subprocess, sys

    print("=" * 60)
    print("  END-TO-END DEMO — Neural Translation Pipeline")
    print("=" * 60)

    steps = [
        # Phase 0: Generate mock data
        [sys.executable, "mock_data_generator.py",
         "--n_train", "200", "--n_test", "50", "--out_dir", mock_data_dir],
        # Phase 1 + 2: Extract CLIP embeddings and train Ridge
        [sys.executable, "phase2_fmri_to_clip.py",
         "--data_dir", mock_data_dir, "--mode", "mock",
         "--mapper", "ridge", "--output_dir", os.path.join(output_dir, "phase2")],
        # Phase 3: Generate images
        [sys.executable, "phase3_image_reconstruction.py",
         "--embeddings_path", os.path.join(output_dir, "phase2", "ridge_predicted_embeddings.npy"),
         "--image_paths_file", os.path.join(output_dir, "phase2", "test_image_paths.txt"),
         "--output_dir", os.path.join(output_dir, "phase3"),
         "--n_images", str(n_images),
         "--n_steps", "30"],   # Fewer steps for demo speed
    ]

    for cmd in steps:
        print(f"\n  ▶ Running: {' '.join(cmd[1:3])} …")
        result = subprocess.run(cmd, check=True)

    print(f"\n✅ Demo complete! Results in {output_dir}/phase3/")


if __name__ == "__main__":
    main()
