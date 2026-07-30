"""
mock_data_generator.py
======================
Generates synthetic fMRI voxel data and paired "stimulus" images so you can
run and validate the full pipeline locally WITHOUT downloading the massive
GOD (~50 GB) or NSD (~700 GB) datasets.

What this script produces
--------------------------
mock_data/
  fmri/
    train/  sub-01_sess-01_run-XXXX.npy   <- (n_train, n_voxels) voxel arrays
    test/   sub-01_sess-01_run-XXXX.npy   <- (n_test,  n_voxels) voxel arrays
  images/
    train/  XXXX.png    <- 224x224 RGB synthetic "stimulus" images
    test/   XXXX.png

Usage
-----
  python mock_data_generator.py [--n_train 200] [--n_test 50] [--n_voxels 4000]

"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fMRI + stimulus image pairs for pipeline testing."
    )
    parser.add_argument(
        "--n_train", type=int, default=200,
        help="Number of synthetic training samples (default: 200)"
    )
    parser.add_argument(
        "--n_test", type=int, default=50,
        help="Number of synthetic test samples (default: 50)"
    )
    parser.add_argument(
        "--n_voxels", type=int, default=4096,
        help="Number of ROI voxels to simulate (default: 4096, ~V1-V4 ROI size)"
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Square image size in pixels (default: 224, matches CLIP input)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="mock_data",
        help="Root output directory (default: mock_data/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic Image Generator
# ---------------------------------------------------------------------------

# Simple colour palette so images are visually distinct
COLOURS = [
    (220, 50,  50),   # Red
    (50,  150, 220),  # Blue
    (50,  200, 80),   # Green
    (250, 200, 30),   # Yellow
    (180, 80,  220),  # Purple
    (240, 130, 40),   # Orange
    (60,  220, 220),  # Cyan
    (240, 80,  160),  # Pink
]

SHAPES = ["circle", "rectangle", "triangle", "ellipse"]


def generate_synthetic_image(img_size: int, seed: int) -> Image.Image:
    """
    Create a reproducible synthetic RGB image with random coloured geometric
    shapes. In the real pipeline these would be replaced by the actual
    ImageNet / NSD stimulus images.

    Parameters
    ----------
    img_size : int   Width and height of the square output image.
    seed     : int   Controls colours / shape placement for reproducibility.

    Returns
    -------
    PIL.Image.Image   RGB image of shape (img_size, img_size).
    """
    rng = random.Random(seed)

    # Dark background with a random tint
    bg_colour = (rng.randint(5, 40), rng.randint(5, 40), rng.randint(5, 40))
    img = Image.new("RGB", (img_size, img_size), color=bg_colour)
    draw = ImageDraw.Draw(img)

    # Draw 3-6 random geometric shapes
    n_shapes = rng.randint(3, 6)
    for _ in range(n_shapes):
        shape   = rng.choice(SHAPES)
        colour  = rng.choice(COLOURS)
        # Add slight alpha variation by blending with background — keep as RGB
        x0 = rng.randint(0, img_size - 60)
        y0 = rng.randint(0, img_size - 60)
        x1 = x0 + rng.randint(40, 120)
        y1 = y0 + rng.randint(40, 120)
        # Clamp to image bounds
        x1 = min(x1, img_size - 1)
        y1 = min(y1, img_size - 1)
        bbox = [x0, y0, x1, y1]

        if shape == "circle":
            draw.ellipse(bbox, fill=colour)
        elif shape == "rectangle":
            draw.rectangle(bbox, fill=colour)
        elif shape == "ellipse":
            draw.ellipse(bbox, fill=colour, outline=(255, 255, 255), width=2)
        elif shape == "triangle":
            cx = (x0 + x1) // 2
            draw.polygon([(cx, y0), (x0, y1), (x1, y1)], fill=colour)

    return img


# ---------------------------------------------------------------------------
# Synthetic fMRI Generator
# ---------------------------------------------------------------------------

def generate_synthetic_fmri(
    n_samples : int,
    n_voxels  : int,
    image_seeds: list[int],
    rng       : np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic fMRI voxel activation patterns that are *correlated*
    with the image seeds so the mapping model has a learnable signal.

    Strategy
    --------
    Each unique image seed creates a latent "neural prototype" — a fixed
    random vector that represents the brain's response to that visual
    category. The actual fMRI sample is the prototype + Gaussian noise.
    This mimics the real neuroscience observation that repeated exposure to
    similar images produces similar (but noisy) activation patterns.

    Parameters
    ----------
    n_samples    : int             Number of fMRI samples to generate.
    n_voxels     : int             Number of simulated ROI voxels.
    image_seeds  : list[int]       Seed per sample (determines image identity).
    rng          : np.random.Generator

    Returns
    -------
    np.ndarray of shape (n_samples, n_voxels), dtype float32.
    """
    # --- Build a dictionary of "neural prototypes" keyed by image seed ---
    prototypes: dict[int, np.ndarray] = {}

    fmri_matrix = np.zeros((n_samples, n_voxels), dtype=np.float32)

    for i, seed in enumerate(image_seeds):
        if seed not in prototypes:
            # Sparse-ish prototype: most voxels near zero, a subset activated
            proto_rng = np.random.default_rng(seed=seed + 10_000)
            prototype = proto_rng.standard_normal(n_voxels).astype(np.float32)
            # Simulate sparsity: zero out 70% of voxels
            mask = proto_rng.random(n_voxels) < 0.70
            prototype[mask] = 0.0
            prototypes[seed] = prototype

        # Add Gaussian noise to the prototype (SNR ~ 2:1 by default)
        noise = rng.standard_normal(n_voxels).astype(np.float32) * 0.5
        fmri_matrix[i] = prototypes[seed] + noise

    return fmri_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed everything for reproducibility
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Mock Data Generator — Neural Translation Pipeline")
    print(f"{'='*60}")
    print(f"  Training samples  : {args.n_train}")
    print(f"  Test samples      : {args.n_test}")
    print(f"  Voxels per sample : {args.n_voxels}")
    print(f"  Image size        : {args.img_size}×{args.img_size}")
    print(f"  Output directory  : {args.out_dir}/")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Create directory structure
    # -----------------------------------------------------------------------
    splits = {
        "train": args.n_train,
        "test" : args.n_test,
    }

    for split in splits:
        os.makedirs(os.path.join(args.out_dir, "fmri",   split), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "images", split), exist_ok=True)

    # -----------------------------------------------------------------------
    # Generate data per split
    # -----------------------------------------------------------------------
    for split, n_samples in splits.items():
        print(f"[{split.upper()}] Generating {n_samples} samples …")

        # Each sample has an image_id (0–99): limits the "concept space" so
        # the model can learn a generalisable mapping across repetitions.
        n_concepts = 50  # Distinct visual concepts
        image_seeds = rng.integers(0, n_concepts, size=n_samples).tolist()

        # --- Images ---------------------------------------------------------
        for idx, seed in enumerate(image_seeds):
            img = generate_synthetic_image(args.img_size, seed=int(seed))
            img_path = os.path.join(args.out_dir, "images", split, f"{idx:04d}.png")
            img.save(img_path)

        # --- fMRI -----------------------------------------------------------
        fmri_data = generate_synthetic_fmri(n_samples, args.n_voxels, image_seeds, rng)
        fmri_path = os.path.join(args.out_dir, "fmri", split, "bold_roi.npy")
        np.save(fmri_path, fmri_data)

        # --- Metadata (image seed ↔ sample index) ---------------------------
        meta_path = os.path.join(args.out_dir, "fmri", split, "image_seeds.npy")
        np.save(meta_path, np.array(image_seeds))

        print(f"  ✓ Images saved  → {args.out_dir}/images/{split}/")
        print(f"  ✓ fMRI saved    → {fmri_path}  shape={fmri_data.shape}")
        print(f"  ✓ Metadata saved→ {meta_path}\n")

    # -----------------------------------------------------------------------
    # Quick sanity check
    # -----------------------------------------------------------------------
    print("[Sanity Check]")
    train_fmri = np.load(
        os.path.join(args.out_dir, "fmri", "train", "bold_roi.npy")
    )
    print(f"  Train fMRI shape : {train_fmri.shape}  dtype={train_fmri.dtype}")
    print(f"  Train fMRI mean  : {train_fmri.mean():.4f}")
    print(f"  Train fMRI std   : {train_fmri.std():.4f}")
    print(f"\n  ✅ Mock data ready. Run the pipeline with:\n")
    print(f"     python phase1_data_loading.py --data_dir {args.out_dir} --mode mock")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
