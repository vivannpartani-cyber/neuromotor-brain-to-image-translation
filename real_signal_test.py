"""
real_signal_test.py
====================
Extracts a REAL fMRI signal + paired ground truth image from the already-
downloaded Miyawaki 2008 dataset, runs it through neuromotor dev-decode,
and produces a side-by-side comparison PNG saved to ~/Downloads.

Usage:
    python3 real_signal_test.py [--sample 0]

'sample' is the test-set index (0-indexed). Default is 0 (first test trial).
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DOWNLOADS = Path.home() / "Downloads"
NEUROMOTOR_DATA = Path.home() / ".neuromotor"
IMG_DIR = NEUROMOTOR_DATA / "data" / "miyawaki_images"


def load_test_sample(sample_idx: int):
    """Load a real fMRI vector and its paired ground-truth image from Miyawaki test set."""
    print(f"\n[1/4] Loading Miyawaki 2008 test dataset...")
    from nilearn import datasets
    miyawaki = datasets.fetch_miyawaki2008()

    mask = nib.load(miyawaki.mask).get_fdata().astype(bool)

    # Test runs are indices 24-31
    all_fmri = []
    all_labels = []

    for idx in range(24, 32):
        func_file  = miyawaki.func[idx]
        label_file = miyawaki.label[idx]

        func_data  = nib.load(func_file).get_fdata()
        func_data  = np.transpose(func_data, (3, 0, 1, 2))
        masked     = func_data[:, mask]

        labels     = pd.read_csv(label_file, header=None).values
        valid      = ~(labels == -1).all(axis=1)
        masked     = masked[valid]
        labels     = labels[valid]

        all_fmri.append(masked)
        all_labels.append(labels)

    fmri_data   = np.concatenate(all_fmri,   axis=0).astype(np.float32)
    label_data  = np.concatenate(all_labels, axis=0)

    # z-score normalise (same as training pipeline)
    mean = fmri_data.mean(axis=0, keepdims=True)
    std  = fmri_data.std(axis=0, keepdims=True)
    fmri_data = (fmri_data - mean) / (std + 1e-8)

    n_samples = len(fmri_data)
    print(f"    Test set: {n_samples} samples, {fmri_data.shape[1]} voxels each")

    if sample_idx >= n_samples:
        print(f"    ⚠ Sample {sample_idx} out of range. Using sample 0.")
        sample_idx = 0

    fmri_vector  = fmri_data[sample_idx]         # shape (5438,)
    label_vector = label_data[sample_idx]         # binary 10x10 pattern

    return fmri_vector, label_vector, sample_idx


def build_ground_truth_image(label_vector: np.ndarray) -> Image.Image:
    """Convert the Miyawaki binary 10x10 label into a clean 512x512 ground-truth image."""
    pattern = label_vector.reshape(10, 10).astype(np.float32)
    pattern = (pattern * 255).astype(np.uint8)

    # Upscale to 512x512 with pixel-perfect nearest-neighbor
    pil = Image.fromarray(pattern, mode='L').convert('RGB')
    pil = pil.resize((512, 512), Image.NEAREST)
    return pil


def run_dev_decode(fmri_vector: np.ndarray, sample_idx: int) -> Path:
    """Save the fMRI vector and run neuromotor dev-decode on it."""
    sig_path = Path(f"/tmp/real_miyawaki_test_{sample_idx}.npy")
    np.save(sig_path, fmri_vector)

    print(f"\n[2/4] Saved real fMRI signal → {sig_path}  shape={fmri_vector.shape}")
    print(f"\n[3/4] Running neuromotor dev-decode (~3 min on Apple Silicon)...")

    result = subprocess.run(
        [sys.executable, "-m", "neuromotor.cli", "dev-decode", "--signals", str(sig_path)],
        capture_output=True, text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("dev-decode failed!")

    # Find the most recent generated image
    phase3_dir = Path.home() / ".neuromotor" / "outputs" / "phase3"
    latest_dir = max(
        (d for d in phase3_dir.iterdir() if d.is_dir() and d.name.startswith("dev_decoded")),
        key=lambda d: d.stat().st_mtime
    )
    generated = sorted(latest_dir.glob("generated_*.png"))
    if not generated:
        raise RuntimeError(f"No generated image found in {latest_dir}")

    return generated[0]


def build_comparison(ground_truth: Image.Image, reconstruction: Image.Image, sample_idx: int) -> Path:
    """Stitch ground truth and reconstruction side-by-side with labels."""
    W, H  = 512, 512
    pad   = 20
    label_h = 40
    total_w = W * 2 + pad * 3
    total_h = H + pad * 2 + label_h

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 245, 247))

    # Paste images
    canvas.paste(ground_truth.resize((W, H)), (pad, pad + label_h))
    canvas.paste(reconstruction.resize((W, H)), (W + pad * 2, pad + label_h))

    draw = ImageDraw.Draw(canvas)

    # Labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    draw.text((pad, 8), "Ground Truth (Miyawaki 2008 — what subject saw)", fill=(40, 40, 40), font=font)
    draw.text((W + pad * 2, 8), "Neuromotor Reconstruction (from fMRI signal)", fill=(40, 40, 40), font=font)

    # Footer
    draw.text(
        (pad, total_h - 20),
        f"Sample #{sample_idx}  •  5,438 visual cortex voxels  →  CLIP embedding  →  Stable Diffusion v1.5",
        fill=(120, 120, 120),
        font=small_font
    )

    out_path = DOWNLOADS / f"neuromotor_real_comparison_sample{sample_idx:03d}.png"
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Test neuromotor dev-decode with a real Miyawaki fMRI signal")
    parser.add_argument("--sample", type=int, default=0, help="Test sample index (default: 0)")
    args = parser.parse_args()

    fmri_vec, label_vec, sample_idx = load_test_sample(args.sample)
    ground_truth   = build_ground_truth_image(label_vec)
    generated_path = run_dev_decode(fmri_vec, sample_idx)
    reconstruction = Image.open(generated_path).convert("RGB")

    print(f"\n[4/4] Building side-by-side comparison...")
    out_path = build_comparison(ground_truth, reconstruction, sample_idx)

    print(f"\n✅  Comparison saved → {out_path}")
    print(f"    Ground truth: Miyawaki binary 10×10 pattern (what the subject was shown)")
    print(f"    Reconstruction: What Neuromotor decoded from {fmri_vec.shape[0]} voxels")

    # Open it
    import subprocess as sp
    sp.run(["open", str(out_path)])


if __name__ == "__main__":
    main()
