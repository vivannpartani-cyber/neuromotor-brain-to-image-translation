# 🧠 Neural Translation: Brainwaves → Images

> **Proof-of-Concept**  
> Decoding human visual cortex fMRI signals into images using an MLP mapper and Stable Diffusion.

---

## Quick Start — One Command

```bash
./run.sh demo
```

That's it. The script automatically:
- Finds Python 3 on your machine (works whether it's `python`, `python3`, etc.)
- Installs all dependencies
- Downloads the fMRI dataset (~100 MB)
- Trains the MLP mapper (~5-10 min, first run only)
- Downloads Stable Diffusion (~4 GB, first run only)
- Generates brain reconstructions and saves them to your **Downloads folder**

> ⚠️ **First run takes ~25-30 minutes** due to model downloads. Every run after that is instant.

---

## Requirements

- **Python 3.10+** — download from [python.org](https://python.org) if not installed
- **~8 GB free disk space** (for downloaded models)
- **Internet connection** (first run only)

---

## All Commands

```bash
./run.sh demo                                        # Random sample → ~/Downloads
./run.sh list                                        # Show all 5 test reconstructions
./run.sh show --sample 2                             # Open a specific one
./run.sh reconstruct --input ~/Downloads/brain.png  # Reconstruct your own image
```

---

## Overview

```
fMRI Voxels (5,438)  ──MLP──▶  CLIP Embedding (1024-D)  ──Stable Diffusion v1.5──▶  Image
        ↑                                                                                  ↑
  Miyawaki 2008                                                                "What the brain saw"
  (real human data)
```

| Component | Technology |
|-----------|-----------|
| fMRI Dataset | Miyawaki 2008 — real V1-V4 visual cortex signals |
| Voxel-to-CLIP Mapper | 5-layer MLP (5,438 → 1,024 dims) |
| Image Generator | Stable Diffusion v1.5 |
| Accelerator | Apple Silicon MPS / CUDA / CPU (auto-detected) |

---

## Results

| Metric | Value |
|--------|-------|
| Biological fMRI voxels decoded | **5,438** |
| Training trials | **1,512** |
| MLP Cosine Similarity (test) | **0.817** |
| SD denoising steps | 25 |

---

## Project Structure

```
neuromotor/
├── run.sh                        ← START HERE — one-click launcher
├── neuromotor_cli.py             ← CLI tool (called by run.sh)
├── demo.ipynb                    ← Jupyter Notebook (for interactive use)
├── phase1_data_loading.py        ← fMRI data loader
├── phase2_fmri_to_clip.py        ← MLP mapper training
├── phase3_image_reconstruction.py← Stable Diffusion generation
├── requirements.txt              ← Dependencies
└── outputs/
    ├── phase2/mlp_best.pt        ← Trained model weights
    └── phase3/comparison_*.png   ← Ground truth vs reconstructed images
```

---

## For Jupyter Notebook Users

```bash
pip3 install jupyter
jupyter notebook demo.ipynb
```

---

## References

1. Miyawaki et al. (2008). *Visual image reconstruction from human brain activity.* Neuron.
2. Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.
3. Radford et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision.* ICML.
