#!/usr/bin/env python3
"""
neuromotor.cli — Neural Image Reconstruction from Brainwaves
=============================================================
Stanford AIMI Proof-of-Concept

After installing:
    pip3 install git+https://github.com/vivannpartani-cyber/neuromotor

Run with:
    neuromotor demo
    neuromotor list
    neuromotor show --sample 2
    neuromotor reconstruct --input ~/Downloads/brain.png
"""

import os, sys, argparse, time, subprocess, platform, shutil
import numpy as np
import torch
from pathlib import Path

# ───────────────────────────────────────────────────────────────
#  Terminal colours
# ───────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN  = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════╗
║  🧠  N E U R O M O T O R  —  Brainwaves → Images              ║
║      Stanford AIMI Proof-of-Concept  |  Miyawaki 2008 Dataset  ║
╚══════════════════════════════════════════════════════════════════╝{RESET}""")

def step(msg):  print(f"\n{CYAN}{BOLD}▶  {msg}{RESET}")
def ok(msg):    print(f"  {GREEN}✓  {msg}{RESET}")
def info(msg):  print(f"  {DIM}   {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠  {msg}{RESET}")
def fail(msg):  print(f"  {RED}✗  {msg}{RESET}")

def progress_bar(label, n=20):
    for i in range(n + 1):
        pct = i / n
        b = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
        sys.stdout.write(f"\r  {DIM}{label}  {CYAN}|{b}|{RESET} {int(pct*100):3d}%")
        sys.stdout.flush()
        time.sleep(0.02)
    print()

# ───────────────────────────────────────────────────────────────
#  Paths
#  Outputs go to ~/.neuromotor/ so they persist across installs
#  and don't end up buried in site-packages.
# ───────────────────────────────────────────────────────────────
HOME_DIR   = Path.home() / ".neuromotor"
P2         = HOME_DIR / "outputs" / "phase2"
P3         = HOME_DIR / "outputs" / "phase3"
DATA_DIR   = HOME_DIR / "data"
DOWNLOADS  = Path.home() / "Downloads"

MLP_PT         = P2 / "mlp_best.pt"
EMBEDDINGS_NPY = P2 / "mlp_predicted_embeddings.npy"
TEST_PATHS_TXT = P2 / "test_image_paths.txt"

for d in [P2, P3, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────
def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon (MPS)"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"

def open_file(path):
    path = str(path)
    if platform.system() == "Darwin":
        subprocess.run(["open", path], check=False)
    elif platform.system() == "Windows":
        os.startfile(path)
    else:
        subprocess.run(["xdg-open", path], check=False)

def ensure_trained():
    """Train MLP mapper if weights don't exist yet."""
    if MLP_PT.exists():
        ok(f"Pre-trained weights found")
        return
    warn("No model found — training MLP mapper (~5-10 min, one-time only) …")
    info("Downloading Miyawaki fMRI dataset (~100 MB) and CLIP model (~1.7 GB) …")
    subprocess.run([
        sys.executable, "-m", "neuromotor.phase2_fmri_to_clip",
        "--mode",       "miyawaki",
        "--data_dir",   str(DATA_DIR),
        "--mapper",     "mlp",
        "--epochs",     "15",
        "--output_dir", str(P2),
    ], check=True)
    ok("Training complete!")

def ensure_generated(n=5):
    """Generate reconstructions if not already present."""
    existing = sorted(P3.glob("comparison_*.png"))
    if len(existing) >= n:
        ok(f"Found {len(existing)} pre-generated reconstructions")
        return
    warn("Generating reconstructions — first run downloads SD v1.5 (~4 GB) …")
    ensure_trained()
    subprocess.run([
        sys.executable, "-m", "neuromotor.phase3_image_reconstruction",
        "--embeddings_path",  str(EMBEDDINGS_NPY),
        "--image_paths_file", str(TEST_PATHS_TXT),
        "--output_dir",       str(P3),
        "--n_images",         str(n),
        "--n_steps",          "25",
    ], check=True)
    ok("Generation complete!")

# ───────────────────────────────────────────────────────────────
#  Commands
# ───────────────────────────────────────────────────────────────
def cmd_demo(args):
    banner()
    step("Neural Reconstruction Demo — full auto-setup")
    _, device_name = detect_device()
    info(f"Device: {device_name}")
    info(f"Model cache: {HOME_DIR}")
    ensure_generated(n=5)
    import random
    comparisons = sorted(P3.glob("comparison_*.png"))
    comp = random.choice(comparisons)
    idx  = int(comp.stem.split("_")[1])
    gen  = P3 / f"generated_{idx:04d}.png"
    step("Saving reconstruction to Downloads")
    dest = DOWNLOADS / f"neuromotor_reconstruction_{idx:04d}.png"
    shutil.copy(gen, dest)
    ok(f"Saved → {dest}")
    step("Opening results")
    open_file(comp)
    open_file(dest)
    print(f"\n{GREEN}{BOLD}  🧠  Done! Reconstructed image saved to ~/Downloads.{RESET}\n")
    print(f"  {DIM}Run 'neuromotor list' to see all 5 samples.{RESET}\n")


def cmd_list(args):
    banner()
    step("All generated reconstructions")
    ensure_generated(n=5)
    for i, p in enumerate(sorted(P3.glob("comparison_*.png"))):
        gen  = P3 / f"generated_{i:04d}.png"
        mark = f"{GREEN}ready{RESET}" if gen.exists() else f"{RED}missing{RESET}"
        print(f"  {BOLD}[{i}]{RESET}  {p.name}  ({mark})")
    print(f"\n  {DIM}Use:  neuromotor show --sample <index>{RESET}\n")


def cmd_show(args):
    banner()
    ensure_generated(n=args.sample + 1)
    comp = P3 / f"comparison_{args.sample:04d}.png"
    if not comp.exists():
        fail(f"Sample {args.sample} not found."); sys.exit(1)
    step(f"Opening Sample {args.sample}")
    ok(f"Comparison  → {comp}")
    open_file(comp)


def cmd_reconstruct(args):
    banner()
    step("Neural Image Reconstruction from Custom Stimulus")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        fail(f"File not found: {input_path}")
        info("Put a PNG or JPG in ~/Downloads, then run:")
        info("  neuromotor reconstruct --input ~/Downloads/myimage.png")
        sys.exit(1)
    _, device_name = detect_device()
    info(f"Device: {device_name}")
    info(f"Input : {input_path}")

    # Ensure mapper trained
    ensure_trained()

    # Encode → simulate fMRI
    step("Step 1/4  Encoding stimulus → simulated fMRI voxels")
    progress_bar("Simulating visual cortex activity")
    from PIL import Image as PILImage
    from torchvision import transforms
    img   = PILImage.open(input_path).convert("RGB")
    tf    = transforms.Compose([transforms.Resize((10,10)), transforms.Grayscale(), transforms.ToTensor()])
    small = tf(img).numpy().squeeze().flatten()
    small = np.pad(small, (0, max(0, 100-small.size)))[:100]
    np.random.seed(42)
    proj     = np.random.randn(5438, 100) * 0.01
    fmri_vec = (proj @ small).astype(np.float32)
    fmri_vec = (fmri_vec - fmri_vec.mean()) / (fmri_vec.std() + 1e-8)
    ok(f"fMRI vector: {fmri_vec.shape} voxels")

    # MLP → CLIP
    step("Step 2/4  Mapping voxels → CLIP visual features")
    from neuromotor.phase2_fmri_to_clip import MLPMapper
    mlp = MLPMapper(input_dim=5438, output_dim=1024)
    mlp.load_state_dict(torch.load(MLP_PT, map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        clip_emb = mlp(torch.tensor(fmri_vec).unsqueeze(0)).numpy()
    ok(f"CLIP embedding: {clip_emb.shape}")

    # Stable Diffusion
    step("Step 3/4  Generating image via Stable Diffusion (~30-90s)")
    tmp_emb  = HOME_DIR / "_cli_tmp_emb.npy"
    tmp_imgs = HOME_DIR / "_cli_tmp_paths.txt"
    out_dir  = P3 / f"cli_{input_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_emb, clip_emb)
    tmp_imgs.write_text(str(input_path) + "\n")
    subprocess.run([
        sys.executable, "-m", "neuromotor.phase3_image_reconstruction",
        "--embeddings_path",  str(tmp_emb),
        "--image_paths_file", str(tmp_imgs),
        "--output_dir",       str(out_dir),
        "--n_images",         "1",
        "--n_steps",          "30",
    ], check=True)
    tmp_emb.unlink(missing_ok=True)
    tmp_imgs.unlink(missing_ok=True)

    # Save to Downloads
    step("Step 4/4  Saving to Downloads")
    generated  = sorted(out_dir.glob("generated_*.png"))
    comparison = sorted(out_dir.glob("comparison_*.png"))
    if not generated:
        fail("No output image produced."); sys.exit(1)
    dest  = DOWNLOADS / f"neuromotor_{input_path.stem}_reconstructed.png"
    cdest = DOWNLOADS / f"neuromotor_{input_path.stem}_comparison.png"
    shutil.copy(generated[0], dest)
    ok(f"Reconstructed image  → {dest}")
    if comparison:
        shutil.copy(comparison[0], cdest)
        ok(f"Comparison grid      → {cdest}")
    open_file(dest)
    print(f"\n{GREEN}{BOLD}  🧠  Done! Check your Downloads folder.{RESET}\n")


# ───────────────────────────────────────────────────────────────
#  Entry point
# ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="neuromotor",
        description="Neural Image Reconstruction from Brainwaves — Stanford AIMI PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""{DIM}
Getting started:
  neuromotor demo                              ← auto-setup, saves to ~/Downloads
  neuromotor list                              ← show all reconstructions
  neuromotor show --sample 0                  ← open a specific one
  neuromotor reconstruct --input ~/Downloads/brain.png{RESET}
"""
    )
    sub = parser.add_subparsers(title="commands", dest="command")
    sub.add_parser("demo",  help="Full auto-setup demo — saves result to ~/Downloads").set_defaults(func=cmd_demo)
    sub.add_parser("list",  help="List all generated reconstructions").set_defaults(func=cmd_list)
    p_show = sub.add_parser("show", help="Open a specific sample")
    p_show.add_argument("--sample", type=int, default=0, metavar="N")
    p_show.set_defaults(func=cmd_show)
    p_rec = sub.add_parser("reconstruct", help="Reconstruct from your own image")
    p_rec.add_argument("--input", required=True, metavar="PATH")
    p_rec.set_defaults(func=cmd_reconstruct)
    args = parser.parse_args()
    if args.command is None:
        banner(); parser.print_help(); return
    args.func(args)

if __name__ == "__main__":
    main()
