"""
phase2_fmri_to_clip.py
======================
Phase 2: The Latent Bridge — fMRI Activations → CLIP Embeddings
----------------------------------------------------------------
This is the core scientific contribution of the pipeline. We train a
regression model that learns to map raw brain activity (voxel patterns)
to the semantic representation space used by Stable Diffusion (CLIP).

Why CLIP?
---------
Stable Diffusion's image decoder operates on CLIP image embeddings — a
768- or 1024-dimensional vector that encodes the semantic content of an
image. By training a model to predict these embeddings from fMRI data,
we give Stable Diffusion a "brain-derived caption" to reconstruct from.

Architecture of this module
----------------------------
  CLIPEmbeddingExtractor  — Wraps CLIP ViT and extracts image embeddings
                            from stimulus images (ground-truth targets).

  RidgeRegressionMapper   — Sklearn Ridge baseline. Fast, interpretable.
                            Good starting point; often surprisingly effective
                            for fMRI decoding (see Naselaris et al. 2011).

  MLPMapper               — PyTorch MLP: 3-4 hidden layers with dropout and
                            BatchNorm. Can capture non-linear voxel→CLIP maps.

  train_mapper()          — Training loop with MSE + cosine similarity loss.
  evaluate_mapper()       — Computes R² and cosine similarity on test set.

Usage
-----
  python phase2_fmri_to_clip.py \
      --data_dir mock_data \
      --mode mock \
      --mapper mlp \
      --epochs 50 \
      --output_dir outputs/phase2

"""

import argparse
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Local imports (Phase 1)
from .phase1_data_loading import get_dataloaders


# =============================================================================
# SECTION 1 — CLIP Embedding Extractor
# =============================================================================

class CLIPEmbeddingExtractor:
    """
    Extracts 768-dimensional CLIP image embeddings from stimulus images using
    the pretrained openai/clip-vit-large-patch14 vision transformer.

    These embeddings serve as the regression TARGETS in our mapping model:
    given an fMRI pattern, predict the CLIP embedding of the image the subject
    was viewing.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: "openai/clip-vit-large-patch14" (768-D)
        Alternative: "openai/clip-vit-base-patch32" (512-D, lighter)
    device     : str   "cuda", "mps", or "cpu"
    batch_size : int   Images to process per forward pass.
    """

    # Default CLIP model — same encoder used internally by SD 1.x
    DEFAULT_MODEL = "openai/clip-vit-large-patch14"

    def __init__(
        self,
        model_name : str = DEFAULT_MODEL,
        device     : str = "cpu",
        batch_size : int = 64,
    ):
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self._model     = None
        self._processor = None

    def _load_model(self):
        """Lazy-load CLIP model (avoids slow import at module level)."""
        from transformers import CLIPProcessor, CLIPVisionModel
        print(f"  [CLIP] Loading {self.model_name} on {self.device} …")
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model     = CLIPVisionModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print(f"  [CLIP] Model loaded. Embedding dim: {self._model.config.hidden_size}")

    @property
    def embedding_dim(self) -> int:
        """Return the CLIP embedding dimension (e.g. 768 for ViT-L/14)."""
        if self._model is None:
            self._load_model()
        return self._model.config.hidden_size

    @torch.no_grad()
    def extract_from_paths(self, image_paths: list) -> np.ndarray:
        """
        Extract CLIP image embeddings from a list of image file paths.

        The model's [CLS] token output is used as the global image embedding.
        This is the same embedding that Stable Diffusion's cross-attention
        layers condition on during image generation.

        Parameters
        ----------
        image_paths : list[str]   Paths to stimulus images.

        Returns
        -------
        np.ndarray, shape (n_images, embedding_dim), dtype float32.
        """
        from PIL import Image

        if self._model is None:
            self._load_model()

        all_embeddings = []
        n = len(image_paths)

        for start in tqdm(range(0, n, self.batch_size), desc="  Extracting CLIP embeddings"):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            # CLIPProcessor handles resizing (224×224), normalisation, etc.
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass through CLIP vision encoder
            outputs  = self._model(**inputs)

            # pooler_output: shape (batch, hidden_size) — global [CLS] embedding
            embeddings = outputs.pooler_output.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def extract_from_tensors(self, image_tensors: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings directly from a float32 tensor of shape
        (N, 3, H, W) with values in [0, 1]. Useful when images are already
        loaded in memory.
        """
        if self._model is None:
            self._load_model()

        all_embeddings = []
        n = len(image_tensors)

        for start in range(0, n, self.batch_size):
            batch = image_tensors[start : start + self.batch_size].to(self.device)

            # Manually normalise for CLIP: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            batch_norm = (batch - mean) / std

            # CLIP expects pixel_values as the input key
            outputs    = self._model(pixel_values=batch_norm)
            embeddings = outputs.pooler_output.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# SECTION 2 — Ridge Regression Baseline
# =============================================================================

class RidgeRegressionMapper:
    """
    Scikit-learn Ridge Regression baseline for fMRI→CLIP mapping.

    Why Ridge?
    ----------
    fMRI data is high-dimensional (n_voxels >> n_samples) and heavily
    collinear. Ridge regression (L2 regularisation) is well-suited to this
    regime and has been the workhorse of brain decoding studies since
    Miyawaki et al. (2008).

    Performance note: Ridge is fast (<1 min on 200 samples, 4096 voxels)
    and often achieves 40-70% of MLP performance, making it an excellent
    sanity check and strong baseline.

    Parameters
    ----------
    alpha : float   Regularisation strength. Higher = stronger shrinkage.
                    Tune via cross-validation with RidgeCV in production.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        # One Ridge model per CLIP dimension (multi-output regression)
        self.model = Ridge(alpha=alpha, fit_intercept=True, max_iter=3000)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "RidgeRegressionMapper":
        """
        Fit Ridge on training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_voxels)   — Z-scored fMRI
        Y_train : np.ndarray, shape (n_train, clip_dim)   — CLIP embeddings

        Returns
        -------
        self
        """
        print(f"  [Ridge] Fitting α={self.alpha} on X={X_train.shape}, Y={Y_train.shape} …")
        self.model.fit(X_train, Y_train)
        self.is_fitted = True

        # Quick training R²
        train_r2 = self.model.score(X_train, Y_train)
        print(f"  [Ridge] Training R² = {train_r2:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CLIP embeddings from fMRI voxel vectors."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")
        return self.model.predict(X).astype(np.float32)

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
        """
        Compute R² and cosine similarity on test data.

        Returns
        -------
        dict with keys: "r2", "cosine_similarity"
        """
        Y_pred = self.predict(X_test)
        r2     = r2_score(Y_test, Y_pred)

        # Cosine similarity: dot product of L2-normalised vectors
        Y_pred_norm = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)
        Y_test_norm = Y_test / (np.linalg.norm(Y_test, axis=1, keepdims=True) + 1e-8)
        cos_sim = (Y_pred_norm * Y_test_norm).sum(axis=1).mean()

        print(f"  [Ridge] Test R²              = {r2:.4f}")
        print(f"  [Ridge] Test cosine similarity = {cos_sim:.4f}")
        return {"r2": r2, "cosine_similarity": float(cos_sim)}

    def save(self, path: str):
        """Save model weights with joblib."""
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [Ridge] Saved → {path}")

    @classmethod
    def load(cls, path: str, alpha: float = 1.0) -> "RidgeRegressionMapper":
        import joblib
        obj = cls(alpha=alpha)
        obj.model = joblib.load(path)
        obj.is_fitted = True
        return obj


# =============================================================================
# SECTION 3 — MLP Mapper (PyTorch)
# =============================================================================

class MLPMapper(nn.Module):
    """
    A lightweight Multi-Layer Perceptron that maps fMRI voxel vectors to
    CLIP image embeddings.

    Architecture
    ------------
    Input  → Linear(n_voxels, 2048) → BN → GELU → Dropout(0.3)
           → Linear(2048, 1024)     → BN → GELU → Dropout(0.2)
           → Linear(1024, 512)      → BN → GELU → Dropout(0.2)
           → Linear(512, clip_dim)
    Output → L2-normalised embedding  (unit sphere, matching CLIP's space)

    Design choices
    --------------
    - GELU activation: smoother gradient, empirically better than ReLU for
      brain decoding tasks.
    - BatchNorm: stabilises training on small fMRI datasets (n < 1000).
    - Dropout: prevents over-fitting in the high-dim → low-dim compression.
    - L2 normalisation of output: CLIP embeddings live on the unit hypersphere;
      matching this geometry improves downstream generation quality.

    Parameters
    ----------
    n_voxels : int   Number of ROI voxels (input dimension).
    clip_dim : int   CLIP embedding dimension (output dimension). 768 for ViT-L/14.
    hidden   : list  Hidden layer widths. Default: [2048, 1024, 512].
    dropout  : list  Per-layer dropout probabilities. Length = len(hidden).
    """

    def __init__(
        self,
        n_voxels : int,
        clip_dim : int,
        hidden   : list = None,
        dropout  : list = None,
    ):
        super().__init__()

        if hidden is None:
            hidden = [2048, 1024, 512]
        if dropout is None:
            dropout = [0.3, 0.2, 0.2]

        assert len(dropout) == len(hidden), "dropout list must match hidden list length"

        # Build layers dynamically from the hidden list
        layers = []
        in_dim = n_voxels

        for h_dim, drop_p in zip(hidden, dropout):
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(p=drop_p),
            ]
            in_dim = h_dim

        # Final projection to CLIP embedding space (no activation)
        layers.append(nn.Linear(in_dim, clip_dim))

        self.network = nn.Sequential(*layers)

        # Store config for serialisation
        self.config = {
            "n_voxels": n_voxels,
            "clip_dim": clip_dim,
            "hidden"  : hidden,
            "dropout" : dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.FloatTensor, shape (batch, n_voxels)

        Returns
        -------
        torch.FloatTensor, shape (batch, clip_dim)
            L2-normalised — lies on the unit hypersphere like CLIP embeddings.
        """
        out = self.network(x)                       # (batch, clip_dim)
        out = nn.functional.normalize(out, dim=-1)  # L2 normalise
        return out


# =============================================================================
# SECTION 4 — MLP Training Loop
# =============================================================================

def train_mlp_mapper(
    model       : MLPMapper,
    X_train     : np.ndarray,
    Y_train     : np.ndarray,
    X_val       : np.ndarray,
    Y_val       : np.ndarray,
    epochs      : int  = 100,
    batch_size  : int  = 32,
    lr          : float = 3e-4,
    weight_decay: float = 1e-4,
    device      : str  = "cpu",
    checkpoint_path: Optional[str] = None,
) -> dict:
    """
    Train the MLPMapper with combined MSE + cosine similarity loss.

    Loss function
    -------------
    L = λ·MSE(pred, target) + (1 - λ)·(1 - CosineSimilarity(pred, target))

    MSE penalises absolute magnitude errors; cosine similarity loss ensures
    the direction of the predicted embedding is correct. In CLIP space,
    directional alignment is what matters most for conditioning generation.

    Parameters
    ----------
    model            : MLPMapper (uninitialised weights)
    X_train, Y_train : Training fMRI and CLIP arrays.
    X_val, Y_val     : Validation fMRI and CLIP arrays.
    epochs           : Number of training epochs.
    batch_size       : Mini-batch size.
    lr               : Adam learning rate.
    weight_decay     : L2 regularisation for Adam.
    device           : "cuda" | "mps" | "cpu"
    checkpoint_path  : If set, save best model weights to this path.

    Returns
    -------
    dict with keys: "train_losses", "val_losses", "best_val_loss"
    """
    model = model.to(device)

    # --- Convert numpy → PyTorch tensors ------------------------------------
    X_tr = torch.from_numpy(X_train)
    Y_tr = torch.from_numpy(Y_train)
    X_vl = torch.from_numpy(X_val).to(device)
    Y_vl = torch.from_numpy(Y_val).to(device)

    train_ds     = TensorDataset(X_tr, Y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --- Optimiser & scheduler ----------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Loss components ----------------------------------------------------
    mse_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss()

    lam = 0.5  # weight balancing MSE vs cosine loss

    # --- Training loop -------------------------------------------------------
    history = {"train_losses": [], "val_losses": []}
    best_val_loss = float("inf")

    print(f"\n  [MLP] Training {epochs} epochs on {len(X_train)} samples …")
    print(f"  [MLP] Device: {device}  |  Batch: {batch_size}  |  LR: {lr}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)   # (batch, clip_dim)

            # Combine MSE and cosine losses
            # CosinEmbeddingLoss expects a target tensor of +1 (similar pairs)
            cos_target = torch.ones(len(pred), device=device)
            loss = (
                lam * mse_loss(pred, Y_batch)
                + (1 - lam) * cos_loss(pred, Y_batch, cos_target)
            )

            loss.backward()
            # Gradient clipping prevents exploding gradients on small datasets
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        history["train_losses"].append(avg_train_loss)

        # --- Validation -----------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_pred   = model(X_vl)
            cos_target = torch.ones(len(val_pred), device=device)
            val_loss   = (
                lam * mse_loss(val_pred, Y_vl)
                + (1 - lam) * cos_loss(val_pred, Y_vl, cos_target)
            ).item()
        history["val_losses"].append(val_loss)

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path:
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "val_loss"   : val_loss,
                    "config"     : model.config,
                }, checkpoint_path)

    history["best_val_loss"] = best_val_loss
    print(f"\n  [MLP] Training complete. Best val loss: {best_val_loss:.4f}")
    if checkpoint_path:
        print(f"  [MLP] Best model saved → {checkpoint_path}")
    return history


def evaluate_mlp_mapper(
    model : MLPMapper,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """
    Compute R² and cosine similarity on the test set using the trained MLP.

    Returns
    -------
    dict with keys: "r2", "cosine_similarity", "predictions" (np.ndarray)
    """
    model.eval().to(device)
    with torch.no_grad():
        X_t    = torch.from_numpy(X_test).to(device)
        Y_pred = model(X_t).cpu().numpy()

    r2  = r2_score(Y_test, Y_pred)

    # Per-sample cosine similarity
    Y_pred_n = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)
    Y_test_n = Y_test / (np.linalg.norm(Y_test, axis=1, keepdims=True) + 1e-8)
    cos_sim  = (Y_pred_n * Y_test_n).sum(axis=1).mean()

    print(f"  [MLP] Test R²              = {r2:.4f}")
    print(f"  [MLP] Test cosine similarity = {cos_sim:.4f}")
    return {"r2": r2, "cosine_similarity": float(cos_sim), "predictions": Y_pred}


# =============================================================================
# SECTION 5 — Batch CLIP extraction helper (extracts ALL images at once)
# =============================================================================

def extract_all_clip_embeddings(
    train_image_paths: list,
    test_image_paths : list,
    extractor        : CLIPEmbeddingExtractor,
    save_dir         : Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CLIP embeddings for all training and test images.
    Optionally save to disk to avoid re-computing on repeated runs.

    Returns
    -------
    (train_clip_embeddings, test_clip_embeddings)
    Both: np.ndarray of shape (n_samples, clip_dim), dtype float32.
    """
    if save_dir:
        train_path = os.path.join(save_dir, "train_clip_embeddings.npy")
        test_path  = os.path.join(save_dir, "test_clip_embeddings.npy")

        # Load from cache if available
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"  [CLIP] Loading cached embeddings from {save_dir}")
            return np.load(train_path), np.load(test_path)

    print(f"\n  Extracting CLIP embeddings for {len(train_image_paths)} training images …")
    train_embs = extractor.extract_from_paths(train_image_paths)

    print(f"  Extracting CLIP embeddings for {len(test_image_paths)} test images …")
    test_embs  = extractor.extract_from_paths(test_image_paths)

    print(f"  [CLIP] Train embeddings: {train_embs.shape}")
    print(f"  [CLIP] Test  embeddings: {test_embs.shape}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(train_path, train_embs)
        np.save(test_path,  test_embs)
        print(f"  [CLIP] Saved embeddings → {save_dir}")

    return train_embs, test_embs


# =============================================================================
# SECTION 6 — CLI entry point
# =============================================================================

def _detect_device() -> str:
    """Return the best available compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: fMRI → CLIP mapping model"
    )
    parser.add_argument("--data_dir",   type=str,   default="mock_data")
    parser.add_argument("--mode",       type=str,   default="mock",
                        choices=["mock", "god", "nsd", "miyawaki"])
    parser.add_argument("--mapper",     type=str,   default="both",
                        choices=["ridge", "mlp", "both"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--ridge_alpha",type=float, default=1.0)
    parser.add_argument("--output_dir", type=str,   default="outputs/phase2")
    parser.add_argument("--clip_model", type=str,
                        default="openai/clip-vit-large-patch14")
    args = parser.parse_args()

    device = _detect_device()
    print(f"\n{'='*60}")
    print(f"  Phase 2 — fMRI → CLIP Mapping")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load and preprocess data (Phase 1)
    # -------------------------------------------------------------------------
    train_loader, test_loader, prep = get_dataloaders(
        data_dir    = args.data_dir,
        mode        = args.mode,
        batch_size  = args.batch_size,
        num_workers = 0,
    )

    # Collect raw numpy arrays (needed for sklearn Ridge)
    # We reuse the already-normalised data from Phase 1
    train_fmri_list, test_fmri_list = [], []
    train_img_paths, test_img_paths = [], []
    train_img_tensors, test_img_tensors = [], []

    for fmri, imgs, paths in train_loader:
        train_fmri_list.append(fmri.numpy())
        train_img_paths.extend(paths)
        train_img_tensors.append(imgs)

    for fmri, imgs, paths in test_loader:
        test_fmri_list.append(fmri.numpy())
        test_img_paths.extend(paths)
        test_img_tensors.append(imgs)

    X_train = np.concatenate(train_fmri_list, axis=0)  # (n_train, n_voxels)
    X_test  = np.concatenate(test_fmri_list,  axis=0)  # (n_test,  n_voxels)

    # -------------------------------------------------------------------------
    # Step 2: Extract CLIP embeddings from stimulus images
    # -------------------------------------------------------------------------
    extractor = CLIPEmbeddingExtractor(
        model_name = args.clip_model,
        device     = device,
        batch_size = 32,
    )
    Y_train, Y_test = extract_all_clip_embeddings(
        train_img_paths,
        test_img_paths,
        extractor,
        save_dir = os.path.join(args.output_dir, "clip_cache"),
    )
    clip_dim = Y_train.shape[1]
    n_voxels = X_train.shape[1]
    print(f"\n  fMRI shape  : X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  CLIP shape  : Y_train={Y_train.shape}, Y_test={Y_test.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Train mapper(s)
    # -------------------------------------------------------------------------
    results = {}

    # --- Ridge Regression Baseline -------------------------------------------
    if args.mapper in ("ridge", "both"):
        print(f"\n{'─'*50}")
        print(f"  Baseline: Ridge Regression (α={args.ridge_alpha})")
        print(f"{'─'*50}")
        ridge = RidgeRegressionMapper(alpha=args.ridge_alpha)
        ridge.fit(X_train, Y_train)
        results["ridge"] = ridge.evaluate(X_test, Y_test)
        ridge.save(os.path.join(args.output_dir, "ridge_mapper.pkl"))

        # Save Ridge predictions for Phase 3
        ridge_preds = ridge.predict(X_test)
        np.save(os.path.join(args.output_dir, "ridge_predicted_embeddings.npy"), ridge_preds)

    # --- MLP Mapper -----------------------------------------------------------
    if args.mapper in ("mlp", "both"):
        print(f"\n{'─'*50}")
        print(f"  MLP Mapper ({n_voxels}→2048→1024→512→{clip_dim})")
        print(f"{'─'*50}")

        # Use 10% of training data as a validation split
        n_val   = max(10, int(0.1 * len(X_train)))
        X_val   = X_train[-n_val:]
        Y_val   = Y_train[-n_val:]
        X_tr    = X_train[:-n_val]
        Y_tr    = Y_train[:-n_val]

        mlp = MLPMapper(n_voxels=n_voxels, clip_dim=clip_dim)
        history = train_mlp_mapper(
            model           = mlp,
            X_train         = X_tr,
            Y_train         = Y_tr,
            X_val           = X_val,
            Y_val           = Y_val,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            lr              = args.lr,
            device          = device,
            checkpoint_path = os.path.join(args.output_dir, "mlp_best.pt"),
        )

        # Load best checkpoint for evaluation
        ckpt = torch.load(
            os.path.join(args.output_dir, "mlp_best.pt"),
            map_location=device,
        )
        mlp.load_state_dict(ckpt["model_state"])

        results["mlp"] = evaluate_mlp_mapper(mlp, X_test, Y_test, device=device)

        # Save MLP predictions for Phase 3
        mlp_preds = results["mlp"]["predictions"]
        np.save(os.path.join(args.output_dir, "mlp_predicted_embeddings.npy"), mlp_preds)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Phase 2 Complete — Results Summary")
    print(f"{'='*60}")
    for name, metrics in results.items():
        print(f"  [{name.upper()}]  R² = {metrics['r2']:.4f}  |  "
              f"Cosine Sim = {metrics['cosine_similarity']:.4f}")

    # Save test image paths for Phase 3 to reference
    test_paths_file = os.path.join(args.output_dir, "test_image_paths.txt")
    with open(test_paths_file, "w") as f:
        f.write("\n".join(test_img_paths))
    print(f"\n  Test image paths saved → {test_paths_file}")
    print(f"  Predicted embeddings   → {args.output_dir}/")
    print(f"\n  ✅ Run Phase 3 to generate images:\n")
    print(f"     python phase3_image_reconstruction.py "
          f"--embeddings_path {args.output_dir}/mlp_predicted_embeddings.npy "
          f"--image_paths_file {test_paths_file}")
    print()
