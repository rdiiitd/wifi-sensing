"""
svm_classify_empty_vs_video.py
==============================
Binary SVM classifier: Empty Room (class 0) vs Video Game (class 1)

Data source: sanitised CSI matrices produced by the SHARP pipeline
  final_output/emp/empty_room_stream_0.mat  -> 'csi_matrix_processed'
  final_output/vid/video_game_stream_0.mat  -> 'csi_matrix_processed'

The .mat matrix has shape: (N_packets, N_subcarriers, 2)
  [:, :, 0] = amplitude
  [:, :, 1] = phase

This script:
  1. Loads both .mat files and slides a window along the packet axis.
  2. For each window it builds a complex CSI matrix and calls feature_extraction().
  3. Labels windows: 0 = empty room, 1 = video game.
  4. Runs k-fold cross-validation with an RBF-SVM.
  5. Prints fold-level and mean accuracy, per-class accuracy, and confusion matrix.

Usage:
    python3 svm_classify_empty_vs_video.py
    python3 svm_classify_empty_vs_video.py --window 200 --stride 100 --k 5 --C 10 --gamma scale
"""

import argparse
import math
import warnings

import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import feature_extraction

# ── default configuration (tweak with CLI args) ───────────────────────────────
DEFAULT_WINDOW  = 200       # packets per window  (≈ 1-3 s depending on packet rate)
DEFAULT_STRIDE  = 100       # stride between windows (50 % overlap)
DEFAULT_K       = 5         # number of CV folds
DEFAULT_C       = 10.0      # SVM regularisation  (search: 0.1, 1, 10, 100)
DEFAULT_GAMMA   = "scale"   # SVM gamma           ("scale", "auto", or float)
DEFAULT_KERNEL  = "rbf"     # SVM kernel
DEFAULT_SEED    = 42
ZERO_PADDING    = 0         # zero-padding for FFT inside feature_extraction()


# ── file paths ────────────────────────────────────────────────────────────────
MAT_PATHS = {
    0: "final_output/emp/empty_room_stream_0.mat",   # class 0 = empty room
    1: "final_output/vid/video_game_stream_0.mat",   # class 1 = video game
}
CLASS_NAMES = {0: "Empty Room", 1: "Video Game"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csi_complex(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a sanitised .mat file produced by the SHARP pipeline.

    Returns
    -------
    csi_complex : complex ndarray, shape (N_packets, N_subcarriers)
    amplitude   : float ndarray,   shape (N_packets, N_subcarriers)
    """
    mdic = sio.loadmat(mat_path)
    csi = mdic["csi_matrix_processed"]          # (N, S, 2)
    amplitude = csi[:, :, 0].astype(np.float64)
    phase     = csi[:, :, 1].astype(np.float64)
    csi_complex = amplitude * np.exp(1j * phase)
    return csi_complex, amplitude


def sliding_windows(n_packets: int, window: int, stride: int) -> list[tuple[int, int]]:
    windows = []
    for start in range(0, n_packets - window + 1, stride):
        windows.append((start, start + window))
    return windows


def extract_window_features(
    csi_complex: np.ndarray,
    amplitude:   np.ndarray,
    start: int,
    end:   int,
    zero_padding: int = 0,
) -> np.ndarray | None:
    """
    Extract a single feature vector from one window of CSI data.
    
    feature_extraction() expects:
      CSI_data_array : complex, shape (N_packets_in_window, N_subcarriers)
                       — note: rows = time, cols = subcarriers
      zero_padding   : int
      rssi_array     : proxy for RSSI — we use mean amplitude per packet as RSSI

    Returns None if the window cannot produce valid features.
    """
    # Slice the window
    csi_win  = csi_complex[start:end, :]   # (W, S)
    amp_win  = amplitude[start:end, :]     # (W, S)

    # Use mean amplitude per packet as a proxy RSSI signal
    rssi_proxy = np.mean(amp_win, axis=1)  # (W,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            feats = feature_extraction(csi_win, zero_padding, rssi_proxy)
        except Exception:
            return None

    feat_vec = np.array(feats, dtype=np.float64)
    if not np.all(np.isfinite(feat_vec)):
        return None
    return feat_vec


def build_dataset(window: int, stride: int, zero_padding: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X (features) and y (labels) by sliding a window over both recordings.
    """
    X_list, y_list = [], []

    for label, mat_path in MAT_PATHS.items():
        class_name = CLASS_NAMES[label]
        print(f"  Loading {class_name} from {mat_path} ...")
        csi_complex, amplitude = load_csi_complex(mat_path)
        n_packets = csi_complex.shape[0]
        windows   = sliding_windows(n_packets, window, stride)
        print(f"    {n_packets} packets → {len(windows)} windows (W={window}, S={stride})")

        valid, skipped = 0, 0
        for i, (s, e) in enumerate(windows):
            if (i + 1) % 100 == 0 or (i + 1) == len(windows):
                print(f"    → window {i+1}/{len(windows)} ...", end="\r", flush=True)
            fv = extract_window_features(csi_complex, amplitude, s, e, zero_padding)
            if fv is None:
                skipped += 1
                continue
            X_list.append(fv)
            y_list.append(label)
            valid += 1
        print(f"\n    {valid} valid, {skipped} skipped windows for {class_name}")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=int)
    return X, y


def k_fold_split(n: int, k: int, fold: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Return train/val index arrays for the given fold using stratified-like chunking.
    (Simple contiguous-block split to keep temporal ordering within each class.)
    """
    indices = np.arange(n)
    fold_size = n // k
    val_start = fold * fold_size
    val_end   = val_start + fold_size if fold < k - 1 else n
    val_idx   = indices[val_start:val_end]
    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
    return train_idx, val_idx


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window",  type=int,   default=DEFAULT_WINDOW,  help="Packets per window")
    p.add_argument("--stride",  type=int,   default=DEFAULT_STRIDE,  help="Stride between windows")
    p.add_argument("--k",       type=int,   default=DEFAULT_K,       help="Number of CV folds")
    p.add_argument("--C",       type=float, default=DEFAULT_C,       help="SVM C parameter")
    p.add_argument("--gamma",   type=str,   default=DEFAULT_GAMMA,   help="SVM gamma ('scale','auto', or float)")
    p.add_argument("--kernel",  type=str,   default=DEFAULT_KERNEL,  help="SVM kernel")
    p.add_argument("--seed",    type=int,   default=DEFAULT_SEED,    help="Random seed")
    p.add_argument("--zero-padding", type=int, default=ZERO_PADDING, help="Zero-padding for FFT in feature_extraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    gamma_val = args.gamma if args.gamma in ("scale", "auto") else float(args.gamma)

    print("=" * 60)
    print("  SVM — Empty Room vs Video Game (CSI feature classification)")
    print("=" * 60)
    print(f"  Window={args.window} packets | Stride={args.stride} | K={args.k}")
    print(f"  SVM: kernel={args.kernel}, C={args.C}, gamma={gamma_val}")
    print()

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("Building feature dataset ...")
    X, y = build_dataset(args.window, args.stride, args.zero_padding)
    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt} samples")

    # ── 2. Shuffle dataset (over entire recording, not block-wise per class) ──
    shuffle_idx = rng.permutation(len(y))
    X_shuf = X[shuffle_idx]
    y_shuf = y[shuffle_idx]

    # ── 3. K-fold cross-validation ────────────────────────────────────────────
    print(f"\nRunning {args.k}-fold cross-validation ...")
    fold_accuracy  = []
    fold_acc_cls0  = []   # Empty Room recall
    fold_acc_cls1  = []   # Video Game recall

    for fold in range(args.k):
        n = len(y_shuf)
        train_idx, val_idx = k_fold_split(n, args.k, fold, rng)

        X_train, X_val = X_shuf[train_idx], X_shuf[val_idx]
        y_train, y_val = y_shuf[train_idx], y_shuf[val_idx]

        # Scale features per fold (fit on train only)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Train SVM
        clf = SVC(kernel=args.kernel, C=args.C, gamma=gamma_val, random_state=args.seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        cm  = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc_cls0 = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        acc_cls1 = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        fold_accuracy.append(acc)
        fold_acc_cls0.append(acc_cls0)
        fold_acc_cls1.append(acc_cls1)

        print(
            f"  Fold {fold+1:2d}/{args.k}: overall={acc:.3f} | "
            f"EmptyRoom={acc_cls0:.3f} | VideoGame={acc_cls1:.3f}"
        )

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Mean overall accuracy : {np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}")
    print(f"  Mean Empty Room acc   : {np.nanmean(fold_acc_cls0):.4f} ± {np.nanstd(fold_acc_cls0):.4f}")
    print(f"  Mean Video Game acc   : {np.nanmean(fold_acc_cls1):.4f} ± {np.nanstd(fold_acc_cls1):.4f}")
    print("=" * 60)

    # ── 5. Final model on all data (for inspection) ───────────────────────────
    print("\nTraining final model on all data for classification report ...")
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X_shuf)
    clf_final = SVC(kernel=args.kernel, C=args.C, gamma=gamma_val, random_state=args.seed)
    clf_final.fit(X_final, y_shuf)
    y_final_pred = clf_final.predict(X_final)

    print("\nClassification Report (training data — sanity check):")
    print(classification_report(
        y_shuf, y_final_pred,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]
    ))


if __name__ == "__main__":
    main()
