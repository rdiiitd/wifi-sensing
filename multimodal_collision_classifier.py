"""
multimodal_collision_classifier.py
===================================
Multimodal CSI + IMU collision detection experiment.

Task
----
Predict whether a person is within the near-distance threshold at the current
time (default: label_now; future labels remain selectable via --label).

Features compared
-----------------
  CSI-only          : 6 CSI-derived features
  IMU-only          : 9 IMU-derived features
  CSI+IMU           : 15 features (concatenation of both)

Validation
----------
  - Chronological 80 / 20 hold-out (boundary before filtering -1 labels)
  - Forward-chaining CV with gap enforced on original row positions
  - StandardScaler + classifier inside a sklearn Pipeline (no leakage)
  - F1, precision, recall, accuracy, balanced accuracy, and ROC AUC

Models evaluated
----------------
  - SVM with RBF kernel       (main model)
  - Random Forest
  - Gradient Boosting         (usually strongest on small tabular data)
  - K-Nearest Neighbours      (simple distance-based baseline)
  - Naive Bayes               (probabilistic baseline)
  - AdaBoost                  (ensemble of weak learners)
  - MLP / Neural Network      (small feedforward net)
  - Logistic Regression       (linear baseline)

Feature sets
------------
  CSI-only, IMU-only, and CSI+IMU.

Output
------
  results/z01_multimodal_cv_results.csv   -- per-fold metrics
  results/z01_multimodal_summary.csv      -- mean +/- std across folds + test score

Usage
-----
  python3 scripts/multimodal_collision_classifier.py
  python3 scripts/multimodal_collision_classifier.py --data data/z_folder/z01_test_table.npz
  python3 scripts/multimodal_collision_classifier.py --label label_in1s --n_splits 4
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA   = "data/z_folder/z01_test_table.npz"
DEFAULT_LABEL  = "label_now"
DEFAULT_SPLITS = 5
DEFAULT_GAP    = 10            # 10-row gap = 2.5 s safety margin between folds
DEFAULT_SEED   = 42
RESULTS_DIR    = Path("results")


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",     default=DEFAULT_DATA,
                   help="Path to z01_test_table.npz")
    p.add_argument("--label",    default=DEFAULT_LABEL,
                   help="Label array to use (label_now / label_in1s / label_in2s)")
    p.add_argument("--n_splits", type=int, default=DEFAULT_SPLITS,
                   help="TimeSeriesSplit folds")
    p.add_argument("--gap",      type=int, default=DEFAULT_GAP,
                   help="Gap rows between train and val folds")
    p.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip confusion-matrix plots")
    return p.parse_args()


def load_data(npz_path: str, label_key: str) -> dict:
    """
    Load the feature table and return a dict with everything needed.

    Chronological 80/20 split is set on ALL rows BEFORE filtering -1 labels.
    This preserves the intended temporal gap between train and test.
    """
    raw = np.load(npz_path, allow_pickle=False)

    csi     = raw["csi"]         # (N, 6)
    imu     = raw["imu"]         # (N, 9)
    y_all   = raw[label_key]     # (N,)  -- -1 = invalid, 0 = no collision, 1 = collision
    t_video = raw["t_video"]     # (N,)  -- video-relative timestamps (seconds)
    parts   = raw["part"]        # (N,)  -- 'alone', 'approach', 'other'

    N = len(y_all)
    n_train_boundary = int(0.8 * N)   # chronological boundary on full dataset

    all_train_rows = np.arange(n_train_boundary)
    all_test_rows  = np.arange(n_train_boundary, N)

    # Filter -1 within each split
    train_valid = all_train_rows[y_all[all_train_rows] != -1]
    test_valid  = all_test_rows[y_all[all_test_rows]  != -1]

    return dict(
        csi=csi, imu=imu, y_all=y_all,
        t_video=t_video, parts=parts,
        train_idx=train_valid,
        test_idx=test_valid,
        csi_names=list(raw["csi_names"]),
        imu_names=list(raw["imu_names"]),
        window_s=float(raw["window_s"]),
        step_s=float(raw["step_s"]),
        label_key=label_key,
        n_total=N,
        n_train_boundary=n_train_boundary,
    )


def build_pipelines(seed: int) -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines (scaler always fitted inside each fold)."""
    return {
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                random_state=seed,
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="euclidean",
            )),
        ]),
        "NaiveBayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.5,
                random_state=seed,
                algorithm="SAMME",
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            )),
        ]),
    }


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute thresholded metrics and score-based AUC, handling single-class folds."""
    if len(np.unique(y_true)) < 2:
        return dict(f1=np.nan, precision=np.nan, recall=np.nan, accuracy=np.nan,
                    balanced_accuracy=np.nan, auc=np.nan,
                    single_class=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return dict(
            f1        = f1_score(y_true, y_pred, zero_division=0),
            precision = precision_score(y_true, y_pred, zero_division=0),
            recall    = recall_score(y_true, y_pred, zero_division=0),
            accuracy  = accuracy_score(y_true, y_pred),
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred),
            auc       = roc_auc_score(y_true, y_score),
            single_class=False,
        )


def positive_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return continuous positive-class scores for ROC AUC."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    original_rows: np.ndarray,
    pipelines: dict[str, Pipeline],
    n_splits: int,
    gap: int,
) -> dict[str, list[dict]]:
    """
    Forward-chaining TimeSeriesSplit cross-validation.
    Returns per-fold metric dicts, keyed by model name.
    """
    # Split valid examples into forward folds, then enforce the requested gap
    # using their original row positions so dropped -1 labels do not shrink it.
    cv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    results: dict[str, list[dict]] = {name: [] for name in pipelines}

    for fold_i, (fold_train, fold_val) in enumerate(cv.split(X_train), start=1):
        val_start_row = original_rows[fold_val[0]]
        fold_train = fold_train[original_rows[fold_train] < val_start_row - gap]
        X_f_tr, y_f_tr   = X_train[fold_train], y_train[fold_train]
        X_f_val, y_f_val = X_train[fold_val],   y_train[fold_val]

        # Skip the fold entirely if either split has only one class
        train_single = len(y_f_tr) == 0 or len(np.unique(y_f_tr)) < 2
        val_single   = len(np.unique(y_f_val)) < 2

        for name, pipeline in pipelines.items():
            if train_single or val_single:
                reason = "train" if train_single else "val"
                m = dict(f1=np.nan, precision=np.nan, recall=np.nan,
                         accuracy=np.nan, balanced_accuracy=np.nan, auc=np.nan,
                         single_class=True)
                m["fold"]         = fold_i
                m["n_train"]      = len(y_f_tr)
                m["n_val"]        = len(y_f_val)
                m["n_class0_val"] = int((y_f_val == 0).sum())
                m["n_class1_val"] = int((y_f_val == 1).sum())
                results[name].append(m)
                print(f"    [{name}] Fold {fold_i}/{n_splits}: "
                      f"SKIP (single class in {reason})  "
                      f"(train={len(y_f_tr)}, val={len(y_f_val)})")
                continue

            pipe = clone(pipeline)
            pipe.fit(X_f_tr, y_f_tr)
            y_pred = pipe.predict(X_f_val)
            y_score = positive_scores(pipe, X_f_val)

            m = safe_metrics(y_f_val, y_pred, y_score)
            m["fold"]          = fold_i
            m["n_train"]       = len(y_f_tr)
            m["n_val"]         = len(y_f_val)
            m["n_class0_val"]  = int((y_f_val == 0).sum())
            m["n_class1_val"]  = int((y_f_val == 1).sum())
            results[name].append(m)

            status = f"F1={m['f1']:.3f} AUC={m['auc']:.3f}"
            print(f"    [{name}] Fold {fold_i}/{n_splits}: {status}  "
                  f"(train={len(y_f_tr)}, val={len(y_f_val)})")

    return results


def run_test(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    pipelines: dict[str, Pipeline],
) -> dict[str, dict]:
    """Fit on full train set, evaluate on chronological hold-out test set."""
    test_results = {}
    for name, pipeline in pipelines.items():
        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score = positive_scores(pipe, X_test)
        m = safe_metrics(y_test, y_pred, y_score)
        m["y_pred"]   = y_pred
        m["y_score"]  = y_score
        m["confusion"] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        test_results[name] = m
    return test_results


def summarise_cv(cv_results: dict[str, list[dict]]) -> dict[str, dict]:
    """Compute mean +/- std across folds (ignoring single-class folds)."""
    summary = {}
    for name, folds in cv_results.items():
        valid   = [f for f in folds if not f["single_class"]]
        skipped = len(folds) - len(valid)
        if not valid:
            summary[name] = dict(
                f1_mean=np.nan, f1_std=np.nan,
                auc_mean=np.nan, auc_std=np.nan,
                precision_mean=np.nan, recall_mean=np.nan,
                accuracy_mean=np.nan, balanced_accuracy_mean=np.nan,
                skipped_folds=skipped,
            )
        else:
            summary[name] = dict(
                f1_mean        = np.mean([f["f1"]        for f in valid]),
                f1_std         = np.std( [f["f1"]        for f in valid]),
                auc_mean       = np.mean([f["auc"]       for f in valid]),
                auc_std        = np.std( [f["auc"]       for f in valid]),
                precision_mean = np.mean([f["precision"] for f in valid]),
                recall_mean    = np.mean([f["recall"]    for f in valid]),
                accuracy_mean  = np.mean([f["accuracy"]  for f in valid]),
                balanced_accuracy_mean = np.mean([f["balanced_accuracy"] for f in valid]),
                skipped_folds  = skipped,
            )
    return summary


def print_summary_table(
    feature_set: str,
    cv_summary: dict[str, dict],
    test_results: dict[str, dict],
) -> None:
    print(f"\n{'─'*72}")
    print(f"  Feature set: {feature_set}")
    print(f"{'─'*72}")
    hdr = f"  {'Model':<22} {'CV F1':>7} {'CV AUC':>7} {'CV BAcc':>8} {'Test F1':>8} {'Test AUC':>8} {'Test BAcc':>9}"
    print(hdr)
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")
    for name in cv_summary:
        cv = cv_summary[name]
        te = test_results[name]
        cv_f1  = f"{cv['f1_mean']:.3f}"       if not np.isnan(cv['f1_mean'])      else "  n/a"
        te_f1  = f"{te['f1']:.3f}"             if not np.isnan(te['f1'])           else "  n/a"
        cv_auc = f"{cv['auc_mean']:.3f}"        if not np.isnan(cv['auc_mean'])     else "  n/a"
        cv_bac = f"{cv['balanced_accuracy_mean']:.3f}" if not np.isnan(cv['balanced_accuracy_mean']) else "  n/a"
        te_auc = f"{te['auc']:.3f}"             if not np.isnan(te['auc'])          else "  n/a"
        te_bac = f"{te['balanced_accuracy']:.3f}" if not np.isnan(te['balanced_accuracy']) else "  n/a"
        skip   = f"  [{cv['skipped_folds']} fold(s) skipped]" if cv['skipped_folds'] else ""
        print(f"  {name:<22} {cv_f1:>7} {cv_auc:>7} {cv_bac:>8} {te_f1:>8} {te_auc:>8} {te_bac:>9}{skip}")


def save_confusion_plots(
    test_results: dict[str, dict],
    y_test: np.ndarray,
    feature_set: str,
    out_dir: Path,
    label_key: str,
) -> None:
    """Save confusion-matrix figure for each model in this feature set."""
    n_models = len(test_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, test_results.items()):
        cm = res["confusion"]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Negative (0)", "Positive (1)"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        f1_str = f"F1={res['f1']:.3f}" if not np.isnan(res['f1']) else "F1=n/a"
        ax.set_title(f"{name}\n{f1_str}", fontsize=10)

    safe_name = feature_set.replace("+", "_plus_").replace(" ", "_").lower()
    fig.suptitle(
        f"Test Confusion Matrices — {feature_set}\n(label: {label_key})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / f"confusion_{safe_name}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def save_csv_results(
    all_cv_folds:  dict[str, dict],
    all_summaries: dict[str, dict],
    all_test:      dict[str, dict],
    out_dir: Path,
    label_key: str,
) -> None:
    """Write per-fold and summary CSV files to results/."""

    # Per-fold CSV
    fold_path = out_dir / "z01_multimodal_cv_results.csv"
    fold_fields = ["feature_set", "model", "fold", "n_train", "n_val",
                   "n_class0_val", "n_class1_val", "f1", "precision",
                   "recall", "accuracy", "balanced_accuracy", "auc", "single_class"]
    with open(fold_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fold_fields, extrasaction="ignore")
        w.writeheader()
        for fset, models in all_cv_folds.items():
            for model_name, folds in models.items():
                for fold_d in folds:
                    row = {"feature_set": fset, "model": model_name, **fold_d}
                    w.writerow(row)
    print(f"\n  Saved fold CSV : {fold_path}")

    # Summary CSV
    summary_path = out_dir / "z01_multimodal_summary.csv"
    summary_fields = [
        "feature_set", "model", "label",
        "cv_f1_mean", "cv_f1_std", "cv_auc_mean", "cv_auc_std",
        "cv_precision_mean", "cv_recall_mean", "cv_accuracy_mean",
        "cv_balanced_accuracy_mean", "cv_skipped_folds",
        "test_f1", "test_precision", "test_recall", "test_accuracy",
        "test_balanced_accuracy", "test_auc",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for fset, models in all_summaries.items():
            for model_name, s in models.items():
                te = all_test[fset][model_name]
                w.writerow({
                    "feature_set":        fset,
                    "model":              model_name,
                    "label":              label_key,
                    "cv_f1_mean":         s["f1_mean"],
                    "cv_f1_std":          s["f1_std"],
                    "cv_auc_mean":        s["auc_mean"],
                    "cv_auc_std":         s["auc_std"],
                    "cv_precision_mean":  s["precision_mean"],
                    "cv_recall_mean":     s["recall_mean"],
                    "cv_accuracy_mean":   s["accuracy_mean"],
                    "cv_balanced_accuracy_mean": s["balanced_accuracy_mean"],
                    "cv_skipped_folds":   s["skipped_folds"],
                    "test_f1":            te["f1"],
                    "test_precision":     te["precision"],
                    "test_recall":        te["recall"],
                    "test_accuracy":      te["accuracy"],
                    "test_balanced_accuracy": te["balanced_accuracy"],
                    "test_auc":            te["auc"],
                })
    print(f"  Saved summary CSV: {summary_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Multimodal CSI + IMU Collision Detection")
    print(f"{'='*72}")
    print(f"  Data   : {args.data}")
    print(f"  Label  : {args.label}")
    print(f"  CV     : TimeSeriesSplit(n_splits={args.n_splits}, gap={args.gap})")
    print(f"  Seed   : {args.seed}")

    d = load_data(args.data, args.label)

    y_train = d["y_all"][d["train_idx"]]
    y_test  = d["y_all"][d["test_idx"]]

    print(f"\n{'─'*72}")
    print("  Dataset summary")
    print(f"{'─'*72}")
    print(f"  Total windows          : {d['n_total']}")
    print(f"  Chrono boundary (80%)  : row {d['n_train_boundary']}  "
          f"(t ≈ {d['t_video'][d['n_train_boundary']-1]:.1f} s)")
    print(f"  Train valid windows    : {len(d['train_idx'])}  "
          f"(class 0: {(y_train==0).sum()}, class 1: {(y_train==1).sum()})")
    print(f"  Test valid windows     : {len(d['test_idx'])}  "
          f"(class 0: {(y_test==0).sum()},  class 1: {(y_test==1).sum()})")
    print(f"  Train time range       : "
          f"{d['t_video'][d['train_idx'][0]]:.1f} s – "
          f"{d['t_video'][d['train_idx'][-1]]:.1f} s")
    print(f"  Test time range        : "
          f"{d['t_video'][d['test_idx'][0]]:.1f} s – "
          f"{d['t_video'][d['test_idx'][-1]]:.1f} s")
    print(f"  Window / step          : "
          f"{d['window_s']} s / {d['step_s']} s  (50 % overlap)")

    # ── 2. Define feature sets ────────────────────────────────────────────────
    feature_sets = {
        "CSI-only": d["csi"],
        "IMU-only": d["imu"],
        "CSI+IMU" : np.concatenate([d["csi"], d["imu"]], axis=1),
    }

    # ── 3. Run experiments ────────────────────────────────────────────────────
    all_cv_folds:  dict[str, dict] = {}
    all_summaries: dict[str, dict] = {}
    all_test:      dict[str, dict] = {}

    for fset_name, X_all in feature_sets.items():
        X_train = X_all[d["train_idx"]]
        X_test  = X_all[d["test_idx"]]

        print(f"\n{'='*72}")
        print(f"  Feature set: {fset_name}  ({X_train.shape[1]} feature(s))")
        print(f"{'='*72}")

        pipelines = build_pipelines(args.seed)

        # Cross-validation
        print(f"\n  Cross-validation "
              f"(TimeSeriesSplit, {args.n_splits} folds, gap={args.gap}) ...")
        cv_folds = run_cv(
            X_train, y_train, d["train_idx"], pipelines, args.n_splits, args.gap
        )

        # Hold-out test set
        print(f"\n  Evaluating on chronological test set ...")
        test_res = run_test(X_train, y_train, X_test, y_test, pipelines)
        for name, res in test_res.items():
            f1_str = f"{res['f1']:.3f}" if not np.isnan(res['f1']) else "n/a"
            print(f"    [{name}] Test F1={f1_str}  AUC={res['auc']:.3f}  "
                  f"BalancedAcc={res['balanced_accuracy']:.3f}")
            if not np.isnan(res['f1']):
                print(f"      Confusion matrix (rows=true, cols=pred):\n{res['confusion']}")

        # Summary table
        cv_summary = summarise_cv(cv_folds)
        print_summary_table(fset_name, cv_summary, test_res)

        # Classification report on test set
        for name, res in test_res.items():
            if not np.isnan(res["f1"]):
                print(f"\n  Classification report — {name}  (test set):")
                print(classification_report(
                    y_test, res["y_pred"],
                    target_names=["Negative (0)", "Positive (1)"],
                    zero_division=0,
                ))

        # Confusion matrix plots
        if not args.no_plots:
            save_confusion_plots(
                test_res, y_test, fset_name, RESULTS_DIR, args.label
            )

        all_cv_folds[fset_name]  = cv_folds
        all_summaries[fset_name] = cv_summary
        all_test[fset_name]      = test_res

    # ── 4. Final cross-feature-set comparison ─────────────────────────────────
    print(f"\n\n{'='*72}")
    print("  FINAL COMPARISON — Test F1 across feature sets and models")
    print(f"{'='*72}")
    all_models = list(build_pipelines(args.seed).keys())
    print(f"  {'Feature set':<12}", end="")
    for m in all_models:
        print(f"  {m:<22}", end="")
    print()
    print(f"  {'─'*12}", end="")
    for _ in all_models:
        print(f"  {'─'*22}", end="")
    print()
    for fset in feature_sets:
        print(f"  {fset:<12}", end="")
        for model in all_models:
            f1 = all_test[fset][model]["f1"]
            cell = f"{f1:.3f}" if not np.isnan(f1) else " n/a"
            print(f"  {cell:<22}", end="")
        print()

    print("\n  Compare modalities using CV first; use the chronological test set")
    print("  once for final evaluation after selecting a model in CV.")

    # ── 5. Save CSVs ──────────────────────────────────────────────────────────
    save_csv_results(all_cv_folds, all_summaries, all_test, RESULTS_DIR, args.label)

    print(f"\n{'='*72}")
    print("  Done.")
    print(f"{'='*72}\n")

    # ── 6. Limitations reminder ───────────────────────────────────────────────
    print("  Limitations of this run:")
    print("    * Single session only -- results do not generalise to new wearers/days.")
    print(f"    * Test set is {len(d['test_idx'])} labeled windows; treat scores as indicative.")
    print("    * Labels are auto-generated; verify a subset against video before publishing.")
    print("    * 'alone' and 'other' parts contain no valid labels for this binary task.")
    print()


if __name__ == "__main__":
    main()
