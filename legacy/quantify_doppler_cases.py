"""
Quantify Doppler spectrograms and compare cases using feature extraction.

This script works on the pickled Doppler matrices saved under `doppler_output/`.
It does not cluster PNG images directly. Instead, it:

1. Loads each Doppler matrix.
2. Extracts summary features for the full recording.
3. Splits each recording into fixed windows and extracts window-level features.
4. Runs KMeans, Gaussian Mixture Models, and Agglomerative clustering.
5. Saves CSV summaries and PCA scatter plots so cases can be compared visually.

Example:
    python3 quantify_doppler_cases.py

    python3 quantify_doppler_cases.py \
        --cases 5g_on_mot_los_stream_0,5g_off_mot_los_stream_0 \
        --window-frames 128
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DEFAULT_METADATA = {
    "empty_room_stream_0": {
        "case": "empty_room",
        "label": "Empty Room",
        "packet_rate": 72.0,
    },
    "video_game_stream_0": {
        "case": "video_game",
        "label": "Video Game",
        "packet_rate": 153.0,
    },
    "wa1_normal_stream_0": {
        "case": "wa1_normal",
        "label": "Walking 1",
        "packet_rate": 10.4,
    },
    "wa2_second_stream_0": {
        "case": "wa2_second",
        "label": "Walking 2",
        "packet_rate": 10.3,
    },
    "5g_on_mot_los_stream_0": {
        "case": "5g_on_mot_los",
        "label": "5G Fan-ON Motion LOS",
        "packet_rate": 94.9,
    },
    "5g_off_mot_los_stream_0": {
        "case": "5g_off_mot_los",
        "label": "5G Fan-OFF Motion LOS",
        "packet_rate": 95.7,
    },
    "5g_on_per_los_stream_0": {
        "case": "5g_on_per_los",
        "label": "5G Fan-ON Person LOS",
        "packet_rate": 98.3,
    },
    "5g_off_per_los_stream_0": {
        "case": "5g_off_per_los",
        "label": "5G Fan-OFF Person LOS",
        "packet_rate": 100.2,
    },
    "5g_on_per_stream_0": {
        "case": "5g_on_per",
        "label": "5G Fan-ON Person",
        "packet_rate": 97.6,
    },
    "5g_off_per_stream_0": {
        "case": "5g_off_per",
        "label": "5G Fan-OFF Person",
        "packet_rate": 107.4,
    },
}


FEATURE_COLUMNS = [
    "mean_power",
    "std_power",
    "p90_power",
    "motion_energy_ratio",
    "static_energy_ratio",
    "pos_energy_ratio",
    "neg_energy_ratio",
    "directional_asymmetry",
    "spectral_centroid",
    "spectral_abs_centroid",
    "spectral_spread",
    "spectral_entropy",
    "spectral_flatness",
    "peak_abs_position",
    "peak_prominence",
    "temporal_variation",
    "temporal_entropy",
    "frame_motion_ratio_mean",
    "frame_motion_ratio_std",
    "frame_abs_centroid_mean",
    "frame_abs_centroid_std",
    "frame_spread_mean",
    "frame_spread_std",
]


@dataclass
class Experiment:
    stem: str
    path: Path
    case: str
    label: str
    packet_rate: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="doppler_output",
        help="Directory containing pickled Doppler matrices.",
    )
    parser.add_argument(
        "--output-dir",
        default="quantification_results",
        help="Directory used for CSV outputs and plots.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated file stems to analyze. Leave empty to auto-discover all non-RU traces.",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=128,
        help="Window size in Doppler frames for window-level features.",
    )
    parser.add_argument(
        "--step-frames",
        type=int,
        default=128,
        help="Stride between windows in Doppler frames.",
    )
    parser.add_argument(
        "--center-band-fraction",
        type=float,
        default=0.10,
        help="Half-width of the near-zero Doppler band on the normalized axis [-1, 1].",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Number of clusters to use. Defaults to the number of selected cases.",
    )
    parser.add_argument(
        "--include-ru",
        action="store_true",
        help="Include RU/grouped outputs. By default only the standard Doppler outputs are used.",
    )
    return parser.parse_args()


def load_doppler_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        matrix = pickle.load(handle)
    return np.asarray(matrix, dtype=np.float64)


def sanitize_case_name(stem: str) -> str:
    return re.sub(r"_stream_\d+$", "", stem)


def make_label(name: str) -> str:
    return name.replace("_", " ").title()


def build_experiment(path: Path) -> Experiment:
    stem = path.stem
    meta = DEFAULT_METADATA.get(stem, {})
    case = meta.get("case", sanitize_case_name(stem))
    label = meta.get("label", make_label(case))
    packet_rate = meta.get("packet_rate")
    return Experiment(stem=stem, path=path, case=case, label=label, packet_rate=packet_rate)


def discover_experiments(input_dir: Path, selected_cases: set[str], include_ru: bool) -> list[Experiment]:
    experiments: list[Experiment] = []
    for path in sorted(input_dir.rglob("*.txt")):
        if not include_ru and any("RU" in part for part in path.parts):
            continue
        experiment = build_experiment(path)
        if selected_cases and experiment.stem not in selected_cases and experiment.case not in selected_cases:
            continue
        experiments.append(experiment)
    return experiments


def iter_windows(n_frames: int, window_frames: int, step_frames: int) -> list[tuple[int, int]]:
    if n_frames <= 0:
        return []
    if n_frames <= window_frames:
        return [(0, n_frames)]

    windows = []
    for start in range(0, n_frames - window_frames + 1, step_frames):
        windows.append((start, start + window_frames))
    if windows and windows[-1][1] < n_frames:
        windows.append((n_frames - window_frames, n_frames))
    return windows


def spectral_entropy(probabilities: np.ndarray) -> float:
    eps = 1e-12
    norm = probabilities / (np.sum(probabilities) + eps)
    return float(-(norm * np.log(norm + eps)).sum() / np.log(len(norm)))


def extract_features(matrix: np.ndarray, center_band_fraction: float) -> dict[str, float]:
    eps = 1e-12
    power = np.asarray(matrix, dtype=np.float64)
    power = np.nan_to_num(power, nan=eps, posinf=eps, neginf=eps)
    power = np.clip(power, eps, None)

    n_frames, n_bins = power.shape
    doppler_axis = np.linspace(-1.0, 1.0, n_bins)
    center_mask = np.abs(doppler_axis) <= center_band_fraction
    motion_mask = ~center_mask
    pos_mask = doppler_axis > center_band_fraction
    neg_mask = doppler_axis < -center_band_fraction

    total_power = float(power.sum())
    mean_power = float(power.mean())
    std_power = float(power.std())
    p90_power = float(np.percentile(power, 90))

    freq_profile = power.sum(axis=0)
    time_profile = power.sum(axis=1)
    freq_norm = freq_profile / (freq_profile.sum() + eps)
    time_norm = time_profile / (time_profile.sum() + eps)

    spectral_centroid = float(np.sum(doppler_axis * freq_norm))
    spectral_abs_centroid = float(np.sum(np.abs(doppler_axis) * freq_norm))
    spectral_spread = float(np.sqrt(np.sum(((doppler_axis - spectral_centroid) ** 2) * freq_norm)))

    motion_energy = float(freq_profile[motion_mask].sum() / (total_power + eps))
    static_energy = float(freq_profile[center_mask].sum() / (total_power + eps))
    pos_energy = float(freq_profile[pos_mask].sum() / (total_power + eps))
    neg_energy = float(freq_profile[neg_mask].sum() / (total_power + eps))
    directional_asymmetry = float((pos_energy - neg_energy) / (pos_energy + neg_energy + eps))

    peak_idx = int(np.argmax(freq_profile))
    peak_abs_position = float(abs(doppler_axis[peak_idx]))
    peak_prominence = float(freq_profile[peak_idx] / (freq_profile.mean() + eps))

    spectral_flatness = float(np.exp(np.mean(np.log(freq_profile + eps))) / (np.mean(freq_profile) + eps))
    temporal_variation = float(np.std(time_profile) / (np.mean(time_profile) + eps))

    row_totals = power.sum(axis=1, keepdims=True) + eps
    row_motion_ratio = power[:, motion_mask].sum(axis=1) / row_totals[:, 0]
    row_centroids = (power * doppler_axis[None, :]).sum(axis=1) / row_totals[:, 0]
    row_spreads = np.sqrt(
        ((power * (doppler_axis[None, :] - row_centroids[:, None]) ** 2).sum(axis=1)) / row_totals[:, 0]
    )

    return {
        "mean_power": mean_power,
        "std_power": std_power,
        "p90_power": p90_power,
        "motion_energy_ratio": motion_energy,
        "static_energy_ratio": static_energy,
        "pos_energy_ratio": pos_energy,
        "neg_energy_ratio": neg_energy,
        "directional_asymmetry": directional_asymmetry,
        "spectral_centroid": spectral_centroid,
        "spectral_abs_centroid": spectral_abs_centroid,
        "spectral_spread": spectral_spread,
        "spectral_entropy": spectral_entropy(freq_profile),
        "spectral_flatness": spectral_flatness,
        "peak_abs_position": peak_abs_position,
        "peak_prominence": peak_prominence,
        "temporal_variation": temporal_variation,
        "temporal_entropy": float(-(time_norm * np.log(time_norm + eps)).sum() / np.log(len(time_norm))),
        "frame_motion_ratio_mean": float(np.mean(row_motion_ratio)),
        "frame_motion_ratio_std": float(np.std(row_motion_ratio)),
        "frame_abs_centroid_mean": float(np.mean(np.abs(row_centroids))),
        "frame_abs_centroid_std": float(np.std(np.abs(row_centroids))),
        "frame_spread_mean": float(np.mean(row_spreads)),
        "frame_spread_std": float(np.std(row_spreads)),
    }


def build_feature_tables(
    experiments: list[Experiment],
    window_frames: int,
    step_frames: int,
    center_band_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recording_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for experiment in experiments:
        matrix = load_doppler_matrix(experiment.path)
        n_frames, n_bins = matrix.shape
        duration_sec = None
        if experiment.packet_rate:
            duration_sec = n_frames / experiment.packet_rate

        recording_row: dict[str, object] = {
            "stem": experiment.stem,
            "case": experiment.case,
            "label": experiment.label,
            "path": str(experiment.path),
            "n_frames": n_frames,
            "n_bins": n_bins,
            "packet_rate": experiment.packet_rate,
            "duration_sec": duration_sec,
        }
        recording_row.update(extract_features(matrix, center_band_fraction))
        recording_rows.append(recording_row)

        for window_index, (start, end) in enumerate(iter_windows(n_frames, window_frames, step_frames)):
            window = matrix[start:end, :]
            window_row: dict[str, object] = {
                "stem": experiment.stem,
                "case": experiment.case,
                "label": experiment.label,
                "path": str(experiment.path),
                "window_index": window_index,
                "start_frame": start,
                "end_frame": end,
                "window_frames": end - start,
                "packet_rate": experiment.packet_rate,
                "start_sec": None if not experiment.packet_rate else start / experiment.packet_rate,
                "end_sec": None if not experiment.packet_rate else end / experiment.packet_rate,
            }
            window_row.update(extract_features(window, center_band_fraction))
            window_rows.append(window_row)

    recording_df = pd.DataFrame(recording_rows)
    window_df = pd.DataFrame(window_rows)
    return recording_df, window_df


def cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    silhouette = float("nan")
    if len(unique_labels) > 1 and X_scaled.shape[0] > len(unique_labels):
        silhouette = float(silhouette_score(X_scaled, labels))
    return {
        "silhouette": silhouette,
        "adjusted_rand_index": float(adjusted_rand_score(truth, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(truth, labels)),
    }


def run_clustering(window_df: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]], np.ndarray]:
    X = window_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    truth = pd.factorize(window_df["case"])[0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    window_df = window_df.copy()

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    window_df["pca_1"] = pca_coords[:, 0]
    window_df["pca_2"] = pca_coords[:, 1]

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    window_df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    window_df["cluster_gmm"] = gmm.fit_predict(X_scaled)
    window_df["cluster_gmm_confidence"] = gmm.predict_proba(X_scaled).max(axis=1)

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    window_df["cluster_agglomerative"] = agglomerative.fit_predict(X_scaled)

    metrics = {
        "kmeans": cluster_metrics(X_scaled, window_df["cluster_kmeans"].to_numpy(), truth),
        "gmm": cluster_metrics(X_scaled, window_df["cluster_gmm"].to_numpy(), truth),
        "agglomerative": cluster_metrics(X_scaled, window_df["cluster_agglomerative"].to_numpy(), truth),
        "pca": {
            "explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
            "explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
        },
    }
    return window_df, metrics, X_scaled


def compute_feature_rankings(window_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = window_df.groupby("case")
    overall_mean = window_df[FEATURE_COLUMNS].mean()
    for feature in FEATURE_COLUMNS:
        between = 0.0
        within = 0.0
        for _, group in grouped:
            diff = float(group[feature].mean() - overall_mean[feature])
            between += len(group) * diff * diff
            within += float(((group[feature] - group[feature].mean()) ** 2).sum())
        fisher_score = between / within if within > 0 else math.inf
        rows.append({"feature": feature, "fisher_score": fisher_score})
    ranking = pd.DataFrame(rows).sort_values("fisher_score", ascending=False)
    return ranking.reset_index(drop=True)


def compute_case_distances(window_df: pd.DataFrame) -> pd.DataFrame:
    case_means = window_df.groupby("case")[FEATURE_COLUMNS].mean()
    rows = []
    for case_a, case_b in combinations(case_means.index, 2):
        vec_a = case_means.loc[case_a].to_numpy(dtype=np.float64)
        vec_b = case_means.loc[case_b].to_numpy(dtype=np.float64)
        distance = float(np.linalg.norm(vec_a - vec_b))
        rows.append({"case_a": case_a, "case_b": case_b, "euclidean_distance": distance})
    return pd.DataFrame(rows).sort_values("euclidean_distance", ascending=False).reset_index(drop=True)


def plot_pca(window_df: pd.DataFrame, color_column: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 7))
    unique_values = list(dict.fromkeys(window_df[color_column].tolist()))
    cmap = plt.get_cmap("tab20", len(unique_values))

    for index, value in enumerate(unique_values):
        subset = window_df[window_df[color_column] == value]
        plt.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=40,
            alpha=0.8,
            label=str(value),
            color=cmap(index),
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_summary(
    output_path: Path,
    experiments: list[Experiment],
    recording_df: pd.DataFrame,
    window_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    feature_ranking: pd.DataFrame,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Doppler Quantification Summary\n")
        handle.write("=============================\n\n")

        handle.write(f"Experiments analyzed: {len(experiments)}\n")
        handle.write(f"Unique cases: {window_df['case'].nunique()}\n")
        handle.write(f"Window samples: {len(window_df)}\n\n")

        handle.write("Cases\n")
        handle.write("-----\n")
        for experiment in experiments:
            handle.write(
                f"- {experiment.case}: {experiment.path} "
                f"(packet_rate={experiment.packet_rate if experiment.packet_rate is not None else 'unknown'})\n"
            )

        handle.write("\nClustering metrics\n")
        handle.write("------------------\n")
        for method, values in metrics.items():
            handle.write(f"{method}\n")
            for key, value in values.items():
                handle.write(f"  {key}: {value:.4f}\n")

        handle.write("\nTop discriminative features\n")
        handle.write("---------------------------\n")
        for _, row in feature_ranking.head(10).iterrows():
            handle.write(f"- {row['feature']}: fisher_score={row['fisher_score']:.4f}\n")

        handle.write("\nRecording-level feature snapshot\n")
        handle.write("-------------------------------\n")
        cols = ["case", "motion_energy_ratio", "directional_asymmetry", "spectral_spread", "temporal_variation"]
        handle.write(recording_df[cols].to_string(index=False))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = {item.strip() for item in args.cases.split(",") if item.strip()}
    experiments = discover_experiments(input_dir, selected_cases, args.include_ru)
    if not experiments:
        raise SystemExit("No Doppler files found for the requested selection.")

    recording_df, window_df = build_feature_tables(
        experiments=experiments,
        window_frames=args.window_frames,
        step_frames=args.step_frames,
        center_band_fraction=args.center_band_fraction,
    )

    n_cases = window_df["case"].nunique()
    if n_cases < 2:
        raise SystemExit("At least two cases are required for clustering.")

    n_clusters = args.clusters if args.clusters is not None else n_cases
    if n_clusters < 2:
        raise SystemExit("Cluster count must be at least 2.")
    if n_clusters > len(window_df):
        raise SystemExit("Cluster count cannot exceed the number of window samples.")

    clustered_window_df, metrics, _ = run_clustering(window_df, n_clusters)
    feature_ranking = compute_feature_rankings(clustered_window_df)
    case_distances = compute_case_distances(clustered_window_df)

    recording_df.to_csv(output_dir / "recording_features.csv", index=False)
    clustered_window_df.to_csv(output_dir / "window_features.csv", index=False)
    feature_ranking.to_csv(output_dir / "feature_ranking.csv", index=False)
    case_distances.to_csv(output_dir / "case_distances.csv", index=False)

    plot_pca(
        clustered_window_df,
        color_column="case",
        output_path=output_dir / "pca_by_case.png",
        title="Window features in PCA space by case",
    )
    plot_pca(
        clustered_window_df,
        color_column="cluster_kmeans",
        output_path=output_dir / "pca_by_kmeans.png",
        title="Window features in PCA space colored by KMeans cluster",
    )
    plot_pca(
        clustered_window_df,
        color_column="cluster_gmm",
        output_path=output_dir / "pca_by_gmm.png",
        title="Window features in PCA space colored by GMM cluster",
    )

    write_summary(
        output_path=output_dir / "summary.txt",
        experiments=experiments,
        recording_df=recording_df,
        window_df=clustered_window_df,
        metrics=metrics,
        feature_ranking=feature_ranking,
    )

    print(f"Saved results to: {output_dir.resolve()}")
    print(f"Experiments analyzed: {len(experiments)}")
    print(f"Cases analyzed: {n_cases}")
    print(f"Window samples: {len(clustered_window_df)}")
    print("Top features:")
    for _, row in feature_ranking.head(5).iterrows():
        print(f"  - {row['feature']}: {row['fisher_score']:.4f}")
    print("Clustering metrics:")
    for method in ("kmeans", "gmm", "agglomerative"):
        values = metrics[method]
        print(
            f"  - {method}: silhouette={values['silhouette']:.4f}, "
            f"ARI={values['adjusted_rand_index']:.4f}, "
            f"NMI={values['normalized_mutual_info']:.4f}"
        )


if __name__ == "__main__":
    main()
