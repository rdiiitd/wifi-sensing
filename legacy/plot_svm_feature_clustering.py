"""
Plot KMeans and DBSCAN clustering on the same feature vectors used by the SVM.

The script reads the cached SVM feature matrices, standardizes the 33 features,
projects them to 2D with PCA, and creates one 3-panel plot per experiment:

    1. Ground-truth labels
    2. KMeans clusters
    3. GMM clusters
    4. DBSCAN clusters

It also writes a compact CSV with ARI, NMI, silhouette, DBSCAN eps, and noise
counts for report use.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Experiment:
    name: str
    cache_path: Path
    class0: str
    class1: str
    output_stem: str


EXPERIMENTS = [
    Experiment(
        name="Empty Room vs Video Game",
        cache_path=Path("svm_empty_vs_video_features.npz"),
        class0="Empty Room",
        class1="Video Game",
        output_stem="empty_vs_video",
    ),
    Experiment(
        name="5GHz Person LOS vs 5GHz Person",
        cache_path=Path("svm_5g_off_person_los_vs_person_features.npz"),
        class0="5GHz Person LOS",
        class1="5GHz Person",
        output_stem="5g_person_los_vs_person",
    ),
    Experiment(
        name="5GHz Person LOS vs 5GHz Motion LOS",
        cache_path=Path("svm_5g_off_person_los_vs_motion_los_features.npz"),
        class0="5GHz Person LOS",
        class1="5GHz Motion LOS",
        output_stem="5g_person_los_vs_motion_los",
    ),
]


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if len(unique_labels) < 2:
        return float("nan")

    mask = labels != -1
    if np.sum(mask) <= len(unique_labels):
        return float("nan")
    return float(silhouette_score(X[mask], labels[mask]))


def choose_dbscan(X_scaled: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Choose DBSCAN eps from k-distance percentiles without using class labels."""
    n_samples = len(X_scaled)
    min_samples = max(5, min(10, int(round(np.sqrt(n_samples)))))
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    distances, _ = neighbors.fit(X_scaled).kneighbors(X_scaled)
    kth_distances = distances[:, -1]

    best_labels: np.ndarray | None = None
    best_eps = float(np.percentile(kth_distances, 90))
    best_score = (-1, -n_samples)

    for percentile in (55, 60, 65, 70, 75, 80, 85, 90, 95):
        eps = float(np.percentile(kth_distances, percentile))
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        clusters = {label for label in labels if label != -1}
        noise_count = int(np.sum(labels == -1))
        clustered_count = n_samples - noise_count
        if not clusters:
            score = (-1, -noise_count)
        elif len(clusters) == 1:
            score = (0, clustered_count)
        else:
            # Prefer useful multi-cluster solutions, then fewer noise points.
            score = (1, clustered_count - abs(len(clusters) - 2) * 10)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = eps

    if best_labels is None:
        best_labels = DBSCAN(eps=best_eps, min_samples=min_samples).fit_predict(X_scaled)
    return best_labels, best_eps, min_samples


def cluster_metrics(X_scaled: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    cluster_count = len({label for label in labels if label != -1})
    return {
        "clusters": float(cluster_count),
        "noise": float(np.sum(labels == -1)),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "silhouette": safe_silhouette(X_scaled, labels),
    }


def scatter_panel(
    ax: plt.Axes,
    X_pca: np.ndarray,
    labels: np.ndarray,
    title: str,
    label_names: dict[int, str] | None = None,
) -> None:
    unique = sorted(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique), 2)))
    for idx, label in enumerate(unique):
        mask = labels == label
        if label == -1:
            color = "0.65"
            marker = "x"
            name = "Noise"
            edgecolors = None
        else:
            color = colors[idx]
            marker = "o"
            name = label_names.get(int(label), f"Cluster {label}") if label_names else f"Cluster {label}"
            edgecolors = "black"
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=34,
            c=[color],
            marker=marker,
            edgecolors=edgecolors,
            linewidths=0.35,
            alpha=0.88,
            label=name,
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=8, frameon=True)


def plot_experiment(experiment: Experiment, output_dir: Path) -> dict[str, object]:
    data = np.load(experiment.cache_path, allow_pickle=False)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=int)

    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    kmeans_labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(X_scaled)
    gmm_labels = GaussianMixture(n_components=2, covariance_type="full", random_state=42).fit_predict(X_scaled)
    dbscan_labels, dbscan_eps, dbscan_min_samples = choose_dbscan(X_scaled)

    class_names = {0: experiment.class0, 1: experiment.class1}
    km_metrics = cluster_metrics(X_scaled, y, kmeans_labels)
    gmm_metrics = cluster_metrics(X_scaled, y, gmm_labels)
    db_metrics = cluster_metrics(X_scaled, y, dbscan_labels)

    fig, axes = plt.subplots(1, 4, figsize=(17.6, 4.2), constrained_layout=True)
    scatter_panel(axes[0], X_pca, y, "Ground Truth", class_names)
    scatter_panel(
        axes[1],
        X_pca,
        kmeans_labels,
        f"KMeans\nARI={km_metrics['ari']:.2f}, silhouette={km_metrics['silhouette']:.2f}",
    )
    scatter_panel(
        axes[2],
        X_pca,
        gmm_labels,
        f"GMM\nARI={gmm_metrics['ari']:.2f}, silhouette={gmm_metrics['silhouette']:.2f}",
    )
    scatter_panel(
        axes[3],
        X_pca,
        dbscan_labels,
        f"DBSCAN\nARI={db_metrics['ari']:.2f}, clusters={int(db_metrics['clusters'])}, noise={int(db_metrics['noise'])}",
    )
    fig.suptitle(f"{experiment.name} using SVM feature vectors", fontsize=13, fontweight="bold")

    output_path = output_dir / f"{experiment.output_stem}_kmeans_dbscan.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return {
        "experiment": experiment.name,
        "samples": len(y),
        "class0_samples": int(np.sum(y == 0)),
        "class1_samples": int(np.sum(y == 1)),
        "plot": str(output_path),
        "kmeans_ari": km_metrics["ari"],
        "kmeans_nmi": km_metrics["nmi"],
        "kmeans_silhouette": km_metrics["silhouette"],
        "gmm_ari": gmm_metrics["ari"],
        "gmm_nmi": gmm_metrics["nmi"],
        "gmm_silhouette": gmm_metrics["silhouette"],
        "dbscan_ari": db_metrics["ari"],
        "dbscan_nmi": db_metrics["nmi"],
        "dbscan_silhouette": db_metrics["silhouette"],
        "dbscan_clusters": int(db_metrics["clusters"]),
        "dbscan_noise": int(db_metrics["noise"]),
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
    }


def main() -> None:
    output_dir = Path("feature_clustering_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment in EXPERIMENTS:
        print(f"Plotting {experiment.name}...")
        rows.append(plot_experiment(experiment, output_dir))

    summary_path = output_dir / "kmeans_dbscan_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    for row in rows:
        print(
            f"{row['experiment']}: "
            f"KMeans ARI={row['kmeans_ari']:.3f}, "
            f"GMM ARI={row['gmm_ari']:.3f}, "
            f"DBSCAN ARI={row['dbscan_ari']:.3f}, "
            f"plot={row['plot']}"
        )


if __name__ == "__main__":
    main()
