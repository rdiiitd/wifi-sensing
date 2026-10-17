"""
Doppler Window Clustering — PCA Projection
-------------------------------------------

For each 5-second window of a Doppler spectrogram we extract high-level
features (total energy, Doppler spread, centroid, mean frequency profile).

We then pool windows from a Fan-ON and Fan-OFF pair, project them to 2-D
via PCA, and show three panels side by side:
    1. True Labels   (which experiment each window came from)
    2. KMeans (K=2)  clusters
    3. DBSCAN         clusters

This gives a clean, presentation-ready scatter plot that visually proves
whether KMeans / DBSCAN can distinguish the two conditions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# =============================================================================
# Configuration
# =============================================================================
WINDOW_SEC = 5.0

EXPERIMENTS = {
    # Original walking data
    "wa1_normal":    {"file": "doppler_output/wa1/wa1_normal_stream_0.txt",
                      "rate": 10.4,  "label": "Walking 1 (ON)"},
    "wa2_second":    {"file": "doppler_output/wa2/wa2_second_stream_0.txt",
                      "rate": 10.3,  "label": "Walking 2 (OFF)"},
    # 5 GHz Fan-OFF
    "5g_off_mot_los":{"file": "doppler_output/5g_/5g_off_mot_los_stream_0.txt",
                      "rate": 95.7,  "label": "Fan-OFF Motion LOS"},
    "5g_off_per_los":{"file": "doppler_output/5g_/5g_off_per_los_stream_0.txt",
                      "rate": 100.2, "label": "Fan-OFF Person LOS"},
    "5g_off_per":    {"file": "doppler_output/5g_/5g_off_per_stream_0.txt",
                      "rate": 107.4, "label": "Fan-OFF Person"},
    # 5 GHz Fan-ON
    "5g_on_mot_los": {"file": "doppler_output/5g_/5g_on_mot_los_stream_0.txt",
                      "rate": 94.9,  "label": "Fan-ON Motion LOS"},
    "5g_on_per_los": {"file": "doppler_output/5g_/5g_on_per_los_stream_0.txt",
                      "rate": 98.3,  "label": "Fan-ON Person LOS"},
    "5g_on_per":     {"file": "doppler_output/5g_/5g_on_per_stream_0.txt",
                      "rate": 97.6,  "label": "Fan-ON Person"},
}

# Pairs to compare (key_on, key_off, plot title)
COMPARISONS = [
    ("5g_on_mot_los", "5g_off_mot_los", "Motion in LOS"),
    ("5g_on_per_los", "5g_off_per_los", "Person in LOS"),
    ("5g_on_per",     "5g_off_per",     "Person (not LOS)"),
    ("wa1_normal",    "wa2_second",     "Walking (original)"),
]


# =============================================================================
# Feature Extraction
# =============================================================================
def extract_window_features(chunk):
    """Extracts high-level statistical features from a 5-second chunk."""
    total_energy = np.sum(chunk)

    freq_bins = chunk.shape[1]
    freq_axis = np.linspace(-50, 50, freq_bins)

    power_dist = np.sum(chunk, axis=0)
    power_sum = np.sum(power_dist) + 1e-10

    centroid = np.sum(power_dist * freq_axis) / power_sum
    spread = np.sqrt(np.sum(power_dist * ((freq_axis - centroid)**2)) / power_sum)

    mean_profile = np.mean(chunk, axis=0)

    features = np.concatenate([[total_energy, spread, centroid], mean_profile])
    return features


def process_file_in_windows(filepath, packet_rate, window_sec):
    """Load a Doppler pickle, threshold, split into windows, extract features."""
    with open(filepath, "rb") as f:
        d = pickle.load(f)

    thresh = np.percentile(d, 80)
    d_clean = np.where(d > thresh, d, 0)

    chunk_frames = int(round(packet_rate * window_sec))
    num_chunks = len(d_clean) // chunk_frames

    X = []
    for i in range(num_chunks):
        start = i * chunk_frames
        end = start + chunk_frames
        chunk = d_clean[start:end, :]
        feats = extract_window_features(chunk)
        X.append(feats)

    return np.array(X)


# =============================================================================
# PCA + Clustering Plot (3-panel)
# =============================================================================
def pca_cluster_plot(key_on, key_off, title):
    """
    Produces a 3-panel PCA scatter plot:
        Panel 1: True Labels
        Panel 2: KMeans (K=2)
        Panel 3: DBSCAN
    """
    cfg_on  = EXPERIMENTS[key_on]
    cfg_off = EXPERIMENTS[key_off]

    X_on  = process_file_in_windows(cfg_on["file"],  cfg_on["rate"],  WINDOW_SEC)
    X_off = process_file_in_windows(cfg_off["file"], cfg_off["rate"], WINDOW_SEC)

    n_on  = len(X_on)
    n_off = len(X_off)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  {cfg_on['label']}: {n_on} windows  |  {cfg_off['label']}: {n_off} windows")
    print(f"{'='*60}")

    if n_on + n_off < 3:
        print("  ⚠ Not enough windows for PCA clustering (need ≥3 total). Skipping.")
        return

    if n_on == 0 or n_off == 0:
        print("  ⚠ Not enough data for clustering (need ≥1 window per class).")
        return

    # Pool windows and create true labels
    X_all = np.vstack([X_on, X_off])
    y_true = np.array([0]*n_on + [1]*n_off)  # 0 = Fan-ON, 1 = Fan-OFF

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # PCA → 2-D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_km = kmeans.fit_predict(X_scaled)
    ari_km = adjusted_rand_score(y_true, y_km)
    sil_km = silhouette_score(X_scaled, y_km) if (len(set(y_km)) > 1 and len(X_all) > 2) else 0

    # --- DBSCAN ---
    db = DBSCAN(eps=1.5, min_samples=2)
    y_db = db.fit_predict(X_scaled)
    n_clusters_db = len(set(y_db)) - (1 if -1 in y_db else 0)
    ari_db = adjusted_rand_score(y_true, y_db) if n_clusters_db > 0 else 0

    print(f"  KMeans  → Silhouette: {sil_km:.3f}  |  ARI: {ari_km:.2f}")
    print(f"  DBSCAN  → Clusters: {n_clusters_db}  |  Noise: {np.sum(y_db==-1)}  |  ARI: {ari_db:.2f}")

    # ====================== PLOTTING ======================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: True Labels ---
    ax = axes[0]
    ax.scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1],
               c='royalblue', marker='o', s=80, edgecolors='k', linewidths=0.5,
               label=cfg_on['label'], zorder=3)
    ax.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1],
               c='crimson', marker='x', s=80, linewidths=2,
               label=cfg_off['label'], zorder=3)
    ax.set_title("True Labels", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: KMeans ---
    ax = axes[1]
    cmap_km = {0: 'mediumorchid', 1: 'orange'}
    for c in [0, 1]:
        mask = y_km == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=cmap_km[c], marker='o', s=80, edgecolors='k', linewidths=0.5,
                   label=f'Cluster {c}', zorder=3)
    ax.set_title(f"KMeans (ARI: {ari_km:.2f})", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: DBSCAN ---
    ax = axes[2]
    db_labels = sorted(set(y_db))
    cmap_db = ['gray', 'forestgreen', 'dodgerblue', 'tomato', 'gold', 'cyan', 'magenta']
    for i, cl in enumerate(db_labels):
        mask = y_db == cl
        name = "Noise (-1)" if cl == -1 else f"Cluster {cl}"
        marker = 'x' if cl == -1 else 'o'
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=cmap_db[i % len(cmap_db)], marker=marker, s=80,
                   edgecolors='k' if cl != -1 else 'none', linewidths=0.5,
                   label=name, zorder=3)
    ax.set_title(f"DBSCAN (ARI: {ari_db:.2f})", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PCA Projection of Doppler Quantification ({WINDOW_SEC:.0f}s Windows) — {title}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    safe = title.replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"pca_clustering_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Plot saved → {fname}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PCA Doppler Window Clustering")
    print("=" * 60)

    for key_on, key_off, title in COMPARISONS:
        pca_cluster_plot(key_on, key_off, title)

    print("\nDone! All PCA clustering plots saved.")
