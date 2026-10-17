"""
quantify_doppler.py — Doppler Spectrogram Quantification
=========================================================

Two types of analysis:

1. INTRA-spectrogram: KMeans K=2 on a single spectrogram's point cloud
   to find motion vs stillness *within* one recording.

2. INTER-spectrogram: Extract per-window features from two experiments,
   pool them, run KMeans K=2 + DBSCAN, and see if the algorithm can
   *separate* the two activities in PCA space.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# =============================================================================
# Configuration
# =============================================================================
MAX_DOPPLER_HZ = 50
SLIDING = 1
WINDOW_SEC = 2.0   # Window size for inter-spectrogram analysis

EXPERIMENTS = {
    # 5 GHz Fan-ON
    "5g_on_mot_los": {"file": "doppler_output/5g_/5g_on_mot_los_stream_0.txt",
                       "rate": 94.9,  "label": "5G Fan-ON Motion LOS"},
    "5g_on_per_los": {"file": "doppler_output/5g_/5g_on_per_los_stream_0.txt",
                       "rate": 98.3,  "label": "5G Fan-ON Person LOS"},
    "5g_on_per":     {"file": "doppler_output/5g_/5g_on_per_stream_0.txt",
                       "rate": 97.6,  "label": "5G Fan-ON Person"},
    # 5 GHz Fan-OFF
    "5g_off_mot_los":{"file": "doppler_output/5g_/5g_off_mot_los_stream_0.txt",
                       "rate": 95.7,  "label": "5G Fan-OFF Motion LOS"},
    "5g_off_per_los":{"file": "doppler_output/5g_/5g_off_per_los_stream_0.txt",
                       "rate": 100.2, "label": "5G Fan-OFF Person LOS"},
    "5g_off_per":    {"file": "doppler_output/5g_/5g_off_per_stream_0.txt",
                       "rate": 107.4, "label": "5G Fan-OFF Person"},
}

# Pairs for comparison
PAIRS = [
    ("5g_off_mot_los", "5g_off_per", "Fan-OFF: Motion-LOS vs Person"),
]


# =============================================================================
# Helpers
# =============================================================================
def load_doppler(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def spectrogram_to_cloud(doppler_matrix, packet_rate):
    """Convert [time × freq] matrix → (N, 3) array of [time_s, freq_Hz, energy_dB]."""
    n_time, n_freq = doppler_matrix.shape
    energy_db = 10 * np.log10(doppler_matrix + 1e-10)

    time_axis = np.arange(n_time) * SLIDING / packet_rate
    freq_axis = np.linspace(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ, n_freq)
    t_grid, f_grid = np.meshgrid(time_axis, freq_axis, indexing='ij')

    return np.column_stack([t_grid.ravel(), f_grid.ravel(), energy_db.ravel()])


# =============================================================================
# PART 1 — Intra-spectrogram: KMeans (K=2) Motion vs Stillness
# =============================================================================
def kmeans_motion_stillness(doppler_matrix, packet_rate, label):
    """
    Run KMeans K=2 on the full spectrogram point cloud.
    Returns the results dict and all data needed to plot.
    """
    n_time, n_freq = doppler_matrix.shape
    cloud = spectrogram_to_cloud(doppler_matrix, packet_rate)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cloud)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_km = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, y_km, sample_size=min(30000, len(cloud)))

    avg_e = [np.mean(cloud[y_km == c, 2]) for c in range(2)]
    motion_c = int(np.argmax(avg_e))
    still_c  = 1 - motion_c

    labels_grid = y_km.reshape(n_time, n_freq)
    motion_frac = np.mean(labels_grid == motion_c, axis=1)
    row_is_motion = motion_frac > 0.30

    time_axis = np.arange(n_time) * SLIDING / packet_rate
    total_duration = time_axis[-1] if len(time_axis) > 1 else 0

    duty_cycle = np.mean(row_is_motion) * 100
    energy_contrast = avg_e[motion_c] - avg_e[still_c]

    freq_axis = np.linspace(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ, n_freq)
    motion_rows = np.where(row_is_motion)[0]
    if len(motion_rows) > 0:
        energy_db_grid = 10 * np.log10(doppler_matrix + 1e-10)
        motion_spectrum = np.mean(energy_db_grid[motion_rows], axis=0)
        weights = 10 ** (motion_spectrum / 10)
        dominant_freq = np.sum(weights * np.abs(freq_axis)) / np.sum(weights)
    else:
        dominant_freq = 0.0

    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  Shape: {n_time} × {n_freq}   Duration: {total_duration:.1f}s")
    print(f"  Silhouette:       {sil:.3f}")
    print(f"  Motion points:    {np.sum(y_km==motion_c):,}   ({np.sum(y_km==motion_c)/len(cloud)*100:.1f}%)")
    print(f"  Still  points:    {np.sum(y_km==still_c):,}   ({np.sum(y_km==still_c)/len(cloud)*100:.1f}%)")
    print(f"  Energy contrast:  {energy_contrast:.2f} dB")
    print(f"  Duty cycle:       {duty_cycle:.1f}%")
    print(f"  Dominant |freq|:  {dominant_freq:.2f} Hz")

    return {
        'label': label,
        'cloud': cloud,
        'y_km': y_km,
        'motion_c': motion_c,
        'still_c': still_c,
        'sil': sil,
        'duty_cycle': duty_cycle,
        'energy_contrast': energy_contrast,
        'dominant_freq': dominant_freq,
        'total_duration': total_duration,
        'n_motion': np.sum(y_km == motion_c),
        'n_still': np.sum(y_km == still_c),
    }


def plot_single(result, save=True):
    """Plot a single experiment: spectrogram + cluster scatter side by side."""
    cloud = result['cloud']
    y_km  = result['y_km']
    mc    = result['motion_c']
    label = result['label']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    n_pts = len(cloud)
    n_freq = 100
    n_time = n_pts // n_freq
    energy_2d = cloud[:, 2].reshape(n_time, n_freq)
    axes[0].imshow(energy_2d.T, aspect='auto', cmap='jet', origin='lower',
                   extent=[0, result['total_duration'], -MAX_DOPPLER_HZ, MAX_DOPPLER_HZ])
    axes[0].set_title(f"{label}\nOriginal Doppler Spectrogram", fontsize=12)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Doppler Freq [Hz]")

    MAX_SCATTER = 25000
    if n_pts > MAX_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pts, MAX_SCATTER, replace=False)
    else:
        idx = np.arange(n_pts)

    colors = np.where(y_km[idx] == mc, '#E74C3C', '#3498DB')
    axes[1].scatter(cloud[idx, 0], cloud[idx, 1],
                    c=colors, s=4, alpha=0.5, edgecolors='none')
    axes[1].set_xlim(0, result['total_duration'])
    axes[1].set_ylim(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ)
    axes[1].set_title(f"KMeans (K=2)  ·  Silhouette = {result['sil']:.3f}\n"
                      f"Motion: {result['n_motion']:,} pts  |  "
                      f"Still: {result['n_still']:,} pts", fontsize=12)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Doppler Freq [Hz]")

    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=9, label=f'Motion ({result["duty_cycle"]:.1f}%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
               markersize=9, label=f'Still ({100 - result["duty_cycle"]:.1f}%)')
    ]
    axes[1].legend(handles=legend, loc='upper right', fontsize=10)

    plt.tight_layout()
    if save:
        safe = label.replace(" ", "_").replace("-", "_")
        fname = f"quant_kmeans_{safe}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {fname}")
        plt.close()
    else:
        plt.show()


def plot_pair(res_on, res_off, pair_title, save=True):
    """Side-by-side cluster scatter: Fan-ON (left) vs Fan-OFF (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    MAX_SCATTER = 25000

    for ax, res, side in [(axes[0], res_on, "Fan ON"), (axes[1], res_off, "Fan OFF")]:
        cloud = res['cloud']
        y_km  = res['y_km']
        mc    = res['motion_c']

        n_pts = len(cloud)
        if n_pts > MAX_SCATTER:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_pts, MAX_SCATTER, replace=False)
        else:
            idx = np.arange(n_pts)

        colors = np.where(y_km[idx] == mc, '#E74C3C', '#3498DB')
        ax.scatter(cloud[idx, 0], cloud[idx, 1],
                   c=colors, s=5, alpha=0.5, edgecolors='none')
        ax.set_xlim(0, res['total_duration'])
        ax.set_ylim(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ)
        ax.set_title(f"{side}: {res['label']}\n"
                     f"Duty={res['duty_cycle']:.1f}%  |  "
                     f"ΔE={res['energy_contrast']:.2f}dB  |  "
                     f"Sil={res['sil']:.3f}", fontsize=11)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Doppler Freq [Hz]")

        legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
                   markersize=8, label=f'Motion ({res["duty_cycle"]:.1f}%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
                   markersize=8, label=f'Still ({100-res["duty_cycle"]:.1f}%)')
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=9)

    fig.suptitle(f"KMeans Motion vs Stillness — {pair_title}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        safe = pair_title.replace(" ", "_").replace("(", "").replace(")", "")
        fname = f"quant_pair_{safe}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {fname}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# PART 2 — Inter-spectrogram: Separate ACTIVITIES across spectrograms
# =============================================================================
def extract_window_features(chunk):
    """
    From a [time_frames × freq_bins] chunk, extract a feature vector:
      - Total energy
      - Doppler spread (variance of frequency distribution)
      - Doppler centroid
      - Peak energy
      - Energy in low-freq band (|f| < 10 Hz) — breathing / stillness
      - Energy in high-freq band (|f| > 20 Hz) — fast motion
      - Mean frequency profile (100 bins)
    Total: 6 scalars + 100 profile = 106 features per window.
    """
    freq_bins = chunk.shape[1]
    freq_axis = np.linspace(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ, freq_bins)

    total_energy = np.sum(chunk)
    peak_energy  = np.max(chunk)

    power_dist = np.sum(chunk, axis=0)
    power_sum  = np.sum(power_dist) + 1e-10

    centroid = np.sum(power_dist * freq_axis) / power_sum
    spread   = np.sqrt(np.sum(power_dist * ((freq_axis - centroid) ** 2)) / power_sum)

    # Band energies
    low_mask  = np.abs(freq_axis) < 10
    high_mask = np.abs(freq_axis) > 20
    low_energy  = np.sum(chunk[:, low_mask])
    high_energy = np.sum(chunk[:, high_mask])

    mean_profile = np.mean(chunk, axis=0)

    return np.concatenate([[total_energy, spread, centroid,
                            peak_energy, low_energy, high_energy],
                           mean_profile])


def extract_all_windows(filepath, packet_rate, window_sec):
    """Load Doppler, threshold, split into windows, return feature matrix."""
    with open(filepath, "rb") as f:
        d = pickle.load(f)

    thresh = np.percentile(d, 80)
    d_clean = np.where(d > thresh, d, 0)

    chunk_frames = int(round(packet_rate * window_sec))
    num_chunks = len(d_clean) // chunk_frames

    features = []
    for i in range(num_chunks):
        start = i * chunk_frames
        end   = start + chunk_frames
        feat  = extract_window_features(d_clean[start:end, :])
        features.append(feat)

    return np.array(features) if features else np.array([]).reshape(0, 0)


def inter_spectrogram_clustering(key_a, key_b, pair_title):
    """
    Pool windows from two experiments, run KMeans + DBSCAN,
    project to 2D via PCA, and plot 3-panel scatter.
    """
    cfg_a = EXPERIMENTS[key_a]
    cfg_b = EXPERIMENTS[key_b]

    X_a = extract_all_windows(cfg_a["file"], cfg_a["rate"], WINDOW_SEC)
    X_b = extract_all_windows(cfg_b["file"], cfg_b["rate"], WINDOW_SEC)

    n_a, n_b = len(X_a), len(X_b)
    print(f"\n  {'═'*60}")
    print(f"  {pair_title}")
    print(f"  {cfg_a['label']}: {n_a} windows  |  {cfg_b['label']}: {n_b} windows")
    print(f"  {'═'*60}")

    if n_a + n_b < 4:
        print(f"  ⚠ Only {n_a + n_b} windows total — need ≥4 for meaningful PCA. Skipping.")
        return

    # Pool and label: 0 = experiment A, 1 = experiment B
    X_all  = np.vstack([X_a, X_b])
    y_true = np.array([0] * n_a + [1] * n_b)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # PCA → 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans K=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_km = kmeans.fit_predict(X_scaled)
    ari_km = adjusted_rand_score(y_true, y_km)
    sil_km = silhouette_score(X_scaled, y_km) if len(X_all) > 2 else 0

    # DBSCAN
    db = DBSCAN(eps=1.5, min_samples=2)
    y_db = db.fit_predict(X_scaled)
    n_db_clusters = len(set(y_db)) - (1 if -1 in y_db else 0)
    ari_db = adjusted_rand_score(y_true, y_db) if n_db_clusters > 0 else 0

    print(f"  KMeans → ARI: {ari_km:.3f}  |  Silhouette: {sil_km:.3f}")
    print(f"  DBSCAN → Clusters: {n_db_clusters}  |  Noise: {np.sum(y_db==-1)}  |  ARI: {ari_db:.3f}")

    # ====================== 3-PANEL PLOT ======================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: True Labels
    ax = axes[0]
    ax.scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1],
               c='royalblue', marker='o', s=90, edgecolors='k', linewidths=0.5,
               label=cfg_a['label'], zorder=3)
    ax.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1],
               c='crimson', marker='X', s=90, edgecolors='k', linewidths=0.5,
               label=cfg_b['label'], zorder=3)
    ax.set_title("True Labels", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: KMeans
    ax = axes[1]
    km_colors = {0: 'mediumorchid', 1: 'orange'}
    for c in [0, 1]:
        mask = y_km == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=km_colors[c], marker='o', s=90, edgecolors='k', linewidths=0.5,
                   label=f'Cluster {c}', zorder=3)
    ax.set_title(f"KMeans K=2 (ARI: {ari_km:.2f})", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: DBSCAN
    ax = axes[2]
    db_labels = sorted(set(y_db))
    db_colors = ['gray', 'forestgreen', 'dodgerblue', 'tomato', 'gold', 'cyan']
    for i, cl in enumerate(db_labels):
        mask = y_db == cl
        name = "Noise" if cl == -1 else f"Cluster {cl}"
        mk   = 'x' if cl == -1 else 'o'
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=db_colors[i % len(db_colors)], marker=mk, s=90,
                   edgecolors='k' if cl != -1 else 'none', linewidths=0.5,
                   label=name, zorder=3)
    ax.set_title(f"DBSCAN (ARI: {ari_db:.2f})", fontsize=13, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PCA — Separating Activities ({WINDOW_SEC:.0f}s Windows) — {pair_title}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    safe = pair_title.replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"quant_pca_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {fname}")
    plt.close()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # ── PART 1: Intra-spectrogram clustering ──
    print("=" * 60)
    print("  PART 1 — KMeans (Motion vs Stillness within each file)")
    print("=" * 60)

    results = {}
    
    # Extract keys we actually need
    keys_to_process = set()
    for p in PAIRS:
        keys_to_process.add(p[0])
        keys_to_process.add(p[1])

    for key, cfg in EXPERIMENTS.items():
        if key in keys_to_process:
            mat = load_doppler(cfg["file"])
            results[key] = kmeans_motion_stillness(mat, cfg["rate"], cfg["label"])
            plot_single(results[key])

    for key_on, key_off, title in PAIRS:
        plot_pair(results[key_on], results[key_off], title)

    # ── PART 2: Inter-spectrogram clustering ──
    print(f"\n\n{'='*60}")
    print("  PART 2 — Separating Activities (Fan ON vs Fan OFF)")
    print(f"{'='*60}")

    for key_on, key_off, title in PAIRS:
        inter_spectrogram_clustering(key_on, key_off, title)

    # ── Summary ──
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY TABLE (Intra-spectrogram)")
    print(f"{'='*80}")
    print(f"  {'Experiment':<26s}  {'Dur(s)':>6s}  {'Sil':>6s}  "
          f"{'Duty%':>7s}  {'ΔE(dB)':>8s}  {'|f|(Hz)':>8s}")
    print(f"  {'-'*72}")
    for key_on, key_off, title in PAIRS:
        for key in [key_on, key_off]:
            r = results[key]
            print(f"  {r['label']:<26s}  {r['total_duration']:>6.1f}  {r['sil']:>6.3f}  "
                  f"{r['duty_cycle']:>6.1f}%  {r['energy_contrast']:>8.2f}  "
                  f"{r['dominant_freq']:>8.2f}")
        print(f"  {'·'*72}")

    print("\nDone!")

