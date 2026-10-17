"""
Doppler Spectrogram Quantification using KMeans and DBSCAN
----------------------------------------------------------

Approach:
  The Doppler spectrogram is a matrix [time_steps x freq_bins].
  We treat each cell as a point (time, frequency, energy_dB).

  KMeans (K=2) separates MOTION from NON-MOTION regions. We then
  map the cluster labels back onto the time axis to produce a
  quantitative "motion timeline" and compute concrete metrics:
    - Motion Duty Cycle (% of time with detected motion)
    - Energy Contrast (dB gap between motion and still clusters)
    - Dominant Doppler Shift during motion (average speed indicator)
    - Per-window (5s) motion score

  DBSCAN is used to find dense motion "events" and isolate noise.

Datasets:
  - Walking 1 (Fan ON):  wa1_normal_stream_0
  - Walking 2 (Fan OFF): wa2_second_stream_0
  - 5GHz Fan-ON / Fan-OFF experiments (motion, person-LOS, person)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =============================================================================
# Configuration
# =============================================================================
FILE_1 = "doppler_output/wa1/wa1_normal_stream_0.txt"   # Walking (Fan ON)
FILE_2 = "doppler_output/wa2/wa2_second_stream_0.txt"   # Walking 2 (Fan OFF)
LABEL_1 = "Walking 1 (Fan ON)"
LABEL_2 = "Walking 2 (Fan OFF)"

RATE_1 = 10.4   # packets per second
RATE_2 = 10.3
SLIDING = 1

N_FREQ_BINS = 100
MAX_DOPPLER_HZ = 50
WINDOW_SEC = 5.0          # 5-second windows (motion/still alternation)


# =============================================================================
# Helpers
# =============================================================================
def load_doppler(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def doppler_to_cloud(doppler_matrix, packet_rate):
    """Convert [time x freq] matrix -> (N, 3) array of [time_s, freq_Hz, energy_dB]."""
    n_time, n_freq = doppler_matrix.shape
    energy_db = 10 * np.log10(doppler_matrix + 1e-10)

    time_axis = np.arange(n_time) * SLIDING / packet_rate
    freq_axis = np.linspace(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ, n_freq)
    t_grid, f_grid = np.meshgrid(time_axis, freq_axis, indexing='ij')

    return np.column_stack([t_grid.ravel(), f_grid.ravel(), energy_db.ravel()])


# =============================================================================
# KMeans Quantification
# =============================================================================
def kmeans_quantify(doppler_matrix, packet_rate, label, window_sec=5.0):
    """
    Run KMeans (K=2) on the full spectrogram point cloud and produce
    quantitative metrics that distinguish motion from non-motion.
    """
    n_time, n_freq = doppler_matrix.shape
    cloud = doppler_to_cloud(doppler_matrix, packet_rate)

    # Scale and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cloud)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_km = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, y_km, sample_size=min(30000, len(cloud)))

    # Identify motion cluster = higher average energy
    avg_e = [np.mean(cloud[y_km == c, 2]) for c in range(2)]
    motion_c = int(np.argmax(avg_e))
    still_c  = 1 - motion_c

    # Reshape cluster labels back to the [time x freq] grid
    labels_grid = y_km.reshape(n_time, n_freq)

    # ---- PER-ROW (per time-step) motion fraction ----
    # For each row, what fraction of freq bins are in the motion cluster?
    motion_frac_per_row = np.mean(labels_grid == motion_c, axis=1)

    # A row is classified as "motion" if > 30% of its freq bins are motion
    MOTION_ROW_THRESH = 0.30
    row_is_motion = motion_frac_per_row > MOTION_ROW_THRESH

    time_axis = np.arange(n_time) * SLIDING / packet_rate
    total_duration = time_axis[-1]

    # ---- GLOBAL METRICS ----
    motion_duty_cycle = np.mean(row_is_motion) * 100  # percentage
    energy_contrast  = avg_e[motion_c] - avg_e[still_c]  # dB gap

    # Dominant Doppler shift during motion rows
    freq_axis = np.linspace(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ, n_freq)
    motion_rows_idx = np.where(row_is_motion)[0]
    if len(motion_rows_idx) > 0:
        energy_db_grid = 10 * np.log10(doppler_matrix + 1e-10)
        motion_spectrum = np.mean(energy_db_grid[motion_rows_idx], axis=0)
        # Weighted centroid (exclude DC bin at center)
        weights = 10 ** (motion_spectrum / 10)
        dominant_freq = np.sum(weights * np.abs(freq_axis)) / np.sum(weights)
    else:
        dominant_freq = 0.0

    # ---- PER-WINDOW (5s) motion score ----
    window_frames = int(round(packet_rate * window_sec))
    n_windows = n_time // window_frames
    window_scores = []
    for w in range(n_windows):
        s = w * window_frames
        e = s + window_frames
        score = np.mean(row_is_motion[s:e]) * 100  # % of rows with motion
        window_scores.append(score)
    window_scores = np.array(window_scores)

    # ---- PRINT RESULTS ----
    print(f"\n{'='*60}")
    print(f"  KMeans Quantification: {label}")
    print(f"{'='*60}")
    print(f"  Spectrogram shape:          {n_time} x {n_freq}")
    print(f"  Total duration:             {total_duration:.1f} s")
    print(f"  Silhouette Score:           {sil:.3f}")
    print(f"")
    print(f"  --- Cluster Statistics ---")
    print(f"  Motion cluster avg energy:  {avg_e[motion_c]:.2f} dB  ({np.sum(y_km==motion_c)} points)")
    print(f"  Still  cluster avg energy:  {avg_e[still_c]:.2f} dB  ({np.sum(y_km==still_c)} points)")
    print(f"  Energy Contrast (gap):      {energy_contrast:.2f} dB")
    print(f"")
    print(f"  --- Motion Detection ---")
    print(f"  Motion Duty Cycle:          {motion_duty_cycle:.1f}%")
    print(f"  Motion rows:                {np.sum(row_is_motion)} / {n_time}")
    print(f"  Dominant Doppler |freq|:    {dominant_freq:.2f} Hz")
    print(f"")
    print(f"  --- Per {window_sec}s Window Motion Scores ---")
    for i, sc in enumerate(window_scores):
        t_start = i * window_sec
        t_end   = t_start + window_sec
        tag = "MOTION" if sc > 50 else "STILL "
        bar = "█" * int(sc / 5) + "░" * (20 - int(sc / 5))
        print(f"    [{t_start:6.1f}s - {t_end:6.1f}s]  {bar}  {sc:5.1f}%  {tag}")

    # ---- PLOTS ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [3, 3, 1, 2]})

    # Row 1: Original spectrogram
    energy_db_full = 10 * np.log10(doppler_matrix.T + 1e-10)
    axes[0].imshow(energy_db_full, aspect='auto', cmap='jet', origin='lower',
                   extent=[0, total_duration, -MAX_DOPPLER_HZ, MAX_DOPPLER_HZ])
    axes[0].set_title(f"{label} — Original Doppler Spectrogram")
    axes[0].set_ylabel("Doppler Freq [Hz]")

    # Row 2: KMeans cluster scatter plot (motion = orange, still = blue)
    # Subsample for plotting performance
    MAX_SCATTER = 20000
    if len(cloud) > MAX_SCATTER:
        rng = np.random.default_rng(42)
        scatter_idx = rng.choice(len(cloud), MAX_SCATTER, replace=False)
    else:
        scatter_idx = np.arange(len(cloud))
    scatter_colors = np.where(y_km[scatter_idx] == motion_c, 'orangered', 'steelblue')
    axes[1].scatter(cloud[scatter_idx, 0], cloud[scatter_idx, 1],
                    c=scatter_colors, s=3, alpha=0.5, edgecolors='none')
    axes[1].set_xlim(0, total_duration)
    axes[1].set_ylim(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ)
    axes[1].set_title(f"KMeans Clusters (K=2) — "
                      f"●Motion ({np.sum(y_km==motion_c)} pts)  "
                      f"●Still ({np.sum(y_km==still_c)} pts)  |  "
                      f"Silhouette = {sil:.3f}")
    axes[1].set_ylabel("Doppler Freq [Hz]")
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='orangered',
                              markersize=8, label=f'Motion ({motion_duty_cycle:.1f}%)'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                              markersize=8, label=f'Still ({100-motion_duty_cycle:.1f}%)')]
    axes[1].legend(handles=legend_elements, loc='upper right')

    # Row 3: Motion timeline (binary motion/still per time step)
    axes[2].fill_between(time_axis, 0, row_is_motion.astype(float),
                         color='orangered', alpha=0.8, label='Motion detected')
    axes[2].set_xlim(0, total_duration)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_ylabel("Motion")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Still", "Motion"])
    axes[2].set_title(f"Motion Timeline (duty cycle = {motion_duty_cycle:.1f}%)")
    axes[2].legend(loc='upper right')

    # Row 4: Per-window bar chart
    window_centers = [(i + 0.5) * window_sec for i in range(n_windows)]
    colors = ['orangered' if s > 50 else 'steelblue' for s in window_scores]
    axes[3].bar(window_centers, window_scores, width=window_sec * 0.85,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[3].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    axes[3].set_xlim(0, total_duration)
    axes[3].set_ylim(0, 105)
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Motion Score (%)")
    axes[3].set_title(f"Per-{window_sec}s Window Motion Score")
    axes[3].legend()

    plt.tight_layout()
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"kmeans_quantification_{safe}.png"
    plt.savefig(fname, dpi=150)
    print(f"\n  Plot saved -> {fname}")
    plt.close()

    return {
        'label': label,
        'duration': total_duration,
        'silhouette': sil,
        'motion_energy': avg_e[motion_c],
        'still_energy': avg_e[still_c],
        'energy_contrast': energy_contrast,
        'duty_cycle': motion_duty_cycle,
        'dominant_freq': dominant_freq,
        'window_scores': window_scores,
        'motion_rows': row_is_motion,
    }


# =============================================================================
# DBSCAN Quantification
# =============================================================================
def dbscan_quantify(doppler_matrix, packet_rate, label):
    """
    Run DBSCAN on points with energy above the median to find dense
    motion "events" (bursts of Doppler activity).

    Only high-energy points are kept so DBSCAN can detect density
    contrast between motion bursts and the sparse background.
    """
    n_time, n_freq = doppler_matrix.shape
    cloud = doppler_to_cloud(doppler_matrix, packet_rate)

    # Only keep top-quartile energy points — this creates density contrast
    # that DBSCAN needs to find clusters
    thresh = np.percentile(cloud[:, 2], 75)
    mask_high = cloud[:, 2] > thresh
    cloud_high = cloud[mask_high]

    # Subsample if still too large
    MAX_PTS = 15000
    if len(cloud_high) > MAX_PTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cloud_high), MAX_PTS, replace=False)
        idx.sort()
        cloud_sub = cloud_high[idx]
    else:
        cloud_sub = cloud_high

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cloud_sub)

    # Run DBSCAN — eps=0.3 with min_samples=10 for tighter clusters
    db = DBSCAN(eps=0.3, min_samples=10)
    y_db = db.fit_predict(X_scaled)

    unique_labels = np.unique(y_db)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(y_db == -1)

    time_axis = np.arange(n_time) * SLIDING / packet_rate
    total_duration = time_axis[-1]

    print(f"\n{'='*60}")
    print(f"  DBSCAN Quantification: {label}")
    print(f"{'='*60}")
    print(f"  Points used:   {len(cloud_sub)} (top 25% energy, subsampled)")
    print(f"  Clusters:      {n_clusters}")
    print(f"  Noise points:  {n_noise} ({n_noise/len(cloud_sub)*100:.1f}%)")

    for cl in unique_labels:
        m = y_db == cl
        avg_t = np.mean(cloud_sub[m, 0])
        avg_f = np.mean(np.abs(cloud_sub[m, 1]))
        avg_e = np.mean(cloud_sub[m, 2])
        tag = "Noise" if cl == -1 else f"Cluster {cl}"
        print(f"    {tag:12s}: {np.sum(m):5d} pts | "
              f"time={avg_t:.1f}s | |freq|={avg_f:.1f}Hz | energy={avg_e:.2f}dB")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].imshow(10 * np.log10(doppler_matrix.T + 1e-10),
                   aspect='auto', cmap='jet', origin='lower',
                   extent=[0, total_duration, -MAX_DOPPLER_HZ, MAX_DOPPLER_HZ])
    axes[0].set_title(f"{label}\nOriginal Doppler Spectrogram")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Doppler Freq [Hz]")

    cmap_list = ['gray', 'lime', 'magenta', 'cyan', 'yellow', 'orange',
                 'dodgerblue', 'red', 'white', 'pink']
    for i, cl in enumerate(unique_labels):
        m = y_db == cl
        c = cmap_list[i % len(cmap_list)]
        name = "Noise" if cl == -1 else f"Cluster {cl}"
        axes[1].scatter(cloud_sub[m, 0], cloud_sub[m, 1],
                        c=c, s=2, alpha=0.6, label=name)
    axes[1].set_title(f"{label}\nDBSCAN (eps=0.3, top 25% energy)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Doppler Freq [Hz]")
    axes[1].legend(loc='upper right', markerscale=4, fontsize=8)
    axes[1].set_xlim(0, total_duration)
    axes[1].set_ylim(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ)

    plt.tight_layout()
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"dbscan_quantification_{safe}.png"
    plt.savefig(fname, dpi=150)
    print(f"  Plot saved -> {fname}")
    plt.close()


# =============================================================================
# Comparison helper
# =============================================================================
def print_comparison(r_on, r_off, title):
    """Pretty-print a side-by-side comparison table for Fan ON vs Fan OFF."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Metric':<30s}  {'Fan ON':>14s}  {'Fan OFF':>14s}")
    print(f"  {'-'*62}")
    print(f"  {'Duration (s)':<30s}  {r_on['duration']:>14.1f}  {r_off['duration']:>14.1f}")
    print(f"  {'Silhouette Score':<30s}  {r_on['silhouette']:>14.3f}  {r_off['silhouette']:>14.3f}")
    print(f"  {'Motion Energy (dB)':<30s}  {r_on['motion_energy']:>14.2f}  {r_off['motion_energy']:>14.2f}")
    print(f"  {'Still Energy (dB)':<30s}  {r_on['still_energy']:>14.2f}  {r_off['still_energy']:>14.2f}")
    print(f"  {'Energy Contrast (dB)':<30s}  {r_on['energy_contrast']:>14.2f}  {r_off['energy_contrast']:>14.2f}")
    print(f"  {'Motion Duty Cycle (%)':<30s}  {r_on['duty_cycle']:>14.1f}  {r_off['duty_cycle']:>14.1f}")
    print(f"  {'Dominant |Doppler| (Hz)':<30s}  {r_on['dominant_freq']:>14.2f}  {r_off['dominant_freq']:>14.2f}")

    n_on  = np.sum(r_on['window_scores'] > 50)
    n_off = np.sum(r_off['window_scores'] > 50)
    print(f"  {'Motion Windows (>50%)':<30s}  {n_on:>10d}/{len(r_on['window_scores']):<3d}  {n_off:>10d}/{len(r_off['window_scores']):<3d}")

    # Delta summary
    delta_duty  = r_on['duty_cycle'] - r_off['duty_cycle']
    delta_econ  = r_on['energy_contrast'] - r_off['energy_contrast']
    delta_freq  = r_on['dominant_freq'] - r_off['dominant_freq']
    print(f"\n  {'--- Delta (ON − OFF) ---':}")
    print(f"  {'Δ Duty Cycle':<30s}  {delta_duty:>+14.1f} pp")
    print(f"  {'Δ Energy Contrast':<30s}  {delta_econ:>+14.2f} dB")
    print(f"  {'Δ Dominant |Doppler|':<30s}  {delta_freq:>+14.2f} Hz")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------
    # Define ALL experiments
    # -----------------------------------------------------------------
    EXPERIMENTS = {
        # Original walking data
        "wa1_normal":    {"file": "doppler_output/wa1/wa1_normal_stream_0.txt",
                          "rate": 10.4,  "label": "Walking 1 (Fan ON)"},
        "wa2_second":    {"file": "doppler_output/wa2/wa2_second_stream_0.txt",
                          "rate": 10.3,  "label": "Walking 2 (Fan OFF)"},
        # 5 GHz Fan-OFF
        "5g_off_mot_los":{"file": "doppler_output/5g_/5g_off_mot_los_stream_0.txt",
                          "rate": 95.7,  "label": "5G Fan-OFF Motion LOS"},
        "5g_off_per_los":{"file": "doppler_output/5g_/5g_off_per_los_stream_0.txt",
                          "rate": 100.2, "label": "5G Fan-OFF Person LOS"},
        "5g_off_per":    {"file": "doppler_output/5g_/5g_off_per_stream_0.txt",
                          "rate": 107.4, "label": "5G Fan-OFF Person"},
        # 5 GHz Fan-ON
        "5g_on_mot_los": {"file": "doppler_output/5g_/5g_on_mot_los_stream_0.txt",
                          "rate": 94.9,  "label": "5G Fan-ON Motion LOS"},
        "5g_on_per_los": {"file": "doppler_output/5g_/5g_on_per_los_stream_0.txt",
                          "rate": 98.3,  "label": "5G Fan-ON Person LOS"},
        "5g_on_per":     {"file": "doppler_output/5g_/5g_on_per_stream_0.txt",
                          "rate": 97.6,  "label": "5G Fan-ON Person"},
    }

    # Fan-ON vs Fan-OFF comparison pairs
    COMPARISONS = [
        ("5g_on_mot_los", "5g_off_mot_los", "MOTION in LOS — Fan ON vs Fan OFF"),
        ("5g_on_per_los", "5g_off_per_los", "PERSON in LOS — Fan ON vs Fan OFF"),
        ("5g_on_per",     "5g_off_per",     "PERSON (not LOS) — Fan ON vs Fan OFF"),
        ("wa1_normal",    "wa2_second",     "WALKING — Fan ON vs Fan OFF (original)"),
    ]

    # -----------------------------------------------------------------
    # KMeans quantification for every experiment
    # -----------------------------------------------------------------
    km_results = {}
    for key, cfg in EXPERIMENTS.items():
        print(f"\n{'#'*60}")
        print(f"  Loading: {cfg['label']}")
        print(f"{'#'*60}")
        mat = load_doppler(cfg["file"])
        r = kmeans_quantify(mat, cfg["rate"], cfg["label"], window_sec=WINDOW_SEC)
        km_results[key] = r

    # -----------------------------------------------------------------
    # Comparative tables — Fan ON vs Fan OFF
    # -----------------------------------------------------------------
    for key_on, key_off, title in COMPARISONS:
        print_comparison(km_results[key_on], km_results[key_off], title)

    # -----------------------------------------------------------------
    # Grand summary table (all experiments at a glance)
    # -----------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print(f"  GRAND SUMMARY — ALL EXPERIMENTS")
    print(f"{'='*100}")
    header = (f"  {'Experiment':<28s}  {'Dur(s)':>7s}  {'Sil':>6s}  "
              f"{'Duty%':>7s}  {'ΔE(dB)':>8s}  {'|f|(Hz)':>8s}  {'MotWin':>8s}")
    print(header)
    print(f"  {'-'*94}")
    for key in ["5g_on_mot_los", "5g_off_mot_los",
                "5g_on_per_los", "5g_off_per_los",
                "5g_on_per",     "5g_off_per",
                "wa1_normal",    "wa2_second"]:
        r = km_results[key]
        n_mw = np.sum(r['window_scores'] > 50)
        tot  = len(r['window_scores'])
        print(f"  {r['label']:<28s}  {r['duration']:>7.1f}  {r['silhouette']:>6.3f}  "
              f"{r['duty_cycle']:>6.1f}%  {r['energy_contrast']:>8.2f}  "
              f"{r['dominant_freq']:>8.2f}  {n_mw:>4d}/{tot:<3d}")

    # -----------------------------------------------------------------
    # DBSCAN quantification for 5 GHz experiments
    # -----------------------------------------------------------------
    print(f"\n\n{'#'*60}")
    print(f"  DBSCAN Analysis (5 GHz experiments)")
    print(f"{'#'*60}")
    for key in ["5g_on_mot_los", "5g_off_mot_los",
                "5g_on_per_los", "5g_off_per_los",
                "5g_on_per",     "5g_off_per"]:
        cfg = EXPERIMENTS[key]
        mat = load_doppler(cfg["file"])
        dbscan_quantify(mat, cfg["rate"], cfg["label"])

    print("\nDone! All plots saved.")
