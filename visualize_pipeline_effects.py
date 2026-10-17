import argparse
import os
import pickle

LOCAL_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".viz_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", LOCAL_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", LOCAL_CACHE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from CSI_phase_sanitization_signal_preprocessing import hampel_filter_mmad_complex


DELETE_IDXS = np.asarray(
    [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255],
    dtype=int,
)
ALL_SUBCARRIERS = np.arange(256, dtype=int)
VALID_SUBCARRIER_IDXS = np.setdiff1d(ALL_SUBCARRIERS, DELETE_IDXS)
KNOWN_PACKET_RATES = {
    "empty_room": 72.0,
    "video_game": 153.0,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def split_stream_from_raw(csi_buff, stream, nss, ncore, start_idx):
    n_tot = nss * ncore
    end_idx = int(np.floor(csi_buff.shape[0] / n_tot))
    signal_stream = csi_buff[stream:end_idx * n_tot + 1:n_tot, :][start_idx:end_idx, :]
    signal_stream = signal_stream.copy()
    signal_stream[:, 64:] = -signal_stream[:, 64:]
    signal_stream = np.delete(signal_stream, DELETE_IDXS, axis=1)
    return signal_stream.T


def compute_step2_views(raw_stream):
    filtered = hampel_filter_mmad_complex(raw_stream, window_size=11, t=2.0)
    mean_signal = np.mean(np.abs(filtered), axis=0, keepdims=True)
    mean_signal[mean_signal == 0] = 1.0
    normalized = filtered / mean_signal
    return filtered, normalized


def db(x):
    return 20.0 * np.log10(np.abs(x) + 1e-10)


def wrapped_phase_delta(a, b):
    return np.angle(np.exp(1j * (np.angle(a) - np.angle(b))))


def unwrap_phase(x):
    return np.unwrap(np.angle(x), axis=1)


def percentile_limits(data, low=1.0, high=99.0):
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return None, None
    return np.percentile(finite, low), np.percentile(finite, high)


def plot_heatmap(ax, data, title, xlabel, ylabel, cmap="viridis", symmetric=False):
    if symmetric:
        vmax = np.nanpercentile(np.abs(data), 99.0)
        vmin = -vmax
    else:
        vmin, vmax = percentile_limits(data)
    image = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return image


def save_step2_figure(name, raw_stream, filtered_stream, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Step 2 Effect: Hampel Filtering for {name}", fontsize=14)

    amp_raw = db(raw_stream)
    amp_filtered = db(filtered_stream)
    amp_delta = np.abs(filtered_stream - raw_stream)

    phase_raw = unwrap_phase(raw_stream)
    phase_filtered = unwrap_phase(filtered_stream)
    phase_delta = wrapped_phase_delta(filtered_stream, raw_stream)

    images = [
        plot_heatmap(axes[0, 0], amp_raw, "Raw Amplitude (dB)", "Packet Index", "Subcarrier"),
        plot_heatmap(axes[0, 1], amp_filtered, "After Hampel Filter (dB)", "Packet Index", "Subcarrier"),
        plot_heatmap(axes[0, 2], amp_delta, "Amplitude Change |filtered - raw|", "Packet Index", "Subcarrier"),
        plot_heatmap(axes[1, 0], phase_raw, "Raw Phase (unwrapped)", "Packet Index", "Subcarrier", cmap="twilight"),
        plot_heatmap(axes[1, 1], phase_filtered, "After Hampel Filter Phase", "Packet Index", "Subcarrier", cmap="twilight"),
        plot_heatmap(
            axes[1, 2],
            phase_delta,
            "Wrapped Phase Change",
            "Packet Index",
            "Subcarrier",
            cmap="coolwarm",
            symmetric=True,
        ),
    ]

    for ax, image in zip(axes.flat, images):
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{name}_step2_hampel_effect.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_step3_figure(name, step2_signal, tr_matrix, save_dir):
    reconstructed_valid = tr_matrix[VALID_SUBCARRIER_IDXS, :]
    common_time = min(step2_signal.shape[1], reconstructed_valid.shape[1])
    step2_signal = step2_signal[:, :common_time]
    reconstructed_valid = reconstructed_valid[:, :common_time]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Step 3 Effect: LASSO Reconstruction for {name}", fontsize=14)

    amp_step2 = db(step2_signal)
    amp_step3 = db(reconstructed_valid)
    amp_residual = np.abs(reconstructed_valid - step2_signal)

    phase_step2 = unwrap_phase(step2_signal)
    phase_step3 = unwrap_phase(reconstructed_valid)
    phase_residual = wrapped_phase_delta(reconstructed_valid, step2_signal)

    images = [
        plot_heatmap(axes[0, 0], amp_step2, "Step 2 Output Amplitude (dB)", "Packet Index", "Observed Subcarrier"),
        plot_heatmap(axes[0, 1], amp_step3, "Step 3 Reconstructed Amplitude (dB)", "Packet Index", "Observed Subcarrier"),
        plot_heatmap(axes[0, 2], amp_residual, "Reconstruction Residual |step3 - step2|", "Packet Index", "Observed Subcarrier"),
        plot_heatmap(axes[1, 0], phase_step2, "Step 2 Output Phase", "Packet Index", "Observed Subcarrier", cmap="twilight"),
        plot_heatmap(axes[1, 1], phase_step3, "Step 3 Reconstructed Phase", "Packet Index", "Observed Subcarrier", cmap="twilight"),
        plot_heatmap(
            axes[1, 2],
            phase_residual,
            "Wrapped Phase Residual",
            "Packet Index",
            "Observed Subcarrier",
            cmap="coolwarm",
            symmetric=True,
        ),
    ]

    for ax, image in zip(axes.flat, images):
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{name}_step3_lasso_effect.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_doppler_figure(name, doppler_map, save_dir, packet_rate=None, sliding=1):
    fig, ax = plt.subplots(figsize=(12, 5))
    doppler_db = 10.0 * np.log10(doppler_map.T + 1e-10)

    if packet_rate and packet_rate > 0:
        duration = doppler_map.shape[0] * sliding / packet_rate
        extent = [0, duration, -50, 50]
        xlabel = "Time [s]"
    else:
        extent = [0, doppler_map.shape[0], -50, 50]
        xlabel = "Frame Index"

    image = ax.imshow(
        doppler_db,
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=extent,
    )
    ax.set_title(f"Final Doppler Heatmap for {name}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Doppler Bin")
    fig.colorbar(image, ax=ax, label="Power [dB]")
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"{name}_doppler_heatmap.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_metrics(raw_stream, filtered_stream, step2_signal, tr_matrix, doppler_map):
    reconstructed_valid = tr_matrix[VALID_SUBCARRIER_IDXS, :]
    common_time = min(step2_signal.shape[1], reconstructed_valid.shape[1])
    step2_signal = step2_signal[:, :common_time]
    reconstructed_valid = reconstructed_valid[:, :common_time]

    step2_changed = np.abs(filtered_stream - raw_stream) > 1e-12
    step2_change_ratio = 100.0 * np.mean(step2_changed)
    step2_relative_delta = np.linalg.norm(filtered_stream - raw_stream) / (np.linalg.norm(raw_stream) + 1e-12)

    step3_residual = reconstructed_valid - step2_signal
    step3_relative_error = np.linalg.norm(step3_residual) / (np.linalg.norm(step2_signal) + 1e-12)
    step3_amplitude_corr = np.corrcoef(
        np.abs(step2_signal).ravel(),
        np.abs(reconstructed_valid).ravel(),
    )[0, 1]

    doppler_peak_db = np.max(10.0 * np.log10(doppler_map + 1e-10))
    doppler_mean_db = np.mean(10.0 * np.log10(doppler_map + 1e-10))

    return {
        "step2_changed_percent": step2_change_ratio,
        "step2_relative_delta": step2_relative_delta,
        "step3_relative_error": step3_relative_error,
        "step3_amplitude_correlation": step3_amplitude_corr,
        "doppler_peak_db": doppler_peak_db,
        "doppler_mean_db": doppler_mean_db,
    }


def write_metrics(name, metrics, save_dir):
    out_path = os.path.join(save_dir, f"{name}_metrics.txt")
    lines = [
        f"Experiment: {name}",
        "Step 2 note: the current preprocessing script applies hampel_filter_mmad_complex, not the older hampel_filter.",
        f"Step 2 changed samples (%): {metrics['step2_changed_percent']:.4f}",
        f"Step 2 relative delta ||filtered-raw|| / ||raw||: {metrics['step2_relative_delta']:.6f}",
        f"Step 3 relative reconstruction error ||step3-step2|| / ||step2||: {metrics['step3_relative_error']:.6f}",
        f"Step 3 amplitude correlation: {metrics['step3_amplitude_correlation']:.6f}",
        f"Doppler peak power (dB): {metrics['doppler_peak_db']:.4f}",
        f"Doppler mean power (dB): {metrics['doppler_mean_db']:.4f}",
    ]
    with open(out_path, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_path


def analyze_experiment(args, name):
    raw_path = os.path.join(args.raw_dir, f"{name}.mat")
    step2_path = os.path.join(args.phase_dir, f"signal_{name}.txt")
    step3_path = os.path.join(args.phase_dir, f"Tr_vector_{name}_stream_{args.stream}.txt")
    doppler_subdir = os.path.join(args.doppler_dir, name[:3])
    doppler_path = os.path.join(doppler_subdir, f"{name}_stream_{args.stream}.txt")

    required_paths = [raw_path, step2_path, step3_path, doppler_path]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        print(f"\n{name}")
        print("  skipped: missing required files")
        for missing_path in missing:
            print(f"  missing -> {missing_path}")
        return

    raw_mat = sio.loadmat(raw_path)["csi_buff"]
    raw_stream = split_stream_from_raw(raw_mat, args.stream, args.nss, args.ncore, args.start_idx)
    filtered_stream, normalized_stream = compute_step2_views(raw_stream)

    step2_saved = load_pickle(step2_path)[:, :, args.stream]
    tr_matrix = load_pickle(step3_path)
    doppler_map = load_pickle(doppler_path)

    common_time = min(normalized_stream.shape[1], step2_saved.shape[1])
    normalized_stream = normalized_stream[:, :common_time]
    step2_saved = step2_saved[:, :common_time]

    if not np.allclose(normalized_stream, step2_saved, atol=1e-6, rtol=1e-5):
        print(f"[warn] recomputed step 2 output does not exactly match saved file for {name}; plotting saved file for downstream comparison.")

    packet_rate = args.packet_rate
    if packet_rate is None:
        packet_rate = KNOWN_PACKET_RATES.get(name)

    step2_figure = save_step2_figure(name, raw_stream, filtered_stream, args.save_dir)
    step3_figure = save_step3_figure(name, step2_saved, tr_matrix, args.save_dir)
    doppler_figure = save_doppler_figure(name, doppler_map, args.save_dir, packet_rate=packet_rate, sliding=args.sliding)

    metrics = build_metrics(raw_stream, filtered_stream, step2_saved, tr_matrix, doppler_map)
    metrics_path = write_metrics(name, metrics, args.save_dir)

    print(f"\n{name}")
    print(f"  step2 figure : {step2_figure}")
    print(f"  step3 figure : {step3_figure}")
    print(f"  doppler      : {doppler_figure}")
    print(f"  metrics      : {metrics_path}")
    print(f"  step2 changed samples (%) : {metrics['step2_changed_percent']:.4f}")
    print(f"  step3 relative error      : {metrics['step3_relative_error']:.6f}")
    print(f"  step3 amplitude corr      : {metrics['step3_amplitude_correlation']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the effect of step 2 preprocessing, step 3 LASSO reconstruction, and final Doppler heatmaps."
    )
    parser.add_argument(
        "--names",
        default="",
        help="Comma-separated experiment names without .mat extension. Leave empty to auto-detect complete experiments.",
    )
    parser.add_argument("--raw-dir", default="raw_mat_data", help="Directory containing raw .mat files")
    parser.add_argument("--phase-dir", default="phase_processing", help="Directory containing step 2 and step 3 pickle outputs")
    parser.add_argument("--doppler-dir", default="doppler_output", help="Directory containing Doppler pickle outputs")
    parser.add_argument("--save-dir", default="analysis_plots", help="Directory to save generated figures")
    parser.add_argument("--nss", type=int, default=1, help="Number of spatial streams")
    parser.add_argument("--ncore", type=int, default=1, help="Number of cores")
    parser.add_argument("--stream", type=int, default=0, help="Stream index to visualize")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index used by step 2 preprocessing")
    parser.add_argument("--sliding", type=int, default=1, help="Sliding parameter used in Doppler computation")
    parser.add_argument("--packet-rate", type=float, default=None, help="Optional packet rate for Doppler time axis")
    return parser.parse_args()


def detect_complete_experiments(args):
    candidates = []
    for filename in os.listdir(args.phase_dir):
        prefix = f"Tr_vector_"
        suffix = f"_stream_{args.stream}.txt"
        if filename.startswith(prefix) and filename.endswith(suffix):
            candidates.append(filename[len(prefix):-len(suffix)])

    complete = []
    for name in sorted(set(candidates)):
        raw_path = os.path.join(args.raw_dir, f"{name}.mat")
        step2_path = os.path.join(args.phase_dir, f"signal_{name}.txt")
        doppler_path = os.path.join(args.doppler_dir, name[:3], f"{name}_stream_{args.stream}.txt")
        if os.path.exists(raw_path) and os.path.exists(step2_path) and os.path.exists(doppler_path):
            complete.append(name)
    return complete


def main():
    args = parse_args()
    ensure_dir(args.save_dir)
    if args.names.strip():
        names = [name.strip() for name in args.names.split(",") if name.strip()]
    else:
        names = detect_complete_experiments(args)
        if not names:
            raise FileNotFoundError(
                "No complete experiments found. Expected raw, step2, step3, and doppler files for the same experiment."
            )
    for name in names:
        analyze_experiment(args, name)


if __name__ == "__main__":
    main()
