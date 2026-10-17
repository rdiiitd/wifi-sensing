"""
run_full_testcases.py
=====================
Runs the complete SHARP CSI pipeline (extraction → preprocessing → H-estimation
→ signal reconstruction) for both test-case PCAP files in FULL mode
(no sample limit).

Output:
  raw_mat_data/empty_room.mat          ← raw complex CSI matrix
  raw_mat_data/video_game.mat
  final_output/emp/empty_room_stream_0.mat  ← sanitised (amplitude + phase)
  final_output/vid/video_game_stream_0.mat
"""

import os
import sys
import subprocess

# ── add project root to path so we can import extracter ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extracter
import scipy.io as sio

# ── configuration ─────────────────────────────────────────────────────────────
RUNS = [
    {
        "pcap":  "data/pcaps/csi-testcse1.pcap",
        "name":  "empty_room",
        "prefix": "emp",
    },
    {
        "pcap":  "data/pcaps/csi-testcase2 (1).pcap",
        "name":  "video_game",
        "prefix": "vid",
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────
def stale_mat_size_bytes(mat_path):
    """Return file size in bytes, or 0 if file doesn't exist."""
    try:
        return os.path.getsize(mat_path)
    except FileNotFoundError:
        return 0


def remove_if_stale(mat_path, pcap_path, label):
    """
    If the sanitised .mat exists but is suspiciously small
    (< 10 MB — a sign it was generated from a short test run),
    delete it so the pipeline re-generates it from the full data.
    """
    if not os.path.isfile(mat_path):
        return
    sz = os.path.getsize(mat_path)
    # A full run for empty_room produces ~100 MB, video_game ~250 MB.
    # Anything under 10 MB is clearly a test-mode artefact.
    if sz < 10 * 1024 * 1024:
        print(f"   [STALE] {label}: {mat_path} is only {sz // 1024} KB — removing.")
        os.remove(mat_path)
    else:
        print(f"   [OK]   {label}: {mat_path} already exists ({sz // (1024*1024)} MB) — skipping.")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("raw_mat_data",    exist_ok=True)
    os.makedirs("phase_processing", exist_ok=True)
    os.makedirs("final_output",    exist_ok=True)

    for run in RUNS:
        pcap   = run["pcap"]
        name   = run["name"]
        prefix = run["prefix"]

        raw_mat   = f"raw_mat_data/{name}.mat"
        final_mat = f"final_output/{prefix}/{name}_stream_0.mat"

        print(f"\n{'='*60}")
        print(f"  Processing: {name}  ({pcap})")
        print(f"{'='*60}")

        # ── Remove stale sanitised output if it was from a test run ──────────
        remove_if_stale(final_mat, pcap, name)

        # ── Step 1: Extraction (skip if raw mat already exists and is full) ──
        if os.path.isfile(raw_mat) and os.path.getsize(raw_mat) > 10 * 1024 * 1024:
            print(f"1. Raw mat already exists ({os.path.getsize(raw_mat) // (1024*1024)} MB). Skipping extraction.")
        else:
            print(f"1. Extracting raw CSI from {pcap} ...")
            samples = extracter.read_pcap(pcap)
            csi_matrix = samples.csi
            sio.savemat(raw_mat, {"csi_buff": csi_matrix})
            print(f"   -> {csi_matrix.shape[0]:,} samples saved to {raw_mat}")

        # ── Step 2: Signal preprocessing ─────────────────────────────────────
        # Always re-run preprocessing — it is fast and idempotent.
        print("2. Running Signal Preprocessing ...")
        subprocess.run([
            "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_phase_sanitization_signal_preprocessing.py"),
            "raw_mat_data/", "0", name, "1", "1", "0"
        ], check=True)

        # ── Step 3: H estimation ──────────────────────────────────────────────
        print("3. Running H Estimation ...")
        subprocess.run([
            "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_phase_sanitization_H_estimation.py"),
            "raw_mat_data/", "0", name, "1", "1", "0", "-1"
        ], check=True)

        # ── Step 4: Signal reconstruction (produces the sanitised .mat) ───────
        # The reconstruction script skips names whose output already exists.
        # We deleted the stale one above, so it will regenerate it now.
        print("4. Running Signal Reconstruction ...")
        subprocess.run([
            "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_phase_sanitization_signal_reconstruction.py"),
            "./phase_processing/", "./final_output/", "1", "1", "0", "0"
        ], check=True)

        # ── Verify ────────────────────────────────────────────────────────────
        if os.path.isfile(final_mat):
            sz = os.path.getsize(final_mat)
            print(f"\n✅  Sanitised matrix ready: {final_mat}  ({sz // (1024*1024)} MB)")
        else:
            print(f"\n❌  ERROR: Expected output not found: {final_mat}")

    print("\n\n🎉  Both test-case pipelines complete!")
    print("    Sanitised matrices:")
    for run in RUNS:
        p = f"final_output/{run['prefix']}/{run['name']}_stream_0.mat"
        sz = os.path.getsize(p) // (1024*1024) if os.path.isfile(p) else -1
        print(f"      {p}  ({sz} MB)")


if __name__ == "__main__":
    main()
