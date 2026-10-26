import os
import subprocess
from extracter import run_pipeline, choose_sample_length

# Configuration for the 2 files
FILES = [
    {
        "pcap": "data/pcaps/5GHz-fan-on-single-motion-in-the-LOS.pcap",
        "name": "5g_on_mot_los",
        "rate": 94.9
    },
    {
        "pcap": "data/pcaps/5GHz-fan-off-single-motion-in-the-LOS.pcap",
        "name": "5g_off_mot_los",
        "rate": 95.7
    }
]

DOPPLER_TARGET_WINDOW_SEC = 1.0
DOPPLER_SLIDING = 1
DOPPLER_NOISE_LEVEL = -0.5

print("\n--- Running pipeline for 2 selected files (Full Data) ---")
for f in FILES:
    # 1. Skip extraction and signal processing if we already have the final sanitized matrix
    final_mat_path = f"final_output/{f['name'][:3]}/{f['name']}_stream_0.mat"
    
    if os.path.exists(final_mat_path):
        print(f"   -> Found existing sanitized matrix at {final_mat_path}! Skipping heavy pipeline processing.")
    else:
        run_pipeline(f["pcap"], f["name"], max_samples=4000)
    
    # 2. Compute Doppler Spectrogram
    print(f"\n--- Computing Doppler Spectrogram for {f['name']} ---")
    
    sample_length = choose_sample_length(f["rate"], DOPPLER_TARGET_WINDOW_SEC)
    
    # The SHARP doppler computation uses a prefix to filter files
    prefix = f["name"][:3] # "5g_"
    
    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_doppler_computation.py"),
        "final_output/", prefix, "./doppler_output/", "0", "0",
        str(sample_length), str(DOPPLER_SLIDING), str(DOPPLER_NOISE_LEVEL),
        "--packet-rate", str(f["rate"]), "--overwrite"
    ], check=True)

    # 3. Compute New Grouped Doppler Spectrogram (RU-based)
    print(f"\n--- Computing New Grouped Doppler Spectrogram for {f['name']} ---")
    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_doppler_computation_final_grouping.py"),
        "final_output/", prefix, "./doppler_output/", "0", "0",
        str(sample_length), str(DOPPLER_SLIDING), str(DOPPLER_NOISE_LEVEL),
        "RU1", "1", "26", "100"
    ], check=True)

print("\n--- Finished processing both files! ---")
