'''
Interleaved
===========

Fast and efficient methods to extract
Interleaved CSI samples in PCAP files.

~230k samples per second.

Suitable for bcm43455c0 and bcm4339 chips.

Requires Numpy.

Usage
-----

import extracter

samples = extracter.read_pcap('path_to_pcap_file')

Bandwidth is inferred from the pcap file, but
can also be explicitly set:
samples = extracter.read_pcap('path_to_pcap_file', bandwidth=40)
'''

__all__ = [
    'read_pcap'
]

import os
import subprocess
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Indexes of Null and Pilot OFDM subcarriers
# https://www.oreilly.com/library/view/80211ac-a-survival/9781449357702/ch02.html
nulls = {
    20: [x+32 for x in [
        -32, -31, -30, -29,
              31,  30,  29,  0
    ]],

    40: [x+64 for x in [
        -64, -63, -62, -61, -60, -59, -1, 
              63,  62,  61,  60,  59,  1,  0
    ]],

    80: [x+128 for x in [
        -128, -127, -126, -125, -124, -123, -1,
               127,  126,  125,  124,  123,  1,  0
    ]],

    160: [x+256 for x in [
        -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
               255,  254,  253,  252,  251,  129,  128,  127,  5,  4,  3,  3,  1,  0 
    ]]
}

pilots = {
    20: [x+32 for x in [
        -21, -7,
         21,  7
    ]],

    40: [x+64 for x in [
        -53, -25, -11, 
         53,  25,  11
    ]],

    80: [x+128 for x in [
        -103, -75, -39, -11,
         103,  75,  39,  11
    ]],

    160: [x+256 for x in [
        -231, -203, -167, -139, -117, -89, -53, -25,
         231,  203,  167,  139,  117,  89,  53,  25
    ]]
}

class SampleSet(object):
    '''
        A helper class to contain data read
        from pcap files.
    '''
    def __init__(self, samples, bandwidth):
        self.rssi, self.fctl, self.mac, self.seq, self.css, self.csi = samples

        self.nsamples = self.csi.shape[0]
        self.bandwidth = bandwidth
    
    def get_rssi(self, index):
        return self.rssi[index]
    
    def get_fctl(self, index):
        return self.fctl[index]

    def get_mac(self, index):
        return self.mac[index*6: (index+1)*6]

    def get_seq(self, index):
        sc = int.from_bytes( #uint16: SC
            self.seq[index*2: (index+1)*2],
            byteorder = 'little',
            signed = False
        )
        fn = sc % 16 # Fragment Number
        sc = int((sc - fn)/16) # Sequence Number

        return (sc, fn)
    
    def get_css(self, index):
        return self.css[index*2: (index+1)*2]

    def get_csi(self, index, rm_nulls=False, rm_pilots=False):
        csi = self.csi[index].copy()
        if rm_nulls:
            csi[nulls[self.bandwidth]] = 0
        if rm_pilots:
            csi[pilots[self.bandwidth]] = 0

        return csi
   
    def print(self, index):
        # Mac ID
        macid = self.get_mac(index).hex()
        macid = ':'.join([macid[i:i+2] for i in range(0, len(macid), 2)])

        # Sequence control
        sc, fn = self.get_seq(index)

        # Core and Spatial Stream
        css = self.get_css(index).hex()

        rssi = self.get_rssi(index)
        fctl = self.get_fctl(index)

        print(
            f'''
Sample #{index}
---------------
Source Mac ID: {macid}
Sequence: {sc}.{fn}
Core and Spatial Stream: 0x{css}
RSSI: {rssi}
FCTL: {fctl}
            '''
        )


def __find_bandwidth(incl_len):
    '''
        Determines bandwidth
        from length of packets.
        
        incl_len is the 4 bytes
        indicating the length of the
        packet in packet header
        https://wiki.wireshark.org/Development/LibpcapFileFormat/

        This function is immune to small
        changes in packet lengths.
    '''

    pkt_len = int.from_bytes(
        incl_len,
        byteorder='little',
        signed=False
    )

    # The number of bytes before we
    # have CSI data is 60. By adding
    # 128-60 to frame_len, bandwidth
    # will be calculated correctly even
    # if frame_len changes +/- 128
    # Some packets have zero padding.
    # 128 = 20 * 3.2 * 4
    nbytes_before_csi = 60
    pkt_len += (128 - nbytes_before_csi)

    bandwidth = 20 * int(
        pkt_len // (20 * 3.2 * 4)
    )

    return bandwidth



def __find_nsamples_max(pcap_filesize, nsub):
    '''
        Returns an estimate for the maximum possible number
        of samples in the pcap file.

        The size of the pcap file is divided by the size of
        a packet to calculate the number of samples. However,
        some packets have a padding of a few bytes, so the value
        returned is slightly higher than the actual number of
        samples in the pcap file.
    '''

    # PCAP global header is 24 bytes
    # PCAP packet header is 12 bytes
    # Ethernet + IP + UDP headers are 46 bytes
    # Nexmon metadata is 18 bytes
    # CSI is nsub*4 bytes long
    #
    # So each packet is 12 + 46 + 18 + nsub*4 bytes long
    nsamples_max = int(
        (pcap_filesize - 24) / (
            12 + 46 + 18 + (nsub*4)
        )
    )

    return nsamples_max

def read_pcap(pcap_filepath, bandwidth=0, nsamples_max=0):
    '''
        Reads CSI samples from
        a pcap file. A SampleSet
        object is returned.

        Bandwidth and maximum samples
        are inferred from the pcap file by
        default, but you can also set them explicitly.
    '''

    pcap_filesize = os.stat(pcap_filepath).st_size
    with open(pcap_filepath, 'rb') as pcapfile:
        fc = pcapfile.read()
    
    if bandwidth == 0:
        bandwidth = __find_bandwidth(
            # 32-36 is where the incl_len
            # bytes for the first frame are
            # located.
            # https://wiki.wireshark.org/Development/LibpcapFileFormat/
            fc[32:36]
        )
    # Number of OFDM sub-carriers
    nsub = int(bandwidth * 3.2)

    if nsamples_max == 0:
        nsamples_max = __find_nsamples_max(pcap_filesize, nsub)

    # Preallocating memory
    rssi = bytearray(nsamples_max * 1)
    fctl = bytearray(nsamples_max * 1)
    mac = bytearray(nsamples_max * 6)
    seq = bytearray(nsamples_max * 2)
    css = bytearray(nsamples_max * 2)
    csi = bytearray(nsamples_max * nsub * 4)

    # Pointer to current location in file.
    # This is faster than using file.tell()
    # =24 to skip pcap global header
    ptr = 24

    nsamples = 0
    while ptr < pcap_filesize:
        # Read frame header
        # Skip over Eth, IP, UDP
        ptr += 8
        frame_len = int.from_bytes(
            fc[ptr: ptr+4],
            byteorder='little',
            signed=False
        )
        ptr += 50

        # 2 bytes: Magic Bytes               @ 0 - 1
        # 1 bytes: RSSI                      @ 2 - 2
        # 1 bytes: FCTL                      @ 3 - 3
        # 6 bytes: Source Mac ID             @ 4 - 10
        # 2 bytes: Sequence Number           @ 10 - 12
        # 2 bytes: Core and Spatial Stream   @ 12 - 14
        # 2 bytes: ChanSpec                  @ 14 - 16
        # 2 bytes: Chip Version              @ 16 - 18
        # nsub*4 bytes: CSI Data             @ 18 - 18 + nsub*4

        rssi[nsamples] = fc[ptr+2]
        fctl[nsamples] = fc[ptr+3]
        mac[nsamples*6: (nsamples+1)*6] = fc[ptr+4: ptr+10]
        seq[nsamples*2: (nsamples+1)*2] = fc[ptr+10: ptr+12]
        css[nsamples*2: (nsamples+1)*2] = fc[ptr+12: ptr+14]
        csi[nsamples*(nsub*4): (nsamples+1)*(nsub*4)] = fc[ptr+18: ptr+18 + nsub*4]

        ptr += (frame_len - 42)
        nsamples += 1

    # Convert CSI bytes to numpy array
    csi_np = np.frombuffer(
        csi,
        dtype = np.int16,
        count = nsub * 2 * nsamples
    )

    # Cast numpy 1-d array to matrix
    csi_np = csi_np.reshape((nsamples, nsub * 2))

    # Convert csi into complex numbers
    csi_cmplx = np.fft.fftshift(
            csi_np[:nsamples, ::2] + 1.j * csi_np[:nsamples, 1::2], axes=(1,)
    )

    # Convert RSSI to Two's complement form
    rssi = np.frombuffer(rssi, dtype=np.int8, count = nsamples)

    return SampleSet(
        (
            rssi,
            fctl,
            mac,
            seq,
            css,
            csi_cmplx,
        ),
        bandwidth
    )

def extract_2sec_features(raw_mat_path, proc_mat_path, start_sec=0, packet_rate=173, interval_sec=2):
    """
    Extracts Amplitude (dB), Raw Phase, and Sanitized Phase for a 2-second window.
    """
    # Load the raw data (for raw phase) and processed data (for amplitude and sanitized phase)
    raw_data = sio.loadmat(raw_mat_path)['csi_buff']
    proc_data = sio.loadmat(proc_mat_path)['csi_matrix_processed']
    
    # Calculate array indices for the 2-second slice
    start_idx = int(start_sec * packet_rate)
    end_idx = int(start_idx + (interval_sec * packet_rate))
    
    # Ensure we don't go out of bounds
    end_idx = min(end_idx, raw_data.shape[0], proc_data.shape[0])
    
    # Extract the processed slice (Shape: time_samples, subcarriers, 2)
    # Index 0 is Amplitude, Index 1 is Sanitized Phase
    proc_slice = proc_data[start_idx:end_idx, :, :]
    
    # Calculate Amplitude in dB scale to match the paper's visuals
    amplitude_db = 20 * np.log10(proc_slice[:, :, 0] + 1e-10)
    
    sanitized_phase = proc_slice[:, :, 1]
    
    # Extract the raw phase directly from the raw complex matrix
    # We slice [:, 6:-5] to match the 245 subcarriers kept by the SHARP script
    raw_slice_complex = raw_data[start_idx:end_idx, 6:-5] 
    raw_phase = np.unwrap(np.angle(raw_slice_complex), axis=0)
    
    # Transpose all matrices so Sub-channels are on the Y-axis and Time is on the X-axis
    return amplitude_db.T, raw_phase.T, sanitized_phase.T

def plot_fig1_comparison(empty_raw_mat, empty_proc_mat, video_raw_mat, video_proc_mat,
                         empty_start_sec=0, video_start_sec=0,
                         empty_packet_rate=72, video_packet_rate=153):
    """
    Plots a 3x2 grid replicating Figure 1 from the SHARP paper.

    Args:
        empty_packet_rate : packets/sec for the empty room recording (default: 72)
        video_packet_rate : packets/sec for the video game recording (default: 153)
    """
    # Extract features using the correct packet rate for each recording
    amp_empty, raw_ph_empty, san_ph_empty = extract_2sec_features(
        empty_raw_mat, empty_proc_mat, empty_start_sec, packet_rate=empty_packet_rate)
    amp_video, raw_ph_video, san_ph_video = extract_2sec_features(
        video_raw_mat, video_proc_mat, video_start_sec, packet_rate=video_packet_rate)

    # Compute actual time duration of each slice (may be < 2s if near end of recording)
    empty_duration = amp_empty.shape[1] / empty_packet_rate
    video_duration = amp_video.shape[1] / video_packet_rate

    # Create the 3x2 subplot grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # --- ROW 1: AMPLITUDE ---
    im00 = axs[0, 0].imshow(amp_empty, aspect='auto', cmap='viridis', origin='lower',
                             extent=[0, empty_duration, -122, 122])
    axs[0, 0].set_title("Empty Room: Amplitude (dB)")
    axs[0, 0].set_ylabel("Sub-channel")
    fig.colorbar(im00, ax=axs[0, 0])

    im01 = axs[0, 1].imshow(amp_video, aspect='auto', cmap='viridis', origin='lower',
                             extent=[0, video_duration, -122, 122])
    axs[0, 1].set_title("Video Game: Amplitude (dB)")
    fig.colorbar(im01, ax=axs[0, 1])

    # --- ROW 2: RAW PHASE ---
    im10 = axs[1, 0].imshow(raw_ph_empty, aspect='auto', cmap='viridis', origin='lower',
                             extent=[0, empty_duration, -122, 122])
    axs[1, 0].set_title("Empty Room: Raw Phase")
    axs[1, 0].set_ylabel("Sub-channel")
    fig.colorbar(im10, ax=axs[1, 0])

    im11 = axs[1, 1].imshow(raw_ph_video, aspect='auto', cmap='viridis', origin='lower',
                             extent=[0, video_duration, -122, 122])
    axs[1, 1].set_title("Video Game: Raw Phase")
    fig.colorbar(im11, ax=axs[1, 1])

    # --- ROW 3: SANITIZED PHASE ---
    im20 = axs[2, 0].imshow(san_ph_empty, aspect='auto', cmap='viridis', origin='lower',
                             extent=[0, empty_duration, -122, 122])
    axs[2, 0].set_title("Empty Room: Sanitized Phase")
    axs[2, 0].set_ylabel("Sub-channel")
    axs[2, 0].set_xlabel("Time [s]")
    fig.colorbar(im20, ax=axs[2, 0])

    im21 = axs[2, 1].imshow(san_ph_video, aspect='auto', cmap='viridis', origin='lower',
                             extent=[0, video_duration, -122, 122])
    axs[2, 1].set_title("Video Game: Sanitized Phase")
    axs[2, 1].set_xlabel("Time [s]")
    fig.colorbar(im21, ax=axs[2, 1])

    plt.tight_layout()
    plt.show()



def plot_doppler_comparison(empty_doppler_path, video_doppler_path,
                            empty_packet_rate=72, video_packet_rate=153,
                            empty_sample_length=15, video_sample_length=31,
                            sliding=1):
    """
    Plots Doppler spectrograms side-by-side for empty room vs video game.

    Args:
        empty_doppler_path : path to the empty room doppler .txt pickle file
        video_doppler_path : path to the video game doppler .txt pickle file
        empty_packet_rate  : pps of the empty room recording (default: 72)
        video_packet_rate  : pps of the video game recording (default: 153)
        empty_sample_length: Doppler window length in packets for empty room
        video_sample_length: Doppler window length in packets for activity recording
        sliding            : sliding step used during computation (default: 1)
    """
    import pickle

    with open(empty_doppler_path, 'rb') as f:
        doppler_empty = pickle.load(f)   # shape: (n_frames, 100)
    with open(video_doppler_path, 'rb') as f:
        doppler_video = pickle.load(f)   # shape: (n_frames, 100)

    # Use the center time of each Doppler window instead of the window start.
    empty_window_sec = empty_sample_length / empty_packet_rate
    video_window_sec = video_sample_length / video_packet_rate
    empty_start_time = 0.5 * empty_window_sec
    video_start_time = 0.5 * video_window_sec
    empty_end_time = empty_start_time + max(doppler_empty.shape[0] - 1, 0) * sliding / empty_packet_rate
    video_end_time = video_start_time + max(doppler_video.shape[0] - 1, 0) * sliding / video_packet_rate

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axs[0].imshow(
        10 * np.log10(doppler_empty.T + 1e-10),  # dB scale, transposed so velocity is Y
        aspect='auto', cmap='jet', origin='lower',
        extent=[empty_start_time, empty_end_time, -50, 50]
    )
    axs[0].set_title("Empty Room: Doppler Spectrogram")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Doppler Frequency [Hz]")
    fig.colorbar(im0, ax=axs[0], label='Power [dB]')

    im1 = axs[1].imshow(
        10 * np.log10(doppler_video.T + 1e-10),
        aspect='auto', cmap='jet', origin='lower',
        extent=[video_start_time, video_end_time, -50, 50]
    )
    axs[1].set_title("Video Game: Doppler Spectrogram")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Doppler Frequency [Hz]")
    fig.colorbar(im1, ax=axs[1], label='Power [dB]')

    plt.suptitle("Doppler Velocity Spectrogram — Empty vs Activity", fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Pipeline Stage Comparison — Effect of each step
# =============================================================================

def plot_pipeline_stages(experiment_name, packet_rate, start_sec=0, interval_sec=2,
                         sample_length=None, sliding=1):
    """
    Visualises the effect of each pipeline stage on the CSI data.

    Layout (4 rows × 2 cols):
        Row 1: Raw CSI               — Amplitude (dB)  |  Raw Phase
        Row 2: After Hampel (Step 2) — Amplitude (dB)  |  Phase
        Row 3: After LASSO  (Step 4) — Amplitude (dB)  |  Sanitized Phase
        Row 4: Doppler Spectrogram   — (spans both columns)
    """
    import pickle

    prefix = experiment_name[:3]  # emp or vid
    raw_mat_path     = f"raw_mat_data/{experiment_name}.mat"
    signal_txt_path  = f"phase_processing/signal_{experiment_name}.txt"
    proc_mat_path    = f"final_output/{prefix}/{experiment_name}_stream_0.mat"
    doppler_txt_path = f"doppler_output/{prefix}/{experiment_name}_stream_0.txt"

    # Null subcarrier indices (same as used in preprocessing)
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)

    # --- Time window ---
    start_idx = int(start_sec * packet_rate)
    end_idx   = int(start_idx + interval_sec * packet_rate)

    # ======================= STAGE 1: Raw CSI ============================
    raw = sio.loadmat(raw_mat_path)['csi_buff']                # (N, 256) complex
    end1 = min(end_idx, raw.shape[0])
    raw_slice = np.delete(raw[start_idx:end1, :], delete_idxs, axis=1)  # (T, 242)
    raw_amp   = 20 * np.log10(np.abs(raw_slice) + 1e-10).T             # (242, T)
    raw_phase = np.unwrap(np.angle(raw_slice), axis=0).T               # (242, T)

    # ===================== STAGE 2: After Hampel =========================
    with open(signal_txt_path, 'rb') as f:
        signal = pickle.load(f)           # (242, N_total, streams)
    sig_s0 = signal[:, :, 0]              # stream 0 → (242, N_total)
    end2 = min(end_idx, sig_s0.shape[1])
    sig_slice    = sig_s0[:, start_idx:end2]                           # (242, T)
    hampel_amp   = 20 * np.log10(np.abs(sig_slice) + 1e-10)           # (242, T)
    hampel_phase = np.unwrap(np.angle(sig_slice), axis=1)              # (242, T)

    # ===================== STAGE 3: After LASSO ==========================
    proc = sio.loadmat(proc_mat_path)['csi_matrix_processed']  # (N, 245, 2)
    end3 = min(end_idx, proc.shape[0])
    proc_slice   = proc[start_idx:end3, :, :]
    lasso_amp    = (20 * np.log10(proc_slice[:, :, 0] + 1e-10)).T     # (245, T)
    lasso_phase  = proc_slice[:, :, 1].T                               # (245, T)

    # ======================== Doppler ====================================
    with open(doppler_txt_path, 'rb') as f:
        doppler = pickle.load(f)          # (frames, 100)
    if sample_length is None:
        sample_length = choose_sample_length(packet_rate)
    doppler_window_sec = sample_length / packet_rate
    doppler_start_time = start_sec + 0.5 * doppler_window_sec
    doppler_end_time = doppler_start_time + max(doppler.shape[0] - 1, 0) * sliding / packet_rate

    # ======================== PLOTTING ===================================
    duration = (end1 - start_idx) / packet_rate
    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.30)

    titles = [
        ("Raw: Amplitude (dB)",          "Raw: Phase"),
        ("After Hampel: Amplitude (dB)", "After Hampel: Phase"),
        ("After LASSO: Amplitude (dB)",  "After LASSO: Sanitized Phase"),
    ]
    data_pairs = [
        (raw_amp,    raw_phase),
        (hampel_amp, hampel_phase),
        (lasso_amp,  lasso_phase),
    ]

    for row, ((amp, phase), (t_amp, t_phase)) in enumerate(zip(data_pairs, titles)):
        ax_l = fig.add_subplot(gs[row, 0])
        im_l = ax_l.imshow(amp, aspect='auto', cmap='viridis', origin='lower',
                           extent=[0, duration, 0, amp.shape[0]])
        ax_l.set_title(t_amp)
        ax_l.set_ylabel("Subcarrier")
        fig.colorbar(im_l, ax=ax_l)

        ax_r = fig.add_subplot(gs[row, 1])
        im_r = ax_r.imshow(phase, aspect='auto', cmap='viridis', origin='lower',
                           extent=[0, duration, 0, phase.shape[0]])
        ax_r.set_title(t_phase)
        fig.colorbar(im_r, ax=ax_r)

    # Row 4: Doppler (full width)
    ax_d = fig.add_subplot(gs[3, :])
    im_d = ax_d.imshow(10 * np.log10(doppler.T + 1e-10),
                       aspect='auto', cmap='jet', origin='lower',
                       extent=[doppler_start_time, doppler_end_time, -50, 50])
    ax_d.set_title("Doppler Spectrogram")
    ax_d.set_xlabel("Time [s]")
    ax_d.set_ylabel("Doppler Freq [Hz]")
    fig.colorbar(im_d, ax=ax_d, label='Power [dB]')

    fig.suptitle(f"Pipeline Stages — {experiment_name}", fontsize=15, fontweight='bold', y=1.0)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.show()


# =============================================================================
# Full SHARP Pipeline
# =============================================================================

def run_pipeline(pcap_filename, experiment_name, max_samples=0):
    '''
    Runs the complete extraction + SHARP preprocessing pipeline.

    Args:
        max_samples : if > 0, only process this many packets (test mode). 0 = all.
    '''
    print(f"\n--- Starting Pipeline for {experiment_name} ---")
    if max_samples > 0:
        print(f"   [TEST MODE] Processing only {max_samples} samples")

    # ---------------------------------------------------------
    # STEP 1: Extraction
    # ---------------------------------------------------------
    print("1. Extracting raw CSI from PCAP...")
    samples = read_pcap(pcap_filename)
    csi_matrix = samples.csi
    if max_samples > 0:
        csi_matrix = csi_matrix[:max_samples, :]

    os.makedirs('raw_mat_data',    exist_ok=True)
    os.makedirs('phase_processing', exist_ok=True)
    os.makedirs('final_output',    exist_ok=True)

    mat_filepath = f"raw_mat_data/{experiment_name}.mat"
    sio.savemat(mat_filepath, {'csi_buff': csi_matrix})
    print(f"   -> Extracted {csi_matrix.shape[0]} samples. Saved to {mat_filepath}")

    # ---------------------------------------------------------
    # STEP 2: Preprocessing
    # ---------------------------------------------------------
    print("2. Running Signal Preprocessing...")
    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_phase_sanitization_signal_preprocessing.py"),
        "raw_mat_data/", "0", experiment_name, "1", "1", "0"
    ], check=True)

    # ---------------------------------------------------------
    # STEP 3: H Estimation
    # ---------------------------------------------------------
    print("3. Running H Estimation (Isolating the strongest path)...")
    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_phase_sanitization_H_estimation.py"),
        "raw_mat_data/", "0", experiment_name, "1", "1", "0", "-1"
    ], check=True)

    # ---------------------------------------------------------
    # STEP 4: Signal Reconstruction
    # ---------------------------------------------------------
    print("4. Running Signal Reconstruction...")
    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_phase_sanitization_signal_reconstruction.py"),
        "./phase_processing/", "./final_output/", "1", "1", "0", "0"
    ], check=True)
    print(f"--- Pipeline Complete for {experiment_name}! ---")


def choose_sample_length(packet_rate, target_window_sec=0.2):
    sample_length = max(3, int(round(packet_rate * target_window_sec)))
    if sample_length % 2 == 0:
        sample_length += 1
    return sample_length


if __name__ == "__main__":
    # Set TEST_SAMPLES to a small number for a quick sanity check.
    # Set to 0 to run the full dataset (takes several hours).
    TEST_SAMPLES = 1500

    PACKET_RATES = {
        "empty_room": 72,
        "video_game": 153,
    }
    DOPPLER_TARGET_WINDOW_SEC = 1.0
    DOPPLER_SLIDING = 1
    DOPPLER_NOISE_LEVEL = -0.5

    # 1. Process the Empty Room Data
    run_pipeline("data/pcaps/csi-testcse1.pcap", "empty_room", max_samples=TEST_SAMPLES)

    # 2. Process the Video Game Data
    run_pipeline("data/pcaps/csi-testcase2 (1).pcap", "video_game", max_samples=TEST_SAMPLES)

    print("\n--- Computing Doppler Spectrograms with packet-rate-aware windows ---")
    os.makedirs('./doppler_output', exist_ok=True)
    empty_sample_length = choose_sample_length(PACKET_RATES["empty_room"], DOPPLER_TARGET_WINDOW_SEC)
    video_sample_length = choose_sample_length(PACKET_RATES["video_game"], DOPPLER_TARGET_WINDOW_SEC)

    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_doppler_computation.py"),
        "final_output/", "emp/", "./doppler_output/", "0", "0",
        str(empty_sample_length), str(DOPPLER_SLIDING), str(DOPPLER_NOISE_LEVEL),
        "--packet-rate", str(PACKET_RATES["empty_room"]), "--overwrite"
    ], check=True)

    subprocess.run([
        "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSI_doppler_computation.py"),
        "final_output/", "vid/", "./doppler_output/", "0", "0",
        str(video_sample_length), str(DOPPLER_SLIDING), str(DOPPLER_NOISE_LEVEL),
        "--packet-rate", str(PACKET_RATES["video_game"]), "--overwrite"
    ], check=True)

    # 3. Plot the Comparison
    # start_sec=0 is safe for test mode (300 samples ≈ 1.7 seconds)
    # Change to start_sec=5 / start_sec=10 when running full dataset
    print("\n--- Generating Figure 1 Comparison Plot ---")
    plot_fig1_comparison(
        empty_raw_mat="raw_mat_data/empty_room.mat",
        empty_proc_mat="final_output/emp/empty_room_stream_0.mat",
        video_raw_mat="raw_mat_data/video_game.mat",
        video_proc_mat="final_output/vid/video_game_stream_0.mat",
        empty_start_sec=0,   # 0 = start from beginning (safe for 300-sample test)
        video_start_sec=0
    )

    # 4. Plot Doppler Spectrograms side-by-side
    print("\n--- Generating Doppler Spectrogram Plot ---")
    plot_doppler_comparison(
        empty_doppler_path="doppler_output/emp/empty_room_stream_0.txt",
        video_doppler_path="doppler_output/vid/video_game_stream_0.txt",
        empty_packet_rate=PACKET_RATES["empty_room"],
        video_packet_rate=PACKET_RATES["video_game"],
        empty_sample_length=empty_sample_length,
        video_sample_length=video_sample_length,
        sliding=DOPPLER_SLIDING
    )

    # 5. Show the effect of each pipeline stage
    print("\n--- Generating Pipeline Stage Comparison ---")
    plot_pipeline_stages(
        "empty_room",
        packet_rate=PACKET_RATES["empty_room"],
        start_sec=0,
        sample_length=empty_sample_length,
        sliding=DOPPLER_SLIDING,
    )
    plot_pipeline_stages(
        "video_game",
        packet_rate=PACKET_RATES["video_game"],
        start_sec=0,
        sample_length=video_sample_length,
        sliding=DOPPLER_SLIDING,
    )
