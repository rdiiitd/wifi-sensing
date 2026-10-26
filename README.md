# VR collision sensing with Wi-Fi CSI

Detect a person approaching a VR user from Wi-Fi channel state information (CSI) recorded by a
Raspberry Pi 4 (Nexmon CSI, 1 antenna) worn on the headset, with the headset IMU to account for
the wearer's own motion.

Status and plan: see the shared doc *VR Wi-Fi Sensing: Pipeline, SHARP, CornerSense and Next Steps*.

**In short:** another moving person is clearly detected while the wearer stands still or turns
their head (AUC 0.92–0.94); while the wearer walks it is not yet reliable (AUC ~0.63). The main
limit is the single receive antenna.

## Pipeline (`sharp_doppler.py`)

1. **Read CSI** from the pcap: router and laptop packets, 52 live tones (`index_pcap`, `load_csi`).
2. **Sort packets by the router's two transmit antennas** (`track_modes`) → 4 links.
3. **Phase cleaning**, SHARP Lasso path decomposition referenced to a tracked path (`sanitize_sharp`).
4. **Resample** to 200 Hz, gaps > 50 ms flagged (`resample_uniform`).
5. **Doppler spectrum**: 180 ms window, 100-point FFT, every 20 ms, ±5.8 m/s (`doppler_spectrogram`).
6. **Motion score**: share of power faster than 0.7 m/s, averaged over the 4 links (`dynamic_fraction`).

With the IMU (`audit/z01_selfspeed.py`): headset speed from steps + head rotation, and a
**speed gate** that only counts power faster than 2 × headset speed + 0.64 m/s.

Based on SHARP (Meneghello et al., IEEE TMC) and the own-motion idea of CornerSense
(He et al., ACM IMWUT 2025, <https://doi.org/10.1145/3770699>, code <https://github.com/hj5202/CornerSense>).

## Layout

| Path | What |
|---|---|
| `sharp_doppler.py` | The pipeline (steps 1–6) |
| `make_figures.py` | Full-capture spectrograms (cached) and figures 01–08 for m01 / x02 / m_empty |
| `audit/z01_check.py` | z01 overview and antenna-mode separation (figure 09) |
| `audit/z01_imu.py` | IMU alignment, IMU-matched leftover (figure 10) |
| `audit/z01_modes.py` | Antenna-sorting check |
| `audit/z01_selfspeed.py` | Headset speed from the IMU and the speed gate (figure 12) |
| `audit/z01_walking.py` | Walking experiments and the evaluation (T1, T2, fair comparison, block-bootstrap intervals) |
| `audit/z01_walking_sanitize.py` / `_resolution.py` / `_pca.py` | Phase-cleaning, window-length and CornerSense-PCA tests |
| `audit/z01_walking_figure.py` | Summary of the walking experiments (figure 13) |
| `audit/sim_*.py` | Simulations (approach, one vs two antennas) |
| `audit/check_*.py` | Earlier data checks (tone order, frames, AGC, rate) |
| `tests/` | `pytest tests/` |
| `sharp_output/figures/` | Result figures |
| `legacy/` | Earlier pipeline (feature extraction, SVM/kNN, clustering) and the original SHARP repo scripts; not used by the current code |

## Data (not in the repository)

Captures are large and contain video of people, so they stay local. The code expects:

```
data/<name>.pcap              Nexmon CSI pcap (tcpdump on wlan0, port 5500)
data/<name>_meta.json         epoch_first_frame, video_clock_scale, sync flashes, ...
data/<name>_events.csv        time_sec (video seconds), event_type
data/z01/imu/imu.csv, clock.csv, meta.json      Quest 3S raw IMU (800 Hz)
data/z01/z_folder/z01_test_auto_distance.csv    per-frame distance from the video (YOLO)
data/z01/z_folder/z01_test_table.npz            collector's feature table
```

## Run

```
pip install -r requirements.txt
python3 make_figures.py              # figures 01-08 (m01, x02, m_empty)
python3 audit/z01_check.py           # z01 spectrograms (slow first time) + figure 09
python3 audit/z01_imu.py             # figure 10
python3 audit/z01_selfspeed.py       # figure 12
python3 audit/z01_walking.py         # walking experiments
python3 audit/z01_walking_resolution.py && python3 audit/z01_walking_pca.py
python3 audit/z01_walking_figure.py  # figure 13
pytest tests/
```

Results are printed and saved to `sharp_output/cache/*.log`; figures go to `sharp_output/figures/`.
