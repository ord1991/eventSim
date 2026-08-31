# ⚡ Prophesee EVK4 Event Camera Controller & Live Viewer ⚡

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Metavision SDK](https://img.shields.io/badge/Metavision%20SDK-4.2%2B-orange.svg)](https://www.prophesee.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

A high-performance, responsive graphical user interface (GUI) designed for **Prophesee EVK4 (IMX636)** neuromorphic event-based vision sensors. The application is written in Python and leverages the standard **Metavision SDK** (fully compatible with version 4.2+ up to the latest releases).

The interface is dark-themed and completely localized in English. It runs event polling and frame accumulation on an asynchronous background worker thread to guarantee a fluid, high-frame-rate user experience without GUI freezes or stutters.

---

## 🔬 Neuromorphic Sensing: Engineering & Physics Principles

Unlike traditional frame-based cameras that capture static full-frame snapshots at fixed intervals (e.g., 30 FPS or 60 FPS), **event-based vision sensors** (such as the Sony IMX636 / Prophesee EVK4) operate on bio-inspired neuromorphic principles:

1. **Asynchronous Delta Modulated Photoreceptors**:
   Each independent pixel on the sensor contains an autonomous logarithmic photoreceptor circuit. Rather than transmitting intensity values on a clock, a pixel fires an **event** only when the local change in logarithmic illuminance ($\Delta \ln I$) exceeds a predefined physical threshold:
   $$\Delta \ln I = \ln I(t) - \ln I(t - \Delta t) \ge \pm C$$
   - **ON Event ($+C$)**: Fired when relative light intensity increases.
   - **OFF Event ($-C$)**: Fired when relative light intensity decreases.

2. **Microsecond Temporal Resolution & High Dynamic Range**:
   Events are timestamped with microsecond precision ($\mu s$), allowing tracking of ultra-fast movements (up to >10,000 events/sec per pixel) without motion blur, while achieving high dynamic range ($>120\text{ dB}$).

3. **Data Efficiency & Zero Redundancy**:
   When scene illuminance remains constant, zero events are transmitted—minimizing data rate, latency, and power consumption.

---

## ✨ Key Features

### 1. 🎥 Real-Time Live Event Visualization & Modes
- Visualizes raw asynchronous event streams as accumulated 2D video frames.
- **Visualization Modes**: Toggle between **Accumulation** and **Time-Surface Decay** (exponential event temporal trails).
- **Custom Color Palettes**: Choose between **Monochrome** (White/Gray), **Red/Blue**, **Green/Red**, and **Heatmap** modes.
- **Region of Interest (ROI) Selection**: Drag a bounding box on the stream to isolate and count events only within a targeted region.
- **Snapshot Capture**: Capture high-resolution viewer frames directly to `.png` with a single click.

### 2. 📊 Dynamic Analytics Grid & Graph Visibility Controls
- **Toggle Dropdown Menu (`Graphs Select 📊`)**: Interactively enable or disable individual analytics charts with real-time automatic grid reconfiguration:
  1. **Event Rate Timeline**: Rolling 10-second history of camera event throughput (`kEvt/sec`).
  2. **ON / OFF Event Ratio**: Tracks polarity sensitivity balance ($+C$ vs $-C$).
  3. **2D Spatial Activity Profile**: X-axis event spatial density distribution across sensor width.
  4. **Inter-Event Interval (ISI)**: Temporal delta-t distribution for thermal noise vs signal discrimination.

### 3. 🎛️ Full Hardware Bias Tuning & Auto-Calibration Wizard
Directly configure low-level sensor analog bias parameters on physical hardware:
- `bias_diff`, `bias_diff_on`, `bias_diff_off`, `bias_fo`, `bias_hpf`, `bias_refr`.
- **Auto-Calibrate Biases Wizard 🪄**: One-click automated calibration routine that samples background noise and auto-tunes contrast thresholds.
- **Event Rate Controller (ERC)** & **Event Trail Filter**: Bandwidth constraint and microsecond noise filtering modules.

### 4. 💾 RAW Recording, Replay Player & MP4 Export
- **RAW Recording**: High-speed logging of binary `.raw` streams directly to disk.
- **RAW File Replay Player 🎬**: Load and play back `.raw` files with Play/Pause and variable playback speeds (0.25x, 0.5x, 1.0x, 2.0x).
- **RAW to MP4 Export 🎥**: Convert recorded `.raw` event streams into playable `.mp4` video files.

---

## 💻 Operating System & Virtualenv Setup

The application is fully optimized for **Windows** and **Linux** platforms.

Because Metavision is a native C++/Python platform SDK installed globally on host systems (e.g. under `C:\Program Files\Prophesee` on Windows or `/usr/include/metavision` on Linux), running inside isolated virtual environments (`venv`) may require system site-packages access.

To ensure your virtual environment accesses global Metavision bindings:
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## 📦 Requirements

Install prerequisites in your active environment:

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## 🚀 Running the Application

To launch the event viewer and controller application:

```bash
python3 event_recorder_app.py
```

Click **"Connect Camera 🔌"** in the top header bar to automatically detect and initialize your USB Prophesee EVK4 sensor.

---

## 🧪 Running Unit Tests

To run automated verification tests:

```bash
python3 -m unittest test_app.py
```
