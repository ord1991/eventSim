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

### 1. 🎥 Real-Time Live Event Visualization
- Visualizes raw asynchronous event streams as accumulated 2D video frames (**ON events** as white pixels, **OFF events** as gray pixels).
- Employs zero-allocation vectorized NumPy masking and uncompressed raw PPM binary rendering to update Tkinter image widgets without CPU-intensive PNG encoding overhead.

### 2. 📊 High-Performance Event Rate Timeline Plot
- Embeds a real-time `Matplotlib` dashboard tracking camera event throughput in thousands of events per second (`kEvt/sec`).
- **Memory & Lag Optimization**: To eliminate memory leaks, GC pauses, or progressive GUI slowdown:
  1. Utilizes `collections.deque` for $O(1)$ constant-time history trimming, bounding the sliding timeline window to **10 seconds**.
  2. Uses direct line dataset updates (`line.set_data()`) and idle redraw calls (`canvas.draw_idle()`) rate-limited to 500ms intervals instead of costly `ax.clear()` axis wipes.
- **Dynamic Accumulation Slider**: Adjusts accumulation time-windows on the fly (1ms to 100ms), modifying rate integration bases and live frame compilation dynamically.

### 3. 🎛️ Full Hardware Bias & Parameter Tuning Sidebar
Directly configure low-level sensor analog bias parameters on physical hardware via Metavision SDK HAL bindings:
- `bias_diff`: Photoreceptor output reference level.
- `bias_diff_on`: Contrast sensitivity threshold for ON events ($+C$).
- `bias_diff_off`: Contrast sensitivity threshold for OFF events ($-C$).
- `bias_fo`: Photoreceptor low-pass filter cutoff frequency.
- `bias_hpf`: Differential amplifier high-pass filter cutoff frequency.
- `bias_refr`: Pixel refractory dead-time period delay.
- **Event Rate Controller (ERC)**: Hardware-level maximum event rate constraint module to throttle bandwidth spikes.
- **Event Trail Filter**: Microsecond temporal noise filter targeting background thermal fluctuations.

### 4. 💾 RAW Event Stream Recording
- High-speed logging of uncompressed binary `.raw` event streams directly to disk.
- Live real-time metric panel displaying duration, output file size (MB), accumulated event totals, and instantaneous event rate.

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
