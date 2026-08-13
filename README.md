# Prophesee EVK4 Event Camera Controller & Live Viewer

A high-performance, responsive graphical user interface (GUI) designed for **Prophesee EVK4 (IMX636)** event-based cameras. The application is written in Python and leverages the standard **Metavision SDK** (fully compatible with version 4.2+ up to the latest releases).

The interface is completely translated to English and designed with a modern, dark-themed styling. It runs event polling and frame accumulation on an asynchronous background thread to guarantee a highly responsive, fluid user experience without any GUI freezes or stutters.

---

## Key Features

### 1. Real-Time Live Event View
- Visualizes the raw event stream dynamically as a standard 2D video (accumulating ON events as white pixels and OFF events as gray pixels).
- Uses efficient OpenCV canvas mapping and conversion to display native events with microsecond precision.

### 2. High-Performance Event Rate Timeline Plot
- Embeds a real-time `Matplotlib` plot inside the dashboard showing the camera's current event-rate (`kEvt/sec`).
- **Memory & Lag Optimization**: To prevent memory leaks, high CPU usage, or gradual freezing (lag) commonly associated with continuous plotting, the chart:
  1. Limits rolling timeline history to exactly the last **10 seconds** of execution.
  2. Employs direct line data updates (`line.set_data()`) and idle redraws (`canvas.draw_idle()`) instead of expensive axis clearing (`ax.clear()`), making graph rendering practically free for the CPU.
- **Dynamic Accumulation Slider**: Placed directly underneath the chart, this slider lets you modify the accumulation time-window in milliseconds (from 1ms to 100ms) on the fly, immediately updating both the live video rendering and the rate integration basis.

### 3. Full Hardware Bias & Parameter Tuning Sidebar
- Adjusts sensor behaviors directly on the physical hardware via the Metavision SDK HAL facility bindings:
  - `bias_diff`: Photoreceptor output reference level.
  - `bias_diff_on`: Contrast sensitivity threshold for ON events.
  - `bias_diff_off`: Contrast sensitivity threshold for OFF events.
  - `bias_fo`: Low-pass filter cutoff frequency.
  - `bias_hpf`: High-pass filter cutoff frequency.
  - `bias_refr`: Refractory period (pixel dead-time) delay.
- **Event Rate Controller (ERC)**: Checkbox and text entry to dynamically enable and configure maximum events-per-second constraints to throttle bandwidth.
- **Event Trail Filter**: Checkbox and text entry to enable digital event trail noise filtering and configure its microsecond delay threshold.

---

## Operating System & Virtualenv Setup 💡

The application is fully tailored and optimized for **Windows** platforms.

Because Metavision is a native platform SDK (installed globally on your machine under `C:\Program Files\Prophesee`) rather than a pip-installable public library, running it inside isolated Python virtual environments (`venv`) like those created by PyCharm can raise `ImportError` or native DLL load failures.

To solve this, ensure your virtual environment has access to your system's global site-packages:
1. In **PyCharm**, navigate to: `File -> Settings -> Project -> Python Interpreter`.
2. Click the gear icon next to your Interpreter dropdown and select `Show All...`.
3. Edit your active Interpreter configuration and make sure the option **"Inherit global site-packages"** is checked.
4. Alternatively, recreate your environment with:
   ```bash
   python -m venv .venv --system-site-packages
   ```
This grants your project's virtual interpreter full native access to the globally registered Metavision python bindings and its binary C++ dependencies seamlessly!

---

## Requirements

Ensure you have installed the following Python prerequisites inside your interpreter environment:

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## Running the Application

To start the controller and viewer, run:

```bash
python3 event_recorder_app.py
```

Click **"Connect to Physical Camera (USB) 🔌"** in the top header to dynamically scan and connect to your USB-connected Prophesee EVK4 camera.

---

## Running Unit Tests

The repository includes a verification test suite:

```bash
python3 -m unittest test_app.py
```
