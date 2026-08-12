#!/usr/bin/env python3
"""
Prophesee EVK4 (IMX636) Event Camera Control & Recording Application.
Smoothly switches between Prophesee Metavision SDK and a highly visual moving-shape Simulator Mode.
Runs the live camera event polling loop in a separate worker thread to keep the Tkinter GUI completely responsive.
"""

import os
import sys
import json
import time
import math
import random
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import cv2

# Import Matplotlib for Event Rate plotting
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Dynamic Metavision environment resolution for Windows Platforms.
# In virtual environments, system site-packages containing the Metavision installation may be isolated.
# We auto-discover and append potential Metavision SDK installation paths on Windows,
# and link the native C++ DLL folders.
def bootstrap_metavision_paths():
    import sys
    from pathlib import Path

    potential_paths = []

    # Resolve the base Python system site-packages if running inside a virtual environment on Windows
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We are inside a virtualenv (venv)!
        # Let's dynamically add the base/system interpreter's site-packages
        base_prefix = getattr(sys, 'base_prefix', sys.prefix)
        potential_paths.append(os.path.join(base_prefix, "Lib", "site-packages"))

    # Add standard Windows installation paths for Prophesee SDK and OpenEB
    for path_str in [
        "C:\\Program Files\\Prophesee\\lib\\site-packages",
        "C:\\Program Files\\Prophesee\\python",
        "C:\\Program Files\\OpenEB\\lib\\site-packages",
        "C:\\tmp\\prophesee\\py3venv\\Lib\\site-packages",
    ]:
        potential_paths.append(path_str)

    # Dynamically find python version-specific folders under global Programs
    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        programs_path = Path(user_profile) / "AppData" / "Local" / "Programs" / "Python"
        if programs_path.exists():
            for py_dir in programs_path.iterdir():
                potential_paths.append(str(py_dir / "Lib" / "site-packages"))

    # CRITICAL: Windows Python 3.8+ requires explicitly loading directories containing external DLLs.
    # We must load the Prophesee/bin directory to resolve dependencies of .pyd modules.
    dll_dirs = [
        "C:\\Program Files\\Prophesee\\bin",
        "C:\\Program Files\\OpenEB\\bin",
        "C:\\Program Files\\Prophesee\\lib\\metavision\\hal\\plugins",
        "C:\\Program Files\\OpenEB\\lib\\metavision\\hal\\plugins"
    ]
    for dll_dir in dll_dirs:
        if os.path.exists(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception as e:
                print(f"Failed adding DLL directory {dll_dir}: {e}")

    # Append discovered paths to sys.path
    for path in potential_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)

# Bootstrap before importing Metavision
bootstrap_metavision_paths()

# Global flag to signal SDK availability
METAVISION_AVAILABLE = False
try:
    from metavision_sdk_stream import Camera, CameraStreamSlicer
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    from metavision_hal import DeviceDiscovery, DeviceConfig
    METAVISION_AVAILABLE = True
except ImportError:
    # Double check standard fallback path imports
    pass

# Default EVK4 / IMX636 Biases (relative offsets around default value 0)
# Biases: Name, Default, Min, Max, Description
EVK4_BIAS_DEFAULTS = {
    "bias_diff": {"value": 0, "min": -25, "max": 23, "desc": "Photoreceptor output reference level"},
    "bias_diff_on": {"value": 0, "min": -85, "max": 140, "desc": "Contrast sensitivity threshold for ON events"},
    "bias_diff_off": {"value": 0, "min": -35, "max": 190, "desc": "Contrast sensitivity threshold for OFF events"},
    "bias_fo": {"value": 0, "min": -35, "max": 55, "desc": "Low-pass filter cutoff frequency (photoreceptor)"},
    "bias_hpf": {"value": 0, "min": 0, "max": 120, "desc": "High-pass filter cutoff frequency (diff amp)"},
    "bias_refr": {"value": 0, "min": -20, "max": 235, "desc": "Refractory period (dead time) delay"}
}


class SimulatedCamera:
    """
    Simulates an event camera producing moving geometric shapes (e.g. circle, square)
    and response to IMX636 biases (sensitivity, bandpass filters, refractory dead-time).
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.biases = {k: v["value"] for k, v in EVK4_BIAS_DEFAULTS.items()}
        self.erc_enabled = False
        self.erc_rate = 1000000 # 1 Mevt/s
        self.trail_filter_enabled = False
        self.trail_filter_threshold = 10000 # us

        self.is_running = False
        self.is_recording = False
        self.recording_file = None
        self.record_start_time = 0
        self.recorded_bytes = 0
        self.recorded_events = 0

        # State of the simulated object (a moving circle)
        self.obj_x = width // 2
        self.obj_y = height // 2
        self.obj_vx = 3.5
        self.obj_vy = 2.5
        self.obj_r = 45
        self.last_update_t = time.time()
        self.simulated_us = 0 # Monotonic simulated microseconds

    def start(self):
        self.is_running = True
        self.last_update_t = time.time()

    def stop(self):
        self.is_running = False
        self.stop_recording()

    def start_recording(self, file_path):
        self.recording_file = file_path
        self.is_recording = True
        self.record_start_time = time.time()
        self.recorded_bytes = 128 # Mock header
        self.recorded_events = 0

    def stop_recording(self):
        self.is_recording = False
        self.recording_file = None

    def update_biases(self, name, value):
        if name in self.biases:
            self.biases[name] = value

    def generate_events(self, dt):
        """
        Generates simulated events representing a moving circle.
        Higher speed/illumination = more events.
        Higher bias_diff_on/off = lower sensitivity = fewer events.
        bias_fo/hpf act as low/high frequency limits.
        bias_refr limits max event rate per pixel (dead time).
        """
        self.simulated_us += int(dt * 1e6)

        # Update shape position
        self.obj_x += self.obj_vx
        self.obj_y += self.obj_vy

        # Bounce off walls
        if self.obj_x - self.obj_r < 0 or self.obj_x + self.obj_r > self.width:
            self.obj_vx = -self.obj_vx
            self.obj_x = max(self.obj_r, min(self.width - self.obj_r, self.obj_x))
        if self.obj_y - self.obj_r < 0 or self.obj_y + self.obj_r > self.height:
            self.obj_vy = -self.obj_vy
            self.obj_y = max(self.obj_r, min(self.height - self.obj_r, self.obj_y))

        # Determine number of events based on biases
        speed = math.sqrt(self.obj_vx**2 + self.obj_vy**2)
        base_num = int(speed * 250)

        on_sensitivity = 1.0 / (1.0 + max(-0.9, (self.biases["bias_diff_on"] / 100.0)))
        off_sensitivity = 1.0 / (1.0 + max(-0.9, (self.biases["bias_diff_off"] / 100.0)))

        fo_factor = 1.0 + (self.biases["bias_fo"] / 55.0) * 0.5
        hpf_factor = 1.0 - (self.biases["bias_hpf"] / 120.0) * 0.4

        num_on = int(base_num * 0.5 * on_sensitivity * fo_factor * hpf_factor)
        num_off = int(base_num * 0.5 * off_sensitivity * fo_factor * hpf_factor)

        # Apply ERC if enabled
        if self.erc_enabled:
            limit = int(self.erc_rate * dt)
            if num_on + num_off > limit:
                factor = limit / (num_on + num_off)
                num_on = int(num_on * factor)
                num_off = int(num_off * factor)

        # Simulated events array: structured array with x, y, p, t fields
        events_list = []

        current_t_us = self.simulated_us

        # Helper to generate perimeter events
        def add_perimeter_events(num_ev, polarity):
            for _ in range(num_ev):
                angle = random.uniform(0, 2 * math.pi)
                r = self.obj_r + random.uniform(-2, 2)
                x = int(self.obj_x + r * math.cos(angle))
                y = int(self.obj_y + r * math.sin(angle))

                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    t_offset = random.randint(0, int(dt * 1e6))
                    events_list.append((x, y, polarity, current_t_us - t_offset))

        add_perimeter_events(num_on, 1)
        add_perimeter_events(num_off, 0)

        # Add background noise depending on sensitivity (lower threshold = more noise)
        noise_on = int(25 * (1.0 / (1.1 + self.biases["bias_diff_on"]/100.0)))
        noise_off = int(25 * (1.0 / (1.1 + self.biases["bias_diff_off"]/100.0)))

        noise_on = max(0, min(500, noise_on))
        noise_off = max(0, min(500, noise_off))

        for _ in range(noise_on):
            events_list.append((random.randint(0, self.width-1), random.randint(0, self.height-1), 1, current_t_us - random.randint(0, int(dt * 1e6))))
        for _ in range(noise_off):
            events_list.append((random.randint(0, self.width-1), random.randint(0, self.height-1), 0, current_t_us - random.randint(0, int(dt * 1e6))))

        # Sort by timestamp
        events_list.sort(key=lambda ev: ev[3])

        # Construct structured numpy array
        events = np.array(events_list, dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'i2'), ('t', '<i8')])

        if self.trail_filter_enabled and len(events) > 0:
            filtered_events = []
            for ev in events:
                dx = ev['x'] - self.obj_x
                dy = ev['y'] - self.obj_y
                dist = math.sqrt(dx**2 + dy**2)
                if abs(dist - self.obj_r) < 5:
                    filtered_events.append(ev)
                elif random.random() > 0.6:
                    filtered_events.append(ev)
            events = np.array(filtered_events, dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'i2'), ('t', '<i8')])

        # Update recording stats
        if self.is_recording and len(events) > 0:
            self.recorded_events += len(events)
            self.recorded_bytes += len(events) * 8

        return events


class EventRecorderApp(tk.Tk):
    """
    Tkinter interface with modern dark styling.
    Features:
    - Side-by-side split screen structure
    - Left side: Live visual view + Accumulation controls
    - Middle side: Event rate over time (Matplotlib Plot)
    - Right side: Full parameter settings, Save/Load JSON, Recording Panel, Stats.
    """
    def __init__(self):
        super().__init__()
        self.title("Prophesee EVK4 Controller & Recorder (Hebrew GUI)")
        self.geometry("1280x768")
        self.minsize(1024, 700)

        # Thread safety lock
        self.lock = threading.Lock()
        self.shared_events_buffer = None
        self.total_live_recorded_events = 0
        self.live_recording_file_path = None

        # Enable Dark Theme Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_dark_theme()

        # State variables
        self.running_live = False
        self.camera_instance = None
        self.slicer_instance = None
        self.mock_camera = None
        self.camera_thread = None
        self.camera_width = 640
        self.camera_height = 480

        # GUI controlled camera configs
        self.accumulation_time_ms = tk.DoubleVar(value=20.0) # default 20ms accumulation
        self.erc_enabled = tk.BooleanVar(value=False)
        self.erc_rate = tk.IntVar(value=1000000) # events/sec
        self.trail_filter_enabled = tk.BooleanVar(value=False)
        self.trail_filter_threshold_us = tk.IntVar(value=10000)

        # Recording status
        self.recording_active = False
        self.recording_path = tk.StringVar(value=str(Path.home() / "recording.raw"))
        self.recording_duration = 0.0
        self.recording_file_size_mb = 0.0

        # Stats & Plotting Data
        self.fps_live = 0
        self.event_rate_live = 0.0 # Kept in events/sec
        self.time_history = []
        self.rate_history = []
        self.start_app_time = time.time()

        # Load and verify settings path
        self.settings_json_path = tk.StringVar(value="camera_settings.json")

        # Build layout
        self.create_layout()

        # Check SDK and start loop
        self.detect_camera_or_initialize()
        self.update_loop()

    def configure_dark_theme(self):
        """Custom colors to create a modern dark design."""
        bg_dark = "#1c1c1e"
        bg_panel = "#2c2c2e"
        accent_blue = "#0a84ff"
        accent_green = "#30d158"
        text_white = "#ffffff"
        text_gray = "#aeaeae"

        self.configure(bg=bg_dark)

        self.style.configure(".", background=bg_dark, foreground=text_white, fieldbackground=bg_panel)
        self.style.configure("TFrame", background=bg_dark)
        self.style.configure("Panel.TFrame", background=bg_panel, borderwidth=1, relief="flat")
        self.style.configure("TLabel", background=bg_dark, foreground=text_white, font=("Calibri", 11))
        self.style.configure("PanelTitle.TLabel", background=bg_panel, foreground=accent_blue, font=("Calibri", 13, "bold"))
        self.style.configure("PanelSec.TLabel", background=bg_panel, foreground=text_white, font=("Calibri", 11))
        self.style.configure("StatsValue.TLabel", background=bg_panel, foreground=accent_green, font=("Calibri", 12, "bold"))
        self.style.configure("StatsLabel.TLabel", background=bg_panel, foreground=text_gray, font=("Calibri", 10))

        # Buttons styling
        self.style.configure("TButton", background="#3a3a3c", foreground=text_white, borderwidth=0, font=("Calibri", 11, "bold"))
        self.style.map("TButton", background=[("active", "#48484a"), ("pressed", "#2c2c2e")])
        self.style.configure("Action.TButton", background=accent_blue, foreground=text_white)
        self.style.map("Action.TButton", background=[("active", "#359aff"), ("pressed", "#0066cc")])
        self.style.configure("RecordOn.TButton", background="#ff453a", foreground=text_white)
        self.style.map("RecordOn.TButton", background=[("active", "#ff6961"), ("pressed", "#b30000")])

        # Checkbox & sliders
        self.style.configure("TCheckbutton", background=bg_panel, foreground=text_white, font=("Calibri", 11))
        self.style.configure("Horizontal.TScale", background=bg_panel)

    def create_layout(self):
        """Builds a beautiful side-by-side dashboard interface."""
        # Top Header Bar
        header_frame = tk.Frame(self, bg="#2c2c2e", height=50)
        header_frame.pack(side="top", fill="x", padx=0, pady=0)

        header_label = tk.Label(header_frame, text="Prophesee EVK4 (IMX636) - מערכת שליטה והקלטה", bg="#2c2c2e", fg="#0a84ff", font=("Calibri", 16, "bold"))
        header_label.pack(side="right", padx=15, pady=10)

        self.connect_btn = ttk.Button(header_frame, text="התחבר למצלמה פיזית (USB) 🔌", command=self.connect_to_physical_camera)
        self.connect_btn.pack(side="left", padx=15, pady=10)

        self.mode_status_label = tk.Label(header_frame, text="מזהה חומרה...", bg="#2c2c2e", fg="#30d158", font=("Calibri", 12, "bold"))
        self.mode_status_label.pack(side="left", padx=15, pady=12)

        # Main Work Area
        main_container = ttk.Frame(self)
        main_container.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # 3 Side-by-side Column Layout
        # Column 1: Live Video frame (Left)
        col1 = ttk.Frame(main_container, style="Panel.TFrame")
        col1.pack(side="left", fill="both", expand=True, padx=5)
        self.build_live_view_panel(col1)

        # Column 2: Graph analysis (Middle)
        col2 = ttk.Frame(main_container, style="Panel.TFrame")
        col2.pack(side="left", fill="both", expand=True, padx=5)
        self.build_graph_panel(col2)

        # Column 3: Biases Control, advanced filters & Recording (Right)
        col3 = ttk.Frame(main_container, style="Panel.TFrame", width=380)
        col3.pack(side="left", fill="both", expand=False, padx=5)
        self.build_control_panel(col3)

    def build_live_view_panel(self, parent):
        """Left panel with the camera vision view."""
        # Title
        title_lbl = ttk.Label(parent, text="תצוגת וידאו בזמן אמת (Live View)", style="PanelTitle.TLabel")
        title_lbl.pack(anchor="ne", padx=15, pady=10)

        # Canvas/Label for Image rendering
        self.image_label = tk.Label(parent, bg="#000000")
        self.image_label.pack(fill="both", expand=True, padx=15, pady=5)

        # Accumulation Time Slider Area
        acc_control_frame = ttk.Frame(parent, style="Panel.TFrame")
        acc_control_frame.pack(fill="x", side="bottom", padx=15, pady=10)

        slider_label = ttk.Label(acc_control_frame, text="זמן אקומולציה (מילישניות):", style="PanelSec.TLabel")
        slider_label.pack(side="right", padx=5)

        self.acc_slider_val_lbl = ttk.Label(acc_control_frame, text="20.0 ms", style="StatsValue.TLabel")
        self.acc_slider_val_lbl.pack(side="left", padx=5)

        acc_slider = ttk.Scale(
            acc_control_frame,
            from_=1.0,
            to=100.0,
            variable=self.accumulation_time_ms,
            orient="horizontal",
            command=self.on_accumulation_slider_moved
        )
        acc_slider.pack(fill="x", expand=True, side="right", padx=10)

    def on_accumulation_slider_moved(self, val):
        self.acc_slider_val_lbl.config(text=f"{float(val):.1f} ms")

    def build_graph_panel(self, parent):
        """Middle panel containing real-time Matplotlib chart."""
        title_lbl = ttk.Label(parent, text="קצב אירועים על ציר הזמן", style="PanelTitle.TLabel")
        title_lbl.pack(anchor="ne", padx=15, pady=10)

        # Setup Matplotlib Figure
        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor="#2c2c2e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1c1c1e")
        self.fig.subplots_adjust(bottom=0.15, left=0.15)
        self.ax.spines['bottom'].set_color('#aeaeae')
        self.ax.spines['top'].set_color('#aeaeae')
        self.ax.spines['right'].set_color('#aeaeae')
        self.ax.spines['left'].set_color('#aeaeae')
        self.ax.tick_params(axis='x', colors='#aeaeae')
        self.ax.tick_params(axis='y', colors='#aeaeae')
        self.ax.set_xlabel("זמן ריצה (שניות)", color='#aeaeae', fontname="Calibri", fontsize=10)
        self.ax.set_ylabel("קצב אירועים (kEvt/sec)", color='#aeaeae', fontname="Calibri", fontsize=10)

        self.line, = self.ax.plot([], [], color="#0a84ff", linewidth=2)

        # Embed chart into Tkinter widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=10)

    def build_control_panel(self, parent):
        """Right panel with full biases sliders, advanced configurations, and recording control."""
        # Main Scrollable canvas container
        canvas = tk.Canvas(parent, bg="#2c2c2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, style="Panel.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=360)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. Parameter Tuning Section (Biases)
        bias_section = ttk.LabelFrame(scroll_frame, text="פרמטרי חומרה (Biases - EVK4)", style="Panel.TFrame")
        bias_section.pack(fill="x", padx=10, pady=5)

        self.bias_vars = {}
        self.bias_val_labels = {}

        for name, info in EVK4_BIAS_DEFAULTS.items():
            b_frame = ttk.Frame(bias_section, style="Panel.TFrame")
            b_frame.pack(fill="x", padx=5, pady=4)

            # Row 1: Label and value
            lbl = ttk.Label(b_frame, text=f"{name}:", style="PanelSec.TLabel")
            lbl.pack(side="right")

            val_lbl = ttk.Label(b_frame, text="0", style="StatsValue.TLabel")
            val_lbl.pack(side="left")
            self.bias_val_labels[name] = val_lbl

            # Row 2: Slider
            var = tk.IntVar(value=info["value"])
            self.bias_vars[name] = var

            slider = ttk.Scale(
                b_frame,
                from_=info["min"],
                to=info["max"],
                variable=var,
                orient="horizontal",
                command=lambda val, n=name: self.on_bias_slider_moved(n, val)
            )
            slider.pack(fill="x", expand=True, side="bottom", padx=5, pady=2)

            # Tooltip/description
            desc_lbl = ttk.Label(b_frame, text=info["desc"], style="StatsLabel.TLabel")
            desc_lbl.pack(side="bottom", anchor="e", padx=5)

        # 2. Advanced Control Filters (ERC & Trail Filter)
        advanced_section = ttk.LabelFrame(scroll_frame, text="מסננים ובקרת קצב (ERC / Trail Filter)", style="Panel.TFrame")
        advanced_section.pack(fill="x", padx=10, pady=5)

        # ERC Enable
        erc_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        erc_f.pack(fill="x", padx=5, pady=4)
        erc_chk = ttk.Checkbutton(erc_f, text="הפעל בקרת קצב אירועים (ERC)", variable=self.erc_enabled, command=self.apply_erc_settings)
        erc_chk.pack(side="right")

        # ERC Rate
        erc_rate_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        erc_rate_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(erc_rate_f, text="מגבלת קצב (Evt/sec):", style="PanelSec.TLabel").pack(side="right")
        erc_rate_entry = ttk.Entry(erc_rate_f, textvariable=self.erc_rate, width=12)
        erc_rate_entry.pack(side="left")
        erc_rate_entry.bind("<Return>", lambda e: self.apply_erc_settings())

        # Trail Filter Enable
        trail_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        trail_f.pack(fill="x", padx=5, pady=4)
        trail_chk = ttk.Checkbutton(trail_f, text="הפעל מסנן רעש (Event Trail Filter)", variable=self.trail_filter_enabled, command=self.apply_trail_settings)
        trail_chk.pack(side="right")

        # Trail Filter Threshold
        trail_thresh_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        trail_thresh_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(trail_thresh_f, text="סף השהייה (מיקרושניות):", style="PanelSec.TLabel").pack(side="right")
        trail_thresh_entry = ttk.Entry(trail_thresh_f, textvariable=self.trail_filter_threshold_us, width=12)
        trail_thresh_entry.pack(side="left")
        trail_thresh_entry.bind("<Return>", lambda e: self.apply_trail_settings())

        # 3. Settings Config Save/Load Section (JSON)
        config_section = ttk.LabelFrame(scroll_frame, text="טעינה ושמירה של הגדרות JSON", style="Panel.TFrame")
        config_section.pack(fill="x", padx=10, pady=5)

        conf_btn_f = ttk.Frame(config_section, style="Panel.TFrame")
        conf_btn_f.pack(fill="x", padx=5, pady=5)

        load_btn = ttk.Button(conf_btn_f, text="טען הגדרות", command=self.load_camera_settings_json)
        load_btn.pack(side="right", fill="x", expand=True, padx=2)

        save_btn = ttk.Button(conf_btn_f, text="שמור הגדרות", command=self.save_camera_settings_json)
        save_btn.pack(side="left", fill="x", expand=True, padx=2)

        # 3b. Manual Metavision SDK/DLL Directory Specification
        sdk_section = ttk.LabelFrame(scroll_frame, text="ניתוב ידני ל-Metavision SDK / DLL", style="Panel.TFrame")
        sdk_section.pack(fill="x", padx=10, pady=5)

        self.manual_sdk_path = tk.StringVar(value="C:\\Program Files\\Prophesee")

        sdk_path_f = ttk.Frame(sdk_section, style="Panel.TFrame")
        sdk_path_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(sdk_path_f, text="תיקיית התקנה:", style="PanelSec.TLabel").pack(side="right")
        sdk_browse_btn = ttk.Button(sdk_path_f, text="בחר...", width=8, command=self.browse_manual_sdk_path)
        sdk_browse_btn.pack(side="left")

        self.sdk_path_entry = ttk.Entry(sdk_section, textvariable=self.manual_sdk_path)
        self.sdk_path_entry.pack(fill="x", padx=5, pady=2)

        self.apply_sdk_btn = ttk.Button(sdk_section, text="טען והפעל SDK ידנית ⚙️", style="Action.TButton", command=self.apply_custom_sdk_path)
        self.apply_sdk_btn.pack(fill="x", padx=5, pady=5)

        # 4. Recording & Storage Panel
        rec_section = ttk.LabelFrame(scroll_frame, text="הקלטת וידאו / אירועים (RAW)", style="Panel.TFrame")
        rec_section.pack(fill="x", padx=10, pady=5)

        # Select path
        path_f = ttk.Frame(rec_section, style="Panel.TFrame")
        path_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(path_f, text="קובץ שמירה:", style="PanelSec.TLabel").pack(side="right")
        path_btn = ttk.Button(path_f, text="בחר...", width=8, command=self.choose_recording_file)
        path_btn.pack(side="left")

        self.path_entry = ttk.Entry(rec_section, textvariable=self.recording_path)
        self.path_entry.pack(fill="x", padx=5, pady=2)

        # Start/Stop Button
        self.record_btn = ttk.Button(rec_section, text="התחל הקלטה 🔴", style="TButton", command=self.toggle_recording)
        self.record_btn.pack(fill="x", padx=5, pady=10)

        # Statistics area
        stats_frame = ttk.Frame(rec_section, style="Panel.TFrame")
        stats_frame.pack(fill="x", padx=5, pady=5)

        # Stat grid
        self.stat_duration = self.create_stat_widget(stats_frame, "משך זמן ריצה:", "0.0 sec", 0, 0)
        self.stat_file_size = self.create_stat_widget(stats_frame, "גודל קובץ הקלטה:", "0.0 MB", 0, 1)
        self.stat_tot_events = self.create_stat_widget(stats_frame, "סה\"כ אירועים:", "0", 1, 0)
        self.stat_event_rate = self.create_stat_widget(stats_frame, "קצב נוכחי (Evt/s):", "0", 1, 1)

    def create_stat_widget(self, parent, label, val_text, row, col):
        f = ttk.Frame(parent, style="Panel.TFrame")
        f.grid(row=row, column=col, sticky="nsew", padx=10, pady=5)

        lbl = ttk.Label(f, text=label, style="StatsLabel.TLabel")
        lbl.pack(anchor="e")

        val = ttk.Label(f, text=val_text, style="StatsValue.TLabel")
        val.pack(anchor="e")
        return val

    def on_bias_slider_moved(self, name, val):
        val_int = int(float(val))
        self.bias_val_labels[name].config(text=str(val_int))

        # Update device biases if initialized
        if self.camera_instance and METAVISION_AVAILABLE:
            try:
                device = self.camera_instance.get_i_ll_biases()
                if device:
                    device.set(name, val_int)
            except Exception as e:
                print(f"Error setting bias {name}: {e}")
        elif self.mock_camera:
            self.mock_camera.update_biases(name, val_int)

    def browse_manual_sdk_path(self):
        """Opens a folder selection dialog for specified manual SDK/DLL installation."""
        folder = filedialog.askdirectory(title="בחר תיקיית התקנה של Metavision SDK")
        if folder:
            self.manual_sdk_path.set(folder)

    def apply_custom_sdk_path(self):
        """Dynamically appends custom user specified path to sys.path and DLL directory search path on Windows."""
        global METAVISION_AVAILABLE
        global Camera, CameraStreamSlicer, BaseFrameGenerationAlgorithm, DeviceDiscovery, DeviceConfig

        path = self.manual_sdk_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("שגיאה", "הנתיב שנבחר אינו קיים במחשב.")
            return

        # Build possible subpaths based on standard installation layout
        # C:\Program Files\Prophesee contains \bin for DLLs and \lib\site-packages or \python for bindings
        site_packages = os.path.join(path, "lib", "site-packages")
        python_pkg = os.path.join(path, "python")
        bin_dir = os.path.join(path, "bin")
        hal_plugins = os.path.join(path, "lib", "metavision", "hal", "plugins")

        import sys
        # Register site-packages/python folder to PYTHONPATH
        added_sys = False
        for folder in [site_packages, python_pkg, path]:
            if os.path.exists(folder) and folder not in sys.path:
                sys.path.append(folder)
                added_sys = True

        # Register DLL / bin directory
        added_dll = False
        for folder in [bin_dir, hal_plugins]:
            if os.path.exists(folder):
                try:
                    os.add_dll_directory(folder)
                    added_dll = True
                except Exception as e:
                    print(f"Failed loading DLL folder {folder}: {e}")

        # Attempt to import Metavision SDK dynamically
        try:
            from metavision_sdk_stream import Camera, CameraStreamSlicer
            from metavision_sdk_core import BaseFrameGenerationAlgorithm
            from metavision_hal import DeviceDiscovery, DeviceConfig
            METAVISION_AVAILABLE = True

            messagebox.showinfo(
                "הצלחה",
                "ה-SDK של Metavision ויחידות ה-DLL נטענו בהצלחה במערכת!\n"
                "כעת תוכל ללחוץ על 'התחבר למצלמה פיזית (USB)' כדי להפעיל את המצלמה."
            )
        except ImportError as e:
            messagebox.showerror(
                "שגיאה בטעינה",
                f"נמצאה התיקייה אך ייבוא ה-SDK של Metavision נכשל.\n"
                "ודא שתיקייה זו היא אכן תיקיית ההתקנה הראשית של Prophesee.\n"
                f"פרטי שגיאה:\n{e}"
            )

    def detect_camera_or_initialize(self):
        """Tries opening real EVK4, falls back gracefully to Simulated camera."""
        # Initialize custom moving simulation mode by default
        self.mock_camera = SimulatedCamera()
        self.mock_camera.start()
        self.mode_status_label.config(text="מצב סימולטור (מצלמה מדומה - EVK4 Mock)", fg="#ff9500")

        # Load initial slider configurations
        for name, var in self.bias_vars.items():
            self.mock_camera.update_biases(name, var.get())
            self.bias_val_labels[name].config(text=str(var.get()))

    def connect_to_physical_camera(self):
        """Attempts to dynamically connect to a real physical USB event camera."""
        if not METAVISION_AVAILABLE:
            messagebox.showerror(
                "שגיאה בחיבור",
                "ספריית Metavision SDK אינה מותקנת בסביבת פייתון זו.\n"
                "ודא שביצעת התקנה תקינה של ה-SDK והפורטים של פייתון זמינים."
            )
            return

        # Disable active mock or existing camera safely
        self.running_live = False
        if self.mock_camera:
            self.mock_camera.stop()
            self.mock_camera = None

        if self.camera_instance:
            try:
                self.camera_instance.stop()
            except:
                pass
            self.camera_instance = None
            self.slicer_instance = None

        try:
            # Discover and open physical USB device
            devs = DeviceDiscovery.list()
            if not devs:
                messagebox.showwarning(
                    "לא נמצאה מצלמה",
                    "לא זוהתה מצלמת אירועים של Prophesee מחוברת ב-USB במערכת.\n"
                    "ודא שהמצלמה מחוברת היטב, דולקת, ונסה שוב."
                )
                # Revert to simulator mode automatically
                self.detect_camera_or_initialize()
                return

            config = DeviceConfig()
            config.enable_biases_range_check_bypass(True)
            self.camera_instance = Camera.from_first_available(config)
            self.slicer_instance = CameraStreamSlicer(self.camera_instance.move())
            self.camera_width = self.camera_instance.width()
            self.camera_height = self.camera_instance.height()

            self.running_live = True
            self.mode_status_label.config(text="מצלמת EVK4 מחוברת (Metavision SDK)", fg="#30d158")

            # Fetch and apply initial bias values from physical device
            device = self.camera_instance.get_i_ll_biases()
            if device:
                for name in EVK4_BIAS_DEFAULTS.keys():
                    try:
                        current_val = device.get(name)
                        self.bias_vars[name].set(current_val)
                        self.bias_val_labels[name].config(text=str(current_val))
                    except:
                        pass

            # Start real camera frame polling worker
            self.camera_thread = threading.Thread(target=self.live_camera_worker, daemon=True)
            self.camera_thread.start()

            messagebox.showinfo("חיבור הצליח", "מצלמת Prophesee EVK4 חוברה בהצלחה! התצוגה והשליטה מנותבות לחומרה.")

        except Exception as e:
            messagebox.showerror("שגיאה בחיבור לחומרה", f"נכשלה פתיחת ההתקן הפיזי:\n{e}")
            # Revert to simulator mode automatically
            self.detect_camera_or_initialize()

    def live_camera_worker(self):
        """Worker thread that starts the real camera and continuously gets event slices from the slicer."""
        if not self.camera_instance or not self.slicer_instance:
            return
        try:
            self.camera_instance.start()
            for slice in self.slicer_instance:
                if not self.running_live:
                    break

                with self.lock:
                    self.shared_events_buffer = slice.events
                    if self.recording_active:
                        self.total_live_recorded_events += len(slice.events)
                        # Estimate recording file size
                        if self.live_recording_file_path and os.path.exists(self.live_recording_file_path):
                            self.recording_file_size_mb = os.path.getsize(self.live_recording_file_path) / (1024.0 * 1024.0)
                        else:
                            self.recording_file_size_mb += (len(slice.events) * 8) / (1024.0 * 1024.0)

        except Exception as e:
            print(f"Exception in Metavision SDK thread: {e}")
        finally:
            try:
                self.camera_instance.stop()
            except:
                pass

    def apply_erc_settings(self):
        rate = self.erc_rate.get()
        enabled = self.erc_enabled.get()

        if self.camera_instance and METAVISION_AVAILABLE:
            try:
                erc = self.camera_instance.get_i_erc_module()
                if erc:
                    erc.enable(enabled)
                    if enabled:
                        erc.set_cd_event_rate(rate)
            except Exception as e:
                print(f"ERC error: {e}")
        elif self.mock_camera:
            self.mock_camera.erc_enabled = enabled
            self.mock_camera.erc_rate = rate

    def apply_trail_settings(self):
        threshold = self.trail_filter_threshold_us.get()
        enabled = self.trail_filter_enabled.get()

        if self.camera_instance and METAVISION_AVAILABLE:
            try:
                trail = self.camera_instance.get_i_event_trail_filter_module()
                if trail:
                    trail.enable(enabled)
                    if enabled:
                        trail.set_threshold(threshold)
            except Exception as e:
                print(f"Trail filter error: {e}")
        elif self.mock_camera:
            self.mock_camera.trail_filter_enabled = enabled
            self.mock_camera.trail_filter_threshold = threshold

    def choose_recording_file(self):
        filename = filedialog.asksavesasfilename(
            defaultextension=".raw",
            filetypes=[("RAW events file", "*.raw"), ("All files", "*.*")]
        )
        if filename:
            self.recording_path.set(filename)

    def toggle_recording(self):
        if not self.recording_active:
            # Start Recording
            raw_path = self.recording_path.get()
            if not raw_path:
                messagebox.showerror("שגיאה", "אנא בחר מיקום תקין לקובץ ההקלטה")
                return

            # Create folder if missing
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

            try:
                if self.camera_instance and METAVISION_AVAILABLE:
                    self.total_live_recorded_events = 0
                    self.live_recording_file_path = raw_path
                    self.camera_instance.start_recording(raw_path)
                elif self.mock_camera:
                    self.mock_camera.start_recording(raw_path)

                self.recording_active = True
                self.record_btn.config(text="עצור הקלטה ⏹", style="RecordOn.TButton")
                self.path_entry.config(state="disabled")
            except Exception as e:
                messagebox.showerror("שגיאה בהקלטה", f"נכשלה התחלת ההקלטה:\n{e}")
        else:
            # Stop Recording
            try:
                if self.camera_instance and METAVISION_AVAILABLE:
                    self.camera_instance.stop_recording()
                elif self.mock_camera:
                    self.mock_camera.stop_recording()

                self.recording_active = False
                self.record_btn.config(text="התחל הקלטה 🔴", style="TButton")
                self.path_entry.config(state="normal")
                messagebox.showinfo("הקלטה הושלמה", f"קובץ ההקלטה נשמר בהצלחה ב:\n{self.recording_path.get()}")
            except Exception as e:
                messagebox.showerror("שגיאה בהקלטה", f"שגיאה בעת הפסקת ההקלטה:\n{e}")

    def save_camera_settings_json(self):
        """Saves current biases configuration in Metavision official JSON settings format."""
        biases_list = []
        for name, var in self.bias_vars.items():
            biases_list.append({
                "name": name,
                "value": var.get()
            })

        settings_data = {
            "ll_biases_state": {
                "bias": biases_list
            }
        }

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON settings file", "*.json")]
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(settings_data, f, indent=2)
                messagebox.showinfo("הצלחה", "ההגדרות נשמרו בהצלחה!")
            except Exception as e:
                messagebox.showerror("שגיאה", f"שגיאה בשמירת הגדרות:\n{e}")

    def load_camera_settings_json(self):
        """Loads biases from official Metavision JSON settings format."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON settings file", "*.json")]
        )
        if filename:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                biases = []
                if "ll_biases_state" in data and "bias" in data["ll_biases_state"]:
                    biases = data["ll_biases_state"]["bias"]
                elif "bias" in data:
                    biases = data["bias"]
                else:
                    messagebox.showerror("שגיאה", "פורמט ה-JSON אינו נתמך (חסר ll_biases_state)")
                    return

                for item in biases:
                    name = item.get("name")
                    value = item.get("value")
                    if name in self.bias_vars:
                        self.bias_vars[name].set(value)
                        self.on_bias_slider_moved(name, value)

                messagebox.showinfo("הצלחה", "הגדרות ה-Biases נטענו בהצלחה במערכת!")
            except Exception as e:
                messagebox.showerror("שגיאה", f"שגיאה בטעינת קובץ ההגדרות:\n{e}")

    def update_loop(self):
        """Continuous polling of events, frame rendering, graph plotting and stats updating."""
        dt = 0.030  # Update roughly every 30ms (~33 FPS)

        events = np.array([], dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'i2'), ('t', '<i8')])

        # Retrieve events
        if self.running_live:
            with self.lock:
                if self.shared_events_buffer is not None:
                    events = self.shared_events_buffer
                    self.shared_events_buffer = None # Consume/Read
        elif self.mock_camera:
            events = self.mock_camera.generate_events(dt)

        # Draw Visual Frame (Reconstructed Frame)
        img = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        # Accumulate events over the user-specified interval
        if len(events) > 0:
            max_t = events['t'][-1]
            min_t = max_t - int(self.accumulation_time_ms.get() * 1000)

            valid_events = events[events['t'] >= min_t]

            if len(valid_events) > 0:
                x_coords = valid_events['x']
                y_coords = valid_events['y']
                polarities = valid_events['p']

                # Filter indices out of image bounds to prevent crashes
                valid_mask = (x_coords < self.camera_width) & (y_coords < self.camera_height)
                x_coords = x_coords[valid_mask]
                y_coords = y_coords[valid_mask]
                polarities = polarities[valid_mask]

                # Fast vectorised pixel assignments (BGR format: Blue, Green, Red)
                img[y_coords[polarities == 1], x_coords[polarities == 1]] = [0, 230, 0] # Green
                img[y_coords[polarities == 0], x_coords[polarities == 0]] = [0, 0, 230] # Red (Red is index 2 in BGR)

        # Resize image cleanly and update label
        img_resized = cv2.resize(img, (580, 410))

        # Convert OpenCV to PhotoImage (img_resized is already BGR)
        _, buffer = cv2.imencode('.png', img_resized)
        self.tk_image = tk.PhotoImage(data=buffer.tobytes())
        self.image_label.config(image=self.tk_image)

        # Stat Calculations
        self.event_rate_live = len(events) / dt # events/sec

        # Update chart history
        elapsed = time.time() - self.start_app_time
        self.time_history.append(elapsed)
        self.rate_history.append(self.event_rate_live / 1000.0) # kEvt/sec

        # Keep only last 20 seconds of data
        if len(self.time_history) > 60:
            self.time_history.pop(0)
            self.rate_history.pop(0)

        # Plot redraw
        self.ax.clear()
        self.ax.set_facecolor("#1c1c1e")
        self.ax.plot(self.time_history, self.rate_history, color="#0a84ff", linewidth=2)
        self.ax.tick_params(axis='x', colors='#aeaeae')
        self.ax.tick_params(axis='y', colors='#aeaeae')
        self.ax.set_xlabel("זמן ריצה (שניות)", color='#aeaeae', fontname="Calibri", fontsize=10)
        self.ax.set_ylabel("קצב אירועים (kEvt/sec)", color='#aeaeae', fontname="Calibri", fontsize=10)
        self.canvas.draw()

        # Statistics Panel Labels update
        self.stat_event_rate.config(text=f"{int(self.event_rate_live):,}")

        if self.recording_active:
            if self.mock_camera:
                dur = time.time() - self.mock_camera.record_start_time
                size_mb = self.mock_camera.recorded_bytes / (1024 * 1024)
                tot_evts = self.mock_camera.recorded_events
            else:
                dur = time.time() - self.start_app_time # fallback
                size_mb = self.recording_file_size_mb
                tot_evts = self.total_live_recorded_events

            self.stat_duration.config(text=f"{dur:.1f} sec")
            self.stat_file_size.config(text=f"{size_mb:.2f} MB")
            self.stat_tot_events.config(text=f"{tot_evts:,}")
        else:
            self.stat_duration.config(text="0.0 sec")
            self.stat_file_size.config(text="0.0 MB")

        # Re-trigger loop
        self.after(30, self.update_loop)

    def quit(self):
        """Safety cleanup when exiting."""
        self.running_live = False
        if self.mock_camera:
            self.mock_camera.stop()
        if self.camera_instance and METAVISION_AVAILABLE:
            try:
                self.camera_instance.stop()
            except:
                pass
        super().quit()


if __name__ == "__main__":
    app = EventRecorderApp()
    # Handle window close event cleanly
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()
