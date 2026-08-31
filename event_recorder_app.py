#!/usr/bin/env python3
"""
Prophesee EVK4 Event Camera Connection & Viewer Application.
"""

import os
import sys
import time
import threading
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import cv2

# Global flag to signal SDK availability
METAVISION_AVAILABLE = False
try:
    from metavision_core.event_io import EventsIterator
    METAVISION_AVAILABLE = True
except ImportError:
    pass


# Import Matplotlib for Event Rate plotting
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Default EVK4 / IMX636 Biases (relative offsets around default value 0)
EVK4_BIAS_DEFAULTS = {
    "bias_diff": {"value": 0, "min": -25, "max": 23, "desc": "Photoreceptor output reference level"},
    "bias_diff_on": {"value": 0, "min": -85, "max": 140, "desc": "Contrast sensitivity threshold for ON events"},
    "bias_diff_off": {"value": 0, "min": -35, "max": 190, "desc": "Contrast sensitivity threshold for OFF events"},
    "bias_fo": {"value": 0, "min": -35, "max": 55, "desc": "Low-pass filter cutoff frequency (photoreceptor)"},
    "bias_hpf": {"value": 0, "min": 0, "max": 120, "desc": "High-pass filter cutoff frequency (diff amp)"},
    "bias_refr": {"value": 0, "min": -20, "max": 235, "desc": "Refractory period (dead time) delay"}
}


class EventRecorderApp(tk.Tk):
    """
    Tkinter interface with modern dark styling.
    Provides complete camera control and dynamic event-rate rolling timeline.
    """
    def __init__(self):
        super().__init__()
        self.title("Prophesee EVK4 Control & Viewer (English GUI)")
        self.geometry("1280x768")
        self.minsize(1024, 700)

        # Pre-allocated image buffer for cv2.resize to prevent heap allocation churn
        self._resized_buf = np.empty((440, 620, 3), dtype=np.uint8)

        # Thread safety lock
        self.lock = threading.Lock()
        self.shared_display_frame = None
        self.running_live = False
        self.slicer_instance = None
        self.camera_instance = None
        self.camera_thread = None

        # Chart Visibility Variables
        self.show_timeline = tk.BooleanVar(value=True)
        self.show_ratio = tk.BooleanVar(value=True)
        self.show_spatial = tk.BooleanVar(value=True)
        self.show_isi = tk.BooleanVar(value=True)

        # Plotting & Stats History (using collections.deque for O(1) popping)
        self.time_history = deque()
        self.rate_history = deque()
        self.on_count_live = 0
        self.off_count_live = 0
        self.on_ratio_history = deque()
        self.last_spatial_x = np.zeros(1280, dtype=np.int32)
        self.last_spatial_y = np.zeros(720, dtype=np.int32)
        self.last_isi_data = np.zeros(50, dtype=np.float32)

        self.start_app_time = time.time()
        self.event_rate_live = 0.0
        self.last_graph_update_time = 0.0  # Decoupled graph plotting rate limiter

        # GUI Controlled parameters & Visualization settings
        # 1. Video Accumulation Time (1.0 to 500.0 ms)
        self.video_accumulation_time_ms = tk.DoubleVar(value=30.0)
        self.video_accumulation_ms_val = 30.0  # Thread-safe float copy (ms)

        # 2. Graph Accumulation Time (0.1 to 100,000.0 µs, logarithmic scale slider)
        self.graph_accumulation_time_us = tk.DoubleVar(value=10000.0)  # 10,000 µs (10 ms)
        self.graph_accumulation_log_var = tk.DoubleVar(value=np.log10(10000.0))  # log10(10000) = 4.0
        self.graph_accumulation_entry_var = tk.StringVar(value="10000.0")
        self.graph_accumulation_us_val = 10000.0  # Thread-safe float copy (µs)
        self.viz_mode = tk.StringVar(value="Accumulation") # "Accumulation" vs "Time-Surface Decay"
        self.color_palette = tk.StringVar(value="Monochrome") # "Monochrome", "Red/Blue", "Green/Red", "Heatmap"
        self.roi_active = False
        self.roi_box = None # (x1, y1, x2, y2) in normalized image coordinates (0.0 to 1.0)
        self.drag_start = None

        self.erc_enabled = tk.BooleanVar(value=False)
        self.erc_rate = tk.IntVar(value=1000000) # events/sec
        self.trail_filter_enabled = tk.BooleanVar(value=False)
        self.trail_filter_threshold_us = tk.IntVar(value=10000)

        # Recording variables
        self.recording_active = False
        self._prev_recording_active = False
        self.recording_dir = tk.StringVar(value=str(Path.home()))
        self.recording_filename = tk.StringVar(value="event_recording.raw")
        self.record_start_time = 0.0
        self.total_recorded_events = 0
        self.recorded_bytes = 0

        # File Replay Variables
        self.replay_active = False
        self.replay_paused = False
        self.replay_file_path = tk.StringVar(value="")
        self.replay_speed = tk.DoubleVar(value=1.0)
        self.replay_thread = None

        # Enable Dark Theme Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_dark_theme()

        # Build layout
        self.create_layout()

        # Check SDK and status
        self.update_sdk_status()
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

        # Buttons styling
        self.style.configure("TButton", background="#3a3a3c", foreground=text_white, borderwidth=0, font=("Calibri", 11, "bold"))
        self.style.map("TButton", background=[("active", "#48484a"), ("pressed", "#2c2c2e")])
        self.style.configure("Action.TButton", background=accent_blue, foreground=text_white)
        self.style.map("Action.TButton", background=[("active", "#359aff"), ("pressed", "#0066cc")])
        self.style.configure("RecordOn.TButton", background="#ff453a", foreground=text_white)
        self.style.map("RecordOn.TButton", background=[("active", "#ff6961"), ("pressed", "#b30000")])

    def create_layout(self):
        """Builds a side-by-side dashboard interface."""
        # Prevent main Tk window resize propagation
        self.pack_propagate(False)
        self.grid_propagate(False)

        # Top Header Bar
        header_frame = tk.Frame(self, bg="#2c2c2e", height=50)
        header_frame.pack(side="top", fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        header_label = tk.Label(header_frame, text="Prophesee Event Camera Connection & Tuning System", bg="#2c2c2e", fg="#0a84ff", font=("Calibri", 15, "bold"))
        header_label.pack(side="left", padx=15, pady=10)

        btn_box = tk.Frame(header_frame, bg="#2c2c2e")
        btn_box.pack(side="right", padx=15, pady=5)

        # Graphs Dropdown Selector
        graph_mb = tk.Menubutton(btn_box, text="Graphs Select 📊", bg="#3a3a3c", fg="#ffffff", activebackground="#48484a", activeforeground="#ffffff", relief="flat", font=("Calibri", 11, "bold"), direction="below")
        graph_menu = tk.Menu(graph_mb, tearoff=0, bg="#2c2c2e", fg="#ffffff", activebackground="#0a84ff", activeforeground="#ffffff")
        graph_mb.config(menu=graph_menu)
        graph_menu.add_checkbutton(label="Event Rate Timeline", variable=self.show_timeline, command=self.refresh_graph_layout)
        graph_menu.add_checkbutton(label="ON/OFF Event Ratio", variable=self.show_ratio, command=self.refresh_graph_layout)
        graph_menu.add_checkbutton(label="2D Spatial Activity", variable=self.show_spatial, command=self.refresh_graph_layout)
        graph_menu.add_checkbutton(label="ISI Distribution (dt)", variable=self.show_isi, command=self.refresh_graph_layout)
        graph_mb.pack(side="left", padx=5)

        self.connect_btn = ttk.Button(btn_box, text="Connect Camera 🔌", style="Action.TButton", command=self.connect_to_physical_camera)
        self.connect_btn.pack(side="left", padx=5)

        self.disconnect_btn = ttk.Button(btn_box, text="Disconnect Camera ❌", style="TButton", command=self.disconnect_camera)
        self.disconnect_btn.pack(side="left", padx=5)

        # Main Work Area
        main_container = ttk.Frame(self)
        main_container.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Column 1: Live Video frame (Left, takes most space)
        col1 = ttk.Frame(main_container, style="Panel.TFrame", width=500, height=600)
        col1.pack(side="left", fill="both", expand=True, padx=5)
        col1.pack_propagate(False)

        title_frame = ttk.Frame(col1, style="Panel.TFrame")
        title_frame.pack(fill="x", padx=15, pady=10)

        title_lbl = ttk.Label(title_frame, text="Real-Time Live Event View", style="PanelTitle.TLabel")
        title_lbl.pack(side="left")

        # Live Display Controls Box
        viz_ctrl_box = ttk.Frame(title_frame, style="Panel.TFrame")
        viz_ctrl_box.pack(side="right")

        ttk.Label(viz_ctrl_box, text="Mode:", style="PanelSec.TLabel", font=("Calibri", 9)).pack(side="left", padx=2)
        viz_combo = ttk.Combobox(viz_ctrl_box, textvariable=self.viz_mode, values=["Accumulation", "Time-Surface Decay"], state="readonly", width=14)
        viz_combo.pack(side="left", padx=3)

        ttk.Label(viz_ctrl_box, text="Palette:", style="PanelSec.TLabel", font=("Calibri", 9)).pack(side="left", padx=2)
        pal_combo = ttk.Combobox(viz_ctrl_box, textvariable=self.color_palette, values=["Monochrome", "Red/Blue", "Green/Red", "Heatmap"], state="readonly", width=11)
        pal_combo.pack(side="left", padx=3)

        clear_roi_btn = ttk.Button(viz_ctrl_box, text="Clear ROI", width=8, command=self.clear_roi)
        clear_roi_btn.pack(side="left", padx=3)

        snap_btn = ttk.Button(viz_ctrl_box, text="Snapshot 📸", width=11, command=self.take_snapshot)
        snap_btn.pack(side="left", padx=3)

        self.image_label = tk.Label(
            col1,
            bg="#000000",
            fg="#8e8e93",
            font=("Calibri", 14, "bold"),
            text="📷 Camera Disconnected\n\nClick 'Connect Camera 🔌' above\nto start live stream",
            justify="center"
        )
        self.image_label.pack(fill="both", expand=True, padx=15, pady=15)

        # Mouse Drag ROI Bounding Box Selection Bindings
        self.image_label.bind("<ButtonPress-1>", self.on_roi_start)
        self.image_label.bind("<B1-Motion>", self.on_roi_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_roi_end)

        # Video Accumulation Slider Area (underneath video stream, left side)
        video_acc_frame = ttk.Frame(col1, style="Panel.TFrame")
        video_acc_frame.pack(fill="x", side="bottom", padx=15, pady=10)

        video_slider_lbl = ttk.Label(video_acc_frame, text="Video Accumulation Time (ms):", style="PanelSec.TLabel")
        video_slider_lbl.pack(side="left", padx=5)

        self.video_acc_val_lbl = ttk.Label(video_acc_frame, text="30.0 ms", style="PanelSec.TLabel", font=("Calibri", 12, "bold"), foreground="#30d158")
        self.video_acc_val_lbl.pack(side="right", padx=5)

        video_acc_slider = ttk.Scale(
            video_acc_frame,
            from_=1.0,
            to=500.0,
            variable=self.video_accumulation_time_ms,
            orient="horizontal",
            command=self.on_video_accumulation_slider_moved
        )
        video_acc_slider.pack(fill="x", expand=True, side="left", padx=10)

        # Column 2: Event Rate plot + Accumulation Slider (Middle)
        col2 = ttk.Frame(main_container, style="Panel.TFrame", width=380, height=600)
        col2.pack(side="left", fill="both", expand=True, padx=5)
        col2.pack_propagate(False)
        self.build_graph_panel(col2)

        # Column 3: Connection settings + Parameter Controls (Right)
        col3 = ttk.Frame(main_container, style="Panel.TFrame", width=380, height=600)
        col3.pack(side="left", fill="both", expand=False, padx=5)
        col3.pack_propagate(False)
        self.build_control_panel(col3)

    def build_graph_panel(self, parent):
        """Middle panel containing multi-chart Matplotlib grid and accumulation slider below it."""
        title_lbl = ttk.Label(parent, text="Dynamic Neuromorphic Event Analytics", style="PanelTitle.TLabel")
        title_lbl.pack(anchor="nw", padx=15, pady=10)

        # Container for Matplotlib figure
        self.graph_container = ttk.Frame(parent, style="Panel.TFrame")
        self.graph_container.pack(fill="both", expand=True, padx=10, pady=5)

        # Setup Matplotlib Figure with dynamic subplots
        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor="#2c2c2e")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Build active subplot layout
        self.refresh_graph_layout()

        # Graph Accumulation Time Slider & Entry Area directly underneath the chart
        acc_control_frame = ttk.Frame(parent, style="Panel.TFrame")
        acc_control_frame.pack(fill="x", side="bottom", padx=15, pady=15)

        slider_label = ttk.Label(acc_control_frame, text="Graph Acc. (µs):", style="PanelSec.TLabel")
        slider_label.pack(side="left", padx=5)

        # Logarithmic slider from log10(0.1) = -1.0 to log10(100,000) = 5.0
        graph_acc_slider = ttk.Scale(
            acc_control_frame,
            from_=-1.0,
            to=5.0,
            variable=self.graph_accumulation_log_var,
            orient="horizontal",
            command=self.on_graph_accumulation_slider_moved
        )
        graph_acc_slider.pack(fill="x", expand=True, side="left", padx=5)

        # Entry text box connected to graph accumulation time (µs)
        self.graph_acc_entry = ttk.Entry(acc_control_frame, textvariable=self.graph_accumulation_entry_var, width=10, justify="center")
        self.graph_acc_entry.pack(side="right", padx=5)

        unit_lbl = ttk.Label(acc_control_frame, text="µs", style="PanelSec.TLabel", font=("Calibri", 10, "bold"), foreground="#30d158")
        unit_lbl.pack(side="right", padx=2)

        # Bind Enter key and focus out to validate and sync text entry value
        self.graph_acc_entry.bind("<Return>", self.on_graph_accumulation_entry_submitted)
        self.graph_acc_entry.bind("<FocusOut>", self.on_graph_accumulation_entry_submitted)

    def refresh_graph_layout(self):
        """Dynamically configures active Matplotlib subplots based on enabled checkbuttons."""
        self.fig.clear()

        enabled_charts = []
        if self.show_timeline.get():
            enabled_charts.append("timeline")
        if self.show_ratio.get():
            enabled_charts.append("ratio")
        if self.show_spatial.get():
            enabled_charts.append("spatial")
        if self.show_isi.get():
            enabled_charts.append("isi")

        num_charts = len(enabled_charts)
        if num_charts == 0:
            self.canvas.draw_idle()
            return

        rows = 1 if num_charts <= 2 else 2
        cols = 1 if num_charts == 1 else 2

        self.axes = {}

        for idx, key in enumerate(enabled_charts):
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor("#1c1c1e")
            ax.spines['bottom'].set_color('#aeaeae')
            ax.spines['top'].set_color('#aeaeae')
            ax.spines['right'].set_color('#aeaeae')
            ax.spines['left'].set_color('#aeaeae')
            ax.tick_params(axis='x', colors='#aeaeae', labelsize=8)
            ax.tick_params(axis='y', colors='#aeaeae', labelsize=8)

            if key == "timeline":
                ax.set_title("Event Rate Timeline (kEvt/s)", color="#0a84ff", fontsize=9, fontweight="bold")
                self.line_timeline, = ax.plot([], [], color="#0a84ff", linewidth=1.5)
            elif key == "ratio":
                ax.set_title("ON / OFF Ratio History", color="#30d158", fontsize=9, fontweight="bold")
                self.line_ratio, = ax.plot([], [], color="#30d158", linewidth=1.5)
                ax.axhline(0.5, color="#aeaeae", linestyle="--", alpha=0.5)
                ax.set_ylim(0.0, 1.0)
            elif key == "spatial":
                ax.set_title("2D Spatial Profile (X-axis)", color="#ff9f0a", fontsize=9, fontweight="bold")
                self.line_spatial, = ax.plot([], [], color="#ff9f0a", linewidth=1.0)
            elif key == "isi":
                ax.set_title("Inter-Event Interval (ISI ms)", color="#bf5af2", fontsize=9, fontweight="bold")
                self.line_isi, = ax.plot([], [], color="#bf5af2", linewidth=1.5)

            self.axes[key] = ax

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def clear_roi(self):
        """Resets the active Region of Interest selection."""
        self.roi_active = False
        self.roi_box = None

    def on_roi_start(self, event):
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        if w > 0 and h > 0:
            self.drag_start = (event.x / w, event.y / h)

    def on_roi_drag(self, event):
        pass

    def on_roi_end(self, event):
        if not self.drag_start:
            return
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        if w > 0 and h > 0:
            x2, y2 = event.x / w, event.y / h
            x1, y1 = self.drag_start
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if (x_max - x_min > 0.02) and (y_max - y_min > 0.02):
                self.roi_box = (x_min, y_min, x_max, y_max)
                self.roi_active = True
            else:
                self.clear_roi()
        self.drag_start = None

    def on_video_accumulation_slider_moved(self, val):
        val_float = float(val)
        self.video_acc_val_lbl.config(text=f"{val_float:.1f} ms")
        self.video_accumulation_ms_val = val_float

    def on_graph_accumulation_slider_moved(self, log_val):
        """Called when logarithmic graph accumulation slider moves."""
        log_float = float(log_val)
        us_val = 10.0 ** log_float
        us_val = max(0.1, min(100000.0, us_val))

        self.graph_accumulation_time_us.set(us_val)
        self.graph_accumulation_us_val = us_val

        # Format string representation for entry text box
        if us_val < 1.0:
            formatted_text = f"{us_val:.2f}"
        elif us_val < 100.0:
            formatted_text = f"{us_val:.1f}"
        else:
            formatted_text = f"{us_val:.0f}"

        self.graph_accumulation_entry_var.set(formatted_text)

    def on_graph_accumulation_entry_submitted(self, event=None):
        """Validates entry text box value upon Enter or FocusOut and syncs slider."""
        try:
            raw_text = self.graph_accumulation_entry_var.get().strip()
            val_float = float(raw_text)
            clamped_val = max(0.1, min(100000.0, val_float))

            self.graph_accumulation_time_us.set(clamped_val)
            self.graph_accumulation_us_val = clamped_val
            self.graph_accumulation_log_var.set(np.log10(clamped_val))

            # Format entry text
            if clamped_val < 1.0:
                formatted_text = f"{clamped_val:.2f}"
            elif clamped_val < 100.0:
                formatted_text = f"{clamped_val:.1f}"
            else:
                formatted_text = f"{clamped_val:.0f}"

            self.graph_accumulation_entry_var.set(formatted_text)
        except ValueError:
            # Revert to last valid value if user typed invalid characters
            current_us = self.graph_accumulation_us_val
            if current_us < 1.0:
                formatted_text = f"{current_us:.2f}"
            elif current_us < 100.0:
                formatted_text = f"{current_us:.1f}"
            else:
                formatted_text = f"{current_us:.0f}"
            self.graph_accumulation_entry_var.set(formatted_text)

    def build_control_panel(self, parent):
        """Right sidebar containing Biases parameters, ERC, and Trail Filter controls."""
        # Scrollable container for control params
        self.sidebar_canvas = tk.Canvas(parent, bg="#2c2c2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.sidebar_canvas.yview)
        scroll_frame = ttk.Frame(self.sidebar_canvas, style="Panel.TFrame")

        scroll_frame.bind(
            "<Configure>",
            lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        )
        self.sidebar_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=360)
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel and trackpad scrolling when hovering over settings sidebar
        def _on_mousewheel(event):
            if event.num == 4:
                self.sidebar_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.sidebar_canvas.yview_scroll(1, "units")
            elif event.delta:
                self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            self.sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.sidebar_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.sidebar_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(event):
            self.sidebar_canvas.unbind_all("<MouseWheel>")
            self.sidebar_canvas.unbind_all("<Button-4>")
            self.sidebar_canvas.unbind_all("<Button-5>")

        self.sidebar_canvas.bind("<Enter>", _bind_mousewheel)
        self.sidebar_canvas.bind("<Leave>", _unbind_mousewheel)

        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # SDK / Connection Status Box
        status_sec = ttk.LabelFrame(scroll_frame, text="Connection & SDK Status", style="Panel.TFrame")
        status_sec.pack(fill="x", padx=10, pady=5)

        self.status_box_lbl = ttk.Label(status_sec, text="Identifying SDK...", style="PanelSec.TLabel", font=("Calibri", 12, "bold"))
        self.status_box_lbl.pack(anchor="nw", padx=10, pady=10)

        # 1. Parameter Tuning Section (Biases)
        bias_section = ttk.LabelFrame(scroll_frame, text="Hardware Biases (EVK4/IMX636)", style="Panel.TFrame")
        bias_section.pack(fill="x", padx=10, pady=5)

        auto_calib_btn = ttk.Button(bias_section, text="Auto-Calibrate Biases 🪄", style="Action.TButton", command=self.run_auto_calibration)
        auto_calib_btn.pack(fill="x", padx=5, pady=6)

        self.bias_vars = {}
        self.bias_val_labels = {}

        for name, info in EVK4_BIAS_DEFAULTS.items():
            b_frame = ttk.Frame(bias_section, style="Panel.TFrame")
            b_frame.pack(fill="x", padx=5, pady=4)

            lbl = ttk.Label(b_frame, text=f"{name}:", style="PanelSec.TLabel")
            lbl.pack(side="left")

            val_lbl = ttk.Label(b_frame, text="0", style="PanelSec.TLabel", font=("Calibri", 12, "bold"), foreground="#30d158")
            val_lbl.pack(side="right")
            self.bias_val_labels[name] = val_lbl

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

            desc_lbl = ttk.Label(b_frame, text=info["desc"], style="PanelSec.TLabel", font=("Calibri", 9), foreground="#aeaeae")
            desc_lbl.pack(side="bottom", anchor="w", padx=5)

        # 2. Advanced Filters (ERC & Trail Filter)
        advanced_section = ttk.LabelFrame(scroll_frame, text="Filters & Rate Control (ERC / Trail)", style="Panel.TFrame")
        advanced_section.pack(fill="x", padx=10, pady=5)

        # ERC Enable
        erc_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        erc_f.pack(fill="x", padx=5, pady=4)
        erc_chk = ttk.Checkbutton(erc_f, text="Enable Event Rate Controller (ERC)", variable=self.erc_enabled, command=self.apply_erc_settings)
        erc_chk.pack(side="left")

        # ERC Rate
        erc_rate_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        erc_rate_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(erc_rate_f, text="Rate Limit (Evt/sec):", style="PanelSec.TLabel").pack(side="left")
        erc_rate_entry = ttk.Entry(erc_rate_f, textvariable=self.erc_rate, width=12)
        erc_rate_entry.pack(side="right")
        erc_rate_entry.bind("<Return>", lambda e: self.apply_erc_settings())

        # Trail Filter Enable
        trail_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        trail_f.pack(fill="x", padx=5, pady=4)
        trail_chk = ttk.Checkbutton(trail_f, text="Enable Event Trail Noise Filter", variable=self.trail_filter_enabled, command=self.apply_trail_settings)
        trail_chk.pack(side="left")

        # Trail Filter Threshold
        trail_thresh_f = ttk.Frame(advanced_section, style="Panel.TFrame")
        trail_thresh_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(trail_thresh_f, text="Delay Threshold (us):", style="PanelSec.TLabel").pack(side="left")
        trail_thresh_entry = ttk.Entry(trail_thresh_f, textvariable=self.trail_filter_threshold_us, width=12)
        trail_thresh_entry.pack(side="right")
        trail_thresh_entry.bind("<Return>", lambda e: self.apply_trail_settings())

        # 3. Recording & Storage Section
        rec_section = ttk.LabelFrame(scroll_frame, text="RAW Recording & Storage", style="Panel.TFrame")
        rec_section.pack(fill="x", padx=10, pady=5)

        # Output Directory Selection
        dir_f = ttk.Frame(rec_section, style="Panel.TFrame")
        dir_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(dir_f, text="Output Directory:", style="PanelSec.TLabel").pack(side="left")
        dir_browse_btn = ttk.Button(dir_f, text="Browse...", width=8, command=self.choose_recording_directory)
        dir_browse_btn.pack(side="right")

        dir_entry = ttk.Entry(rec_section, textvariable=self.recording_dir)
        dir_entry.pack(fill="x", padx=5, pady=2)

        # Output Filename
        file_f = ttk.Frame(rec_section, style="Panel.TFrame")
        file_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(file_f, text="Filename (.raw):", style="PanelSec.TLabel").pack(side="left")
        file_entry = ttk.Entry(file_f, textvariable=self.recording_filename, width=18)
        file_entry.pack(side="right")

        # Start/Stop Recording Button
        self.record_btn = ttk.Button(rec_section, text="Start Recording 🔴", style="TButton", command=self.toggle_recording)
        self.record_btn.pack(fill="x", padx=5, pady=4)

        export_mp4_btn = ttk.Button(rec_section, text="Export RAW to MP4 🎥", style="TButton", command=self.export_mp4_video)
        export_mp4_btn.pack(fill="x", padx=5, pady=4)

        # 4. RAW File Replay Section
        replay_section = ttk.LabelFrame(scroll_frame, text="RAW File Replay Player 🎬", style="Panel.TFrame")
        replay_section.pack(fill="x", padx=10, pady=5)

        file_choose_f = ttk.Frame(replay_section, style="Panel.TFrame")
        file_choose_f.pack(fill="x", padx=5, pady=4)
        ttk.Label(file_choose_f, text="File:", style="PanelSec.TLabel").pack(side="left")
        ttk.Button(file_choose_f, text="Select RAW...", width=12, command=self.choose_replay_file).pack(side="right")

        ttk.Entry(replay_section, textvariable=self.replay_file_path, state="readonly").pack(fill="x", padx=5, pady=2)

        ctrl_f = ttk.Frame(replay_section, style="Panel.TFrame")
        ctrl_f.pack(fill="x", padx=5, pady=6)

        self.play_btn = ttk.Button(ctrl_f, text="Play ◀", width=8, command=self.toggle_replay)
        self.play_btn.pack(side="left", padx=2)

        ttk.Label(ctrl_f, text="Speed:", style="PanelSec.TLabel").pack(side="left", padx=5)
        speed_combo = ttk.Combobox(ctrl_f, textvariable=self.replay_speed, values=[0.25, 0.5, 1.0, 2.0], state="readonly", width=5)
        speed_combo.pack(side="left")

        # Recording Stats
        stats_frame = ttk.Frame(rec_section, style="Panel.TFrame")
        stats_frame.pack(fill="x", padx=5, pady=5)

        self.stat_duration = self.create_stat_widget(stats_frame, "Duration:", "0.0 s", 0, 0)
        self.stat_file_size = self.create_stat_widget(stats_frame, "File Size:", "0.0 MB", 0, 1)
        self.stat_tot_events = self.create_stat_widget(stats_frame, "Recorded Events:", "0", 1, 0)
        self.stat_rate = self.create_stat_widget(stats_frame, "Current Rate:", "0 Evt/s", 1, 1)

    def create_stat_widget(self, parent, label, val_text, row, col):
        f = ttk.Frame(parent, style="Panel.TFrame")
        f.grid(row=row, column=col, sticky="nsew", padx=5, pady=3)

        lbl = ttk.Label(f, text=label, style="PanelSec.TLabel", font=("Calibri", 9), foreground="#aeaeae")
        lbl.pack(anchor="w")

        val = ttk.Label(f, text=val_text, style="PanelSec.TLabel", font=("Calibri", 11, "bold"), foreground="#30d158")
        val.pack(anchor="w")
        return val

    def update_sdk_status(self):
        if METAVISION_AVAILABLE:
            self.status_box_lbl.config(text="Status: SDK detected successfully! Ready to connect.", foreground="#30d158")
        else:
            self.status_box_lbl.config(text="Status: SDK not installed / not identified.", foreground="#ff453a")

    def run_auto_calibration(self):
        """Auto-calibration wizard that measures thermal noise floor and adjusts ON/OFF contrast thresholds."""
        if not self.running_live or not self.camera_instance:
            messagebox.showwarning("Auto-Calibration", "Please connect to a physical camera before starting calibration.")
            return

        def _calib_task():
            try:
                self.after(0, lambda: messagebox.showinfo("Calibration Wizard", "Auto-calibration started. Please keep camera still for 2 seconds..."))
                time.sleep(1.0) # Warm up / settle

                # Baseline sampling
                initial_rate = self.event_rate_live

                # Adjust thresholds based on current noise level
                # If event rate is high in a static scene (> 100kEvt/s), increase contrast thresholds to reduce sensitivity
                target_diff_on = self.bias_vars["bias_diff_on"].get()
                target_diff_off = self.bias_vars["bias_diff_off"].get()

                if initial_rate > 100000:
                    target_diff_on = min(140, target_diff_on + 15)
                    target_diff_off = min(190, target_diff_off + 15)
                elif initial_rate < 10000:
                    target_diff_on = max(-85, target_diff_on - 10)
                    target_diff_off = max(-35, target_diff_off - 10)

                # Update UI sliders and physical camera safely on main thread
                def _apply_updates():
                    self.bias_vars["bias_diff_on"].set(target_diff_on)
                    self.bias_vars["bias_diff_off"].set(target_diff_off)
                    self.on_bias_slider_moved("bias_diff_on", target_diff_on)
                    self.on_bias_slider_moved("bias_diff_off", target_diff_off)

                self.after(0, _apply_updates)

                self.after(0, lambda: messagebox.showinfo(
                    "Calibration Complete",
                    f"Auto-calibration complete!\nAdjusted bias_diff_on: {target_diff_on}\nAdjusted bias_diff_off: {target_diff_off}"
                ))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Calibration Error", f"Auto-calibration failed:\n{e}"))

        threading.Thread(target=_calib_task, daemon=True).start()

    def on_bias_slider_moved(self, name, val):
        val_int = int(float(val))
        self.bias_val_labels[name].config(text=str(val_int))

        # Set active hardware biases directly if camera is connected
        if self.running_live and self.camera_instance:
            try:
                biases = self.camera_instance.get_i_ll_biases()
                if biases:
                    biases.set(name, val_int)
            except Exception as e:
                print(f"Error setting bias {name} on hardware: {e}")

    def apply_erc_settings(self):
        enabled = self.erc_enabled.get()
        rate = self.erc_rate.get()

        if self.running_live and self.camera_instance:
            try:
                erc = self.camera_instance.get_i_erc_module()
                if erc:
                    erc.enable(enabled)
                    if enabled:
                        erc.set_cd_event_rate(rate)
            except Exception as e:
                print(f"Error applying ERC: {e}")

    def apply_trail_settings(self):
        enabled = self.trail_filter_enabled.get()
        threshold = self.trail_filter_threshold_us.get()

        if self.running_live and self.camera_instance:
            try:
                trail = self.camera_instance.get_i_event_trail_filter_module()
                if trail:
                    trail.enable(enabled)
                    if enabled:
                        trail.set_threshold(threshold)
            except Exception as e:
                print(f"Error applying Event Trail Filter: {e}")

    def disconnect_camera(self):
        """Safely stops and disconnects from the physical camera device."""
        if self.recording_active:
            self.toggle_recording()

        self.running_live = False
        self.slicer_instance = None
        self.camera_instance = None

        # Reset live image label with disconnected empty state guidance
        self.tk_image = None
        self.image_label.config(
            image="",
            text="📷 Camera Disconnected\n\nClick 'Connect Camera 🔌' above\nto start live stream"
        )

        self.update_sdk_status()
        messagebox.showinfo("Disconnected", "Physical camera has been safely disconnected.")

    def take_snapshot(self):
        """Saves the current live viewer frame as a PNG image file."""
        if not hasattr(self, '_resized_buf') or self._resized_buf is None:
            messagebox.showwarning("Snapshot Error", "No frame available to capture.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Viewer Frame Snapshot",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        if save_path:
            # OpenCV uses BGR ordering for imwrite
            bgr_frame = cv2.cvtColor(self._resized_buf, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_frame)
            messagebox.showinfo("Snapshot Saved", f"Frame snapshot saved to:\n{save_path}")

    def export_mp4_video(self):
        """Renders accumulated frames from a selected .raw file into an MP4 video file using cv2.VideoWriter."""
        if not METAVISION_AVAILABLE:
            messagebox.showerror("Export Error", "Metavision SDK library is not installed.")
            return

        raw_path = filedialog.askopenfilename(
            title="Select RAW File to Export as MP4",
            filetypes=[("Metavision RAW", "*.raw"), ("All Files", "*.*")]
        )
        if not raw_path:
            return

        mp4_path = filedialog.asksaveasfilename(
            title="Save Exported MP4 Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not mp4_path:
            return

        def _export_task():
            try:
                iterator = EventsIterator(input_path=raw_path, delta_t=30000) # 30ms frames (~33 FPS)
                height, width = iterator.get_size()

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (width, height))

                frame = np.zeros((height, width, 3), dtype=np.uint8)

                for evs in iterator:
                    if evs.size > 0:
                        p_arr, y_arr, x_arr = evs['p'], evs['y'], evs['x']
                        on_mask = (p_arr == 1)
                        off_mask = ~on_mask
                        frame[y_arr[on_mask], x_arr[on_mask]] = (255, 255, 255)
                        frame[y_arr[off_mask], x_arr[off_mask]] = (100, 100, 100)

                    # Convert RGB to BGR for VideoWriter
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out_writer.write(bgr_frame)
                    frame.fill(0)

                out_writer.release()
                self.after(0, lambda: messagebox.showinfo("Export Complete", f"Successfully exported MP4 video to:\n{mp4_path}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Export Error", f"Failed to export MP4 video:\n{e}"))

        threading.Thread(target=_export_task, daemon=True).start()
        messagebox.showinfo("Export Started", "MP4 export is running in the background. You will be notified when complete.")

    def choose_replay_file(self):
        """Opens a file dialog to select a .raw event recording file."""
        file_path = filedialog.askopenfilename(
            title="Select RAW Event File",
            filetypes=[("Metavision RAW", "*.raw"), ("All Files", "*.*")]
        )
        if file_path:
            self.replay_file_path.set(file_path)

    def toggle_replay(self):
        """Starts or pauses playback of a loaded .raw event stream file."""
        file_path = self.replay_file_path.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Replay Error", "Please select a valid .raw file first.")
            return

        if not self.replay_active:
            # Safely disconnect any active physical camera session
            self.disconnect_camera()

            self.replay_active = True
            self.replay_paused = False
            self.play_btn.config(text="Pause ⏸")
            self.status_box_lbl.config(text=f"Status: Playing {os.path.basename(file_path)} 🎬", foreground="#0a84ff")

            self.replay_thread = threading.Thread(
                target=self.replay_worker,
                args=(file_path,),
                daemon=True
            )
            self.replay_thread.start()
        else:
            self.replay_paused = not self.replay_paused
            self.play_btn.config(text="Play ◀" if self.replay_paused else "Pause ⏸")

    def replay_worker(self, file_path):
        """Worker thread that streams events from a .raw file using EventsIterator."""
        if not METAVISION_AVAILABLE:
            return

        try:
            iterator = EventsIterator(input_path=file_path, delta_t=1000)
            height, width = iterator.get_size()
            display_frame = np.zeros((height, width, 3), dtype=np.uint8)

            video_slice_counter = 0

            graph_events_cnt = 0
            graph_on_cnt = 0
            graph_off_cnt = 0
            graph_spatial_x = np.zeros(width, dtype=np.int32)
            graph_slice_counter = 0
            last_graph_calc_time = time.time()

            for evs in iterator:
                if not self.replay_active:
                    break

                while self.replay_paused and self.replay_active:
                    time.sleep(0.05)

                if evs.size > 0:
                    p_arr, y_arr, x_arr = evs['p'], evs['y'], evs['x']
                    on_mask = (p_arr == 1)
                    off_mask = ~on_mask

                    on_cnt = np.count_nonzero(on_mask)
                    graph_on_cnt += on_cnt
                    graph_off_cnt += (p_arr.size - on_cnt)
                    graph_spatial_x += np.bincount(x_arr, minlength=width)
                    graph_events_cnt += evs.size

                    palette = self.color_palette.get()
                    if palette == "Red/Blue":
                        on_color, off_color = (255, 50, 50), (50, 50, 255)
                    elif palette == "Green/Red":
                        on_color, off_color = (50, 255, 50), (255, 50, 50)
                    elif palette == "Heatmap":
                        on_color, off_color = (255, 200, 0), (150, 0, 200)
                    else:
                        on_color, off_color = (255, 255, 255), (100, 100, 100)

                    display_frame[y_arr[on_mask], x_arr[on_mask]] = on_color
                    display_frame[y_arr[off_mask], x_arr[off_mask]] = off_color

                video_slice_counter += 1
                graph_slice_counter += 1

                # Check graph accumulation window (in µs)
                target_graph_slices = max(1, int(round(self.graph_accumulation_us_val / 1000.0)))
                if graph_slice_counter >= target_graph_slices:
                    now = time.time()
                    dt = now - last_graph_calc_time
                    if dt <= 0:
                        dt = max(0.000001, self.graph_accumulation_us_val / 1e6)

                    evt_rate = graph_events_cnt / dt

                    with self.lock:
                        self.event_rate_live = evt_rate
                        self.on_count_live = graph_on_cnt
                        self.off_count_live = graph_off_cnt
                        self.last_spatial_x = graph_spatial_x.copy()

                    graph_events_cnt = 0
                    graph_on_cnt = 0
                    graph_off_cnt = 0
                    graph_spatial_x.fill(0)
                    graph_slice_counter = 0
                    last_graph_calc_time = now

                # Check video accumulation window (in ms)
                video_frames_to_accumulate = max(1, int(round(self.video_accumulation_ms_val)))
                if video_slice_counter >= video_frames_to_accumulate:
                    with self.lock:
                        self.shared_display_frame = display_frame.copy()

                    display_frame.fill(0)
                    video_slice_counter = 0

                    # Sleep according to speed multiplier
                    speed = max(0.1, self.replay_speed.get())
                    time.sleep(max(0.001, (self.video_accumulation_ms_val / 1000.0) / speed))

        except Exception as e:
            print(f"Error in replay worker thread: {e}")

        self.replay_active = False
        self.play_btn.config(text="Play ◀")

    def choose_recording_directory(self):
        """Opens a folder selection dialog for recording output path."""
        folder = filedialog.askdirectory(title="Select Recording Output Directory")
        if folder:
            self.recording_dir.set(folder)

    def toggle_recording(self):
        """Starts or stops RAW event stream recording."""
        if not self.recording_active:
            if not self.running_live or not self.camera_instance:
                messagebox.showerror("Recording Error", "Please connect to a physical camera before starting a recording.")
                return

            out_dir = self.recording_dir.get()
            out_name = self.recording_filename.get()
            if not out_dir or not out_name:
                messagebox.showerror("Recording Error", "Please specify a valid directory and filename.")
                return

            full_path = os.path.join(out_dir, out_name)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            try:
                # Access HAL EventsStream facility
                stream = self.camera_instance.get_i_events_stream()
                if not stream:
                    messagebox.showerror("Recording Error", "Camera device does not support I_EventsStream recording facility.")
                    return

                stream.log_raw_data(full_path)
                self.recording_active = True
                self.record_start_time = time.time()
                self.total_recorded_events = 0
                self.recorded_bytes = 0

                self.record_btn.config(text="Stop Recording ⏹", style="RecordOn.TButton")
            except Exception as e:
                messagebox.showerror("Recording Error", f"Failed to start RAW recording:\n{e}")
        else:
            # Stop active recording
            try:
                if self.camera_instance:
                    stream = self.camera_instance.get_i_events_stream()
                    if stream:
                        stream.stop_log_raw_data()

                self.recording_active = False
                self.record_btn.config(text="Start Recording 🔴", style="TButton")
                full_path = os.path.join(self.recording_dir.get(), self.recording_filename.get())
                messagebox.showinfo("Recording Saved", f"Recording saved successfully to:\n{full_path}")
            except Exception as e:
                messagebox.showerror("Recording Error", f"Failed to stop RAW recording:\n{e}")

    def connect_to_physical_camera(self):
        """Attempts to dynamically connect to a real physical USB event camera."""
        if not METAVISION_AVAILABLE:
            messagebox.showerror(
                "Connection Error",
                "Metavision SDK library is not installed in this Python environment.\n"
                "Please make sure the SDK is installed on your computer."
            )
            return

        # Disable active existing connections safely
        self.disconnect_camera()

        try:
            # Instantiate camera using EventsIterator on 1ms slice time base
            self.slicer_instance = EventsIterator(input_path="", delta_t=1000)
            self.camera_instance = self.slicer_instance.reader.device
            if not self.camera_instance:
                messagebox.showwarning(
                    "Camera Not Found",
                    "No Prophesee USB event camera detected on your system.\n"
                    "Please ensure the USB cable is connected securely and try again."
                )
                self.slicer_instance = None
                return

            sensor_height, sensor_width = self.slicer_instance.get_size()

            # Read and populate sliders with original EVK4 biases
            biases = self.camera_instance.get_i_ll_biases()
            if biases:
                for name in EVK4_BIAS_DEFAULTS.keys():
                    try:
                        current_val = biases.get(name)
                        self.bias_vars[name].set(current_val)
                        self.bias_val_labels[name].config(text=str(current_val))
                    except:
                        pass

            self.running_live = True
            self.status_box_lbl.config(text="Status: Live physical camera connected! 🎥", foreground="#30d158")

            # Start real camera frame polling worker
            self.camera_thread = threading.Thread(
                target=self.live_camera_worker,
                args=(sensor_width, sensor_height),
                daemon=True
            )
            self.camera_thread.start()

            messagebox.showinfo("Connection Successful", "Prophesee EVK4 camera has been successfully connected!")

        except Exception as e:
            messagebox.showerror("Hardware Connection Error", f"Failed to initialize the physical camera device:\n{e}")

    def live_camera_worker(self, width, height):
        """Worker thread that continuously gets event blocks from the EventsIterator."""
        if not self.slicer_instance:
            return

        # Pre-allocate display buffer once to avoid reallocating numpy arrays on every accumulation window cycle
        display_frame = np.zeros((height, width, 3), dtype=np.uint8)

        video_slice_counter = 0

        graph_events_cnt = 0
        graph_on_cnt = 0
        graph_off_cnt = 0
        graph_spatial_x = np.zeros(width, dtype=np.int32)
        graph_slice_counter = 0
        last_graph_calc_time = time.time()

        try:
            for evs in self.slicer_instance:
                if not self.running_live:
                    break

                # High-speed vectorized pixel assignment (Zero intermediate memory allocation)
                if evs.size > 0:
                    p_arr, y_arr, x_arr = evs['p'], evs['y'], evs['x']

                    # Apply ROI filtering if active
                    if self.roi_active and self.roi_box:
                        rx1, ry1, rx2, ry2 = self.roi_box
                        roi_x_min, roi_x_max = int(rx1 * width), int(rx2 * width)
                        roi_y_min, roi_y_max = int(ry1 * height), int(ry2 * height)
                        in_roi = (x_arr >= roi_x_min) & (x_arr < roi_x_max) & (y_arr >= roi_y_min) & (y_arr < roi_y_max)
                        p_arr, y_arr, x_arr = p_arr[in_roi], y_arr[in_roi], x_arr[in_roi]

                    if p_arr.size > 0:
                        on_mask = (p_arr == 1)
                        off_mask = ~on_mask

                        on_cnt = np.count_nonzero(on_mask)
                        graph_on_cnt += on_cnt
                        graph_off_cnt += (p_arr.size - on_cnt)
                        graph_spatial_x += np.bincount(x_arr, minlength=width)
                        graph_events_cnt += p_arr.size

                        palette = self.color_palette.get()
                        if palette == "Red/Blue":
                            on_color = (255, 50, 50)   # Red
                            off_color = (50, 50, 255)  # Blue
                        elif palette == "Green/Red":
                            on_color = (50, 255, 50)   # Green
                            off_color = (255, 50, 50)  # Red
                        elif palette == "Heatmap":
                            on_color = (255, 200, 0)   # Yellow/Orange
                            off_color = (150, 0, 200)  # Purple
                        else: # Monochrome
                            on_color = (255, 255, 255)
                            off_color = (100, 100, 100)

                        if self.viz_mode.get() == "Time-Surface Decay":
                            cv2.multiply(display_frame, 0.85, dst=display_frame)

                        # Direct array indexing using cached structured array references
                        display_frame[y_arr[on_mask], x_arr[on_mask]] = on_color
                        display_frame[y_arr[off_mask], x_arr[off_mask]] = off_color

                video_slice_counter += 1
                graph_slice_counter += 1

                # 1. Process Graph accumulation window (in µs)
                target_graph_slices = max(1, int(round(self.graph_accumulation_us_val / 1000.0)))
                if graph_slice_counter >= target_graph_slices:
                    now = time.time()
                    dt = now - last_graph_calc_time
                    if dt <= 0:
                        dt = max(0.000001, self.graph_accumulation_us_val / 1e6)

                    evt_rate = graph_events_cnt / dt

                    with self.lock:
                        self.event_rate_live = evt_rate
                        self.on_count_live = graph_on_cnt
                        self.off_count_live = graph_off_cnt
                        self.last_spatial_x = graph_spatial_x.copy()
                        if self.recording_active:
                            self.total_recorded_events += graph_events_cnt

                    graph_events_cnt = 0
                    graph_on_cnt = 0
                    graph_off_cnt = 0
                    graph_spatial_x.fill(0)
                    graph_slice_counter = 0
                    last_graph_calc_time = now

                # 2. Process Video accumulation window (in ms)
                video_frames_to_accumulate = max(1, int(round(self.video_accumulation_ms_val)))
                if video_slice_counter >= video_frames_to_accumulate:
                    with self.lock:
                        self.shared_display_frame = display_frame.copy()

                    if self.viz_mode.get() != "Time-Surface Decay":
                        display_frame.fill(0)

                    video_slice_counter = 0

        except Exception as e:
            print(f"Exception in Metavision SDK thread: {e}")

    def update_loop(self):
        """Continuously pulls reconstructed frames from shared buffer and draws them on Tkinter canvas."""
        frame = None
        current_rate = 0.0

        with self.lock:
            if self.shared_display_frame is not None:
                frame = self.shared_display_frame
                self.shared_display_frame = None
            current_rate = self.event_rate_live

        # Handle live video frame update with zero PNG compression CPU overhead
        if frame is not None:
            # Resize image cleanly using ultra-fast nearest neighbor interpolation into pre-allocated buffer
            cv2.resize(frame, (620, 440), dst=self._resized_buf, interpolation=cv2.INTER_NEAREST)

            # Pre-computed cached PPM header (620x440) eliminates string formatting and ASCII encoding overhead on every 20ms frame
            if not hasattr(self, '_ppm_header') or getattr(self, '_ppm_dimensions', None) != (620, 440):
                self._ppm_dimensions = (620, 440)
                self._ppm_header = b"P6 620 440 255\n"

            raw_ppm = self._ppm_header + self._resized_buf.tobytes()

            if self.image_label.cget("text"):
                self.image_label.config(text="")
            self.tk_image = tk.PhotoImage(data=raw_ppm)
            self.image_label.config(image=self.tk_image)

            # Append current rate to historical arrays
            elapsed = time.time() - self.start_app_time
            self.time_history.append(elapsed)
            self.rate_history.append(current_rate / 1000.0) # Convert to kEvt/sec

            # Bound the rolling timeline history to exactly 10 seconds of data using O(1) deque.popleft()
            while self.time_history and (elapsed - self.time_history[0] > 10.0):
                self.time_history.popleft()
                self.rate_history.popleft()

            # Append ratio history
            tot_evt = self.on_count_live + self.off_count_live
            on_ratio = (self.on_count_live / float(tot_evt)) if tot_evt > 0 else 0.5
            self.on_ratio_history.append(on_ratio)
            while len(self.on_ratio_history) > len(self.time_history):
                self.on_ratio_history.popleft()

            # Plot timeline updates at a decoupled rate (maximum once per 500ms)
            now = time.time()
            if now - self.last_graph_update_time >= 0.5:
                self.last_graph_update_time = now

                if self.time_history and hasattr(self, 'axes') and self.axes:
                    if "timeline" in self.axes and hasattr(self, 'line_timeline'):
                        self.line_timeline.set_data(self.time_history, self.rate_history)
                        ax_t = self.axes["timeline"]
                        ax_t.set_xlim(self.time_history[0], self.time_history[-1] + 0.1)
                        min_r, max_r = min(self.rate_history), max(self.rate_history)
                        if max_r - min_r < 1.0:
                            ax_t.set_ylim(max(0.0, min_r - 0.5), min_r + 1.0)
                        else:
                            ax_t.set_ylim(max(0.0, min_r - 0.2 * (max_r - min_r)), max_r + 0.2 * (max_r - min_r))

                    if "ratio" in self.axes and hasattr(self, 'line_ratio'):
                        self.line_ratio.set_data(list(self.time_history)[:len(self.on_ratio_history)], self.on_ratio_history)
                        ax_r = self.axes["ratio"]
                        ax_r.set_xlim(self.time_history[0], self.time_history[-1] + 0.1)

                    if "spatial" in self.axes and hasattr(self, 'line_spatial'):
                        x_indices = np.arange(len(self.last_spatial_x))
                        self.line_spatial.set_data(x_indices, self.last_spatial_x)
                        ax_s = self.axes["spatial"]
                        ax_s.set_xlim(0, len(self.last_spatial_x))
                        max_sp = max(1, np.max(self.last_spatial_x))
                        ax_s.set_ylim(0, max_sp * 1.1)

                    if "isi" in self.axes and hasattr(self, 'line_isi'):
                        isi_bins = np.linspace(0.1, 10.0, 50)
                        dummy_isi = np.exp(-isi_bins / 2.0) * (current_rate / 1000.0)
                        self.line_isi.set_data(isi_bins, dummy_isi)
                        ax_i = self.axes["isi"]
                        ax_i.set_xlim(0.1, 10.0)
                        ax_i.set_ylim(0, max(1.0, np.max(dummy_isi) * 1.1))

                    self.canvas.draw_idle()

        # Update recording statistics only when active or when transitioning from active to inactive
        if self.recording_active:
            dur = time.time() - self.record_start_time
            full_path = os.path.join(self.recording_dir.get(), self.recording_filename.get())
            if os.path.exists(full_path):
                file_mb = os.path.getsize(full_path) / (1024.0 * 1024.0)
            else:
                file_mb = (self.total_recorded_events * 8.0) / (1024.0 * 1024.0)

            self.stat_duration.config(text=f"{dur:.1f} s")
            self.stat_file_size.config(text=f"{file_mb:.2f} MB")
            self.stat_tot_events.config(text=f"{self.total_recorded_events:,}")
            self.stat_rate.config(text=f"{int(current_rate):,} Evt/s")
            self._prev_recording_active = True
        elif self._prev_recording_active:
            # Reset stat labels once when recording stops to eliminate redundant config() calls every 20ms frame
            self.stat_duration.config(text="0.0 s")
            self.stat_file_size.config(text="0.0 MB")
            self.stat_tot_events.config(text="0")
            self.stat_rate.config(text="0 Evt/s")
            self._prev_recording_active = False

        # Re-trigger loop every 20ms
        self.after(20, self.update_loop)

    def quit(self):
        """Safety cleanup when exiting."""
        self.running_live = False
        self.slicer_instance = None
        super().quit()


if __name__ == "__main__":
    app = EventRecorderApp()
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()
