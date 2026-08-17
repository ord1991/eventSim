#!/usr/bin/env python3
"""
Prophesee EVK4 Event Camera Connection & Viewer Application.
"""

import os
import sys
import time
import threading
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

        # Thread safety lock
        self.lock = threading.Lock()
        self.shared_display_frame = None
        self.running_live = False
        self.slicer_instance = None
        self.camera_instance = None
        self.camera_thread = None

        # Plotting & Stats History
        self.time_history = []
        self.rate_history = []
        self.start_app_time = time.time()
        self.event_rate_live = 0.0
        self.last_graph_update_time = 0.0  # Decoupled graph plotting rate limiter

        # GUI Controlled parameters
        self.accumulation_time_ms = tk.DoubleVar(value=30.0) # default 30ms accumulation
        self.accumulation_ms_val = 30.0  # Thread-safe float copy of accumulation_time_ms
        self.erc_enabled = tk.BooleanVar(value=False)
        self.erc_rate = tk.IntVar(value=1000000) # events/sec
        self.trail_filter_enabled = tk.BooleanVar(value=False)
        self.trail_filter_threshold_us = tk.IntVar(value=10000)

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

        self.connect_btn = ttk.Button(header_frame, text="Connect to Physical Camera (USB) 🔌", style="Action.TButton", command=self.connect_to_physical_camera)
        self.connect_btn.pack(side="right", padx=15, pady=10)

        # Main Work Area
        main_container = ttk.Frame(self)
        main_container.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Column 1: Live Video frame (Left, takes most space)
        col1 = ttk.Frame(main_container, style="Panel.TFrame", width=500, height=600)
        col1.pack(side="left", fill="both", expand=True, padx=5)
        col1.pack_propagate(False)

        title_lbl = ttk.Label(col1, text="Real-Time Live Event View", style="PanelTitle.TLabel")
        title_lbl.pack(anchor="nw", padx=15, pady=10)

        self.image_label = tk.Label(col1, bg="#000000")
        self.image_label.pack(fill="both", expand=True, padx=15, pady=15)

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
        """Middle panel containing real-time Matplotlib chart and accumulation slider below it."""
        title_lbl = ttk.Label(parent, text="Event Rate Rolling Timeline", style="PanelTitle.TLabel")
        title_lbl.pack(anchor="nw", padx=15, pady=10)

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
        self.ax.set_xlabel("Elapsed Time (seconds)", color='#aeaeae', fontname="Calibri", fontsize=10)
        self.ax.set_ylabel("Event Rate (kEvt/sec)", color='#aeaeae', fontname="Calibri", fontsize=10)

        self.line, = self.ax.plot([], [], color="#0a84ff", linewidth=2)

        # Embed chart into Tkinter widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=10)

        # Accumulation Time Slider Area directly underneath the chart
        acc_control_frame = ttk.Frame(parent, style="Panel.TFrame")
        acc_control_frame.pack(fill="x", side="bottom", padx=15, pady=15)

        slider_label = ttk.Label(acc_control_frame, text="Accumulation Time (ms):", style="PanelSec.TLabel")
        slider_label.pack(side="left", padx=5)

        self.acc_slider_val_lbl = ttk.Label(acc_control_frame, text="30.0 ms", style="PanelSec.TLabel", font=("Calibri", 12, "bold"), foreground="#30d158")
        self.acc_slider_val_lbl.pack(side="right", padx=5)

        acc_slider = ttk.Scale(
            acc_control_frame,
            from_=1.0,
            to=100.0,
            variable=self.accumulation_time_ms,
            orient="horizontal",
            command=self.on_accumulation_slider_moved
        )
        acc_slider.pack(fill="x", expand=True, side="left", padx=10)

    def on_accumulation_slider_moved(self, val):
        val_float = float(val)
        self.acc_slider_val_lbl.config(text=f"{val_float:.1f} ms")
        self.accumulation_ms_val = val_float

    def build_control_panel(self, parent):
        """Right sidebar containing Biases parameters, ERC, and Trail Filter controls."""
        # Scrollable container for control params
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

        # SDK / Connection Status Box
        status_sec = ttk.LabelFrame(scroll_frame, text="Connection & SDK Status", style="Panel.TFrame")
        status_sec.pack(fill="x", padx=10, pady=5)

        self.status_box_lbl = ttk.Label(status_sec, text="Identifying SDK...", style="PanelSec.TLabel", font=("Calibri", 12, "bold"))
        self.status_box_lbl.pack(anchor="nw", padx=10, pady=10)

        # 1. Parameter Tuning Section (Biases)
        bias_section = ttk.LabelFrame(scroll_frame, text="Hardware Biases (EVK4/IMX636)", style="Panel.TFrame")
        bias_section.pack(fill="x", padx=10, pady=5)

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

    def update_sdk_status(self):
        if METAVISION_AVAILABLE:
            self.status_box_lbl.config(text="Status: SDK detected successfully! Ready to connect.", foreground="#30d158")
        else:
            self.status_box_lbl.config(text="Status: SDK not installed / not identified.", foreground="#ff453a")

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
        self.running_live = False
        if self.slicer_instance:
            self.slicer_instance = None

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

        # Draw frame using user's white and gray accumulation logic
        display_frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame_counter = 0
        accumulated_events_count = 0
        last_calc_time = time.time()

        try:
            for evs in self.slicer_instance:
                if not self.running_live:
                    break

                # 1. Accumulate events
                if evs.size > 0:
                    display_frame[evs['y'], evs['x']] = np.where(evs['p'][:, None] == 1, [255, 255, 255], [100, 100, 100])
                    accumulated_events_count += evs.size

                frame_counter += 1

                # Read dynamically from thread-safe variable (converted to integer ms)
                frames_to_accumulate = max(1, int(self.accumulation_ms_val))

                if frame_counter >= frames_to_accumulate:
                    now = time.time()
                    dt = now - last_calc_time
                    if dt <= 0:
                        dt = 0.001

                    # Calculate event rate (events/sec) for this accumulated interval
                    evt_rate = accumulated_events_count / dt

                    # Thread-safe dispatch frame and stats to UI loop
                    with self.lock:
                        self.shared_display_frame = display_frame.copy()
                        self.event_rate_live = evt_rate

                    # Clear canvas for the next block to prevent smearing
                    display_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame_counter = 0
                    accumulated_events_count = 0
                    last_calc_time = now

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

        # Handle live video frame update (very fast, occurs on every loop)
        if frame is not None:
            # Resize image cleanly and update label
            img_resized = cv2.resize(frame, (620, 440))

            # Convert OpenCV to PhotoImage
            _, buffer = cv2.imencode('.png', img_resized)
            self.tk_image = tk.PhotoImage(data=buffer.tobytes())
            self.image_label.config(image=self.tk_image)

            # Append current rate to historical arrays
            elapsed = time.time() - self.start_app_time
            self.time_history.append(elapsed)
            self.rate_history.append(current_rate / 1000.0) # Convert to kEvt/sec

            # Bound the rolling timeline history to exactly 10 seconds of data.
            while self.time_history and (elapsed - self.time_history[0] > 10.0):
                self.time_history.pop(0)
                self.rate_history.pop(0)

            # Plot timeline updates at a decoupled rate (maximum once per 500ms)
            # This completely resolves CPU exhaustion and progressive GUI freezing/lag!
            now = time.time()
            if now - self.last_graph_update_time >= 0.5:
                self.last_graph_update_time = now

                if self.time_history:
                    self.line.set_data(self.time_history, self.rate_history)

                    # Auto-adjust plot limits with small padding
                    self.ax.set_xlim(self.time_history[0], self.time_history[-1] + 0.1)

                    min_rate = min(self.rate_history)
                    max_rate = max(self.rate_history)
                    # Ensure minimum 1.0 vertical span
                    if max_rate - min_rate < 1.0:
                        self.ax.set_ylim(max(0.0, min_rate - 0.5), min_rate + 1.0)
                    else:
                        self.ax.set_ylim(max(0.0, min_rate - 0.2 * (max_rate - min_rate)), max_rate + 0.2 * (max_rate - min_rate))

                    self.canvas.draw_idle()

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
