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


class EventRecorderApp(tk.Tk):
    """
    Tkinter interface with modern dark styling.
    Focuses purely on connecting to the physical USB camera and displaying its live video.
    """
    def __init__(self):
        super().__init__()
        self.title("Prophesee EVK4 Viewer (Hebrew GUI)")
        self.geometry("1024x700")
        self.minsize(800, 600)

        # Thread safety lock
        self.lock = threading.Lock()
        self.shared_display_frame = None
        self.running_live = False
        self.slicer_instance = None
        self.camera_thread = None

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
        # Top Header Bar
        header_frame = tk.Frame(self, bg="#2c2c2e", height=50)
        header_frame.pack(side="top", fill="x", padx=0, pady=0)

        header_label = tk.Label(header_frame, text="מערכת חיבור ותצוגה למצלמת אירועים Prophesee", bg="#2c2c2e", fg="#0a84ff", font=("Calibri", 15, "bold"))
        header_label.pack(side="right", padx=15, pady=10)

        self.connect_btn = ttk.Button(header_frame, text="התחבר למצלמה פיזית (USB) 🔌", style="Action.TButton", command=self.connect_to_physical_camera)
        self.connect_btn.pack(side="left", padx=15, pady=10)

        # Main Work Area
        main_container = ttk.Frame(self)
        main_container.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Column 1: Live Video frame (Left, takes most space)
        col1 = ttk.Frame(main_container, style="Panel.TFrame")
        col1.pack(side="left", fill="both", expand=True, padx=5)

        title_lbl = ttk.Label(col1, text="תצוגת וידאו בזמן אמת (Live View)", style="PanelTitle.TLabel")
        title_lbl.pack(anchor="ne", padx=15, pady=10)

        self.image_label = tk.Label(col1, bg="#000000")
        self.image_label.pack(fill="both", expand=True, padx=15, pady=15)

        # Column 2: Connection settings (Right, small panel)
        col2 = ttk.Frame(main_container, style="Panel.TFrame", width=340)
        col2.pack(side="left", fill="both", expand=False, padx=5)

        side_title = ttk.Label(col2, text="הגדרות וחיבור", style="PanelTitle.TLabel")
        side_title.pack(anchor="ne", padx=15, pady=10)

        # Connection status label
        self.status_box_lbl = ttk.Label(col2, text="מזהה SDK...", style="PanelSec.TLabel", font=("Calibri", 12, "bold"))
        self.status_box_lbl.pack(anchor="ne", padx=15, pady=10)

    def update_sdk_status(self):
        if METAVISION_AVAILABLE:
            self.status_box_lbl.config(text="סטטוס: ה-SDK זוהה בהצלחה! מוכן לחיבור.", foreground="#30d158")
        else:
            self.status_box_lbl.config(text="סטטוס: ה-SDK לא מותקן / לא זוהה.", foreground="#ff453a")

    def connect_to_physical_camera(self):
        """Attempts to dynamically connect to a real physical USB event camera."""
        if not METAVISION_AVAILABLE:
            messagebox.showerror(
                "שגיאה בחיבור",
                "ספריית Metavision SDK אינה מותקנת בסביבת פייתון זו.\n"
                "ודא שביצעת התקנה תקינה של ה-SDK או הפנה אליה באופן ידני."
            )
            return

        # Disable active existing connections safely
        self.running_live = False
        if self.slicer_instance:
            self.slicer_instance = None

        try:
            # Instantiate camera using EventsIterator on 1ms slice time base
            self.slicer_instance = EventsIterator(input_path="", delta_t=1000)
            device = self.slicer_instance.reader.device
            if not device:
                messagebox.showwarning(
                    "לא נמצאה מצלמה",
                    "לא זוהתה מצלמת אירועים של Prophesee מחוברת ב-USB במערכת.\n"
                    "ודא שהמצלמה מחוברת היטב, דולקת, ונסה שוב."
                )
                self.slicer_instance = None
                return

            sensor_height, sensor_width = self.slicer_instance.get_size()

            self.running_live = True
            self.status_box_lbl.config(text="סטטוס: מצלמה פיזית מחוברת ומזרימה וידאו חי! 🎥", foreground="#30d158")

            # Start real camera frame polling worker
            self.camera_thread = threading.Thread(
                target=self.live_camera_worker,
                args=(sensor_width, sensor_height),
                daemon=True
            )
            self.camera_thread.start()

            messagebox.showinfo("חיבור הצליח", "מצלמת Prophesee EVK4 חוברה בהצלחה!")

        except Exception as e:
            messagebox.showerror("שגיאה בחיבור לחומרה", f"נכשלה פתיחת ההתקן הפיזי:\n{e}")

    def live_camera_worker(self, width, height):
        """Worker thread that continuously gets event blocks from the EventsIterator."""
        if not self.slicer_instance:
            return

        # Draw frame using user's white and gray accumulation logic
        display_frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames_to_accumulate = 30 # Accumulate 30 cycles of 1ms before presenting
        frame_counter = 0

        try:
            for evs in self.slicer_instance:
                if not self.running_live:
                    break

                # 1. Accumulate events
                if evs.size > 0:
                    display_frame[evs['y'], evs['x']] = np.where(evs['p'][:, None] == 1, [255, 255, 255], [100, 100, 100])

                frame_counter += 1
                if frame_counter >= frames_to_accumulate:
                    # Thread-safe dispatch frame to UI loop
                    with self.lock:
                        self.shared_display_frame = display_frame.copy()

                    # Clear canvas for the next block to prevent smearing
                    display_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    frame_counter = 0

        except Exception as e:
            print(f"Exception in Metavision SDK thread: {e}")

    def update_loop(self):
        """Continuously pulls reconstructed frames from shared buffer and draws them on Tkinter canvas."""
        frame = None
        with self.lock:
            if self.shared_display_frame is not None:
                frame = self.shared_display_frame
                self.shared_display_frame = None

        if frame is not None:
            # Resize image cleanly and update label
            img_resized = cv2.resize(frame, (620, 440))

            # Convert OpenCV to PhotoImage
            _, buffer = cv2.imencode('.png', img_resized)
            self.tk_image = tk.PhotoImage(data=buffer.tobytes())
            self.image_label.config(image=self.tk_image)

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
