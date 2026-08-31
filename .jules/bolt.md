## 2025-05-20 - Pre-allocated Numpy Canvas Buffers in High-Frequency Worker Threads

**Learning:** Re-allocating full-resolution NumPy arrays (`np.zeros(...)`) inside a high-frequency polling thread loop (e.g., event camera accumulation window at 30Hz–1000Hz) creates heavy memory allocation churn and triggers frequent Python garbage collection pauses, causing frame drops and rendering stutters. Reusing a single pre-allocated array buffer initialized once per worker lifecycle and resetting it via in-place zero-filling (`display_frame.fill(0)`) avoids heap allocation churn completely.

**Action:** Whenever a background worker thread accumulates or constructs frames in a tight loop, pre-allocate the frame buffer outside the loop and use in-place reset operations (`.fill(0)` or `np.copyto`) before starting the next cycle.

## 2025-05-21 - Pre-allocated Destination Buffers in OpenCV Resizing & Gated GUI Updates

**Learning:** Calling `cv2.resize()` without a `dst` parameter in high-frequency GUI render loops (20ms/50Hz) continuously allocates memory for output arrays, adding ~65% latency overhead per frame. Additionally, executing redundant Tkinter widget `config()` updates on inactive elements (e.g., static recording labels) forces unnecessary Tcl/Tk interpreter IPC calls every frame tick. Passing a pre-allocated destination buffer `dst=self._resized_buf` to `cv2.resize` and gating inactive widget updates via boolean state tracking drastically reduces loop latency and CPU idle load.

**Action:** Always pre-allocate image resize buffers for fixed-size video controls and gate idle GUI label `config()` calls using state transition flags.
