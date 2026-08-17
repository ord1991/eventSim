## 2025-05-20 - Pre-allocated Numpy Canvas Buffers in High-Frequency Worker Threads

**Learning:** Re-allocating full-resolution NumPy arrays (`np.zeros(...)`) inside a high-frequency polling thread loop (e.g., event camera accumulation window at 30Hz–1000Hz) creates heavy memory allocation churn and triggers frequent Python garbage collection pauses, causing frame drops and rendering stutters. Reusing a single pre-allocated array buffer initialized once per worker lifecycle and resetting it via in-place zero-filling (`display_frame.fill(0)`) avoids heap allocation churn completely.

**Action:** Whenever a background worker thread accumulates or constructs frames in a tight loop, pre-allocate the frame buffer outside the loop and use in-place reset operations (`.fill(0)` or `np.copyto`) before starting the next cycle.
