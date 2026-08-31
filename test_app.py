import os
import unittest

class TestEventRecorderApp(unittest.TestCase):
    def test_imports(self):
        # Verify basic imports work
        import numpy as np
        import cv2
        self.assertTrue(True)

    def test_ui_empty_state_and_scroll_bindings(self):
        import event_recorder_app
        try:
            app = event_recorder_app.EventRecorderApp()
        except Exception as e:
            self.skipTest(f"Tkinter display not available: {e}")

        try:
            # Check empty state text on initial launch
            empty_text = app.image_label.cget("text")
            self.assertIn("Camera Disconnected", empty_text)
            self.assertIn("Connect Camera", empty_text)

            # Check sidebar_canvas exists
            self.assertTrue(hasattr(app, "sidebar_canvas"))
            self.assertIsNotNone(app.sidebar_canvas)

            # Check new visualization variables and controls
            self.assertEqual(app.viz_mode.get(), "Accumulation")
            self.assertEqual(app.color_palette.get(), "Monochrome")

            # Check chart visibility variables
            self.assertTrue(app.show_timeline.get())
            self.assertTrue(app.show_ratio.get())

            # Test chart visibility toggling & dynamic grid refresh
            app.show_isi.set(False)
            app.refresh_graph_layout()

            # Test ROI clear
            app.clear_roi()
            self.assertFalse(app.roi_active)

            # Test separated accumulation sliders
            # 1. Video slider
            self.assertAlmostEqual(app.video_accumulation_time_ms.get(), 30.0)
            app.on_video_accumulation_slider_moved(150.0)
            self.assertAlmostEqual(app.video_accumulation_ms_val, 150.0)
            self.assertEqual(app.video_acc_val_lbl.cget("text"), "150.0 ms")

            # 2. Graph logarithmic slider and text entry sync
            self.assertAlmostEqual(app.graph_accumulation_us_val, 10000.0)
            # Simulate moving slider to log10(500) ≈ 2.69897
            import numpy as np
            app.on_graph_accumulation_slider_moved(np.log10(500.0))
            self.assertAlmostEqual(app.graph_accumulation_us_val, 500.0, places=1)
            self.assertEqual(app.graph_accumulation_entry_var.get(), "500")

            # Simulate typing value into text entry and pressing enter / focus out
            app.graph_accumulation_entry_var.set("0.5")
            app.on_graph_accumulation_entry_submitted()
            self.assertAlmostEqual(app.graph_accumulation_us_val, 0.5)
            self.assertAlmostEqual(app.graph_accumulation_log_var.get(), np.log10(0.5))

            # Simulate typing value out of bounds (> 100,000 µs)
            app.graph_accumulation_entry_var.set("200000")
            app.on_graph_accumulation_entry_submitted()
            self.assertAlmostEqual(app.graph_accumulation_us_val, 100000.0)
            self.assertEqual(app.graph_accumulation_entry_var.get(), "100000")

            # Simulate invalid string input
            app.graph_accumulation_entry_var.set("abc")
            app.on_graph_accumulation_entry_submitted()
            self.assertAlmostEqual(app.graph_accumulation_us_val, 100000.0)
            self.assertEqual(app.graph_accumulation_entry_var.get(), "100000")
        finally:
            app.destroy()

if __name__ == "__main__":
    unittest.main()
