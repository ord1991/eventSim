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

    def test_ux_enhancements_tooltips_status_and_shortcuts(self):
        import event_recorder_app
        try:
            app = event_recorder_app.EventRecorderApp()
        except Exception as e:
            self.skipTest(f"Tkinter display not available: {e}")

        try:
            # Test ToolTip class initialization and show/hide
            tooltip = event_recorder_app.ToolTip(app.connect_btn, "Test tooltip text")
            self.assertEqual(tooltip.text, "Test tooltip text")
            tooltip.show_tooltip()
            self.assertIsNotNone(tooltip.tip_window)
            tooltip.hide_tooltip()
            self.assertIsNone(tooltip.tip_window)

            # Test non-blocking status message helper
            app.show_status_message("Test Status Message")
            self.assertEqual(app.status_msg_lbl.cget("text"), "Test Status Message")

            # Test Escape shortcut clears ROI and updates status bar directly
            app.roi_active = True
            app.roi_box = (0.1, 0.1, 0.5, 0.5)
            app.clear_roi()
            self.assertFalse(app.roi_active)
            self.assertIsNone(app.roi_box)
            self.assertEqual(app.status_msg_lbl.cget("text"), "ROI selection cleared.")

            # Test header recording badge existence
            self.assertTrue(hasattr(app, "header_rec_badge"))
            self.assertEqual(app.header_rec_badge.cget("text"), "🔴 REC")
        finally:
            app.destroy()

    def test_path_traversal_prevention(self):
        import event_recorder_app
        from pathlib import Path
        try:
            app = event_recorder_app.EventRecorderApp()
        except Exception as e:
            self.skipTest(f"Tkinter display not available: {e}")

        try:
            base_test_dir = str(Path("/tmp/evk_test_recordings").resolve())
            app.recording_dir.set(base_test_dir)

            # 1. Test normal safe filename
            app.recording_filename.get()
            app.recording_filename.set("normal_recording.raw")
            base_dir, full_path = app.get_safe_recording_path()
            self.assertIsNotNone(base_dir)
            self.assertIsNotNone(full_path)
            self.assertEqual(full_path, Path(base_test_dir) / "normal_recording.raw")

            # 2. Test relative path traversal injection (e.g., ../../etc/passwd)
            app.recording_filename.set("../../etc/passwd")
            base_dir, full_path = app.get_safe_recording_path()
            self.assertIsNotNone(base_dir)
            self.assertIsNotNone(full_path)
            self.assertEqual(full_path, Path(base_test_dir) / "passwd")
            self.assertTrue(full_path.is_relative_to(Path(base_test_dir)))

            # 3. Test absolute path traversal injection (e.g., /etc/shadow)
            app.recording_filename.set("/etc/shadow")
            base_dir, full_path = app.get_safe_recording_path()
            self.assertIsNotNone(base_dir)
            self.assertIsNotNone(full_path)
            self.assertEqual(full_path, Path(base_test_dir) / "shadow")
            self.assertTrue(full_path.is_relative_to(Path(base_test_dir)))

            # 4. Test invalid / empty filename
            app.recording_filename.set("    ")
            base_dir, full_path = app.get_safe_recording_path()
            self.assertIsNone(base_dir)
            self.assertIsNone(full_path)
        finally:
            app.destroy()

if __name__ == "__main__":
    unittest.main()
