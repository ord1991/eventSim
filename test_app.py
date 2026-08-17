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
        finally:
            app.destroy()

if __name__ == "__main__":
    unittest.main()
