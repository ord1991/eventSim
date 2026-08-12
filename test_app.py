import os
import unittest
import json
import tempfile
import numpy as np
from event_recorder_app import SimulatedCamera, EVK4_BIAS_DEFAULTS

class TestSimulatedCamera(unittest.TestCase):
    def setUp(self):
        self.camera = SimulatedCamera(width=640, height=480)

    def test_initial_state(self):
        self.assertFalse(self.camera.is_running)
        self.assertFalse(self.camera.is_recording)
        self.assertEqual(self.camera.width, 640)
        self.assertEqual(self.camera.height, 480)

        # Verify initial bias defaults are configured properly
        for name, info in EVK4_BIAS_DEFAULTS.items():
            self.assertEqual(self.camera.biases[name], info["value"])

    def test_start_stop(self):
        self.camera.start()
        self.assertTrue(self.camera.is_running)
        self.camera.stop()
        self.assertFalse(self.camera.is_running)

    def test_recording_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_record.raw")
            self.camera.start()
            self.camera.start_recording(file_path)
            self.assertTrue(self.camera.is_recording)
            self.assertEqual(self.camera.recording_file, file_path)

            # Generate events and verify recording increments
            events = self.camera.generate_events(0.030)
            self.assertGreater(len(events), 0)
            self.assertGreater(self.camera.recorded_events, 0)
            self.assertGreater(self.camera.recorded_bytes, 128)

            self.camera.stop_recording()
            self.assertFalse(self.camera.is_recording)
            self.assertIsNone(self.camera.recording_file)

    def test_update_biases(self):
        # Update specific biases and ensure values are set correctly
        self.camera.update_biases("bias_diff_on", 50)
        self.assertEqual(self.camera.biases["bias_diff_on"], 50)

        self.camera.update_biases("bias_diff_off", -15)
        self.assertEqual(self.camera.biases["bias_diff_off"], -15)

    def test_erc_and_trail_filter_configurations(self):
        self.assertFalse(self.camera.erc_enabled)
        self.assertFalse(self.camera.trail_filter_enabled)

        # Configure ERC
        self.camera.erc_enabled = True
        self.camera.erc_rate = 500
        self.assertTrue(self.camera.erc_enabled)
        self.assertEqual(self.camera.erc_rate, 500)

        # Configure Trail filter
        self.camera.trail_filter_enabled = True
        self.camera.trail_filter_threshold = 5000
        self.assertTrue(self.camera.trail_filter_enabled)
        self.assertEqual(self.camera.trail_filter_threshold, 5000)

    def test_generate_events_format(self):
        self.camera.start()
        events = self.camera.generate_events(0.030)

        # Check that events have the expected structured dtype fields
        self.assertTrue(isinstance(events, np.ndarray))
        self.assertIn('x', events.dtype.names)
        self.assertIn('y', events.dtype.names)
        self.assertIn('p', events.dtype.names)
        self.assertIn('t', events.dtype.names)

        # Verify coordinates are within range
        for ev in events:
            self.assertTrue(0 <= ev['x'] < 640)
            self.assertTrue(0 <= ev['y'] < 480)
            self.assertTrue(ev['p'] in (0, 1))

class TestAppMockConnection(unittest.TestCase):
    def test_mock_fallback_logic(self):
        # Verify app is safely fallback ready by inspecting components without rendering
        from event_recorder_app import SimulatedCamera, EVK4_BIAS_DEFAULTS
        mock_cam = SimulatedCamera()
        self.assertFalse(mock_cam.is_running)
        mock_cam.start()
        self.assertTrue(mock_cam.is_running)
        mock_cam.stop()
        self.assertFalse(mock_cam.is_running)

class TestRawFileDecoder(unittest.TestCase):
    def test_evt2_decoding(self):
        # Create a mock EVT2.0 binary raw data
        import struct
        from event_recorder_app import RawFileDecoder

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.raw")
            with open(file_path, "wb") as f:
                # Header
                f.write(b"% This is a mock header\n")
                # EVT_TIME_HIGH (type 8, value = 1) -> 0x80000001
                f.write(struct.pack("<I", 0x80000001))
                # CD_ON (type 1, ts_lsb = 10, x = 100, y = 200)
                # Word construction:
                # type (4 bits) = 1 -> 0x10000000
                # ts_lsb (6 bits) = 10 -> (10 << 22) = 0x02800000
                # x (11 bits) = 100 -> (100 << 11) = 0x00032000
                # y (11 bits) = 200 -> 0x000000C8
                # Combined = 0x128320C8
                f.write(struct.pack("<I", 0x128320C8))

            decoder = RawFileDecoder(file_path)
            events = decoder.read_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['x'], 100)
            self.assertEqual(events[0]['y'], 200)
            self.assertEqual(events[0]['p'], 1)
            self.assertEqual(events[0]['t'], (1 << 6) | 10)
            decoder.close()

if __name__ == "__main__":
    unittest.main()
