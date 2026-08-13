import os
import unittest

class TestEventRecorderApp(unittest.TestCase):
    def test_imports(self):
        # Verify basic imports work
        import numpy as np
        import cv2
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
