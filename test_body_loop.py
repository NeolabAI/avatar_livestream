import unittest

import numpy as np

from utils.image import analyze_loop_frames, forward_loop_index


class BodyLoopTests(unittest.TestCase):
    def test_forward_loop_never_reverses(self):
        values = [forward_loop_index(10, i, 2, 6) for i in range(10)]
        self.assertEqual(values, [2, 3, 4, 5, 2, 3, 4, 5, 2, 3])

    def test_analyzer_chooses_stable_similar_endpoints(self):
        frames = []
        for i in range(60):
            value = 30 + (i % 20)
            if 20 <= i < 30:
                value = 220 if i % 2 else 0
            frames.append(np.full((32, 32, 3), value, dtype=np.uint8))

        profile = analyze_loop_frames(frames, fps=5, min_seconds=4, max_seconds=6)

        self.assertEqual(profile["mode"], "forward")
        self.assertGreaterEqual(profile["loop_frames"], 20)
        self.assertLessEqual(profile["loop_frames"], 30)
        self.assertTrue(profile["abrupt_transitions"])


if __name__ == "__main__":
    unittest.main()
