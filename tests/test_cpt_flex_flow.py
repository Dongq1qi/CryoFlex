import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpt_flex import compute_weighted_flow_and_flex_grids


class WeightedFlowGridTest(unittest.TestCase):
    def test_single_point_produces_voxel_flow_and_legacy_flex(self):
        source = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
        target = np.array([[3.0, 4.0, 5.0]], dtype=np.float64)
        mrc = np.ones((5, 5, 5), dtype=np.float32)

        flex, flow = compute_weighted_flow_and_flex_grids(
            source_idx=source,
            target_idx=target,
            mrc_zyx=mrc,
            voxel_size_A=1.0,
            radius_vox=0.25,
            mrc_threshold=0.0,
            weighting_scheme="uniform",
        )

        self.assertEqual(flex.shape, (5, 5, 5))
        self.assertEqual(flow.shape, (5, 5, 5, 3))
        np.testing.assert_allclose(flow[2, 2, 2], [1.0, 2.0, 3.0], rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(flex[2, 2, 2]), float(np.sqrt(14.0)), places=6)
        self.assertEqual(float(flex[0, 0, 0]), 0.0)
        np.testing.assert_allclose(flow[0, 0, 0], [0.0, 0.0, 0.0], rtol=0, atol=0)

    def test_mask_is_applied_to_flex_and_flow(self):
        source = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
        target = np.array([[3.0, 4.0, 5.0]], dtype=np.float64)
        mrc = np.ones((5, 5, 5), dtype=np.float32)
        mrc[2, 2, 2] = 0.0

        flex, flow = compute_weighted_flow_and_flex_grids(
            source_idx=source,
            target_idx=target,
            mrc_zyx=mrc,
            voxel_size_A=1.0,
            radius_vox=0.25,
            mrc_threshold=0.0,
            weighting_scheme="uniform",
        )

        self.assertEqual(float(flex[2, 2, 2]), 0.0)
        np.testing.assert_allclose(flow[2, 2, 2], [0.0, 0.0, 0.0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
