import numpy as np
from pathlib import Path

from module_1 import map_correspondence_coords_to_full_points


ROOT = Path(__file__).resolve().parents[1]


def test_maps_correspondence_coordinates_to_nearest_full_points():
    source = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    )
    target = np.array(
        [[0.0, 5.0, 0.0], [10.0, 5.0, 0.0], [20.0, 5.0, 0.0]]
    )
    source_corr = np.array([[19.8, 0.0, 0.0], [0.2, 0.0, 0.0]])
    target_corr = np.array([[10.1, 5.0, 0.0], [19.9, 5.0, 0.0]])

    indices, distances = map_correspondence_coords_to_full_points(
        source,
        target,
        source_corr,
        target_corr,
    )

    np.testing.assert_array_equal(indices, [[2, 1], [0, 2]])
    np.testing.assert_allclose(distances, [[0.2, 0.1], [0.2, 0.1]])
    assert indices.dtype == np.int64
    assert indices.shape == distances.shape == (2, 2)


def test_direct_pipeline_consumers_use_full_point_correspondence_indices():
    script = (ROOT / "scripts" / "run_light_pipeline.sh").read_text()
    readme = (ROOT / "README.md").read_text()

    assert '--corr "${WORK_DIR}/corr_full_indices_7pqg_7pqq.npy"' in script
    assert "--corr /path/to/data/corr_full_indices_source_target.npy" in readme


if __name__ == "__main__":
    test_maps_correspondence_coordinates_to_nearest_full_points()
    test_direct_pipeline_consumers_use_full_point_correspondence_indices()
    print("MODULE1_CORRESPONDENCE_MAPPING_TEST_OK")
