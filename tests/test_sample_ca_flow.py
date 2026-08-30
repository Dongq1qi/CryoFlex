import csv
import sys
import tempfile
import unittest
from pathlib import Path

import mrcfile
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sample_ca_flow import sample_ca_flow_to_csv


def pdb_atom_line(serial, atom_name, resname, chain_id, resseq, x, y, z):
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id:1s}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}           C\n"
    )


class SampleCaFlowTest(unittest.TestCase):
    def test_samples_nearest_flow_at_ca_positions_and_reorders_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            flow = np.zeros((3, 4, 5, 3), dtype=np.float32)
            flow[1, 2, 3] = [10.0, 20.0, 30.0]  # dz, dy, dx
            flow_path = tmp_path / "flow.npy"
            np.save(flow_path, flow)

            mrc_path = tmp_path / "ref.mrc"
            with mrcfile.new(mrc_path, overwrite=True) as mrc:
                mrc.set_data(np.ones((3, 4, 5), dtype=np.float32))
                mrc.voxel_size = 1.0

            pdb_path = tmp_path / "model.pdb"
            pdb_path.write_text(
                pdb_atom_line(1, "CA", "ALA", "A", 7, 3.0, 2.0, 1.0)
                + pdb_atom_line(2, "CB", "ALA", "A", 7, 4.0, 2.0, 1.0)
                + "END\n",
                encoding="utf-8",
            )
            out_csv = tmp_path / "ca_flow.csv"

            sample_ca_flow_to_csv(
                flow_npy=str(flow_path),
                flow_mrc_prefix=None,
                reference_mrc=str(mrc_path),
                pdb_path=str(pdb_path),
                output_csv=str(out_csv),
                method="nearest",
            )

            with out_csv.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["atom_name"], "CA")
            self.assertEqual(row["chain_id"], "A")
            self.assertEqual(int(row["resseq"]), 7)
            self.assertAlmostEqual(float(row["flow_dz"]), 10.0)
            self.assertAlmostEqual(float(row["flow_dy"]), 20.0)
            self.assertAlmostEqual(float(row["flow_dx"]), 30.0)
            self.assertAlmostEqual(float(row["flow_x"]), 30.0)
            self.assertAlmostEqual(float(row["flow_y"]), 20.0)
            self.assertAlmostEqual(float(row["flow_z"]), 10.0)
            self.assertAlmostEqual(float(row["flow_mag"]), float(np.sqrt(1400.0)), places=5)


if __name__ == "__main__":
    unittest.main()
