#!/usr/bin/env python3
"""Sample voxel-level CryoFlex flow at C-alpha positions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mrcfile
import numpy as np


@dataclass
class CaAtom:
    model_id: int
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atom_name: str
    x: float
    y: float
    z: float


def read_mrc_voxel_and_origin_xyz(mrc_path: str) -> tuple[np.ndarray, np.ndarray]:
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        header = mrc.header
        nx, ny, nz = int(header.nx), int(header.ny), int(header.nz)
        vx = float(header.cella.x) / max(nx, 1)
        vy = float(header.cella.y) / max(ny, 1)
        vz = float(header.cella.z) / max(nz, 1)
        if min(vx, vy, vz) <= 0:
            vx, vy, vz = float(mrc.voxel_size.x), float(mrc.voxel_size.y), float(mrc.voxel_size.z)

        origin = np.array(
            [float(header.origin.x), float(header.origin.y), float(header.origin.z)],
            dtype=np.float64,
        )
        nstart = np.array([int(header.nxstart), int(header.nystart), int(header.nzstart)], dtype=np.float64)
        voxel = np.array([vx, vy, vz], dtype=np.float64)
        if np.allclose(origin, 0.0) and np.any(nstart != 0):
            origin = nstart * voxel
    return voxel, origin


def load_flow(flow_npy: Optional[str], flow_mrc_prefix: Optional[str]) -> np.ndarray:
    if flow_npy:
        flow = np.load(flow_npy)
    elif flow_mrc_prefix:
        components = []
        for name in ("z", "y", "x"):
            path = f"{flow_mrc_prefix}_{name}.mrc"
            with mrcfile.open(path, permissive=True) as mrc:
                components.append(np.asarray(mrc.data, dtype=np.float32))
        flow = np.stack(components, axis=-1)
    else:
        raise ValueError("需要提供 flow_npy 或 flow_mrc_prefix")

    if flow.ndim != 4 or flow.shape[-1] != 3:
        raise ValueError("flow 必须是 shape=(nz, ny, nx, 3)，分量顺序为 (dz, dy, dx)")
    return np.asarray(flow, dtype=np.float32)


def parse_ca_atoms(pdb_path: str, chains: Optional[set[str]] = None) -> list[CaAtom]:
    atoms: list[CaAtom] = []
    model_id = 0
    saw_model = False
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record == "MODEL":
                saw_model = True
                try:
                    model_id = int(line[10:14].strip())
                except ValueError:
                    model_id += 1
                continue
            if record == "ENDMDL" and saw_model:
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21].strip() or "_"
            if chains is not None and chain_id not in chains:
                continue
            try:
                atoms.append(
                    CaAtom(
                        model_id=model_id,
                        chain_id=chain_id,
                        resseq=int(line[22:26]),
                        icode=line[26].strip(),
                        resname=line[17:20].strip(),
                        atom_name=atom_name,
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                    )
                )
            except ValueError:
                continue
    return atoms


def world_xyz_to_index_xyz(coord_xyz: np.ndarray, voxel_xyz: np.ndarray, origin_xyz: np.ndarray) -> np.ndarray:
    return (coord_xyz - origin_xyz) / voxel_xyz


def nearest_sample_flow(flow: np.ndarray, idx_xyz: np.ndarray) -> np.ndarray:
    x, y, z = idx_xyz
    xi, yi, zi = int(np.round(x)), int(np.round(y)), int(np.round(z))
    nz, ny, nx, _ = flow.shape
    if xi < 0 or yi < 0 or zi < 0 or xi >= nx or yi >= ny or zi >= nz:
        return np.zeros(3, dtype=np.float32)
    return flow[zi, yi, xi].astype(np.float32)


def trilinear_sample_scalar(volume: np.ndarray, idx_xyz: np.ndarray) -> float:
    x, y, z = idx_xyz
    nx, ny, nz = volume.shape[2], volume.shape[1], volume.shape[0]
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    if x0 < 0 or y0 < 0 or z0 < 0 or x1 >= nx or y1 >= ny or z1 >= nz:
        return 0.0

    xd, yd, zd = x - x0, y - y0, z - z0
    c000 = volume[z0, y0, x0]
    c100 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c110 = volume[z0, y1, x1]
    c001 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c011 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]
    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return float(c0 * (1 - zd) + c1 * zd)


def trilinear_sample_flow(flow: np.ndarray, idx_xyz: np.ndarray) -> np.ndarray:
    return np.array(
        [trilinear_sample_scalar(flow[..., i], idx_xyz) for i in range(3)],
        dtype=np.float32,
    )


def sample_flow(flow: np.ndarray, idx_xyz: np.ndarray, method: str) -> np.ndarray:
    if method == "nearest":
        return nearest_sample_flow(flow, idx_xyz)
    if method == "trilinear":
        return trilinear_sample_flow(flow, idx_xyz)
    raise ValueError(f"未知采样方法: {method}")


def sample_ca_flow_to_csv(
    flow_npy: Optional[str],
    flow_mrc_prefix: Optional[str],
    reference_mrc: str,
    pdb_path: str,
    output_csv: str,
    method: str = "trilinear",
    chains: Optional[Iterable[str]] = None,
) -> None:
    flow = load_flow(flow_npy, flow_mrc_prefix)
    voxel_xyz, origin_xyz = read_mrc_voxel_and_origin_xyz(reference_mrc)
    chain_set = None if chains is None else {c for c in chains if c}
    atoms = parse_ca_atoms(pdb_path, chain_set)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "chain_id",
        "resseq",
        "icode",
        "resname",
        "atom_name",
        "x",
        "y",
        "z",
        "idx_x",
        "idx_y",
        "idx_z",
        "flow_dz",
        "flow_dy",
        "flow_dx",
        "flow_x",
        "flow_y",
        "flow_z",
        "flow_mag",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for atom in atoms:
            coord = np.array([atom.x, atom.y, atom.z], dtype=np.float64)
            idx_xyz = world_xyz_to_index_xyz(coord, voxel_xyz, origin_xyz)
            flow_dzdyx = sample_flow(flow, idx_xyz, method)
            dz, dy, dx = [float(v) for v in flow_dzdyx]
            writer.writerow(
                {
                    "model_id": atom.model_id,
                    "chain_id": atom.chain_id,
                    "resseq": atom.resseq,
                    "icode": atom.icode,
                    "resname": atom.resname,
                    "atom_name": atom.atom_name,
                    "x": atom.x,
                    "y": atom.y,
                    "z": atom.z,
                    "idx_x": float(idx_xyz[0]),
                    "idx_y": float(idx_xyz[1]),
                    "idx_z": float(idx_xyz[2]),
                    "flow_dz": dz,
                    "flow_dy": dy,
                    "flow_dx": dx,
                    "flow_x": dx,
                    "flow_y": dy,
                    "flow_z": dz,
                    "flow_mag": float(np.linalg.norm(flow_dzdyx)),
                }
            )
    print(f"[OK] sampled {len(atoms)} C-alpha atoms -> {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample CryoFlex voxel flow at C-alpha atom positions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--flow_npy", help="flow.npy, shape=(nz,ny,nx,3), components=(dz,dy,dx)")
    group.add_argument("--flow_mrc_prefix", help="Prefix for flow_z.mrc/flow_y.mrc/flow_x.mrc")
    parser.add_argument("--reference_mrc", required=True, help="Reference MRC used to define grid origin and voxel size")
    parser.add_argument("--pdb", required=True, help="PDB aligned to the reference MRC")
    parser.add_argument("--output_csv", required=True, help="Output CSV with C-alpha sampled flow")
    parser.add_argument("--method", choices=["nearest", "trilinear"], default="trilinear")
    parser.add_argument("--chains", default=None, help="Optional comma-separated chain IDs to keep, e.g. A,B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chains = None if args.chains is None else [c.strip() for c in args.chains.split(",")]
    sample_ca_flow_to_csv(
        flow_npy=args.flow_npy,
        flow_mrc_prefix=args.flow_mrc_prefix,
        reference_mrc=args.reference_mrc,
        pdb_path=args.pdb,
        output_csv=args.output_csv,
        method=args.method,
        chains=chains,
    )


if __name__ == "__main__":
    main()
