"""
Write the single CSV the PatchTST repo reads (main.py --data ...), from the
per-vehicle training and test files. Data values are copied unchanged.

Columns: vehicle_id, vehicle_index, idx, velocity, delta_x, delta_v, acceleration
350 rows per vehicle: the 248 training rows followed by the 102 test rows.
The extra idx column lets the patched repo save which row each prediction is for.

Usage:
    python prepare_patchtst_data.py [output.csv]
    (default: PatchTST-Vehicle-Acceleration/data/ngsim_30_vehicles.csv)
"""
import sys
from pathlib import Path

import pandas as pd

import common as C


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("PatchTST-Vehicle-Acceleration/data/ngsim_30_vehicles.csv")
    frames = []
    for v, (train, test) in C.load_all().items():
        rows = pd.concat([train, test], ignore_index=True)
        frames.append(pd.DataFrame({
            "vehicle_id": v,
            "vehicle_index": range(len(rows)),
            "idx": rows["idx"],
            "velocity": rows["v1"],
            "delta_x": rows["delta_x"],
            "delta_v": rows["delta_v"],
            "acceleration": rows["a1"],
        }))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
