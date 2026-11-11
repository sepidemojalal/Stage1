"""
Average and Median fusion of the four methods' test predictions (Stage 1).

For each vehicle and each variant (User-specific, CDT Full, CDT Limited), the
predicted accelerations of Helly, IDM, LSTM and Transformer for the same target
row are combined:
    Average fusion = mean of the four predictions
    Median fusion  = median of the four predictions (mean of the middle two)
Test data only. MAE and MSE are computed against the actual acceleration.

Inputs: the per-vehicle prediction files of the four methods (settings.BASE_METHODS).
Usage:
    python fusion.py
"""
import argparse

import numpy as np
import pandas as pd

import common as C
import settings as S


def main():
    p = argparse.ArgumentParser(description="Average and median fusion of the four methods")
    p.add_argument("--vehicles", type=int, nargs="+", default=None)
    args = p.parse_args()
    vehicles = S.VEHICLES if args.vehicles is None else args.vehicles

    metric_rows = []
    for v in vehicles:
        frames = {m: C.read_predictions(folder, m, v) for m, folder in S.BASE_METHODS.items()}
        ref = frames["Helly"]
        for m, df in frames.items():
            C.check_same_targets(df["idx"].to_numpy(), ref["idx"].to_numpy())
            if not np.allclose(df["a1_true"], ref["a1_true"], atol=1e-4):
                raise ValueError(f"Vehicle {v}: a1_true of {m} differs from Helly")
        y_true = ref["a1_true"].to_numpy(float)

        for name, combine in (("AverageFusion", np.mean), ("MedianFusion", np.median)):
            preds = {}
            for variant, col in S.PRED_COLUMN.items():
                stack = np.column_stack([frames[m][col].to_numpy(float) for m in S.BASE_METHODS])
                preds[variant] = combine(stack, axis=1)
            C.save_predictions(S.FUSION_METHODS[name], name, v, ref["idx"], y_true, preds)
            metric_rows += C.metrics_table(name, v, y_true, preds)

    out = pd.DataFrame(metric_rows)
    for name in S.FUSION_METHODS:
        out[out.method == name].to_csv(S.OUTPUT_DIR / name / "metrics.csv", index=False)
    print(f"Fusion: wrote {', '.join(str(f) for f in S.FUSION_METHODS.values())}")


if __name__ == "__main__":
    main()
