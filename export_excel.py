"""
One Excel workbook per vehicle with the estimated accelerations of all six
methods (Helly, IDM, LSTM, Transformer, Average fusion, Median fusion) for the
three variants, and their test-set MAE and MSE.

  Metrics sheet : MAE and MSE (live Excel formulas over the method sheets) and,
                  if bca.py has been run, the BCa 95% confidence interval of the MAE
  Method sheets : target idx, actual a1, estimated a1 for each variant
  Setup sheet   : the settings used to produce the predictions

Usage:
    python export_excel.py
"""
import argparse

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import common as C
import settings as S

SHEETS = {"Helly": "Helly", "IDM": "IDM", "LSTM": "LSTM", "Transformer": "Transformer",
          "AverageFusion": "Average Fusion", "MedianFusion": "Median Fusion"}
FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEAD = Font(name="Arial", size=10, bold=True, color="FFFFFF")
FILL = PatternFill("solid", start_color="1F4E78")
TITLE = Font(name="Arial", size=12, bold=True)
NOTE = Font(name="Arial", size=9, italic=True)


def header(ws, row, names, widths):
    for j, (name, w) in enumerate(zip(names, widths), 1):
        c = ws.cell(row=row, column=j, value=name)
        c.font, c.fill = HEAD, FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[row].height = 30


def put(ws, r, c, value, fmt=None, font=FONT):
    cell = ws.cell(row=r, column=c, value=value)
    cell.font = font
    if fmt:
        cell.number_format = fmt
    return cell


def setup_rows():
    return [
        ("Test targets", f"{S.N_TEST_TARGETS} per vehicle (test rows 11–102)"),
        ("Helly / IDM inputs", f"v1, delta_x, delta_v, a1 of row τ−{S.REACTION_STEPS} → a1 of row τ (T = 1 s)"),
        ("LSTM / Transformer inputs", f"rows τ−{S.WINDOW} … τ−1 → a1 of row τ (next row)"),
        ("User-specific", "trained on the vehicle's 248 training rows"),
        ("CDT Full", "trained on all 30 vehicles' training rows"),
        ("CDT Limited", f"{S.LIMITED_N} random training rows per vehicle (numpy default_rng({S.LIMITED_SEED}), "
                        "same rows for all methods)"),
        ("LSTM / Transformer windows", "built over the rows joined in vehicle order (PatchTST repo design)"),
        ("Validation", "none. Training uses the training files only; the test files are used only "
                       "for the final scores"),
        ("LSTM / Transformer training", f"{S.LSTM['epochs']} epochs, constant learning rate {S.LSTM['lr']}, "
                                        "final weights kept"),
        ("Training MAE / MSE", "on each model's own training data: User-specific on this vehicle, CDT Full on "
                               "all 30 vehicles, CDT Limited on its 270 rows (CDT values repeat in every file)"),
        ("Helly / IDM calibration", f"Metropolis–Hastings, {S.MCMC['n_chains']} chains, {S.MCMC['n_epochs']} epochs, "
                                    f"{S.MCMC['n_burn']} burn-in + {S.MCMC['n_samples']} kept per chain and epoch; "
                                    "posterior means"),
        ("IDM fixed values", f"v0 = {S.IDM_FIXED['v0']} m/s, s0 = {S.IDM_FIXED['s0']} m, δ = {S.IDM_FIXED['delta']}"),
        ("IDM Δv", "Δv_n = −delta_v" if S.IDM_DELTA_V_SIGN == -1 else "Δv_n = delta_v"),
        ("Fusion", "average and median of Helly, IDM, LSTM and Transformer for each target row"),
        ("BCa", f"B = {S.BCA['B']:,} resamples of the 92 rows, 95% two-sided, MAE"),
    ]


def main():
    p = argparse.ArgumentParser(description="Per-vehicle Excel workbooks")
    p.add_argument("--vehicles", type=int, nargs="+", default=None)
    args = p.parse_args()
    vehicles = S.VEHICLES if args.vehicles is None else args.vehicles
    folders = {**S.BASE_METHODS, **S.FUSION_METHODS}
    bca_file = S.OUTPUT_DIR / "BCa" / "bca_all_methods.csv"
    bca = pd.read_csv(bca_file) if bca_file.exists() else None
    train = pd.concat([pd.read_csv(f) for f in S.METRICS_FILES.values() if f.exists()],
                      ignore_index=True) if any(f.exists() for f in S.METRICS_FILES.values()) else None
    out_dir = S.OUTPUT_DIR / "Excel"
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in vehicles:
        wb = Workbook()
        ws_m = wb.active
        ws_m.title = "Metrics"
        n = S.N_TEST_TARGETS
        first, last = 5, 5 + n - 1

        for m, sheet in SHEETS.items():
            df = C.read_predictions(folders[m], m, v)
            ws = wb.create_sheet(sheet)
            put(ws, 1, 1, f"Vehicle {v} — {sheet}: estimated acceleration on the test set", font=TITLE)
            put(ws, 2, 1, "Units: m/s². Target idx = row being predicted.", font=NOTE)
            header(ws, 4, ["Prediction #", "Target idx", "Actual a1"] +
                   [f"Estimated a1 — {var}" for var in S.VARIANTS], [12, 12, 12, 24, 24, 24])
            for k, row in enumerate(df.itertuples(index=False)):
                r = first + k
                put(ws, r, 1, k + 1)
                put(ws, r, 2, int(row.idx))
                put(ws, r, 3, float(row.a1_true), "0.0000")
                for j, var in enumerate(S.VARIANTS):
                    put(ws, r, 4 + j, float(getattr(row, S.PRED_COLUMN[var])), "0.0000")
            ws.freeze_panes = "A5"

        put(ws_m, 1, 1, f"Vehicle {v} — test-set MAE and MSE ({n} predictions)", font=TITLE)
        put(ws_m, 2, 1, "Test MAE and MSE are formulas over the method sheets; the BCa interval comes from "
                        "bca.py. Training MAE and MSE: each model on its own training data (fusion: test only).",
            font=NOTE)
        header(ws_m, 4, ["Method", "Variant", "Test MAE (m/s²)", "Test MSE (m²/s⁴)",
                         "BCa 95% CI low", "BCa 95% CI high", "Training MAE (m/s²)", "Training MSE (m²/s⁴)"],
               [16, 16, 14, 14, 15, 15, 16, 16])
        r = 5
        for m, sheet in SHEETS.items():
            act = f"'{sheet}'!$C${first}:$C${last}"
            for j, var in enumerate(S.VARIANTS):
                col = get_column_letter(4 + j)
                est = f"'{sheet}'!${col}${first}:${col}${last}"
                put(ws_m, r, 1, sheet, font=BOLD if j == 0 else FONT)
                put(ws_m, r, 2, var)
                put(ws_m, r, 3, f"=SUMPRODUCT(ABS({est}-{act}))/COUNT({act})", "0.0000")
                put(ws_m, r, 4, f"=SUMPRODUCT(({est}-{act})^2)/COUNT({act})", "0.0000")
                if bca is not None:
                    hit = bca[(bca.method == m) & (bca.vehicle == v) & (bca.variant == var)]
                    if len(hit):
                        put(ws_m, r, 5, float(hit.ci_low.iloc[0]), "0.0000")
                        put(ws_m, r, 6, float(hit.ci_high.iloc[0]), "0.0000")
                if train is not None and m in S.METRICS_FILES:
                    hit = train[(train.method == m) & (train.vehicle == v) & (train.variant == var)]
                    if len(hit) and pd.notna(hit.train_MAE.iloc[0]):
                        put(ws_m, r, 7, float(hit.train_MAE.iloc[0]), "0.0000")
                        put(ws_m, r, 8, float(hit.train_MSE.iloc[0]), "0.0000")
                r += 1

        ws_s = wb.create_sheet("Setup")
        ws_s.column_dimensions["A"].width = 34
        ws_s.column_dimensions["B"].width = 100
        put(ws_s, 1, 1, "Settings used", font=TITLE)
        for k, (a, b) in enumerate(setup_rows(), 3):
            put(ws_s, k, 1, a, font=BOLD).alignment = Alignment(wrap_text=True, vertical="top")
            put(ws_s, k, 2, b).alignment = Alignment(wrap_text=True, vertical="top")

        wb.save(out_dir / f"Stage1_Vehicle_{v:02d}.xlsx")
    print(f"Excel: wrote {len(vehicles)} workbooks to {out_dir}")


if __name__ == "__main__":
    main()
