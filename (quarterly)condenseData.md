```python
import polars as pl
import os

DATA_DIR = "out"                        # where the quarter‑files live
os.makedirs(DATA_DIR, exist_ok=True)    # create if missing

# ── helper: common read options ───────────────────────────────────────────
READ_KWARGS = dict(
    separator=" ",                      # files are space‑separated
    infer_schema_length=10_000,
    schema_overrides={
        "reporterCode":  pl.Int64,
        "partnerCode":   pl.Int64,
        "cmdCode":       pl.Utf8,       # cast to Int64 after load
        "flowCode":      pl.Utf8,
        "qtyUnitCode":   pl.Int64,
        "qty":           pl.Float64,
        "primaryValue":  pl.Float64,
    },
    ignore_errors=True,
)

START_YEAR = 2020
END_YEAR   = 2025   # inclusive, but we’ll stop after 2025Q1

for year in range(START_YEAR, END_YEAR + 1):
    for q in range(1, 5):               # Q1 .. Q4
        # stop at 2025Q1
        if year == 2025 and q > 1:
            break

        tag      = f"{year}Q{q}"
        in_file  = os.path.join(DATA_DIR, f"q_unmapped_edges{tag}.txt")
        out_file = os.path.join(DATA_DIR, f"q_agg_unmapped{tag}.txt")

        if not os.path.exists(in_file):
            print(f"[WARN] {in_file} not found – skipping.")
            continue

        print(f"→ aggregating {in_file}")

        # ── load & clean ──────────────────────────────────────────────────
        df = pl.read_csv(in_file, **READ_KWARGS)
        df = (
            df.with_columns(pl.col("cmdCode").cast(pl.Int64, strict=False))
              .drop_nulls(subset=["cmdCode"])
        )

        # ── aggregate ────────────────────────────────────────────────────
        agg = (
            df.group_by(["reporterCode", "partnerCode", "cmdCode", "flowCode"])
              .agg([
                  pl.col("qtyUnitCode").mode().first().alias("qtyUnitCode"),
                  pl.col("qty").sum().alias("qty"),
                  pl.col("primaryValue").sum().alias("primaryValue"),
              ])
        )

        # ── save ─────────────────────────────────────────────────────────
        agg.write_csv(out_file, separator="\t")
        print(f"   saved → {out_file}")
```
