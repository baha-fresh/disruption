```python
import polars as pl
import pandas as pd
import os, pathlib

# ── folder with quarterly files ───────────────────────────────────────────
DATA_DIR = "out"  # both input and output live here
os.makedirs(DATA_DIR, exist_ok=True)

# ── load numeric‑code → ISO‑3 lookup tables once ─────────────────────────
_PROJECT_ROOT = pathlib.Path("/project/bi_dpi/data/UN_Comtrade")
_partner_df   = pl.from_pandas(pd.read_feather(_PROJECT_ROOT / "partnerAreas.arrow"))
_reporter_df  = pl.from_pandas(pd.read_feather(_PROJECT_ROOT / "Reporters.arrow"))

_CODE_TO_A3 = {**dict(zip(_partner_df["PartnerCode"],
                           _partner_df["PartnerCodeIsoAlpha3"])),
               **dict(zip(_reporter_df["reporterCode"],
                           _reporter_df["reporterCodeIsoAlpha3"]))}

code_map = pl.DataFrame({"num": list(_CODE_TO_A3.keys()),
                         "iso": list(_CODE_TO_A3.values())})

# ── flowCode buckets ─────────────────────────────────────────────────────
IMPORT_CODES = {"FM", "M", "MIP", "MOP", "RM"}
EXPORT_CODES = {"DX", "RX", "X", "XIP", "XOP"}

# ── read options for quarter‑agg files ───────────────────────────────────
READ_KWARGS = dict(
    separator="\t",
    infer_schema_length=1_000,
    schema_overrides={
        "reporterCode":  pl.Int64,
        "partnerCode":   pl.Int64,
        "cmdCode":       pl.Int64,
        "flowCode":      pl.Utf8,
        "qtyUnitCode":   pl.Int64,
        "qty":           pl.Float64,
        "primaryValue":  pl.Float64,
    },
    ignore_errors=True,
)

START_YEAR = 2020
END_YEAR   = 2025  # inclusive, but we’ll only take Q1 in 2025

for year in range(START_YEAR, END_YEAR + 1):
    for q in range(1, 5):               # Q1 .. Q4
        if year == 2025 and q > 1:      # stop after 2025Q1
            break

        tag      = f"{year}Q{q}"
        in_file  = os.path.join(DATA_DIR, f"q_agg_unmapped{tag}.txt")
        out_file = os.path.join(DATA_DIR, f"q_transformed_{tag}.txt")

        if not os.path.exists(in_file):
            print(f"[WARN] {in_file} not found — skipping.")
            continue

        print(f"→ transforming {in_file}")

        # ── load & filter recognised flow codes ──────────────────────────
        df = (
            pl.read_csv(in_file, **READ_KWARGS)
              .filter(pl.col("flowCode").is_in(IMPORT_CODES | EXPORT_CODES))
        )

        # ── figure out importer / exporter numeric codes ─────────────────
        df = df.with_columns([
            pl.when(pl.col("flowCode").is_in(IMPORT_CODES))
              .then(pl.col("reporterCode"))
              .otherwise(pl.col("partnerCode"))
              .alias("importerCode"),

            pl.when(pl.col("flowCode").is_in(IMPORT_CODES))
              .then(pl.col("partnerCode"))
              .otherwise(pl.col("reporterCode"))
              .alias("exporterCode"),
        ])

        # ── map numeric codes → ISO‑3 ────────────────────────────────────
        df = (
            df.join(code_map.rename({"num": "importerCode", "iso": "importer"}),
                    on="importerCode", how="left")
              .join(code_map.rename({"num": "exporterCode", "iso": "exporter"}),
                    on="exporterCode", how="left")
              .drop_nulls(subset=["importer", "exporter"])
        )

        # ── aggregate by importer / exporter / commodity ────────────────
        transformed = (
            df.group_by(["importer", "exporter", "cmdCode"])
              .agg([
                  pl.col("qtyUnitCode").mode().first().alias("qtyUnitCode"),
                  pl.col("qty").sum().alias("qty"),
                  pl.col("primaryValue").sum().alias("primaryValue"),
              ])
              .sort(["importer", "exporter", "cmdCode"])
        )

        # ── save ────────────────────────────────────────────────────────
        transformed.write_csv(out_file, separator="\t")
        print(f"   saved → {out_file}")

print("✓ all quarters processed.")
```
