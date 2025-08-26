```python
import os, pathlib
import pandas as pd
import polars as pl
from pathlib import Path
from pyarrow import ipc  # for reading .arrow files with pyarrow

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR      = "out"  # all intermediate & final files live here
META_JSON     = "/project/bi_dpi/data/UN_Comtrade/bulk/meta.json"
PROJECT_ROOT  = Path("/project/bi_dpi/data/UN_Comtrade")

os.makedirs(DATA_DIR, exist_ok=True)

# flow direction sets (match your previous logic)
IMPORT_CODES = {"FM", "M", "MIP", "MOP", "RM"}
EXPORT_CODES = {"DX", "RX", "X", "XIP", "XOP"}

# print all Polars rows by default (handy for debugging)
pl.Config.set_tbl_rows(-1)

# ──────────────────────────────────────────────────────────────────────────────
# LOOKUP: numeric UN codes → ISO-3
# ──────────────────────────────────────────────────────────────────────────────
_partner_df  = pl.from_pandas(pd.read_feather(PROJECT_ROOT / "partnerAreas.arrow"))
_reporter_df = pl.from_pandas(pd.read_feather(PROJECT_ROOT / "Reporters.arrow"))

_CODE_TO_A3 = {
    **dict(zip(_partner_df["PartnerCode"],  _partner_df["PartnerCodeIsoAlpha3"])),
    **dict(zip(_reporter_df["reporterCode"], _reporter_df["reporterCodeIsoAlpha3"]))
}
code_map = pl.DataFrame({"num": list(_CODE_TO_A3.keys()),
                         "iso": list(_CODE_TO_A3.values())})

# ──────────────────────────────────────────────────────────────────────────────
# partitionData: from meta.json pick all .arrow file paths for a given YEAR
# (this is your function, unchanged except path is configurable)
# ──────────────────────────────────────────────────────────────────────────────
def get_list_files(meta_json_path: str, Y: int):
    Flist = []
    df_files = pd.read_json(meta_json_path)
    for i in range(len(df_files)):
        x = str(df_files.loc[i, 'data']['period'])
        if x.startswith(str(Y)):
            Flist.append('/project/bi_dpi/data/UN_Comtrade/bulk/' +
                         df_files.loc[i, 'data']['rowKey'] + '.arrow')
        # Alternatively, match by refYear:
        # if df_files.loc[i, 'data']['refYear'] == Y:
        #     Flist.append('/project/bi_dpi/data/UN_Comtrade/bulk/' +
        #                  df_files.loc[i, 'data']['rowKey'] + '.feather')
    return Flist

# ──────────────────────────────────────────────────────────────────────────────
# filterData: read all Arrow files → write space-separated x_edges{Y}.txt
# Columns: reporterCode partnerCode cmdCode qtyUnitCode qty primaryValue flowCode
# ──────────────────────────────────────────────────────────────────────────────
def write_x_edges_for_year(year: int):
    Flist = get_list_files(META_JSON, year)
    out_file = os.path.join(DATA_DIR, f"x_edges{year}.txt")
    if not Flist:
        print(f"[WARN] No Arrow files found in meta.json for {year} — skipping x_edges.")
        return None

    print(f"→ building {out_file} from {len(Flist)} Arrow file(s)")
    with open(out_file, "w") as file:
        # header row (space-separated)
        file.write("reporterCode partnerCode cmdCode qtyUnitCode qty primaryValue flowCode\n")

        for arrow_file in Flist:
            try:
                table = ipc.open_file(arrow_file).read_all()
            except Exception as e:
                print(f"[WARN] failed to read {arrow_file}: {e}")
                continue

            df = pl.DataFrame(table.to_pandas())  # to Polars

            # keep only needed columns; missing columns become Null and will stringify as "null"
            df = df.select([
                pl.col("reporterCode"),
                pl.col("partnerCode"),
                pl.col("cmdCode"),
                pl.col("qtyUnitCode"),
                pl.col("qty"),
                pl.col("primaryValue"),
                pl.col("flowCode")
            ])

            # stream rows as space-separated text
            for row in df.iter_rows():
                file.write(" ".join(map(str, row)) + "\n")

    return out_file

# ──────────────────────────────────────────────────────────────────────────────
# condenseData: read x_edges{Y}.txt → group/aggregate → write z_edges{Y}.txt
# NOTE: we PRESERVE flowCode by grouping on it, so direction is not lost.
# ──────────────────────────────────────────────────────────────────────────────
def condense_to_z_edges(year: int):
    in_file  = os.path.join(DATA_DIR, f"x_edges{year}.txt")
    out_file = os.path.join(DATA_DIR, f"z_edges{year}.txt")

    if not os.path.exists(in_file):
        print(f"[WARN] {in_file} not found — skipping z_edges.")
        return None

    # read space-separated file
    df = pl.read_csv(
        in_file,
        separator=" ",
        infer_schema_length=10_000,
        schema_overrides={
            "reporterCode": pl.Int64,
            "partnerCode":  pl.Int64,
            "cmdCode":      pl.Utf8,    # may contain non-numeric; cast below
            "qtyUnitCode":  pl.Int64,
            "qty":          pl.Float64,
            "primaryValue": pl.Float64,
            "flowCode":     pl.Utf8,
        },
        ignore_errors=True,
    )

    # cast cmdCode to integer (drop rows that fail)
    df = (df.with_columns(pl.col("cmdCode").cast(pl.Int64, strict=False))
            .drop_nulls(subset=["cmdCode"]))

    # group BY direction as well (flowCode) so we can map importer/exporter later
    grouped = (
        df.group_by(["reporterCode", "partnerCode", "cmdCode", "flowCode"])
          .agg([
              pl.col("qtyUnitCode").mode().first().alias("qtyUnitCode"),
              pl.col("qty").sum().alias("qty"),
              pl.col("primaryValue").sum().alias("primaryValue"),
          ])
    )

    grouped.write_csv(out_file, separator="\t")
    print(f"   saved → {out_file}")
    return out_file

# ──────────────────────────────────────────────────────────────────────────────
# transform: read z_edges{Y}.txt → map to importer/exporter ISO → aggregate
#           → write flow_transformed_{Y}.txt (tab-separated)
# ──────────────────────────────────────────────────────────────────────────────
def transform_to_importer_exporter_iso(year: int):
    in_file  = os.path.join(DATA_DIR, f"z_edges{year}.txt")
    out_file = os.path.join(DATA_DIR, f"flow_transformed_{year}.txt")

    if not os.path.exists(in_file):
        print(f"[WARN] {in_file} not found — skipping transformed.")
        return None

    READ_KWARGS_TAB = dict(
        separator="\t",
        infer_schema_length=10_000,
        schema_overrides={
            "reporterCode": pl.Int64,
            "partnerCode":  pl.Int64,
            "cmdCode":      pl.Int64,
            "flowCode":     pl.Utf8,
            "qtyUnitCode":  pl.Int64,
            "qty":          pl.Float64,
            "primaryValue": pl.Float64,
        },
        ignore_errors=True,
    )

    df = (pl.read_csv(in_file, **READ_KWARGS_TAB)
            .filter(pl.col("flowCode").is_in(IMPORT_CODES | EXPORT_CODES)))

    # derive importer/exporter numeric codes from flow direction
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

    # numeric → ISO-3
    df = (
        df.join(code_map.rename({"num": "importerCode", "iso": "importer"}),
                on="importerCode", how="left")
          .join(code_map.rename({"num": "exporterCode", "iso": "exporter"}),
                on="exporterCode", how="left")
          .drop_nulls(subset=["importer", "exporter"])
    )

    # final aggregation at importer/exporter/commodity level
    transformed = (
        df.group_by(["importer", "exporter", "cmdCode"])
          .agg([
              pl.col("qtyUnitCode").mode().first().alias("qtyUnitCode"),
              pl.col("qty").sum().alias("qty"),
              pl.col("primaryValue").sum().alias("primaryValue"),
          ])
          .sort(["importer", "exporter", "cmdCode"])
    )

    transformed.write_csv(out_file, separator="\t")
    print(f"   saved → {out_file}")
    return out_file

# ──────────────────────────────────────────────────────────────────────────────
# DRIVER: run annual pipeline
# ──────────────────────────────────────────────────────────────────────────────
START_YEAR = 2017
END_YEAR   = 2024  # adjust as needed

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n===== YEAR {year} =====")
    x_path = write_x_edges_for_year(year)          # partitionData + filterData
    if x_path:
        z_path = condense_to_z_edges(year)         # condenseData
        if z_path:
            transform_to_importer_exporter_iso(year)  # final transformed file

print("✓ all years processed.")
```
