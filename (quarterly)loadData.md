```python
import polars as pl
import os

# ── settings ──
FOLDER      = "out"
START_YEAR  = 2020
END_YEAR    = 2025          # only up to Q1 for 2025
FILENAME_TMPL = "q_transformed_{year}Q{q}.txt"

# set typing for data in dataframe
READ_KWARGS = dict(
    separator="\t",
    infer_schema_length=1_000,
    schema_overrides={
        "importer":      pl.Utf8,
        "exporter":      pl.Utf8,
        "cmdCode":       pl.Int64,
        "qtyUnitCode":   pl.Int64,
        "qty":           pl.Float64,
        "primaryValue":  pl.Float64,
    },
    ignore_errors=True,
)

# ── load quarter names into a list for quarters we're observing (format 2020Q1 etc.) ──
quarter_tags = []
for year in range(START_YEAR, END_YEAR + 1):
    for q in range(1, 5):
        if year == 2025 and q > 1:   # stop after 2025Q1
            break
        quarter_tags.append(f"{year}Q{q}")

# ── load existing files into a dict keyed by "YYYYQ#" ──
dfs_q = {}
for tag in quarter_tags:
    in_path = os.path.join(FOLDER, FILENAME_TMPL.format(year=tag[:4], q=tag[-1]))
    if os.path.exists(in_path):
        dfs_q[tag] = pl.read_csv(in_path, **READ_KWARGS)
    else:
        print(f"[WARN] {in_path} not found – skipping.")

print(f"Loaded {len(dfs_q)} quarterly DataFrames:",
      ", ".join(sorted(dfs_q.keys())))

# ── optional: create globals like df_2020Q1, df_2020Q2, … ────────────────
for tag, df in dfs_q.items():
    globals()[f"df_{tag}"] = df

# ── optional: single concatenated DataFrame of all quarters ──────────────
if dfs_q:
    df_all_quarters = pl.concat(dfs_q.values(), how="vertical_relaxed")
    print("Combined df_all_quarters rows:", df_all_quarters.height)
else:
    df_all_quarters = None

# ── optional: dictionary grouped by year (list of quarters per year) ─────
dfs_by_year = {}
for tag, df in dfs_q.items():
    yr = tag[:4]
    dfs_by_year.setdefault(yr, []).append(df)
```
