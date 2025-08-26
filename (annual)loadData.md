```python
import polars as pl
import os, itertools

# ── settings ───────────────────────────────────────────────────────────────
FOLDER = "out"                       # where the transformed files live
YEARS  = range(2017, 2025)           # 2017–2024 inclusive
TEMPLATE = "flow_transformed_{year}.txt"
READ_KWARGS = dict(separator="\t", infer_schema_length=1000)

# ── load existing files into a dict of DataFrames ─────────────────────────
dfs = {
    year: pl.read_csv(os.path.join(FOLDER, TEMPLATE.format(year=year)),
                      **READ_KWARGS)
    for year in YEARS
    if os.path.exists(os.path.join(FOLDER, TEMPLATE.format(year=year)))
}

# ── (optional) unpack into standalone variables like df_2017, df_2018, … ──
for year, df in dfs.items():
    globals()[f"df_{year}"] = df

print(f"Loaded {len(dfs)} DataFrames:", ", ".join(map(str, dfs.keys())))
```
