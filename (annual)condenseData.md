```python
import polars as pl
import os

# Define output directory and ensure it exists
output_dir = "/project/bii_nssac/people/anil/DPI"
os.makedirs(output_dir, exist_ok=True)

# Load data
file_path = "out/x_edges{year}.txt"
df = pl.read_csv(
    file_path,
    separator=" ",  
    infer_schema_length=10000,  
    schema_overrides={
        "reporterCode": pl.Int64,
        "partnerCode": pl.Int64,
        "cmdCode": pl.Utf8,
        "qtyUnitCode": pl.Int64,
        "qty": pl.Float64,
        "primaryValue": pl.Float64,
    },
    ignore_errors=True
)

# Convert cmdCode to numeric, filtering out non-numeric values
df = df.with_columns(
    pl.col("cmdCode").cast(pl.Int64, strict=False)  
).drop_nulls(subset=["cmdCode"])

# ✅ Keep `qtyUnitCode` using mode (most frequent value)
df_grouped = df.group_by(["reporterCode", "partnerCode", "cmdCode"]).agg(
    [
        pl.col("qtyUnitCode").mode().first().alias("qtyUnitCode"),  # Get most frequent unit code
        pl.col("qty").sum(),
        pl.col("primaryValue").sum(),
    ]
)

# Save the condensed data
output_file = os.path.join(output_dir, "out/z_edges2000.txt")
df_grouped.write_csv(output_file, separator="\t")
```
