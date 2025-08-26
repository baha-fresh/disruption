```python
import pyarrow.ipc as ipc
import polars as pl

def write_quarterly_files(files_by_quarter, output_dir):
    for quarter, file_list in files_by_quarter.items():
        output_file_path = f"{output_dir}/q_unmapped_edges2020{quarter}.txt"

        with open(output_file_path, 'w') as file:
            # Write header
            file.write('reporterCode partnerCode cmdCode flowCode qtyUnitCode qty primaryValue\n')

            for arrow_file in file_list:
                table = ipc.open_file(arrow_file).read_all()
                df = pl.DataFrame(table.to_pandas())

                df = df.select([
                    pl.col("reporterCode"),
                    pl.col("partnerCode"),
                    pl.col("cmdCode"),
                    pl.col("flowCode"),
                    pl.col("qtyUnitCode"),
                    pl.col("qty"),
                    pl.col("primaryValue")
                ])

                for row in df.iter_rows():
                    file.write(' '.join(map(str, row)) + '\n')

        print(f"✅ Finished writing: {output_file_path} ({len(file_list)} files)")
```
