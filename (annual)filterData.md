```python
with open(output_file_path, 'w') as file:
    # Write the header (optional)
    file.write('reporterCode partnerCode cmdCode qtyUnitCode qty primaryValue\n')
    
    # Iterate through the list of .arrow files
    for arrow_file in Flist:
        # Read the Arrow file as a Polars DataFrame
        table = ipc.open_file(arrow_file).read_all()
        df = pl.DataFrame(table.to_pandas())  # Convert PyArrow Table to Polars
        
        # Process and filter the required columns
        df = (
            df
            .select([
                pl.col("reporterCode"),
                pl.col("partnerCode"),
                pl.col("cmdCode")
                pl.col("qtyUnitCode"),
                pl.col("qty"),
                pl.col("primaryValue"),
                pl.col("flowCode")
            ])
        )
        
        # Write the DataFrame to the output file
        for row in df.iter_rows():  # Efficient row iteration
            file.write(' '.join(map(str, row)) + '\n')
```
