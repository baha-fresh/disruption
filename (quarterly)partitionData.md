python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def get_list_files_by_quarter(meta_path, year):
    files_by_quarter = { 'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [] }

    df_files = pd.read_json(meta_path)

    for i in range(len(df_files)):
        period = str(df_files.loc[i, 'data']['period'])  # e.g., '201810'
        if period.startswith(str(year)):
            month = int(period[-2:])   # Extract last 2 digits → month
            if 1 <= month <= 3:
                quarter = 'Q1'
            elif 4 <= month <= 6:
                quarter = 'Q2'
            elif 7 <= month <= 9:
                quarter = 'Q3'
            elif 10 <= month <= 12:
                quarter = 'Q4'
            else:
                continue  # Skip invalid months

            file_path = '/project/bi_dpi/data/UN_Comtrade/bulk/' + df_files.loc[i, 'data']['rowKey'] + '.arrow'
            files_by_quarter[quarter].append(file_path)

    return files_by_quarter
```
