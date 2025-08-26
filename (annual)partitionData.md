```python
def get_list_files(f, Y):
    Flist = []
    df_files = pd.read_json(f)
    for i in range(len(df_files)):
        x = str(df_files.loc[i, 'data']['period'])
        if x.startswith(str(Y)):
            Flist.append('/project/bi_dpi/data/UN_Comtrade/bulk/'+df_files.loc[i, 'data']['rowKey']+'.arrow')
        # x = df_files.loc[i, 'data']['refYear']
        # if x == Y:
        #     Flist.append('/project/bi_dpi/data/UN_Comtrade/bulk/'+df_files.loc[i, 'data']['rowKey']+'.feather')
    return Flist
```
