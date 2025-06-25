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

    Loaded 8 DataFrames: 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024



```python
import polars as pl
import networkx as nx
import os
from typing import Iterable, Union, Optional

def network_flow_disruption(
    cmd_code: int,
    year: int,
    *,
    remove: Union[str, Iterable[str]],
    importer: Optional[str] = None,
    metric: str = "value",                # "value"│"qty"
    folder: str = "out",
    template: str = "flow_transformed_{yr}.txt",
) -> float:
    """
    % flow lost when *remove* (1 or many ISO-3 codes) disappear from the network.

    Works for a global network (importer=None) or an importer-centric network.
    """
    metric_col = {"value": "primaryValue", "qty": "qty"}[metric.lower()]
    rm_set = {remove} if isinstance(remove, str) else set(remove)

    # ── read & pre-filter rows ──────────────────────────────────────────────
    path = os.path.join(folder, template.format(yr=year))
    df = (
        pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=1000,
            columns=["importer", "exporter", "cmdCode", metric_col],
        )
        .filter(
            (pl.col("cmdCode") == cmd_code)
            & (pl.col("importer") != "W00")
            & (pl.col("exporter") != "W00")
        )
    )

    if importer:
        df = df.filter(pl.col("importer") == importer)

    if df.is_empty():
        raise ValueError(f"No trade rows for {cmd_code} in {year} (importer={importer})")

    # ── build graph & compute disruption exactly ───────────────────────────
    G = nx.DiGraph()
    for exp_, imp_, w in df.select(["exporter", "importer", metric_col]).iter_rows():
        G.add_edge(exp_, imp_, weight=float(w))

    baseline = G.size(weight="weight")

    G_removed = G.copy()
    G_removed.remove_nodes_from(rm_set)
    remaining = G_removed.size(weight="weight")

    return (baseline - remaining) / baseline * 100.0
```


```python
loss_world = network_flow_disruption(282200, 2017, remove={"COD", "ZAF"})
```


```python
print(f"Global flow loss: {loss_world:.2f}%")
```

    Global flow loss: 82.65%



```python
import polars as pl
import os
from typing import Iterable, Optional

def disruption_table(
    cmd_code: int,
    *,
    importer: Optional[str] = None,        # None = global
    k: Optional[int] = None,               # top-k disruptors to keep
    metric: str = "value",                 # "value" | "qty"
    years: Iterable[int] = range(2017, 2025),
    folder: str = "out",
    template: str = "flow_transformed_{yr}.txt",
) -> pl.DataFrame:
    """
    Fast year-over-year table of single-country disruption percentages.
    Uses an O(|E|) edge scan per year instead of rebuilding the graph
    for each candidate country.
    """
    metric_col = {"value": "primaryValue", "qty": "qty"}[metric.lower()]
    long_rows = []

    for yr in years:
        path = os.path.join(folder, template.format(yr=yr))
        if not os.path.exists(path):
            print(f"[warn] {path} missing – skipping {yr}.")
            continue

        # ── load once per year ────────────────────────────────────────────
        df = (
            pl.read_csv(
                path,
                separator="\t",
                infer_schema_length=1000,
                columns=["importer", "exporter", "cmdCode", metric_col],
            )
            .filter(
                (pl.col("cmdCode") == cmd_code)
                & (pl.col("importer") != "W00")
                & (pl.col("exporter") != "W00")
            )
        )

        if importer:                           # importer-centric slice
            df = df.filter(pl.col("importer") == importer)

        if df.is_empty():
            raise ValueError(f"No rows for {cmd_code} in {yr} (importer={importer})")

        # ── baseline total flow ───────────────────────────────────────────
        baseline = df[metric_col].sum()

        # ── gather edge weights twice (exporter & importer) ───────────────
        ends = (
            pl.concat([
                df.select(pl.col("exporter").alias("country"), pl.col(metric_col)),
                df.select(pl.col("importer").alias("country"), pl.col(metric_col)),
            ])
            .group_by("country")
            .agg(pl.col(metric_col).sum().alias("node_flow"))
        )

        # exclude central importer itself from candidates
        if importer:
            ends = ends.filter(pl.col("country") != importer)

        # add rows to long list
        long_rows.extend(
            {"year": yr,
             "country": row["country"],
             "disruption": row["node_flow"] / baseline * 100}
            for row in ends.iter_rows(named=True)
        )

    # ── long → wide, optional top-k filter ───────────────────────────────
    table_long = pl.DataFrame(long_rows)

    if k is not None:
        top_k = (
            table_long.group_by("country")
                      .agg(pl.col("disruption").mean().alias("avg"))
                      .sort("avg", descending=True)
                      .head(k)
                      .get_column("country")
                      .to_list()
        )
        table_long = table_long.filter(pl.col("country").is_in(top_k))

    table_wide = (
        table_long.pivot(index="year", columns="country", values="disruption")
                  .sort("year")
    )

    return table_wide
```


```python
tbl = disruption_table(282200, k=10, importer = "USA")
display(tbl.to_pandas()) 
```

    /tmp/ipykernel_726686/2479395516.py:90: DeprecationWarning: The argument `columns` for `DataFrame.pivot` is deprecated. It has been renamed to `on`.
      table_long.pivot(index="year", columns="country", values="disruption")



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>CHN</th>
      <th>FIN</th>
      <th>BEL</th>
      <th>GBR</th>
      <th>KOR</th>
      <th>EUR</th>
      <th>BRA</th>
      <th>NLD</th>
      <th>S19</th>
      <th>NAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>8.334140</td>
      <td>3.298633</td>
      <td>5.608430</td>
      <td>55.311847</td>
      <td>0.559741</td>
      <td>25.419363</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>5.402089</td>
      <td>2.845652</td>
      <td>13.453707</td>
      <td>34.681442</td>
      <td>0.299253</td>
      <td>42.369138</td>
      <td>0.213830</td>
      <td>0.090191</td>
      <td>0.230807</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>2.802488</td>
      <td>2.543324</td>
      <td>10.014450</td>
      <td>42.114241</td>
      <td>1.212047</td>
      <td>39.786455</td>
      <td>NaN</td>
      <td>0.287986</td>
      <td>0.643655</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>0.824781</td>
      <td>4.299153</td>
      <td>6.607371</td>
      <td>65.402119</td>
      <td>0.649244</td>
      <td>20.035741</td>
      <td>NaN</td>
      <td>1.192393</td>
      <td>0.495963</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>0.337760</td>
      <td>4.705841</td>
      <td>8.945052</td>
      <td>55.117506</td>
      <td>0.295553</td>
      <td>26.960040</td>
      <td>0.635542</td>
      <td>1.132669</td>
      <td>1.080810</td>
      <td>0.500138</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>0.444726</td>
      <td>7.069731</td>
      <td>8.482241</td>
      <td>48.150995</td>
      <td>0.224573</td>
      <td>32.158467</td>
      <td>0.078234</td>
      <td>2.183389</td>
      <td>0.829592</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023</td>
      <td>1.140298</td>
      <td>13.115517</td>
      <td>5.544123</td>
      <td>52.265388</td>
      <td>NaN</td>
      <td>23.383883</td>
      <td>NaN</td>
      <td>1.736523</td>
      <td>1.165673</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024</td>
      <td>0.518670</td>
      <td>11.388502</td>
      <td>10.577688</td>
      <td>45.038511</td>
      <td>0.060812</td>
      <td>29.460625</td>
      <td>NaN</td>
      <td>1.807485</td>
      <td>0.612933</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
usa_tbl = disruption_table(282200, importer="ESP", k = 5)
print(usa_tbl)
```

    shape: (8, 6)
    ┌──────┬───────────┬───────────┬───────────┬───────────┬───────────┐
    │ year ┆ ITA       ┆ GBR       ┆ CHN       ┆ BEL       ┆ FIN       │
    │ ---  ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
    │ i64  ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       │
    ╞══════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
    │ 2017 ┆ 49.711885 ┆ 5.183092  ┆ 25.310386 ┆ 5.870094  ┆ 11.776622 │
    │ 2018 ┆ 41.651907 ┆ 10.397389 ┆ 28.532144 ┆ 8.928974  ┆ 8.098787  │
    │ 2019 ┆ 24.760464 ┆ 3.078637  ┆ 41.189559 ┆ 21.109104 ┆ 7.96677   │
    │ 2020 ┆ 19.118196 ┆ 1.621299  ┆ 23.51117  ┆ 38.745807 ┆ 16.170917 │
    │ 2021 ┆ 0.467355  ┆ 1.160161  ┆ 44.438795 ┆ 42.103325 ┆ 10.396603 │
    │ 2022 ┆ 0.582082  ┆ 0.44172   ┆ 39.206442 ┆ 47.894788 ┆ 10.417384 │
    │ 2023 ┆ 0.273089  ┆ 0.012185  ┆ 37.957132 ┆ 56.569719 ┆ 2.308899  │
    │ 2024 ┆ 1.112722  ┆ 0.012238  ┆ 44.458997 ┆ 49.42917  ┆ 0.726922  │
    └──────┴───────────┴───────────┴───────────┴───────────┴───────────┘


    /tmp/ipykernel_726686/2479395516.py:90: DeprecationWarning: The argument `columns` for `DataFrame.pivot` is deprecated. It has been renamed to `on`.
      table_long.pivot(index="year", columns="country", values="disruption")



```python
import polars as pl
import itertools
import os
from typing import Optional, Iterable, List, Tuple, Dict, Set

##############################################################################
# 1)  max_disruption_set  – brute-force top-N country set for ONE year
##############################################################################
def max_disruption_set(
    cmd_code: int,
    year: int,
    n: int,
    *,
    metric: str = "value",                  # "value" | "qty"
    importer: Optional[str] = None,         # None → global, "USA" → USA-centric
    folder: str = "out",
    template: str = "flow_transformed_{yr}.txt",
) -> Tuple[Tuple[str, ...], float]:
    """
    Return the size-n country set S that maximises flow disruption
    (percent lost) for a given commodity & year.

    Parameters
    ----------
    cmd_code : int
    year     : int
    n        : int           (size of S, e.g. 2 for top-2 cut)
    metric   : {"value","qty"}
    importer : str | None
    folder, template : where the transformed files live

    Returns
    -------
    (best_set, disruption_pct) where
        best_set         – tuple of ISO-3 codes (lexicographic order)
        disruption_pct   – float, 0-100
    """
    metric_col = {"value": "primaryValue", "qty": "qty"}[metric.lower()]

    path = os.path.join(folder, template.format(yr=year))
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    # ── load & filter once ────────────────────────────────────────────────
    df = (
        pl.read_csv(
            path,
            separator="\t",
            columns=["importer", "exporter", "cmdCode", metric_col],
            infer_schema_length=1000,
        )
        .filter(
            (pl.col("cmdCode") == cmd_code)
            & (pl.col("importer") != "W00")
            & (pl.col("exporter") != "W00")
        )
    )
    if importer:
        df = df.filter(pl.col("importer") == importer)

    if df.is_empty():
        raise ValueError(f"No rows for cmdCode {cmd_code} in {year} (importer={importer})")

    edges: List[Tuple[str, str, float]] = [
        (exp, imp, float(w))
        for exp, imp, w in df.select(["exporter", "importer", metric_col]).iter_rows()
    ]
    baseline_total = sum(w for _, _, w in edges)

    # ── prepare candidate node list ───────────────────────────────────────
    if importer:
        candidates: List[str] = df["exporter"].unique().to_list()
    else:
        candidates = df.select(["importer"]).unique().get_column("importer").to_list()
        candidates += df.select(["exporter"]).unique().get_column("exporter").to_list()
        candidates = sorted(set(candidates))         # unique & stable order

    if importer and importer in candidates:
        candidates.remove(importer)                  # never drop the hub

    if n > len(candidates):
        raise ValueError(f"n ({n}) larger than number of candidate countries ({len(candidates)})")

    # ── brute-force search ────────────────────────────────────────────────
    best_set: Tuple[str, ...] = ()
    best_loss: float = -1.0

    for combo in itertools.combinations(candidates, n):
        combo_set: Set[str] = set(combo)
        removed_weight = sum(
            w for u, v, w in edges if u in combo_set or v in combo_set
        )
        loss_pct = removed_weight / baseline_total * 100.0
        if loss_pct > best_loss:
            best_loss, best_set = loss_pct, combo

    return tuple(sorted(best_set)), best_loss
```


```python
def yearly_max_disruption_sets(
    cmd_code: int,
    n: int,
    *,
    metric: str = "value",                  # "value" | "qty"
    importer: Optional[str] = None,         # None → global
    years: Iterable[int] = range(2017, 2025),
    folder: str = "out",
    template: str = "flow_transformed_{yr}.txt",
) -> pl.DataFrame:
    """
    For each year, compute the top-N disruptor set and its disruption score.

    Returns a tidy Polars table:
        year | countries | disruption
        -----+-----------+-----------
        2017 | (CHN,IDN) | 62.7
        2018 | (CHN,IDN) | 60.3
        ...
    """
    records = []
    for yr in years:
        try:
            best_set, loss = max_disruption_set(
                cmd_code, yr, n,
                metric=metric,
                importer=importer,
                folder=folder,
                template=template,
            )
            records.append({
                "year": yr,
                "countries": ", ".join(best_set),
                "disruption": loss,
            })
        except FileNotFoundError:
            print(f"[warn] missing {yr} file – skipped")
        except ValueError as e:
            print(f"[warn] {e} – skipped {yr}")

    return pl.DataFrame(records).sort("year")
```


```python
best_2017 = max_disruption_set(282200, 2017, n=2, importer = "USA", metric = "qty")
print("2017 top-2:", best_2017)
```

    2017 top-2: (('EUR', 'GBR'), 80.78745328546408)



```python
tbl = yearly_max_disruption_sets(280530, n=2, importer = "CHN")
display(tbl.to_pandas())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>countries</th>
      <th>disruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>USA, VNM</td>
      <td>64.252072</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>BDI, ESP</td>
      <td>81.556191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>KOR, R4</td>
      <td>57.092755</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>KOR, THA</td>
      <td>68.856007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>KOR, VNM</td>
      <td>82.359819</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>KOR, VNM</td>
      <td>59.880800</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023</td>
      <td>THA, VNM</td>
      <td>61.543120</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024</td>
      <td>THA, USA</td>
      <td>97.419068</td>
    </tr>
  </tbody>
</table>
</div>



```python
# ─── PARAMS ────────────────────────────────────────────────────────────────
CMD_CODE   = 280530          # HS-6 commodity (e.g. 282200 for cobalt chemicals)
IMPORTER   = None            # None → global network, or e.g. "USA"
YEARS      = range(2017, 2025)
OUT_FOLDER = "out"           # where flow_transformed_YYYY.txt live
# ───────────────────────────────────────────────────────────────────────────

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

# helper to collect disruption & set info for one metric --------------------
def collect(metric: str) -> pl.DataFrame:
    rows = []
    for k in (1, 2, 3):
        tbl = yearly_max_disruption_sets(
            CMD_CODE, k,
            metric=metric,
            importer=IMPORTER,
            years=YEARS,
            folder=OUT_FOLDER,
        )
        tbl = tbl.with_columns([
            pl.lit(metric).alias("metric"),
            pl.lit(k).alias("k")
        ])
        rows.append(tbl)
    return pl.concat(rows)

value_df = collect("value")
qty_df   = collect("qty")
full_df  = pl.concat([value_df, qty_df]).sort(["metric", "k", "year"])

# ─── plotting ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))

for k in (1, 2, 3):
    # solid → value
    subset = value_df.filter(pl.col("k") == k).to_pandas()
    plt.plot(subset["year"], subset["disruption"],
             label=f"k={k} (value)", linestyle="-")
    # dotted → qty
    subset = qty_df.filter(pl.col("k") == k).to_pandas()
    plt.plot(subset["year"], subset["disruption"],
             label=f"k={k} (qty)",   linestyle=":")

plt.title(f"Top-k Disruption over Time – cmdCode {CMD_CODE}"
          + (f" – importer {IMPORTER}" if IMPORTER else " – global"))
plt.xlabel("Year")
plt.ylabel("Flow lost (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── table of the best sets S ──────────────────────────────────────────────
display_df = (
    full_df
    .select([
        "year", "metric", "k", "countries", "disruption"
    ])
    .sort(["year", "metric", "k"])
)
pd.set_option("display.max_colwidth", None)
display(display_df.to_pandas())

```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[6], line 31
         28     return pl.concat(rows)
         30 value_df = collect("value")
    ---> 31 qty_df   = collect("qty")
         32 full_df  = pl.concat([value_df, qty_df]).sort(["metric", "k", "year"])
         34 # ─── plotting ──────────────────────────────────────────────────────────────


    Cell In[6], line 16, in collect(metric)
         14 rows = []
         15 for k in (1, 2, 3):
    ---> 16     tbl = yearly_max_disruption_sets(
         17         CMD_CODE, k,
         18         metric=metric,
         19         importer=IMPORTER,
         20         years=YEARS,
         21         folder=OUT_FOLDER,
         22     )
         23     tbl = tbl.with_columns([
         24         pl.lit(metric).alias("metric"),
         25         pl.lit(k).alias("k")
         26     ])
         27     rows.append(tbl)


    Cell In[5], line 24, in yearly_max_disruption_sets(cmd_code, n, metric, importer, years, folder, template)
         22 for yr in years:
         23     try:
    ---> 24         best_set, loss = max_disruption_set(
         25             cmd_code, yr, n,
         26             metric=metric,
         27             importer=importer,
         28             folder=folder,
         29             template=template,
         30         )
         31         records.append({
         32             "year": yr,
         33             "countries": ", ".join(best_set),
         34             "disruption": loss,
         35         })
         36     except FileNotFoundError:


    Cell In[4], line 46, in max_disruption_set(cmd_code, year, n, metric, importer, folder, template)
         42     raise FileNotFoundError(f"{path} not found")
         44 # ── load & filter once ────────────────────────────────────────────────
         45 df = (
    ---> 46     pl.read_csv(
         47         path,
         48         separator="\t",
         49         columns=["importer", "exporter", "cmdCode", metric_col],
         50         infer_schema_length=1000,
         51     )
         52     .filter(
         53         (pl.col("cmdCode") == cmd_code)
         54         & (pl.col("importer") != "W00")
         55         & (pl.col("exporter") != "W00")
         56     )
         57 )
         58 if importer:
         59     df = df.filter(pl.col("importer") == importer)


    File ~/.local/lib/python3.11/site-packages/polars/_utils/deprecation.py:92, in deprecate_renamed_parameter.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
         87 @wraps(function)
         88 def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
         89     _rename_keyword_argument(
         90         old_name, new_name, kwargs, function.__qualname__, version
         91     )
    ---> 92     return function(*args, **kwargs)


    File ~/.local/lib/python3.11/site-packages/polars/_utils/deprecation.py:92, in deprecate_renamed_parameter.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
         87 @wraps(function)
         88 def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
         89     _rename_keyword_argument(
         90         old_name, new_name, kwargs, function.__qualname__, version
         91     )
    ---> 92     return function(*args, **kwargs)


    File ~/.local/lib/python3.11/site-packages/polars/_utils/deprecation.py:92, in deprecate_renamed_parameter.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
         87 @wraps(function)
         88 def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
         89     _rename_keyword_argument(
         90         old_name, new_name, kwargs, function.__qualname__, version
         91     )
    ---> 92     return function(*args, **kwargs)


    File ~/.local/lib/python3.11/site-packages/polars/io/csv/functions.py:534, in read_csv(source, has_header, columns, new_columns, separator, comment_prefix, quote_char, skip_rows, skip_lines, schema, schema_overrides, null_values, missing_utf8_is_empty_string, ignore_errors, try_parse_dates, n_threads, infer_schema, infer_schema_length, batch_size, n_rows, encoding, low_memory, rechunk, use_pyarrow, storage_options, skip_rows_after_header, row_index_name, row_index_offset, sample_size, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, glob)
        526 else:
        527     with prepare_file_arg(
        528         source,
        529         encoding=encoding,
       (...)
        532         storage_options=storage_options,
        533     ) as data:
    --> 534         df = _read_csv_impl(
        535             data,
        536             has_header=has_header,
        537             columns=columns if columns else projection,
        538             separator=separator,
        539             comment_prefix=comment_prefix,
        540             quote_char=quote_char,
        541             skip_rows=skip_rows,
        542             skip_lines=skip_lines,
        543             schema_overrides=schema_overrides,
        544             schema=schema,
        545             null_values=null_values,
        546             missing_utf8_is_empty_string=missing_utf8_is_empty_string,
        547             ignore_errors=ignore_errors,
        548             try_parse_dates=try_parse_dates,
        549             n_threads=n_threads,
        550             infer_schema_length=infer_schema_length,
        551             batch_size=batch_size,
        552             n_rows=n_rows,
        553             encoding=encoding if encoding == "utf8-lossy" else "utf8",
        554             low_memory=low_memory,
        555             rechunk=rechunk,
        556             skip_rows_after_header=skip_rows_after_header,
        557             row_index_name=row_index_name,
        558             row_index_offset=row_index_offset,
        559             eol_char=eol_char,
        560             raise_if_empty=raise_if_empty,
        561             truncate_ragged_lines=truncate_ragged_lines,
        562             decimal_comma=decimal_comma,
        563             glob=glob,
        564         )
        566 if new_columns:
        567     return _update_columns(df, new_columns)


    File ~/.local/lib/python3.11/site-packages/polars/io/csv/functions.py:682, in _read_csv_impl(source, has_header, columns, separator, comment_prefix, quote_char, skip_rows, skip_lines, schema, schema_overrides, null_values, missing_utf8_is_empty_string, ignore_errors, try_parse_dates, n_threads, infer_schema_length, batch_size, n_rows, encoding, low_memory, rechunk, skip_rows_after_header, row_index_name, row_index_offset, sample_size, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, glob)
        678         raise ValueError(msg)
        680 projection, columns = parse_columns_arg(columns)
    --> 682 pydf = PyDataFrame.read_csv(
        683     source,
        684     infer_schema_length,
        685     batch_size,
        686     has_header,
        687     ignore_errors,
        688     n_rows,
        689     skip_rows,
        690     skip_lines,
        691     projection,
        692     separator,
        693     rechunk,
        694     columns,
        695     encoding,
        696     n_threads,
        697     path,
        698     dtype_list,
        699     dtype_slice,
        700     low_memory,
        701     comment_prefix,
        702     quote_char,
        703     processed_null_values,
        704     missing_utf8_is_empty_string,
        705     try_parse_dates,
        706     skip_rows_after_header,
        707     parse_row_index_args(row_index_name, row_index_offset),
        708     eol_char=eol_char,
        709     raise_if_empty=raise_if_empty,
        710     truncate_ragged_lines=truncate_ragged_lines,
        711     decimal_comma=decimal_comma,
        712     schema=schema,
        713 )
        714 return wrap_df(pydf)


    KeyboardInterrupt: 



```python
import polars as pl
import itertools, os
import matplotlib.pyplot as plt
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────
# helpers reused from earlier cells ----------------------------------------
#   → max_disruption_set()
#   → network_flow_disruption()      (used inside max_disruption_set)
#   (make sure those are already defined)
# ───────────────────────────────────────────────────────────────────────────


def plot_disruption_waterfall(
    cmd_code: int,
    *,
    importer: str | None = None,          # None = global view
    k: int = 3,                           # size of best set S
    years=range(2017, 2025),
    folder: str = "out",
    template: str = "flow_transformed_{yr}.txt",
):
    """
    Draws a stacked-bar “waterfall” for both value & qty disruption
    (two adjacent bars per year).  Each bar is the *k-country* set S that
    maximises disruption that year.

    Also prints a table:  year | metric | S | disruption %.
    """

    def best_sets(metric: str) -> pl.DataFrame:
        rows = []
        for yr in years:
            try:
                S, loss = max_disruption_set(
                    cmd_code, yr, k,
                    metric=metric,
                    importer=importer,
                    folder=folder,
                    template=template,
                )
                rows.append({"year": yr, "metric": metric,
                             "countries": list(S), "disruption": loss})
            except (FileNotFoundError, ValueError):
                # file missing or no data for that year—skip
                continue
        return pl.DataFrame(rows)

    value_df = best_sets("value")
    qty_df   = best_sets("qty")
    full_df  = pl.concat([value_df, qty_df])

    if full_df.is_empty():
        print("No data found for the requested parameters.")
        return

    # ── build contributions per country (incremental) ─────────────────────
    # We'll compute each country's incremental share so stack sums exactly.
    contrib_rows = []
    for row in full_df.iter_rows(named=True):
        yr       = row["year"]
        metric   = row["metric"]
        base_set = row["countries"]
        base_set_sorted = sorted(base_set)   # stable order for reproducibility
        prev_loss = 0.0
        so_far    = set()

        for c in base_set_sorted:
            loss = network_flow_disruption(
                cmd_code, yr,
                remove=so_far | {c},
                importer=importer,
                metric=metric,
                folder=folder,
                template=template,
            )
            contrib = loss - prev_loss
            contrib_rows.append({
                "year": yr,
                "metric": metric,
                "country": c,
                "contrib": contrib,
            })
            prev_loss += contrib
            so_far.add(c)

    contrib_df = pl.DataFrame(contrib_rows)

    # ── plotting ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_w   = 0.35
    x_base  = list(range(len(years)))

    for j, (metric, linestyle) in enumerate([("value", "-"), ("qty", ":")]):
        offset = -bar_w/2 if metric == "value" else bar_w/2
        bottoms = {yr: 0.0 for yr in years}

        # iterate countries in a deterministic order for colour stability
        for country in sorted(contrib_df["country"].unique()):
            part = contrib_df.filter(
                (pl.col("metric") == metric) & (pl.col("country") == country)
            )
            if part.is_empty():
                continue
            xs   = [x_base[years.index(yr)] + offset for yr in part["year"]]
            ys   = part["contrib"].to_list()
            ax.bar(xs, ys, bar_w, bottom=[bottoms[yr] for yr in part["year"]])
            # update bottoms
            for yr, h in zip(part["year"], ys):
                bottoms[yr] += h

    ax.set_xticks(x_base, years)
    ax.set_ylabel("Flow lost (%)")
    title = f"Top-{k} disruption waterfall – cmd {cmd_code}"
    if importer:
        title += f" – imports of {importer}"
    ax.set_title(title)
    ax.legend(["value (solid stack)", "qty (dotted stack)"])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── table with sets S and totals ──────────────────────────────────────
    display_df = (
        full_df
        .select(["year", "metric", "countries", "disruption"])
        .sort(["year", "metric"])
    )
    pd.set_option("display.max_colwidth", None)
    display(display_df.to_pandas())


# ─── Example call ─────────────────────────────────────────────────────────
# plot_disruption_waterfall(282200, importer=None, k=3)
```


```python
plot_disruption_waterfall(810530, importer=None, k=3)
```


    
![png](output_13_0.png)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>countries</th>
      <th>disruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>[CAN, GBR, USA]</td>
      <td>85.655576</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>value</td>
      <td>[CAN, GBR, USA]</td>
      <td>87.572407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018</td>
      <td>qty</td>
      <td>[CAN, GBR, USA]</td>
      <td>85.404525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018</td>
      <td>value</td>
      <td>[CAN, GBR, USA]</td>
      <td>86.782166</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019</td>
      <td>qty</td>
      <td>[CAN, GBR, USA]</td>
      <td>83.783567</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>value</td>
      <td>[CAN, GBR, USA]</td>
      <td>85.757565</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>qty</td>
      <td>[CAN, GBR, USA]</td>
      <td>84.212205</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020</td>
      <td>value</td>
      <td>[CAN, DEU, GBR]</td>
      <td>84.191797</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2021</td>
      <td>qty</td>
      <td>[DEU, GBR, USA]</td>
      <td>83.823495</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2021</td>
      <td>value</td>
      <td>[DEU, GBR, USA]</td>
      <td>88.676989</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2022</td>
      <td>qty</td>
      <td>[DEU, GBR, USA]</td>
      <td>85.175804</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2022</td>
      <td>value</td>
      <td>[DEU, GBR, USA]</td>
      <td>89.201137</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023</td>
      <td>qty</td>
      <td>[GBR, JPN, USA]</td>
      <td>82.793305</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023</td>
      <td>value</td>
      <td>[DEU, GBR, USA]</td>
      <td>91.932434</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024</td>
      <td>qty</td>
      <td>[CAN, DEU, GBR]</td>
      <td>79.193515</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024</td>
      <td>value</td>
      <td>[DEU, GBR, USA]</td>
      <td>88.574937</td>
    </tr>
  </tbody>
</table>
</div>



```python
import polars as pl
import matplotlib.pyplot as plt
import os

def plot_k123_disruption(
    cmd_code: int,
    *,
    importer: str | None = None,           # None = global
    years = range(2017, 2025),
    folder: str = "out",
    template: str = "flow_transformed_{yr}.txt",
):
    """
    Two stacked-bar charts (value & qty).  Each bar shows cumulative disruption
    for k=1 → k=3; colours represent the incremental jump when k grows.

    Parameters
    ----------
    cmd_code : int
    importer : str | None      focal importer (None = global network)
    years    : iterable[int]
    folder, template : file locations for flow_transformed_YYYY.txt
    """

    def gather(metric: str) -> pl.DataFrame:
        rows = []
        for yr in years:
            # compute best loss for k = 1, 2, 3
            losses = []
            for k in (1, 2, 3):
                try:
                    _, loss = max_disruption_set(
                        cmd_code, yr, k,
                        metric=metric, importer=importer,
                        folder=folder, template=template,
                    )
                    losses.append(loss)
                except (FileNotFoundError, ValueError):
                    break       # file missing or no data; skip year
            if len(losses) == 3:
                seg1   = losses[0]
                seg2   = losses[1] - losses[0]
                seg3   = losses[2] - losses[1]
                rows.append({
                    "year": yr,
                    "seg1": seg1,
                    "seg2": seg2,
                    "seg3": seg3,
                    "metric": metric,
                })
        return pl.DataFrame(rows).sort("year")

    value_df = gather("value")
    qty_df   = gather("qty")

    if value_df.is_empty() or qty_df.is_empty():
        print("No data found for the requested parameters.")
        return

    # ── plotting helper ────────────────────────────────────────────────────
    def stacked(ax, data: pl.DataFrame, title_suffix: str):
        xs   = data["year"].to_list()
        seg1 = data["seg1"].to_list()
        seg2 = data["seg2"].to_list()
        seg3 = data["seg3"].to_list()

        ax.bar(xs, seg1, label="k=1")
        ax.bar(xs, seg2, bottom=seg1, label="k=2 incremental")
        ax.bar(xs, seg3, bottom=[s1+s2 for s1, s2 in zip(seg1, seg2)],
               label="k=3 incremental")

        ax.set_xlabel("Year")
        ax.set_ylabel("Flow lost (%)")
        ax.set_title(f"Top-k disruption ({title_suffix}) – cmd {cmd_code}"
                     + (f" – imports of {importer}" if importer else " – global"))
        ax.set_xticks(xs, xs)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend()

    # ── draw the two charts side-by-side ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    stacked(axes[0], value_df, "value")
    stacked(axes[1], qty_df,   "qty")
    plt.tight_layout()
    plt.show()


# ─── Example call ─────────────────────────────────────────────────────────
# global network
# plot_k123_disruption(282200, importer=None)

# USA-centric imports
# plot_k123_disruption(282200, importer="USA")
```


```python
plot_k123_disruption(810530, importer=None)
```


    
![png](output_15_0.png)
    



```python
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextlib, io, pathlib
import itertools, matplotlib as mpl

# ── stable colour map: ISO-3 → RGB ─────────────────────────────
_BASE_PALETTE   = list(mpl.colormaps["tab20"].colors)   # 20 nicely spaced hues
_palette_cycle  = itertools.cycle(_BASE_PALETTE)        # endless generator
_ISO2COLOR: dict[str, tuple[float, float, float]] = {}  # cache

def _color_for_iso(iso: str) -> tuple:
    """Return the same RGB tuple every time this ISO-3 code is requested."""
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = next(_palette_cycle)
    return _ISO2COLOR[iso]

# ──────────────────────────────────────────────────────────────────────────
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_k123_disruption_coloured(
    cmd_code: int,
    *,
    importer: str | None = None,    # None = global
    years        = range(2017, 2025),
    folder       = "out",
    template     = "flow_transformed_{yr}.txt",
):
    """
    Two stacked bars per year (value & qty), each bar shows cumulative
    disruption for k=1,2,3, and each coloured segment is the country
    responsible for the incremental jump.
    Compatible with old Polars builds (≤0.17).
    """

    # ----------------------- helpers ----------------------------
    def best_losses(metric: str):
        """Return dict {year: [(country, inc1), (country, inc2), (country, inc3)]}"""
        res = {}
        for yr in years:
            try:
                S3, _ = max_disruption_set(cmd_code, yr, 3,
                                           metric=metric, importer=importer,
                                           folder=folder, template=template)
            except (FileNotFoundError, ValueError):
                continue

            so_far, prev = set(), 0.0
            triples = []
            for c in S3:                        # deterministic order
                loss = network_flow_disruption(
                    cmd_code, yr,
                    remove=so_far | {c},
                    importer=importer, metric=metric,
                    folder=folder, template=template,
                )
                triples.append((c, loss - prev))
                so_far.add(c)
                prev = loss
            res[yr] = triples                 # list length == 3
        return res

    val_loss = best_losses("value")
    qty_loss = best_losses("qty")

    if not val_loss or not qty_loss:
        print("No data for requested parameters.")
        return

    # -------- plotting routine (no .list(), no .apply()) --------
    def plot_one(ax, loss_dict, metric_name):
        yrs = sorted(loss_dict)
        bottoms = [0.0]*len(yrs)             # cumulative bottoms per bar
        used_countries = set(c for triple in loss_dict.values() for c, _ in triple)

        # stable colour per ISO-3
        colours = {c: _color_for_iso(c) for c in sorted(used_countries)}

        for tier in range(3):                # 0 → k=1, 1 → inc to k=2 …
            heights = []
            tier_countries = []
            for y in yrs:
                c, h = loss_dict[y][tier]
                heights.append(h)
                tier_countries.append(c)
            ax.bar(yrs, heights, 0.6,
                   bottom=bottoms,
                   color=[colours[c] for c in tier_countries],
                   edgecolor="black", linewidth=.3)
            bottoms = [b+h for b, h in zip(bottoms, heights)]

        ax.set_xticks(yrs, yrs)
        ax.set_ylabel("Flow lost (%)")
        ttl = f"k=1→3 disruption ({metric_name}) – cmd {cmd_code}"
        ttl += f" – imports of {importer}" if importer else " – global"
        ax.set_title(ttl)
        ax.grid(axis="y", ls="--", alpha=.4)

        # legend for countries
        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0], color=colours[c], lw=6) for c in sorted(colours)]
        ax.legend(handles, sorted(colours), ncol=2, title="Country", frameon=False)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6), sharey=True)
    plot_one(ax1, val_loss, "value")
    plot_one(ax2, qty_loss, "qty")
    plt.tight_layout()
    plt.show()

    # -------- simple table -------------------------------------
    tbl = []
    for metric, d in [("value", val_loss), ("qty", qty_loss)]:
        for y in sorted(d):
            tot = sum(h for _, h in d[y])
            countries = ", ".join(c for c, _ in d[y])
            tbl.append([y, metric, f"{tot:.1f} %", countries])
    print(pd.DataFrame(tbl, columns=["Year","Metric","k=3 Disruption","Countries"]))

# ─── Example call ----------------------------------------------------------
# plot_k123_disruption_coloured(282200, importer=None)
# plot_k123_disruption_coloured(282200, importer="USA")
```


```python
plot_k123_disruption_coloured(280530, importer="USA")
```


    
![png](output_17_0.png)
    


        Year Metric k=3 Disruption      Countries
    0   2017  value         83.4 %  CHN, EUR, GBR
    1   2018  value         81.8 %  CHN, GBR, RUS
    2   2019  value         81.1 %  CHN, EUR, GBR
    3   2020  value         87.3 %  CHN, GBR, RUS
    4   2021  value         94.8 %  CHN, GBR, RUS
    5   2022  value         95.9 %  CAN, CHN, GBR
    6   2023  value         93.0 %  CHN, DEU, GBR
    7   2024  value         93.2 %  CAN, CHN, GBR
    8   2017    qty         96.7 %  CHN, EUR, GBR
    9   2018    qty         93.2 %  CHN, EUR, GBR
    10  2019    qty         82.6 %  CHN, GBR, HKG
    11  2020    qty         94.5 %  CHN, GBR, RUS
    12  2021    qty         96.3 %  CHN, EUR, GBR
    13  2022    qty         99.3 %  CAN, CHN, GBR
    14  2023    qty         99.2 %  CAN, CHN, GBR
    15  2024    qty         75.6 %  CAN, CHN, GBR



```python
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
import os

# make sure max_disruption_set & network_flow_disruption are already defined
# and the colour helper _color_for_iso is in memory

def plot_k123_disruption_lines(
    cmd_code: int,
    *,
    importer: str | None = None,           # None → global
    years      = range(2017, 2025),
    folder     = "out",
    template   = "flow_transformed_{yr}.txt",
):
    """
    Line chart: top-k disruption (k = 1,2,3) for value & qty.
    • Each k uses one colour (solid = value, dashed = qty).
    • Y-axis always starts at 0.
    • Table lists disruption % and country set S for every entry.
    """

    # ------------- gather data (loss + country set) -----------------------
    rows = []
    for metric in ("value", "qty"):
        for k in (1, 2, 3):
            for yr in years:
                try:
                    S, loss = max_disruption_set(
                        cmd_code, yr, k,
                        metric=metric, importer=importer,
                        folder=folder, template=template
                    )
                    rows.append({
                        "year": yr,
                        "metric": metric,
                        "k": k,
                        "loss": loss,
                        "countries": ", ".join(S)
                    })
                except (FileNotFoundError, ValueError):
                    continue  # skip this (yr, metric, k) if data missing

    df_all = pl.DataFrame(rows)
    if df_all.is_empty():
        print("No data for requested parameters.")
        return

    # ------------- plotting ----------------------------------------------
    colours = dict(zip((1, 2, 3), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in (1, 2, 3):
        col = colours[k]

        # value
        sub_v = df_all.filter((pl.col("k") == k) & (pl.col("metric") == "value")).to_pandas()
        ax.plot(sub_v["year"], sub_v["loss"],
                color=col, linestyle="-", linewidth=2,
                label=f"k={k} value")

        # qty
        sub_q = df_all.filter((pl.col("k") == k) & (pl.col("metric") == "qty")).to_pandas()
        ax.plot(sub_q["year"], sub_q["loss"],
                color=col, linestyle="--", linewidth=2,
                label=f"k={k} qty")

    ax.set_xlabel("Year")
    ax.set_ylabel("Flow lost (%)")
    ax.set_ylim(bottom=0)                        # y-axis starts at 0
    title = f"Top-k disruption (cmd {cmd_code})"
    title += f" – imports of {importer}" if importer else " – global"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------- table --------------------------------------------------
    table = (df_all
             .sort(["year", "metric", "k"])
             .to_pandas())
    pd.set_option("display.max_colwidth", None)
    display(table)
```


```python
plot_k123_disruption_lines(280530)
```


    
![png](output_19_0.png)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>74.360851</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>86.053310</td>
      <td>AUS, CHN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>91.884872</td>
      <td>CHN, MYS, R4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>42.214424</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>72.926405</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>85.439368</td>
      <td>AUS, CHN, JPN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>67.003421</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>86.415377</td>
      <td>AUS, CHN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>91.869033</td>
      <td>CHN, MYS, R4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>44.760091</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>67.981158</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>84.154150</td>
      <td>AUS, CHN, JPN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>66.533981</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>86.008878</td>
      <td>AUS, CHN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>93.275575</td>
      <td>AUS, CHN, JPN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>45.441409</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>64.839781</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>81.593335</td>
      <td>AUS, JPN, THA</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>68.702711</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>83.392159</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>92.193326</td>
      <td>AUS, CHN, JPN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>51.673592</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>70.870402</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>86.600443</td>
      <td>AUS, JPN, THA</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>55.462545</td>
      <td>MYS</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>79.815841</td>
      <td>CHN, MYS</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>90.476399</td>
      <td>CHN, MYS, THA</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>46.634172</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>69.536660</td>
      <td>JPN, THA</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>87.157264</td>
      <td>AUS, JPN, THA</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>75.580866</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>89.589594</td>
      <td>AUS, CHN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>95.890029</td>
      <td>AUS, CHN, JPN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>57.037654</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>75.764558</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>92.096955</td>
      <td>AUS, JPN, THA</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>66.001366</td>
      <td>AUS</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>86.742110</td>
      <td>AUS, CHN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>94.278985</td>
      <td>AUS, CHN, THA</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>55.913887</td>
      <td>JPN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>76.121935</td>
      <td>AUS, JPN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>87.524004</td>
      <td>AUS, JPN, THA</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>68.353023</td>
      <td>MYS</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>86.567318</td>
      <td>CHN, MYS</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>94.564643</td>
      <td>CHN, JPN, MYS</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>50.703585</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>73.266456</td>
      <td>JPN, MYS</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>88.013898</td>
      <td>AUS, JPN, THA</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_k123_disruption_lines(280530, importer = "USA")
```


    
![png](output_20_0.png)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>59.024332</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>82.404753</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>96.701798</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>46.035253</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>70.224233</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>83.371395</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>75.683746</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>86.896844</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>93.191616</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>46.273702</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>64.841697</td>
      <td>CHN, RUS</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>81.841433</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>42.324030</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>68.114638</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>82.596354</td>
      <td>CHN, GBR, HKG</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>36.914700</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>66.097164</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>81.109126</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>81.979921</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>91.367510</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>94.501324</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>56.782958</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>74.731963</td>
      <td>CHN, RUS</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>87.284671</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>84.654999</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>91.471400</td>
      <td>CHN, EUR</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>96.332250</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>55.223164</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>84.000554</td>
      <td>CHN, RUS</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>94.819295</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>64.983596</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>89.098496</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>99.338405</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>64.745651</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>92.302091</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>95.927329</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>88.595644</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>96.862984</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>99.237071</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>70.543627</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>85.879285</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>93.022353</td>
      <td>CHN, DEU, GBR</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>30.973441</td>
      <td>CAN</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>56.965138</td>
      <td>CAN, CHN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>75.595116</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>37.032169</td>
      <td>GBR</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>72.250108</td>
      <td>CAN, GBR</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>93.164083</td>
      <td>CAN, CHN, GBR</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_k123_disruption_lines(280530, importer = "CHN")
```


    
![png](output_21_0.png)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>54.323473</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>74.661504</td>
      <td>BDI, VNM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>81.336022</td>
      <td>BDI, USA, VNM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>54.390159</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>64.252072</td>
      <td>USA, VNM</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>71.587350</td>
      <td>KOR, USA, VNM</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>81.561507</td>
      <td>BDI</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>89.618431</td>
      <td>BDI, R4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>93.853954</td>
      <td>BDI, ESP, R4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>70.917034</td>
      <td>BDI</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>81.556191</td>
      <td>BDI, ESP</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>87.266518</td>
      <td>BDI, ESP, KOR</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>63.918571</td>
      <td>R4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>95.876033</td>
      <td>PHL, R4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>96.591011</td>
      <td>KOR, PHL, R4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>43.081320</td>
      <td>KOR</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>57.092755</td>
      <td>KOR, R4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>66.994537</td>
      <td>ESP, KOR, R4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>58.961950</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>72.330992</td>
      <td>R4 , THA</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>81.153192</td>
      <td>KOR, R4 , THA</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>53.221600</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>68.856007</td>
      <td>KOR, THA</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>81.013597</td>
      <td>KOR, R4 , THA</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>72.687934</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>82.761516</td>
      <td>JPN, VNM</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>88.072546</td>
      <td>JPN, KOR, VNM</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>68.145243</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>82.359819</td>
      <td>KOR, VNM</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>88.317739</td>
      <td>JPN, KOR, VNM</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>35.131782</td>
      <td>KOR</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>68.066507</td>
      <td>KOR, VNM</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>83.814161</td>
      <td>JPN, KOR, VNM</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>30.602019</td>
      <td>KOR</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>59.880800</td>
      <td>KOR, VNM</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>75.118009</td>
      <td>JPN, KOR, VNM</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>66.894712</td>
      <td>LAO</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>79.185224</td>
      <td>LAO, VNM</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>91.180100</td>
      <td>LAO, R4 , VNM</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>41.884781</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>61.543120</td>
      <td>THA, VNM</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>77.207418</td>
      <td>LAO, THA, VNM</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>95.843817</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>99.584664</td>
      <td>KOR, THA</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>99.784694</td>
      <td>JPN, KOR, THA</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>88.212066</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>97.419068</td>
      <td>THA, USA</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>99.826190</td>
      <td>KOR, THA, USA</td>
    </tr>
  </tbody>
</table>
</div>



```python
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import contextlib, io, pathlib
import itertools, matplotlib as mpl

# ── make sure _color_for_iso, max_disruption_set, and network_flow_disruption
#    are defined in the notebook before you run this cell
# -------------------------------------------------------------------------
# ── stable colour map: ISO-3 → RGB ─────────────────────────────
_BASE_PALETTE   = list(mpl.colormaps["tab20"].colors)   # 20 nicely spaced hues
_palette_cycle  = itertools.cycle(_BASE_PALETTE)        # endless generator
_ISO2COLOR: dict[str, tuple[float, float, float]] = {}  # cache

#increase line thickness in patterns
mpl.rcParams["hatch.linewidth"] = 2.0

def _color_for_iso(iso: str) -> tuple:
    """Return the same RGB tuple every time this ISO-3 code is requested."""
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = next(_palette_cycle)
    return _ISO2COLOR[iso]

def plot_top3_disruption(
    cmd_code: int,
    importer: str,                       # REQUIRED focal importer (ISO-3)
    *,
    years      = range(2017, 2025),
    folder     = "out",
    template   = "flow_transformed_{yr}.txt",
):
    """
    Two stacked-bar charts (value & qty) showing the additive disruption
    from removing the top-1, top-2, and top-3 exporters to *importer*.

    Bars: bottom = S1, middle = incremental (S2 – S1), top = incremental (S3 – S2)
    Colours: country of the exporter added at each level.
    """

    def collect(metric: str):
        recs = []
        for yr in years:
            path = os.path.join(folder, template.format(yr=yr))
            if not os.path.exists(path):
                continue

            df = (
                pl.read_csv(
                    path,
                    separator="\t",
                    columns=["importer", "exporter", "cmdCode", metric],
                    infer_schema_length=1000,
                )
                .filter(
                    (pl.col("cmdCode") == cmd_code) &
                    (pl.col("importer") == importer) &
                    (pl.col("exporter") != "W00")
                )
            )
            if df.is_empty():
                continue

            total = df[metric].sum()
            top3  = (
                df.group_by("exporter")
                  .agg(pl.col(metric).sum().alias("flow"))
                  .sort("flow", descending=True)
                  .head(3)
                  .to_dict(as_series=False)
            )

            exporters = top3["exporter"]
            flows     = top3["flow"]

            if not exporters:        # no data
                continue

            # build incremental pieces
            pieces, cum = [], 0.0
            for exp, fl in zip(exporters, flows):
                share = fl / total * 100.0
                pieces.append((exp, share))
                cum += share

            recs.append({"year": yr,
                         "metric": metric,
                         "pieces": pieces})      # list of (country, share)
        return recs

    data_val = collect("primaryValue")
    data_qty = collect("qty")

    if not data_val or not data_qty:
        print("No data for requested parameters.")
        return

    def plot_panel(ax, records, mlabel):
        yrs = [r["year"] for r in records]
        bottoms = [0.0]*len(records)
        used = {c for r in records for c, _ in r["pieces"]}

        colours = {c: _color_for_iso(c) for c in sorted(used)}

        # three layers max
        for tier in range(3):
            heights, cs = [], []
            for r in records:
                if tier < len(r["pieces"]):
                    c, h = r["pieces"][tier]
                else:                 # fewer than 3 exporters that year
                    c, h = None, 0.0
                heights.append(h)
                cs.append(c)

            bars = ax.bar(yrs, heights, 0.55,
                          bottom=bottoms,
                          color=[colours.get(c, "#cccccc") for c in cs],
                          edgecolor="black", linewidth=.3)

            bottoms = [b+h for b, h in zip(bottoms, heights)]

        ax.set_xticks(yrs, yrs)
        ax.set_ylabel("Flow lost (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{mlabel} – cmd {cmd_code} – imports of {importer}")
        ax.grid(axis="y", linestyle="--", alpha=.4)

        # legend
        handles = [Line2D([0],[0], color=colours[c], lw=6) for c in sorted(colours)]
        ax.legend(handles, sorted(colours), ncol=2,
                  title="Exporter", frameon=False, fontsize=8, title_fontsize=8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_panel(ax1, data_val, "value")
    plot_panel(ax2, data_qty, "quantity")
    plt.tight_layout()
    plt.show()

    # ---------- summary table --------------------------------------------
    rows = []
    for metric, records in [("value", data_val), ("quantity", data_qty)]:
        for r in records:
            tot = sum(h for _, h in r["pieces"])
            countries = ", ".join(c for c, _ in r["pieces"])
            rows.append([r["year"], metric, f"{tot:.1f} %", countries])

    print(pd.DataFrame(rows,
                       columns=["Year", "Metric", "k=3 Disruption", "Countries"]))
```


```python
plot_top3_disruption(282200, importer="USA")
```


    
![png](output_23_0.png)
    


        Year    Metric k=3 Disruption      Countries
    0   2017     value         89.1 %  GBR, EUR, CHN
    1   2018     value         90.5 %  EUR, GBR, BEL
    2   2019     value         91.9 %  GBR, EUR, BEL
    3   2020     value         92.0 %  GBR, EUR, BEL
    4   2021     value         91.0 %  GBR, EUR, BEL
    5   2022     value         88.8 %  GBR, EUR, BEL
    6   2023     value         88.8 %  GBR, EUR, FIN
    7   2024     value         85.9 %  GBR, EUR, FIN
    8   2017  quantity         89.3 %  GBR, EUR, CHN
    9   2018  quantity         88.9 %  EUR, GBR, BEL
    10  2019  quantity         88.4 %  EUR, GBR, BEL
    11  2020  quantity         89.0 %  GBR, EUR, BEL
    12  2021  quantity         86.7 %  GBR, EUR, BEL
    13  2022  quantity         86.1 %  GBR, EUR, BEL
    14  2023  quantity         83.1 %  EUR, GBR, FIN
    15  2024  quantity         85.0 %  GBR, EUR, BEL



```python
# ────────────────────────────────────────────────────────────────────────────
#  Persistent ISO-3 → colour mapping  (CHN = dark-red, USA = dark-blue)
# ────────────────────────────────────────────────────────────────────────────
import json, itertools, matplotlib as mpl
from pathlib import Path

COLOR_PATH  = Path("/sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI")
COLOR_PATH.mkdir(parents=True, exist_ok=True)
_COLOR_FILE = COLOR_PATH / "iso2color.json"

_BASE_PALETTE = (list(mpl.colormaps["tab20"].colors) +
                 list(mpl.colormaps["tab20b"].colors) +
                 list(mpl.colormaps["tab20c"].colors))
_palette_cycle = itertools.cycle(_BASE_PALETTE)

if _COLOR_FILE.exists():
    with open(_COLOR_FILE, "r") as f:
        _ISO2COLOR = {k: tuple(v) for k, v in json.load(f).items()}
else:
    _ISO2COLOR = {}

_ISO2COLOR["CHN"] = mpl.colors.to_rgb("darkred")
_ISO2COLOR["USA"] = mpl.colors.to_rgb("blue")

def _next_unused_colour():
    for col in _palette_cycle:
        if col not in _ISO2COLOR.values():
            return col
    raise RuntimeError("ran out of palette colours")

def _color_for_iso(iso: str):
    iso = iso.upper()
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = _next_unused_colour()
        with open(_COLOR_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _ISO2COLOR.items()}, f, indent=2)
    return _ISO2COLOR[iso]

# ────────────────────────────────────────────────────────────────────────────
#  Plot helper – grouped value/qty bars, top-3 exporters per year
# ────────────────────────────────────────────────────────────────────────────
import polars as pl, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
mpl.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 15, "legend.title_fontsize": 17,
    "hatch.linewidth": 1.5,
})

def plot_top3_disruption_grouped(
    cmd_code: int,
    importer: str,
    *,
    years    = range(2017, 2025),
    folder   = "out",
    template = "flow_transformed_{yr}.txt",
):
    """Grouped value/qty bars, stacked by k=1-3; colours = exporters."""
    def collect(metric: str):
        recs = []
        for yr in years:
            fp = Path(folder) / template.format(yr=yr)
            if not fp.exists(): continue
            df = (
                pl.read_csv(fp, separator="\t",
                            columns=["importer","exporter","cmdCode",metric],
                            infer_schema_length=1000)
                  .filter(
                      (pl.col("cmdCode") == cmd_code) &
                      (pl.col("importer") == importer) &
                      (pl.col("exporter") != "W00") &
                      (pl.col("importer") != "W00") &          # NEW filter
                      (pl.col("exporter") != pl.col("importer"))
                  )
            )
            if df.is_empty(): continue
            total = df[metric].sum()
            top3 = (df.group_by("exporter")
                      .agg(pl.col(metric).sum().alias("flow"))
                      .sort("flow", descending=True)
                      .head(3))
            pieces = [(r["exporter"], r["flow"] / total * 100)
                      for r in top3.iter_rows(named=True)]
            recs.append({"year": yr, "pieces": pieces})
        return recs

    dv, dq = collect("primaryValue"), collect("qty")
    if not dv or not dq:
        print("No data."); return

    # ---------------- title flag (commodity families) --------------------
    title_flag = ""
    if cmd_code in {280530, 284690, 850511, 720299}:
        title_flag = "(Nd/Dy)"
    if cmd_code in {811299, 285000, 381800}:
        title_flag = "(Ga/In)"
    # add more families here if desired
    # --------------------------------------------------------------------

    used = {c for rec in dv+dq for c,_ in rec["pieces"]}
    colours = {c: _color_for_iso(c) for c in used}

    yrs = sorted({r["year"] for r in dv})
    x   = np.arange(len(yrs)); w = .35
    fig, ax = plt.subplots(figsize=(14,6)); fig.subplots_adjust(right=.68)

    b_val = {y:0 for y in yrs}; b_q = {y:0 for y in yrs}
    for tier in range(3):
        for rec in dv:
            if tier < len(rec["pieces"]):
                c,h = rec["pieces"][tier]
                ax.bar(x[yrs.index(rec["year"])]-w/2, h, w,
                       bottom=b_val[rec["year"]],
                       color=colours[c], edgecolor="black", lw=.3)
                b_val[rec["year"]] += h
        for rec in dq:
            if tier < len(rec["pieces"]):
                c,h = rec["pieces"][tier]
                ax.bar(x[yrs.index(rec["year"])]+w/2, h, w,
                       bottom=b_q[rec["year"]],
                       color=colours[c], edgecolor="black", lw=.3, hatch="//")
                b_q[rec["year"]] += h

    ax.set_xticks(x, yrs)
    ax.set_ylabel("Flow lost (%)"); ax.set_ylim(0,100)
    ax.set_title(f"Top-3 Disruption – {cmd_code} {title_flag} – {importer} imports")
    ax.grid(axis="y", ls="--", alpha=.4)

    exp_handles = [Line2D([0],[0], color=colours[c], lw=6) for c in sorted(colours)]
    exp_leg = ax.legend(exp_handles, sorted(colours), title="Exporter",
                        bbox_to_anchor=(1.02,1.0), loc="upper left",
                        frameon=False)
    ax.add_artist(exp_leg)
    ax.legend([Patch(fc="black"),
               Patch(fc="white", ec="black", hatch="//")],
              ["Value","Quantity"], title="Metric",
              bbox_to_anchor=(1.02,0.20), loc="upper left",
              frameon=False)

    plt.tight_layout(); plt.show()
```


```python
plot_top3_disruption_grouped(280530, importer="USA")
plot_top3_disruption_grouped(284690, importer="USA")
plot_top3_disruption_grouped(850511, importer="USA")
plot_top3_disruption_grouped(720299, importer="USA")
```


    
![png](output_25_0.png)
    



    
![png](output_25_1.png)
    



    
![png](output_25_2.png)
    



    
![png](output_25_3.png)
    



```python
plot_top3_disruption_grouped(280530, importer="CHN")
plot_top3_disruption_grouped(284690, importer="CHN")
plot_top3_disruption_grouped(850511, importer="CHN")
plot_top3_disruption_grouped(720299, importer="CHN")
```


    
![png](output_26_0.png)
    



    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    



    
![png](output_26_3.png)
    



```python
import polars as pl
import pandas as pd
from typing import Iterable, Optional

# make sure max_disruption_set & network_flow_disruption are already defined

def max_disruption_across_codes(
    cmd_codes: Iterable[int],
    *,
    importer: Optional[str] = None,          # None = global network
    years     = range(2017, 2025),
    folder    = "out",
    template  = "flow_transformed_{yr}.txt",
) -> pl.DataFrame:
    """
    For every (year, metric, k) find the HS-6 code in `cmd_codes` that produces
    the *largest* disruption when the best k-exporter set is removed.

    Returns a Polars DataFrame with columns:
        year | metric | k | loss | cmdCode | countries
    where `countries` is "ISO1, ISO2, …" (comma-separated).
    """
    rows = []
    for yr in years:
        for metric in ("value", "qty"):
            for k in (1, 2, 3):
                best = {"loss": -1}
                for code in cmd_codes:
                    try:
                        S, loss = max_disruption_set(
                            code, yr, k,
                            metric=metric,
                            importer=importer,
                            folder=folder,
                            template=template,
                        )
                        if loss > best["loss"]:
                            best = {"loss": loss,
                                    "cmd": code,
                                    "countries": ", ".join(S)}
                    except (FileNotFoundError, ValueError):
                        # skip missing files or no data for that code/year
                        continue
                if best["loss"] >= 0:
                    rows.append({"year": yr,
                                 "metric": metric,
                                 "k": k,
                                 "loss": best["loss"],
                                 "cmdCode": best["cmd"],
                                 "countries": best["countries"]})
    return pl.DataFrame(rows).sort(["year", "metric", "k"])


# ─── Example --------------------------------------------------------------
# codes_of_interest = {282200, 810590, 811299}
# df = max_disruption_across_codes(codes_of_interest, importer="USA")
# display(df.to_pandas())
```


```python
codes_of_interest = {280530, 284690, 850511, 720299}
df = max_disruption_across_codes(codes_of_interest, importer="USA")
display(df.to_pandas())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>cmdCode</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>59.024332</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>82.404753</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>96.701798</td>
      <td>280530</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>46.035253</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>70.224233</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>83.371395</td>
      <td>280530</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>75.683746</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>86.896844</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>93.191616</td>
      <td>280530</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>46.273702</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>64.841697</td>
      <td>280530</td>
      <td>CHN, RUS</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>81.841433</td>
      <td>280530</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>62.086862</td>
      <td>284690</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>74.873297</td>
      <td>284690</td>
      <td>CHN, MYS</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>83.228076</td>
      <td>284690</td>
      <td>CHN, MYS, R4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>47.462872</td>
      <td>284690</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>66.097164</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>81.109126</td>
      <td>280530</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>81.979921</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>91.367510</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>94.501324</td>
      <td>280530</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>56.782958</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>74.731963</td>
      <td>280530</td>
      <td>CHN, RUS</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>87.284671</td>
      <td>280530</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>84.654999</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>91.471400</td>
      <td>280530</td>
      <td>CHN, EUR</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>96.332250</td>
      <td>280530</td>
      <td>CHN, EUR, GBR</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>55.223164</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>84.000554</td>
      <td>280530</td>
      <td>CHN, RUS</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>94.819295</td>
      <td>280530</td>
      <td>CHN, GBR, RUS</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>64.983596</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>89.098496</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>99.338405</td>
      <td>280530</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>64.745651</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>92.302091</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>95.927329</td>
      <td>280530</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>88.595644</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>96.862984</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>99.237071</td>
      <td>280530</td>
      <td>CAN, CHN, GBR</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>70.543627</td>
      <td>280530</td>
      <td>CHN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>85.879285</td>
      <td>280530</td>
      <td>CHN, GBR</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>93.022353</td>
      <td>280530</td>
      <td>CHN, DEU, GBR</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>54.881546</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>75.393539</td>
      <td>720299</td>
      <td>BRA, CHN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>80.774520</td>
      <td>284690</td>
      <td>CHN, EST, ZAF</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>45.050006</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>72.250108</td>
      <td>280530</td>
      <td>CAN, GBR</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>93.164083</td>
      <td>280530</td>
      <td>CAN, CHN, GBR</td>
    </tr>
  </tbody>
</table>
</div>



```python
codes_of_interest = {280530, 284690, 850511, 720299}
df = max_disruption_across_codes(codes_of_interest, importer="CHN")
display(df.to_pandas())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>cmdCode</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>54.323473</td>
      <td>280530</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>91.209468</td>
      <td>284690</td>
      <td>MYS, R4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>93.966792</td>
      <td>284690</td>
      <td>EUR, MYS, R4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>54.390159</td>
      <td>280530</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>87.717036</td>
      <td>284690</td>
      <td>MYS, R4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>90.666307</td>
      <td>284690</td>
      <td>EUR, MYS, R4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>81.561507</td>
      <td>280530</td>
      <td>BDI</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>89.618431</td>
      <td>280530</td>
      <td>BDI, R4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>93.853954</td>
      <td>280530</td>
      <td>BDI, ESP, R4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>70.917034</td>
      <td>280530</td>
      <td>BDI</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>81.556191</td>
      <td>280530</td>
      <td>BDI, ESP</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>87.266518</td>
      <td>280530</td>
      <td>BDI, ESP, KOR</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>63.918571</td>
      <td>280530</td>
      <td>R4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>95.876033</td>
      <td>280530</td>
      <td>PHL, R4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>96.591011</td>
      <td>280530</td>
      <td>KOR, PHL, R4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>43.081320</td>
      <td>280530</td>
      <td>KOR</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>57.092755</td>
      <td>280530</td>
      <td>KOR, R4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>66.994537</td>
      <td>280530</td>
      <td>ESP, KOR, R4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>58.961950</td>
      <td>280530</td>
      <td>THA</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>79.610569</td>
      <td>284690</td>
      <td>MMR, USA</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>88.291783</td>
      <td>284690</td>
      <td>MMR, MYS, USA</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>78.288082</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>83.956420</td>
      <td>720299</td>
      <td>BRA, JPN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>87.741192</td>
      <td>720299</td>
      <td>BRA, JPN, R4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>72.687934</td>
      <td>280530</td>
      <td>VNM</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>89.046903</td>
      <td>284690</td>
      <td>MMR, USA</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>93.931146</td>
      <td>284690</td>
      <td>MMR, MYS, USA</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>70.095149</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>85.722592</td>
      <td>720299</td>
      <td>BRA, COD</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>89.632039</td>
      <td>720299</td>
      <td>BRA, COD, JPN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>72.616688</td>
      <td>284690</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>83.668607</td>
      <td>284690</td>
      <td>MMR, USA</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>90.750670</td>
      <td>284690</td>
      <td>MMR, MYS, USA</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>73.979433</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>80.572752</td>
      <td>720299</td>
      <td>BRA, COD</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>86.117557</td>
      <td>720299</td>
      <td>BRA, COD, R4</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>66.894712</td>
      <td>280530</td>
      <td>LAO</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>79.185224</td>
      <td>280530</td>
      <td>LAO, VNM</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>93.717462</td>
      <td>720299</td>
      <td>BRA, IDN, ZAF</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>75.692845</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>91.553896</td>
      <td>720299</td>
      <td>BRA, IDN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>93.788390</td>
      <td>720299</td>
      <td>BRA, IDN, ZAF</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>96.260502</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>99.584664</td>
      <td>280530</td>
      <td>KOR, THA</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>99.784694</td>
      <td>280530</td>
      <td>JPN, KOR, THA</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>97.446948</td>
      <td>720299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>98.011269</td>
      <td>720299</td>
      <td>BRA, ZAF</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>99.826190</td>
      <td>280530</td>
      <td>KOR, THA, USA</td>
    </tr>
  </tbody>
</table>
</div>



```python
codes_of_interest = {280530, 284690, 850511, 720299}
df = max_disruption_across_codes(codes_of_interest, importer=None)
display(df.to_pandas())
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[18], line 2
          1 codes_of_interest = {280530, 284690, 850511, 720299}
    ----> 2 df = max_disruption_across_codes(codes_of_interest, importer=None)
          3 display(df.to_pandas())


    Cell In[16], line 30, in max_disruption_across_codes(cmd_codes, importer, years, folder, template)
         28 for code in cmd_codes:
         29     try:
    ---> 30         S, loss = max_disruption_set(
         31             code, yr, k,
         32             metric=metric,
         33             importer=importer,
         34             folder=folder,
         35             template=template,
         36         )
         37         if loss > best["loss"]:
         38             best = {"loss": loss,
         39                     "cmd": code,
         40                     "countries": ", ".join(S)}


    Cell In[6], line 90, in max_disruption_set(cmd_code, year, n, metric, importer, folder, template)
         88 for combo in itertools.combinations(candidates, n):
         89     combo_set: Set[str] = set(combo)
    ---> 90     removed_weight = sum(
         91         w for u, v, w in edges if u in combo_set or v in combo_set
         92     )
         93     loss_pct = removed_weight / baseline_total * 100.0
         94     if loss_pct > best_loss:


    Cell In[6], line 91, in <genexpr>(.0)
         88 for combo in itertools.combinations(candidates, n):
         89     combo_set: Set[str] = set(combo)
         90     removed_weight = sum(
    ---> 91         w for u, v, w in edges if u in combo_set or v in combo_set
         92     )
         93     loss_pct = removed_weight / baseline_total * 100.0
         94     if loss_pct > best_loss:


    KeyboardInterrupt: 



```python
plot_top3_disruption_grouped(811299, importer="USA")
plot_top3_disruption_grouped(285000, importer="USA")
plot_top3_disruption_grouped(381800, importer="USA")
```


    
![png](output_31_0.png)
    



    
![png](output_31_1.png)
    



    
![png](output_31_2.png)
    



```python
plot_top3_disruption_grouped(811299, importer="CHN")
plot_top3_disruption_grouped(285000, importer="CHN")
plot_top3_disruption_grouped(381800, importer="CHN")
```


    
![png](output_32_0.png)
    



    
![png](output_32_1.png)
    



    
![png](output_32_2.png)
    



```python
import json, os, itertools, matplotlib as mpl
from pathlib import Path

# -----------------------------------------------------------------------
# 1️⃣ location for the persistent mapping file
# -----------------------------------------------------------------------
_COLOR_FILE = Path("iso2color.json")     # pick any writable location

# -----------------------------------------------------------------------
# 2️⃣ base palette (≈60 distinct hues) and cycle iterator
# -----------------------------------------------------------------------
_BASE_PALETTE = (
      list(mpl.colormaps["tab20"].colors)
    + list(mpl.colormaps["tab20b"].colors)
    + list(mpl.colormaps["tab20c"].colors)
)
_palette_cycle = itertools.cycle(_BASE_PALETTE)

# -----------------------------------------------------------------------
# 3️⃣ load existing mapping or start fresh
# -----------------------------------------------------------------------
if _COLOR_FILE.exists():
    with open(_COLOR_FILE, "r") as f:
        _ISO2COLOR = {k: tuple(v) for k, v in json.load(f).items()}
else:
    _ISO2COLOR = {}

# -----------------------------------------------------------------------
# 4️⃣ hard-wire CHN / USA (over-write if file had something else)
# -----------------------------------------------------------------------
_ISO2COLOR["CHN"] = mpl.colors.to_rgb("darkred")
_ISO2COLOR["USA"] = mpl.colors.to_rgb("darkblue")

# -----------------------------------------------------------------------
# 5️⃣ helper: find next unused palette colour
# -----------------------------------------------------------------------
def _next_unused_colour():
    for col in _palette_cycle:
        if col not in _ISO2COLOR.values():
            return col
    raise RuntimeError("Ran out of palette colours.")

# -----------------------------------------------------------------------
# 6️⃣ public accessor
# -----------------------------------------------------------------------
def _color_for_iso(iso: str):
    """
    Stable ISO-3 → RGB mapping that persists across notebook restarts.
    Adds a new entry to iso2color.json the first time an unseen country appears.
    """
    iso = iso.upper()
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = _next_unused_colour()
        # ---- write back to disk immediately ----------------------------
        with open(_COLOR_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _ISO2COLOR.items()}, f,
                      indent=2)
    return _ISO2COLOR[iso]
```


```python
pwd
```




    '/sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI'




```python
import json, os, itertools, matplotlib as mpl
from pathlib import Path

# -----------------------------------------------------------------------
# 1️⃣ location for the persistent mapping file  (your DPI directory)
# -----------------------------------------------------------------------
_BASE_DIR   = Path("/sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI")
_BASE_DIR.mkdir(parents=True, exist_ok=True)     # create folder if needed
_COLOR_FILE = _BASE_DIR / "iso2color.json"       # full path to the file

# -----------------------------------------------------------------------
# the rest of the block is identical
# -----------------------------------------------------------------------
_BASE_PALETTE = (
      list(mpl.colormaps["tab20"].colors)
    + list(mpl.colormaps["tab20b"].colors)
    + list(mpl.colormaps["tab20c"].colors)
)
_palette_cycle = itertools.cycle(_BASE_PALETTE)

if _COLOR_FILE.exists():
    with open(_COLOR_FILE, "r") as f:
        _ISO2COLOR = {k: tuple(v) for k, v in json.load(f).items()}
else:
    _ISO2COLOR = {}

_ISO2COLOR["CHN"] = mpl.colors.to_rgb("darkred")
_ISO2COLOR["USA"] = mpl.colors.to_rgb("darkblue")

def _next_unused_colour():
    for col in _palette_cycle:
        if col not in _ISO2COLOR.values():
            return col
    raise RuntimeError("Ran out of palette colours.")

def _color_for_iso(iso: str):
    iso = iso.upper()
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = _next_unused_colour()
        with open(_COLOR_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _ISO2COLOR.items()}, f, indent=2)
    return _ISO2COLOR[iso]
```


```python
pwd
```




    '/sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI'




```python
!ls -l /sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI
```

    total 258545
    -rwxrwx---+  1 vyb3yf bii_nssac   5703801 Jan 14 02:56  2804.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac   5025987 May 28 12:06  2811.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    500619 Mar  3 09:38  8107.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac   1174219 Jan 14 02:38  8110.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac  18705772 Jan 14 02:21  8541.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac  14335004 Jan 14 02:38  8542.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac     58655 Apr  6 14:40  aluminum_imports_quantity.png
    -rwxrwx---+  1 vyb3yf bii_nssac     56404 Apr  6 14:40  aluminum_imports_value.png
    drwxrws---+  3 vyb3yf bii_nssac      4096 Dec 20 02:28  codes
    -rwxrwx---+  1 vyb3yf bii_nssac   1750838 Nov 26  2024  commodityCodes.json
    -rwxrwx---+  1 vyb3yf bii_nssac    694030 Mar  3 01:52  componentElasticityPlots.ipynb
    drwxrws---+  3 vyb3yf bii_nssac      4096 Dec 30 06:11 'Component Materials'
    -rwxrwxrwx+  1 asv9v  bii_nssac     12684 May 28 12:06  comtrade-analysis.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac   1694834 Jun  4 02:06  comtrade-network-analysis-a.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac   1247849 May 28 12:08  Co_Nd_Dy_ResiliencyAnalysis.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac     39247 Dec 11 04:49  countryPosition.json
    -rwxrwx---+  1 vyb3yf bii_nssac    256359 Dec 11 06:26  f_df.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac   2588055 Jun  9 15:38  flow.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac   1003546 Jun  3 13:23  GaN_InP_NetowrkResilience.ipynb
    -rwxrwx---+  1 asv9v  bii_nssac     28252 Feb 13 11:26  GSDB-analysis.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac 175755486 Dec 18 02:46  HS2804_Exports_2018.html
    -rwxrwx---+  1 vyb3yf bii_nssac   2732612 Dec 30 04:23  HS2804_Exports_2023.html
    -rwxrwx---+  1 vyb3yf bii_nssac   2367727 Dec 30 06:27  HS2811_Exports_2023.html
    -rwxrwx---+  1 vyb3yf bii_nssac      2768 Dec 30 06:09  HS8107_Exports_2023.html
    -rwxrwx---+  1 vyb3yf bii_nssac    352472 Dec 30 05:42  HS8110_Exports_2023.html
    -rwxrwx---+  1 vyb3yf bii_nssac   9623228 Dec 30 04:50  HS8541_Exports_2023.html
    -rwxrwx---+  1 vyb3yf bii_nssac   7359290 Dec 30 05:15  HS8542_Exports_2023.html
    -rwxrwx---+  1 vyb3yf bii_nssac    205337 May 28 12:06  InitialAluminumAnalysis.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    390076 Apr 24 16:05  InitialChinaUSAluminumTrade.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    832592 Mar 31 07:28  InitialChinaUSSteelHSDeepDive.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    590372 Mar 31 07:57  InitialSteelAnalysis.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    808250 Mar 31 08:33  InitialUSChinaSteelTradeViz.ipynb
    -rwxrwx---+  1 asv9v  bii_nssac        76 Feb 11 18:59  links.txt
    -rwxrwx---+  1 vyb3yf bii_nssac      7772 Dec 11 06:34  mapVisualization.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    302833 May 14 12:56  marketShares.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac     20739 Dec 11 03:54  medicalMarketShares.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    771293 Apr  7 17:47  MiscChinese232RetaltiatoryTariffs.ipynb
    -rwxrwx---+  1 asv9v  bii_nssac     26157 Dec 18 15:20  misc.ipynb
    -rwxrwxrwx+  1 asv9v  bii_nssac      4400 Nov 22  2024  network.py
    -rwxrwxrwx+  1 asv9v  bii_nssac       426 Nov 20  2024  network.slurm
    drwxrwsrwx+  6 asv9v  bii_nssac      4096 Jun  4 04:26  out
    drwxrws---+ 14 vyb3yf bii_nssac      4096 Dec 20 08:27  pickles
    -rwxrwx---+  1 vyb3yf bii_nssac      3827 Feb 16 18:34  quantityNetwork.py
    -rwxrwxrwx+  1 asv9v  bii_nssac         0 Nov 20  2024  slurm-47874.out
    -rwxrwx---+  1 vyb3yf bii_nssac     60520 Apr  6 14:38  steel_imports_quantity.png
    -rwxrwx---+  1 vyb3yf bii_nssac     63747 Apr  6 14:38  steel_imports_value.png
    -rwxrwx---+  1 vyb3yf bii_nssac   2717185 Mar 28 21:50  steelTariffsChinaUSA.ipynb
    drwxrwsrwx+  3 asv9v  bii_nssac      4096 Dec 16 23:14  tmp
    -rwxrwx---+  1 vyb3yf bii_nssac      1141 Jan 28 02:46  tradeE_8110_156.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac      1042 Feb  1 21:31  tradeE_8110_392.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac      1042 Feb  2 02:48  tradeE_8110_704.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac      1042 Feb  1 23:03  tradeE_8110_762.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac      1042 Feb  2 01:00  tradeE_8110_842.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac      1287 Jan 27 17:21  tradeE_8110.pkl
    -rwxrwx---+  1 vyb3yf bii_nssac   1027395 Feb 28 13:44  tradeElasticity8110.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    280688 Feb  1 23:22  tradeElasticity.ipynb
    drwxrwsrwx+  4 asv9v  bii_nssac      4096 Nov 26  2024  UN_Comtrade
    drwxrws---+  2 vyb3yf bii_nssac      4096 Dec 17 23:07 'Untitled Folder'
    -rwxrwx---+  1 vyb3yf bii_nssac   3010270 Dec 11 06:32  USA_HS2850_Exports.html
    -rwxrwx---+  1 vyb3yf bii_nssac    171693 Nov 24  2024  visualization.ipynb
    -rwxrwx---+  1 vyb3yf bii_nssac    208316 Nov 26  2024  visualizationMedical.ipynb
    -rwxrwxrwx+  1 asv9v  bii_nssac       607 Nov 12  2024  x



```python
# use a dummy ISO code you know will not appear in real data
_ = _color_for_iso("ZZZ")        # triggers a new entry and writes the file

print("Colour assigned to ZZZ:", _)
print("Mapping file lives at :", _COLOR_FILE)
```

    Colour assigned to ZZZ: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    Mapping file lives at : /sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI/iso2color.json



```python
codes_of_interest = {811299, 285000, 381800}
df = max_disruption_across_codes(codes_of_interest, importer="USA")
display(df.to_pandas())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>cmdCode</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>53.662042</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>73.779314</td>
      <td>811299</td>
      <td>BRA, RUS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>83.835773</td>
      <td>811299</td>
      <td>BRA, EUR, RUS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>46.019847</td>
      <td>285000</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>75.115054</td>
      <td>285000</td>
      <td>DEU, EUR</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>84.837177</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>54.493143</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>73.024581</td>
      <td>285000</td>
      <td>CHN, DOM</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>85.592420</td>
      <td>285000</td>
      <td>CHN, DEU, DOM</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>44.766672</td>
      <td>285000</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>71.220523</td>
      <td>285000</td>
      <td>DEU, EUR</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>85.053676</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>61.070595</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>74.563977</td>
      <td>811299</td>
      <td>BRA, RUS</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>81.208904</td>
      <td>811299</td>
      <td>BRA, EST, RUS</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>39.848809</td>
      <td>285000</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>65.297426</td>
      <td>285000</td>
      <td>DEU, EUR</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>76.133778</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>54.723112</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>68.122983</td>
      <td>285000</td>
      <td>CHN, DEU</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>77.951563</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>43.554602</td>
      <td>285000</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>68.066650</td>
      <td>285000</td>
      <td>DEU, EUR</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>77.711258</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>73.588640</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>82.670543</td>
      <td>811299</td>
      <td>BRA, EST</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>86.619128</td>
      <td>811299</td>
      <td>BRA, EST, EUR</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>38.448983</td>
      <td>285000</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>63.246141</td>
      <td>285000</td>
      <td>DEU, EUR</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>73.063236</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>76.585921</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>88.892001</td>
      <td>811299</td>
      <td>BRA, EST</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>92.917234</td>
      <td>811299</td>
      <td>BRA, EST, EUR</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>47.314148</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>63.609411</td>
      <td>811299</td>
      <td>BRA, EST</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>73.667985</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>71.786151</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>86.893353</td>
      <td>811299</td>
      <td>BRA, CHN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>91.200073</td>
      <td>811299</td>
      <td>BRA, CHN, EST</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>46.366339</td>
      <td>285000</td>
      <td>DEU</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>72.975272</td>
      <td>285000</td>
      <td>DEU, EUR</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>80.072137</td>
      <td>285000</td>
      <td>CHN, DEU, EUR</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>94.754788</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>96.923955</td>
      <td>811299</td>
      <td>BRA, CHN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>97.931330</td>
      <td>811299</td>
      <td>BRA, CHN, MYS</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>65.844966</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>82.274370</td>
      <td>811299</td>
      <td>BRA, DEU</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>88.780559</td>
      <td>811299</td>
      <td>BRA, DEU, EUR</td>
    </tr>
  </tbody>
</table>
</div>



```python
codes_of_interest = {811299, 285000, 381800}
df = max_disruption_across_codes(codes_of_interest, importer="CHN")
display(df.to_pandas())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>metric</th>
      <th>k</th>
      <th>loss</th>
      <th>cmdCode</th>
      <th>countries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>qty</td>
      <td>1</td>
      <td>53.228697</td>
      <td>811299</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>qty</td>
      <td>2</td>
      <td>71.912410</td>
      <td>811299</td>
      <td>BRA, USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>qty</td>
      <td>3</td>
      <td>85.997966</td>
      <td>811299</td>
      <td>BRA, R4 , USA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>value</td>
      <td>1</td>
      <td>45.883864</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>value</td>
      <td>2</td>
      <td>68.970431</td>
      <td>285000</td>
      <td>DEU, USA</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>value</td>
      <td>3</td>
      <td>80.558453</td>
      <td>285000</td>
      <td>DEU, EUR, USA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>qty</td>
      <td>1</td>
      <td>86.127819</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018</td>
      <td>qty</td>
      <td>2</td>
      <td>89.791549</td>
      <td>285000</td>
      <td>DEU, USA</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018</td>
      <td>qty</td>
      <td>3</td>
      <td>92.953878</td>
      <td>285000</td>
      <td>DEU, EUR, USA</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018</td>
      <td>value</td>
      <td>1</td>
      <td>54.969019</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>value</td>
      <td>2</td>
      <td>69.947675</td>
      <td>285000</td>
      <td>DEU, USA</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>value</td>
      <td>3</td>
      <td>79.077184</td>
      <td>285000</td>
      <td>DEU, EUR, USA</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>qty</td>
      <td>1</td>
      <td>65.452391</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>qty</td>
      <td>2</td>
      <td>73.175943</td>
      <td>285000</td>
      <td>EUR, USA</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019</td>
      <td>qty</td>
      <td>3</td>
      <td>79.292237</td>
      <td>285000</td>
      <td>DEU, EUR, USA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019</td>
      <td>value</td>
      <td>1</td>
      <td>53.095944</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019</td>
      <td>value</td>
      <td>2</td>
      <td>66.903599</td>
      <td>285000</td>
      <td>DEU, USA</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019</td>
      <td>value</td>
      <td>3</td>
      <td>76.057980</td>
      <td>285000</td>
      <td>DEU, KOR, USA</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020</td>
      <td>qty</td>
      <td>1</td>
      <td>75.654632</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020</td>
      <td>qty</td>
      <td>2</td>
      <td>81.980969</td>
      <td>285000</td>
      <td>EUR, USA</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020</td>
      <td>qty</td>
      <td>3</td>
      <td>86.655227</td>
      <td>285000</td>
      <td>DEU, EUR, USA</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020</td>
      <td>value</td>
      <td>1</td>
      <td>41.044016</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020</td>
      <td>value</td>
      <td>2</td>
      <td>59.022238</td>
      <td>811299</td>
      <td>EST, USA</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>value</td>
      <td>3</td>
      <td>74.708843</td>
      <td>811299</td>
      <td>BRA, EST, USA</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>qty</td>
      <td>1</td>
      <td>61.008973</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021</td>
      <td>qty</td>
      <td>2</td>
      <td>73.632201</td>
      <td>811299</td>
      <td>BRA, EST</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2021</td>
      <td>qty</td>
      <td>3</td>
      <td>86.009958</td>
      <td>811299</td>
      <td>BRA, EST, EUR</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021</td>
      <td>value</td>
      <td>1</td>
      <td>43.447599</td>
      <td>811299</td>
      <td>EST</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021</td>
      <td>value</td>
      <td>2</td>
      <td>62.079965</td>
      <td>811299</td>
      <td>EST, USA</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2021</td>
      <td>value</td>
      <td>3</td>
      <td>80.546548</td>
      <td>811299</td>
      <td>BRA, EST, USA</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2022</td>
      <td>qty</td>
      <td>1</td>
      <td>60.962112</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2022</td>
      <td>qty</td>
      <td>2</td>
      <td>79.367668</td>
      <td>811299</td>
      <td>BRA, EST</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022</td>
      <td>qty</td>
      <td>3</td>
      <td>91.512978</td>
      <td>811299</td>
      <td>BRA, EST, EUR</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022</td>
      <td>value</td>
      <td>1</td>
      <td>49.387890</td>
      <td>811299</td>
      <td>EST</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022</td>
      <td>value</td>
      <td>2</td>
      <td>72.980099</td>
      <td>811299</td>
      <td>BRA, EST</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022</td>
      <td>value</td>
      <td>3</td>
      <td>85.801836</td>
      <td>811299</td>
      <td>BRA, EST, EUR</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2023</td>
      <td>qty</td>
      <td>1</td>
      <td>72.474927</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2023</td>
      <td>qty</td>
      <td>2</td>
      <td>79.232014</td>
      <td>285000</td>
      <td>JPN, USA</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2023</td>
      <td>qty</td>
      <td>3</td>
      <td>85.107019</td>
      <td>285000</td>
      <td>EUR, JPN, USA</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2023</td>
      <td>value</td>
      <td>1</td>
      <td>38.829273</td>
      <td>285000</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2023</td>
      <td>value</td>
      <td>2</td>
      <td>58.713431</td>
      <td>811299</td>
      <td>BRA, SGP</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2023</td>
      <td>value</td>
      <td>3</td>
      <td>74.471472</td>
      <td>811299</td>
      <td>BRA, R4 , SGP</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2024</td>
      <td>qty</td>
      <td>1</td>
      <td>76.167113</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2024</td>
      <td>qty</td>
      <td>2</td>
      <td>91.818740</td>
      <td>811299</td>
      <td>BRA, USA</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>qty</td>
      <td>3</td>
      <td>95.287860</td>
      <td>811299</td>
      <td>BRA, SGP, USA</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2024</td>
      <td>value</td>
      <td>1</td>
      <td>61.052052</td>
      <td>811299</td>
      <td>BRA</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2024</td>
      <td>value</td>
      <td>2</td>
      <td>76.947943</td>
      <td>811299</td>
      <td>BRA, USA</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2024</td>
      <td>value</td>
      <td>3</td>
      <td>88.985736</td>
      <td>811299</td>
      <td>BRA, SGP, USA</td>
    </tr>
  </tbody>
</table>
</div>



```python
###############################################################################
#  Persistent ISO-3 → colour mapping  (CHN = dark-red, USA = dark-blue)
###############################################################################
import json, itertools, matplotlib as mpl
from pathlib import Path
import polars as pl, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLOR_PATH  = Path("/sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI")
COLOR_PATH.mkdir(parents=True, exist_ok=True)
_COLOR_FILE = COLOR_PATH / "iso2color.json"

_BASE_PALETTE = (list(mpl.colormaps["tab20"].colors) +
                 list(mpl.colormaps["tab20b"].colors) +
                 list(mpl.colormaps["tab20c"].colors))
_palette_cycle = itertools.cycle(_BASE_PALETTE)

if _COLOR_FILE.exists():
    with open(_COLOR_FILE, "r") as f:
        _ISO2COLOR = {k: tuple(v) for k, v in json.load(f).items()}
else:
    _ISO2COLOR = {}

_ISO2COLOR["CHN"] = mpl.colors.to_rgb("darkred")
_ISO2COLOR["USA"] = mpl.colors.to_rgb("blue")

def _next_unused_colour():
    for col in _palette_cycle:
        if col not in _ISO2COLOR.values():
            return col
    raise RuntimeError("palette exhausted")

def _color_for_iso(iso: str):
    iso = iso.upper()
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = _next_unused_colour()
        with open(_COLOR_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _ISO2COLOR.items()}, f, indent=2)
    return _ISO2COLOR[iso]


###############################################################################
#  Plot helper – choose max-loss commodity from a set of codes
###############################################################################
mpl.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 15, "legend.title_fontsize": 17,
    "hatch.linewidth": 1.5,
})

def plot_max_disruption_from_codes(
    cmd_codes,
    *,
    k: int = 3,
    importer: str | None = None,        # None ⇒ global
    years = range(2017, 2025),
    folder   = "out",
    template = "flow_transformed_{yr}.txt",
):
    """
    For each year pick the HS-6 in `cmd_codes` whose best k-exporter removal
    maximises disruption; show value & quantity bars side-by-side.
    """
    cmd_codes = list(cmd_codes)
    if k < 1:
        raise ValueError("k must be ≥1")

    # -------- convenience title tag -------------------------------------
    ND_DY_SET = {280530, 284690, 850511, 720299}
    GA_IN_SET = {811299, 285000, 381800}
    title_tag = ""
    if set(cmd_codes) == ND_DY_SET:
        title_tag = "(Nd/Dy)"
    if set(cmd_codes) == GA_IN_SET:
        title_tag = "(Ga/In)"

    # -------- helper : get top-k pieces for a code ----------------------
    def pieces_for(code, yr, metric):
        fp = Path(folder) / template.format(yr=yr)
        if not fp.exists():
            return None
        df = (
            pl.read_csv(fp, separator="\t",
                        columns=["importer","exporter","cmdCode",metric],
                        infer_schema_length=1000)
              .filter(
                  (pl.col("cmdCode")==code) &
                  ((pl.col("importer")==importer) if importer else True) &
                  (pl.col("exporter")!="W00") &
                  (pl.col("importer")!="W00") &
                  (pl.col("exporter")!=pl.col("importer"))
              )
        )
        if df.is_empty():
            return None
        total = df[metric].sum()
        topk  = (df.group_by("exporter")
                   .agg(pl.col(metric).sum().alias("flow"))
                   .sort("flow", descending=True)
                   .head(k))
        return [(r["exporter"], r["flow"]/total*100)
                for r in topk.iter_rows(named=True)]

    # -------- pick code with max loss per year & metric ------------------
    rows_v, rows_q = [], []
    for yr in years:
        best_v = {"loss":-1}; best_q = {"loss":-1}
        for code in cmd_codes:
            pcs_v = pieces_for(code, yr, "primaryValue")
            pcs_q = pieces_for(code, yr, "qty")
            if pcs_v:
                lv = sum(h for _,h in pcs_v)
                if lv > best_v["loss"]:
                    best_v = {"loss": lv, "cmd": code, "pieces": pcs_v}
            if pcs_q:
                lq = sum(h for _,h in pcs_q)
                if lq > best_q["loss"]:
                    best_q = {"loss": lq, "cmd": code, "pieces": pcs_q}
        if best_v["loss"]>=0: rows_v.append({"year":yr, **best_v})
        if best_q["loss"]>=0: rows_q.append({"year":yr, **best_q})

    if not rows_v or not rows_q:
        print("No data."); return

    # colours
    used = {iso for r in rows_v+rows_q for iso,_ in r["pieces"]}
    colours = {iso:_color_for_iso(iso) for iso in used}

    yrs = sorted({r["year"] for r in rows_v})
    x   = np.arange(len(yrs)); w=.35
    fig, ax = plt.subplots(figsize=(14,6)); fig.subplots_adjust(right=.68)
    bv={y:0 for y in yrs}; bq={y:0 for y in yrs}

    def stack(rows, offset, bottoms, hatch=False):
        for tier in range(k):
            for r in rows:
                if tier < len(r["pieces"]):
                    iso,h = r["pieces"][tier]
                    ax.bar(x[yrs.index(r["year"])] + offset, h, w,
                           bottom=bottoms[r["year"]],
                           color=colours[iso], edgecolor="black", lw=.3,
                           hatch="//" if hatch else "")
                    bottoms[r["year"]] += h
    stack(rows_v,-w/2,bv,False)
    stack(rows_q, w/2,bq,True)

    ax.set_xticks(x, yrs); ax.set_ylim(0,100)
    scope = f"{importer} Imports" if importer else "Global"
    ax.set_title(f"Max-Disruption Across Bucket {title_tag} – {scope}".strip())
    ax.set_ylabel("Flow lost (%)"); ax.grid(axis="y", ls="--", alpha=.4)

    # legends
    exp_handles=[Line2D([0],[0],color=colours[c],lw=6) for c in sorted(colours)]
    exp_leg=ax.legend(exp_handles,sorted(colours),title="Exporter",
                      bbox_to_anchor=(1.02,1.0),loc="upper left",frameon=False)
    ax.add_artist(exp_leg)
    ax.legend([Patch(fc="black"),
               Patch(fc="white",ec="black",hatch="//")],
              ["Value","Quantity"], title="Metric",
              bbox_to_anchor=(1.02,0.20),loc="upper left",frameon=False)
    plt.tight_layout(); plt.show()

    # table
    tbl=[]
    for lbl,rows in [("Value",rows_v),("Quantity",rows_q)]:
        for r in rows:
            tbl.append({"year":r["year"],"metric":lbl,
                        "cmdCode":r["cmd"],
                        "loss":f"{r['loss']:.1f} %",
                        "countries":", ".join(i for i,_ in r["pieces"])})
    display(pl.DataFrame(tbl).sort(["year","metric"]))
```


```python
codes = {280530,284690,850511,720299}
plot_max_disruption_from_codes(codes, k=1, importer="USA")
```


    
![png](output_42_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;59.0 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;46.0 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;75.7 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;46.3 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>284690</td><td>&quot;62.1 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;64.7 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;88.6 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;70.5 %&quot;</td><td>&quot;CHN&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>720299</td><td>&quot;54.9 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;45.1 %&quot;</td><td>&quot;BRA&quot;</td></tr></tbody></table></div>



```python
codes = {280530,284690,850511,720299}
plot_max_disruption_from_codes(codes, k=2, importer="USA")
```


    
![png](output_43_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;82.4 %&quot;</td><td>&quot;CHN, GBR&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;70.2 %&quot;</td><td>&quot;CHN, GBR&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;86.9 %&quot;</td><td>&quot;CHN, GBR&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;64.8 %&quot;</td><td>&quot;CHN, RUS&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>284690</td><td>&quot;74.9 %&quot;</td><td>&quot;CHN, MYS&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;92.3 %&quot;</td><td>&quot;CHN, GBR&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;96.9 %&quot;</td><td>&quot;CHN, GBR&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;85.9 %&quot;</td><td>&quot;CHN, GBR&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>720299</td><td>&quot;75.4 %&quot;</td><td>&quot;BRA, CHN&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;72.3 %&quot;</td><td>&quot;GBR, CAN&quot;</td></tr></tbody></table></div>



```python
codes = {280530,284690,850511,720299}
plot_max_disruption_from_codes(codes, k=3, importer="USA")
```


    
![png](output_44_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;96.7 %&quot;</td><td>&quot;CHN, GBR, EUR&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;83.4 %&quot;</td><td>&quot;CHN, GBR, EUR&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;93.2 %&quot;</td><td>&quot;CHN, GBR, EUR&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;81.8 %&quot;</td><td>&quot;CHN, RUS, GBR&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>284690</td><td>&quot;83.2 %&quot;</td><td>&quot;CHN, MYS, R4 &quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;95.9 %&quot;</td><td>&quot;CHN, GBR, CAN&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;99.2 %&quot;</td><td>&quot;CHN, GBR, CAN&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;93.0 %&quot;</td><td>&quot;CHN, GBR, DEU&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>284690</td><td>&quot;80.8 %&quot;</td><td>&quot;CHN, EST, ZAF&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;93.2 %&quot;</td><td>&quot;GBR, CAN, CHN&quot;</td></tr></tbody></table></div>



```python
codes = {280530,284690,850511,720299}
plot_max_disruption_from_codes(codes, k=1, importer="CHN")
```


    
![png](output_45_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;58.6 %&quot;</td><td>&quot;VNM&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;58.7 %&quot;</td><td>&quot;VNM&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;81.6 %&quot;</td><td>&quot;BDI&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;71.0 %&quot;</td><td>&quot;BDI&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;65.3 %&quot;</td><td>&quot;R4 &quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;74.6 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;66.9 %&quot;</td><td>&quot;LAO&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;75.7 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>720299</td><td>&quot;96.3 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;97.4 %&quot;</td><td>&quot;BRA&quot;</td></tr></tbody></table></div>



```python
codes = {280530,284690,850511,720299}
plot_max_disruption_from_codes(codes, k=2, importer="CHN")
```


    
![png](output_46_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>284690</td><td>&quot;91.3 %&quot;</td><td>&quot;R4 , MYS&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>284690</td><td>&quot;87.8 %&quot;</td><td>&quot;MYS, R4 &quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;89.6 %&quot;</td><td>&quot;BDI, R4 &quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;81.6 %&quot;</td><td>&quot;BDI, ESP&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;97.9 %&quot;</td><td>&quot;R4 , PHL&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;81.3 %&quot;</td><td>&quot;BRA, COD&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;79.2 %&quot;</td><td>&quot;LAO, VNM&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;91.6 %&quot;</td><td>&quot;BRA, IDN&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;99.6 %&quot;</td><td>&quot;THA, KOR&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;98.0 %&quot;</td><td>&quot;BRA, ZAF&quot;</td></tr></tbody></table></div>



```python
codes = {280530,284690,850511,720299}
plot_max_disruption_from_codes(codes, k=3, importer="CHN")
```


    
![png](output_47_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>284690</td><td>&quot;94.0 %&quot;</td><td>&quot;R4 , MYS, EUR&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>284690</td><td>&quot;90.8 %&quot;</td><td>&quot;MYS, R4 , EUR&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;93.9 %&quot;</td><td>&quot;BDI, R4 , ESP&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;87.4 %&quot;</td><td>&quot;BDI, ESP, KOR&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;98.6 %&quot;</td><td>&quot;R4 , PHL, KOR&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;86.9 %&quot;</td><td>&quot;BRA, COD, R4 &quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>720299</td><td>&quot;93.7 %&quot;</td><td>&quot;IDN, BRA, ZAF&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>720299</td><td>&quot;93.8 %&quot;</td><td>&quot;BRA, IDN, ZAF&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>280530</td><td>&quot;99.8 %&quot;</td><td>&quot;THA, KOR, JPN&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>280530</td><td>&quot;99.8 %&quot;</td><td>&quot;THA, USA, KOR&quot;</td></tr></tbody></table></div>



```python
###############################################################################
# 1.  Persistent ISO-3 → colour mapping (saved to iso2color.json)
###############################################################################
import json, itertools, matplotlib as mpl
from pathlib import Path

_COLOR_FILE = Path("/sfs/gpfs/tardis/project/bii_nssac/people/anil/DPI/iso2color.json")
_COLOR_FILE.parent.mkdir(parents=True, exist_ok=True)

_BASE_PALETTE = (list(mpl.colormaps["tab20"].colors) +
                 list(mpl.colormaps["tab20b"].colors) +
                 list(mpl.colormaps["tab20c"].colors))
_palette_cycle = itertools.cycle(_BASE_PALETTE)

_ISO2COLOR = json.load(_COLOR_FILE.open()) if _COLOR_FILE.exists() else {}
_ISO2COLOR["CHN"] = mpl.colors.to_rgb("darkred")   # fixed colours
_ISO2COLOR["USA"] = mpl.colors.to_rgb("blue")

def _next_colour():
    for c in _palette_cycle:
        if c not in _ISO2COLOR.values():
            return c
    raise RuntimeError("exhausted palette")

def _color_for_iso(iso: str):
    iso = iso.upper()
    if iso not in _ISO2COLOR:
        _ISO2COLOR[iso] = _next_colour()
        with _COLOR_FILE.open("w") as f:
            json.dump({k: list(v) for k, v in _ISO2COLOR.items()}, f, indent=2)
    return _ISO2COLOR[iso]


###############################################################################
# 2.  Max-disruption plot: k = 1,2,3 bars per year (value or quantity)
###############################################################################
import polars as pl, numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 15, "legend.title_fontsize": 17,
    "hatch.linewidth": 1.5,
})

def plot_max_disruption_k123(
    cmd_codes,                          # iterable of HS-6 codes
    *,                                  # keyword-only after this
    importer: str | None = None,        # None ⇒ global  |  "USA" ⇒ USA-centric
    metric: str = "value",              # "value" | "qty"
    years = range(2017, 2025),
    folder="out",
    template="flow_transformed_{yr}.txt",
):
    metric = metric.lower()
    metric_col = {"value": "primaryValue", "qty": "qty"}[metric]
    metric_tag = "(Value)" if metric == "value" else "(Quantity)"

    cmd_codes = list(cmd_codes)
    ND_DY_SET = {280530, 284690, 850511, 720299}
    GA_IN_SET = {811299, 285000, 381800}
    title_tag = ""
    if set(cmd_codes) == ND_DY_SET:
        title_tag = "(Nd/Dy)"
    if set(cmd_codes) == GA_IN_SET:
        title_tag = "(Ga/In)"

    # ---------- helper: top-3 exporter pieces for one HS-6 / year ----------
    def pieces_for(code, yr):
        fp = Path(folder) / template.format(yr=yr)
        if not fp.exists():
            return None
        df = (pl.read_csv(fp, separator="\t",
                          columns=["importer","exporter","cmdCode",metric_col],
                          infer_schema_length=1000)
                .filter(
                    (pl.col("cmdCode") == code) &
                    ((pl.col("importer") == importer) if importer else True) &
                    (pl.col("exporter") != "W00") &
                    (pl.col("importer") != "W00") &
                    (pl.col("exporter") != pl.col("importer"))
                ))
        if df.is_empty():
            return None
        tot = df[metric_col].sum()
        top3 = (df.group_by("exporter")
                  .agg(pl.col(metric_col).sum().alias("flow"))
                  .sort("flow", descending=True)
                  .head(3))
        return [(r["exporter"], r["flow"] / tot * 100)
                for r in top3.iter_rows(named=True)]

    # ---------- for each year & k pick HS-6 with max loss ------------------
    rows = []
    for yr in years:
        for k in (1, 2, 3):
            best = {"loss": -1}
            for code in cmd_codes:
                pcs = pieces_for(code, yr)
                if pcs and len(pcs) >= k:
                    loss = sum(h for _, h in pcs[:k])
                    if loss > best["loss"]:
                        best = {"cmd": code, "loss": loss, "pieces": pcs[:k]}
            if best["loss"] >= 0:
                rows.append({"year": yr, "k": k, **best})
    if not rows:
        print("No data for requested parameters."); return

    # ---------- colour map -------------------------------------------------
    used_iso = {iso for r in rows for iso, _ in r["pieces"]}
    colours  = {iso: _color_for_iso(iso) for iso in used_iso}

    # ---------- plotting ---------------------------------------------------
    yrs = sorted({r["year"] for r in rows})
    x   = np.arange(len(yrs)); bar_w = 0.22
    fig, ax = plt.subplots(figsize=(14, 6)); fig.subplots_adjust(right=0.72)
    bottoms = {(yr, k): 0 for yr in yrs for k in (1, 2, 3)}

    for tier in range(3):                       # exporter tier 0/1/2
        for r in rows:
            if tier < len(r["pieces"]):
                iso, h = r["pieces"][tier]
                xpos = x[yrs.index(r["year"])] + (r["k"] - 2) * bar_w
                ax.bar(xpos, h, bar_w,
                       bottom=bottoms[(r["year"], r["k"])],
                       color=colours[iso], edgecolor="black", linewidth=1)
                bottoms[(r["year"], r["k"])] += h

    ax.set_xticks(x, yrs)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Flow lost (%)")
    scope = f"{importer} Imports" if importer else "Global"
    ax.set_title(f"Max-Disruption Across Bucket {title_tag} {metric_tag} – {scope}".strip())
    ax.grid(axis="y", ls="--", alpha=0.4)

    # exporter legend only
    exp_handles = [Line2D([0], [0], color=colours[c], lw=6) for c in sorted(colours)]
    ax.legend(exp_handles, sorted(colours), title="Exporter",
              bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)

    plt.tight_layout()
    plt.show()
    
    table = pd.DataFrame(
        [{
            "Year": r["year"],
            "k":    r["k"],
            "Loss %": f"{r['loss']:.1f}",
            "cmdCode": r["cmd"],
            "Exporters": ", ".join(iso for iso,_ in r["pieces"])
        } for r in rows]
    ).sort_values(["Year","k"]).reset_index(drop=True)

    display(table)
    return table
```


```python
codes = {280530, 284690, 850511, 720299}
plot_max_disruption_k123(codes, importer="USA", metric="value")
```


    
![png](output_49_0.png)
    



```python
codes = {280530, 284690, 850511, 720299}
plot_max_disruption_k123(codes, importer="USA", metric="qty")
```


    
![png](output_50_0.png)
    



```python
codes = {280530, 284690, 850511, 720299}
plot_max_disruption_k123(codes, importer="CHN", metric="value")
```


    
![png](output_51_0.png)
    



```python
codes = {280530, 284690, 850511, 720299}
plot_max_disruption_k123(codes, importer="CHN", metric="qty")
```


    
![png](output_52_0.png)
    



```python
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

def plot_max_disruption_map(
    cmd_code: int,
    importer_iso: str,
    *,
    metric: str = "value",           # "value" | "qty"
    years = range(2017, 2025),
    folder = "out",
    template = "flow_transformed_{yr}.txt",
):
    """
    World map: every exporter is shaded by the *maximum* disruption share
    (% of the importer’s inbound flow that would vanish in any one year).
    The importing country itself is highlighted in blue.
    """
    metric_col   = {"value": "primaryValue", "qty": "qty"}[metric.lower()]
    importer_iso = importer_iso.upper()

    # ── 1) gather max share for each exporter ────────────────────────────
    max_share = {}                # exporter ISO → max %
    for yr in years:
        fp = Path(folder) / template.format(yr=yr)
        if not fp.exists():
            continue
        df = (
            pl.read_csv(
                fp, separator="\t",
                columns=["importer","exporter","cmdCode",metric_col],
                infer_schema_length=1000,
            )
            .filter(
                (pl.col("cmdCode") == cmd_code) &
                (pl.col("importer")  == importer_iso) &
                (pl.col("exporter") != "W00") &
                (pl.col("exporter") != pl.col("importer"))
            )
        )
        if df.is_empty():
            continue
        total = df[metric_col].sum()
        shares = (
            df.group_by("exporter")
              .agg((pl.col(metric_col).sum() / total * 100).alias("share"))
        )
        for exp, sh in shares.iter_rows():
            max_share[exp] = max(max_share.get(exp, 0.0), sh)

    if not max_share:
        print("No data for given parameters."); return

    # ── 2) load world geometry (GeoDatasets key search + fallback) ───────
    try:
        import geodatasets as gd
        for key in ("ne_110m_admin_0_countries",
                    "naturalearth_countries",
                    "naturalearth_lowres"):
            try:
                world_path = gd.get_path(key)
                break
            except (ValueError, AttributeError):
                continue
        else:
            raise ImportError
    except Exception:
        cache = Path.home() / ".cache" / "ne_110m_countries.geojson"
        if not cache.exists():
            url = ("https://raw.githubusercontent.com/nvkelso/"
                   "natural-earth-vector/master/geojson/"
                   "ne_110m_admin_0_countries.geojson")
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, cache)
        world_path = cache

    world = gpd.read_file(world_path)

    # detect ISO column name
    for col in ("iso_a3", "ISO_A3", "adm0_a3", "ADM0_A3"):
        if col in world.columns:
            iso_col = col
            break
    else:
        raise ValueError("Could not find ISO-3 column in world layer")

    world["iso_code"] = world[iso_col].str.upper()
    world["share"]    = world["iso_code"].map(max_share).fillna(0.0)

    # ── 3) plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7))

    # base: thin country borders
    world.boundary.plot(ax=ax, linewidth=0.4, color="black")

    # exporters shaded by max-share
    world.plot(
        column="share",
        cmap="Reds",
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        legend=True,
        legend_kwds={"label": "Max disruption (%)", "shrink": 0.6},
        vmin=0,
        vmax=max(max_share.values()),
    )

    # highlight the importing country in blue (on top)
    imp_geom = world[world["iso_code"] == importer_iso]
    if not imp_geom.empty:
        imp_geom.plot(ax=ax, facecolor="blue", edgecolor="black", linewidth=0.8)

    ax.set_axis_off()
    title_metric = "Value" if metric.lower() == "value" else "Quantity"
    ax.set_title(
        f"Max Disruption Share {title_metric}\n"
        f"HS {cmd_code} – {importer_iso} Imports, {years.start}-{years.stop-1}",
        fontsize=18,
    )
    plt.tight_layout()
    plt.show()
```


```python
plot_max_disruption_map(
    cmd_code      = 381800,
    importer_iso  = "USA",
    metric        = "value",
)
```


    
![png](output_54_0.png)
    



```python
codes = {811299, 285000, 381800}
plot_max_disruption_from_codes(codes, k=1, importer="USA")
```


    
![png](output_55_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;53.7 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;46.0 %&quot;</td><td>&quot;DEU&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;54.5 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;44.8 %&quot;</td><td>&quot;DEU&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;61.1 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;47.3 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;71.8 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;46.4 %&quot;</td><td>&quot;DEU&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;94.8 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;65.8 %&quot;</td><td>&quot;BRA&quot;</td></tr></tbody></table></div>



```python
codes = {811299, 285000, 381800}
plot_max_disruption_from_codes(codes, k=2, importer="USA")
```


    
![png](output_56_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;73.8 %&quot;</td><td>&quot;BRA, RUS&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;75.1 %&quot;</td><td>&quot;DEU, EUR&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;73.0 %&quot;</td><td>&quot;CHN, DOM&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;71.2 %&quot;</td><td>&quot;DEU, EUR&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;74.6 %&quot;</td><td>&quot;BRA, RUS&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;63.6 %&quot;</td><td>&quot;BRA, EST&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;86.9 %&quot;</td><td>&quot;BRA, CHN&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;73.0 %&quot;</td><td>&quot;DEU, EUR&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;96.9 %&quot;</td><td>&quot;BRA, CHN&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;82.3 %&quot;</td><td>&quot;BRA, DEU&quot;</td></tr></tbody></table></div>



```python
codes = {811299, 285000, 381800}
plot_max_disruption_from_codes(codes, k=3, importer="USA")
```


    
![png](output_57_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;83.8 %&quot;</td><td>&quot;BRA, RUS, EUR&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;84.8 %&quot;</td><td>&quot;DEU, EUR, CHN&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;85.6 %&quot;</td><td>&quot;CHN, DOM, DEU&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;85.1 %&quot;</td><td>&quot;DEU, EUR, CHN&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;81.2 %&quot;</td><td>&quot;BRA, RUS, EST&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;73.7 %&quot;</td><td>&quot;DEU, EUR, CHN&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;91.2 %&quot;</td><td>&quot;BRA, CHN, EST&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;80.1 %&quot;</td><td>&quot;DEU, EUR, CHN&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;97.9 %&quot;</td><td>&quot;BRA, CHN, MYS&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;88.8 %&quot;</td><td>&quot;BRA, DEU, EUR&quot;</td></tr></tbody></table></div>



```python
codes = {811299, 285000, 381800}
plot_max_disruption_from_codes(codes, k=1, importer="CHN")
```


    
![png](output_58_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;53.2 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;46.0 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;86.2 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;55.1 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;65.5 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;49.4 %&quot;</td><td>&quot;EST&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;72.5 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;38.9 %&quot;</td><td>&quot;USA&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;76.2 %&quot;</td><td>&quot;BRA&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;61.1 %&quot;</td><td>&quot;BRA&quot;</td></tr></tbody></table></div>



```python
codes = {811299, 285000, 381800}
plot_max_disruption_from_codes(codes, k=2, importer="CHN")
```


    
![png](output_59_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;71.9 %&quot;</td><td>&quot;USA, BRA&quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;69.1 %&quot;</td><td>&quot;USA, DEU&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;89.9 %&quot;</td><td>&quot;USA, DEU&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;70.1 %&quot;</td><td>&quot;USA, DEU&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;73.2 %&quot;</td><td>&quot;USA, EUR&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;73.0 %&quot;</td><td>&quot;EST, BRA&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;79.3 %&quot;</td><td>&quot;USA, JPN&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;58.7 %&quot;</td><td>&quot;BRA, SGP&quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;91.8 %&quot;</td><td>&quot;BRA, USA&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;76.9 %&quot;</td><td>&quot;BRA, USA&quot;</td></tr></tbody></table></div>



```python
codes = {811299, 285000, 381800}
plot_max_disruption_from_codes(codes, k=3, importer="CHN")
```


    
![png](output_60_0.png)
    



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (16, 5)</small><table border="1" class="dataframe"><thead><tr><th>year</th><th>metric</th><th>cmdCode</th><th>loss</th><th>countries</th></tr><tr><td>i64</td><td>str</td><td>i64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>2017</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;86.0 %&quot;</td><td>&quot;USA, BRA, R4 &quot;</td></tr><tr><td>2017</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;80.7 %&quot;</td><td>&quot;USA, DEU, EUR&quot;</td></tr><tr><td>2018</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;93.0 %&quot;</td><td>&quot;USA, DEU, EUR&quot;</td></tr><tr><td>2018</td><td>&quot;Value&quot;</td><td>285000</td><td>&quot;79.2 %&quot;</td><td>&quot;USA, DEU, EUR&quot;</td></tr><tr><td>2019</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;79.3 %&quot;</td><td>&quot;USA, EUR, DEU&quot;</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>2022</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;85.8 %&quot;</td><td>&quot;EST, BRA, EUR&quot;</td></tr><tr><td>2023</td><td>&quot;Quantity&quot;</td><td>285000</td><td>&quot;85.1 %&quot;</td><td>&quot;USA, JPN, EUR&quot;</td></tr><tr><td>2023</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;74.5 %&quot;</td><td>&quot;BRA, SGP, R4 &quot;</td></tr><tr><td>2024</td><td>&quot;Quantity&quot;</td><td>811299</td><td>&quot;95.3 %&quot;</td><td>&quot;BRA, USA, SGP&quot;</td></tr><tr><td>2024</td><td>&quot;Value&quot;</td><td>811299</td><td>&quot;89.0 %&quot;</td><td>&quot;BRA, USA, SGP&quot;</td></tr></tbody></table></div>



```python
codes = {811299, 285000, 381800}
plot_max_disruption_k123(codes, importer="USA", metric="value")
```


    
![png](output_61_0.png)
    



```python
codes = {811299, 285000, 381800}
plot_max_disruption_k123(codes, importer="USA", metric="qty")
```


    
![png](output_62_0.png)
    



```python
codes = {811299, 285000, 381800}
plot_max_disruption_k123(codes, importer="CHN", metric="value")
```


    
![png](output_63_0.png)
    



```python
codes = {811299, 285000, 381800}
plot_max_disruption_k123(codes, importer="CHN", metric="qty")
```


    
![png](output_64_0.png)
    



```python

```
