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
