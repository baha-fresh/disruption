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
