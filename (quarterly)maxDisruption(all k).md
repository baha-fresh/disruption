```python
###############################################################################
# 1.  Persistent ISO‑3 → colour mapping (saved to iso2color.json)
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
_ISO2COLOR["CHN"] = mpl.colors.to_rgb("darkred")      # fixed colours
_ISO2COLOR["USA"] = mpl.colors.to_rgb("blue")
_ISO2COLOR["COL"] = mpl.colors.to_rgb("xkcd:mustard yellow")
_ISO2COLOR["ISR"] = mpl.colors.to_rgb("xkcd:light brown")

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
# 2.  Max‑disruption plot (k = 1,2,3) – QUARTERLY VERSION
###############################################################################
import polars as pl, numpy as np, matplotlib.pyplot as plt, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 15, "legend.title_fontsize": 17,
    "hatch.linewidth": 1.5,
})

def plot_max_disruption_k123_quarterly(
    cmd_codes,                           # iterable of HS‑6 codes
    *,                                   # keyword‑only after this
    importer: str | None = None,         # None ⇒ global, "USA" ⇒ USA‑centric
    metric: str = "value",               # "value" | "qty"
    quarters = None,                     # ["2020Q1","2020Q2", ...], None defaults to all, which is 2020Q1 - 2025Q1
    folder = "out",
    template = "q_transformed_{tag}.txt"
):
    """
    Quarterly analogue of the k‑disruption plot.
    For each quarter, choose the HS‑6 that maximises the k‑exporter loss.
    """

    # --------------------------- defaults ---------------------------------
    if quarters is None:
        quarters = []
        for yr in range(2020, 2026):
            for q in range(1, 5):
                if yr == 2025 and q > 1:       # stop at 2025Q1
                    break
                quarters.append(f"{yr}Q{q}")

    metric = metric.lower()
    metric_col = {"value": "primaryValue", "qty": "qty"}[metric]
    metric_tag = "(Value)" if metric == "value" else "(Quantity)"

    cmd_codes = list(cmd_codes)

    # Code to place a title tag in generated plot
    ND_DY_SET = {280530, 284690, 850511, 720299}
    GA_IN_SET = {811299, 285000, 381800}
    CO_SET    = {260500, 282200, 282739, 283329, 283699,291529, 810520, 810530, 810590}
    B_SET = {252800,281000,281290,282690,283990,284011,284019,284020,284520,284990,285000}
    GE_SET = {261790,281219,281290,282560,285000,811292}
    IR_SET = {261690,284390,381512,711019,711041,711049,711292}
    P_SET = {251010,251020,280470,280910,280920,281212,281213,281214,281390,283510,283522,283524,283525,283529,283531,283539,285390,291990,310319,310530,740500}
    SI_SET = {250510,280461,280469,281122,283911,283990,284920,285000,293190,391000,720221}
    GA_SET = {260600,260800,281219,282590,283329,283429,285000,285390,381800,811292,811299}
    REE_SET = {261220,261790,280530,284690}
    REE1_SET = {261220,261790,280530,284690,280530,284690,850511,720299}
    GA1_SET = {260600,260800,281219,282590,283329,283429,285000,285390,381800,811292,811299,811291}
    CE_SET = {261790,280530,284610,360690}
    RGAS_SET = {280429}
    TA_SET = {261590,282590,284990,285000,810320,810330,810391,810399}
    PRP_SET = {291469,292700,370191,390319,390530,390610,390690,390730,390940,990211}

    title_tag = ""
    if set(cmd_codes) == ND_DY_SET:
        title_tag = "(Nd/Dy)"
    if set(cmd_codes) == GA_IN_SET:
        title_tag = "(Ga/In)"
    if set(cmd_codes) == CO_SET:
        title_tag = "(Co)"          
    if set(cmd_codes) == B_SET:
        title_tag = "(B)"
    if set(cmd_codes) == GE_SET:
        title_tag = "(Ge)"
    if set(cmd_codes) == IR_SET:
        title_tag = "(Ir)"
    if set(cmd_codes) == P_SET:
        title_tag = "(P)"
    if set(cmd_codes) == SI_SET:
        title_tag = "(Si)"
    if set(cmd_codes) == GA_SET:
        title_tag = "(Ga)"
    if set(cmd_codes) == REE_SET:
        title_tag = "(REE)"
    if set(cmd_codes) == REE1_SET:
        title_tag = "(REE[a])"
    if set(cmd_codes) == GA1_SET:
        title_tag = "(Ga-GaN-InP)"
    if set(cmd_codes) == CE_SET:
        title_tag = "(Ce)"
    if set(cmd_codes) == RGAS_SET:
        title_tag = "(Rare Gases)"
    if set(cmd_codes) == TA_SET:
        title_tag = "(Ta)"
    if set(cmd_codes) == PRP_SET:
        title_tag = "(PR-polymers)"

    # ---------------- helper: top‑3 exporter shares for one HS‑6 / quarter
    #main logic for calculating max disruption
    def pieces_for(code, tag):
        from pathlib import Path
        fp = Path(folder) / template.format(tag=tag)
        if not fp.exists():
            return None
        df = (
            pl.read_csv(
                fp,
                separator="\t",
                columns=["importer", "exporter", "cmdCode", metric_col],
                infer_schema_length=1000
            )
            .filter(
                (pl.col("cmdCode") == code)
                & ((pl.col("importer") == importer) if importer else True)
                & (pl.col("exporter") != "W00")
                & (pl.col("importer") != "W00")
                & (pl.col("exporter") != pl.col("importer"))
            )
        )
        if df.is_empty():
            return None
        tot = df[metric_col].sum()
        if tot is None or tot <= 0:
            return None
        top3 = (
            df.group_by("exporter")
              .agg(pl.col(metric_col).sum().alias("flow"))
              .sort("flow", descending=True)
              .head(3)
        )
        return [(r["exporter"], r["flow"] / tot * 100)
                for r in top3.iter_rows(named=True)]

    # ---------------- compute best HS‑6 per quarter and k -----------------
    # iterate through all codes in the bucket and find greatest disrupting set at each size k
    rows = []
    for tag in quarters:
        for k in (1, 2, 3):
            best = {"loss": -1}
            for code in cmd_codes:
                pcs = pieces_for(code, tag)
                if pcs:
                    loss = sum(h for _, h in pcs[:k])
                    if loss > best["loss"]:
                        best = {"cmd": code, "loss": loss, "pieces": pcs[:k]}
            if best["loss"] >= 0:
                rows.append({"tag": tag, "k": k, **best})
    if not rows:
        print("No data for requested parameters."); return

    # ---------------- colour map -----------------------------------------
    used_iso = {iso for r in rows for iso, _ in r["pieces"]}
    colours  = {iso: _color_for_iso(iso) for iso in used_iso}

    # ---------------- plotting -------------------------------------------
    tags = sorted({r["tag"] for r in rows})
    x    = np.arange(len(tags))
    bar_w = 0.22
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.subplots_adjust(right=0.80)     # leave room for 2‑column legend

    bottoms = {(t, k): 0 for t in tags for k in (1, 2, 3)}

    for tier in range(3):                       # exporter tier 0/1/2
        for r in rows:
            if tier < len(r["pieces"]):
                iso, h = r["pieces"][tier]
                xpos   = x[tags.index(r["tag"])] + (r["k"] - 2) * bar_w
                ax.bar(xpos, h, bar_w,
                       bottom=bottoms[(r["tag"], r["k"])],
                       color=colours[iso], edgecolor="black", linewidth=1)
                bottoms[(r["tag"], r["k"])] += h

    ax.set_xticks(x, tags, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Flow lost (%)")
    scope = f"{importer} Imports" if importer else "Global"
    ax.set_title(f"Max‑Disruption Across Bucket {title_tag} {metric_tag} – {scope}".strip())
    ax.grid(axis="y", ls="--", alpha=0.4)

    # ---------------- legends --------------------------------------------
    exp_handles = [Line2D([0], [0], color=colours[c], lw=6) for c in sorted(colours)]
    exp_leg = ax.legend(
        exp_handles,
        sorted(colours),
        title="Exporter",
        ncol=2,
        bbox_to_anchor=(0.99, 1.0),
        loc="upper left",
        frameon=False,
        fontsize = 13,
        handlelength = 1.6,
        handleheight = 1.05,
        borderpad = 0.45,
        labelspacing = 0.35,
        columnspacing = 0.6   
    )
    ax.add_artist(exp_leg)

    ax.legend(
        [Patch(fc="black"), Patch(fc="white", ec="black", hatch="//")],
        ["Value", "Quantity"],
        title="Metric",
        bbox_to_anchor=(1.02, 0.20),
        loc="upper left",
        frameon=False,
        
    )

    plt.tight_layout()
    plt.show()

    # ---------------- summary table --------------------------------------
    table = pd.DataFrame(
        [{
            "Quarter": r["tag"],
            "k":       r["k"],
            "Loss %":  f"{r['loss']:.1f}",
            "cmdCode": r["cmd"],
            "Exporters": ", ".join(iso for iso, _ in r["pieces"])
        } for r in rows]
    ).sort_values(["Quarter", "k"]).reset_index(drop=True)

    display(table)
    return table
```
