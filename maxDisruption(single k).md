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
#Columbia and Israel were also hard-coded because of similarity of randomly generated colors to already existing ones for other countries
_ISO2COLOR["COL"] = mpl.colors.to_rgb("xkcd:mustard yellow")
_ISO2COLOR["ISR"] = mpl.colors.to_rgb("xkcd:light brown")

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
#  Matplotlib defaults
###############################################################################
mpl.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 15, "legend.title_fontsize": 17,
    "hatch.linewidth": 1.5,
})


###############################################################################
#  Quarterly disruption plot helper
###############################################################################
def plot_max_disruption_from_codes_quarterly(
    cmd_codes,
    *,
    k: int = 3,
    importer: str | None = None,          #If no importer is specified, defaults to global perspective
    quarters = None,                      #Can specify desired time frame, defaults to entire breadth (2020Q1 - 2025Q1)
    folder   = "out",
    template = "q_transformed_{tag}.txt",  # expects {tag} = YYYYQ#
):
    """
    Quarterly analogue of the yearly disruption plot.

    For each quarter, pick the HS-6 code in `cmd_codes` whose removal of the
    top-k exporters yields the largest percentage 'disruption' (share of flow
    accounted for by the top k exporters). Plots value vs quantity.

    Parameters
    ----------
    cmd_codes : iterable[int]
        HS-6 candidate codes.
    k : int (≥1)
        Number of top exporters whose combined share defines disruption.
    importer : str | None
        Restrict to a single importer ISO (e.g., "USA") or None for global.
    quarters : list[str] | None
        Quarter tags ("YYYYQ#"). If None: full span 2020Q1 .. 2025Q1.
    folder : str
        Directory containing the quarterly transformed files.
    template : str
        Filename template containing '{tag}' placeholder.
    """
    cmd_codes = list(cmd_codes)
    if k < 1:
        raise ValueError("k must be ≥1")

    # default quarter coverage
    if quarters is None:
        quarters = []
        for yr in range(2020, 2026):
            for q in range(1, 5):
                if yr == 2025 and q > 1:  # stop after 2025Q1
                    break
                quarters.append(f"{yr}Q{q}")

    # optional title tags, used for putting a title tag in plots generated
    ND_DY_SET = {280530, 284690, 850511, 720299}
    GA_IN_SET = {811299, 285000, 381800}
    CO_SET = {260500,282200,282739,283329,283699,291529,810520,810530,810590}
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

    #helper to compute top-k exporter shares for a code & quarter
    #main logic for computing max disruption
    def pieces_for(code, tag, metric):
        from pathlib import Path
        fp = Path(folder) / template.format(tag=tag)
        if not fp.exists():
            return None
        #read in data
        df = (
            pl.read_csv(
                fp,
                separator="\t",
                columns=["importer", "exporter", "cmdCode", metric],
                infer_schema_length=1000
            )
            .filter(
                (pl.col("cmdCode") == code)
                & ((pl.col("importer") == importer) if importer else True)
                #Filter out when importer/exporter is unspecified, and when exporter and importer are the same
                #W00 is the Iso-3 code for partner and reporter code 0 which means the exporter/importer wasn't specified 
                & (pl.col("exporter") != "W00")
                & (pl.col("importer") != "W00")
                & (pl.col("exporter") != pl.col("importer"))
            )
        )
        if df.is_empty():
            return None
        #sums total flow based on metric provided (qty or value)
        total = df[metric].sum()
        if total is None or total <= 0:
            return None
        topk = (
            #calculates the raw flow each exporter is accountable for
            df.group_by("exporter")
              .agg(pl.col(metric).sum().alias("flow"))
              .sort("flow", descending=True)
              .head(k)
        )
        #converts the raw flow for each exporter into a proportion of overall flow to the importer from all sources
        return [(r["exporter"], r["flow"] / total * 100)
                for r in topk.iter_rows(named=True)]

    # evaluate quarters
    rows_v, rows_q = [], []
    for tag in quarters:
        best_v = {"loss": -1}
        best_qt = {"loss": -1}
        for code in cmd_codes:
            pcs_v = pieces_for(code, tag, "primaryValue")
            pcs_q = pieces_for(code, tag, "qty")
            if pcs_v:
                lv = sum(h for _, h in pcs_v)
                if lv > best_v["loss"]:
                    best_v = {"loss": lv, "cmd": code, "pieces": pcs_v, "tag": tag}
            if pcs_q:
                lq = sum(h for _, h in pcs_q)
                if lq > best_qt["loss"]:
                    best_qt = {"loss": lq, "cmd": code, "pieces": pcs_q, "tag": tag}
        if best_v["loss"] >= 0:
            rows_v.append(best_v)
        if best_qt["loss"] >= 0:
            rows_q.append(best_qt)

    if not rows_v or not rows_q:
        print("No data.")
        return

    #visual effects and graphing
    #colours
    used = {iso for r in rows_v + rows_q for iso, _ in r["pieces"]}
    colours = {iso: _color_for_iso(iso) for iso in used}

    tags = sorted({r["tag"] for r in rows_v})
    x = np.arange(len(tags))
    w = 0.35
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.subplots_adjust(right=0.80)
    bv = {t: 0 for t in tags}
    bq = {t: 0 for t in tags}

    def stack(rows, offset, bottoms, hatch=False):
        for tier in range(k):
            for r in rows:
                if tier < len(r["pieces"]):
                    iso, h = r["pieces"][tier]
                    idx = tags.index(r["tag"])
                    ax.bar(
                        x[idx] + offset,
                        h,
                        w,
                        bottom=bottoms[r["tag"]],
                        color=colours[iso],
                        edgecolor="black",
                        lw=0.3,
                        hatch="//" if hatch else ""
                    )
                    bottoms[r["tag"]] += h

    stack(rows_v, -w / 2, bv, False)
    stack(rows_q,  w / 2, bq, True)

    ax.set_xticks(x, tags, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    scope = f"{importer} Imports" if importer else "Global"
    ax.set_title(f"Max-Disruption Across Bucket {title_tag} – {scope}".strip())
    ax.set_ylabel("Flow lost (%)")
    ax.grid(axis="y", ls="--", alpha=0.4)

    # legends
    # toggle lw = 6 or 5 as ness
    exp_handles = [Line2D([0], [0], color=colours[c], lw=5) for c in sorted(colours)]
    exp_leg = ax.legend(
        exp_handles,
        sorted(colours),
        title="Exporter",
        ncol=2,
        bbox_to_anchor=(0.99, 1.0),
        loc="upper left",
        frameon=False,
        ##uncomment if nessessary and adjust comma on top
        fontsize = 13,
        handlelength = 1.6,
        handleheight = 1.05,
        borderpad = 0.45,
        labelspacing = 0.35,
        columnspacing = 0.6
    )
    ax.add_artist(exp_leg)
    ax.legend(
        [Patch(fc="black"),
         Patch(fc="white", ec="black", hatch="//")],
        ["Value", "Quantity"],
        title="Metric",
        bbox_to_anchor=(1.0, 0.25),
        loc="upper left",
        frameon=False
    )
    plt.tight_layout()
    plt.show()

    # summary table
    tbl = []
    for lbl, rows in [("Value", rows_v), ("Quantity", rows_q)]:
        for r in rows:
            tbl.append({
                "quarter": r["tag"],
                "metric": lbl,
                "cmdCode": r["cmd"],
                "loss": f"{r['loss']:.1f} %",
                "countries": ", ".join(i for i, _ in r["pieces"])
            })
    display(pl.DataFrame(tbl).sort(["quarter", "metric"]))
```
