"""Reproduce the corresponding manuscript figure from the released results."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "_mpl_cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image, ImageOps

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": .65, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": True, "savefig.facecolor": "white",
})
COLORS = {"intercept_ML": "#0072B2", "shrink_intercept": "#D98200",
          "intercept_slope_ML": "#009E73", "unchanged": "#777777"}
LABELS = {"intercept_ML": "Ordinary intercept", "shrink_intercept": "Penalized intercept (κ=5)",
          "intercept_slope_ML": "Intercept + slope"}
METHODS = ["intercept_ML", "shrink_intercept", "intercept_slope_ML"]
MARKERS = ["o", "s", "^"]
STYLES = ["-", "-", "-"]


def filehash(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def export(fig, name, claim, data_files, transformations, dimensions):
    """Independent exporter: fixed canvas preserves exact template width."""
    paths = {}
    for suffix in ["svg", "pdf", "png"]:
        path = HERE / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths[suffix] = path.name
    with Image.open(HERE / f"{name}.png") as im:
        ImageOps.grayscale(im).save(HERE / f"{name}_grayscale.png", dpi=(300, 300))
    fig.savefig(HERE / f"{name}_preview.png", dpi=150, facecolor="white")
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    text_outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.width+1 or box.y1 > canvas.height+1:
            text_outside.append(artist.get_text())
    manifest = {
        "claim": claim, "generation_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_code": Path(__file__).name, "source_code_sha256": filehash(__file__),
        "data_files": [{"path": p, "sha256": filehash(HERE/p)} for p in data_files],
        "transformations": transformations, "exports": paths,
        "size_inches": list(dimensions), "png_dpi": 300,
        "template": {"full_width_inches": 7.24028, "rendered_width_inches": 7.24, "single_column_inches": 3.495, "column_gap_inches": .25},
        "style": {"method_colors": COLORS, "legend_order": METHODS, "line_styles": STYLES, "markers": MARKERS,
                  "inside_title": False, "font": "Times New Roman", "minimum_text_pt": 8,
                  "all_figure_text_pt": 8, "template_font_compliance": "All visible labels, ticks, and legend text use 8 pt Times New Roman; Unicode superscripts avoid a separate math font.",
                  "marker_face_color": "white", "zero_line_style": "solid", "grid_line_style": "solid",
                  "vector_text": True, "no_smoothing": True, "no_dual_axes": True},
        "automatic_text_bounds_audit": {"outside_canvas": text_outside},
    }
    (HERE/f"{name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(name, "size", dimensions, "text outside canvas", text_outside)
    plt.close(fig)


def budget_response():
    data = pd.read_csv(HERE/"fig2_budget_response_data.csv")
    assert set(data.evaluation)=={"common"}
    assert set(data.model)=={"Average"} and set(data.reference)=={"unchanged"}
    assert (data.n_allocations==300).all()
    cohorts = ["SCANB_holdout","METABRIC_external","GSE42568_external"]
    cohort_labels = ["SCAN-B","METABRIC","GSE42568"]
    limits = [(-.5,9),(-2.5,11),(-25,1)]
    yticks = [[0,4,8],[-2,0,5,10],[-20,-10,0]]
    fig, axes = plt.subplots(3,2,figsize=(7.24,4.1),dpi=150)
    fig.subplots_adjust(left=.105,right=.985,bottom=.205,top=.935,wspace=.34,hspace=.64)
    handles = [Line2D([],[],color=COLORS[m],label=LABELS[m],marker=MARKERS[j],
                      linestyle=STYLES[j],lw=1.15,ms=3.5,mfc="white",mew=.9) for j,m in enumerate(METHODS)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.55,.008),
               frameon=False,ncol=3,columnspacing=1.3,handlelength=2.1,handletextpad=.5)
    # These are shared axis labels, not narrative titles.
    fig.text(.025,.57,"Mean ΔBrier (×10⁻³)",rotation=90,ha="center",va="center",fontsize=8)
    fig.text(.535,.57,"Worsening (%)",rotation=90,ha="center",va="center",fontsize=8)
    fig.text(.55,.100,"Calibration sample size",ha="center",va="center",fontsize=8)
    for row, (cohort,clab) in enumerate(zip(cohorts,cohort_labels)):
        for col, metric in enumerate(["mean_delta_x1e3","harm_percent"]):
            ax = axes[row,col]
            subset = data[data.cohort==cohort]
            expected = [25,50] if row==2 else [25,50,100,200]
            assert sorted(subset.budget.unique())==expected
            for j,method in enumerate(METHODS):
                vals = subset[subset.method==method].sort_values("budget")
                assert len(vals)==len(expected)
                ax.plot(vals.budget, vals[metric],color=COLORS[method],marker=MARKERS[j],
                        linestyle=STYLES[j],lw=1.15,ms=3.5,mew=.9,mfc="white")
            ax.set_axisbelow(True)
            ax.grid(axis="y",color="#E0E5EA",linewidth=.5,linestyle="-")
            ax.axhline(0,color=COLORS["unchanged"],lw=.65,linestyle="-",zorder=1)
            ax.spines["left"].set_color("#9AA6B2")
            ax.spines["bottom"].set_color("#9AA6B2")
            ax.set_xscale("log",base=2)
            ax.set_xlim((22,57) if row==2 else (21.5,232))
            ax.set_xticks(expected)
            ax.set_xticklabels([str(x) for x in expected])
            ax.minorticks_off()
            ax.tick_params(length=2.3,width=.65,pad=2)
            ax.text(0,1.10,f"({chr(97+2*row+col)})"+(f" {clab}" if col==0 else ""),
                    transform=ax.transAxes,fontsize=8,fontweight="bold",ha="left",va="bottom")
            if col==0:
                ax.set_ylim(*limits[row]);ax.set_yticks(yticks[row])
            else:
                ax.set_ylim(-2,102);ax.set_yticks([0,50,100])
    export(fig,"fig3_budget_response", "The trade-off between mean Brier change and worsening frequency depends on cohort and calibration budget; fixed shrinkage reduces allocation risk without uniformly giving the best mean.",
           ["fig2_budget_response_data.csv","fig2_budget_response_paired_allocations.csv"],
           ["Common evaluation patients fixed across calibration budgets within each permutation; post hoc sensitivity analysis.",
            "For each allocation and method, first calculate predictor-specific Brier differences vs unchanged, then average equally over Clinical, Gene and Pathway.",
            "Summarize 300 allocation-level averages per cohort-budget-method: left arithmetic mean; right 100 * fraction(delta>0).",
            "Left multiplied by 1000; units marked ×10^-3. Right in percent. No interval bands; no patient-level confidence intervals inferred.",
            "Only observed budgets connected; log2 horizontal axis exposes doubling in sample size; all tick labels show actual sample sizes.",
            "GSE42568 has only 25 and 50 patients and uses a shorter horizontal range. Left y scales differ and are explicitly labeled; right shares 0–100%.",
            "Mentor-guided visual revision retains the 7.24 by 4.1 inch canvas and 8 pt text. Open markers, light horizontal grids, left and bottom spines, and a shared bottom legend follow the supplied Fig. 2. All lines are solid. Data values, axis limits, tick values, method order, and CSV inputs are unchanged."],(7.24,4.1))


if __name__=="__main__":
    budget_response()
