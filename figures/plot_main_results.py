"""Reproduce the corresponding manuscript figure from the released results."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import csv
import json
import os

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "results" / "table2_main_remaining_average.csv"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.unicode_minus": True, "axes.linewidth": .6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
})
METHODS = ["intercept_ML", "shrink_intercept", "intercept_slope_ML"]
COLORS = ["#0072B2", "#D98200", "#009E73"]
LABELS = ["Ordinary intercept", "Penalized intercept (κ = 5)", "Intercept + slope"]
COHORTS = [
    ("SCANB_holdout", "SCAN-B", 100),
    ("METABRIC_external", "METABRIC", 100),
    ("GSE42568_external", "GSE42568", 50),
]
POSITIONS = [0, .76, 1.52, 3, 3.76, 4.52, 6, 6.76, 7.52]
SIZE = (7.24, 3.0)


def signed(value):
    return f"{value:+.2f}".replace("-", "−")


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    lookup = {(r["cohort"], r["method"]): r for r in source_rows}
    rows = []
    for cohort_id, cohort, budget in COHORTS:
        for method in METHODS:
            r = lookup[cohort_id, method]
            assert r["evaluation"] == "remaining" and r["model"] == "Average"
            assert int(r["budget"]) == budget and int(r["n_allocations"]) == 300
            values = [
                -1000 * float(r["mean_delta"]),
                float(r["harm_percent"]),
                -1000 * float(r["worst10_mean_delta"]),
            ]
            rows.append({"cohort": cohort, "calibration_n": budget,
                         "method": method, "values": values,
                         "labels": [signed(values[0]), f"{values[1]:.1f}%", signed(values[2])]})
    assert rows[8]["labels"][2] == "−26.67"
    # Reading the raw float also avoids double-rounding METABRIC's −4.994908.
    assert rows[3]["labels"][2] == "−4.99"

    fig = plt.figure(figsize=SIZE, dpi=150)
    axes = [fig.add_axes([x, .248, .252, .588]) for x in [.145, .434, .723]]
    titles = ["(a) Average improvement ↑", "(b) Worsening ↓", "(c) Worst 10% improvement ↑"]
    xlabels = ["Brier reduction (×10⁻³)", "Allocations (%)", "Brier reduction (×10⁻³)"]
    limits = [(-12, 32), (0, 120), (-42, 17)]
    ticks = [[0, 10, 20, 30], [0, 50, 100], [-30, -20, -10, 0, 10]]
    data_text = []
    for panel, ax in enumerate(axes):
        for start in [-.44, 5.56]:
            ax.axhspan(start, start + 2.4, facecolor="#F1F4F6", linewidth=0, zorder=0)
        ax.set_xlim(limits[panel])
        ax.set_ylim(8.08, -.56)
        ax.set_xticks(ticks[panel])
        ax.set_yticks([])
        ax.grid(axis="x", color="#D9E0E5", linewidth=.4, linestyle="-", zorder=1)
        ax.axvline(0, color="#77828C", linewidth=.65, linestyle="-", zorder=2)
        ax.set_axisbelow(True)
        for side in ["left", "top", "right"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#8C969F")
        ax.tick_params(axis="x", direction="out", width=.5, length=2.5, pad=2)
        ax.set_xlabel(xlabels[panel], labelpad=4)
        ax.set_title(titles[panel], pad=9, fontweight="bold")
        for y, row in zip(POSITIONS, rows):
            value = row["values"][panel]
            method_index = METHODS.index(row["method"])
            ax.barh(y, value, height=.45, color=COLORS[method_index],
                    edgecolor="none", linewidth=0, zorder=3)
            text = ax.annotate(row["labels"][panel], (value, y),
                               xytext=((3 if value >= 0 else -3), 0),
                               textcoords="offset points", va="center",
                               ha="left" if value >= 0 else "right", fontsize=8,
                               color="#20262B", clip_on=False, zorder=4)
            data_text.append((panel, text))

    for (_, cohort, budget), y in zip(COHORTS, [.76, 3.76, 6.76]):
        display_y = axes[0].transData.transform((0, y))[1]
        figure_y = fig.transFigure.inverted().transform((0, display_y))[1]
        fig.text(.130, figure_y, f"{cohort}\nm = {budget}",
                 ha="right", va="center", fontsize=8, linespacing=1.18)

    fig.legend([Patch(facecolor=c, edgecolor="none") for c in COLORS], LABELS,
               loc="lower center", bbox_to_anchor=(.55, .015), ncol=3,
               frameon=False, handlelength=1.3, handleheight=.7,
               handletextpad=.55, columnspacing=1.5, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    text_bounds = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        text_bounds.append({"text": artist.get_text(),
                            "bounds_inches": [round(v / fig.dpi, 5) for v in bbox.extents]})
        if bbox.x0 < -.5 or bbox.y0 < -.5 or bbox.x1 > canvas.width + .5 or bbox.y1 > canvas.height + .5:
            outside.append(artist.get_text())
    data_outside_panel = []
    for panel, artist in data_text:
        b = artist.get_window_extent(renderer)
        p = axes[panel].get_window_extent(renderer)
        if b.x0 < p.x0 - .5 or b.x1 > p.x1 + .5:
            data_outside_panel.append({"panel": panel + 1, "text": artist.get_text()})
    assert not outside, outside
    assert not data_outside_panel, data_outside_panel
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(HERE / f"fig2_main_results.{ext}", dpi=300, facecolor="white")
    manifest = {
        "figure": 2,
        "source_csv": str(SOURCE.relative_to(HERE.parent)),
        "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "source_code": Path(__file__).name,
        "source_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "size_inches": list(SIZE), "dpi": 300,
        "png_dimensions": [2172, 900],
        "font": "Times New Roman", "minimum_text_pt": 8,
        "all_lines_solid": True, "method_colors": dict(zip(METHODS, COLORS)),
        "reference_design_adopted": "Three parallel panels, nine method rows grouped by cohort, alternating pale backgrounds, signed bar-end labels, and one bottom legend.",
        "panel_transformations": ["−1000 × mean_delta", "harm_percent", "−1000 × worst10_mean_delta"],
        "uncertainty": "No confidence interval is plotted. The right panel summarizes the 30 largest signed allocation-level delta values among 300 allocations.",
        "rounding": "Brier reductions rounded once to two decimals from the raw floating-point columns; worsening to one decimal.",
        "data_values_and_labels": rows,
        "checks": {"values_verified": 27, "text_outside_canvas": outside,
                   "bar_labels_outside_panel": data_outside_panel,
                   "GSE42568_slope_worst_label": rows[8]["labels"][2],
                   "METABRIC_ordinary_worst_label": rows[3]["labels"][2]},
        "automatic_text_bounds_audit": {"outside_canvas": outside,
                                        "bar_labels_outside_panel": data_outside_panel},
        "text_bounds_inches": text_bounds,
    }
    (HERE / "fig2_main_results.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"size_inches": SIZE, "values_verified": 27,
                      "text_outside_canvas": outside,
                      "bar_labels_outside_panel": data_outside_panel}, ensure_ascii=False))


if __name__ == "__main__":
    main()
