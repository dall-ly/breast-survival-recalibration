"""Reproduce the corresponding manuscript figure from the released results."""
from pathlib import Path
import os
import csv
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '_mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MPath
from PIL import Image, ImageOps

W, H = 521.3 / 72, 2.50
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8,
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    'svg.hashsalt': 'icbcb-workflow-davis-amia-2019',
})
INK = '#000000'
LW = 0.65


def drawing():
    fig = plt.figure(figsize=(W, H), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis('off')

    def text(x, y, label, bold=False, ha='center', **kw):
        return ax.text(x, y, label, fontsize=8, ha=ha, va='center',
                       color=INK, linespacing=1.10,
                       fontweight='bold' if bold else 'normal', **kw)

    def line(points):
        xx, yy = zip(*points)
        ax.plot(xx, yy, color=INK, lw=LW, solid_capstyle='butt', zorder=1)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                    mutation_scale=7, color=INK, lw=LW,
                    shrinkA=0, shrinkB=0, zorder=2))

    def process(x, y, w, h, label):
        cut = .105
        points = [(x + cut, y), (x + w - cut, y), (x + w, y + h / 2),
                  (x + w - cut, y + h), (x + cut, y + h), (x, y + h / 2)]
        ax.add_patch(Polygon(points, closed=True, facecolor='white',
                             edgecolor=INK, lw=LW, zorder=3))
        text(x + w / 2, y + h / 2, label, zorder=4)

    def data(x, y, w, h, label):
        # Conventional data cylinder, as in Davis AMIA Fig. 1. No shading.
        cap = .075
        p = MPath([(x, y + h - cap / 2), (x, y + cap / 2),
                   (x, y - cap / 6), (x + w, y - cap / 6), (x + w, y + cap / 2),
                   (x + w, y + h - cap / 2), (x, y + h - cap / 2)],
                  [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
                   MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY])
        ax.add_patch(PathPatch(p, facecolor='white', edgecolor=INK, lw=LW, zorder=3))
        ax.add_patch(Ellipse((x + w / 2, y + h - cap / 2), w, cap,
                             facecolor='white', edgecolor=INK, lw=LW, zorder=3))
        text(x + w / 2, y + h / 2 - .018, label, zorder=4)

    text(.08, 2.43, '(a) Paired allocation and evaluation', bold=True, ha='left')
    data(.08, 1.59, 1.16, .59, 'Target cohort\nFixed risks\nand outcomes')
    process(1.49, 1.59, 1.16, .59, '300 paired\npermutations\nNested budgets')
    arrow((1.24, 1.885), (1.49, 1.885))
    line([(2.65, 1.885), (2.82, 1.885)])
    line([(2.82, 1.49), (2.82, 2.11)])
    arrow((2.82, 2.11), (3.03, 2.11))
    arrow((2.82, 1.49), (3.03, 1.49))
    data(3.03, 1.89, 1.19, .44, 'Calibration U\nFirst m patients')
    data(3.03, 1.25, 1.19, .48, 'Evaluation V\nRemaining or\ncommon suffix')
    process(4.49, 1.89, 2.65, .44,
            'Fixed comparison: four strategies\nNone / ordinary / κ=5 / intercept + slope')
    process(4.49, 1.25, 2.65, .48,
            'Paired IPCW Brier scores\nMean Δ and fraction Δ > 0')
    arrow((4.22, 2.11), (4.49, 2.11))
    arrow((4.22, 1.49), (4.49, 1.49))
    arrow((5.815, 1.89), (5.815, 1.73))
    text(1.55, 1.32, 'Same allocations across predictors and methods', ha='center')

    text(.08, 1.07,
         '(b) Exploratory selection within U: none, κ=5, ordinary',
         bold=True, ha='left')
    process(.08, .13, 1.25, .71,
            'Bootstrap stage 1\n100 replicates\nStore OOB losses')
    data(1.58, .13, 1.13, .71,
         'Pooled OOB\npatient–replicate\nrecords')
    process(2.96, .13, 1.25, .71,
            'Bootstrap stage 2\n1,000 paired\nrecord resamples')
    data(4.46, .13, 1.13, .71,
         'Distributions\nof paired Brier\nscore differences')
    process(5.84, .13, 1.30, .71,
            'Minimum error\nor preference\nSelect and refit')
    for left, right in [(1.33, 1.58), (2.71, 2.96), (4.21, 4.46), (5.59, 5.84)]:
        arrow((left, .485), (right, .485))
    # The OOB route enters the same disjoint evaluation; four fixed comparators
    # do not feed the selector. Evaluation outcomes never flow back into U.
    arrow((6.49, .84), (6.49, 1.25))
    return fig


def main():
    fig = drawing()
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(ROOT / f'fig1_workflow.{ext}', dpi=300)
    with matplotlib.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(ROOT / 'fig1_workflow_word.svg', dpi=300)
    fig.savefig(ROOT / 'fig1_workflow_preview.png', dpi=180)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            box = t.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                clipped.append(t.get_text())
    with Image.open(ROOT / 'fig1_workflow.png') as im:
        ImageOps.grayscale(im).save(ROOT / 'fig1_workflow_grayscale.png', dpi=(300, 300))
    plt.close(fig)
    rows = [
        ('predictors', 'Clinical;Gene;Pathway', 'Fixed 3-year risks'),
        ('allocation', '300 paired permutations', 'Shared across predictors and methods; nested calibration prefixes'),
        ('budgets', '25;50;100;200', 'GSE42568 uses 25 and 50 only'),
        ('calibration_U', 'First m patients', 'Fitting and exploratory selection use only calibration outcomes'),
        ('evaluation_V', 'Remaining or common suffix', 'Post hoc common suffix begins after maximum calibration budget'),
        ('fixed_comparison', 'None;Ordinary;kappa5;Intercept+slope', 'Four independent prespecified comparators; not inputs to selection'),
        ('selection_candidates', 'None;kappa5;Ordinary', 'Exploratory three-candidate selection only'),
        ('stage1', '100 bootstrap replicates', 'Fit with in-bag censoring weights; OOB loss uses calibration weights'),
        ('pooled_records', 'Patient-replicate records', 'Repeated OOB records are pooled without prediction averaging'),
        ('stage2', '1000 paired record resamples', 'Calibration-size draws; shared indices across candidates; not patient-cluster resampling'),
        ('decision', 'Minimum error or preference', 'Select per predictor; refit selected strategy on full calibration U'),
        ('summary', 'Mean Delta;fraction Delta>0', 'Each allocation Delta averages three predictor-specific IPCW Brier differences versus unchanged'),
    ]
    with (ROOT / 'fig1_workflow_data.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f); writer.writerow(['component', 'setting', 'definition']); writer.writerows(rows)
    caption = (
        'Fig. 1. Study workflow. (a) Fixed clinical, gene and pathway risks are evaluated '
        'using 300 paired cohort allocations. Calibration U contains the first m patients; '
        'evaluation V contains the remaining patients or, in the post hoc common-evaluation '
        'analysis, the suffix after the largest budget. Four fixed strategies and exploratory '
        'selection are evaluated separately. Δ averages three predictor-specific Brier differences '
        'versus unchanged. (b) At main budgets, calibration-only selection pools OOB patient–replicate records '
        'before paired record resampling. Hexagons denote processes; cylinders denote data, '
        'following Davis et al. [11].'
    )
    (ROOT / 'fig1_caption.txt').write_text(caption + '\n', encoding='utf-8')
    manifest = {
        'purpose': 'Distinguish calibration-only updating and selection from disjoint paired evaluation.',
        'size_inches': [W, H], 'width_points': 521.3, 'font_size_pt': 8,
        'fonts': 'Times New Roman throughout, including Greek symbols and bold panel labels',
        'png_dpi': 300, 'text_outside_canvas': clipped,
        'visual_reference': {
            'citation': 'Davis SE et al. Comparison of Prediction Model Performance Updating Protocols: Using a Data-Driven Testing Procedure to Guide Updating. AMIA Annu Symp Proc. 2019:1002–1010.',
            'figure': 'Figure 1',
            'article': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7153129/',
            'figure_page': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7153129/figure/f1-3203079/',
            'borrowed_conventions': 'Monochrome thin outlines; hexagonal processes; cylindrical data; left-to-right bootstrap/pool/bootstrap/distribution/decision sequence.',
            'artwork_reuse': 'No original image, text, or plotting code incorporated; original study-specific vector artwork.',
        },
        'supporting_references_inspected': [
            {'article': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513/', 'figure': 'Figure 1', 'role': 'Methodological context; not the direct visual template.'},
            {'article': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10714456/', 'figure': 'Figure 2', 'role': 'Checked separation of update and evaluation data; not the direct visual template.'},
        ],
        'template_body_width_calculation': '(612 - 2*45.35) points = 521.3 points',
        'original_code': True, 'all_outputs_relative_to_script': True,
        'source_code': 'plot_workflow.py',
        'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_snapshot': 'fig1_workflow_data.csv',
        'data_sha256': hashlib.sha256((ROOT / 'fig1_workflow_data.csv').read_bytes()).hexdigest(),
        'word_svg': 'fig1_workflow_word.svg; all text outlined as vector paths',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / 'fig1_workflow_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'text_outside_canvas': clipped, 'size_inches': [W, H], 'caption_words': len(caption.split())}))


if __name__ == '__main__':
    main()
