"""Rebuild paper summaries and figures from the released allocation scores."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script):
    subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)


if __name__ == '__main__':
    run('analysis/summarize.py')
    run('analysis/sensitivity_tables.py')
    for source, target in [
        ('figure_common_budget_average.csv', 'fig2_budget_response_data.csv'),
        ('figure_common_budget_average_paired_allocations.csv', 'fig2_budget_response_paired_allocations.csv'),
    ]:
        shutil.copyfile(ROOT / 'results' / source, ROOT / 'figures' / target)
    for name in ['plot_workflow.py', 'plot_main_results.py', 'plot_budget.py']:
        run('figures/' + name)
    print('Paper summaries and figures are ready in results/ and figures/.')
