# Historical fixed predictors

These source files document construction of the clinical, gene-residual, and pathway-residual predictors used in the recalibration study. `frozen_predictor_snapshot.json` records the selected parameters, feature names, coefficients, and source-model hashes.

| Component | Implementation |
|---|---|
| SCAN-B/METABRIC input harmonization | `scripts/analyze_gse96058_os.py` |
| Clinical forest and censoring calculations | `scripts/survival_experiments.py`, `scripts/experiment_clinical_anchor.py` |
| Molecular residual and ranking objective | `scripts/program_ranking_v5.py` |
| Six-variable clinical anchor | `scripts/program_node_v5.py` |
| GSE42568 platform transport and refitting | `scripts/prepare_external_cohort_v5.py` |
| Prediction assembly | `scripts/analyze_expansion_v5.py` |

The remaining local modules are imported dependencies. `requirements-original.txt` records the historical environment. These scripts expect the original `processed/` and `runs/` research data layout, which is not distributed here. They preserve the historical pipeline rather than providing a single-command download-and-train workflow.

For the two external data locations, set `SCANB_PROCESSED_DIR` and `METABRIC_RECURRENCE_FILE`. Defaults are relative to `source_models/data/`.

SCAN-B/METABRIC use 18,150 ranked genes and a six-variable clinical anchor. GSE42568 uses a source refit on 16,317 shared genes and a five-variable anchor. Source development uses 2,618 SCAN-B patients; the recalibration comparison uses fixed risks for the three evaluation cohorts.
