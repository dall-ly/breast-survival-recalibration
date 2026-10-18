# Data and fixed-risk input

The study uses SCAN-B/GSE96058, METABRIC, and GSE42568. Source records are available through their original study repositories:

- SCAN-B: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058
- METABRIC: Curtis et al., https://doi.org/10.1038/nature10983
- GSE42568: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568

The repository includes aggregate allocation scores, not patient-level records. To rerun model fitting, obtain the fixed-risk research input from the authors or reconstruct it from the source studies and archived predictor pipeline under the applicable data terms.

## Required CSV

| Column | Meaning |
|---|---|
| `cohort` | `SCANB_holdout`, `METABRIC_external`, or `GSE42568_external` |
| `patient_id` | Stable string identifier, sorted before permutation |
| `os_days` | Observed overall-survival follow-up in days |
| `os_event` | Death indicator, 0 or 1 |
| `p_Clinical` | Fixed clinical three-year risk |
| `p_Gene` | Fixed gene-residual three-year risk |
| `p_Pathway` | Fixed pathway-residual three-year risk |

The archived input has 655 SCAN-B, 1,978 METABRIC, and 104 GSE42568 rows. Its SHA-256 is `7df3d64e64695d256fc4506ff46ca307e62f0156d3165271b644224e9ac0a916`.

The archived sequential identifiers preserve the original sorted order. Changing identifiers can change the random allocations even when the seed is unchanged. Exact byte identity is useful for matching the archived file; equivalent CSV formatting can change the hash without changing the data.

Place local input under `data/private/`, which is excluded from Git. `scripts/prepare_inputs.py` checks the schema and copies it to the two locations used by the original analysis scripts.
