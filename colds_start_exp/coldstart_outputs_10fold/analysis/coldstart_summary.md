# ResPepBind held-out cold-start test results

Root: `/home/ubuntu/storage/peptide-protein-binding/For_publication/scripts_finalised_model/respepbind_coldstart/coldstart_outputs_seed2026_10fold`

Valid evaluations: **120 / 120**

| Scenario | Identity | Folds | AUC (mean ± SD) | AUPR (mean ± SD) | ACC (mean ± SD) | F1 (mean ± SD) |
|---|---|---:|---|---|---|---|
| novel_peptide | 0.6 | 10 | 0.8292 ± 0.0160 | 0.7968 ± 0.0263 | 0.7495 ± 0.0163 | 0.7702 ± 0.0115 |
| novel_peptide | 0.7 | 10 | 0.8239 ± 0.0203 | 0.7964 ± 0.0244 | 0.7472 ± 0.0173 | 0.7642 ± 0.0162 |
| novel_peptide | 0.8 | 10 | 0.8353 ± 0.0134 | 0.8065 ± 0.0193 | 0.7546 ± 0.0139 | 0.7752 ± 0.0081 |
| novel_peptide | 0.9 | 10 | 0.8399 ± 0.0171 | 0.8100 ± 0.0205 | 0.7570 ± 0.0213 | 0.7817 ± 0.0148 |
| novel_protein | 0.6 | 10 | 0.8186 ± 0.0148 | 0.7895 ± 0.0168 | 0.7412 ± 0.0131 | 0.7617 ± 0.0197 |
| novel_protein | 0.7 | 10 | 0.8173 ± 0.0155 | 0.7841 ± 0.0166 | 0.7441 ± 0.0199 | 0.7651 ± 0.0190 |
| novel_protein | 0.8 | 10 | 0.8251 ± 0.0103 | 0.7917 ± 0.0162 | 0.7480 ± 0.0105 | 0.7654 ± 0.0135 |
| novel_protein | 0.9 | 10 | 0.8285 ± 0.0155 | 0.7980 ± 0.0190 | 0.7467 ± 0.0184 | 0.7730 ± 0.0177 |
| double_cold | 0.6 | 10 | 0.8068 ± 0.0183 | 0.7697 ± 0.0376 | 0.7177 ± 0.0286 | 0.7433 ± 0.0364 |
| double_cold | 0.7 | 10 | 0.8160 ± 0.0475 | 0.7868 ± 0.0504 | 0.7430 ± 0.0234 | 0.7607 ± 0.0208 |
| double_cold | 0.8 | 10 | 0.8117 ± 0.0367 | 0.8016 ± 0.0498 | 0.7371 ± 0.0307 | 0.7654 ± 0.0342 |
| double_cold | 0.9 | 10 | 0.8303 ± 0.0236 | 0.8045 ± 0.0338 | 0.7617 ± 0.0246 | 0.7828 ± 0.0268 |

Results are calculated from `independent_test_metrics.csv` on `coldstart_test`, not internal validation.
SD is the sample standard deviation across completed folds (ddof=1).
Comparisons with PepLM-GNN Figure 3 are literature-level rather than paired, identical-split comparisons.
Inspect `coldstart_missing_or_invalid.csv` and the split exclusion policy before publication.
