# Fixed-model evaluation summary

Interpretation: models were fixed before evaluation; independent results are descriptive, not tuning criteria.

| Seed/run | CV AUPR mean ± SD | Checkpoint loss gap | Independent ensemble AUPR (Test251 / Test1440 / Test167 / Leadspep / NewInd) |
|---|---|---|---|
| pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed123 | 0.8271 ± 0.0150 | 0.3080 | 0.9811 / 0.9779 / 0.9816 / 0.9635 / NA |
| pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed1997 | 0.8297 ± 0.0103 | 0.2666 | 0.9702 / 0.9700 / 0.9708 / 0.9189 / NA |
| pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed2026 | 0.8387 ± 0.0125 | 0.2554 | 0.9725 / 0.9685 / 0.9733 / 0.9340 / NA |
| pepresmlp_pqabsdiff_HD128_MLP3_B32_lr1e-3_seed42 | 0.8356 ± 0.0158 | 0.2999 | 0.9777 / 0.9781 / 0.9783 / 0.9470 / NA |

## Important limitations

- Fold-specific best-epoch validation metrics are used for CV summaries; they are selected on the same fold and may be optimistic. Pooled OOF AUC/AP, if saved, are in JSON and CSV.
- Training/validation loss gap at each selected checkpoint is a descriptive diagnostic, not proof of no overfitting.
- Verify duplicate pairs, peptide/protein homology and leakage before claiming independent generalization.
- Parameter count excludes the pretrained embedding model; 10-fold ensemble weights are ten times the individual classifier weights.
- AUPR is sklearn average precision; comparisons require consistent dataset construction and prevalence.
