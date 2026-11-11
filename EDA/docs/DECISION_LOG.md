# Decision Log

Record decisions that affect scientific validity or reproducibility.

---

## DEC-001 — Preserve FI-2010 temporal ordering
**Status:** Locked

**Decision:** We will not use a random train/test split as the primary evaluation.

**Reason:** FI-2010 is temporally ordered high-frequency market data. The benchmark supplies day-based / anchored evaluation, and DeepLOB evaluates using chronological setups.

**Consequence:** Any tuning must happen using only the development/training period. Final test data must not influence hyperparameter selection.

---

## DEC-002 — DeepLOB is an extension, not the whole project
**Status:** Locked

**Decision:** The project remains centered on Regression Techniques. DeepLOB is the nonlinear spatio-temporal benchmark.

**Reason:** This allows us to study interpretability, regularisation, temporal context, and nonlinear representation learning without turning the course project into only a neural-network reproduction.

---

## DEC-003 — Begin with NoAuction + Decimal Precision files
**Status:** Initial choice; verify during audit

**Decision:** Begin with the widely used DeepLOB-compatible FI-2010 NoAuction Decimal Precision representation.

**Reason:** It gives us a clean link to the published DeepLOB experimental setup and includes the full FI-2010 feature/label matrix.

**Verification required:** File names, shape, orientation, labels, and integrity must be confirmed from the actual downloaded files.


---

## DEC-004 — Freeman FI-2010 is the primary project dataset
**Status:** Locked

**Decision:** Use the FreemanOne Kaggle FI-2010 dataset as the primary dataset for the project.

**Exact representation:** NoAuction Z-score, CF9 anchored split.

**Reason:** This is the exact dataset referenced by the project team before implementation began. Using it preserves consistency between the proposal, dataset link and final experiments.

**Verified files:**

- `FI2010_train.csv`
  - 362,400 rows
  - 149 data columns
  - SHA-256: `53e86cb52a15a50f266a6b41532bd74e54f5cb11da288981d9aa2ecc0bc611ca`

- `FI2010_test.csv`
  - 31,937 rows
  - 149 data columns
  - SHA-256: `fa01243fcae34e4146d6ffbed1d5f4d3f35a06afdd608c22f68f56255360d4e5`

Both hashes match publicly available copies.

No NaN or infinite values were detected.

**Secondary data:** The previously audited NoAuction Decimal-Precision files are retained only as a reference / possible robustness dataset. They are not the primary experimental dataset.

**Consequence:** All primary regression, EDA and DeepLOB experiments must use the Freeman CF9 dataset unless an experiment is explicitly labelled as a secondary robustness analysis.

---

## DEC-005 — Keep engineered feature names conservative
**Status:** Locked

**Decision:** Use exact names for the verified u1 raw LOB columns and five
targets. Use family-relative names for u2--u9 members unless a source verifies
the flattened CSV ordering.

**Reason:** The paper defines engineered families but does not independently
verify every flattened CSV member position.

**Consequence:** The dictionary remains honest and machine-readable without
inventing unsupported semantics.

---

## DEC-006 — Interpret predictors only in standardized units
**Status:** Locked

**Decision:** Do not reconstruct physical prices, volumes, spreads, or
imbalance ratios from independently Z-scored columns.

**Reason:** Arithmetic on separately standardized variables does not generally
recover the corresponding raw-market quantity.

**Consequence:** EDA language and axes use standardized units only.

---

## DEC-007 — Use effect-size drift diagnostics
**Status:** Locked

**Decision:** Report full-split SMD and PSI plus deterministic bounded-sample
KS statistics; do not use large-N p-values as the primary drift result.

**Reason:** With this sample size, statistical significance is not a useful
proxy for practical distribution change.

**Consequence:** KS uses fixed seeds and at most 50,000 rows per split. PSI
uses train-decile bins, duplicate-edge removal, full counts, and 0.5-bin
smoothing. Their limitations are documented.

---

## DEC-008 — Separate constants from condition diagnostics
**Status:** Locked

**Decision:** Inventory zero-variance predictors separately and compute the
correlation eigen-spectrum, rank, and condition indices on the nonconstant
subspace.

**Reason:** Pearson correlation and standardized-design condition diagnostics
are undefined for constant variables.

**Consequence:** Nine constants are explicit; the remaining 135 predictors
have numerical rank 89. No undefined correlation is silently replaced by zero.

---

## DEC-009 — Define very high correlation as |r| >= 0.95
**Status:** Locked for EDA

**Decision:** Use absolute train Pearson correlation of at least 0.95 for the
near-redundancy inventory.

**Reason:** This is a transparent screening threshold, not a causal statement
or automatic feature-deletion rule.

**Consequence:** The canonical EDA reports 544 distinct pairs; later modelling
must choose any treatment using train-only validation.

---

## DEC-010 — Process Freeman splits sequentially
**Status:** Locked

**Decision:** Fully process and release train before loading test, retaining
only derived summaries and a bounded deterministic sample.

**Reason:** The train CSV is approximately 593 MB and simultaneous full-frame
loading is unnecessary.

**Consequence:** The canonical EDA is reproducible on more memory-constrained
machines without changing full-split summary calculations.

---

## DEC-011 — Lock the post-EDA modelling safeguards
**Status:** Locked

**Decision:** Fit every transformation on train only, remove train-constant
predictors using a train-derived rule, handle multicollinearity explicitly,
and keep day ten untouched for final evaluation.

**Reason:** EDA found constants, rank deficiency, extensive correlation, and
material temporal drift.

**Consequence:** The project is ready to enter regression modelling, but no
test-guided scaling, screening, feature selection, or hyperparameter tuning is
permitted.

---

## Effective numerical-rank tolerance

The FI-2010 predictor correlation matrix contains a pronounced eigenspectrum
gap after component 89. The 89th eigenvalue is approximately 8.41e-03,
whereas subsequent eigenvalues are approximately 1e-12 and can change sign
or cross machine-epsilon-based thresholds across BLAS/LAPACK implementations.

Therefore, effective numerical rank and condition-index calculations use a
fixed relative eigenvalue tolerance of 1e-10 with respect to the largest
eigenvalue.

This gives a platform-stable effective numerical rank of 89 and a maximum
finite positive-subspace condition index of approximately 75.44.

This tolerance affects descriptive multicollinearity diagnostics only. It is
not selected using the held-out test outcome and is not a predictive-model
hyperparameter.
