# Evaluation

The protocol was fixed before any test-set number existed, and test data were never used for any choice.

## Split protocol

| Role | Sessions | Use |
|---|---|---|
| Training and validation pool | 11 Collection A sessions from 2026-08-25, 2026-08-26, 2026-09-01 | Training, epoch selection, threshold |
| Primary test | `2026-08-28/2p_kid_1`, `2p_kid_2` (vendor `fixed-test`) | Headline result, read once per selected model |
| Secondary shift test | 8 labeled Collection B sessions from 2026-09-11 and 2026-09-17 | Reported separately, never tuned on |
| Excluded | 2026-09-14 sessions, transition, unknown, unlabeled time | Not used |

Validation scheme: **leave one date out** over the pool. For each of the three dates, train on the other two, validate on the held-out date. Each candidate therefore yields three models per seed. Every model scores its validation date, the primary test and the shift test. The split unit is the session, and a session never appears in two roles.

The README of the handoff forbids using the fixed test for thresholds, feature selection, tuning or normalisation baselines. This protocol respects that.

Leakage checks (automated tests): no session in two roles; no segment in two roles; no window's epoch span crosses a segment boundary; baseline statistics, class weights and the feature scaler use training windows only; the selection code has no access to test predictions.

## Selection

- **Checkpoint:** the epoch with the highest validation partial AUC over false-alarm rates from 0 to 0.05 (standardised, `roc_auc_score(max_fpr=0.05)`), computed on `eval_nonoverlap` validation windows. Ties go to the earlier epoch.
- **Threshold:** no fixed false-alarm cap. The threshold is the smallest score above the highest score of any negative validation window, so validation raises zero false alarms. Because one outlier window can push it high, recall at that point can be very low. The report shows this openly next to the full curve.
- **`M` (smoothing length):** the value in `{1, 3, 5}` with the highest validation recall at the zero-false-alarm threshold. Ties go to smaller `M`.
- **Candidate:** the higher mean validation partial AUC across dates and seeds, with the tie rule from `ARCHITECTURE.md`.

## Metrics

Primary, on the primary test at the validation threshold:

1. Recall of `single-child` windows.
2. Realised false-alarm rate on negative windows. The aim is as low as possible, with no fixed target. The realised value on the test sessions is reported even when it is above zero.

Secondary:

3. Recall when the test threshold is set for 0%, 0.1%, 1% and 5% false alarms. These are operating points on the test ROC, labeled "oracle threshold", and never used for selection.
4. AUROC and average precision.
5. False-alarm rate per negative type: `empty`, `single-adult`, `multiple`. `single-adult` appears only in the shift test.
6. Per-session table: windows, alarms, recall or false-alarm rate. Two test sessions make this the most honest view.
7. Confusion matrix over the four classes with support.

Diagnostics: predicted-class histogram, fraction of windows above threshold, epoch selected, validation-threshold value, and the validation recall at that threshold.

All windows in reported metrics are non-overlapping 5 s windows.

## Baselines

B0 (never alarm) and B1 (feature logistic regression) run through the same code, splits and metrics. Both are fitted on training windows only.

## Reporting

- Per fold and seed rows, then means and standard deviations across the three validation dates and the seeds.
- Pooled out-of-fold validation results appear as a separate table, because the estimands differ from per-model test results.
- The primary test and the shift test appear in separate tables. They are never averaged together.
- The Collection A to B comparison states the size and direction of the gap and lists the changes that come with it: bandwidth (512 vs 256 subcarriers), packet rate, dates, class mix and the subcarrier pooling. It does not attribute the gap to one cause.

## Report integrity

The report reads `run.json` of every run and refuses to combine runs with different cache id, split hash, source hashes or configuration. It also requires the complete job matrix (2 neural candidates and 2 baselines, 3 dates, 3 seeds for neural). It writes `manifest.json` with the identity and every accepted run. Recall at fixed false-alarm rates is labeled "(oracle)" everywhere, because those thresholds come from the evaluated split's own scores. A second table keeps only windows at least 20 s from a segment edge (see `PIPELINE.md`).

## Interpretation limits

- The child is a dummy infant. Results say nothing about real children.
- The pool has 11 positive segments; the primary test has 2; the shift test has 6. Window counts are large because windows are correlated. I report no p-values and no window-level confidence intervals. I show per-session values.
- The primary test has no lone-adult windows, so its false-alarm rate does not cover that case.
- Three validation dates give a coarse threshold. With 400 to 1,500 negative validation windows per fold and heavy correlation, a zero-false-alarm threshold rests on few independent negative segments, and the realised test rate can be above zero.
- One car and a small set of people (**Assumed**) mean generalisation to new cars is untested.
- Date, collection, sample rate and class mix change together.
- With three seeds, differences smaller than the spread between seeds are noise.
- A result does not have to beat any published number. DeepCPD and RapidPD report results on other hardware, cars and data.
