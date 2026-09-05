# LFD-IDS

**Bagging-based data poisoning attacks against cyberattack detection in
connected vehicles — and a clustering defence against them.**

Connected vehicles stream tyre temperature, tyre pressure and GPS position to
a cloud intrusion detection system that labels each record benign or
malicious. Because that training data is collected from distributed,
potentially compromised sensors and travels over 5G/4G, an adversary can
invert its labels before the model ever sees them.

This project builds the whole loop as a working system: the acquisition
pipeline, two label-flipping attacks (**BOOTLFA** and **BAGLFA**), the deep
learning IDS they target, and the **K-means Clustering Defence (KCD)** that
repairs the poisoned labels and restores the detector.

---

## Headline result

Measured on a clean held-out test set, averaged over 3 seeds
(see [docs/RESULTS.md](docs/RESULTS.md) for the full tables):

| | Accuracy | FNR (missed intrusions) | AUC |
| --- | --- | --- | --- |
| Clean baseline | **99.45%** | 0.98% | 0.9990 |
| BAGLFA @ 50% poisoning | **42.6%** | 53.5% | 0.383 |
| BAGLFA @ 50% + BAGKCD | **98.6%** | 2.8% | 0.9980 |
| BOOTLFA @ 50% poisoning | **59.8%** | 39.6% | 0.669 |
| BOOTLFA @ 50% + BOOTKCD | **98.7%** | 2.5% | 0.9961 |

Half the training labels inverted takes the detector to the random-guessing
floor or below — BAGLFA's AUC of 0.383 means the model has learned an actively
*inverted* ranking. The defence puts it back within a point of baseline and
cuts missed intrusions by **19×**.

---

## Quick start

```bash
pip install -e .            # core install (NumPy backend, no TensorFlow needed)
pip install -e ".[dev]"     # + pytest and scikit-learn for the test suite

lfd-ids demo                # ~30 s end-to-end walkthrough
lfd-ids run -c configs/default.yaml --repeats 3 -o results
```

`run` writes `results/results.csv`, `results/summary.csv`,
`results/experiment_report.json` and a set of figures under
`results/figures/`.

### Other commands

```bash
lfd-ids generate-data --out data/cv_sensors.csv   # Module 1 only, export CSV
lfd-ids attack                                    # poison and report the damage
lfd-ids defend                                    # run KCD, report label recovery
lfd-ids figures                                   # re-render plots from a finished run
```

### Configuration

Everything is driven by [`configs/default.yaml`](configs/default.yaml). Any key
can be overridden on the command line, before or after the subcommand:

```bash
lfd-ids run --set model.epochs=50 --set attack.intensities=0.2,0.35,0.5
lfd-ids run --set attack.selection=malicious_to_benign     # a directed attack
lfd-ids run --set defence.outer_policy=keep                # the relabel-only rule
lfd-ids run --backend keras                                # TensorFlow instead of NumPy
```

### Using a real dataset

The simulator is the default source, but the pipeline runs on a real capture
just as well:

```bash
lfd-ids run --csv data/real_capture.csv \
            --set acquisition.csv_label_column=is_attack
```

The CSV needs one column per name in `acquisition.feature_set` plus a binary
label column. Optional `attack_type` and `vehicle_id` columns enable the
per-attack-type breakdown and the alert module's per-vehicle streak tracking.

---

## How it works

Four modules, matching the project architecture. Full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Module 1 — acquisition (`src/lfd_ids/module1_acquisition/`)

A simulated fleet emits telemetry that obeys real tyre physics: pressure
follows the ideal gas law against tyre temperature, temperature rises with
sustained speed, fuel drains with distance, position tracks a road network.
Five attack types break that structure in different ways — TPMS deflation
spoofing, thermal injection, GPS spoofing, replay and fuzzing — arriving in
bursts, with a configurable share blended back towards benign values so the
classes are not trivially separable. Records travel a 5G/4G channel with
packet loss and clock skew, are validated and clock-aligned by the ingestion
layer, and land in a SQLite cloud database as the labelled set `S = (xᵢ, yᵢ)`.

### Module 2 — the attacks (`src/lfd_ids/module2_attack/`)

Both attacks apply the same operator — select a fraction `p` of samples,
invert their labels with `y' = |1 − y|` — but distribute the budget
differently:

- **BOOTLFA** accumulates flip votes over `B` bootstrap resamples and keeps
  the most-voted samples, so corruption clusters where the bootstrap
  distribution over-samples (*distribution-level* noise).
- **BAGLFA** fills the budget round-robin across `B` bagging subsets, so every
  subset carries an equal share (*variance-reduced* noise).

Both flip exactly `round(p·N)` labels, so at each intensity they differ in
*which* labels move, not how many.

### Module 3 — the IDS (`src/lfd_ids/module3_ids/`)

```
Input(4) → Dense(64, relu) → Dense(32, tanh) → Dense(16, relu)
         → Dense(8, tanh)  → Dense(1, sigmoid)        3,073 parameters
```

Two interchangeable backends build it: a self-contained **NumPy**
implementation (the default — deterministic and ~8× faster at this size) and
a **Keras** one for parity with the original toolchain. 60/20/20 stratified
split, min-max scaling fitted on train only. Reported metrics: accuracy,
precision, recall, F1, **FNR**, FPR, specificity and AUC.

### Module 4 — the defence (`src/lfd_ids/module4_defense/`)

KCD assumes the adversary corrupted *labels*, not *features* — so the geometry
of the poisoned data still carries the class structure. It clusters with
k-means (`k = 2`), maps each cluster to a class, measures each sample's
distance to its centroid, and relabels everything closer than the mean
distance `μ`. **BOOTKCD** and **BAGKCD** run that over bootstrap resamples and
bagging subsets respectively and vote. The retrained model then passes a
robustness gate before it is published to the fleet.

---

## What the experiments show

Three findings, in order of how much they matter. Numbers and tables in
[docs/RESULTS.md](docs/RESULTS.md).

**1. At 50% poisoning the IDS collapses.** Accuracy falls from 99.45% to 42.6%
(BAGLFA) and 59.8% (BOOTLFA); AUC falls to 0.383 and 0.669. The training labels
carry no class signal left to learn. These cells are also by far the most
unstable in the grid (±15 and ±13 points across three seeds) — at exactly 50%
inversion the model latches onto whatever accidental structure the poisoned
labels happen to hold that run.

**2. Below 50%, uniform random flipping is far less damaging than it looks.**
At `p = 30%` the attacked model still scores 98.3–98.8%. This is not a bug — it
is the well-known robustness of a classifier to *symmetric* label noise: while
fewer than half the labels flip, the majority label in each region of feature
space is still the correct one, so the decision boundary survives. The damage
at 30–40% comes almost entirely from the model memorising noise late in
training, not from the boundary moving.

That robustness disappears the moment the adversary aims its flips. With
`attack.selection=malicious_to_benign`, accuracy collapses to ~53% at
`p = 30%` — the same damage the random attack needs 50% to achieve. **A
directed attacker needs a little over half the budget.**

**3. The defence works, but only with the outer shell removed.** KCD relabels
only samples within the mean distance `μ` of their centroid — about 53% of
the data — so a single pass can never repair more than ~55% of the flips.
Keeping the rest with their poisoned labels leaves 14–22% residual label error
and recovers only to ~83% at `p = 50%`. Dropping them from the retraining set
instead (`outer_policy: drop`, the default) restores **98.5–99.0% at every
intensity**, for both attacks. Running the defence on a *clean* training set
costs only 0.59 points, so it can be left permanently enabled.

---

## Project layout

```
src/lfd_ids/
  config.py                  typed configuration for all four modules
  pipeline.py                the experiment grid: baseline → attack → defence
  cli.py                     command-line interface
  figures.py                 result plots
  module1_acquisition/       sensors, V2X channel, ingestion, cloud database
  module2_attack/            BOOTLFA, BAGLFA, attacker model
  module3_ids/               preprocessing, MLP, metrics, alert module
  module4_defense/           k-means, KCD, BOOTKCD/BAGKCD, retrain & deploy
configs/default.yaml         the reference configuration
tests/                       113 tests across all four modules
docs/ARCHITECTURE.md         module-by-module design
docs/RESULTS.md              measured results and analysis
scripts/run_experiments.sh   reproduces every table in RESULTS.md
scripts/make_report_tables.py renders those tables from a run directory
```

## Tests

```bash
pytest -q          # 113 tests, ~25 s
```

The suite checks the sensor physics (attacked records really do break the gas
law), that both attacks hit their exact flip budgets, that the MLP's parameter
count and layer stack match the specification, and cross-checks the project's
own k-means against scikit-learn's and its AUC against
`sklearn.metrics.roc_auc_score`. It also asserts the whole pipeline is
reproducible under a seed.

## Reproducing the results

```bash
./scripts/run_experiments.sh      # main grid + every ablation in RESULTS.md
```

## Requirements

Python ≥ 3.10, NumPy, pandas, matplotlib, PyYAML. TensorFlow is **optional** —
only needed for `--backend keras`.

## Notes and limitations

- **The dataset is simulated.** No public connected-vehicle sensor capture with
  the exact feature set was available, so Module 1 generates one from a
  physical model of tyre behaviour. It is calibrated to be realistic
  (`P ∝ T` at constant volume, ~5 psi gain from 28 °C to 60 °C) and to give
  the >99% clean baseline the project specifies, but it is not a substitute
  for a field capture. `--csv` swaps in a real one.
- **KCD assumes the classes are recoverable as clusters.** The whole defence
  rests on `k = 2` k-means finding the benign/malicious split in feature
  space. Where the classes are multi-modal or heavily overlapping, cluster
  assignment degrades and so does the repair.
- **Validation and test labels are never poisoned.** The attack targets the
  training split, matching the threat model; the defender is assumed to hold a
  clean holdout for model selection, and a small trusted anchor set
  (5% by default) for cluster-to-class assignment at high poisoning
  intensities.

## Licence

MIT.
