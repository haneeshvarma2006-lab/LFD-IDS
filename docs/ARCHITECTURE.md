# LFD-IDS architecture

The system reconstructs the four-module architecture of the project
specification. Data flows top to bottom; the deployment feedback closes the
loop back to the fleet.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 1  Connected Vehicle Sensor Data Acquisition                      │
│  Vehicle Sensor Suite → V2V/V2I Link → Cloud Ingestion → Cloud Database  │
│  tyre temp/pressure     5G/4G/DSRC     validation +      raw store +     │
│  GPS, speed, fuel       loss, jitter   clock sync        labelled set S  │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ clean dataset D
┌──────────────────────────────▼───────────────────────────────────────────┐
│ MODULE 2  Label-Flipping Data Poisoning Attack Engine                    │
│  BOOTLFA              BAGLFA                    Attacker Model           │
│  bootstrap resample   bagging subsets           MITM / compromised       │
│  fraction-p selector  fraction-p selector       labeller; white- or      │
│  y' = |1 − y|         y' = |1 − y|              black-box knowledge      │
│  M₁..M_B + aggregator M₁..M_B + majority vote                            │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ poisoned dataset D′
┌──────────────────────────────▼───────────────────────────────────────────┐
│ MODULE 3  Deep-Learning IDS Classification (MLP)                         │
│  Preprocessing      Sequential MLP        Evaluation        Alert        │
│  60/20/20 split     64-32-16-8-1          confusion matrix  flag trigger │
│  min-max scaling    relu/tanh/relu/tanh   acc/prec/rec/F1   model        │
│                     sigmoid output        FNR/AUC           broadcast    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ degraded-accuracy model
┌──────────────────────────────▼───────────────────────────────────────────┐
│ MODULE 4  K-Means Clustering-Based Defence (KCD)                         │
│  Clustering & Distance   Label Correction      Retrain & Deploy          │
│  k-means (k = 2)         d(xᵢ) < μ  →          retrain on D″             │
│  centroid distances      cluster label         robustness gate           │
│  mean threshold μ        BOOTKCD / BAGKCD      publish to fleet ─────────┼──┐
└──────────────────────────────────────────────────────────────────────────┘  │
                          deployed model feedback to Module 3 ◄───────────────┘
```

## Module 1 — acquisition

| Block | Code | What it does |
| --- | --- | --- |
| Vehicle Sensor Suite | `module1_acquisition/sensors.py` | Generates per-vehicle telemetry. Benign readings obey tyre physics; attacks break it. |
| V2V/V2I Communication | `module1_acquisition/v2x.py` | 5G/4G/DSRC uplink with packet loss, latency jitter and clock skew; exposes the MITM hook. |
| Cloud Ingestion Layer | `module1_acquisition/ingestion.py` | Range/format validation, timestamp re-synchronisation, record aggregation. |
| Cloud Database | `module1_acquisition/database.py` | SQLite raw store; materialises the labelled set `S = (xᵢ, yᵢ)`. |

**The benign physics.** Tyre pressure follows the ideal gas law at constant
volume, `P_abs = P₀_abs · T/T₀`, so a tyre at 34 psi and 28 °C reads about
39 psi at 60 °C. Temperature rises with sustained speed through a
first-order thermal lag, fuel falls with distance, and position tracks a
polyline route inside the fleet's operating region.

**The attacks.** Five attack types each break that structure in a different
way — `tpms_deflation_spoof` (pressure collapses while temperature climbs),
`thermal_injection` (impossible temperatures with a decoupled pressure),
`gps_spoofing` (position displaced far outside the region), `replay_freeze`
(stale frames), and `fuzzing` (values driven to the rails). Attacks arrive in
bursts, as a real intrusion does, and a configurable `stealth_fraction` of
them is blended back towards benign values so the classes are not trivially
separable.

**Feature vector.** The four features the architecture names —
`tyre_temperature`, `tyre_pressure`, `latitude`, `longitude`. `speed` and
`fuel_level` are collected too and can be switched in via
`acquisition.feature_set`.

## Module 2 — the attack engine

Both attacks are built from the two primitives the architecture names: a
**fraction-p sample selector** and the **label inversion operator**
`y' = |1 − y|` (`module2_attack/base.py`). They differ in how the flip budget
is distributed across resampled views of the training set.

**BOOTLFA** (`bootlfa.py`) draws `B` bootstrap resamples of size `N` with
replacement and inverts a fraction `p` of the rows of each. Because a
bootstrap over-represents some samples and omits about 36.8% of the training
set, the flip votes accumulated across the resamples land unevenly; BOOTLFA
keeps the samples that collected the most votes. That is the
*distribution-level* label noise of the specification.

**BAGLFA** (`baglfa.py`) generates `B` bagging subsets (a fraction `f` of the
training set, drawn without replacement) and fills the flip budget round-robin
across them, so every subset carries an equal share — the *variance-reduced*
noise profile. A sample that lands in no subset is out of reach, which is why
BAGLFA's realised flip count can fall marginally short of its budget at small
`B` or `f` (reported as `uncovered_fraction`).

**Comparability.** Both hold the global budget at `round(p·N)` labels, so at
each intensity the two attacks differ in *which* labels they invert, not how
many. Both also retain the per-subset views so `M₁..M_B` and their aggregator
(soft average for BOOTLFA, majority vote for BAGLFA) can be trained and
scored.

**Attacker model** (`attacker.py`) fixes what the adversary can do:
white-box (reads the labelled dataset) or black-box (packets only); MITM
interception or a compromised in-vehicle labeller; fleet-wide or
single-vehicle scope. A black-box attacker is refused label-directed
selection strategies, because it cannot see the labels it would need to aim.

## Module 3 — the IDS

```
Input(4) → Dense(64, relu) → Dense(32, tanh) → Dense(16, relu)
         → Dense(8, tanh) → Dense(1, sigmoid)          3,073 parameters
```

The abstract names the last four layers (32-tanh, 16-relu, 8-tanh, sigmoid);
the architecture diagram adds the Dense-64 input projection, whose activation
the specification leaves open — it defaults to `relu` and is configurable.

Two interchangeable backends implement it. **`numpy`** is the default: a
self-contained forward/backward pass with Adam and mini-batch binary
cross-entropy, bit-for-bit reproducible under a seed and roughly eight times
faster than Keras at this model size. **`keras`** builds the same topology as
a `tf.keras.Sequential` for parity with the original toolchain; both report
the same 3,073 parameters.

Preprocessing splits 60/20/20 with stratification and fits the min-max scaler
on the training split only. The split is computed once on the clean dataset
and reused for every attack and defence run, so only the training *labels*
vary across the grid.

The evaluation engine reports the confusion matrix, accuracy, precision,
recall, F1, FNR, FPR, specificity and ROC AUC. AUC is computed from the
Mann-Whitney U statistic with mid-ranks for ties and is verified against
scikit-learn in the tests.

The alert module raises a cyberattack flag only after `consecutive_trigger`
malicious classifications in a row *from the same vehicle*, and records each
model broadcast to the fleet.

## Module 4 — the defence

KCD assumes the adversary corrupted **labels but not features**. The geometry
of `D′` therefore still carries the class structure, and the labels can be
re-derived from it:

1. Cluster the poisoned training features with k-means (`k = 2`).
2. Assign each cluster a class label.
3. Measure each sample's Euclidean distance `d(xᵢ)` to its cluster centroid.
4. Estimate the mean distance `μ` (per cluster by default).
5. Relabel every sample with `d(xᵢ) < μ` to its cluster's label.

Step 5 is deliberately conservative: samples near a centroid are the ones
whose cluster membership is most trustworthy, so those are the only labels
the defence overwrites. For a roughly Gaussian cluster in 4-D that is about
53% of the mass — which caps the fraction of flips a single pass can undo.

**Cluster-to-class assignment** needs a source of truth the adversary has not
touched. `majority` votes the poisoned labels inside each cluster (the
published rule — sound while fewer than half of a cluster's labels were
inverted, a coin flip at exactly `p = 50%`). `anchor` votes a small trusted
subset the defender holds back, modelling records collected under attestation.
`auto`, the default, uses `majority` when its margin is decisive and falls
back to `anchor` otherwise; the rule that actually fired is reported as
`assignment_method`.

**The outer shell.** Samples beyond `μ` keep whatever label the adversary left
on them, and `defence.outer_policy` decides what happens to them: `drop`
(default) removes them from `D″`, `keep` reproduces the literal relabel-only
rule, `downweight` keeps them at reduced loss weight. This choice dominates
the defence's effectiveness — see [RESULTS.md](RESULTS.md).

**BOOTKCD / BAGKCD** (`variants.py`) mirror the structure of the attack they
answer: they run KCD over `B` bootstrap resamples or bagging subsets and keep,
per sample, the label that won a majority of the passes covering it, falling
back to the single-pass repair for anything uncovered.

**Retraining and deployment** (`retrain.py`) refits the IDS on `D″`, gates it
on accuracy, FNR, AUC and drift from the clean baseline, and publishes it only
if it clears the gate; rejections are recorded and the previous model stays
live.

## Reproducibility

Every stochastic component draws from a seeded `numpy.random.Generator`
threaded down from `ExperimentConfig.seed`: the sensor simulator, the channel,
the split, the weight initialisation and mini-batch shuffling, the attack's
sample selection, k-means seeding, and the trusted-anchor draw. Two runs of
the same configuration produce identical result rows, which the test suite
asserts.
