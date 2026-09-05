# Results

All numbers below are measured on a **clean held-out test set** (3,957 records,
20% of the dataset) that the attacker never touches, averaged over **3 seeds**
(42, 43, 44) and reported as mean ± standard deviation.

Reproduce with:

```bash
lfd-ids run -c configs/default.yaml --repeats 3 -o results
python scripts/make_report_tables.py results
```

**Setup.** 40 vehicles × 500 readings → 19,782 records after channel loss and
ingestion validation, balanced 50/50 benign/malicious. Feature vector: tyre
temperature, tyre pressure, latitude, longitude. Split 60/20/20; only the
**training** labels are poisoned. The IDS is the specified sequential MLP
(64-relu / 32-tanh / 16-relu / 8-tanh / 1-sigmoid, 3,073 parameters) trained
for 200 epochs with Adam, no early stopping — deliberately the *undefended*
pipeline the project is characterising.

Because the classes are balanced, **50% accuracy is exactly the random-guessing
floor**, which is what makes the collapse at p = 50% legible.

---

## 1. The main grid

| Scenario | Attack | Defence | p | Accuracy | Precision | Recall | F1 | FNR | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | — | — | 0% | 0.9945 ± 0.0004 | 0.9988 ± 0.0009 | 0.9902 ± 0.0009 | 0.9945 ± 0.0004 | 0.0098 ± 0.0009 | 0.9990 ± 0.0001 |
| attacked | BAGLFA | — | 30% | 0.9832 ± 0.0030 | 1.0000 ± 0.0000 | 0.9663 ± 0.0060 | 0.9829 ± 0.0031 | 0.0337 ± 0.0060 | 0.9954 ± 0.0017 |
| defended | BAGLFA | BAGKCD | 30% | 0.9864 ± 0.0016 | 0.9997 ± 0.0005 | 0.9732 ± 0.0031 | 0.9863 ± 0.0016 | 0.0268 ± 0.0031 | 0.9976 ± 0.0009 |
| attacked | BAGLFA | — | 40% | 0.9780 ± 0.0038 | 0.9957 ± 0.0057 | 0.9603 ± 0.0099 | 0.9776 ± 0.0040 | 0.0397 ± 0.0099 | 0.9894 ± 0.0017 |
| defended | BAGLFA | BAGKCD | 40% | 0.9853 ± 0.0009 | 0.9997 ± 0.0005 | 0.9710 ± 0.0020 | 0.9851 ± 0.0009 | 0.0290 ± 0.0020 | 0.9960 ± 0.0019 |
| attacked | BAGLFA | — | 50% | 0.4259 ± 0.1463 | 0.5837 ± 0.2963 | 0.4654 ± 0.1897 | 0.4369 ± 0.0659 | 0.5346 ± 0.1897 | 0.3830 ± 0.0033 |
| defended | BAGLFA | BAGKCD | 50% | 0.9855 ± 0.0016 | 0.9988 ± 0.0017 | 0.9722 ± 0.0040 | 0.9853 ± 0.0017 | 0.0278 ± 0.0040 | 0.9980 ± 0.0009 |
| attacked | BOOTLFA | — | 30% | 0.9878 ± 0.0026 | 0.9976 ± 0.0017 | 0.9779 ± 0.0067 | 0.9877 ± 0.0027 | 0.0221 ± 0.0067 | 0.9940 ± 0.0032 |
| defended | BOOTLFA | BOOTKCD | 30% | 0.9903 ± 0.0021 | 0.9998 ± 0.0002 | 0.9808 ± 0.0039 | 0.9902 ± 0.0021 | 0.0192 ± 0.0039 | 0.9970 ± 0.0016 |
| attacked | BOOTLFA | — | 40% | 0.9790 ± 0.0020 | 1.0000 ± 0.0000 | 0.9581 ± 0.0040 | 0.9786 ± 0.0021 | 0.0419 ± 0.0040 | 0.9823 ± 0.0077 |
| defended | BOOTLFA | BOOTKCD | 40% | 0.9863 ± 0.0021 | 0.9998 ± 0.0002 | 0.9727 ± 0.0042 | 0.9861 ± 0.0021 | 0.0273 ± 0.0042 | 0.9972 ± 0.0015 |
| attacked | BOOTLFA | — | 50% | 0.5979 ± 0.1262 | 0.6983 ± 0.2299 | 0.6038 ± 0.2038 | 0.5901 ± 0.1046 | 0.3962 ± 0.2038 | 0.6687 ± 0.1396 |
| defended | BOOTLFA | BOOTKCD | 50% | 0.9870 ± 0.0015 | 0.9993 ± 0.0010 | 0.9747 ± 0.0023 | 0.9869 ± 0.0016 | 0.0253 ± 0.0023 | 0.9961 ± 0.0019 |

![accuracy](../results/figures/accuracy_vs_intensity.png)

### What this says

**The clean IDS works.** 99.45% accuracy, 0.98% false-negative rate, AUC
0.9990. That is the detector the fleet would ship.

**At p = 50% it collapses.** BAGLFA drives accuracy to 42.6% and BOOTLFA to
59.8% — either side of the random-guessing floor — with AUC at 0.383 and 0.669.
An AUC *below* 0.5 (BAGLFA) means the model has learned an actively inverted
ranking: it is worse than a coin flip. The false-negative rate rises from 1% to
**53%** (BAGLFA) and **40%** (BOOTLFA): roughly every second intrusion walks
straight through.

**Those p = 50% numbers are wildly unstable** — ±14.6 and ±12.6 points across
just three seeds, the largest variance anywhere in the grid. That is a finding,
not noise to average away. At exactly 50% inversion the training labels carry
no class signal at all, so what the model latches onto is whichever accidental
structure the poisoned labels happen to contain that run. A single-seed
experiment here would be close to meaningless, which is why the grid is run
three times.

**The defence restores the detector at every intensity**: 98.5–99.0% accuracy,
FNR back down to 2.5–2.9%, AUC ≥ 0.996. At p = 50% BAGKCD takes accuracy from
42.6% to 98.6% and cuts missed intrusions from 53% to 2.8% — a **19× reduction
in false negatives**, which is the number that matters for a safety system.

**Between 30% and 40% the attack barely bites, and so the defence has little to
undo** (98.3% → 98.6% for BAGLFA at p = 30%). Section 3 explains why, and why
that is not the reassurance it looks like.

![false negatives](../results/figures/fnr_vs_intensity.png)

---

## 2. How much of the poisoning the defence actually undoes

| Attack | Defence | p | Label error before | after | Flips corrected | Newly corrupted | Cluster mapping |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAGLFA | BAGKCD | 30% | 0.300 | 0.130 | 57.7% | 40 | majority |
| BAGLFA | BAGKCD | 40% | 0.400 | 0.171 | 58.0% | 38 | majority |
| BAGLFA | BAGKCD | 50% | 0.500 | 0.214 | 57.7% | 30 | anchor |
| BOOTLFA | BOOTKCD | 30% | 0.300 | 0.130 | 57.7% | 43 | majority |
| BOOTLFA | BOOTKCD | 40% | 0.400 | 0.171 | 58.0% | 36 | majority |
| BOOTLFA | BOOTKCD | 50% | 0.500 | 0.212 | 58.1% | 29 | anchor |

KCD corrects a **strikingly consistent ~58% of the adversary's flips**,
regardless of attack or intensity. That number is not a coincidence — it is
structural. KCD only rewrites labels of samples lying within the mean distance
`μ` of their cluster centroid, and for a roughly Gaussian cluster in 4-D that
is about 53–58% of the mass. **The published relabelling rule therefore has a
hard ceiling: it can never repair more than about three-fifths of the damage**,
no matter how good the clustering is.

The residual label error after the defence is 13% (p = 30%), 17% (p = 40%) and
21% (p = 50%). The defence also *introduces* a few errors of its own — 29 to 43
training labels out of 11,869, well under half a percent — where a sample near
a centroid genuinely belonged to the other class.

So how does a training set still carrying 21% wrong labels produce a 98.6%
detector? Because of **where** the errors are. The repaired core of each
cluster is clean; what remains corrupted is the sparse outer shell, which the
default `outer_policy: drop` removes from the retraining set entirely. Section
4 measures exactly what that choice is worth.

**Cluster-to-class mapping** switches rule at p = 50%, as designed: at 30% and
40% the majority vote inside each cluster is still decisive, so `majority`
fires; at 50% the vote is a coin flip and the defence falls back to the trusted
`anchor` set. Without that fallback the defence is unreliable at exactly 50% —
Section 4 quantifies it.

![label recovery](../results/figures/label_recovery.png)

---

## 3. Random flipping is far weaker than it looks

The grid shows only a 1.6-point drop at p = 30%. That is not a flaw in the
attack implementation — it is a real and well-understood property of
**symmetric** label noise.

If the adversary flips labels uniformly at random, then while fewer than half
of them flip, the *majority* label in every region of feature space is still
the correct one. The Bayes-optimal decision boundary does not move. A
classifier that does not overfit will find roughly the same boundary it would
have found on clean data. The damage that does appear at 30–40% comes from the
model memorising individual noisy labels late in training, not from the
boundary shifting.

**That robustness evaporates the moment the adversary aims.** Restricting the
same flip budget to one direction (`attack.selection=malicious_to_benign`)
makes the noise *asymmetric*, which does move the boundary:

| Selection | p = 30% | p = 40% | p = 50% |
| --- | --- | --- | --- |
| `random` (uniform) | 0.9689 | 0.9512 | 0.3045 |
| `malicious_to_benign` (directed) | **0.5265** | **0.5055** | **0.5004** |
| `benign_to_malicious` (directed) | **0.4996** | — | — |

*(single seed, 20 vehicles × 300 samples; see `results/ablation_directed_m2b/`
for the full 3-seed run at the default scale)*

A directed attacker reaches the random-guessing floor at **p = 30%** — the same
damage uniform flipping needs 50% for. **The practical takeaway is that
poisoning intensity alone is a poor threat measure.** An adversary who knows
which class to target needs a little over half the budget, and at 30% poisoning
an operator watching only accuracy would see a healthy-looking detector under
the random attack and a destroyed one under the directed attack.

---

## 4. What the defence's design choices are worth

KCD relabels only inside the mean-distance shell, which leaves the outer shell
holding whatever labels the adversary left. `defence.outer_policy` decides what
happens to those samples, and it dominates the defence's effectiveness:

| `outer_policy` | p = 30% | p = 40% | p = 50% | Training rows kept |
| --- | --- | --- | --- | --- |
| `keep` (literal relabel-only rule) | 0.9672 | 0.9579 | **0.8343** | 3,566 |
| `downweight` (weight 0.25) | 0.9773 | 0.9638 | 0.9142 | 3,566 |
| `drop` (default) | 0.9882 | 0.9899 | **0.9882** | 1,983 |

*(BOOTLFA, single seed at reduced scale; the 3-seed runs at full scale are in
`results/ablation_outer_keep/` and `results/ablation_outer_downweight/`)*

**Reading the published rule literally — relabel the core, keep the rest —
recovers only 83% at p = 50%.** Dropping the untrusted outer shell instead
restores 98.8%, and does so uniformly across intensities. The reasoning is
simple: KCD has no basis for trusting those labels, so treating them as ground
truth is worse than treating them as unusable. Dropping them costs 44% of the
training rows and is still the better trade by 15 points.

This is the single most consequential implementation decision in the project,
and it is why `drop` is the default. `keep` remains available as an ablation.

**What the defence costs when there is no attack.** Running KCD on a completely
clean training set (`attack.intensities=0.0`) gives 98.65% against a 99.24%
baseline — a **0.59-point cost, with no change in false-negative rate**. That
is cheap enough that the defence can be left permanently enabled rather than
switched on once poisoning is suspected, which matters because in practice you
do not know when you are being poisoned.

---

## 5. The ensemble aggregator makes things worse under heavy poisoning

Both attacks also train the per-subset models `M₁..M_B` and their aggregator,
as the architecture specifies:

| Attack | p | Aggregation | Members | Mean member accuracy | Aggregated accuracy | Single model on D′ |
| --- | --- | --- | --- | --- | --- | --- |
| BAGLFA | 30% | majority_vote | 10 | 0.9868 ± 0.0035 | 0.9910 | 0.9832 |
| BAGLFA | 40% | majority_vote | 10 | 0.9651 ± 0.0121 | 0.9805 | 0.9780 |
| BAGLFA | 50% | majority_vote | 10 | 0.2991 ± 0.1130 | 0.2414 | 0.4259 |
| BOOTLFA | 30% | soft_average | 10 | 0.9892 ± 0.0029 | 0.9933 | 0.9878 |
| BOOTLFA | 40% | soft_average | 10 | 0.9721 ± 0.0092 | 0.9896 | 0.9790 |
| BOOTLFA | 50% | soft_average | 10 | 0.4899 ± 0.1308 | 0.3730 | 0.5979 |

At 30% and 40% the aggregator behaves exactly as ensemble theory predicts —
voting beats the single model (99.10% vs 98.32% for BAGLFA at 30%; 98.96% vs
97.90% for BOOTLFA at 40%), because member errors are partly independent and
cancel.

**At 50% that reverses, sharply.** The BAGLFA aggregator scores 24.1% against
the single model's 42.6%, and BOOTLFA's 37.3% against 59.8%. Ensembling makes
the poisoned detector *worse than one model trained on the same poisoned data*.
The reason is that every member is trained on views of the **same** corrupted
label set, so their errors are correlated rather than independent — and voting
over correlated errors amplifies them instead of cancelling them. Bagging is
not a defence against label poisoning; under heavy poisoning it is a liability.

---

## 6. Detection by attack type

Per-attack-type recall on the clean baseline, from
`results/experiment_report.json` (`extras.per_attack_recall`):

| Attack type | What it forges | Records | Baseline recall |
| --- | --- | --- | --- |
| `fuzzing` | sensor values driven to their rails | 1,998 | 0.9955 |
| `thermal_injection` | impossible tyre temperature, decoupled pressure | 1,942 | 0.9933 |
| `replay_freeze` | stale frame replayed while the vehicle moves | 2,024 | 0.9920 |
| `gps_spoofing` | position displaced outside the operating region | 1,996 | 0.9857 |
| `tpms_deflation_spoof` | pressure collapse while temperature climbs | 1,938 | 0.9850 |

Every attack type is detected above 98.5%, but the spread is not arbitrary.
The easiest are the crude ones — fuzzing drives sensors to their rails, and
thermal injection reports temperatures no tyre can reach; both are detectable
from a single feature.

The two hardest, by a clear margin, are the TPMS deflation spoof (0.9850) and
GPS spoofing (0.9857) — and the deflation spoof is the most physically
plausible forgery of the five. It keeps every sensor inside its own valid
range and violates only the *relationship* between temperature and pressure.
Nothing about any single reading looks wrong; the record is only anomalous
jointly. That is precisely the structure the MLP has to learn, and it is where
the stealthy variants (`stealth_fraction`, blended back towards benign values)
land. **A more capable adversary would concentrate on exactly this kind of
physically-consistent forgery rather than the rail-slamming ones.**

---

## 7. Limitations

- **The dataset is simulated.** No public connected-vehicle capture with this
  exact feature set was available, so Module 1 generates one from a physical
  model of tyre behaviour. It is calibrated to be realistic and to give the
  >99% clean baseline the project specifies, but it is not a field capture.
  `lfd-ids run --csv <file>` swaps in a real one.
- **KCD assumes the classes are recoverable as two clusters.** The entire
  defence rests on k-means (k = 2) finding the benign/malicious split in
  feature space. Where classes are multi-modal or heavily overlapping, cluster
  assignment degrades and the repair degrades with it. This is a property of
  the published method, not of this implementation.
- **Validation and test labels are never poisoned**, matching the threat model
  (the attack targets training data). The defender is assumed to hold a clean
  holdout for model selection and a 5% trusted anchor set for cluster-to-class
  assignment. The anchor set is what makes the defence work at exactly
  p = 50%; `results/ablation_majority_only/` shows the majority-vote-only
  behaviour without it.
- **Three seeds is few** for the p = 50% cells, where the standard deviation
  reaches ±15 points. The direction of the effect is unambiguous; the exact
  value is not.

---

## Appendix: files

| File | Contents |
| --- | --- |
| `results/results.csv` | one row per (scenario, attack, defence, intensity, seed) |
| `results/summary.csv` | the same, averaged across seeds |
| `results/experiment_report.json` | full detail: poison metadata, defence internals, label recovery, per-attack recall, deployment log |
| `results/figures/` | the six figures referenced above |
| `results/ablation_*/` | the ablation runs behind Sections 3 and 4 |
