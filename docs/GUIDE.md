# LFD-IDS — a plain guide

This is the "read me first" document. It explains what the project does, how
to run it, and how to read what comes out — without assuming you already know
the terminology.

[ARCHITECTURE.md](ARCHITECTURE.md) covers the design in depth, and
[RESULTS.md](RESULTS.md) has the full experimental tables. Start here.

---

## 1. What problem is this about?

Modern cars send sensor readings — tyre temperature, tyre pressure, GPS
position — over a mobile network to a server. The server runs an **Intrusion
Detection System (IDS)**: a machine-learning model that looks at each reading
and decides *"is this normal, or is this a cyberattack?"*

To learn that, the model needs training examples that are already marked
`benign` or `malicious`. Those marks are called **labels**.

**The vulnerability:** the training data comes from thousands of cars over a
public network. If an attacker can flip some of those labels before training —
marking attacks as normal and normal readings as attacks — the model learns the
wrong lesson. It gets trained to ignore the very attacks it was built to catch.

That is a **label-flipping data poisoning attack**, and it is what this project
builds, measures, and then defends against.

### The two questions this project answers

1. **How much damage can label flipping actually do?** Not in theory — measured,
   on a working detector.
2. **Can you repair the damage without knowing which labels were flipped?**

Short answers: it can take a 99% detector down to a coin flip, and yes — by
using the *shape* of the data instead of trusting its labels.

---

## 2. The idea in one page

### The attack

The attacker picks a fraction of the training rows — 30%, 40% or 50% — and
flips each of their labels. `benign` becomes `malicious`, `malicious` becomes
`benign`. That fraction is called the **poisoning intensity**, written `p`.

The project implements two ways of choosing *which* rows to flip:

| Name | How it picks rows |
| --- | --- |
| **BOOTLFA** | Draws random samples of the data repeatedly (*bootstrapping*) and flips the rows that keep coming up. Corruption ends up clumped. |
| **BAGLFA** | Splits the data into overlapping chunks (*bagging*) and spreads the flips evenly across all of them. Corruption ends up smooth. |

Both flip the same *number* of labels — they differ only in *which* ones. That
is deliberate, so the two can be compared fairly.

### The defence — KCD

The defence is called **KCD** (K-means Clustering Defence). Its key insight:

> The attacker changed the **labels**, but not the **sensor readings**.

So the readings still carry the truth. Attacked readings genuinely look
different from normal ones — a tyre reporting 120 °C and 15 psi at the same
time is not physically possible. The labels lie; the numbers don't.

KCD therefore ignores the labels and works from the numbers:

1. **Group the data into 2 clusters** using k-means. Because normal and
   attacked readings really do look different, one cluster fills up with normal
   readings and the other with attacked ones.
2. **Decide which cluster is which.** Mostly by majority vote of the labels
   inside each cluster — but at `p = 50%` that vote is a coin flip, so the
   defence falls back to a small **trusted set** of labels known to be genuine
   (5% by default). This matters a lot; see §8.
3. **Measure how far each reading sits from its cluster's centre.**
4. **Fix the labels closest to the centre.** Those are the ones we're most
   confident about, so they get their cluster's label — overwriting whatever the
   attacker put there.
5. **Throw away the far-out ones.** For readings far from any centre, the
   defence can't tell — so it drops them rather than guessing.
6. **Retrain the detector** on the repaired data.

Applied to BOOTLFA it's called **BOOTKCD**; to BAGLFA, **BAGKCD**.

### Why step 5 is not cheating

It looks like the defence is discarding data to make its numbers look good. It
isn't — it's the opposite. KCD *declined to relabel* those rows because it
wasn't confident. Feeding them to the retrainer anyway would mean handing the
model labels the defence itself doesn't believe. Dropping them costs about 40%
of the training rows and is still worth 10–13 accuracy points. Section 4 of
[RESULTS.md](RESULTS.md) measures all three options side by side.

---

## 3. Install

You need **Python 3.10 or newer**. Then:

```bash
git clone https://github.com/haneeshvarma2006-lab/LFD-IDS.git
cd LFD-IDS
pip install -e .
```

That's it. No GPU, no TensorFlow, no dataset to download — the project
generates its own data.

To also run the tests:

```bash
pip install -e ".[dev]"
pytest -q          # 114 tests, about 20 seconds
```

---

## 4. Your first run

```bash
lfd-ids demo
```

Takes about **12 seconds** and prints a table like this:

```
scenario   attack    defence       p      acc     prec      rec       f1      fnr      auc
------------------------------------------------------------------------------------------
baseline   -         -          0.00   0.9983   1.0000   0.9966   0.9983   0.0034   1.0000
attacked   BOOTLFA   -          0.30   0.9781   1.0000   0.9562   0.9776   0.0438   0.9995
defended   BOOTLFA   BOOTKCD    0.30   0.9899   1.0000   0.9798   0.9898   0.0202   0.9993
attacked   BOOTLFA   -          0.50   0.7643   1.0000   0.5286   0.6916   0.4714   0.6621
defended   BOOTLFA   BOOTKCD    0.50   0.9899   1.0000   0.9798   0.9898   0.0202   0.9992
attacked   BAGLFA    -          0.30   0.9815   1.0000   0.9630   0.9811   0.0370   0.9995
defended   BAGLFA    BAGKCD     0.30   0.9916   1.0000   0.9832   0.9915   0.0168   0.9994
attacked   BAGLFA    -          0.50   0.3535   0.4044   0.6195   0.4894   0.3805   0.4680
defended   BAGLFA    BAGKCD     0.50   0.9916   1.0000   0.9832   0.9915   0.0168   0.9995
```

**How to read it, one row at a time:**

- `baseline` — the detector with nobody attacking it. **99.83%**. This is the
  target to get back to.
- `attacked … 0.50` for BAGLFA — **35.35%**. Half the training labels flipped,
  and the detector is now worse than guessing.
- `defended … 0.50` — **99.16%**. KCD repaired the labels and the detector came
  back.

That contrast is the whole project.

> **Note:** `demo` uses a deliberately small dataset so it finishes fast, so its
> numbers bounce around between runs. The real experiment (§6) uses nearly 7×
> more data and averages 3 runs.

---

## 5. The commands

| Command | What it does |
| --- | --- |
| `lfd-ids demo` | Fast end-to-end run. Start here. |
| `lfd-ids run` | The full experiment: baseline → both attacks → both defences. |
| `lfd-ids generate-data --out data.csv` | Just make the sensor dataset and save it as CSV. |
| `lfd-ids attack` | Show what the attacks do, without defending. |
| `lfd-ids defend` | Show how many flipped labels the defence recovers. |
| `lfd-ids figures` | Redraw the charts from a finished run. |

Every command accepts `-o DIR` to choose where output goes, and `--set key=value`
to change any setting (see §7).

---

## 6. The full experiment

```bash
lfd-ids run -c configs/default.yaml --repeats 3 -o results
```

This takes about **an hour** on 4 CPU cores. It runs the whole grid — both
attacks × three intensities × their defences — three times with different
random seeds, and averages the results so you can see which differences are
real and which are noise.

For a faster version (about 5 minutes), skip the ensemble training:

```bash
lfd-ids run --repeats 1 --set attack.train_ensemble=false -o results
```

### What you get

```
results/
├── results.csv               one row per experiment, per seed
├── summary.csv               the same, averaged across seeds
├── experiment_report.json    everything, in full detail
└── figures/
    ├── accuracy_vs_intensity.png    ← the main chart
    ├── fnr_vs_intensity.png         ← missed attacks
    ├── metric_grid.png              ← all six metrics
    ├── confusion_matrices.png       ← clean vs poisoned vs defended
    ├── label_recovery.png           ← how many labels got fixed
    └── feature_space.png            ← what the data looks like
```

To reproduce every table in [RESULTS.md](RESULTS.md), including the ablations:

```bash
./scripts/run_experiments.sh
```

---

## 7. Understanding the numbers

Each row of the results is one trained detector, scored on a **clean test set**
the attacker never touched.

| Column | Meaning | Good value |
| --- | --- | --- |
| `acc` | Accuracy — fraction of readings classified correctly | near 1.0 |
| `prec` | Precision — of the alarms raised, how many were real attacks | near 1.0 |
| `rec` | Recall — of the real attacks, how many were caught | near 1.0 |
| `f1` | Balance of precision and recall | near 1.0 |
| **`fnr`** | **False-negative rate — attacks that got through** | **near 0.0** |
| `auc` | How well the model ranks attacks above normal readings | near 1.0 |

**`fnr` is the one that matters most here.** A false negative is a real
cyberattack the IDS waved through. In the full experiment it rises from 0.98%
(clean) to 53% (poisoned) and back down to 2.8% (defended) — the defence cuts
missed attacks by about 19×.

**Two values worth recognising:**

- **`acc ≈ 0.50`** means the detector is guessing. The dataset is half benign
  and half malicious, so 50% is what a coin gets.
- **`auc < 0.50`** is worse than guessing — the model has learned the pattern
  *backwards*. This actually happens at `p = 50%`.

---

## 8. Four things the experiments showed

Full numbers in [RESULTS.md](RESULTS.md); this is the summary.

**1. At 50% poisoning the detector collapses.** From 99.45% down to 42.6%
(BAGLFA) and 59.8% (BOOTLFA). At exactly half the labels flipped, the training
data carries no usable signal at all.

**2. Below 50%, random flipping does surprisingly little damage.** At `p = 30%`
the attacked detector still scores 98–99%. That's not a bug in the attack — it's
a real property of *random* flipping. As long as fewer than half the labels in
any region are wrong, the majority is still right, so the model finds roughly
the correct boundary anyway.

**But that protection vanishes if the attacker aims.** Flipping only
`malicious → benign` — same number of flips — drops the detector to **50.6% at
`p = 30%`**, the same damage random flipping needs 50% for. **The lesson: how
many labels were flipped tells you much less than which ones.**

**3. The defence works, and it is cheap to leave on.** 98.5–99.0% recovery at
every intensity, against both attacks, and against the aimed attack too.
Running it when there's *no* attack costs only 0.8 accuracy points — so you can
leave it permanently enabled rather than trying to detect when you're under
attack.

**4. The trusted set is load-bearing.** At `p = 50%` the majority vote inside
each cluster is an exact coin flip. With the trusted set disabled, the defence
sometimes picks the mapping *backwards* and drives the detector to **4.5%
accuracy — worse than the attack it was repairing**. A defence that can invert
your detector is not a partial defence. If you present this project, this is
the honest limitation to lead with.

---

## 9. Changing things

All settings live in [`configs/default.yaml`](../configs/default.yaml), and any
of them can be overridden from the command line:

```bash
# train for fewer epochs (faster, slightly worse)
lfd-ids run --set model.epochs=50

# try different poisoning intensities
lfd-ids run --set attack.intensities=0.1,0.2,0.3,0.6

# run the aimed attack instead of the random one
lfd-ids run --set attack.selection=malicious_to_benign

# use the literal relabel-only rule instead of dropping uncertain rows
lfd-ids run --set defence.outer_policy=keep

# turn off the trusted set and watch the defence become unreliable
lfd-ids run --set defence.label_assignment=majority --set defence.trusted_fraction=0.0

# use TensorFlow/Keras instead of the built-in NumPy trainer
pip install -e ".[keras]"
lfd-ids run --backend keras
```

### Settings worth knowing

| Setting | Default | What it controls |
| --- | --- | --- |
| `acquisition.n_vehicles` | 40 | How many cars to simulate |
| `acquisition.samples_per_vehicle` | 500 | Readings per car |
| `attack.intensities` | 0.3, 0.4, 0.5 | Poisoning levels to test |
| `attack.selection` | `random` | `random` or aimed (`malicious_to_benign`) |
| `defence.outer_policy` | `drop` | What to do with uncertain rows |
| `defence.trusted_fraction` | 0.05 | Size of the trusted label set |
| `model.epochs` | 200 | Training length |
| `model.backend` | `auto` | `numpy` (default) or `keras` |

---

## 10. Using your own data

The simulator is the default, but the whole pipeline runs on a real CSV:

```bash
lfd-ids run --csv path/to/your_data.csv
```

Your CSV needs one column per sensor plus a label column:

```csv
tyre_temperature,tyre_pressure,latitude,longitude,label
45.2,36.1,12.98,77.61,0
118.7,17.3,12.99,77.62,1
```

- Labels must be binary (`0`/`1`, or any two values).
- Different column names? Point the config at them:
  `--set acquisition.feature_set=temp,psi,lat,lon --set acquisition.csv_label_column=is_attack`
- Optional `attack_type` and `vehicle_id` columns unlock the per-attack-type
  breakdown and the alert module's per-vehicle tracking.

---

## 11. Where the code lives

The code is organised as the four modules of the system architecture:

```
src/lfd_ids/
├── module1_acquisition/     Making the data
│   ├── sensors.py             simulated car sensors + the 5 attack types
│   ├── v2x.py                 the 5G/4G link (packet loss, delays)
│   ├── ingestion.py           server-side validation
│   └── database.py            stores it all in SQLite
├── module2_attack/          Poisoning the labels
│   ├── bootlfa.py             the BOOTLFA attack
│   ├── baglfa.py              the BAGLFA attack
│   └── attacker.py            what the attacker can and can't do
├── module3_ids/             The detector
│   ├── preprocessing.py       splitting and scaling the data
│   ├── model.py               the neural network
│   ├── evaluate.py            accuracy, FNR, AUC and the rest
│   └── alert.py               raising alarms
├── module4_defense/         Repairing the damage
│   ├── kmeans.py              clustering
│   ├── kcd.py                 the KCD defence
│   ├── variants.py            BOOTKCD and BAGKCD
│   └── retrain.py             retrain, check, deploy
├── pipeline.py              runs the whole experiment
└── cli.py                   the lfd-ids command
```

### The detector itself

A small neural network, exactly as the project specifies:

```
4 sensor readings in
  → 64 neurons (relu)
  → 32 neurons (tanh)
  → 16 neurons (relu)
  →  8 neurons (tanh)
  →  1 output  (sigmoid)  → probability this reading is an attack
```

3,073 trainable parameters — small enough to train in seconds on a laptop CPU.

---

## 12. Troubleshooting

**`lfd-ids: command not found`**
The install didn't finish, or you're in a different Python environment. Re-run
`pip install -e .` from the project folder. You can always fall back to
`python -m lfd_ids.cli` instead of `lfd-ids`.

**`lfd-ids run` is taking forever**
It's training 73 neural networks per seed. Use
`--set attack.train_ensemble=false` to skip the ensemble models — that's about
90% of the work — or `--repeats 1` for a single seed.

**The numbers don't match RESULTS.md exactly**
Check you're using the full config (`-c configs/default.yaml`) and 3 repeats.
`demo` uses a much smaller dataset. At `p = 50%` results genuinely vary a lot
between seeds — that's finding #1, not a mistake.

**`No space left on device`**
Delete old run folders: `rm -rf results/ablation_*`.

**Can I run this without TensorFlow?**
Yes — that's the default. The NumPy trainer is built in and is actually about
8× faster for a network this small. TensorFlow is only needed for
`--backend keras`.

---

## 13. If you have to present this

Questions you're most likely to be asked, and honest answers:

**"Is the data real?"**
No — it's simulated, because no public dataset had these exact sensors. But it
isn't arbitrary: benign tyre pressure follows the ideal gas law against tyre
temperature (a tyre at 34 psi and 28 °C reads about 39 psi at 60 °C), and the
attacks break that physical relationship. There's a test that verifies attacked
readings really do violate it. And `--csv` swaps in real data.

**"Why does 50% poisoning break it so completely?"**
Because at exactly half the labels flipped, the labels become statistically
independent of the truth. There is no signal left to learn. Below 50% a
majority still points the right way, which is why 30% barely hurts.

**"Isn't dropping data cheating?"**
No — see §2. The defence drops exactly the rows it wasn't confident enough to
relabel. Keeping them means training on labels the defence itself distrusts.
Section 4 of RESULTS.md measures keeping, down-weighting and dropping side by
side.

**"What's the weakest part?"**
Two things, and say them before you're asked. First, KCD only works if the two
classes actually form two clusters — if they overlap heavily, it fails. Second,
the trusted label set is doing critical work: without it the defence is a coin
flip at `p = 50%` and can invert the detector entirely (§8, finding 4).

**"What would you do next?"**
Test on a real vehicle capture; handle more than two classes; and look at
attackers who poison the *sensor values* as well as the labels, since KCD's
entire premise is that the values are still trustworthy.
