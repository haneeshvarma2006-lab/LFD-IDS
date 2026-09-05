#!/usr/bin/env bash
# Reproduce every table in docs/RESULTS.md.
#
# Usage:  ./scripts/run_experiments.sh [output_dir]
#
# Runs the main grid plus the four ablations. Each writes its own directory
# containing results.csv, summary.csv, experiment_report.json and figures/.
set -euo pipefail

OUT="${1:-results}"
CONFIG="configs/default.yaml"
SEEDS="${SEEDS:-3}"

# The ablations vary the attack or the defence, not the ensemble aggregator,
# so they skip training M_1..M_B -- which is ~90% of a run's cost. Set
# ABLATION_EXTRA="" to train the ensembles there too.
ABLATION_EXTRA="${ABLATION_EXTRA:---set attack.train_ensemble=false}"

# Runtime on 4 CPU cores, NumPy backend: roughly an hour for the main grid at
# SEEDS=3, plus about 10 minutes per ablation. Lower SEEDS to shorten it.

cd "$(dirname "$0")/.."

if ! command -v lfd-ids >/dev/null 2>&1; then
    echo "lfd-ids not on PATH; run 'pip install -e .' first." >&2
    exit 1
fi

run() {
    local name="$1"; shift
    echo
    echo "=============================================================="
    echo "  $name  ->  $OUT/$name"
    echo "=============================================================="
    lfd-ids run -c "$CONFIG" --repeats "$SEEDS" -o "$OUT/$name" "$@"
}

ablation() {
    local name="$1"; shift
    # shellcheck disable=SC2086
    run "$name" $ABLATION_EXTRA "$@"
}

# 1. Main grid: BOOTLFA and BAGLFA at p = 30/40/50%, answered by BOOTKCD/BAGKCD.
run main

# 2. Outer-shell ablation: what the defence achieves under the literal
#    relabel-only rule, and under distance-weighted retraining.
ablation ablation_outer_keep       --set defence.outer_policy=keep
ablation ablation_outer_downweight --set defence.outer_policy=downweight

# 3. Directed attacks: the adversary aims its flip budget at one class
#    instead of spending it uniformly.
ablation ablation_directed_m2b --set attack.selection=malicious_to_benign
ablation ablation_directed_b2m --set attack.selection=benign_to_malicious

# 4. Cluster-to-class assignment: majority vote alone, with no trusted anchors,
#    which is the regime where p = 50% becomes a coin flip.
ablation ablation_majority_only \
    --set defence.label_assignment=majority \
    --set defence.trusted_fraction=0.0

# 5. The cost of the defence when there is no attack: p = 0% means nothing is
#    poisoned, so this measures what KCD gives up on a clean training set.
ablation ablation_no_attack --set attack.intensities=0.0

# 6. Backend parity: the same grid on TensorFlow/Keras, if it is installed.
if python3 -c "import tensorflow" >/dev/null 2>&1; then
    ablation backend_keras --backend keras
else
    echo
    echo "TensorFlow not installed - skipping the Keras parity run."
    echo "Install it with: pip install -e '.[keras]'"
fi

echo
echo "All runs complete. Results under $OUT/"
