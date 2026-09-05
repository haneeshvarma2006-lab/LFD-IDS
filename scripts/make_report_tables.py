#!/usr/bin/env python3
"""Render the result tables of docs/RESULTS.md straight from a run directory.

Reading numbers out of ``results.csv`` rather than transcribing them by hand
keeps the report and the experiment in sync.

Usage:
    python scripts/make_report_tables.py results [results/ablation_outer_keep ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("accuracy", "precision", "recall", "f1", "fnr", "auc")


def load(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        raise SystemExit(f"no results.csv in {run_dir}")
    frame = pd.read_csv(path)
    frame["attack"] = frame["attack"].fillna("")
    frame["defence"] = frame["defence"].fillna("")
    return frame


def _cell(values: pd.Series, digits: int = 4) -> str:
    """``mean`` for a single seed, ``mean ± std`` when several were run."""
    values = values.dropna()
    if values.empty:
        return "n/a"
    if values.size == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=0):.{digits}f}"


def main_table(frame: pd.DataFrame) -> str:
    """The headline grid: baseline, each attack, each defence, per intensity."""
    lines = [
        "| Scenario | Attack | Defence | p | Accuracy | Precision | Recall | F1 | FNR | AUC |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = {"baseline": 0, "attacked": 1, "defended": 2}
    grouped = frame.groupby(["scenario", "attack", "defence", "intensity"], dropna=False)
    rows = sorted(
        grouped,
        key=lambda kv: (kv[0][1], kv[0][3], order.get(kv[0][0], 9)),
    )
    for (scenario, attack, defence, p), group in rows:
        cells = " | ".join(_cell(group[m]) for m in METRICS)
        lines.append(
            f"| {scenario} | {attack or '—'} | {defence or '—'} | "
            f"{p:.0%} | {cells} |"
        )
    return "\n".join(lines)


def label_recovery_table(run_dir: str | Path) -> str:
    """How many of the adversary's flips each defence actually undid."""
    import json

    payload = json.loads((Path(run_dir) / "experiment_report.json").read_text())
    rows: dict[tuple, list[dict]] = {}
    for report in payload:
        for record in report["records"]:
            recovery = record.get("extras", {}).get("label_recovery")
            if not recovery:
                continue
            key = (record["attack"], record["defence"], record["intensity"])
            rows.setdefault(key, []).append(
                {**recovery, "method": record["extras"]["defence"]["assignment_method"]}
            )
    lines = [
        "| Attack | Defence | p | Label error before | after | Flips corrected | "
        "Newly corrupted | Cluster mapping |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (attack, defence, p), entries in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        before = np.mean([e["label_error_before"] for e in entries])
        after = np.mean([e["label_error_after"] for e in entries])
        rate = np.mean([e["correction_rate"] for e in entries])
        broke = np.mean([e["newly_corrupted"] for e in entries])
        methods = sorted({e["method"] for e in entries})
        lines.append(
            f"| {attack} | {defence} | {p:.0%} | {before:.3f} | {after:.3f} | "
            f"{rate:.1%} | {broke:.0f} | {', '.join(methods)} |"
        )
    return "\n".join(lines)


def ensemble_table(run_dir: str | Path) -> str:
    """The per-subset models M_1..M_B and their aggregator."""
    import json

    payload = json.loads((Path(run_dir) / "experiment_report.json").read_text())
    rows: dict[tuple, list[dict]] = {}
    for report in payload:
        for record in report["records"]:
            ensemble = record.get("extras", {}).get("ensemble")
            if not ensemble:
                continue
            rows.setdefault((record["attack"], record["intensity"]), []).append(
                {**ensemble, "single": record["accuracy"]}
            )
    if not rows:
        return "_(no ensembles were trained in this run)_"
    lines = [
        "| Attack | p | Aggregation | Members | Mean member accuracy | "
        "Aggregated accuracy | Single model on D′ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (attack, p), entries in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines.append(
            f"| {attack} | {p:.0%} | {entries[0]['aggregation']} | "
            f"{entries[0]['n_members']} | "
            f"{np.mean([e['member_accuracy_mean'] for e in entries]):.4f} ± "
            f"{np.mean([e['member_accuracy_std'] for e in entries]):.4f} | "
            f"{np.mean([e['accuracy'] for e in entries]):.4f} | "
            f"{np.mean([e['single'] for e in entries]):.4f} |"
        )
    return "\n".join(lines)


def comparison_table(run_dirs: list[str]) -> str:
    """Defended accuracy across ablation runs, side by side."""
    frames = {Path(d).name: load(d) for d in run_dirs}
    intensities = sorted(
        {p for f in frames.values() for p in f.loc[f.scenario == "defended", "intensity"]}
    )
    header = "| Run | Attack | " + " | ".join(f"p = {p:.0%}" for p in intensities) + " |"
    lines = [header, "| --- | --- |" + " --- |" * len(intensities)]
    for name, frame in frames.items():
        for attack in sorted(frame.loc[frame.scenario == "defended", "attack"].unique()):
            cells = []
            for p in intensities:
                sel = frame[
                    (frame.scenario == "defended")
                    & (frame.attack == attack)
                    & (np.isclose(frame.intensity, p))
                ]
                cells.append(_cell(sel["accuracy"]) if len(sel) else "n/a")
            lines.append(f"| {name} | {attack} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    primary = argv[0]
    frame = load(primary)
    print(f"## Main grid ({primary})\n")
    print(main_table(frame))
    print(f"\n## Label recovery\n")
    print(label_recovery_table(primary))
    print(f"\n## Attack ensembles (M_1..M_B)\n")
    print(ensemble_table(primary))
    if len(argv) > 1:
        print(f"\n## Defended accuracy across runs\n")
        print(comparison_table(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
