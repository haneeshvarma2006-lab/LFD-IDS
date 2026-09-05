"""Result figures for the LFD-IDS experiments.

Renders the plots that carry the project's findings:

1. ``accuracy_vs_intensity`` -- IDS accuracy against poisoning intensity,
   attacked versus defended, per attack.
2. ``fnr_vs_intensity`` -- the same for the false-negative rate, the metric
   that matters for a safety system.
3. ``metric_grid`` -- accuracy / precision / recall / F1 / FNR / AUC together.
4. ``confusion_matrices`` -- baseline, worst attacked and its defended
   counterpart side by side.
5. ``label_recovery`` -- training-label error before and after the defence.
6. ``feature_space`` -- the acquired dataset in sensor space, showing why the
   classes are clusterable and hence why KCD can work.  Drawn from the sample
   each report stores, so it can be redrawn without re-running acquisition.

Matplotlib only -- no seaborn, no styling dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import ensure_dir, get_logger

LOGGER = get_logger("lfd_ids.figures")

ATTACK_COLOURS = {"BOOTLFA": "#c0392b", "BAGLFA": "#8e44ad"}
DEFENDED_COLOUR = "#1e8449"
BASELINE_COLOUR = "#2c3e50"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rows_for(rows: Sequence[dict], scenario: str, attack: str | None = None) -> list[dict]:
    out = [r for r in rows if r["scenario"] == scenario]
    if attack is not None:
        out = [r for r in out if (r.get("attack") or "") == attack]
    return sorted(out, key=lambda r: r["intensity"])


def _mean_by_intensity(rows: Sequence[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated seeds into ``(intensities, means, stds)``."""
    by: dict[float, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        by.setdefault(float(row["intensity"]), []).append(float(value))
    if not by:
        return np.array([]), np.array([]), np.array([])
    xs = np.array(sorted(by))
    means = np.array([np.mean(by[x]) for x in xs])
    stds = np.array([np.std(by[x]) for x in xs])
    return xs, means, stds


def _attacks_in(rows: Sequence[dict]) -> list[str]:
    return sorted({r["attack"] for r in rows if r.get("attack")})


def _metric_vs_intensity(
    rows: Sequence[dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    lower_is_better: bool = False,
) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    baseline_rows = _rows_for(rows, "baseline")
    if baseline_rows:
        value = float(np.mean([r[metric] for r in baseline_rows]))
        ax.axhline(
            value,
            color=BASELINE_COLOUR,
            linestyle="--",
            linewidth=1.4,
            label=f"clean baseline ({value:.3f})",
        )
    for attack in _attacks_in(rows):
        colour = ATTACK_COLOURS.get(attack, "#d35400")
        xs, means, stds = _mean_by_intensity(_rows_for(rows, "attacked", attack), metric)
        if xs.size:
            ax.errorbar(xs * 100, means, yerr=stds, marker="o", capsize=3,
                        color=colour, label=f"{attack} (attacked)")
        dx, dmeans, dstds = _mean_by_intensity(_rows_for(rows, "defended", attack), metric)
        if dx.size:
            ax.errorbar(dx * 100, dmeans, yerr=dstds, marker="s", capsize=3,
                        color=colour, alpha=0.85, linestyle=":",
                        markerfacecolor="white", label=f"{attack} + KCD defence")
    ax.set_xlabel("poisoning intensity p (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="lower left" if not lower_is_better else "upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def accuracy_vs_intensity(rows: Sequence[dict], out_dir: Path) -> Path:
    return _metric_vs_intensity(
        rows,
        "accuracy",
        "test accuracy (clean test set)",
        "IDS accuracy under label-flipping poisoning",
        out_dir / "accuracy_vs_intensity.png",
    )


def fnr_vs_intensity(rows: Sequence[dict], out_dir: Path) -> Path:
    return _metric_vs_intensity(
        rows,
        "fnr",
        "false-negative rate (missed intrusions)",
        "Missed intrusions under poisoning, with and without KCD",
        out_dir / "fnr_vs_intensity.png",
        lower_is_better=True,
    )


def metric_grid(rows: Sequence[dict], out_dir: Path) -> Path:
    plt = _mpl()
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("fnr", "False-negative rate"),
        ("auc", "ROC AUC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        baseline_rows = _rows_for(rows, "baseline")
        if baseline_rows:
            values = [r[metric] for r in baseline_rows if np.isfinite(r.get(metric, np.nan))]
            if values:
                ax.axhline(float(np.mean(values)), color=BASELINE_COLOUR,
                           linestyle="--", linewidth=1.2)
        for attack in _attacks_in(rows):
            colour = ATTACK_COLOURS.get(attack, "#d35400")
            xs, means, _ = _mean_by_intensity(_rows_for(rows, "attacked", attack), metric)
            if xs.size:
                ax.plot(xs * 100, means, marker="o", color=colour, label=f"{attack} attacked")
            dx, dmeans, _ = _mean_by_intensity(_rows_for(rows, "defended", attack), metric)
            if dx.size:
                ax.plot(dx * 100, dmeans, marker="s", linestyle=":", color=colour,
                        markerfacecolor="white", label=f"{attack} defended")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[-1]:
        ax.set_xlabel("poisoning intensity p (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("LFD-IDS metric suite versus poisoning intensity", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = out_dir / "metric_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def confusion_matrices(rows: Sequence[dict], out_dir: Path) -> Path:
    plt = _mpl()
    baseline = _rows_for(rows, "baseline")
    attacked = _rows_for(rows, "attacked")
    if not baseline or not attacked:
        raise ValueError("need at least a baseline and one attacked run")
    worst = min(attacked, key=lambda r: r["accuracy"])
    defended = [
        r
        for r in _rows_for(rows, "defended", worst["attack"])
        if abs(r["intensity"] - worst["intensity"]) < 1e-9
    ]
    panels = [("Clean baseline", baseline[0]),
              (f"{worst['attack']} p={worst['intensity']:.0%}", worst)]
    if defended:
        panels.append((f"{defended[0]['defence']} p={worst['intensity']:.0%}", defended[0]))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (title, row) in zip(axes, panels):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        normed = cm / max(cm.sum(), 1)
        ax.imshow(normed, cmap="Blues", vmin=0, vmax=normed.max() or 1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}\n{normed[i, j]:.1%}",
                        ha="center", va="center", fontsize=11,
                        color="white" if normed[i, j] > normed.max() * 0.6 else "black")
        ax.set_xticks([0, 1], ["pred benign", "pred malicious"], fontsize=9)
        ax.set_yticks([0, 1], ["true benign", "true malicious"], fontsize=9)
        ax.set_title(f"{title}\nacc={row['accuracy']:.3f}  fnr={row['fnr']:.3f}", fontsize=10)
    fig.suptitle("Confusion matrices: clean, poisoned and defended", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def label_recovery(reports: Sequence[Any], out_dir: Path) -> Path:
    """Training-label error before and after the defence, per attack."""
    plt = _mpl()
    records: list[dict] = []
    for report in reports:
        payload = report if isinstance(report, dict) else report.as_dict()
        for rec in payload["records"]:
            recovery = rec.get("extras", {}).get("label_recovery")
            if recovery:
                records.append(
                    {
                        "attack": rec["attack"],
                        "intensity": rec["intensity"],
                        "before": recovery["label_error_before"],
                        "after": recovery["label_error_after"],
                        "corrected": recovery["correction_rate"],
                    }
                )
    if not records:
        raise ValueError("no label-recovery data in the reports")

    attacks = sorted({r["attack"] for r in records})
    intensities = sorted({r["intensity"] for r in records})
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    width = 0.8 / (2 * len(attacks))
    positions = np.arange(len(intensities))
    for a, attack in enumerate(attacks):
        before = [
            np.mean([r["before"] for r in records if r["attack"] == attack and r["intensity"] == p])
            for p in intensities
        ]
        after = [
            np.mean([r["after"] for r in records if r["attack"] == attack and r["intensity"] == p])
            for p in intensities
        ]
        offset = (2 * a - len(attacks) + 0.5) * width
        colour = ATTACK_COLOURS.get(attack, "#d35400")
        ax.bar(positions + offset, before, width, color=colour, alpha=0.85,
               label=f"{attack}: after attack")
        ax.bar(positions + offset + width, after, width, color=DEFENDED_COLOUR, alpha=0.85,
               label=f"{attack}: after defence")
    ax.set_xticks(positions, [f"{p:.0%}" for p in intensities])
    ax.set_xlabel("poisoning intensity p")
    ax.set_ylabel("fraction of training labels that are wrong")
    ax.set_title("Training-label corruption before and after the KCD defence")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "label_recovery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def feature_space_from_reports(reports: Sequence[Any], out_dir: Path) -> Path:
    """Draw the feature-space scatter from a report's saved sample."""
    for report in reports:
        payload = report if isinstance(report, dict) else report.as_dict()
        sample = payload.get("feature_sample") or {}
        if sample.get("X"):
            return feature_space(
                np.asarray(sample["X"], dtype=float),
                np.asarray(sample["y"], dtype=int),
                sample["feature_names"],
                out_dir,
            )
    raise ValueError("no feature sample stored in the reports")


def feature_space(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    out_dir: Path,
) -> Path:
    """Scatter the acquired dataset, showing the structure KCD relies on."""
    plt = _mpl()
    pairs = [(0, 1), (2, 3)] if len(feature_names) >= 4 else [(0, 1)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 5.0))
    axes = np.atleast_1d(axes)
    for ax, (i, j) in zip(axes, pairs):
        for label, colour, name in ((0, "#2980b9", "benign"), (1, "#c0392b", "malicious")):
            mask = y == label
            ax.scatter(X[mask, i], X[mask, j], s=6, alpha=0.35, color=colour, label=name)
        ax.set_xlabel(feature_names[i].replace("_", " "))
        ax.set_ylabel(feature_names[j].replace("_", " "))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, markerscale=2)
    fig.suptitle("Connected-vehicle sensor data: benign versus attacked records", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "feature_space.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_all(
    reports: Sequence[Any],
    rows: Sequence[dict],
    out_dir: str | Path,
) -> list[Path]:
    """Render every figure that the available data supports."""
    out = ensure_dir(out_dir)
    paths: list[Path] = []
    builders = [
        ("accuracy_vs_intensity", lambda: accuracy_vs_intensity(rows, out)),
        ("fnr_vs_intensity", lambda: fnr_vs_intensity(rows, out)),
        ("metric_grid", lambda: metric_grid(rows, out)),
        ("confusion_matrices", lambda: confusion_matrices(rows, out)),
        ("label_recovery", lambda: label_recovery(reports, out)),
        ("feature_space", lambda: feature_space_from_reports(reports, out)),
    ]
    for name, builder in builders:
        try:
            paths.append(builder())
        except (ValueError, KeyError, IndexError) as exc:
            LOGGER.warning("Skipping figure '%s': %s", name, exc)
    return paths
