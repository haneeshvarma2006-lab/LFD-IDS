"""End-to-end orchestration of the four LFD-IDS modules.

``run_experiment`` walks the full grid the project specifies:

    clean baseline
      -> {BOOTLFA, BAGLFA} x {30%, 40%, 50%} poisoning
        -> {BOOTKCD, BAGKCD} defence + retraining + deployment gate

and returns every metric as a structured report that ``cli`` and the figure
scripts consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ExperimentConfig
from .module1_acquisition import AcquisitionResult, acquire
from .module2_attack import AttackerModel, PoisonResult
from .module3_ids import (
    AlertModule,
    DataSplit,
    Metrics,
    build_model,
    describe_architecture,
    evaluate,
    label_recovery_metrics,
    per_attack_recall,
    preprocess,
    resolve_backend,
)
from .module4_defense import (
    DEFENCE_FOR_ATTACK,
    CloudModelPublisher,
    DefenceResult,
    RobustnessValidator,
    build_defence,
    select_trusted_anchors,
)
from .utils import ensure_dir, get_logger, set_global_seed, timer, write_json

LOGGER = get_logger("lfd_ids.pipeline")


@dataclass
class RunRecord:
    """One row of the results table."""

    scenario: str  # baseline | attacked | defended
    attack: str | None
    defence: str | None
    intensity: float
    metrics: Metrics
    seed: int
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        m = self.metrics
        return {
            "scenario": self.scenario,
            "attack": self.attack or "",
            "defence": self.defence or "",
            "intensity": self.intensity,
            "seed": self.seed,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "fnr": m.fnr,
            "fpr": m.fpr,
            "auc": m.auc,
            "tn": m.confusion.tn,
            "fp": m.confusion.fp,
            "fn": m.confusion.fn,
            "tp": m.confusion.tp,
        }

    def as_dict(self) -> dict:
        return {**self.as_row(), "extras": self.extras}


@dataclass
class ExperimentReport:
    """Everything one seed's run produced."""

    config: dict
    acquisition: dict
    split: dict
    architecture: list[str]
    backend: str
    records: list[RunRecord]
    deployment: dict
    alerts: dict
    seed: int
    #: A subsample of the acquired dataset, so the feature-space figure can be
    #: redrawn from a saved report without re-running acquisition.
    feature_sample: dict = field(default_factory=dict)

    def rows(self) -> list[dict]:
        return [r.as_row() for r in self.records]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "backend": self.backend,
            "architecture": self.architecture,
            "config": self.config,
            "acquisition": self.acquisition,
            "split": self.split,
            "records": [r.as_dict() for r in self.records],
            "deployment": self.deployment,
            "alerts": self.alerts,
            "feature_sample": self.feature_sample,
        }

    def baseline(self) -> RunRecord | None:
        return next((r for r in self.records if r.scenario == "baseline"), None)

    def find(self, scenario: str, attack: str, intensity: float) -> RunRecord | None:
        return next(
            (
                r
                for r in self.records
                if r.scenario == scenario
                and r.attack == attack
                and abs(r.intensity - intensity) < 1e-9
            ),
            None,
        )


def _train_and_score(
    cfg: ExperimentConfig,
    split: DataSplit,
    y_train: np.ndarray,
    seed: int,
    X_train: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[Metrics, np.ndarray]:
    """Fit the IDS on ``y_train`` and score it against the clean test split.

    ``X_train``/``sample_weight`` let the defence hand back a sanitised
    training set (rows dropped or re-weighted); both default to the full split.
    """
    X = split.X_train if X_train is None else X_train
    model = build_model(cfg.model, split.X_train.shape[1], seed)
    model.fit(X, y_train, split.X_val, split.y_val, sample_weight=sample_weight)
    scores = model.predict_proba(split.X_test)
    metrics = evaluate(split.y_test, scores, cfg.alert.decision_threshold)
    return metrics, scores


def _train_attack_ensemble(
    cfg: ExperimentConfig,
    split: DataSplit,
    poison: PoisonResult,
    seed: int,
) -> tuple[Metrics, dict]:
    """Train M_1..M_B on the poisoned subsets and score the aggregator.

    BOOTLFA aggregates by averaging the member probabilities; BAGLFA uses the
    majority vote of the member decisions, matching the architecture's
    "Result Aggregator" and "Majority-Voting Aggregator" blocks.
    """
    member_scores: list[np.ndarray] = []
    for b, subset in enumerate(poison.subsets):
        model = build_model(cfg.model, split.X_train.shape[1], seed + 1000 + b)
        model.fit(split.X_train[subset.indices], subset.y, split.X_val, split.y_val)
        member_scores.append(model.predict_proba(split.X_test))
    stacked = np.vstack(member_scores)
    if poison.name == "BAGLFA":
        votes = (stacked >= cfg.alert.decision_threshold).astype(float)
        aggregated = votes.mean(axis=0)
        rule = "majority_vote"
    else:
        aggregated = stacked.mean(axis=0)
        rule = "soft_average"
    metrics = evaluate(split.y_test, aggregated, cfg.alert.decision_threshold)
    member_acc = [
        float(np.mean((s >= cfg.alert.decision_threshold).astype(int) == split.y_test))
        for s in member_scores
    ]
    return metrics, {
        "aggregation": rule,
        "n_members": len(member_scores),
        "member_accuracy_mean": float(np.mean(member_acc)),
        "member_accuracy_std": float(np.std(member_acc)),
    }


def _feature_sample(dataset, rng: np.random.Generator, max_rows: int = 3000) -> dict:
    """Subsample the dataset so the feature-space figure survives in the report."""
    n = len(dataset)
    idx = (
        np.arange(n)
        if n <= max_rows
        else np.sort(rng.choice(n, size=max_rows, replace=False))
    )
    return {
        "X": dataset.X[idx].tolist(),
        "y": dataset.y[idx].tolist(),
        "feature_names": list(dataset.feature_names),
        "n_total": n,
    }


def run_experiment(cfg: ExperimentConfig, seed: int | None = None) -> ExperimentReport:
    """Run the complete attack-and-defence grid for a single seed."""
    seed = cfg.seed if seed is None else seed
    rng = set_global_seed(seed)
    backend = resolve_backend(cfg.model.backend)
    LOGGER.info("Backend: %s | seed: %d", backend, seed)

    # ---- Module 1 ---------------------------------------------------------
    acquisition: AcquisitionResult = acquire(cfg.acquisition, rng)
    dataset = acquisition.dataset

    # ---- Module 3 (preprocessing) -----------------------------------------
    split = preprocess(dataset, cfg.preprocess, rng)
    architecture = describe_architecture(cfg.model, split.X_train.shape[1])
    LOGGER.info("IDS architecture: %s", " -> ".join(architecture))

    vehicle_ids_test = (
        dataset.vehicle_ids[split.test_index] if dataset.vehicle_ids is not None else None
    )
    alert_module = AlertModule(cfg.alert)
    publisher = CloudModelPublisher()
    validator = RobustnessValidator()
    records: list[RunRecord] = []

    # ---- Baseline: the IDS as it would ship without an adversary ----------
    with timer("baseline") as t:
        baseline_metrics, baseline_scores = _train_and_score(cfg, split, split.y_train, seed)
    LOGGER.info("Baseline (clean)          %s", baseline_metrics.headline())
    baseline_extras = {
        "train_seconds": t["seconds"],
        "per_attack_recall": per_attack_recall(
            split.y_test,
            (baseline_scores >= cfg.alert.decision_threshold).astype(int),
            split.attack_types_test,
        )
        if split.attack_types_test is not None
        else {},
    }
    records.append(
        RunRecord("baseline", None, None, 0.0, baseline_metrics, seed, baseline_extras)
    )
    alert_module.evaluate_stream(baseline_scores, vehicle_ids_test)
    verdict = validator.validate(baseline_metrics)
    publisher.publish("ids-v1-baseline", baseline_metrics, "clean dataset D", verdict)
    alert_module.broadcast(
        "ids-v1-baseline", baseline_metrics.accuracy, baseline_metrics.fnr, cfg.acquisition.n_vehicles
    )

    # The defender's trusted anchor set, held out before any poisoning.
    trusted_idx, trusted_y = select_trusted_anchors(
        split.y_train, cfg.defence.trusted_fraction, rng, stratify=cfg.preprocess.stratify
    )

    attacker = AttackerModel(
        rng=rng,
        knowledge=cfg.attack.attacker_knowledge,
        access=cfg.attack.attacker_access,
        scope=cfg.attack.attacker_scope,
    )

    # A boundary-selection attacker needs surrogate scores for the *training*
    # rows it is choosing among. Fit that surrogate once and reuse it.
    surrogate_scores: np.ndarray | None = None
    if cfg.attack.selection == "boundary":
        surrogate_scores = (
            build_model(cfg.model, split.X_train.shape[1], seed)
            .fit(split.X_train, split.y_train, split.X_val, split.y_val)
            .predict_proba(split.X_train)
        )

    version = 1
    for attack_name in cfg.attacks:
        defence_name = DEFENCE_FOR_ATTACK[attack_name]
        for intensity in cfg.attack.intensities:
            # ---- Module 2: poison the training labels ---------------------
            attack = attacker.build_attack(
                attack_name,
                intensity,
                n_estimators=cfg.attack.n_estimators,
                subset_fraction=cfg.attack.subset_fraction,
                selection=cfg.attack.selection,
            )
            poison = attack.poison(split.y_train, surrogate_scores)

            # ---- Module 3: the IDS trained on the poisoned dataset D' -----
            attacked_metrics, attacked_scores = _train_and_score(
                cfg, split, poison.y_poisoned, seed
            )
            attacked_extras = {"poison": poison.summary()}
            if cfg.attack.train_ensemble:
                ens_metrics, ens_info = _train_attack_ensemble(cfg, split, poison, seed)
                attacked_extras["ensemble"] = {**ens_info, **ens_metrics.as_dict()}
            LOGGER.info(
                "%-8s p=%.0f%% attacked    %s",
                attack_name,
                100 * intensity,
                attacked_metrics.headline(),
            )
            records.append(
                RunRecord(
                    "attacked",
                    attack_name,
                    None,
                    intensity,
                    attacked_metrics,
                    seed,
                    attacked_extras,
                )
            )
            alert_module.evaluate_stream(attacked_scores, vehicle_ids_test)

            # ---- Module 4: cleanse, retrain, re-validate, deploy ----------
            defence = build_defence(
                defence_name,
                cfg.defence,
                rng,
                n_estimators=cfg.attack.n_estimators,
                subset_fraction=cfg.attack.subset_fraction,
            )
            result: DefenceResult = defence.defend(
                split.X_train, poison.y_poisoned, trusted_idx, trusted_y
            )
            recovery = label_recovery_metrics(
                split.y_train, poison.y_poisoned, result.y_repaired
            )
            X_clean, y_clean_labels, weights = result.training_set(split.X_train)
            defended_metrics, defended_scores = _train_and_score(
                cfg, split, y_clean_labels, seed, X_clean, weights
            )
            version += 1
            model_version = f"ids-v{version}-{defence_name.lower()}-p{int(round(100 * intensity))}"
            verdict = validator.validate(defended_metrics, baseline_metrics)
            publisher.publish(
                model_version, defended_metrics, f"cleansed dataset D'' ({defence_name})", verdict
            )
            alert_module.broadcast(
                model_version,
                defended_metrics.accuracy,
                defended_metrics.fnr,
                cfg.acquisition.n_vehicles,
            )
            alert_module.evaluate_stream(defended_scores, vehicle_ids_test)

            LOGGER.info(
                "%-8s p=%.0f%% %-8s %s (labels %.3f -> %.3f)",
                attack_name,
                100 * intensity,
                defence_name,
                defended_metrics.headline(),
                recovery["label_error_before"],
                recovery["label_error_after"],
            )
            records.append(
                RunRecord(
                    "defended",
                    attack_name,
                    defence_name,
                    intensity,
                    defended_metrics,
                    seed,
                    {
                        "defence": result.summary(),
                        "label_recovery": recovery,
                        "robustness": verdict.as_dict(),
                        "model_version": model_version,
                        "per_attack_recall": per_attack_recall(
                            split.y_test,
                            (defended_scores >= cfg.alert.decision_threshold).astype(int),
                            split.attack_types_test,
                        )
                        if split.attack_types_test is not None
                        else {},
                    },
                )
            )

    from dataclasses import asdict

    return ExperimentReport(
        feature_sample=_feature_sample(dataset, rng),
        config=asdict(cfg),
        acquisition=acquisition.summary(),
        split=split.summary(),
        architecture=architecture,
        backend=backend,
        records=records,
        deployment=publisher.summary(),
        alerts=alert_module.summary(),
        seed=seed,
    )


def run_repeats(cfg: ExperimentConfig) -> tuple[list[ExperimentReport], list[dict]]:
    """Run the grid ``cfg.n_repeats`` times and return reports plus all rows."""
    reports: list[ExperimentReport] = []
    rows: list[dict] = []
    for i in range(max(cfg.n_repeats, 1)):
        report = run_experiment(cfg, seed=cfg.seed + i)
        reports.append(report)
        rows.extend(report.rows())
    return reports, rows


def aggregate_rows(rows: Sequence[dict]) -> list[dict]:
    """Average the metric columns across seeds, keeping mean and std."""
    import pandas as pd

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    metric_cols = ["accuracy", "precision", "recall", "f1", "fnr", "fpr", "auc"]
    grouped = frame.groupby(["scenario", "attack", "defence", "intensity"], dropna=False)
    agg = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        col[0] if not col[1] else f"{col[0]}_{col[1]}" for col in agg.columns.to_flat_index()
    ]
    agg["n_seeds"] = grouped.size().to_numpy()
    return agg.to_dict(orient="records")


def save_results(
    reports: Sequence[ExperimentReport],
    rows: Sequence[dict],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the report JSON, the per-seed CSV and the aggregated summary."""
    import pandas as pd

    out = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    paths["report"] = write_json(
        out / "experiment_report.json", [r.as_dict() for r in reports]
    )
    frame = pd.DataFrame(list(rows))
    csv_path = out / "results.csv"
    frame.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    aggregated = aggregate_rows(rows)
    paths["summary"] = write_json(out / "summary.json", aggregated)
    if aggregated:
        agg_csv = out / "summary.csv"
        pd.DataFrame(aggregated).to_csv(agg_csv, index=False)
        paths["summary_csv"] = agg_csv
    return paths
