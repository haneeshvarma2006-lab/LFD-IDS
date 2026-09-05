"""Command-line interface for LFD-IDS.

Subcommands
-----------
``generate-data``  Run Module 1 only and export the labelled dataset to CSV.
``run``            Run the full attack-and-defence grid and write results.
``attack``         Poison a dataset and report the damage, without defending.
``defend``         Run KCD on a poisoned dataset and report label recovery.
``figures``        Render the result figures from a finished run.
``demo``           A fast, small end-to-end run for a live walkthrough.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ExperimentConfig, apply_override, load_config, validate
from .utils import ensure_dir, format_pct, get_logger, set_global_seed, write_json

LOGGER = get_logger("lfd_ids.cli")


def _parse_value(text: str):
    """Coerce a ``--set key=value`` payload into a Python scalar or tuple."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in text:
        return tuple(_parse_value(part) for part in text.split(","))
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_config(args.config)
    for assignment in args.set or []:
        if "=" not in assignment:
            raise SystemExit(f"--set expects key=value, got '{assignment}'")
        key, _, raw = assignment.partition("=")
        apply_override(cfg, key.strip(), _parse_value(raw))
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "output", None):
        cfg.output_dir = args.output
    if getattr(args, "backend", None):
        cfg.model.backend = args.backend
    if getattr(args, "repeats", None):
        cfg.n_repeats = args.repeats
    if getattr(args, "csv", None):
        cfg.acquisition.csv_path = args.csv
    validate(cfg)
    return cfg


# ------------------------------------------------------------------ commands


def cmd_generate_data(args: argparse.Namespace) -> int:
    import pandas as pd

    from .module1_acquisition import acquire

    cfg = _build_config(args)
    rng = set_global_seed(cfg.seed)
    result = acquire(cfg.acquisition, rng)
    dataset = result.dataset
    frame = pd.DataFrame(dataset.X, columns=list(dataset.feature_names))
    frame["label"] = dataset.y
    if dataset.attack_types is not None:
        frame["attack_type"] = dataset.attack_types
    if dataset.vehicle_ids is not None:
        frame["vehicle_id"] = dataset.vehicle_ids
    out = Path(args.out or Path(cfg.output_dir) / "cv_sensor_dataset.csv")
    ensure_dir(out.parent)
    frame.to_csv(out, index=False)
    write_json(Path(cfg.output_dir) / "acquisition_summary.json", result.summary())
    print(f"Wrote {len(frame)} records to {out}")
    print(f"  malicious: {format_pct(dataset.summary()['malicious_ratio'])}")
    print(f"  attack mix: {result.attack_breakdown}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import aggregate_rows, run_repeats, save_results

    cfg = _build_config(args)
    reports, rows = run_repeats(cfg)
    paths = save_results(reports, rows, cfg.output_dir)
    _print_table(rows)
    print()
    for name, path in paths.items():
        print(f"  {name:12s} {path}")
    if cfg.save_figures and not args.no_figures:
        try:
            from .figures import render_all

            figures = render_all(reports, rows, Path(cfg.output_dir) / "figures")
            print(f"  figures      {len(figures)} written to "
                  f"{Path(cfg.output_dir) / 'figures'}")
        except ImportError as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("Skipping figures: %s", exc)
    if len(reports) > 1:
        print("\nAggregated across seeds:")
        for row in aggregate_rows(rows):
            print(
                f"  {row['scenario']:9s} {row['attack']:8s} p={row['intensity']:.2f} "
                f"acc={row['accuracy_mean']:.4f}+/-{row.get('accuracy_std') or 0.0:.4f}"
            )
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    from .module1_acquisition import acquire
    from .module2_attack import AttackerModel
    from .module3_ids import build_model, evaluate, preprocess

    cfg = _build_config(args)
    rng = set_global_seed(cfg.seed)
    split = preprocess(acquire(cfg.acquisition, rng).dataset, cfg.preprocess, rng)
    attacker = AttackerModel(
        rng,
        knowledge=cfg.attack.attacker_knowledge,
        access=cfg.attack.attacker_access,
        scope=cfg.attack.attacker_scope,
    )
    baseline = build_model(cfg.model, split.X_train.shape[1], cfg.seed).fit(
        split.X_train, split.y_train, split.X_val, split.y_val
    )
    clean = evaluate(split.y_test, baseline.predict_proba(split.X_test))
    print(f"clean baseline           {clean.headline()}")
    for name in cfg.attacks:
        for p in cfg.attack.intensities:
            attack = attacker.build_attack(
                name,
                p,
                n_estimators=cfg.attack.n_estimators,
                subset_fraction=cfg.attack.subset_fraction,
                selection=cfg.attack.selection,
            )
            poison = attack.poison(split.y_train)
            model = build_model(cfg.model, split.X_train.shape[1], cfg.seed).fit(
                split.X_train, poison.y_poisoned, split.X_val, split.y_val
            )
            metrics = evaluate(split.y_test, model.predict_proba(split.X_test))
            print(f"{name:8s} p={p:.0%}  {metrics.headline()}  "
                  f"(flipped {poison.n_flipped}/{poison.y_clean.size})")
    return 0


def cmd_defend(args: argparse.Namespace) -> int:
    from .module1_acquisition import acquire
    from .module2_attack import AttackerModel
    from .module3_ids import build_model, evaluate, preprocess
    from .module3_ids.evaluate import label_recovery_metrics
    from .module4_defense import DEFENCE_FOR_ATTACK, build_defence, select_trusted_anchors

    cfg = _build_config(args)
    rng = set_global_seed(cfg.seed)
    split = preprocess(acquire(cfg.acquisition, rng).dataset, cfg.preprocess, rng)
    trusted_idx, trusted_y = select_trusted_anchors(
        split.y_train, cfg.defence.trusted_fraction, rng
    )
    attacker = AttackerModel(rng, knowledge=cfg.attack.attacker_knowledge)
    for name in cfg.attacks:
        defence_name = DEFENCE_FOR_ATTACK[name]
        for p in cfg.attack.intensities:
            poison = attacker.build_attack(
                name,
                p,
                n_estimators=cfg.attack.n_estimators,
                subset_fraction=cfg.attack.subset_fraction,
                selection=cfg.attack.selection,
            ).poison(split.y_train)
            defence = build_defence(
                defence_name,
                cfg.defence,
                rng,
                n_estimators=cfg.attack.n_estimators,
                subset_fraction=cfg.attack.subset_fraction,
            )
            result = defence.defend(split.X_train, poison.y_poisoned, trusted_idx, trusted_y)
            recovery = label_recovery_metrics(
                split.y_train, poison.y_poisoned, result.y_repaired
            )
            model = build_model(cfg.model, split.X_train.shape[1], cfg.seed).fit(
                split.X_train, result.y_repaired, split.X_val, split.y_val
            )
            metrics = evaluate(split.y_test, model.predict_proba(split.X_test))
            print(
                f"{name:8s} p={p:.0%} -> {defence_name:8s} "
                f"[{result.assignment_method}] labels "
                f"{recovery['label_error_before']:.3f} -> {recovery['label_error_after']:.3f} "
                f"| {metrics.headline()}"
            )
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    import pandas as pd

    from .figures import render_all
    from .utils import read_json

    cfg = _build_config(args)
    out_dir = Path(cfg.output_dir)
    rows = pd.read_csv(out_dir / "results.csv").to_dict(orient="records")
    reports = read_json(out_dir / "experiment_report.json")
    figures = render_all(reports, rows, out_dir / "figures")
    for path in figures:
        print(f"  {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .pipeline import run_repeats, save_results

    cfg = _build_config(args)
    cfg.acquisition.n_vehicles = 12
    cfg.acquisition.samples_per_vehicle = 250
    cfg.model.epochs = 60
    cfg.attack.train_ensemble = False
    cfg.attack.intensities = (0.30, 0.50)
    cfg.n_repeats = 1
    reports, rows = run_repeats(cfg)
    save_results(reports, rows, cfg.output_dir)
    _print_table(rows)
    return 0


# -------------------------------------------------------------------- output


def _print_table(rows: Sequence[dict]) -> None:
    header = f"{'scenario':10s} {'attack':9s} {'defence':9s} {'p':>5s} " \
             f"{'acc':>8s} {'prec':>8s} {'rec':>8s} {'f1':>8s} {'fnr':>8s} {'auc':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['scenario']:10s} {str(row['attack'] or '-'):9s} "
            f"{str(row['defence'] or '-'):9s} {row['intensity']:5.2f} "
            f"{row['accuracy']:8.4f} {row['precision']:8.4f} {row['recall']:8.4f} "
            f"{row['f1']:8.4f} {row['fnr']:8.4f} "
            f"{row['auc'] if np.isfinite(row['auc']) else float('nan'):8.4f}"
        )


#: Global options, and the dest suffix used for their post-subcommand copies.
GLOBAL_DESTS: tuple[str, ...] = ("config", "set", "seed", "output", "backend")
_POST = "_post"


def _add_global_options(parser: argparse.ArgumentParser, suffix: str = "") -> None:
    """Attach the global options, optionally under suffixed dests.

    The suffixed copies live on the subcommand parsers.  They need their own
    dests because argparse lets a subparser overwrite an attribute the main
    parser already filled in -- with a shared dest, ``lfd-ids --set k=v run``
    would silently discard ``--set``.  :func:`_merge_global_options` folds the
    two sets back together afterwards.
    """
    parser.add_argument(
        "-c", "--config", dest="config" + suffix, help="path to a YAML configuration file"
    )
    parser.add_argument(
        "--set",
        dest="set" + suffix,
        action="append",
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set model.epochs=50 "
             "(repeatable; comma-separated values become tuples)",
    )
    parser.add_argument(
        "--seed", dest="seed" + suffix, type=int, help="override the random seed"
    )
    parser.add_argument(
        "-o", "--output", dest="output" + suffix, help="override the output directory"
    )
    parser.add_argument(
        "--backend",
        dest="backend" + suffix,
        choices=("auto", "numpy", "keras"),
        help="override the model backend",
    )


def _merge_global_options(args: argparse.Namespace) -> argparse.Namespace:
    """Fold post-subcommand global options onto their pre-subcommand values.

    ``--set`` accumulates from both positions; every other option takes the
    later (post-subcommand) value when it was given.
    """
    for dest in GLOBAL_DESTS:
        post = getattr(args, dest + _POST, None)
        delattr(args, dest + _POST) if hasattr(args, dest + _POST) else None
        if post is None:
            continue
        if dest == "set":
            setattr(args, dest, list(getattr(args, dest) or []) + list(post))
        else:
            setattr(args, dest, post)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfd-ids",
        description="Label-flipping poisoning attacks and the KCD defence for "
                    "connected-vehicle intrusion detection.",
    )
    _add_global_options(parser)

    # The same options are attached to every subcommand through a parent
    # parser, so `lfd-ids run -o results` works as naturally as
    # `lfd-ids -o results run`.  argparse would otherwise reject the former.
    shared = argparse.ArgumentParser(add_help=False)
    _add_global_options(shared, suffix=_POST)

    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser(
        "generate-data", parents=[shared], help="run Module 1 and export the dataset"
    )
    p_gen.add_argument("--out", help="CSV path (default: <output>/cv_sensor_dataset.csv)")
    p_gen.set_defaults(func=cmd_generate_data)

    p_run = sub.add_parser(
        "run", parents=[shared], help="run the full attack-and-defence experiment grid"
    )
    p_run.add_argument("--repeats", type=int, help="number of seeds to average over")
    p_run.add_argument("--csv", help="train on a real dataset instead of the simulator")
    p_run.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    p_run.set_defaults(func=cmd_run)

    p_atk = sub.add_parser(
        "attack", parents=[shared], help="poison and report, without defending"
    )
    p_atk.set_defaults(func=cmd_attack)

    p_def = sub.add_parser(
        "defend", parents=[shared], help="run KCD and report label recovery"
    )
    p_def.set_defaults(func=cmd_defend)

    p_fig = sub.add_parser(
        "figures", parents=[shared], help="render figures from a finished run"
    )
    p_fig.set_defaults(func=cmd_figures)

    p_demo = sub.add_parser("demo", parents=[shared], help="fast end-to-end walkthrough")
    p_demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _merge_global_options(parser.parse_args(argv))
    try:
        return int(args.func(args))
    except (KeyError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
