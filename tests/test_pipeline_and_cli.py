"""End-to-end: config handling, the experiment pipeline and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lfd_ids.cli import main
from lfd_ids.config import ExperimentConfig, apply_override, load_config, validate
from lfd_ids.pipeline import aggregate_rows, run_experiment, run_repeats, save_results


# -------------------------------------------------------------------- config


def test_defaults_match_the_specification():
    cfg = load_config()
    assert cfg.model.hidden_units == (64, 32, 16, 8)
    assert cfg.model.activations == ("relu", "tanh", "relu", "tanh")
    assert cfg.model.output_activation == "sigmoid"
    assert cfg.attack.intensities == (0.30, 0.40, 0.50)
    assert cfg.defence.n_clusters == 2
    assert (cfg.preprocess.train_frac, cfg.preprocess.val_frac, cfg.preprocess.test_frac) == (
        0.60,
        0.20,
        0.20,
    )
    assert cfg.acquisition.feature_set == (
        "tyre_temperature",
        "tyre_pressure",
        "latitude",
        "longitude",
    )


def test_yaml_config_loads(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_config(path)
    assert cfg.seed == 42
    assert cfg.attacks == ("BOOTLFA", "BAGLFA")
    assert cfg.defence.outer_policy == "drop"


def test_scalar_sequence_fields_are_widened():
    cfg = load_config()
    apply_override(cfg, "attack.intensities", 0.4)
    apply_override(cfg, "attacks", "BOOTLFA")
    validate(cfg)
    assert cfg.attack.intensities == (0.4,)
    assert cfg.attacks == ("BOOTLFA",)


def test_dotted_overrides():
    cfg = load_config()
    apply_override(cfg, "model.epochs", 7)
    apply_override(cfg, "defence.n_clusters", 3)
    assert cfg.model.epochs == 7 and cfg.defence.n_clusters == 3
    with pytest.raises(KeyError):
        apply_override(cfg, "model.nonexistent", 1)
    with pytest.raises(KeyError):
        apply_override(cfg, "nope.thing", 1)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("preprocess.train_frac", 0.9, "must sum to 1.0"),
        ("acquisition.malicious_ratio", 1.5, "malicious_ratio"),
        ("attack.intensities", (1.5,), "intensity must lie"),
        ("defence.n_clusters", 1, "at least 2"),
        ("attacks", ("NOPE",), "unknown attack"),
        ("model.backend", "jax", "backend must be"),
        ("defence.outer_policy", "burn", "outer_policy must be"),
    ],
)
def test_validation_rejects_bad_configs(key, value, message):
    cfg = load_config()
    apply_override(cfg, key, value)
    with pytest.raises(ValueError, match=message):
        validate(cfg)


def test_mismatched_layer_spec_is_rejected():
    cfg = load_config()
    cfg.model.activations = ("relu",)
    with pytest.raises(ValueError, match="same length"):
        validate(cfg)


# ------------------------------------------------------------------ pipeline


@pytest.fixture(scope="module")
def report():
    cfg = ExperimentConfig()
    cfg.acquisition.n_vehicles = 8
    cfg.acquisition.samples_per_vehicle = 150
    cfg.acquisition.database_path = None
    cfg.model.backend = "numpy"
    cfg.model.epochs = 40
    cfg.attack.n_estimators = 3
    cfg.attack.train_ensemble = True
    cfg.attack.intensities = (0.30, 0.50)
    cfg.defence.n_init = 3
    return run_experiment(cfg)


def test_report_covers_the_full_grid(report):
    scenarios = {(r.scenario, r.attack, r.intensity) for r in report.records}
    assert ("baseline", None, 0.0) in scenarios
    for attack in ("BOOTLFA", "BAGLFA"):
        for p in (0.30, 0.50):
            assert ("attacked", attack, p) in scenarios
            assert ("defended", attack, p) in scenarios
    assert len(report.records) == 1 + 2 * 2 * 2


def test_baseline_is_a_working_detector(report):
    baseline = report.baseline()
    assert baseline is not None
    assert baseline.metrics.accuracy > 0.95
    assert baseline.metrics.auc > 0.98


def test_maximum_poisoning_degrades_the_ids(report):
    """At p = 50% the poisoned labels carry no class signal at all.

    This fixture trains for only 40 epochs to stay fast, so the model
    under-fits the noise and its accuracy does not fall all the way to the
    ~50% the full configuration reaches.  The collapse of AUC to near chance
    is the assumption-free part of the claim, so assert on that.
    """
    baseline = report.baseline()
    for attack in ("BOOTLFA", "BAGLFA"):
        worst = report.find("attacked", attack, 0.50)
        assert worst is not None
        assert worst.metrics.auc < 0.75
        assert baseline.metrics.auc - worst.metrics.auc > 0.20
        assert worst.metrics.accuracy < baseline.metrics.accuracy - 0.15


def test_defence_recovers_accuracy_and_cuts_false_negatives(report):
    for attack in ("BOOTLFA", "BAGLFA"):
        for p in (0.30, 0.50):
            attacked = report.find("attacked", attack, p)
            defended = report.find("defended", attack, p)
            assert defended.metrics.accuracy >= attacked.metrics.accuracy
            assert defended.metrics.accuracy > 0.90
            recovery = defended.extras["label_recovery"]
            assert recovery["label_error_after"] < recovery["label_error_before"]
        # The headline safety claim: fewer missed intrusions at max poisoning.
        assert (
            report.find("defended", attack, 0.50).metrics.fnr
            < report.find("attacked", attack, 0.50).metrics.fnr
        )


def test_feature_sample_is_stored_for_plotting(report):
    sample = report.feature_sample
    assert sample["feature_names"] == list(report.split["feature_names"])
    assert len(sample["X"]) == len(sample["y"]) <= sample["n_total"]


def test_attack_ensembles_are_trained(report):
    ensemble = report.find("attacked", "BAGLFA", 0.30).extras["ensemble"]
    assert ensemble["n_members"] == 3
    assert ensemble["aggregation"] == "majority_vote"
    assert report.find("attacked", "BOOTLFA", 0.30).extras["ensemble"]["aggregation"] == (
        "soft_average"
    )


def test_only_sound_models_reach_the_fleet(report):
    deployed = report.deployment["deployed"]
    assert deployed[0]["version"] == "ids-v1-baseline"
    for model in deployed:
        assert model["accuracy"] >= 0.90 and model["fnr"] <= 0.10


def test_experiment_is_reproducible():
    cfg = ExperimentConfig()
    cfg.acquisition.n_vehicles = 4
    cfg.acquisition.samples_per_vehicle = 120
    cfg.acquisition.database_path = None
    cfg.model.backend = "numpy"
    cfg.model.epochs = 15
    cfg.attack.n_estimators = 2
    cfg.attack.train_ensemble = False
    cfg.attack.intensities = (0.40,)
    cfg.attacks = ("BOOTLFA",)
    cfg.defence.n_init = 2
    a = run_experiment(cfg)
    b = run_experiment(cfg)
    assert [r.as_row() for r in a.records] == [r.as_row() for r in b.records]


def test_save_results_and_aggregation(tmp_path):
    cfg = ExperimentConfig()
    cfg.acquisition.n_vehicles = 4
    cfg.acquisition.samples_per_vehicle = 120
    cfg.acquisition.database_path = None
    cfg.model.backend = "numpy"
    cfg.model.epochs = 12
    cfg.attack.n_estimators = 2
    cfg.attack.train_ensemble = False
    cfg.attack.intensities = (0.40,)
    cfg.attacks = ("BOOTLFA",)
    cfg.defence.n_init = 2
    cfg.n_repeats = 2
    reports, rows = run_repeats(cfg)
    assert len(reports) == 2
    assert {r.seed for r in reports} == {cfg.seed, cfg.seed + 1}

    paths = save_results(reports, rows, tmp_path)
    assert paths["csv"].exists() and paths["report"].exists()
    payload = json.loads(paths["report"].read_text())
    assert len(payload) == 2
    aggregated = aggregate_rows(rows)
    assert all(row["n_seeds"] == 2 for row in aggregated)
    assert any("accuracy_mean" in row for row in aggregated)


# ----------------------------------------------------------------------- CLI


def _fast(*extra: str) -> list[str]:
    return [
        "--backend", "numpy",
        "--set", "acquisition.n_vehicles=4",
        "--set", "acquisition.samples_per_vehicle=120",
        "--set", "acquisition.database_path=none",
        "--set", "model.epochs=12",
        "--set", "attack.n_estimators=2",
        "--set", "attack.train_ensemble=false",
        "--set", "attack.intensities=0.4",
        "--set", "defence.n_init=2",
        *extra,
    ]


def test_cli_generate_data(tmp_path, capsys):
    out = tmp_path / "data.csv"
    assert main(_fast("-o", str(tmp_path), "generate-data", "--out", str(out))) == 0
    assert out.exists()
    assert "Wrote" in capsys.readouterr().out


def test_cli_run_writes_results(tmp_path):
    assert main(_fast("-o", str(tmp_path), "run", "--no-figures")) == 0
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "experiment_report.json").exists()


def test_overrides_before_the_subcommand_actually_apply(tmp_path):
    """Guards against argparse silently dropping pre-subcommand options.

    Asserting only that output files exist is not enough: a run that ignored
    every ``--set`` would still write them, just from the full default
    configuration. So check the values landed in the report.
    """
    assert main(_fast("-o", str(tmp_path), "run", "--no-figures")) == 0
    payload = json.loads((tmp_path / "experiment_report.json").read_text())
    config = payload[0]["config"]
    assert config["model"]["epochs"] == 12
    assert config["acquisition"]["n_vehicles"] == 4
    assert config["acquisition"]["samples_per_vehicle"] == 120
    assert config["attack"]["intensities"] == [0.4]
    assert config["attack"]["train_ensemble"] is False
    assert config["acquisition"]["database_path"] is None


def test_overrides_merge_across_the_subcommand_boundary(tmp_path):
    assert (
        main(_fast("-o", str(tmp_path), "run", "--no-figures", "--set", "model.batch_size=16"))
        == 0
    )
    config = json.loads((tmp_path / "experiment_report.json").read_text())[0]["config"]
    assert config["model"]["batch_size"] == 16   # given after the subcommand
    assert config["model"]["epochs"] == 12       # given before it


def test_cli_attack_and_defend(tmp_path, capsys):
    assert main(_fast("-o", str(tmp_path), "attack")) == 0
    assert "clean baseline" in capsys.readouterr().out
    assert main(_fast("-o", str(tmp_path), "defend")) == 0
    assert "BOOTKCD" in capsys.readouterr().out


def test_cli_demo(tmp_path):
    assert main(_fast("-o", str(tmp_path), "demo")) == 0
    assert (tmp_path / "results.csv").exists()


def test_cli_rejects_a_bad_override(tmp_path):
    with pytest.raises(SystemExit):
        main(["--set", "model.nope=1", "-o", str(tmp_path), "attack"])


def test_cli_value_parsing():
    from lfd_ids.cli import _parse_value

    assert _parse_value("true") is True
    assert _parse_value("none") is None
    assert _parse_value("12") == 12
    assert _parse_value("0.4") == 0.4
    assert _parse_value("0.3,0.5") == (0.3, 0.5)
    assert _parse_value("minmax") == "minmax"


def test_figures_render(tmp_path):
    pytest.importorskip("matplotlib")
    from lfd_ids.figures import render_all

    cfg = ExperimentConfig()
    cfg.acquisition.n_vehicles = 4
    cfg.acquisition.samples_per_vehicle = 120
    cfg.acquisition.database_path = None
    cfg.model.backend = "numpy"
    cfg.model.epochs = 12
    cfg.attack.n_estimators = 2
    cfg.attack.train_ensemble = False
    cfg.attack.intensities = (0.30, 0.50)
    cfg.defence.n_init = 2
    reports, rows = run_repeats(cfg)
    paths = render_all(reports, rows, tmp_path / "figures")
    assert len(paths) == 6
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_global_options_work_after_the_subcommand(tmp_path):
    """`lfd-ids run -o DIR` must behave like `lfd-ids -o DIR run`."""
    assert (
        main([
            "run", "--no-figures",
            "-o", str(tmp_path),
            "--backend", "numpy",
            "--set", "acquisition.n_vehicles=3",
            "--set", "acquisition.samples_per_vehicle=100",
            "--set", "acquisition.database_path=none",
            "--set", "model.epochs=8",
            "--set", "attack.n_estimators=2",
            "--set", "attack.train_ensemble=false",
            "--set", "attack.intensities=0.5",
            "--set", "attacks=BOOTLFA",
            "--set", "defence.n_init=2",
        ])
        == 0
    )
    assert (tmp_path / "results.csv").exists()


def test_boundary_selection_runs_end_to_end(tmp_path):
    """The boundary attacker needs surrogate scores over the training rows."""
    cfg = ExperimentConfig()
    cfg.acquisition.n_vehicles = 4
    cfg.acquisition.samples_per_vehicle = 120
    cfg.acquisition.database_path = None
    cfg.model.backend = "numpy"
    cfg.model.epochs = 15
    cfg.attack.n_estimators = 2
    cfg.attack.train_ensemble = False
    cfg.attack.intensities = (0.30,)
    cfg.attacks = ("BOOTLFA",)
    cfg.attack.selection = "boundary"
    cfg.defence.n_init = 2
    result = run_experiment(cfg)
    attacked = result.find("attacked", "BOOTLFA", 0.30)
    assert attacked.extras["poison"]["selection"] == "boundary"
    assert attacked.extras["poison"]["realised_intensity"] == pytest.approx(0.30, abs=0.01)
