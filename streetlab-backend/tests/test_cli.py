"""The headless CLI. `run` has to prove the sim reacts without a frontend attached."""

import pytest

from server.cli import main


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_no_arguments_prints_help(capsys):
    code, out = run(capsys)
    assert code != 0
    assert "serve" in out and "run" in out


def test_scenarios_lists_the_catalog(capsys):
    code, out = run(capsys, "scenarios")
    assert code == 0
    assert "grid-loop" in out
    assert "grid-night" in out


def test_run_completes_and_reports_a_summary(capsys):
    code, out = run(capsys, "run", "--duration", "6", "--seed", "3")
    assert code == 0
    assert "grid-loop" in out
    assert "distance" in out.lower()


def test_run_prints_a_reaction_log(capsys):
    code, out = run(capsys, "run", "--duration", "12", "--inject-at", "4")
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("t=")]
    assert len(lines) >= 5, "expected a per-interval reaction log"
    assert any("mph" in ln for ln in lines)


def test_run_reports_the_reaction_to_an_injected_hazard(capsys):
    code, out = run(capsys, "run", "--duration", "14", "--inject-at", "5", "--inject", "sudden_brake")
    assert code == 0
    assert "sudden_brake" in out
    assert "slowed" in out.lower() or "ttc" in out.lower()


def test_run_is_deterministic_for_a_seed(capsys):
    _, first = run(capsys, "run", "--duration", "8", "--seed", "11")
    _, second = run(capsys, "run", "--duration", "8", "--seed", "11")
    assert first == second


def test_different_seeds_produce_different_runs(capsys):
    _, first = run(capsys, "run", "--duration", "8", "--seed", "1")
    _, second = run(capsys, "run", "--duration", "8", "--seed", "2")
    assert first != second


def test_run_accepts_a_scenario(capsys):
    code, out = run(capsys, "run", "--scenario", "grid-signals", "--duration", "4")
    assert code == 0
    assert "grid-signals" in out


def test_run_rejects_an_unknown_scenario(capsys):
    code, out = run(capsys, "run", "--scenario", "atlantis", "--duration", "2")
    assert code != 0
    assert "atlantis" in out


def test_run_never_leaves_the_lane(capsys):
    code, out = run(capsys, "run", "--duration", "30")
    assert code == 0
    assert "max lane offset" in out.lower()
    offset = float(out.lower().split("max lane offset")[1].split("m")[0].strip(" :"))
    assert offset < 1.8


@pytest.mark.parametrize(
    "command,cycle",
    [("export-dataset", "5"), ("train", "5"), ("eval", "5")],
)
def test_unimplemented_commands_explain_themselves(capsys, command, cycle):
    code, out = run(capsys, command)
    assert code != 0
    assert "not yet implemented" in out.lower()
    assert f"cycle {cycle}" in out.lower()


# --------------------------------------------------------------------------- #
# `--perception ml` builds a real detector — offline, and without weights.     #
# --------------------------------------------------------------------------- #


def _never_called():  # pragma: no cover - the assertion is that it is not
    raise AssertionError("the weights cache must not be touched for a local path")


def test_ground_truth_builds_no_pipeline():
    from server.cli import build_parser, perception_pipeline_for

    args = build_parser().parse_args(["run"])
    assert perception_pipeline_for(args) is None


def test_a_local_model_path_is_used_directly_without_the_cache(tmp_path, monkeypatch):
    """`--detector-model` is the development path: no download, no hashing,
    no cache directory touched at all."""
    from perception.detector import OnnxDetector
    from server import cli

    monkeypatch.setattr(cli, "model_cache_dir", _never_called)
    weights = tmp_path / "local.onnx"
    weights.write_bytes(b"not really a model, never loaded here")

    detector = cli.build_detector(str(weights))
    assert isinstance(detector, OnnxDetector)
    # Lazily, on the executor: building a session at startup would make a
    # slow model load look like a backend that failed to boot.
    assert detector.provider is None


def test_unresolvable_weights_degrade_to_the_stub_rather_than_refusing_to_start(caplog):
    """A download that failed must cost perception, not the whole backend."""
    from perception.pipeline import StubDetector
    from server.cli import build_detector

    with caplog.at_level("WARNING", logger="streetlab.cli"):
        detector = build_detector("/nonexistent/definitely-not-a-model.onnx")

    assert isinstance(detector, StubDetector)
    assert "detector weights unavailable" in caplog.text
