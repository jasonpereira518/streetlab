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
    [("build", "2"), ("export-dataset", "5"), ("train", "5"), ("eval", "5")],
)
def test_unimplemented_commands_explain_themselves(capsys, command, cycle):
    argv = [command, "somewhere"] if command == "build" else [command]
    code, out = run(capsys, *argv)
    assert code != 0
    assert "not yet implemented" in out.lower()
    assert f"cycle {cycle}" in out.lower()
