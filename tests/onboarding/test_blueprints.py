from pathlib import Path

from onboarding.blueprints import load_blueprints, select_blueprints
from onboarding.models import Blueprint, BlueprintStep, PersonProfile

GLOBAL_YAML = """
scope: global
version: "1"
source: authored
steps:
  - id: sec
    title: Security policy
    requirement: required
"""

BACKEND_YAML = """
scope: area:backend
version: "2"
steps:
  - id: db
    title: Local DB
    requirement: required
"""


def _write(dir_: Path, name: str, content: str) -> None:
    (dir_ / name).write_text(content, encoding="utf-8")


def test_load_blueprints_reads_all_yaml(tmp_path: Path) -> None:
    _write(tmp_path, "global.yaml", GLOBAL_YAML)
    _write(tmp_path, "area-backend.yaml", BACKEND_YAML)

    blueprints = load_blueprints(tmp_path)

    scopes = {b.scope for b in blueprints}
    assert scopes == {"global", "area:backend"}


def test_load_blueprints_skips_invalid_file(tmp_path: Path) -> None:
    _write(tmp_path, "global.yaml", GLOBAL_YAML)
    _write(tmp_path, "broken.yaml", "scope: global\nsteps: : :")

    blueprints = load_blueprints(tmp_path)

    assert [b.scope for b in blueprints] == ["global"]


def test_load_blueprints_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_blueprints(tmp_path / "nope") == []


def test_select_includes_global_and_matching_area() -> None:
    blueprints = [
        Blueprint(scope="global", steps=[BlueprintStep(id="a", title="A")]),
        Blueprint(scope="area:backend", steps=[BlueprintStep(id="b", title="B")]),
        Blueprint(scope="area:frontend", steps=[BlueprintStep(id="c", title="C")]),
    ]
    profile = PersonProfile(working_area="backend", experience="junior")

    selected = select_blueprints(blueprints, profile)

    assert [b.scope for b in selected] == ["global", "area:backend"]


def test_select_unknown_area_yields_global_only() -> None:
    blueprints = [
        Blueprint(scope="global", steps=[BlueprintStep(id="a", title="A")]),
        Blueprint(scope="area:backend", steps=[BlueprintStep(id="b", title="B")]),
    ]
    profile = PersonProfile(working_area="quantum-computing", experience="junior")

    selected = select_blueprints(blueprints, profile)

    assert [b.scope for b in selected] == ["global"]


def test_no_repo_seed_blueprints_shipped() -> None:
    repo_blueprints = Path(__file__).resolve().parents[2] / "blueprints"

    blueprints = load_blueprints(repo_blueprints)

    assert blueprints == []
