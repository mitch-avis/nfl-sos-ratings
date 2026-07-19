"""Policy tests for durable methodology language."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import polars as pl
import pytest

from nfl_sos_ratings import composite_weights
from nfl_sos_ratings.metrics import get_registry
from nfl_sos_ratings.validation import walk_forward

_REPO_ROOT = Path(__file__).resolve().parents[1]

_DURABLE_DOC_PATHS: tuple[Path, ...] = (
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "methodology.md",
    _REPO_ROOT / "docs" / "stats-catalog.md",
    _REPO_ROOT / "docs" / "qb-stats-catalog.md",
)

_SOURCE_ROOT = _REPO_ROOT / "nfl_sos_ratings"
_TESTS_ROOT = _REPO_ROOT / "tests"
_HISTORY_LANGUAGE_EXEMPT_SOURCE = _SOURCE_ROOT / "validation" / "history_strings.py"
_POLICY_TEST_EXEMPT_SOURCE = Path(__file__).resolve()
_LANGUAGE_SCAN_EXEMPT_SOURCES: tuple[Path, ...] = (
    _HISTORY_LANGUAGE_EXEMPT_SOURCE,
    _POLICY_TEST_EXEMPT_SOURCE,
)

_PUBLISHED_RATING_COLUMNS: tuple[str, ...] = (
    "SaOR",
    "SaDR",
    "SaSTR",
    "SaOvR",
    "SaCR",
    "QRaw",
    "QSaOR",
    "QSoS",
    "QSaCR",
    "QOutcome",
)

_BANNED_METHODOLOGY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bStage\s+\d+[a-z]?\b"),
    re.compile(r"\bBlock\s+[A-Z0-9]{1,3}\b"),
    re.compile(r"\bT1Weighted\b"),
    re.compile(r"\bT2Weighted\b"),
    re.compile(r"\bT4Weighted\b"),
    re.compile(r"\bQ1\b"),
    re.compile(r"\bQ2\b"),
    re.compile(r"\bD[1-7]\b"),
    re.compile(r"\bpreregister(?:ed|ing|s)?\b", re.IGNORECASE),
    re.compile(r"\brefreeze(?:d|s|ing)?\b", re.IGNORECASE),
    re.compile(r"\brefrozen\b", re.IGNORECASE),
    re.compile(r"\brefroze\b", re.IGNORECASE),
    re.compile(r"\boverhaul\b", re.IGNORECASE),
    re.compile(r"\bpromoted backbone\b", re.IGNORECASE),
    re.compile(r"\bcutover\b", re.IGNORECASE),
)


def _collect_strings(value: object) -> list[str]:
    """Return all nested string values from a JSON-safe payload."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for nested in value.values():
            strings.extend(_collect_strings(nested))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for nested in value:
            strings.extend(_collect_strings(nested))
        return strings
    return []


def _find_banned_terms(text: str) -> list[str]:
    """Return the sorted banned-term matches found in one text blob."""
    return sorted(
        {
            match.group(0)
            for pattern in _BANNED_METHODOLOGY_PATTERNS
            for match in pattern.finditer(text)
        }
    )


def _iter_language_scan_paths() -> list[Path]:
    """Return all Python source files covered by the durable-language scan."""
    return sorted(
        [
            *(
                path
                for path in _SOURCE_ROOT.rglob("*.py")
                if path not in _LANGUAGE_SCAN_EXEMPT_SOURCES
            ),
            *(
                path
                for path in _TESTS_ROOT.rglob("*.py")
                if path not in _LANGUAGE_SCAN_EXEMPT_SOURCES
            ),
        ]
    )


def _build_validation_cli_help(capsys: pytest.CaptureFixture[str]) -> str:
    """Return the validation command's CLI help text."""
    with pytest.raises(SystemExit):
        walk_forward._parse_args(["--help"])
    return capsys.readouterr().out


def _capture_composite_weight_cli_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Return the composite-weight command's report text with lightweight fixtures."""
    team_rows = pl.DataFrame(
        {
            "season": [2020],
            "next_season": [2021],
            "team": ["A"],
            **{component.name: [0.0] for component in composite_weights.TEAM_SACR_COMPONENTS},
            "target": [0.0],
        }
    )
    qb_rows = pl.DataFrame(
        {
            "season": [2020],
            "next_season": [2021],
            "qb_id": ["qb-a"],
            "qb_dropbacks": [30.0],
            **{component.name: [0.0] for component in composite_weights.QB_QSACR_COMPONENTS},
            "target": [0.0],
        }
    )

    def fake_fit(
        df: pl.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        target_column: str,
        sample_weight_column: str | None = None,
    ) -> dict[str, float]:
        del df, target_column, sample_weight_column
        return {column: float(index + 1) for index, column in enumerate(feature_columns)}

    def fake_eval(
        df: pl.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        target_column: str,
        holdout_column: str,
        sample_weight_column: str | None = None,
    ) -> dict[str, float]:
        del df, feature_columns, target_column, holdout_column, sample_weight_column
        return {
            "weighted_mae": 0.1,
            "weighted_rmse": 0.2,
            "equal_weight_mae": 0.3,
            "equal_weight_rmse": 0.4,
        }

    monkeypatch.setattr(
        composite_weights, "build_team_training_rows", lambda data_dir, seasons: team_rows
    )
    monkeypatch.setattr(
        composite_weights, "build_qb_training_rows", lambda data_dir, seasons: qb_rows
    )
    monkeypatch.setattr(composite_weights, "fit_linear_weights", fake_fit)
    monkeypatch.setattr(composite_weights, "evaluate_leave_one_season_out", fake_eval)

    composite_weights.main()
    return capsys.readouterr().out


def test_durable_methodology_surfaces_avoid_campaign_terminology(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Durable user-facing surfaces should stay free of campaign-history terminology."""
    registry_payload = get_registry().payload()
    baseline_labels = "\n".join(
        str(inspect.signature(function).parameters["baseline_name"].default)
        for function in (
            walk_forward.run_weighted_team_backtest,
            walk_forward.run_weighted_team_special_teams_backtest,
            walk_forward.run_play_level_team_special_teams_backtest,
        )
    )
    named_texts = {
        "registry_payload": "\n".join(_collect_strings(registry_payload)),
        "published_rating_columns": "\n".join(_PUBLISHED_RATING_COLUMNS),
        "validation_cli_help": _build_validation_cli_help(capsys),
        "validation_baseline_labels": baseline_labels,
        "composite_weights_cli": _capture_composite_weight_cli_output(monkeypatch, capsys),
        **{
            path.relative_to(_REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in _iter_language_scan_paths()
        },
        **{
            path.relative_to(_REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in _DURABLE_DOC_PATHS
        },
    }

    matches = {
        name: found for name, text in named_texts.items() if (found := _find_banned_terms(text))
    }

    assert matches == {}


def test_durable_methodology_language_scan_exemptions_stay_explicit() -> None:
    """The repo-wide language scan should keep only the approved source exemptions."""
    assert {path.relative_to(_REPO_ROOT).as_posix() for path in _LANGUAGE_SCAN_EXEMPT_SOURCES} == {
        "nfl_sos_ratings/validation/history_strings.py",
        "tests/test_methodology_language_policy.py",
    }
