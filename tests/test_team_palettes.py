"""Tests for the team color palettes the analyst app offers, derived from nflverse team colors."""

import json
from typing import TYPE_CHECKING

import nflreadpy
import polars as pl
import pytest

from nfl_sos_ratings import team_palettes
from nfl_sos_ratings.config import DIVISIONS
from nfl_sos_ratings.team_palettes import (
    MARK_CONTRAST,
    PALETTE_PATH,
    SURFACES,
    TEXT_CONTRAST,
    TeamColors,
    TeamPalette,
    build_palette,
    build_palettes,
    contrast_ratio,
    hex_to_oklch,
    load_palette_file,
    parse_oklch,
    srgb_to_oklch,
)
from tests.stubs import stub

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_sos_ratings.team_palettes import Mode

_RED_NAVY = TeamColors("NE", "New England Patriots", "AFC", "AFC East", ("#002244", "#C60C30"))
_BLACK_SILVER = TeamColors(
    "LV", "Las Vegas Raiders", "AFC", "AFC West", ("#000000", "#A5ACAF", "#a6aeb0", "#000000")
)


def test_hex_to_oklch_reads_white_and_black() -> None:
    # Act
    white, black = hex_to_oklch("#FFFFFF"), hex_to_oklch("#000000")

    # Assert
    assert white[0] == pytest.approx(1.0, abs=1e-3)
    assert white[1] == pytest.approx(0.0, abs=1e-3)
    assert black[0] == pytest.approx(0.0, abs=1e-3)


def test_contrast_ratio_spans_one_to_twenty_one() -> None:
    # Arrange
    white, black = hex_to_oklch("#FFFFFF"), hex_to_oklch("#000000")

    # Act
    ratios = (contrast_ratio(white, black), contrast_ratio(white, white))

    # Assert
    assert ratios == (pytest.approx(21.0, abs=0.01), pytest.approx(1.0))


def test_hex_to_oklch_round_trips_through_srgb() -> None:
    # Arrange
    color = hex_to_oklch("#FB4F14")

    # Act
    back = srgb_to_oklch(team_palettes.oklch_to_srgb(color))

    # Assert
    assert back == pytest.approx(color, abs=1e-4)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_palette_keeps_text_on_the_accent_readable(mode: Mode) -> None:
    # Act
    palette = build_palette(_RED_NAVY)[mode]

    # Assert
    primary = parse_oklch(palette["primary"])
    foreground = parse_oklch(palette["primary_foreground"])
    assert contrast_ratio(primary, foreground) >= TEXT_CONTRAST
    assert contrast_ratio(primary, SURFACES[mode]["card"]) >= TEXT_CONTRAST
    assert contrast_ratio(parse_oklch(palette["chart_2"]), SURFACES[mode]["card"]) >= MARK_CONTRAST


def test_build_palette_takes_the_more_vivid_main_color_as_the_accent() -> None:
    # Act
    palette = build_palette(_RED_NAVY)

    # Assert
    red_hue = hex_to_oklch("#C60C30")[2]
    assert parse_oklch(palette["light"]["primary"])[2] == pytest.approx(red_hue, abs=1.0)


def test_build_palette_falls_back_to_the_default_heat_scale_without_two_hues() -> None:
    # Act
    palette = build_palette(_BLACK_SILVER)

    # Assert
    assert palette["light"]["heat"] is None
    assert palette["dark"]["heat"] is None


def test_build_palette_tints_the_heat_scale_with_the_team_hues() -> None:
    # Act
    heat = build_palette(_RED_NAVY)["light"]["heat"]

    # Assert
    assert heat is not None
    good, bad = srgb_to_oklch(heat["good"]), srgb_to_oklch(heat["bad"])
    assert abs(good[2] - hex_to_oklch("#C60C30")[2]) < 20
    assert abs(bad[2] - hex_to_oklch("#002244")[2]) < 20


def _teams_frame() -> pl.DataFrame:
    """Return an nflverse-shaped teams table for every current team, plus a relocated one."""
    rows = [
        {
            "team_abbr": team,
            "team_name": f"Team {team}",
            "team_conf": division.split()[0],
            "team_division": division,
            "team_color": "#002244",
            "team_color2": "#C60C30",
            "team_color3": "#B0B7BC",
            "team_color4": "#001532",
        }
        for division, teams in DIVISIONS.items()
        for team in teams
    ]
    rows.append({**rows[0], "team_abbr": "OAK"})
    return pl.DataFrame(rows)


def test_build_palettes_covers_every_current_team_and_keeps_the_broncos() -> None:
    # Act
    palettes = build_palettes(_teams_frame())

    # Assert
    assert sorted(palettes) == sorted(team for teams in DIVISIONS.values() for team in teams)
    assert palettes["DEN"]["light"]["primary"] == "oklch(0.66 0.2 40)"


def test_main_writes_the_palette_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(team_palettes, "load_teams", stub(_teams_frame))
    output = tmp_path / "teamPaletteData.json"

    # Act
    team_palettes.main(["--output", str(output)])

    # Assert
    written = json.loads(output.read_text(encoding="utf-8"))
    assert len(written) == 32
    assert written["KC"]["division"] == "AFC West"


def _committed() -> dict[str, TeamPalette]:
    """Return the palette file the web app ships."""
    return load_palette_file(PALETTE_PATH)


def test_committed_palettes_cover_every_current_team() -> None:
    # Act
    palettes = _committed()

    # Assert
    assert sorted(palettes) == sorted(team for teams in DIVISIONS.values() for team in teams)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_committed_palettes_keep_text_and_marks_readable(mode: Mode) -> None:
    # Arrange
    palettes = _committed()

    # Act
    problems = [
        team
        for team, palette in palettes.items()
        if not team_palettes.is_readable(palette, mode) and not (team == "DEN" and mode == "light")
    ]

    # Assert
    assert problems == []


def test_oklch_to_srgb_round_trips_a_near_black() -> None:
    # Arrange
    color = hex_to_oklch("#010101")

    # Act
    rgb = team_palettes.oklch_to_srgb(color)

    # Assert
    assert rgb == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)


def test_parse_oklch_rejects_other_color_forms() -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="oklch"):
        parse_oklch("#FFFFFF")


def test_build_palette_takes_the_most_distinct_extra_color_after_a_neutral_one() -> None:
    # Arrange
    steelers = TeamColors(
        "PIT",
        "Pittsburgh Steelers",
        "AFC",
        "AFC North",
        ("#000000", "#FFB612", "#C60C30", "#00539B"),
    )

    # Act
    palette = build_palette(steelers)

    # Assert
    blue_hue = hex_to_oklch("#00539B")[2]
    assert parse_oklch(palette["dark"]["chart_2"])[2] == pytest.approx(blue_hue, abs=1.0)


def test_build_palette_drops_a_heat_scale_whose_ends_look_alike() -> None:
    # Arrange
    browns = TeamColors("CLE", "Cleveland Browns", "AFC", "AFC North", ("#FF3C00", "#311D00"))

    # Act
    palette = build_palette(browns)

    # Assert
    assert palette["light"]["heat"] is None


def test_load_teams_reads_the_nflverse_teams_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(nflreadpy, "load_teams", stub(_teams_frame))

    # Act
    teams = team_palettes.load_teams()

    # Assert
    assert teams.height == 33
