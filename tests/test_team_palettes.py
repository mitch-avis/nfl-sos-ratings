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
_GREEN_GOLD = TeamColors(
    "GB", "Green Bay Packers", "NFC", "NFC North", ("#203731", "#FFB612", "#1c2d25", "#eead1e")
)
_BLACK_GOLD = TeamColors(
    "PIT", "Pittsburgh Steelers", "AFC", "AFC North", ("#000000", "#FFB612", "#c60c30", "#00539b")
)
_ORANGE_BLACK = TeamColors(
    "CIN", "Cincinnati Bengals", "AFC", "AFC North", ("#FB4F14", "#000000", "#000000", "#d32f1e")
)
_GREEN_BLACK = TeamColors("NYJ", "New York Jets", "AFC", "AFC East", ("#003F2D", "#000000"))
_NAVY_ORANGE = TeamColors(
    "DEN", "Denver Broncos", "AFC", "AFC West", ("#002244", "#FB4F14", "#00234c", "#ff5200")
)
_FIXTURES = (_RED_NAVY, _BLACK_SILVER, _GREEN_GOLD, _BLACK_GOLD, _ORANGE_BLACK, _GREEN_BLACK)


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


@pytest.mark.parametrize(("mode", "darker_is_better"), [("light", True), ("dark", False)])
def test_build_palette_gives_a_team_without_a_hue_a_lightness_heat_scale(
    mode: Mode, *, darker_is_better: bool
) -> None:
    # Act
    heat = build_palette(_BLACK_SILVER)[mode]["heat"]

    # Assert
    assert heat is not None
    good, mid, bad = (srgb_to_oklch(heat[end])[0] for end in ("good", "mid", "bad"))
    assert (good < mid < bad) if darker_is_better else (good > mid > bad)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_palette_marks_the_bad_end_in_a_muted_second_color_without_two_hues(
    mode: Mode,
) -> None:
    # Act
    heat = build_palette(_GREEN_BLACK)[mode]["heat"]

    # Assert
    assert heat is not None
    good, bad = srgb_to_oklch(heat["good"]), srgb_to_oklch(heat["bad"])
    assert good[2] == pytest.approx(hex_to_oklch("#003F2D")[2], abs=10.0)
    assert bad[1] < team_palettes.NEUTRAL_CHROMA


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


def test_build_palettes_covers_every_current_team_and_builds_the_broncos_like_the_rest() -> None:
    # Act
    palettes = build_palettes(_teams_frame())

    # Assert
    assert sorted(palettes) == sorted(team for teams in DIVISIONS.values() for team in teams)
    assert palettes["DEN"]["light"] == palettes["NE"]["light"]
    assert palettes["DEN"]["mark"] == palettes["NE"]["mark"]


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
        team for team, palette in palettes.items() if not team_palettes.is_readable(palette, mode)
    ]

    # Assert
    assert problems == []


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_committed_palettes_give_every_team_its_own_heat_scale(mode: Mode) -> None:
    # Arrange
    palettes = _committed()

    # Act
    missing = [team for team, palette in palettes.items() if palette[mode]["heat"] is None]

    # Assert
    assert missing == []


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


def test_build_palette_mutes_a_second_color_whose_tint_looks_like_the_accent() -> None:
    # Arrange
    browns = TeamColors("CLE", "Cleveland Browns", "AFC", "AFC North", ("#FF3C00", "#311D00"))

    # Act
    heat = build_palette(browns)["light"]["heat"]

    # Assert
    assert heat is not None
    bad = srgb_to_oklch(heat["bad"])
    assert bad[1] < team_palettes.NEUTRAL_CHROMA
    assert bad[2] == pytest.approx(hex_to_oklch("#311D00")[2], abs=15.0)


def test_load_teams_reads_the_nflverse_teams_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(nflreadpy, "load_teams", stub(_teams_frame))

    # Act
    teams = team_palettes.load_teams()

    # Assert
    assert teams.height == 33


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_palette_leaves_the_page_surfaces_to_the_default(mode: Mode) -> None:
    # Act
    tokens = build_palette(_GREEN_GOLD)[mode]

    # Assert
    assert not {"background", "card", "popover", "muted", "secondary", "sidebar"} & set(tokens)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_palette_tints_the_hint_card_and_borders_it_with_the_accent(mode: Mode) -> None:
    # Act
    tokens = build_palette(_RED_NAVY)[mode]

    # Assert
    accent = parse_oklch(tokens["primary"])
    hint = parse_oklch(tokens["hint"])
    assert hint[2] == pytest.approx(accent[2], abs=1.0)
    assert hint[1] > 0.01
    assert tokens["hint_border"] == tokens["primary"]


def test_is_readable_rejects_unreadable_secondary_text_on_the_hint_card() -> None:
    # Arrange
    palette = build_palette(_RED_NAVY)
    palette["light"]["hint"] = "oklch(0.85 0.03 21.3)"

    # Act
    readable = team_palettes.is_readable(palette, "light")

    # Assert
    assert readable is False


def test_build_palette_tints_the_accent_surfaces_with_the_accent_hue() -> None:
    # Act
    tokens = build_palette(_RED_NAVY)["light"]

    # Assert
    accent_hue = parse_oklch(tokens["primary"])[2]
    for surface in ("accent", "sidebar_accent"):
        tint = parse_oklch(tokens[surface])
        assert tint[2] == pytest.approx(accent_hue, abs=1.0)
        assert tint[1] > SURFACES["light"]["accent"][1]


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("team", _FIXTURES, ids=lambda team: team.team)
def test_build_palette_keeps_every_text_and_mark_readable(team: TeamColors, mode: Mode) -> None:
    # Act
    palette = build_palette(team)

    # Assert
    assert team_palettes.is_readable(palette, mode)


def test_is_readable_rejects_unreadable_text_on_an_accent_surface() -> None:
    # Arrange
    palette = build_palette(_RED_NAVY)
    palette["light"]["accent_foreground"] = palette["light"]["accent"]

    # Act
    readable = team_palettes.is_readable(palette, "light")

    # Assert
    assert readable is False


def test_build_mark_draws_the_logo_in_the_two_main_colors() -> None:
    # Act
    mark = team_palettes.build_mark(_NAVY_ORANGE)

    # Assert
    assert (mark["background"], mark["line"]) == ("#002244", "#FB4F14")


def test_build_mark_takes_an_extra_color_when_the_main_two_are_too_close() -> None:
    # Act
    mark = team_palettes.build_mark(_RED_NAVY)

    # Assert
    assert mark["background"] == "#002244"
    assert contrast_ratio(hex_to_oklch(mark["line"]), hex_to_oklch("#002244")) >= MARK_CONTRAST


def test_build_mark_draws_a_white_line_without_a_contrasting_color() -> None:
    # Act
    mark = team_palettes.build_mark(_GREEN_BLACK)

    # Assert
    assert (mark["background"], mark["line"]) == ("#000000", "#FFFFFF")


def test_build_palette_takes_another_listed_gray_when_the_second_one_is_too_close() -> None:
    # Arrange
    eagles = TeamColors(
        "PHI",
        "Philadelphia Eagles",
        "NFC",
        "NFC East",
        ("#004C54", "#A5ACAF", "#acc0c6", "#000000"),
    )

    # Act
    heat = build_palette(eagles)["dark"]["heat"]

    # Assert
    assert heat is not None
    assert srgb_to_oklch(heat["bad"])[1] < 0.002


def test_build_palette_drops_the_heat_scale_when_no_gray_sets_the_ends_apart() -> None:
    # Arrange
    dull = TeamColors("GB", "Green Bay Packers", "NFC", "NFC North", ("#203731", "#1c2d25"))

    # Act
    heat = build_palette(dull)["dark"]["heat"]

    # Assert
    assert heat is None


def test_is_readable_accepts_a_palette_that_keeps_the_default_heat_scale() -> None:
    # Arrange
    palette = build_palette(
        TeamColors("GB", "Green Bay Packers", "NFC", "NFC North", ("#203731", "#1c2d25"))
    )

    # Act
    readable = team_palettes.is_readable(palette, "dark")

    # Assert
    assert readable is True


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_palette_keeps_the_better_end_in_the_same_team_color_in_both_modes(
    mode: Mode,
) -> None:
    # Act
    heat = build_palette(_GREEN_GOLD)[mode]["heat"]

    # Assert
    assert heat is not None
    assert srgb_to_oklch(heat["good"])[2] == pytest.approx(hex_to_oklch("#203731")[2], abs=12.0)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_build_palette_keeps_every_heat_end_apart_from_the_card(mode: Mode) -> None:
    # Act
    heat = build_palette(_BLACK_SILVER)[mode]["heat"]

    # Assert
    assert heat is not None
    card = SURFACES[mode]["card"]
    for end in ("good", "bad"):
        assert team_palettes.color_distance(srgb_to_oklch(heat[end]), card) >= (
            team_palettes.MIN_CARD_SEPARATION
        )


def test_build_mark_draws_the_dot_in_another_team_color_on_a_white_line() -> None:
    # Act
    mark = team_palettes.build_mark(_GREEN_BLACK)

    # Assert
    assert (mark["line"], mark["dot"]) == ("#FFFFFF", "#003F2D")


def test_build_mark_draws_the_dot_in_the_tile_color_when_no_other_color_stands_out() -> None:
    # Arrange
    two_tone = TeamColors("XX", "Two Tone", "AFC", "AFC East", ("#000000", "#F5F5F5"))

    # Act
    mark = team_palettes.build_mark(two_tone)

    # Assert
    assert (mark["line"], mark["dot"]) == ("#F5F5F5", "#000000")


def test_build_mark_keeps_the_white_dot_on_a_colored_line() -> None:
    # Act
    mark = team_palettes.build_mark(_NAVY_ORANGE)

    # Assert
    assert mark["dot"] == "#FFFFFF"


def test_build_mark_draws_a_black_line_on_a_light_tile_without_a_contrasting_color() -> None:
    # Arrange
    pale = TeamColors("XX", "Pale Team", "AFC", "AFC East", ("#FFF8E1", "#FFFDE7"))

    # Act
    mark = team_palettes.build_mark(pale)

    # Assert
    assert mark["line"] == "#000000"
