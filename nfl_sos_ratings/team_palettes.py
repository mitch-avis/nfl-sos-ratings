"""Team color palettes for the analyst app, derived from nflverse team colors.

The app's ``Palette`` menu offers the default palette and one per team, and a team or QB page
switches to that team's palette. A team palette swaps the accent (buttons, links, focus rings, the
sidebar's active item, the first chart color), the second chart color, the hint cards (tooltips),
the app's logo mark, and the table heat scale for the team's own colors, in light and dark mode
alike. Page surfaces (background, cards, menus, sidebar) always keep the default palette's neutral
colors, so moving between teams never changes them.

Colors come from nflverse's teams table (``team_color`` through ``team_color4``; the first two are
the team's primary and secondary colors). For each mode:

- **Accent:** the more vivid of the two main colors that can reach readable contrast with little
  change, its lightness moved (hue kept, chroma kept where sRGB allows) until text on it and links
  in it reach ``TEXT_CONTRAST`` (WCAG AA for normal text) on every surface links sit on. Hover and
  selected backgrounds (``accent``, ``sidebar_accent``) and the hint card are tints of the accent
  hue, and the hint card's border is the accent itself.
- **Second chart color:** the other main color, or, when that is black, white, silver, or too close
  in hue, the most distinct of the extra listed colors; moved until it reaches ``MARK_CONTRAST``.
- **Heat scale:** pale tints (light mode) or deep shades (dark mode) of the accent hue for the good
  end and the second color's hue for the bad end, around a neutral middle. When the second color is
  black, white, silver, or its tint looks like the accent's, the bad end is a muted gray of it, so
  the team color marks the good end. A team without any hue (black and silver) gets a lightness
  scale instead: darker is better in light mode, lighter in dark mode.

The logo mark is the darker main color with a line in the other one (or the most contrasting listed
color, or white). Run ``nfl-sos-ratings team-palettes`` to regenerate
``web/src/domain/teamPaletteData.json``; it downloads nflverse's teams table and writes only that
file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from nfl_sos_ratings.config import DIVISIONS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

type Oklch = tuple[float, float, float]
type Rgb = tuple[float, float, float]
type Mode = Literal["light", "dark"]
MODES: tuple[Mode, ...] = ("light", "dark")


class HeatScale(TypedDict):
    """A heat scale's two ends and middle as 0-255 sRGB channels."""

    good: list[int]
    bad: list[int]
    mid: list[int]


class ModeTokens(TypedDict):
    """One mode's palette: the CSS colors it sets and its heat scale (``None``: the default)."""

    primary: str
    primary_foreground: str
    ring: str
    sidebar_primary: str
    sidebar_primary_foreground: str
    sidebar_ring: str
    chart_1: str
    chart_2: str
    accent: str
    accent_foreground: str
    sidebar_accent: str
    sidebar_accent_foreground: str
    hint: str
    hint_border: str
    heat: HeatScale | None


class ModePalettes(TypedDict):
    """A palette's light and dark tokens."""

    light: ModeTokens
    dark: ModeTokens


class BrandMark(TypedDict):
    """The app logo's colors for a team: its tile, the line drawn on it, and the line's end dot."""

    background: str
    line: str
    dot: str


class TeamPalette(ModePalettes):
    """One team's entry in the palette file."""

    name: str
    conference: str
    division: str
    source: list[str]
    mark: BrandMark


PALETTE_PATH = (
    Path(__file__).resolve().parents[1] / "web" / "src" / "domain" / "teamPaletteData.json"
)
TEXT_CONTRAST = 4.5
MARK_CONTRAST = 3.0
# Below this OKLCH chroma a color reads as black, white, silver, or gray.
NEUTRAL_CHROMA = 0.025
# The most lightness an accent may move and still count as the team's color with little change.
MAX_ACCENT_SHIFT = 0.15
# Two hues closer than this many degrees do not separate the ends of a heat scale.
MIN_HUE_DISTANCE = 30.0
# The smallest OKLab distance (x100) between the heat scale's two ends, so the gradient reads.
MIN_HEAT_SEPARATION: dict[Mode, float] = {"light": 8.0, "dark": 8.0}
# The smallest OKLab distance (x100) between either heat-scale end and the card, so the best and
# worst cells never look unshaded.
MIN_CARD_SEPARATION = 3.0
# Light-mode accents start no darker than this, so links stay apart from the near-black body text
# (the default palette's accent sits at 0.42).
LIGHT_ACCENT_FLOOR = 0.4
LIGHTNESS_STEP = 0.005
# The default palette's surfaces and text colors (web/src/index.css). Links sit on background, card,
# and muted; ``on_accent`` is the text on the accent.
SURFACES: dict[Mode, dict[str, Oklch]] = {
    "light": {
        "background": (0.985, 0.002, 250.0),
        "card": (1.0, 0.0, 0.0),
        "muted": (0.955, 0.006, 255.0),
        "accent": (0.94, 0.02, 262.0),
        "sidebar_accent": (0.93, 0.015, 262.0),
        "foreground": (0.2, 0.02, 260.0),
        "muted_foreground": (0.52, 0.02, 260.0),
        "accent_foreground": (0.3, 0.05, 262.0),
        "on_accent": (0.985, 0.002, 250.0),
    },
    "dark": {
        "background": (0.17, 0.015, 260.0),
        "card": (0.21, 0.017, 260.0),
        "muted": (0.26, 0.018, 260.0),
        "accent": (0.3, 0.04, 262.0),
        "sidebar_accent": (0.28, 0.03, 262.0),
        "foreground": (0.95, 0.008, 255.0),
        "muted_foreground": (0.7, 0.02, 255.0),
        "accent_foreground": (0.95, 0.008, 255.0),
        "on_accent": (0.17, 0.015, 260.0),
    },
}
LINK_SURFACES = ("background", "card", "muted")
# The surfaces a team palette tints with its accent hue: hover and selected backgrounds, and the
# hint card (tooltips). Page surfaces are never tinted.
ACCENT_SURFACES = ("accent", "sidebar_accent")
# Body text sits on every tinted surface; secondary text (chart tooltip labels) on the hint card.
TEXT_SURFACES = (*ACCENT_SURFACES, "hint")
MUTED_TEXT_SURFACES = ("hint",)
# The chroma of the accent-hue hover and selected backgrounds, and of the hint card.
ACCENT_SURFACE_CHROMA: dict[Mode, float] = {"light": 0.035, "dark": 0.05}
HINT_CHROMA: dict[Mode, float] = {"light": 0.03, "dark": 0.045}
# The hint card's lightness: a little below the card in light mode, a little above it in dark mode.
HINT_LIGHTNESS: dict[Mode, float] = {"light": 0.965, "dark": 0.25}
# Heat-scale tints: light in light mode, deep in dark mode, with a neutral middle.
HEAT_LIGHTNESS: dict[Mode, float] = {"light": 0.84, "dark": 0.36}
HEAT_MAX_CHROMA: dict[Mode, float] = {"light": 0.1, "dark": 0.09}
# The heat scale's neutral middle for team palettes: a cool near-white in light mode, the card in
# dark mode.
HEAT_MID: dict[Mode, list[int]] = {"light": [244, 247, 250], "dark": [22, 27, 34]}
# A muted bad end (a gray of the second color): a little darker than the tints in light mode and a
# little lighter than the card in dark mode, with at most this much chroma.
HEAT_MUTED_LIGHTNESS: dict[Mode, float] = {"light": 0.76, "dark": 0.3}
HEAT_MUTED_CHROMA = 0.012
# A team without any hue gets good, middle, and bad lightnesses instead (darker is better in light
# mode, lighter in dark mode), in its own gray.
HEAT_LIGHTNESS_SCALE: dict[Mode, tuple[float, float, float]] = {
    "light": (0.7, 0.88, 0.97),
    "dark": (0.46, 0.33, 0.25),
}
WHITE = "#FFFFFF"
BLACK = "#000000"
# The logo's dot must stand out from its line by this much (it sits on the line's end).
MIN_DOT_CONTRAST = 2.0
# sRGB transfer function breakpoints, and how far outside 0-1 a linear channel may drift from
# rounding and still count as in gamut.
_SRGB_ENCODED_LIMIT = 0.04045
_SRGB_LINEAR_LIMIT = 0.0031308
_GAMUT_TOLERANCE = 1e-9
_OKLCH_PATTERN = re.compile(r"oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\)")
_COLOR_COLUMNS = ("team_color", "team_color2", "team_color3", "team_color4")


@dataclass(frozen=True, slots=True)
class TeamColors:
    """One team's identity and its listed colors, primary first."""

    team: str
    name: str
    conference: str
    division: str
    colors: tuple[str, ...]


def _to_linear(channel: float) -> float:
    """Return the linear-light value of one gamma-encoded sRGB channel (0-1)."""
    if channel <= _SRGB_ENCODED_LIMIT:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _to_gamma(channel: float) -> float:
    """Return the gamma-encoded value of one linear-light sRGB channel (0-1)."""
    if channel <= _SRGB_LINEAR_LIMIT:
        return 12.92 * channel
    return 1.055 * channel ** (1 / 2.4) - 0.055


def srgb_to_oklch(rgb: Sequence[float]) -> Oklch:
    """Return the OKLCH of a gamma-encoded sRGB color given in 0-255 channels."""
    red, green, blue = (_to_linear(channel / 255.0) for channel in rgb)
    long = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    long_, medium_, short_ = (
        math.copysign(abs(value) ** (1 / 3), value) for value in (long, medium, short)
    )
    lightness = 0.2104542553 * long_ + 0.7936177850 * medium_ - 0.0040720468 * short_
    a = 1.9779984951 * long_ - 2.4285922050 * medium_ + 0.4505937099 * short_
    b = 0.0259040371 * long_ + 0.7827717662 * medium_ - 0.8086757660 * short_
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360.0


def _oklch_to_linear(color: Oklch) -> Rgb:
    """Return the linear-light sRGB of an OKLCH color, unclipped (outside 0-1 when out of gamut)."""
    lightness, chroma, hue = color
    a, b = chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))
    long_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    medium_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    short_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    long, medium, short = long_**3, medium_**3, short_**3
    return (
        4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short,
        -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short,
        -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short,
    )


def oklch_to_srgb(color: Oklch) -> Rgb:
    """Return the gamma-encoded sRGB of an OKLCH color in 0-255 channels, clipped to the gamut."""
    red, green, blue = (
        255.0 * _to_gamma(min(1.0, max(0.0, channel))) for channel in _oklch_to_linear(color)
    )
    return red, green, blue


def _in_gamut(color: Oklch) -> bool:
    """Return whether an OKLCH color is inside the sRGB gamut."""
    return all(
        -_GAMUT_TOLERANCE <= channel <= 1 + _GAMUT_TOLERANCE for channel in _oklch_to_linear(color)
    )


def hex_to_oklch(hex_color: str) -> Oklch:
    """Return the OKLCH of a ``#RRGGBB`` color."""
    digits = hex_color.lstrip("#")
    return srgb_to_oklch([int(digits[index : index + 2], 16) for index in (0, 2, 4)])


def parse_oklch(text: str) -> Oklch:
    """Return the components of an ``oklch(L C h)`` CSS color.

    Raises:
        ValueError: If ``text`` is not that form.

    """
    match = _OKLCH_PATTERN.fullmatch(text.strip())
    if match is None:
        msg = f"not an oklch() color: {text!r}"
        raise ValueError(msg)
    lightness, chroma, hue = (float(part) for part in match.groups())
    return lightness, chroma, hue


def format_oklch(color: Oklch) -> str:
    """Return an ``oklch(L C h)`` CSS color, rounded for a readable file."""
    lightness, chroma, hue = color
    return f"oklch({round(lightness, 3):g} {round(chroma, 3):g} {round(hue, 1):g})"


def _luminance(color: Oklch) -> float:
    """Return the WCAG relative luminance of an OKLCH color."""
    red, green, blue = (min(1.0, max(0.0, channel)) for channel in _oklch_to_linear(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: Oklch, second: Oklch) -> float:
    """Return the WCAG contrast ratio of two colors, from 1 to 21."""
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def color_distance(first: Oklch, second: Oklch) -> float:
    """Return the OKLab distance of two colors, times 100."""

    def lab(color: Oklch) -> tuple[float, float, float]:
        lightness, chroma, hue = color
        return lightness, chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))

    return 100.0 * math.dist(lab(first), lab(second))


def _hue_distance(first: float, second: float) -> float:
    """Return the angle between two hues in degrees, 0-180."""
    gap = abs(first - second) % 360.0
    return min(gap, 360.0 - gap)


def _max_chroma(lightness: float, hue: float, ceiling: float) -> float:
    """Return the largest chroma up to ``ceiling`` that keeps a color in the sRGB gamut."""
    low, high = 0.0, ceiling
    if _in_gamut((lightness, high, hue)):
        return high
    for _ in range(30):
        middle = (low + high) / 2
        if _in_gamut((lightness, middle, hue)):
            low = middle
        else:
            high = middle
    return low


def _is_neutral(color: Oklch) -> bool:
    """Return whether a color reads as black, white, silver, or gray."""
    return color[1] < NEUTRAL_CHROMA


type Surfaces = dict[str, Oklch]


def _passes_accent(color: Oklch, surfaces: Surfaces) -> bool:
    """Return whether text on ``color`` and links in it reach ``TEXT_CONTRAST`` on ``surfaces``."""
    return contrast_ratio(color, surfaces["on_accent"]) >= TEXT_CONTRAST and all(
        contrast_ratio(color, surfaces[surface]) >= TEXT_CONTRAST for surface in LINK_SURFACES
    )


def _passes_mark(color: Oklch, surfaces: Surfaces) -> bool:
    """Return whether ``color`` stands out from the card by ``MARK_CONTRAST``."""
    return contrast_ratio(color, surfaces["card"]) >= MARK_CONTRAST


def _fit(color: Oklch, mode: Mode, surfaces: Surfaces, *, accent: bool) -> tuple[Oklch, float]:
    """Move ``color``'s lightness (darker in light mode, lighter in dark mode) until it passes.

    The hue stays; the chroma stays where the sRGB gamut allows. The check runs on the color as
    the palette file writes it (rounded), so the published value passes. Returns the fitted color
    and how far its lightness moved.
    """
    passes = _passes_accent if accent else _passes_mark
    lightness, chroma, hue = color
    start = max(lightness, LIGHT_ACCENT_FLOOR) if accent and mode == "light" else lightness
    step = -LIGHTNESS_STEP if mode == "light" else LIGHTNESS_STEP
    current = (start, _max_chroma(start, hue, chroma), hue)
    while (
        not passes(parse_oklch(format_oklch(current)), surfaces) and 0.0 < current[0] + step < 1.0
    ):
        next_lightness = current[0] + step
        current = (next_lightness, _max_chroma(next_lightness, hue, chroma), hue)
    return current, abs(current[0] - lightness)


def _accent(main: Sequence[Oklch], mode: Mode, surfaces: Surfaces) -> tuple[int, Oklch]:
    """Return which main color becomes the accent in ``mode``, and its fitted form.

    Among the vivid main colors that move at most ``MAX_ACCENT_SHIFT``, the most vivid wins; with
    none, the main color that moves least.
    """
    vivid = [index for index, color in enumerate(main) if not _is_neutral(color)]
    candidates = vivid or list(range(len(main)))
    fitted = {index: _fit(main[index], mode, surfaces, accent=True) for index in candidates}
    close = [index for index in candidates if fitted[index][1] <= MAX_ACCENT_SHIFT]
    if close:
        best = max(close, key=lambda index: fitted[index][0][1])
    else:
        best = min(candidates, key=lambda index: fitted[index][1])
    return best, fitted[best][0]


def _second(colors: Sequence[Oklch], accent_index: int) -> Oklch | None:
    """Return the second color: the other main color, or the most distinct extra one.

    ``None`` when no listed color is vivid and far enough in hue from the accent's.
    """
    accent_hue = colors[accent_index][2]
    other_main = colors[1 - accent_index] if len(colors) > 1 else None
    if (
        other_main is not None
        and not _is_neutral(other_main)
        and _hue_distance(other_main[2], accent_hue) >= MIN_HUE_DISTANCE
    ):
        return other_main
    extras = [
        color
        for color in colors[2:]
        if not _is_neutral(color) and _hue_distance(color[2], accent_hue) >= MIN_HUE_DISTANCE
    ]
    if not extras:
        return None
    return max(extras, key=lambda color: _hue_distance(color[2], accent_hue))


def _accent_surfaces(accent: Oklch, mode: Mode) -> Surfaces:
    """Return the hover, selected, and hint-card backgrounds and their text, in the accent hue."""
    hue = accent[2]
    tinted: Surfaces = {}
    for surface in ACCENT_SURFACES:
        lightness = SURFACES[mode][surface][0]
        ceiling = min(accent[1], ACCENT_SURFACE_CHROMA[mode])
        tinted[surface] = (lightness, _max_chroma(lightness, hue, ceiling), hue)
    hint_lightness = HINT_LIGHTNESS[mode]
    tinted["hint"] = (
        hint_lightness,
        _max_chroma(hint_lightness, hue, min(accent[1], HINT_CHROMA[mode])),
        hue,
    )
    text_lightness, text_chroma, _ = SURFACES[mode]["accent_foreground"]
    tinted["accent_foreground"] = (
        text_lightness,
        _max_chroma(text_lightness, hue, min(accent[1], text_chroma)),
        hue,
    )
    return tinted


def _rgb(color: Oklch) -> list[int]:
    """Return an OKLCH color as rounded 0-255 sRGB channels."""
    return [round(channel) for channel in oklch_to_srgb(color)]


def _tint(color: Oklch, mode: Mode) -> list[int]:
    """Return the heat-scale tint of ``color``'s hue in ``mode`` as 0-255 sRGB channels."""
    lightness = HEAT_LIGHTNESS[mode]
    hue = color[2]
    chroma = _max_chroma(lightness, hue, min(color[1], HEAT_MAX_CHROMA[mode]))
    return _rgb((lightness, chroma, hue))


def _muted_tint(color: Oklch, mode: Mode) -> list[int]:
    """Return a gray of ``color``'s hue for a heat scale's bad end, set apart by lightness."""
    lightness = HEAT_MUTED_LIGHTNESS[mode]
    chroma = _max_chroma(lightness, color[2], min(color[1], HEAT_MUTED_CHROMA))
    return _rgb((lightness, chroma, color[2]))


def _separated(good: list[int], bad: list[int], mode: Mode) -> bool:
    """Return whether a heat scale's two ends are at least ``MIN_HEAT_SEPARATION`` apart."""
    return color_distance(srgb_to_oklch(good), srgb_to_oklch(bad)) >= MIN_HEAT_SEPARATION[mode]


def _heat(
    colors: Sequence[Oklch], accent_index: int, second: Oklch | None, mode: Mode
) -> HeatScale | None:
    """Return the team's heat scale in ``mode``, or ``None`` when no rule separates its ends.

    ``accent_index`` names the team color that marks the good end; the caller passes the same one
    for both modes, so a team's better end never changes color between light and dark mode. The
    bad end is the second hue's tint when the two tints are far enough apart, otherwise a muted
    gray of the second color (or of the other main color, or failing that of whichever other
    listed color sets the ends farthest apart), so the team's color always means better. A team
    whose color has no hue gets a scale of its own gray by lightness.
    """
    accent = colors[accent_index]
    if _is_neutral(accent):
        lightnesses = dict(zip(("good", "mid", "bad"), HEAT_LIGHTNESS_SCALE[mode], strict=True))
        ceiling = min(accent[1], HEAT_MUTED_CHROMA)
        return {
            "good": _rgb(
                (
                    lightnesses["good"],
                    _max_chroma(lightnesses["good"], accent[2], ceiling),
                    accent[2],
                )
            ),
            "bad": _rgb(
                (lightnesses["bad"], _max_chroma(lightnesses["bad"], accent[2], ceiling), accent[2])
            ),
            "mid": _rgb(
                (lightnesses["mid"], _max_chroma(lightnesses["mid"], accent[2], ceiling), accent[2])
            ),
        }
    good = _tint(accent, mode)
    mid = HEAT_MID[mode]
    if second is not None and _separated(good, bad := _tint(second, mode), mode):
        return {"good": good, "bad": bad, "mid": mid}
    other = second if second is not None else colors[1 - accent_index if len(colors) > 1 else 0]
    rest = [color for index, color in enumerate(colors) if index != accent_index]
    by_separation = sorted(
        (_muted_tint(color, mode) for color in rest),
        key=lambda gray: color_distance(srgb_to_oklch(good), srgb_to_oklch(gray)),
        reverse=True,
    )
    grays = [_muted_tint(other, mode), *by_separation]
    bad = next((gray for gray in grays if _separated(good, gray, mode)), None)
    return None if bad is None else {"good": good, "bad": bad, "mid": mid}


def _mode_palette(colors: Sequence[Oklch], mode: Mode, heat_index: int) -> ModeTokens:
    """Return one mode's palette tokens for a team's listed colors (primary first).

    ``heat_index`` is the main color that marks the heat scale's better end in both modes.
    """
    surfaces = SURFACES[mode]
    main = colors[:2]
    accent_index, accent = _accent(main, mode, surfaces)
    second = _second(colors, accent_index)
    chart2_source = (
        second if second is not None else colors[1 - accent_index if len(colors) > 1 else 0]
    )
    chart2, _ = _fit(chart2_source, mode, surfaces, accent=False)
    accent_surfaces = _accent_surfaces(accent, mode)
    primary = format_oklch(accent)
    on_accent = format_oklch(SURFACES[mode]["on_accent"])
    accent_foreground = format_oklch(accent_surfaces["accent_foreground"])
    return {
        "primary": primary,
        "primary_foreground": on_accent,
        "ring": primary,
        "sidebar_primary": primary,
        "sidebar_primary_foreground": on_accent,
        "sidebar_ring": primary,
        "chart_1": primary,
        "chart_2": format_oklch(chart2),
        "accent": format_oklch(accent_surfaces["accent"]),
        "accent_foreground": accent_foreground,
        "sidebar_accent": format_oklch(accent_surfaces["sidebar_accent"]),
        "sidebar_accent_foreground": accent_foreground,
        "hint": format_oklch(accent_surfaces["hint"]),
        "hint_border": primary,
        "heat": _heat(colors, heat_index, _second(colors, heat_index), mode),
    }


def build_palette(team: TeamColors) -> ModePalettes:
    """Return a team's light and dark palette tokens from its listed colors.

    The accent can differ by mode (a dark green that reads in light mode may not in dark mode),
    but the heat scale's better end always uses the light-mode accent's team color.
    """
    colors = [hex_to_oklch(color) for color in team.colors]
    heat_index, _ = _accent(colors[:2], "light", SURFACES["light"])
    return {
        "light": _mode_palette(colors, "light", heat_index),
        "dark": _mode_palette(colors, "dark", heat_index),
    }


def build_mark(team: TeamColors) -> BrandMark:
    """Return the app logo's colors for a team: its darker main color with a contrasting line.

    The line is the other main color when it stands out from the tile by ``MARK_CONTRAST``,
    otherwise the most contrasting extra listed color that does, otherwise white (black on a
    light tile). The dot at the line's end is white unless the line is too light for it, then the
    listed color other than the tile that stands out from the line most, or the tile's color.
    """
    main = team.colors[:2]
    tile = min(main, key=lambda color: hex_to_oklch(color)[0])
    tile_color = hex_to_oklch(tile)
    others = [color for color in main if color != tile]
    extras = sorted(
        (color for color in team.colors[2:] if color.lower() != tile.lower()),
        key=lambda color: contrast_ratio(hex_to_oklch(color), tile_color),
        reverse=True,
    )
    fallback = WHITE if contrast_ratio(hex_to_oklch(WHITE), tile_color) >= MARK_CONTRAST else BLACK
    line = next(
        (
            color
            for color in (*others, *extras)
            if contrast_ratio(hex_to_oklch(color), tile_color) >= MARK_CONTRAST
        ),
        fallback,
    )
    line_color = hex_to_oklch(line)
    dot = WHITE
    if contrast_ratio(hex_to_oklch(WHITE), line_color) < MIN_DOT_CONTRAST:
        others = [
            color for color in team.colors if color.lower() not in {line.lower(), tile.lower()}
        ]
        best = max(
            others, key=lambda color: contrast_ratio(hex_to_oklch(color), line_color), default=tile
        )
        dot = best if contrast_ratio(hex_to_oklch(best), line_color) >= MIN_DOT_CONTRAST else tile
    return {"background": tile, "line": line, "dot": dot}


def _palette_surfaces(tokens: ModeTokens, mode: Mode) -> Surfaces:
    """Return the default surfaces and text colors, plus a palette's tinted surfaces and text."""
    surfaces = dict(SURFACES[mode])
    for surface in (*TEXT_SURFACES, "accent_foreground", "sidebar_accent_foreground"):
        surfaces[surface] = parse_oklch(tokens[surface])
    return surfaces


def is_readable(palette: ModePalettes, mode: Mode) -> bool:
    """Return whether one mode of a palette meets the readability rules this module builds to.

    Text on the accent and links in it (on the page and in hint cards) reach ``TEXT_CONTRAST``; body
    text reaches it on every tinted surface and secondary text on the hint card; the accent
    surfaces' own text reaches it; both chart colors and the hint card's border reach
    ``MARK_CONTRAST``; and a heat scale keeps text readable on every step while its two ends stay
    ``MIN_HEAT_SEPARATION`` apart and ``MIN_CARD_SEPARATION`` from the card.
    """
    tokens = palette[mode]
    primary = parse_oklch(tokens["primary"])
    surfaces = _palette_surfaces(tokens, mode)
    pairs = [
        (primary, parse_oklch(tokens["primary_foreground"])),
        *((primary, surfaces[surface]) for surface in (*LINK_SURFACES, "hint")),
        *((surfaces["foreground"], surfaces[surface]) for surface in TEXT_SURFACES),
        *((surfaces["muted_foreground"], surfaces[surface]) for surface in MUTED_TEXT_SURFACES),
        (surfaces["accent_foreground"], surfaces["accent"]),
        (surfaces["sidebar_accent_foreground"], surfaces["sidebar_accent"]),
    ]
    readable = all(contrast_ratio(first, second) >= TEXT_CONTRAST for first, second in pairs)
    readable = readable and all(
        _passes_mark(parse_oklch(tokens[key]), surfaces) for key in ("chart_1", "chart_2")
    )
    readable = readable and (
        contrast_ratio(parse_oklch(tokens["hint_border"]), surfaces["hint"]) >= MARK_CONTRAST
    )
    heat = tokens["heat"]
    if heat is None:
        return readable
    steps = [srgb_to_oklch(heat["good"]), srgb_to_oklch(heat["bad"]), srgb_to_oklch(heat["mid"])]
    readable = readable and all(
        contrast_ratio(step, surfaces["foreground"]) >= TEXT_CONTRAST for step in steps
    )
    readable = readable and all(
        color_distance(step, surfaces["card"]) >= MIN_CARD_SEPARATION for step in steps[:2]
    )
    return readable and _separated(heat["good"], heat["bad"], mode)


def load_palette_file(path: Path = PALETTE_PATH) -> dict[str, TeamPalette]:
    """Return the palette file the web app ships, keyed by team."""
    return cast("dict[str, TeamPalette]", json.loads(path.read_text(encoding="utf-8")))


def load_teams() -> pl.DataFrame:
    """Return nflverse's teams table (abbreviation, name, conference, division, colors)."""
    import nflreadpy as nfl  # noqa: PLC0415 - only this command needs the download

    return nfl.load_teams()


def build_palettes(teams: pl.DataFrame) -> dict[str, TeamPalette]:
    """Return every current team's palette, keyed by abbreviation, in division order.

    Current teams are those in ``config.DIVISIONS``; relocated franchises' old rows are skipped.
    """
    rows = {str(row["team_abbr"]): row for row in teams.iter_rows(named=True)}
    palettes: dict[str, TeamPalette] = {}
    for division, members in DIVISIONS.items():
        for team in members:
            row = rows[team]
            colors = tuple(str(row[column]) for column in _COLOR_COLUMNS if row.get(column))
            identity = TeamColors(
                team, str(row["team_name"]), str(row["team_conf"]), division, colors
            )
            modes = build_palette(identity)
            palettes[team] = {
                "name": identity.name,
                "conference": identity.conference,
                "division": division,
                "source": list(colors[:2]),
                "mark": build_mark(identity),
                "light": modes["light"],
                "dark": modes["dark"],
            }
    return palettes


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``team-palettes`` command's options."""
    parser = argparse.ArgumentParser(
        prog="nfl-sos-ratings team-palettes",
        description=(
            "Rebuild the analyst app's team color palettes from nflverse team colors "
            "(downloads the teams table, writes one JSON file)."
        ),
    )
    parser.add_argument(
        "--output", default=str(PALETTE_PATH), help=f"File to write (default: {PALETTE_PATH})."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Download nflverse team colors and write the palette file."""
    args = _parse_args(argv)
    palettes = build_palettes(load_teams())
    output = Path(args.output)
    output.write_text(json.dumps(palettes, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"Wrote {len(palettes)} team palettes to {output}\n")


__all__ = [
    "MARK_CONTRAST",
    "MIN_CARD_SEPARATION",
    "MODES",
    "PALETTE_PATH",
    "SURFACES",
    "TEXT_CONTRAST",
    "BrandMark",
    "HeatScale",
    "ModePalettes",
    "ModeTokens",
    "TeamColors",
    "TeamPalette",
    "build_mark",
    "build_palette",
    "build_palettes",
    "color_distance",
    "contrast_ratio",
    "format_oklch",
    "hex_to_oklch",
    "is_readable",
    "load_palette_file",
    "load_teams",
    "main",
    "oklch_to_srgb",
    "parse_oklch",
    "srgb_to_oklch",
]
