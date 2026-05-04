"""Configuration constants for NFL Strength of Schedule analysis."""

# Season range for the full pipeline run (nfl-sos-pipeline command).
# 2016 is the first season with available Next Gen Stats data.
START_YEAR: int = 2016
END_YEAR: int = 2025

# Single-season target — used by main.py and visualize.py when invoked directly.
SEASON: int = END_YEAR

# Output directory for CSVs
OUTPUT_DIR: str = "output"

# NFL division mapping
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SF", "SEA"],
}

# Build a team-to-division lookup
TEAM_TO_DIVISION = {}
for div, teams in DIVISIONS.items():
    for team in teams:
        TEAM_TO_DIVISION[team] = div

# All 32 teams
ALL_TEAMS = sorted(TEAM_TO_DIVISION.keys())

# Canonicalize nflverse source differences. Schedules/team stats use LA for the Rams in some
# datasets, while Next Gen Stats and this project use LAR.
TEAM_ABBR_ALIASES = {"LA": "LAR"}

# QB Next Gen Stats columns to extract (from load_nextgen_stats passing data)
QB_NGS_COLS = [
    "avg_time_to_throw",
    "avg_completed_air_yards",
    "avg_intended_air_yards",
    "avg_air_distance",
    "max_completed_air_distance",
    "avg_air_yards_to_sticks",
    "attempts",
    "pass_yards",
    "pass_touchdowns",
    "interceptions",
    "passer_rating",
    "completions",
    "completion_percentage",
    "expected_completion_percentage",
    "completion_percentage_above_expectation",
    "avg_air_yards_differential",
    "aggressiveness",
    "max_air_distance",
]
