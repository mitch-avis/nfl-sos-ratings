"""Configuration constants for NFL Strength of Schedule analysis."""

# Completed seasons: the range `nfl-sos-ratings pipeline` builds and `validate` evaluates.
START_YEAR: int = 1999
END_YEAR: int = 2025

# Default for `nfl-sos-ratings season`: the current season, which may still be in progress. A
# partial season rates only the games played so far, and its ratings lean hard on the league
# average until enough games are in.
SEASON: int = 2026

# Data directory for generated Parquet files
DATA_DIR: str = "data"

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
TEAM_TO_DIVISION: dict[str, str] = {}
for div, teams in DIVISIONS.items():
    for team in teams:
        TEAM_TO_DIVISION[team] = div

# Canonicalize nflverse source differences. Schedules/team stats use LA for the Rams in some
# datasets, while Next Gen Stats and this project use LAR.
TEAM_ABBR_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",  # ESPN QBR uses WSH for Washington
}
