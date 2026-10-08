"""Compare every API payload of two running ``nfl-sos-ratings web`` servers, value for value.

Written for the Polars 2.0 parity check (``.agents/current-status.md``): run one server per Polars
version on the same ``data/`` (for example ``--port 8090`` from this repo and ``--port 8092`` from
a worktree on the older lockfile), then compare the seasons list, the metric registry, every
season's tables, and for the given seasons the rank ranges, garbage-time tables at 0, 1, 5, 10, and
20%, and every team's and QB's detail endpoints. Floats must match exactly. Writes to stdout.

Run from the repository root:

    .venv/bin/python .agents/findings_2026_10_08/api_parity.py http://127.0.0.1:8090 \
        http://127.0.0.1:8092 1999 2012 2025 2026
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request

DETAIL_ENDPOINTS = ("game-logs", "rating-history", "rating-pairs", "rank-history")
THRESHOLDS = (0, 1, 5, 10, 20)


def fetch(base: str, path: str) -> object:
    """Return the JSON at ``base + path``."""
    with urllib.request.urlopen(base + path, timeout=120) as response:  # noqa: S310 - local URL
        return json.loads(response.read())


def differences(left: object, right: object, where: str = "$") -> list[str]:
    """Return every place two JSON values differ (keys, lengths, or values)."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = [] if list(left) == list(right) else [f"{where}: keys differ"]
        for key in left.keys() & right.keys():
            found += differences(left[key], right[key], f"{where}.{key}")
        return found
    if isinstance(left, list) and isinstance(right, list):
        found = [] if len(left) == len(right) else [f"{where}: lengths differ"]
        for index, (a, b) in enumerate(zip(left, right, strict=False)):
            found += differences(a, b, f"{where}[{index}]")
        return found
    if isinstance(left, float) and isinstance(right, float):
        same = left == right or (math.isnan(left) and math.isnan(right))
        return [] if same else [f"{where}: {left!r} != {right!r}"]
    return (
        [] if left == right and type(left) is type(right) else [f"{where}: {left!r} != {right!r}"]
    )


def paths(base: str, seasons: list[int]) -> list[str]:
    """Return every endpoint path to compare."""
    listed = fetch(base, "/api/seasons")
    every = listed["seasons"] if isinstance(listed, dict) else []
    found = ["/api/metadata", "/api/seasons", *(f"/api/seasons/{season}" for season in every)]
    for season in seasons:
        payload = fetch(base, f"/api/seasons/{season}")
        if not isinstance(payload, dict):
            msg = f"/api/seasons/{season} did not return an object"
            raise TypeError(msg)
        teams = [row["team"] for row in payload["teams"]["rows"]]
        qbs = [row["qb_id"] for row in payload["qbs"]["rows"]]
        for kind in ("teams", "qbs"):
            found.append(f"/api/seasons/{season}/{kind}/rating-ranges")
            found += [f"/api/seasons/{season}/{kind}/wp-ratings?threshold={t}" for t in THRESHOLDS]
        for kind, ids in (("teams", teams), ("qbs", qbs)):
            found += [
                f"/api/seasons/{season}/{kind}/{entity}/{endpoint}"
                for entity in ids
                for endpoint in DETAIL_ENDPOINTS
            ]
    return found


def main() -> None:
    """Compare the two servers and print each differing payload and the totals."""
    left, right, *seasons = sys.argv[1:]
    compared = differing = missing = 0
    for path in paths(left, [int(season) for season in seasons]):
        try:
            a, b = fetch(left, path), fetch(right, path)
        except urllib.error.HTTPError:
            missing += 1
            continue
        compared += 1
        found = differences(a, b)
        if found:
            differing += 1
            sys.stdout.write(f"{path}: {len(found)} differences, first {found[0]}\n")
    sys.stdout.write(
        f"{compared} payloads compared, {differing} differ, {missing} not served (404)\n"
    )


if __name__ == "__main__":
    main()
