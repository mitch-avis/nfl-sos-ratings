"""Shared pytest setup: import path, Polars threads, published-data coverage, blocked downloads.

- The repo root goes on the import path so ``tests.stubs`` resolves.
- Polars runs on one thread unless ``POLARS_MAX_THREADS`` is already set: the suite builds
  thousands of tiny frames, where a full thread pool spends its time waking threads (the suite ran
  several times slower with all cores). Polars reads the variable when it is first imported, which
  happens only once the test modules load, after this file.
- ``pytest -m published_data`` runs the checks of the generated files in ``data/``. They exercise
  little code, so when every selected test is one of them the coverage floor and report are lifted;
  any run that includes a code test keeps both.
- No test reaches nflverse: every nflreadpy download fails, cache reads included, and so does
  ``urllib.request.urlopen`` (the ESPN QBR release assets), so a loader a test forgets to stub
  raises instead of downloading the real file or reading the user's on-disk cache (which another
  test may have switched nflreadpy to).
"""

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POLARS_THREAD_VARIABLE = "POLARS_MAX_THREADS"
PUBLISHED_DATA_MARKER = "published_data"


class _CoverageOptions(Protocol):
    """The two pytest-cov options the hook sets (``--cov-fail-under`` and ``--cov-report``)."""

    cov_fail_under: float | None
    cov_report: object


class _Marked(Protocol):
    """A collected test, as far as the marker check needs it."""

    def get_closest_marker(self, name: str) -> object | None:
        """Return the named marker, or ``None`` when the test does not carry it."""
        ...


def limit_polars_threads(environ: MutableMapping[str, str]) -> None:
    """Default ``POLARS_MAX_THREADS`` to one thread in ``environ``, keeping any explicit value."""
    environ.setdefault(POLARS_THREAD_VARIABLE, "1")


def only_published_data(items: Sequence[_Marked]) -> bool:
    """Return whether at least one test is selected and every one reads the generated data files."""
    return bool(items) and all(item.get_closest_marker(PUBLISHED_DATA_MARKER) for item in items)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Lift the coverage floor and report when only published-data checks are selected.

    Runs after the ``-m`` deselection, so ``items`` is what will run. Sets the pytest-cov options
    ``--cov-fail-under`` and ``--cov-report`` to nothing for this run only.
    """
    plugin = config.pluginmanager.getplugin("_cov")
    options = cast("_CoverageOptions | None", getattr(plugin, "options", None))
    if options is None or not only_published_data(items):
        return
    options.cov_fail_under = 0
    options.cov_report = {}


@pytest.fixture(autouse=True)
def _block_nflverse_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every nflverse download fail for the test.

    That covers nflreadpy's, before its cache is consulted, and the release assets (ESPN QBR)
    the loaders fetch with ``urllib``.
    """
    # Imported here: nflreadpy imports Polars, which must wait for the thread default below.
    import urllib.request

    from nflreadpy.downloader import NflverseDownloader

    from tests.stubs import NflverseDownloadBlockedError

    def refuse(
        _self: object, repository: str, path: str, *_args: object, **_kwargs: object
    ) -> None:
        msg = f"a test reached nflverse ({repository}: {path}); stub the nflreadpy loader instead"
        raise NflverseDownloadBlockedError(msg)

    def refuse_url(url: object, *_args: object, **_kwargs: object) -> None:
        msg = f"a test opened {url}; stub the download instead"
        raise NflverseDownloadBlockedError(msg)

    monkeypatch.setattr(NflverseDownloader, "download", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse_url)


limit_polars_threads(os.environ)
