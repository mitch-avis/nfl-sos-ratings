"""Typed stand-ins for functions the tests monkeypatch."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def stub[T](factory: Callable[[], T]) -> Callable[..., T]:
    """Return a stand-in that accepts any arguments and returns ``factory()`` on every call.

    Wrapping the result in a zero-argument factory keeps the per-call evaluation of the lambda it
    replaces while giving the type checker a fully known return type.
    """

    def stand_in(*_args: object, **_kwargs: object) -> T:
        return factory()

    return stand_in


def unscaled(values: Sequence[float]) -> np.ndarray:
    """Return ``values`` as a float array, standing in for a z-score that leaves them as-is."""
    return np.array(values, dtype=np.float64)


def unscaled_against(values: Sequence[float], reference_values: Sequence[float]) -> np.ndarray:
    """Return ``values`` as a float array, ignoring the reference a z-score would scale by."""
    del reference_values
    return np.array(values, dtype=np.float64)
