from __future__ import annotations

from typing import Any

import numpy as np


def zero_like(value: Any):
    """Return a zero value with the same container style as ``value``.

    Parameters
    ----------
    value : Any
        Numeric scalar or NumPy array to mirror.

    Returns
    -------
    Any
        Zero scalar or array compatible with ``value``.

    Raises
    ------
    TypeError
        If ``value`` is not a supported numeric scalar or NumPy array.
    """
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if isinstance(value, (float, int, np.number)):
        return type(value)(0)
    raise TypeError("ZeroFault supports numeric scalars and numpy arrays")


def nan_like(value: Any):
    """Return a NaN value with the same container style as ``value``.

    Parameters
    ----------
    value : Any
        Numeric scalar or NumPy array to mirror.

    Returns
    -------
    Any
        NaN scalar or array compatible with ``value``.

    Raises
    ------
    TypeError
        If ``value`` is not a supported numeric scalar or NumPy array.
    """
    if isinstance(value, np.ndarray):
        return np.full(value.shape, np.nan, dtype=float)
    if isinstance(value, (float, int, np.number)):
        return float("nan")
    raise TypeError("DisconnectFault supports numeric scalars and numpy arrays")