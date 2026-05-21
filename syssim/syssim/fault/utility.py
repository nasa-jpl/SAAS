from __future__ import annotations

from typing import Any

import numpy as np


def zero_like(value: Any):
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if isinstance(value, (float, int, np.number)):
        return type(value)(0)
    raise TypeError("ZeroFault supports numeric scalars and numpy arrays")


def nan_like(value: Any):
    if isinstance(value, np.ndarray):
        return np.full(value.shape, np.nan, dtype=float)
    if isinstance(value, (float, int, np.number)):
        return float("nan")
    raise TypeError("DisconnectFault supports numeric scalars and numpy arrays")