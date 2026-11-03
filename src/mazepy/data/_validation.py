"""Internal validation shared by signal containers; no analysis dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def readonly(values, *, dtype=None):
    """Own immutable numeric bytes, reusing already immutable views when possible."""
    array = np.asarray(values, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("Object arrays are not supported for numerical data.")
    owner = array
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    if not array.flags.writeable and isinstance(owner, bytes):
        return array
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def real_array(values, name, *, ndim):
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got {array.shape}.")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numerical values.")
    return array


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return freeze(value.tolist()) if value.dtype.hasobject else readonly(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Metadata must contain mappings, arrays, sequences, or scalars.")


def signal_id(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("Signal IDs must be strings or integers, excluding booleans.")
    return value


def ids(values):
    if isinstance(values, str):
        raise TypeError("Signal IDs must be a sequence of labels, not one string.")
    result = tuple(signal_id(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("Signal IDs must be unique.")
    return result


def positions(selection, size):
    if isinstance(selection, slice):
        result = np.arange(size, dtype=np.intp)[selection]
    else:
        array = np.asarray(selection)
        if array.ndim == 0 and array.dtype.kind in "iu":
            array = array.reshape(1)
        if array.ndim != 1:
            raise ValueError("Sample selection must be one-dimensional.")
        if array.dtype.kind == "b":
            if array.size != size:
                raise ValueError("Boolean selection length must match the sample axis.")
            result = np.flatnonzero(array)
        elif array.size == 0:
            result = np.empty(0, dtype=np.intp)
        elif array.dtype.kind in "iu":
            if np.any(array >= size) or np.any(array < -size):
                raise IndexError("Sample index is outside the recording.")
            result = np.asarray(array, dtype=np.intp)
            result = np.where(result < 0, result + size, result)
        else:
            raise TypeError("Sample selection must contain integers or booleans.")
    if np.any(np.diff(result) <= 0):
        raise ValueError(
            "Sample selection must be strictly increasing without repeats."
        )
    return result


def window(start, stop):
    if isinstance(start, bool) or isinstance(stop, bool):
        raise TypeError("Time bounds must be numerical seconds.")
    start, stop = float(start), float(stop)
    if not np.isfinite([start, stop]).all() or start > stop:
        raise ValueError("Time bounds must be finite and satisfy start <= stop.")
    return start, stop
