"""typing.py - Functions to help with narrowing NumPy types."""

from typing import Any, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt

_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)


def ndarray_has_type(data: np.ndarray, type_: type[_ScalarType_co]) -> TypeGuard[npt.NDArray[_ScalarType_co]]:
    """Confirm that a numpy array contains a certain type."""
    return np.issubdtype(data.dtype, type_)


def is_ndarray_of_type(
    data: Any,  # noqa: ANN401
    type_: type[_ScalarType_co],
) -> TypeGuard[npt.NDArray[_ScalarType_co]]:
    """Confirm that data is a numpy array of some type."""
    return isinstance(data, np.ndarray) and ndarray_has_type(data, type_)
