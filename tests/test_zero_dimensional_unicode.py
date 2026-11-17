# Copyright (c) 2026, CAOShurong
"""Zero-dimensional Unicode arrays retain their values and container metadata."""

import numpy as np
import pytest

import hdf5storage


@pytest.mark.parametrize("mode", ["matlab", "utf16", "utf32"])
@pytest.mark.parametrize(
    "value",
    [
        np.array("sample"),
        np.array("\u6e2c\u91cf"),
        np.array("a\U0001f600b"),
        np.array(""),
        np.array("a", dtype="U5"),
        np.array("a\x00b"),
        np.str_("sample"),
        np.array(["sample"]),
        np.array(["a", "bb"], dtype="U5"),
        np.array([["a", "bb"]], dtype="U5"),
        np.array("a\U0001f600b", dtype="U6"),
    ],
)
def test_unicode_array_roundtrip(tmp_path, value, mode):
    options = {} if mode == "matlab" else {"matlab_compatible": False, "convert_numpy_str_to_utf16": mode == "utf16"}
    filename = tmp_path / "unicode.h5"
    hdf5storage.write(value, path="value", filename=filename, **options)

    actual = hdf5storage.read(path="value", filename=filename)
    assert type(actual) is type(value)
    np.testing.assert_array_equal(actual, value)
    assert actual.shape == value.shape
    assert actual.dtype == value.dtype
