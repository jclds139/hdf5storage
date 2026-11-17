# Copyright (c) 2026, CAOShurong
"""Read strings whose code units were stored as column vectors."""

import numpy as np
import pytest

import hdf5storage


@pytest.mark.parametrize("oned_as", ["row", "column"])
def test_nested_string_dictionary(tmp_path, oned_as):
    data = {"metadata": {"paths": {"0": "string0", "1": "string1", "2": "string2"}}}
    filename = tmp_path / "nested.mat"
    hdf5storage.savemat(filename, data, oned_as=oned_as)

    assert hdf5storage.loadmat(filename) == data


@pytest.mark.parametrize("oned_as", ["row", "column"])
@pytest.mark.parametrize(
    "data",
    [
        "sample",
        "\u6e2c\u91cf",
        "a\U0001f600b",
        b"sample",
        np.str_("sample"),
        np.bytes_(b"sample"),
        np.array(b"sample"),
        np.array(["one", "two"]),
        np.array([b"one", b"two"]),
        np.array(["a\U0001f600b", "other"]),
        np.array([["one"], ["two"]]),
        np.array([[b"one"], [b"two"]]),
        "",
        b"",
        np.array([], dtype="U3"),
        np.array([], dtype="S3"),
    ],
)
def test_string_readback(tmp_path, oned_as, data):
    filename = tmp_path / "strings.h5"
    hdf5storage.write(data, path="value", filename=filename, oned_as=oned_as)

    # Reading an existing file must not require knowing the writer's oned_as.
    actual = hdf5storage.read(path="value", filename=filename)
    assert type(actual) is type(data)
    np.testing.assert_array_equal(actual, data)
    if isinstance(data, np.ndarray):
        assert actual.shape == data.shape
        assert actual.dtype == data.dtype
