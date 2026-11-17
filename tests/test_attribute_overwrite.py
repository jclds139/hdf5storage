# Copyright (c) 2026, CAOShurong
"""Metadata must describe the latest value written to an existing dataset."""

import numpy as np
import pytest

import hdf5storage


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (np.str_("sample"), np.bytes_(b"sample")),
        (np.bytes_(b"sample"), np.str_("sample")),
        (np.str_("sample"), np.array(["one", "two"])),
        (np.array(["one", "two"]), np.str_("sample")),
        (np.array(["one", "two"]), np.array([["one", "two"]])),
        (np.array([["one", "two"]]), np.array(["one", "two"])),
        (np.array([b"one", b"two"]), np.array([[b"one", b"two"]])),
        (np.array([[b"one", b"two"]]), np.array([b"one", b"two"])),
        (np.array(["one", "two"]), np.array(["abc", "def"])),
    ],
)
def test_overwrite_string_metadata(tmp_path, before, after):
    filename = tmp_path / "updated.h5"
    hdf5storage.write(before, path="value", filename=filename)
    hdf5storage.write(after, path="value", filename=filename)

    # Reopen the file so all reconstruction uses the persisted metadata.
    actual = hdf5storage.read(path="value", filename=filename)
    assert type(actual) is type(after)
    np.testing.assert_array_equal(actual, after)
    assert actual.shape == after.shape
    assert actual.dtype == after.dtype
