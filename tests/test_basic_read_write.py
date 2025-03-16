"""Test the basic read/write functions in hdf5storage.__init__."""

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import hdf5storage

try:
    importlib.import_module("scipy.io")
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@pytest.mark.parametrize(
    ("conv_file_path", "conv_internal_path"),
    [
        (lambda path: path, lambda path: path),
        (lambda path: str(path), lambda path: path),
        (lambda path: path, lambda path: str(path)),
        (lambda path: str(path), lambda path: str(path)),
    ],
)
def test_file(conv_file_path: Callable, conv_internal_path: Callable, tmp_path: Path):
    """Test the hdf5storage.File class."""
    file_path = tmp_path / "file.h5"
    internal_path = Path("/abc")
    data = np.array([1.0, 2.0, 3.0], "f8")
    with hdf5storage.File(conv_file_path(file_path), writable=True) as h5file:
        h5file.write(data, conv_internal_path(internal_path))
    assert file_path.is_file()
    with hdf5storage.File(conv_file_path(file_path), writable=False) as h5file:
        output = h5file.read(conv_internal_path(internal_path))
    np.testing.assert_allclose(output, data)


@pytest.mark.parametrize(
    ("conv_file_path", "conv_internal_path"),
    [
        (lambda path: path, lambda path: path),
        (lambda path: str(path), lambda path: path),
        (lambda path: path, lambda path: str(path)),
        (lambda path: str(path), lambda path: str(path)),
    ],
)
def test_writes_reads(conv_file_path: Callable, conv_internal_path: Callable, tmp_path: Path):
    """Test the writes and reads functions."""
    file_path = tmp_path / "file.h5"
    internal_path = Path("/abc")
    data = np.array([1.0, 2.0, 3.0], "f8")
    hdf5storage.writes({conv_internal_path(internal_path): data}, filename=conv_file_path(file_path))
    assert file_path.is_file()
    output = hdf5storage.reads([conv_internal_path(internal_path)], filename=conv_file_path(file_path))
    assert isinstance(output, list)
    assert len(output) == 1
    np.testing.assert_allclose(output[0], data)


@pytest.mark.parametrize(
    ("conv_file_path", "conv_internal_path"),
    [
        (lambda path: path, lambda path: path),
        (lambda path: str(path), lambda path: path),
        (lambda path: path, lambda path: str(path)),
        (lambda path: str(path), lambda path: str(path)),
    ],
)
def test_write_read(conv_file_path: Callable, conv_internal_path: Callable, tmp_path: Path):
    """Test the write and read functions."""
    file_path = tmp_path / "file.h5"
    internal_path = Path("/abc")
    data = np.array([1.0, 2.0, 3.0], "f8")
    hdf5storage.write(data, conv_internal_path(internal_path), filename=conv_file_path(file_path))
    assert file_path.is_file()
    output = hdf5storage.read(conv_internal_path(internal_path), filename=conv_file_path(file_path))
    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(output, data)


@pytest.mark.parametrize("conv_file_path", [lambda path: path, lambda path: str(path)])
def test_savemat_loadmat(conv_file_path: Callable, tmp_path: Path):
    """Test the savemat and loadmat functions."""
    file_path = tmp_path / "file.mat"
    data = {"abc": np.array([1.0, 2.0, 3.0], "f8")}
    hdf5storage.savemat(conv_file_path(file_path), data)
    assert file_path.is_file()
    output = hdf5storage.loadmat(conv_file_path(file_path))
    assert isinstance(output, dict)
    assert set(output.keys()) == {"abc"}
    np.testing.assert_allclose(output["abc"], data["abc"])


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not available")
@pytest.mark.parametrize(
    ("conv_file_path", "fmt"),
    [(lambda path: path, "4"), (lambda path: path, "5"), (lambda path: str(path), "4"), (lambda path: str(path), "5")],
)
def test_savemat_loadmat_notv7p3(conv_file_path: Callable, fmt: str, tmp_path: Path):
    """Test the savemat and loadmat functions, using matfile versions 4 and 5 that are dispatched to scipy.io."""
    file_path = tmp_path / "file.mat"
    data = {"abc": np.array([1.0, 2.0, 3.0], "f8")}
    hdf5storage.savemat(conv_file_path(file_path), data, fmt=fmt)
    assert file_path.is_file()
    output = hdf5storage.loadmat(conv_file_path(file_path))
    assert isinstance(output, dict)
    assert "abc" in output
    np.testing.assert_allclose(output["abc"][0, :], data["abc"])


@pytest.mark.parametrize("conv_file_path", [lambda path: path, lambda path: str(path)])
def test_savemat_appendmat(conv_file_path: Callable, tmp_path: Path):
    """Test the savemat function, automatically appending .mat to the path."""
    file_path = tmp_path / "file"
    expected_file_path = tmp_path / "file.mat"
    data = {"abc": np.array([1.0, 2.0, 3.0], "f8")}
    hdf5storage.savemat(conv_file_path(file_path), data, appendmat=True)
    assert not file_path.is_file()
    assert expected_file_path.is_file()
    output = hdf5storage.loadmat(conv_file_path(file_path), appendmat=True)
    assert isinstance(output, dict)
    assert set(output.keys()) == {"abc"}
    np.testing.assert_allclose(output["abc"], data["abc"])
