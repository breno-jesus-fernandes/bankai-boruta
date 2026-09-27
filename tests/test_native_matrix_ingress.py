import numpy as np
import pytest

from bankai_boruta import _core


@pytest.mark.parametrize(
    "X",
    [
        np.arange(12, dtype=np.float64).reshape(4, 3),
        np.asfortranarray(np.arange(12, dtype=np.float64).reshape(4, 3)),
    ],
)
def test_contiguous_float64_inputs_are_borrowed_for_c_or_fortran_layout(X):
    original = X.copy()

    assert _core.inspect_matrix(X) == (4, 3, 0)
    np.testing.assert_array_equal(X, original)


def test_zero_copy_refuses_noncontiguous_views_with_a_useful_error():
    X = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]

    with pytest.raises(ValueError, match="C-contiguous or Fortran-contiguous"):
        _core.inspect_matrix(X)


@pytest.mark.parametrize(
    "X",
    [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.asfortranarray(np.arange(12, dtype=np.float32).reshape(4, 3)),
    ],
)
def test_float32_inputs_are_also_borrowed_in_contiguous_layouts(X):
    assert _core.inspect_matrix(X) == (4, 3, 0)


@pytest.mark.parametrize(
    "X",
    [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.arange(12, dtype=np.int64).reshape(4, 3),
    ],
)
def test_copy_mode_accepts_convertible_numeric_arrays_and_reports_copy_bytes(X):
    assert _core.inspect_matrix(X, input_mode="copy") == (4, 3, 4 * 3 * 8)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_matrix_ingress_rejects_nonfinite_values(value):
    X = np.array([[0.0, value], [1.0, 2.0]])

    with pytest.raises(ValueError, match="finite"):
        _core.inspect_matrix(X)


def test_copy_mode_is_explicit_and_invalid_modes_fail():
    X = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="input_mode"):
        _core.inspect_matrix(X, input_mode="automatic")


@pytest.mark.parametrize(
    "X",
    [
        np.ones(4, dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.empty((2, 0), dtype=np.float64),
    ],
)
def test_matrix_ingress_requires_a_nonempty_two_dimensional_array(X):
    with pytest.raises((TypeError, ValueError)):
        _core.inspect_matrix(X)


def test_rejects_unaligned_numpy_buffer_before_borrowing():
    source = bytearray(8 * 12 + 1)
    unaligned = np.ndarray((6, 2), dtype=np.float64, buffer=source, offset=1)
    assert unaligned.flags.c_contiguous
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        _core.inspect_matrix(unaligned)
