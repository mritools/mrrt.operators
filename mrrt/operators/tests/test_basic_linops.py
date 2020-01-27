from itertools import product

import numpy as np
from numpy.testing import assert_equal, assert_
import pytest

from mrrt.operators import (
    IdentityOperator,
    ZeroOperator,
    DiagonalOperator,
    MaskingOperator,
    IDiagOperator,
    RDiagOperator,
)
from mrrt.utils import config, masker

all_xp = [np]
if config.have_cupy:
    import cupy

    if cupy.cuda.runtime.getDeviceCount() > 0:
        all_xp += [cupy]


def get_loc(xp):
    """Location arguments corresponding to numpy or CuPy case."""
    if xp is np:
        return dict(loc_in="cpu", loc_out="cpu")
    else:
        return dict(loc_in="gpu", loc_out="gpu")


@pytest.mark.parametrize(
    "xp, order, squeeze_reps", product(all_xp, ["C", "F"], [True, False])
)
def test_identity_linop(xp, order, squeeze_reps):
    a = xp.ones(100)

    I = IdentityOperator(
        nargin=a.size, shape=a.shape, order=order, **get_loc(xp)
    )
    c = I * a
    xp.testing.assert_array_equal(a, c)

    c = I.H * I * a
    xp.testing.assert_array_equal(a, c)

    c = I.H * (I * a)
    xp.testing.assert_array_equal(a, c)

    c = (I.H * I) * a
    xp.testing.assert_array_equal(a, c)

    # scipy compatibility
    xp.testing.assert_array_equal(I * a, I.matvec(a))
    xp.testing.assert_array_equal(I.H * a, I.rmatvec(a))

    I2 = IdentityOperator(
        nargin=a.size, order=order, squeeze_reps=squeeze_reps, **get_loc(xp)
    )
    c = I2 * a
    if squeeze_reps:
        assert_(c.ndim == 1)
        xp.testing.assert_array_equal(a, c)
    else:
        assert_(c.ndim == 2)
        if order == "F":
            assert_(c.shape[-1] == 1)
            xp.testing.assert_array_equal(a, c[:, 0])
        else:
            assert_(c.shape[0] == 1)
            xp.testing.assert_array_equal(a, c[0, :])


@pytest.mark.parametrize("xp", all_xp)
def test_diag_linop(xp):
    A = DiagonalOperator(xp.ones(3), **get_loc(xp))
    e = xp.eye(3)

    # test .full()
    Afull = A.full()
    xp.testing.assert_array_equal(Afull, e)

    # test scalar multiplication
    A5 = 5 * A
    A5full = A5.full()
    xp.testing.assert_array_equal(A5full, 5 * e)

    # scipy compatibility
    xp.testing.assert_array_equal(A * e, A.matvec(e))
    xp.testing.assert_array_equal(A.H * e, A.rmatvec(e))


@pytest.mark.parametrize(
    "xp, in_dtype, op_dtype",
    product(
        all_xp,
        [np.float32, np.float64, np.complex64, np.complex128],
        [np.complex64, np.complex128],
    ),
)
def test_idiag_linop(xp, in_dtype, op_dtype):
    N = 32
    rstate = xp.random.RandomState(5)

    # test various input and operator dtype combinations
    r = rstate.randn(N)
    if in_dtype in [xp.complex64, xp.complex128]:
        r = r + 1j * rstate.randn(N)
    r = xp.asarray(r, dtype=in_dtype)
    dtype_out = xp.result_type(r.dtype, op_dtype)
    Ai = IDiagOperator(N, dtype=op_dtype, **get_loc(xp))
    ri = Ai * r
    xp.testing.assert_array_equal(ri.real, xp.zeros_like(ri.real))
    xp.testing.assert_array_equal(ri.imag, r.imag)
    assert_equal(ri.dtype, dtype_out)
    assert_(ri is not r)  # r itself was not modified in-place

    # 2d input -> 1d output if shape not provided
    r2d = xp.random.randn(N, N)
    Ai = IDiagOperator(r2d.size, **get_loc(xp))
    ri = Ai * r2d
    assert_equal(ri.shape, (r2d.size,))

    # 2d input -> 2d output when shape is provided
    Ai2 = IDiagOperator(r2d.size, shape=r2d.shape, **get_loc(xp))
    ri = Ai2 * r2d
    assert_equal(ri.shape, r2d.shape)
    xp.testing.assert_array_equal(ri.imag, r2d.imag)

    # 1d input -> 2d output when shape is provided
    Ai2 = RDiagOperator(r2d.size, shape=r2d.shape, **get_loc(xp))
    ri = Ai2 * r2d.ravel(order=Ai2.order)
    assert_equal(ri.shape, r2d.shape)
    xp.testing.assert_array_equal(ri.imag, r2d.imag)


@pytest.mark.parametrize(
    "xp, in_dtype, op_dtype",
    product(
        all_xp,
        [np.float32, np.float64, np.complex64, np.complex128],
        [np.complex64, np.complex128],
    ),
)
def test_rdiag_linop(xp, in_dtype, op_dtype):
    N = 32
    rstate = xp.random.RandomState(5)

    # test various input and operator dtype combinations
    r = rstate.randn(N)
    if in_dtype in [xp.complex64, xp.complex128]:
        r = r + 1j * rstate.randn(N)
    r = xp.asarray(r, dtype=in_dtype)
    dtype_out = xp.result_type(r.dtype, op_dtype)
    Ar = RDiagOperator(N, dtype=op_dtype, **get_loc(xp))
    rr = Ar * r
    xp.testing.assert_array_equal(rr.imag, xp.zeros_like(rr.real))
    xp.testing.assert_array_equal(rr.real, r.real)
    assert_equal(rr.dtype, dtype_out)
    assert_(rr is not r)  # r itself was not modified in-place

    # 2d input -> 1d output if shape is not provided
    r2d = xp.random.randn(N, N)
    Ar = RDiagOperator(r2d.size)
    rr = Ar * r2d
    assert_equal(rr.shape, (r2d.size,))

    # 2d input -> 2d output when shape is provided
    Ar2 = RDiagOperator(r2d.size, shape=r2d.shape, **get_loc(xp))
    rr = Ar2 * r2d
    assert_equal(rr.shape, r2d.shape)
    xp.testing.assert_array_equal(rr.real, r2d.real)

    # 1d input -> 2d output when shape is provided
    Ar2 = RDiagOperator(r2d.size, shape=r2d.shape, **get_loc(xp))
    rr = Ar2 * r2d.ravel(order=Ar2.order)
    assert_equal(rr.shape, r2d.shape)
    xp.testing.assert_array_equal(rr.real, r2d.real)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_zero_linop(xp, order):
    nargin = 3
    nargout = 4
    A = ZeroOperator(nargin, nargout, order=order, **get_loc(xp))

    # test .full()
    Afull = A.full()
    xp.testing.assert_array_equal(Afull, xp.zeros((nargout, nargin)))

    e = xp.ones(nargin)
    xp.testing.assert_array_equal(A * e, xp.zeros(nargout))

    # scipy compatibility
    xp.testing.assert_array_equal(A * e, A.matvec(e))
    eH = xp.ones(nargout)
    xp.testing.assert_array_equal(A.H * eH, A.rmatvec(eH))


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_linop_with_mask(xp, order):
    a = xp.ones((100, 100))
    mask = xp.zeros(a.shape, dtype=xp.bool)
    mask[20:-20, 20:-20] = True
    nmask = mask.sum()
    Z = ZeroOperator(
        mask_in=mask,  # TODO: should be able to infer nargin, nargout from mask_in/mask_out
        nargin=nmask,
        nargout=a.size,
        squeeze_reps=True,
        order=order,
        **get_loc(xp),
    )
    az = Z * a[mask]
    assert_(xp.all(az == 0))
    assert_(az.size == a.size)
    assert_(Z.shape[1] == nmask)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_masking_operator(xp, order):
    a = xp.ones(100)
    mask = xp.ones(a.shape, dtype=xp.bool)
    mask[20:40] = 0
    # mask_out only
    I = MaskingOperator(
        mask_out=mask,
        nargin=a.size,
        squeeze_reps=True,
        order=order,
        **get_loc(xp),
    )
    c = I * a
    xp.testing.assert_array_equal(masker(a, mask, order=order), c)

    r = I.H * c
    xp.testing.assert_array_equal(a * mask, r)

    # mask_in only
    I = MaskingOperator(
        mask_in=mask,
        nargout=a.size,
        squeeze_reps=True,
        order=order,
        **get_loc(xp),
    )
    a_masked = masker(a, mask, order=order)
    c = I * a_masked
    xp.testing.assert_array_equal(a * mask, c)

    r = I.H * c
    xp.testing.assert_array_equal(a_masked, r)

    # mask neither
    I = MaskingOperator(
        nargin=a.size, squeeze_reps=True, order=order, **get_loc(xp)
    )
    c = I * a
    xp.testing.assert_array_equal(a, c)

    r = I.H * c
    xp.testing.assert_array_equal(a, r)

    # mask both
    mask_out = xp.ones(a.shape, dtype=xp.bool)
    mask_out[60:80] = 0
    # Note: would be identity if same mask used for input & output
    I = MaskingOperator(
        mask_in=mask,
        mask_out=mask_out,
        squeeze_reps=True,
        order=order,
        **get_loc(xp),
    )
    a_masked = masker(a, mask, order=order)
    c = I * a_masked
    xp.testing.assert_array_equal(masker(a * mask, mask_out, order=order), c)

    r = I.H * c
    xp.testing.assert_array_equal(masker(a * mask_out, mask, order=order), r)

    # scipy compatibility
    xp.testing.assert_array_equal(c, I.matvec(a_masked))
    xp.testing.assert_array_equal(r, I.rmatvec(c))
