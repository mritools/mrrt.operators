from itertools import product

import numpy as np
import pytest

from mrrt.utils import config
from mrrt.operators import CompositeLinOp

OrthoMatrixOperator = pytest.importorskip("mrrt.operators.OrthoMatrixOperator")
dct_matrix = pytest.importorskip("mrrt.operators._OrthoMatrix.dct_matrix")

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


rstate = np.random.RandomState(1234)
c = rstate.randn(32, 32, 3)


@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order, axis",
    product(all_xp, [True, False], [True, False], ["C", "F"], [0, 1, -1]),
)
def test_OrthoMatrixOperator(xp, nd_in, nd_out, order, axis):
    rtol = 1e-4
    atol = 1e-4

    DCTop = OrthoMatrixOperator(
        c.shape,
        order=order,
        axis=axis,
        m=dct_matrix(c.shape[axis], xp=xp),
        nd_input=nd_in,
        nd_output=nd_out,
        **get_loc(xp),
    )

    """
    test forward transform
    """
    tmp = DCTop * c
    if nd_out:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    """
    test adjoint transform
    """
    tmp2 = DCTop.H * tmp
    if nd_in:
        assert tmp2.shape == c.shape
    else:
        assert tmp2.ndim == 1
    assert tmp2.real.dtype == c.dtype

    tmp2 = tmp2.reshape(c.shape, order=order)
    xp.testing.assert_allclose(tmp2, c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order",
    product(all_xp, [True, False], [True, False], ["C", "F"]),
)
def test_OrthoMatrixOperator_2reps(xp, nd_in, nd_out, order):
    """ multiple input case. """
    DCTop = OrthoMatrixOperator(
        c.shape,
        order=order,
        axis=-1,
        m=dct_matrix(c.shape[-1]),  # np.ndarray
        nd_input=nd_in,
        nd_output=nd_out,
        **get_loc(xp),
    )
    """
    test forward transform with 2 repetitions
    """
    nreps = 2
    if order == "F":
        c2 = np.stack((c,) * nreps, axis=-1)
    else:
        c2 = np.stack((c,) * nreps, axis=0)
    tmp = DCTop * c2
    if nd_out:
        if order == "F":
            assert tmp.shape == c.shape + (nreps,)
        else:
            assert tmp.shape == (nreps,) + c.shape
    else:
        assert tmp.ndim == 2
        if order == "F":
            assert tmp.shape[-1] == nreps
        else:
            assert tmp.shape[0] == nreps
    """
    test adjoint transform with 2 repetitions
    """
    tmp2 = DCTop.H * tmp
    if nd_in:
        if order == "F":
            assert tmp2.shape == c.shape + (nreps,)
        else:
            assert tmp2.shape == (nreps,) + c.shape
    else:
        assert tmp2.ndim == 2

    # scipy compatibility
    xp.testing.assert_array_equal(DCTop * c2, DCTop.matvec(c2))
    xp.testing.assert_array_equal(DCTop.H * tmp2, DCTop.rmatvec(tmp2))


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_composite_ortho(xp, order):
    rtol = 1e-4
    atol = 1e-4

    MDWT_Operator = pytest.importorskip("mrrt.operators.MDWT_Operator")

    """ multiple input case. """
    DCTop = OrthoMatrixOperator(
        c.shape,
        order=order,
        axis=-1,
        m="dct",
        nd_input=False,
        nd_output=False,
        **get_loc(xp),
    )
    DWTop = MDWT_Operator(
        arr_shape=c.shape,
        axes=(0, 1),
        order=order,
        nd_input=False,
        nd_output=False,
        level=2,
        filterbank="sym4",
        mode="periodization",
        **get_loc(xp),
    )

    Op = CompositeLinOp([DWTop, DCTop])

    """
    test forward transform
    """
    tmp = Op * xp.asarray(c)
    assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    """
    test adjoint transform
    """
    tmp2 = Op.H * tmp
    assert tmp2.ndim == 1
    assert tmp2.real.dtype == c.dtype

    tmp2 = tmp2.reshape(c.shape, order=order)
    xp.testing.assert_allclose(tmp2, c, rtol=rtol, atol=atol)
