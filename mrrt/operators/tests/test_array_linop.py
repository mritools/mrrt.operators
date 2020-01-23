from itertools import product

import numpy as np
from numpy.testing import assert_
import scipy.sparse
import pytest

from mrrt.operators.LinOp import ArrayOp
from mrrt.utils import config

all_xp = [np]
if config.have_cupy:
    import cupy
    import cupyx.scipy.sparse

    if cupy.cuda.runtime.getDeviceCount() > 0:
        all_xp += [cupy]


# if False:
#    import sys
#    import gc
#
#    n = 5
#    m = 10
#    A = np.random.randn(n, n) + 1j*np.random.randn(n, n)
#    sys.getrefcount(A)
#    Aop = ArrayOp(A, order='F')
#    referrers = gc.get_referrers(A)


# def memtest1(n=20000):
#    A = np.ones((n, n))
#
# def memtest2(n=20000):
#    A = np.ones((n, n))
#    Aop = ArrayOp(A, order='F')
#
# def memtest3(n=20000):
#    A = np.ones((n, n))
#    Aop = array_to_linop(A, order='F')


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_array_linop(xp, order):
    n = 5
    m = 10
    A = xp.random.randn(n, n) + 1j * xp.random.randn(n, n)
    b = xp.random.randn(n, m)
    bcplx = b + 1j * xp.random.randn(n, m)

    if xp is np:
        loc_kwargs = dict(loc_in="cpu", loc_out="cpu")
    else:
        loc_kwargs = dict(loc_in="gpu", loc_out="gpu")

    # ravel the array b to test reshaping
    b_col = b.ravel(order=order)
    bcplx_col = bcplx.ravel(order=order)

    if order == "C":
        with pytest.raises(ValueError):
            Aop = ArrayOp(A, order=order, **loc_kwargs)
        return
    # test 2D array operator
    Aop = ArrayOp(A, order=order, **loc_kwargs)
    assert_(Aop.Aref() is A)
    xp.testing.assert_array_equal(xp.dot(A, b), Aop * b_col)
    xp.testing.assert_array_equal(xp.dot(A.T, b), Aop.T * b_col)
    xp.testing.assert_array_equal(xp.dot(xp.conj(A).T, b), Aop.H * b_col)

    # test matrix operator
    if xp is np:
        Amat = xp.matrix(A)
        Aop = ArrayOp(Amat, order=order, **loc_kwargs)
        assert_(Aop.Aref() is Amat)
        xp.testing.assert_array_equal(Amat * b, Aop * b_col)
        xp.testing.assert_array_equal(Amat.T * b, Aop.T * b_col)
        xp.testing.assert_array_equal(Amat.H * b, Aop.H * b_col)
    else:
        pass  # no matrix class in CuPy

    # test sparse matrix operator

    if xp is np:
        Asparse = scipy.sparse.coo_matrix(A)
    else:
        Asparse = cupyx.scipy.sparse.coo_matrix((A.ravel(), xp.where(A)))
    Aop = ArrayOp(Asparse, order=order, **loc_kwargs)
    assert_(Aop.Aref() is Asparse)
    xp.testing.assert_array_equal(Asparse * b, Aop * b_col)
    xp.testing.assert_array_equal(Asparse.T * b, Aop.T * b_col)
    xp.testing.assert_array_equal(Asparse.H * b, Aop.H * b_col)

    # test consistent composite operator behavior
    xp.testing.assert_array_equal((Aop.H * Aop) * b_col, Aop.H * Aop * b_col)
    xp.testing.assert_array_equal((Aop.H * Aop) * b_col, Aop.H * (Aop * b_col))

    # Note:  have to have paranthesis in Asparse * b below to get the
    # equivalent result from scipy.sparse. otherwise the matrices are
    # multiplied first!
    xp.testing.assert_array_equal(
        Asparse.H * (Asparse * b), Aop.H * Aop * b_col
    )

    # check real array, complex input case
    Areal = A.real
    Aop = ArrayOp(Areal, order=order, **loc_kwargs)
    assert_(Aop.Aref() is Areal)
    xp.testing.assert_array_equal(xp.dot(Areal, bcplx), Aop * bcplx_col)
    xp.testing.assert_array_equal(xp.dot(Areal.T, bcplx), Aop.T * bcplx_col)
    xp.testing.assert_array_equal(
        xp.dot(xp.conj(Areal).T, bcplx), Aop.H * bcplx_col
    )
