import itertools

import pytest
import numpy as np
from numpy import fft
from numpy.testing import (
    run_module_suite,
    assert_allclose,
    assert_equal,
    assert_array_equal,
)

from scipy.fftpack import dct, idct

from mrrt.operators import DCT_Operator
from mrrt.operators._block_transforms import block_dctn, block_idctn


pytest.importorskip("skimage")

rstate = np.random.RandomState(1234)
c = rstate.randn(16, 16)


def test_DCT():
    rtol = 1e-4
    atol = 1e-4

    for shift, nd_in, nd_out, order, ax in itertools.product(
        [True, False], [True, False], [True, False], ["C", "F"], [0, 1]
    ):

        dct_axis = ax
        dct_axes = (ax,)
        DCTop = DCT_Operator(
            c.shape,
            order=order,
            dct_axes=dct_axes,
            use_FFT_shifts=shift,
            nd_input=nd_in,
            nd_output=nd_out,
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

        tmp = tmp.reshape(c.shape, order=order)
        if shift:
            scipy_tmp = fft.fftshift(
                dct(fft.ifftshift(c, axes=dct_axes), axis=dct_axis),
                axes=dct_axes,
            )
        else:
            scipy_tmp = dct(c, axis=dct_axis)

        assert_allclose(tmp, scipy_tmp, rtol=rtol, atol=atol)

        """
        test adjoint transform
        """
        tmp2 = DCTop.H * scipy_tmp
        if nd_in:
            assert tmp2.shape == c.shape
        else:
            assert tmp2.ndim == 1
        assert tmp2.real.dtype == c.dtype

        tmp2 = tmp2.reshape(c.shape, order=order)
        if shift:
            scipy_tmp2 = fft.fftshift(
                idct(fft.ifftshift(scipy_tmp, axes=dct_axes), axis=dct_axis),
                axes=dct_axes,
            )
        else:
            scipy_tmp2 = idct(scipy_tmp, axis=dct_axis)
        assert_allclose(tmp2, scipy_tmp2, rtol=rtol, atol=atol)


def test_DCT_all_axes():
    rtol = 1e-4
    atol = 1e-4
    for shift, nd_in, nd_out, order in itertools.product(
        [True, False], [True, False], [True, False], ["C", "F"]
    ):

        DCTop = DCT_Operator(
            c.shape,
            order=order,
            dct_axes=None,
            use_FFT_shifts=shift,
            nd_input=nd_in,
            nd_output=nd_out,
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

        tmp = tmp.reshape(c.shape, order=order)
        if shift:
            scipy_tmp = fft.fftshift(dct(dct(fft.ifftshift(c), axis=0), axis=1))
        else:
            scipy_tmp = dct(dct(c, axis=0), axis=1)

        assert_allclose(tmp, scipy_tmp, rtol=rtol, atol=atol)

        """
        test adjoint transform
        """
        tmp2 = DCTop.H * scipy_tmp
        if nd_in:
            assert tmp2.shape == c.shape
        else:
            assert tmp2.ndim == 1
        assert tmp2.real.dtype == c.dtype

        tmp2 = tmp2.reshape(c.shape, order=order)
        if shift:
            scipy_tmp2 = fft.fftshift(
                idct(idct(fft.ifftshift(scipy_tmp), axis=0), axis=1)
            )
        else:
            scipy_tmp2 = idct(idct(scipy_tmp, axis=0), axis=1)
        assert_allclose(tmp2, scipy_tmp2, rtol=rtol, atol=atol)


def test_DCT_2reps():
    """ multiple input case. """
    for shift, nd_in, nd_out, order in itertools.product(
        [True, False], [True, False], [True, False], ["C", "F"]
    ):
        DCTop = DCT_Operator(
            c.shape,
            order=order,
            use_FFT_shifts=shift,
            dct_axes=None,
            nd_input=nd_in,
            nd_output=nd_out,
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
        assert_array_equal(DCTop * c2, DCTop.matvec(c2))
        assert_array_equal(DCTop.H * tmp2, DCTop.rmatvec(tmp2))


def test_block_dctn():
    rstate = np.random.RandomState(1234)
    x = rstate.randn(64, 32)
    for block_size in [(2, 2), (8, 8), (16, 16), (1, 8), (8, 1), (1, 1)]:
        for reshape_array in [True, False]:
            out = block_dctn(x, (8, 8), reshape_output=reshape_array)
            if reshape_array:
                assert_equal(out.ndim, x.ndim)
            else:
                assert_equal(out.ndim, 2 * x.ndim)
            out2 = block_idctn(out, (8, 8), reshape_input=reshape_array)
            assert_allclose(x, out2)


if __name__ == "__main__":
    run_module_suite()
