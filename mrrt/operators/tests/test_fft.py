import os
from itertools import product

import numpy as np
import pytest

from mrrt.operators import FFT_Operator, LinearOperatorMulti
from mrrt.utils import config, masker, embed
from mrrt.utils import fftn, fftnc, ifftn, ifftnc


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


def get_data(xp, shape=(128, 127)):
    rstate = xp.random.RandomState(1234)
    # keep one dimension odd to make sure shifts work correctly
    c = rstate.randn(*shape)
    return c


if config.have_pyfftw:
    preplan_pyfftw_vals = [True, False]
else:
    preplan_pyfftw_vals = [False]

if "PYIR_GPUTEST" in os.environ:
    run_gpu_tests = True
else:
    run_gpu_tests = False
run_gpu_tests = False  # TODO: GPU variant not yet implemented

if run_gpu_tests:
    cuda_cases = [True, False]
else:
    cuda_cases = [False]


# @dec.slow
@pytest.mark.parametrize(
    "xp, shift, nd_in, nd_out, order, real_dtype",
    product(
        all_xp,
        [True, False],
        [True, False],
        [True, False],
        ["C", "F"],
        [np.float32, np.float64, np.complex64, np.complex128],
    ),
)
def test_fft_basic(xp, shift, nd_in, nd_out, order, real_dtype):
    rtol = 1e-4
    atol = 1e-4
    cimg = get_data(xp).astype(real_dtype)
    cplx_type = xp.result_type(cimg.dtype, np.complex64)

    FTop = FFT_Operator(
        cimg.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    """
    test forward transform
    """
    tmp = FTop * cimg
    if nd_out:
        assert tmp.shape == cimg.shape
    else:
        assert tmp.ndim == 1
    if xp.isrealobj(cimg):
        assert tmp.real.dtype == cimg.dtype
    else:
        assert tmp.dtype == cimg.dtype

    tmp = tmp.reshape(cimg.shape, order=order)
    if shift:
        numpy_tmp = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(cimg)))
    else:
        numpy_tmp = xp.fft.fftn(cimg)

    if numpy_tmp.dtype != cplx_type:
        numpy_tmp = numpy_tmp.astype(cplx_type)

    xp.testing.assert_allclose(tmp, numpy_tmp, rtol=rtol, atol=atol)

    """
    test adjoint transform
    """
    tmp2 = FTop.H * tmp
    if nd_in:
        assert tmp2.shape == cimg.shape
    else:
        assert tmp2.ndim == 1

    if xp.isrealobj(cimg):
        assert tmp2.real.dtype == cimg.dtype
    else:
        assert tmp2.dtype == cimg.dtype

    tmp2 = tmp2.reshape(cimg.shape, order=order)
    if shift:
        numpy_tmp2 = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(numpy_tmp)))
    else:
        numpy_tmp2 = xp.fft.ifftn(numpy_tmp)
    xp.testing.assert_allclose(tmp2, numpy_tmp2, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order",
    product(all_xp, [True, False], [True, False], ["C", "F"]),
)
def test_fft_roundtrips(xp, nd_in, nd_out, order):
    rtol = 1e-3
    atol = 1e-3

    c = get_data(xp)
    FTop = FFT_Operator(
        c.shape,
        order=order,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=False,
        **get_loc(xp),
    )

    """
    test round trip transform with paranthesis grouping
    """
    tmp = FTop.H * (FTop * c)

    xp.testing.assert_allclose(
        tmp.reshape(c.shape, order=order).real, c, rtol=rtol, atol=atol
    )

    """
    test round trip without paranthesis
    """
    tmp = FTop.H * FTop * c

    xp.testing.assert_allclose(
        tmp.reshape(c.shape, order=order).real, c, rtol=rtol, atol=atol
    )

    """
    test round trip with composite operator
    """
    tmp = (FTop.H * FTop) * c

    xp.testing.assert_allclose(
        tmp.reshape(c.shape, order=order).real, c, rtol=rtol, atol=atol
    )


# # @dec.slow
@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order, ortho, fft_axes, shift",
    product(
        all_xp,
        [True, False],
        [True, False],
        ["C", "F"],
        [True, False],
        [(0,), (1,), (0, 1), None],
        [True, False],
    ),
)
def test_fft_axes_subsets_and_ortho(
    xp, nd_in, nd_out, order, ortho, fft_axes, shift
):
    """ Test applying FFT only along particular axes. """
    rtol = 1e-3
    atol = 1e-3
    c = get_data(xp)
    FTop = FFT_Operator(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        fft_axes=fft_axes,
        ortho=ortho,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    """
    test forward transform
    """
    tmp = FTop * c
    if nd_out:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)
    if ortho:
        fftargs = dict(norm="ortho")
    else:
        fftargs = dict(norm=None)
    if shift:
        numpy_tmp = xp.fft.fftshift(
            xp.fft.fftn(
                xp.fft.ifftshift(c, axes=fft_axes), axes=fft_axes, **fftargs
            ),
            axes=fft_axes,
        )
    else:
        numpy_tmp = xp.fft.fftn(c, axes=fft_axes, **fftargs)
    xp.testing.assert_allclose(tmp, numpy_tmp, rtol=rtol, atol=atol)

    """
    test adjoint transform
    """
    tmp2 = FTop.H * numpy_tmp
    if nd_in:
        assert tmp2.shape == c.shape
    else:
        assert tmp2.ndim == 1
    assert tmp2.real.dtype == c.dtype

    tmp2 = tmp2.reshape(c.shape, order=order)
    if shift:
        numpy_tmp2 = xp.fft.fftshift(
            xp.fft.ifftn(
                xp.fft.ifftshift(numpy_tmp, axes=fft_axes),
                axes=fft_axes,
                **fftargs,
            ),
            axes=fft_axes,
        )
    else:
        numpy_tmp2 = xp.fft.ifftn(numpy_tmp, axes=fft_axes, **fftargs)
    xp.testing.assert_allclose(tmp2, numpy_tmp2, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order, shift",
    product(all_xp, [True, False], [True, False], ["C", "F"], [True, False]),
)
def test_fft_2reps(xp, nd_in, nd_out, order, shift):
    """ multiple input case. """
    c = get_data(xp)
    FTop = FFT_Operator(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
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
    tmp = FTop * c2

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
    tmp2 = FTop.H * tmp
    if nd_in:
        if order == "F":
            assert tmp2.shape == c.shape + (nreps,)
        else:
            assert tmp2.shape == (nreps,) + c.shape
    else:
        assert tmp2.ndim == 2

    # scipy compatibility
    xp.testing.assert_array_equal(tmp, FTop.matvec(c2))
    xp.testing.assert_array_equal(tmp2, FTop.rmatvec(tmp))


@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order, shift",
    product(all_xp, [True, False], [True, False], ["C", "F"], [True, False]),
)
def test_fft_composite(xp, nd_in, nd_out, order, shift):
    """Testing composite forward and adjoint operator."""
    rtol = 1e-4
    atol = 1e-4
    c = get_data(xp)

    FTop = FFT_Operator(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop
    """
    test forward transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)
    xp.testing.assert_allclose(tmp, c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, nd_in, order, shift",
    product(all_xp, [True, False], ["C", "F"], [True, False]),
)
def test_partial_FFT_allsamples(xp, nd_in, order, shift):
    rtol = 1e-4
    atol = 1e-4
    """ masked FFT without missing samples """
    # TODO: when all samples are present, can test against normal FFT

    nd_out = False
    c = get_data(xp)
    FTop = FFT_Operator(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=xp.ones(c.shape),
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )  # no missing samples

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop

    """
    test forward transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)
    xp.testing.assert_allclose(tmp, c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, nd_in, order, shift",
    product(all_xp, [True, False], ["C", "F"], [True, False]),
)
def test_partial_FFT(xp, nd_in, order, shift):
    """ masked FFT with missing samples """
    # TODO: check accuracy against brute force DFT
    nd_out = False
    c = get_data(xp)
    rstate = xp.random.RandomState(1234)
    sample_mask = rstate.rand(*(128, 127)) > 0.5
    FTop = FFT_Operator(
        c.shape,
        order=order,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=sample_mask,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop

    """
    test round trip transform
    """
    tmp = FtF * c
    if nd_in:
        assert tmp.shape == c.shape
    else:
        assert tmp.ndim == 1
    assert tmp.real.dtype == c.dtype

    tmp = tmp.reshape(c.shape, order=order)


@pytest.mark.parametrize(
    "xp, nd_in, order, shift",
    product(
        all_xp, [True, False], ["F"], [True, False],
    ),  # TODO: support order = 'C' as well?
)
def test_partial_FFT_with_im_mask(xp, nd_in, order, shift):
    """ masked FFT with missing samples and masked image domain """
    c = get_data(xp)
    rstate = xp.random.RandomState(1234)
    sample_mask = rstate.rand(*(128, 127)) > 0.5
    x, y = xp.meshgrid(
        xp.arange(-c.shape[0] // 2, c.shape[0] // 2),
        xp.arange(-c.shape[1] // 2, c.shape[1] // 2),
        indexing="ij",
        sparse=True,
    )

    # make a circular mask
    im_mask = xp.sqrt(x * x + y * y) < c.shape[0] // 2

    nd_out = False
    FTop = FFT_Operator(
        c.shape,
        order=order,
        im_mask=im_mask,
        use_fft_shifts=shift,
        nd_input=nd_in,
        nd_output=nd_out,
        sample_mask=sample_mask,
        gpu_force_reinit=False,
        mask_kspace_on_gpu=(not shift),
        **get_loc(xp),
    )

    # create new linear operator for forward followed by inverse transform
    FtF = FTop.H * FTop
    assert isinstance(FtF, LinearOperatorMulti)

    # test forward only
    forw = embed(
        FTop * masker(c, im_mask, order=order), sample_mask, order=order
    )

    if shift:
        expected_forw = sample_mask * fftnc(c * im_mask)
    else:
        expected_forw = sample_mask * fftn(c * im_mask)
    xp.testing.assert_allclose(forw, expected_forw, rtol=1e-7, atol=1e-4)

    # test roundtrip
    roundtrip = FTop.H * (FTop * masker(c, im_mask, order=order))
    if shift:
        expected_roundtrip = masker(
            ifftnc(sample_mask * fftnc(c * im_mask)), im_mask, order=order
        )
    else:
        expected_roundtrip = masker(
            ifftn(sample_mask * fftn(c * im_mask)), im_mask, order=order
        )

    xp.testing.assert_allclose(
        roundtrip, expected_roundtrip, rtol=1e-7, atol=1e-4
    )

    # test roundtrip with 2 reps
    c2 = xp.stack([c] * 2, axis=-1)
    roundtrip = FTop.H * (FTop * masker(c2, im_mask, order=order))
    if shift:
        expected_roundtrip = masker(
            ifftnc(
                sample_mask[..., xp.newaxis]
                * fftnc(c2 * im_mask[..., xp.newaxis], axes=(0, 1)),
                axes=(0, 1),
            ),
            im_mask,
            order=order,
        )
    else:
        expected_roundtrip = masker(
            ifftn(
                sample_mask[..., xp.newaxis]
                * fftn(c2 * im_mask[..., xp.newaxis], axes=(0, 1)),
                axes=(0, 1),
            ),
            im_mask,
            order=order,
        )
    xp.testing.assert_allclose(
        roundtrip, expected_roundtrip, rtol=1e-7, atol=1e-4
    )
