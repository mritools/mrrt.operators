from itertools import product

import numpy as np
from numpy.testing import assert_raises, assert_, assert_equal
import pytest

from mrrt.utils import config, masker, embed

all_xp = [np]
if config.have_cupy:
    import cupy

    if cupy.cuda.runtime.getDeviceCount() > 0:
        all_xp += [cupy]

MDWT_Operator = pytest.importorskip("mrrt.operators.MDWT_Operator")
filters = pytest.importorskip("pyframelets.separable.filters")


def get_loc(xp):
    """Location arguments corresponding to numpy or CuPy case."""
    if xp is np:
        return dict(loc_in="cpu", loc_out="cpu")
    else:
        return dict(loc_in="gpu", loc_out="gpu")


@pytest.mark.parametrize(
    "xp, order, nd_in, nd_out",
    product(all_xp, ["C", "F"], [True, False], [False, True]),
)
def test_DWT(xp, order, nd_in, nd_out):
    rstate = xp.random.RandomState(5)
    c = rstate.randn(128, 128)
    dec_level = 3
    # filterbank = sep.pywt_as_filterbank('db2', xp=xp)
    filterbank = filters.mlt_filterbank(3, xp=xp)
    dwt_mode = "periodization"
    rtol = atol = 1e-7
    kwargs = dict(
        arr_shape=c.shape,
        order=order,
        nd_input=nd_in,
        nd_output=nd_out,
        level=dec_level,
        filterbank=filterbank,
        mode=dwt_mode,
        **get_loc(xp),
    )
    if nd_out:
        assert_raises(ValueError, MDWT_Operator, **kwargs)
        return

    Phi = MDWT_Operator(**kwargs)
    """
    test forward transform
    """
    tmp = Phi * c
    assert_(tmp.ndim == 1)
    assert_(tmp.real.dtype == c.dtype)

    """
    test adjoint transform
    """
    r = Phi.H * tmp
    if nd_in:
        assert_(r.shape == c.shape)
    else:
        assert_(r.ndim == 1)
    assert_(r.real.dtype == c.dtype)

    r = r.reshape(c.shape, order=order)

    xp.testing.assert_allclose(r, c, rtol=rtol, atol=atol)


# def test_DWT_axes_subsets():
# TODO:  implement ability to only transform a subset of the axes


@pytest.mark.parametrize(
    "xp, order, nd_in, decimation",
    product(all_xp, ["C", "F"], [True, False], [1, 2]),
)
def test_DWT_2reps(xp, order, nd_in, decimation):
    """ multiple input case. """
    rstate = xp.random.RandomState(5)
    c = rstate.randn(128, 128)
    dec_level = 3
    filterbank = filters.pywt_as_filterbank("db2", xp=xp, decimation=decimation)
    dwt_mode = "periodization"
    rtol = atol = 1e-7
    Phi = MDWT_Operator(
        c.shape,
        order=order,
        nd_input=nd_in,
        nd_output=False,
        level=dec_level,
        filterbank=filterbank,
        mode=dwt_mode,
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
    tmp = Phi * c2
    assert_(tmp.ndim == 2)
    if order == "F":
        assert_(tmp.shape[-1] == nreps)
    else:
        assert_(tmp.shape[0] == nreps)
    """
    test adjoint transform with 2 repetitions
    """
    tmp2 = Phi.H * tmp
    if nd_in:
        if order == "F":
            assert_(tmp2.shape == c.shape + (nreps,))
        else:
            assert_(tmp2.shape == (nreps,) + c.shape)
    else:
        assert_(tmp2.ndim == 2)

    if order == "F":
        tmp2 = tmp2.reshape(c.shape + (nreps,), order=order)
        # check accuracy of round-trip transform
        xp.testing.assert_allclose(tmp2[..., 0], c, rtol=rtol, atol=atol)
        xp.testing.assert_allclose(tmp2[..., -1], c, rtol=rtol, atol=atol)
    else:
        tmp2 = tmp2.reshape((nreps,) + c.shape, order=order)
        # check accuracy of round-trip transform
        xp.testing.assert_allclose(tmp2[0, ...], c, rtol=rtol, atol=atol)
        xp.testing.assert_allclose(tmp2[-1, ...], c, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "xp, order, nd_in, nd_out",
    product(
        all_xp,
        ["C", "F"],
        [True, False],
        [False],  # only nd_out == False is support
    ),
)
def test_DWT_axes_subset(xp, order, nd_in, nd_out):
    """ multiple input case. """
    rstate = xp.random.RandomState(5)
    c = rstate.randn(128, 128)
    dec_level = 3
    filterbank = filters.pywt_as_filterbank("db2", xp=xp)
    dwt_mode = "periodization"

    if order == "F":
        c3 = xp.stack([c] * 4, axis=-1)
    else:
        c3 = xp.stack([c] * 4, axis=0)
    atol = rtol = 1e-7
    Phi = MDWT_Operator(
        c3.shape,
        axes=(0, 1),  # subset of axes
        order=order,
        nd_input=nd_in,
        nd_output=False,
        level=dec_level,
        filterbank=filterbank,
        mode=dwt_mode,
        **get_loc(xp),
    )

    tmp = Phi * c3
    assert_(Phi.ndim_transform == 2)
    assert_(tmp.ndim == 1)
    """
    test adjoint transform with 2 repetitions
    """
    tmp2 = Phi.H * tmp
    if nd_in:
        assert_(tmp2.shape == c3.shape)
    else:
        assert_(tmp2.ndim == 1)
    tmp2 = tmp2.reshape(c3.shape, order=Phi.order)
    xp.testing.assert_allclose(tmp2, c3, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "xp, order, decimation", product(all_xp, ["C", "F"], [1, 2])
)
def test_DWT_prior(xp, order, decimation):
    rstate = xp.random.RandomState(5)
    c = rstate.randn(128, 128)
    dec_level = 3
    filterbank = filters.pywt_as_filterbank("db2", xp=xp, decimation=decimation)
    dwt_mode = "periodization"
    Phi = MDWT_Operator(
        c.shape,
        order=order,
        level=dec_level,
        filterbank=filterbank,
        mode=dwt_mode,
        prior=c,
        **get_loc(xp),
    )
    coeffs = Phi * c
    # when prior is the same as the image, DWT coeffs should all be zero
    if xp is np:
        assert_equal(coeffs.sum(), 0.0)
    else:
        assert_equal(coeffs.sum().get(), 0.0)

    # adjoint transform should add back the prior
    c2 = Phi.H * coeffs
    c2 = c2.reshape(c.shape, order=Phi.order)
    xp.testing.assert_allclose(c2, c)

    # scipy compatibility
    xp.testing.assert_array_equal(Phi * c, Phi.matvec(c))
    xp.testing.assert_array_equal(Phi.H * coeffs, Phi.rmatvec(coeffs))


@pytest.mark.parametrize(
    "xp, order, decimation", product(all_xp, ["C", "F"], [1, 2])
)
def test_DWT_masked(xp, order, decimation):
    rstate = xp.random.RandomState(5)
    c = rstate.randn(128, 128)
    # generate a random image-domain mask
    # im_mask = xp.random.standard_normal(c.shape) > -0.2

    # test 1D in/outputs but using masked values for the input array
    im_mask = xp.ones(c.shape, dtype=xp.bool)
    im_mask[16:64, 16:64] = 0  # mask out a region
    c_masked = masker(c, im_mask, order=order)
    fb = filters.pywt_as_filterbank("db2", xp=xp, decimation=decimation)
    Wm = MDWT_Operator(
        c.shape,
        level=2,
        filterbank=fb,
        order=order,
        mode="periodization",
        mask_in=im_mask,
        mask_out=None,
        nd_input=False,
        nd_output=False,
        random_shift=True,
        **get_loc(xp),
    )
    coeffs = Wm * c_masked
    assert_(coeffs.ndim == 1)
    if decimation == 2:
        assert_(coeffs.size == c.size)
        coeffs = coeffs.reshape(c.shape, order=order)
    else:
        assert_(coeffs.size > c.size)

    c_recon = Wm.H * coeffs
    assert_(c_recon.ndim == 1)
    if xp is np:
        assert_(c_recon.size == im_mask.sum())
    else:
        assert_(c_recon.size == im_mask.sum().get())
    c_recon = embed(c_recon, im_mask, order=order)
    xp.testing.assert_allclose(c_recon, c * im_mask, rtol=1e-9, atol=1e-9)

    # test 1D in/outputs but using masked values for both input and output
    # arrays
    coeffs_mask = coeffs != 0  # mask out regions of zero-valued coeffs
    Wm2 = MDWT_Operator(
        c.shape,
        level=2,
        filterbank=fb,
        order=order,
        mode="periodization",
        mask_in=im_mask,
        mask_out=coeffs_mask,
        nd_input=False,
        nd_output=False,
        random_shift=True,
        **get_loc(xp),
    )
    coeffs = Wm2 * c_masked
    assert_(coeffs.ndim == 1)
    if decimation == 2:
        if xp is np:
            assert_(coeffs.size == coeffs_mask.sum())
        else:
            assert_(coeffs.size == coeffs_mask.sum().get())
    c_recon = Wm2.H * coeffs
    c_recon = embed(c_recon, im_mask, order=order)
    xp.testing.assert_allclose(c_recon, c * im_mask, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    "xp, order, decimation", product(all_xp, ["C", "F"], [1, 2])
)
def test_DWT_autopad(xp, order, decimation):
    rstate = xp.random.RandomState(5)
    c = rstate.randn(128, 128)
    dec_level = 3
    filterbank = filters.pywt_as_filterbank("db2", xp=xp, decimation=decimation)
    dwt_mode = "periodization"
    rtol = atol = 1e-7

    # c is already divisible by 2**dec_level, so it won't get padded
    Phi = MDWT_Operator(
        c.shape,
        order=order,
        nd_input=False,
        nd_output=False,
        level=dec_level,
        filterbank=filterbank,
        mode=dwt_mode,
        autopad=True,
        **get_loc(xp),
    )
    assert_equal(Phi.pad_width, [(0, 0), (0, 0)])
    c_recon = xp.reshape(Phi.H * Phi * c, c.shape, order=order)
    xp.testing.assert_allclose(c, c_recon, rtol=rtol, atol=atol)

    """
    Now test a case where autopad will round up shape to match c.
    truncate by 5 (< 2**dec_level) so autopad will give a shape equal to c
    """
    c_trunc = c[:-5, :-5]
    Phi = MDWT_Operator(
        c_trunc.shape,
        order=order,
        nd_input=False,
        nd_output=False,
        level=dec_level,
        filterbank=filterbank,
        mode=dwt_mode,
        autopad=True,
        **get_loc(xp),
    )

    assert_equal(c_trunc.shape, Phi.arr_shape)
    assert_equal(c.shape, Phi.pad_shape)

    tmp = Phi * c_trunc
    if decimation == 2:
        assert_equal(tmp.size, c.size)
    else:
        assert_(tmp.size > c.size)

    c_trunc_recon = xp.reshape(Phi.H * tmp, c_trunc.shape, order=order)
    assert_equal(c_trunc_recon.shape, c_trunc.shape)
    xp.testing.assert_allclose(c_trunc, c_trunc_recon, rtol=rtol, atol=atol)
