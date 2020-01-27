from itertools import product
import numpy as np


from numpy.testing import assert_, assert_equal
import pytest

from mrrt.utils import config, ellipse_im, ImageGeometry, masker, embed
from mrrt.operators._TV import TV_Operator


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


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_TV_prior_input(xp, order):
    r = xp.random.randn(16, 16)
    TVop = TV_Operator(
        r.shape, arr_dtype=r.dtype, prior=r, order=order, **get_loc(xp)
    )
    g = TVop * r

    # because prior is equal to r, the gradient will be zero
    # norm and magnitude should also both be zero in this case
    if xp is np:
        assert_equal(g.sum(), 0.0)
        assert_equal(TVop.opnorm(r), 0.0)
        assert_equal(TVop.magnitude(r), 0.0)
        assert_equal(TVop.gradient(r), 0.0)
    else:
        assert_equal(g.sum().get(), 0.0)
        assert_equal(TVop.opnorm(r).get(), 0.0)
        assert_equal(TVop.magnitude(r).get(), 0.0)
        assert_equal(TVop.gradient(r).get(), 0.0)

    # divergence will be zero
    d = TVop.T * g
    if xp is np:
        assert_equal(d.sum(), 0.0)  # ? desired result?
    else:
        assert_equal(d.sum().get(), 0.0)

    # scipy compatibility
    xp.testing.assert_array_equal(
        TVop.H * (TVop * r), TVop.rmatvec(TVop.matvec(r))
    )


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_TV_multiple_inputs(xp, order):
    nreps = 4
    if order == "F":
        r = xp.random.randn(16, 16, nreps)
        tv_shape = r.shape[:-1]
    else:
        r = xp.random.randn(nreps, 16, 16)
        tv_shape = r.shape[1:]
    TVop = TV_Operator(tv_shape, arr_dtype=r.dtype, order=order, **get_loc(xp))
    g = TVop * r
    if order == "F":
        assert_equal(g.shape, (16, 16, 2, nreps))
    else:
        assert_equal(g.shape, (nreps, 2, 16, 16))
    d = TVop.T * g
    assert_equal(d.shape, r.shape)

    # scipy compatibility
    xp.testing.assert_array_equal(
        TVop.H * (TVop * r), TVop.rmatvec(TVop.matvec(r))
    )


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_TV_gradient_complex(xp, order):
    e = ellipse_im(
        ImageGeometry(shape=(256, 256), fov=256), params="shepplogan-mod"
    )[0]
    ec = e + 1j * e.T
    ec = xp.asarray(ec)

    T2 = TV_Operator(e.shape, arr_dtype=ec.dtype, order=order, **get_loc(xp))
    grad_complex = T2.gradient(ec)
    xp.testing.assert_array_almost_equal(grad_complex.real, grad_complex.imag.T)


@pytest.mark.parametrize("xp", all_xp)
def test_TV_masked(xp):
    # generate a random image-domain mask
    # im_mask = xp.random.standard_normal(c.shape) > -0.2

    # test 1D in/outputs but using masked values for the input array
    #  r = pywt.data.camera().astype(float)
    r = xp.random.randn(16, 16)
    x, y = xp.meshgrid(
        xp.arange(-r.shape[0] // 2, r.shape[0] // 2),
        xp.arange(-r.shape[1] // 2, r.shape[1] // 2),
        indexing="ij",
        sparse=True,
    )

    # make a circular mask
    im_mask = xp.sqrt(x * x + y * y) < r.shape[0] // 2
    # TODO: order='C' case is currently broken
    for order, mask_out in product(["F"], [None, im_mask]):
        r_masked = masker(r, im_mask, order=order)
        Wm = TV_Operator(
            r.shape,
            order=order,
            mask_in=im_mask,
            mask_out=mask_out,
            nd_input=True,
            nd_output=False,
            random_shift=True,
            **get_loc(xp),
        )
        out = Wm * r_masked

        if mask_out is None:
            assert_(out.ndim == 1)
            assert_(out.size == r.ndim * r.size)
            out = out.reshape(r.shape + (r.ndim,), order=order)
        else:
            assert_(out.ndim == 1)
            nmask = mask_out.sum()
            if xp != np:
                nmask = nmask.get()
            assert_(out.size == Wm.ndim * nmask)
            out = embed(out, mask_out, order=order)

        Wm.H * (Wm * r_masked)
