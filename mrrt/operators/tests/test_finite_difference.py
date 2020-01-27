from itertools import product

import numpy as np
from numpy.testing import (
    assert_,
    assert_raises,
    assert_equal,
    assert_almost_equal,
)
import pytest

from mrrt.utils import config, ellipse_im, ImageGeometry, masker, embed
from mrrt.operators import (
    TV_Operator,
    FiniteDifferenceOperator,
    DiagonalOperator,
    IdentityOperator,
    linop_shape_args,
)

from mrrt.operators._FiniteDifference import (
    gradient_periodic,
    divergence_periodic,
    forward_diff,
    backward_diff,
)


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


# TODO: replace use of ellipse_im


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_FiniteDifference_operator(xp, order):
    e = ellipse_im(
        ImageGeometry(shape=(256, 256), fov=256), params="shepplogan-mod"
    )[0]
    ec = e + 1j * e.T

    # 2D without corners should match the TV Operator in 'iso' mode
    TV = TV_Operator(
        e.shape, arr_dtype=ec.dtype, tv_type="iso", order=order, **get_loc(xp)
    )
    FD_nocorner = FiniteDifferenceOperator(
        e.shape,
        arr_dtype=ec.dtype,
        order=order,
        use_corners=False,
        **get_loc(xp),
    )
    xp.testing.assert_array_almost_equal(TV * ec, FD_nocorner * ec)

    # with corners included, should have 4 directions for 2D input
    FD = FiniteDifferenceOperator(
        ec.shape,
        arr_dtype=ec.dtype,
        order=order,
        use_corners=True,
        **get_loc(xp),
    )
    y = FD * e
    if order == "F":
        assert_(y.shape[-1] == 4)
    else:
        assert_(y.shape[0] == 4)

    # scipy compatibility
    xp.testing.assert_array_equal(FD.H * (FD * e), FD.rmatvec(FD.matvec(e)))


@pytest.mark.parametrize(
    "xp, nd_output, ndim", product(all_xp, [True, False], [1, 2, 3, 4])
)
def test_FiniteDifference_nd(xp, nd_output, ndim):
    e = xp.ones((8,) * ndim)

    # 2D without corners should match the TV Operator in 'iso' mode
    FD_nocorner = FiniteDifferenceOperator(
        e.shape, use_corners=False, nd_output=nd_output, **get_loc(xp)
    )
    y = FD_nocorner * e
    # last axis shape should equal the expected number of offsets
    if e.ndim == 1 and FD_nocorner.squeeze_reps:
        # assert_(y.ndim == 1)  # TODO
        pass
    else:
        if nd_output:
            assert_(y.shape[-1] == ndim)
        else:
            assert_(y.ndim == 1)
            assert_(y.shape[0] == ndim * e.size)

    # output should be all zeros for a constant amplitude input
    assert_(xp.all(y == 0))

    # with corners included, should have 4 directions for 2D input
    FD = FiniteDifferenceOperator(
        e.shape, use_corners=True, nd_output=nd_output, **get_loc(xp)
    )
    y = FD * e
    # last axis shape should equal the expected number of offsets
    if e.ndim == 1 and FD.squeeze_reps:
        # assert_(y.ndim == 1)  # TODO
        pass
    else:
        if nd_output:
            assert_(y.shape[-1] == (3 ** ndim - 1) // 2)
        else:
            assert_(y.ndim == 1)
            assert_(y.shape[0] == (3 ** ndim - 1) // 2 * e.size)
    # output should be all zeros for a constant amplitude input
    assert_(xp.all(y == 0))

    # check shape of first axis in output
    if FD.nd_output:
        assert_(y.shape[:-1] == e.shape)


@pytest.mark.parametrize(
    "xp, ndim, dtype",
    product(
        all_xp,
        [3],
        [np.float32, np.float64, np.complex64, np.complex128, np.int16],
    ),
)
def test_FiniteDifference_1axis_and_dtypes(xp, ndim, dtype):
    # verify proper operation with transform only along a single axis
    rstate = xp.random.RandomState(1234)
    e = rstate.standard_normal((8,) * ndim).astype(dtype)
    if xp.iscomplexobj(e):
        e = e + 1j * rstate.standard_normal(e.shape).astype(dtype)

    for ax in range(e.ndim):
        FD = FiniteDifferenceOperator(
            e.shape, axes=(ax,), nd_output=True, **get_loc(xp)
        )
        # verify result for forward operation
        d = xp.squeeze(FD * e)
        d2 = xp.roll(e, -1, axis=ax) - e
        xp.testing.assert_array_almost_equal(d, d2)

        # verify result for adjoint operation
        dHd = FD.H * d
        dHd2 = d2 - xp.roll(d2, 1, axis=ax)
        xp.testing.assert_array_almost_equal(dHd, -dHd2)

    # output is always at least float32
    assert_(FD.dtype == np.float32)
    assert_(d.dtype == np.result_type(dtype, FD.dtype))


@pytest.mark.parametrize("xp", all_xp)
def test_operator_combinations(xp):
    # verify proper operation with transform only along a single axis
    rstate = xp.random.RandomState(1234)
    ndim = 3
    dtype = np.complex64
    e = rstate.standard_normal((8,) * ndim).astype(dtype)
    if xp.iscomplexobj(e):
        e += 1j * rstate.standard_normal(e.shape).astype(dtype)

    FD0 = FiniteDifferenceOperator(
        e.shape, axes=(0,), nd_output=True, **get_loc(xp)
    )
    FD1 = FiniteDifferenceOperator(
        e.shape, axes=(1,), nd_output=True, **get_loc(xp)
    )
    FD01 = FiniteDifferenceOperator(
        e.shape, axes=(0, 1), nd_output=True, **get_loc(xp)
    )

    d0 = FD0 * e  # FD along first axis
    d01 = FD01 * e  # FD along both axes
    if False:
        #  TODO: cannot do this linear combination when using a subset of
        #  arrays
        # xp.testing.assert_array_almost_equal(d01, (FD0 + FD1)*e)
        pass
    else:
        xp.testing.assert_array_almost_equal(
            d01.sum(-1, keepdims=True), (FD0 + FD1) * e
        )
    xp.testing.assert_array_almost_equal(3 * d0, (2 * FD0 + FD0) * e)
    xp.testing.assert_array_almost_equal(3 * d0, (FD0 + FD0 * 2) * e)
    xp.testing.assert_array_almost_equal(FD0 * d0, (FD0 * FD0) * e)
    xp.testing.assert_array_almost_equal(xp.zeros_like(d0), (FD0 - FD0) * e)

    I = IdentityOperator(e.size, **linop_shape_args(FD0), **get_loc(xp))
    xp.testing.assert_array_almost_equal(d0 + e[..., np.newaxis], (FD0 + I) * e)

    D = DiagonalOperator(
        xp.arange(e.size, dtype=np.float32),
        **linop_shape_args(FD0),
        **get_loc(xp),
    )
    xp.testing.assert_array_almost_equal(D * d0, (D * FD0) * e)


@pytest.mark.parametrize(
    "xp, fd_axes",
    product(all_xp, [(0, 1), (1, 2), (2, 0), (3,), (2, 1, 0, 3), (-1,), None]),
)
def test_FiniteDifference_axes_subsets(xp, fd_axes):
    # verify proper operation with transform over various axis combinations
    rstate = xp.random.RandomState(1234)
    ndim = 4
    e = rstate.standard_normal((6,) * ndim)
    FD = FiniteDifferenceOperator(
        e.shape, axes=fd_axes, nd_output=True, **get_loc(xp)
    )
    if fd_axes is None:
        fd_axes = tuple(np.arange(ndim))
    else:
        # cannot enable use_corners with custom axes specified
        assert_raises(
            ValueError,
            FiniteDifferenceOperator,
            e.shape,
            axes=(0,),
            use_corners=True,
            **get_loc(xp),
        )

    # verify result for forward operation
    d = FD * e
    d2 = xp.zeros_like(d)
    for n, ax in enumerate(fd_axes):
        d2[..., n] = xp.roll(e, -1, axis=ax) - e
    xp.testing.assert_array_almost_equal(d, d2)

    # verify result for adjoint operation
    dHd = FD.H * d
    dHd2 = xp.zeros_like(d2)
    for n, ax in enumerate(fd_axes):
        dHd2[..., n] = d2[..., n] - xp.roll(d2[..., n], 1, axis=ax)
    dHd2 = dHd2.sum(-1)
    xp.testing.assert_array_almost_equal(dHd, -dHd2)

    # scipy compatibility
    xp.testing.assert_array_equal(d, FD.matvec(e))
    xp.testing.assert_array_equal(dHd, FD.rmatvec(d))

    # verify other concatenation parenthesis placements are equivalent
    xp.testing.assert_array_almost_equal(dHd, FD.H * FD * e)
    xp.testing.assert_array_almost_equal(dHd, (FD.H * FD) * e)


@pytest.mark.parametrize("xp, ndim", product(all_xp, [1, 2, 3, 4]))
def test_custom_offsets(xp, ndim):
    rstate = xp.random.RandomState(1234)
    e = rstate.standard_normal((4,) * ndim)

    # default operation along all axes
    FD1 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(np.arange(ndim)),
        use_corners=False,
        nd_output=True,
        **get_loc(xp),
    )

    # equivalent operation specified via custom_offsets instead
    custom_offsets = np.concatenate(([1], np.cumprod(e.shape)[:-1]))
    FD2 = FiniteDifferenceOperator(
        e.shape,
        custom_offsets=custom_offsets,
        use_corners=False,
        nd_output=True,
        **get_loc(xp),
    )

    d1 = FD1 * e
    d2 = FD2 * e
    # edge won't match because custom_offsets case doesn't have proper
    # periodic boundary conditions!
    xp.testing.assert_array_almost_equal(
        d1[tuple([slice(1, -1)] * ndim)], d2[tuple([slice(1, -1)] * ndim)]
    )

    d1H = FD1.H * d1
    d2H = FD2.H * d2
    # edge won't match because custom_offsets case doesn't have proper
    # periodic boundary conditions!
    xp.testing.assert_array_almost_equal(
        d1H[tuple([slice(1, -1)] * ndim)], d2H[tuple([slice(1, -1)] * ndim)]
    )


@pytest.mark.parametrize("xp", all_xp)
def test_grid_weights(xp):
    rstate = xp.random.RandomState(1234)
    e = rstate.standard_normal((8, 8))

    # default operation along all axes
    FD1 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(xp.arange(ndim)),
        use_corners=False,
        grid_size=None,
        nd_output=True,
        **get_loc(xp),
    )

    grid_size = (0.5, 2)  # different scaling along each axis
    FD2 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(xp.arange(ndim)),
        use_corners=False,
        grid_size=grid_size,
        nd_output=True,
        **get_loc(xp),
    )
    d1 = FD1 * e
    d2 = FD2 * e
    xp.testing.assert_array_almost_equal(d1[..., 0], d2[..., 0] * grid_size[0])
    xp.testing.assert_array_almost_equal(d1[..., 1], d2[..., 1] * grid_size[1])
    # check that adjoint runs and has the expected shape
    d1H = FD1.H * d1
    d2H = FD2.H * d2
    assert_(d1H.shape == d2H.shape)

    # case with same scaling on each axis to easily verify adjoint result
    grid_size = (2, 2)
    FD2 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(xp.arange(ndim)),
        use_corners=False,
        grid_size=grid_size,
        nd_output=True,
        **get_loc(xp),
    )
    d1 = FD1 * e
    d2 = FD2 * e
    xp.testing.assert_array_almost_equal(d1[..., 0], d2[..., 0] * grid_size[0])
    xp.testing.assert_array_almost_equal(d1[..., 1], d2[..., 1] * grid_size[1])
    # check that adjoint matches the expected values
    d1H = FD1.H * d1
    d2H = FD2.H * d2
    xp.testing.assert_array_almost_equal(d1H, d2H * grid_size[0] ** 2)

    # grid_size can be a sequence of arrays, each equal in shape to e
    grid_size = (
        rstate.standard_normal(e.shape),
        rstate.standard_normal(e.shape),
    )
    FD2 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(xp.arange(ndim)),
        use_corners=False,
        grid_size=grid_size,
        nd_output=True,
        **get_loc(xp),
    )
    d2 = FD2 * e
    xp.testing.assert_array_almost_equal(d1[..., 0], d2[..., 0] * grid_size[0])
    xp.testing.assert_array_almost_equal(d1[..., 1], d2[..., 1] * grid_size[1])
    # check that adjoint runs and has the expected shape
    d1H = FD1.H * d1
    d2H = FD2.H * d2
    assert_(d1H.shape == d2H.shape)

    # grid_size arrays can also use broadcasting for memory efficiency
    grid_size = (
        rstate.standard_normal((1, e.shape[1])),
        rstate.standard_normal((e.shape[0], 1)),
    )
    FD2 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(xp.arange(ndim)),
        use_corners=False,
        grid_size=grid_size,
        nd_output=True,
        **get_loc(xp),
    )
    d2 = FD2 * e
    xp.testing.assert_array_almost_equal(d1[..., 0], d2[..., 0] * grid_size[0])
    xp.testing.assert_array_almost_equal(d1[..., 1], d2[..., 1] * grid_size[1])
    # check that adjoint runs and has the expected shape
    d1H = FD1.H * d1
    d2H = FD2.H * d2
    assert_(d1H.shape == d2H.shape)

    # grid_size can be a mixture of integers and arrays
    grid_size = (3, rstate.standard_normal(e.shape))
    FD2 = FiniteDifferenceOperator(
        e.shape,
        axes=None,  # tuple(xp.arange(ndim)),
        use_corners=False,
        grid_size=grid_size,
        nd_output=True,
        **get_loc(xp),
    )
    d2 = FD2 * e
    xp.testing.assert_array_almost_equal(d1[..., 0], d2[..., 0] * grid_size[0])
    xp.testing.assert_array_almost_equal(d1[..., 1], d2[..., 1] * grid_size[1])
    # check that adjoint runs and has the expected shape
    d1H = FD1.H * d1
    d2H = FD2.H * d2
    assert_(d1H.shape == d2H.shape)


@pytest.mark.parametrize("xp, shape", product(all_xp, [(16, 16)]))
def test_FiniteDifference_masked(xp, shape):
    # test 1D in/outputs but using masked values for the input array
    #  r = pywt.data.camera().astype(float)
    rstate = xp.random.RandomState(1234)
    r = rstate.randn(*shape)
    x, y = xp.meshgrid(
        xp.arange(-r.shape[0] // 2, r.shape[0] // 2),
        xp.arange(-r.shape[1] // 2, r.shape[1] // 2),
        indexing="ij",
        sparse=True,
    )

    # make a circular mask
    im_mask = xp.sqrt(x ** 2 + y ** 2) < r.shape[0] // 2
    # TODO: order='C' case is currently broken
    for order, mask_out in product(["F"], [None, im_mask]):
        r_masked = masker(r, im_mask, order=order)
        Wm = FiniteDifferenceOperator(
            r.shape,
            order=order,
            use_corners=True,
            mask_in=im_mask,
            mask_out=mask_out,
            nd_input=True,
            nd_output=mask_out is not None,
            random_shift=True,
            **get_loc(xp),
        )
        out = Wm * r_masked

        if mask_out is None:
            assert_(out.ndim == 1)
            assert_(out.size == Wm.num_offsets * r.size)
            out = out.reshape(r.shape + (Wm.num_offsets,), order=order)
        else:
            assert_(out.ndim == 1)
            assert_(out.size == Wm.num_offsets * mask_out.sum())
            out = embed(out, mask_out, order=order)

        Wm.H * (Wm * r_masked)


@pytest.mark.parametrize("xp, dtype", product(all_xp, [np.float32, np.float64]))
def test_gradient_periodic_forw_back(xp, dtype):
    x = xp.arange(5).astype(dtype)
    g_forw_expected = xp.asarray([1, 1, 1, 1, -4])
    g_back_expected = g_forw_expected[::-1]
    g_forw = gradient_periodic(
        x, deltas=None, direction="forward", grad_axis="last"
    )
    g_back = gradient_periodic(
        x, deltas=[1], direction="backward", grad_axis="last"
    )
    assert_(g_forw.dtype == dtype)
    assert_equal(g_forw.ndim, x.ndim + 1)
    xp.testing.assert_array_almost_equal(xp.squeeze(g_forw), g_forw_expected)
    xp.testing.assert_array_almost_equal(xp.squeeze(g_back), g_back_expected)


@pytest.mark.parametrize(
    "xp, axis, ndim", product(all_xp, [0, -1], [1, 2, 3, 4])
)
def test_gradient_periodic_nd(xp, axis, ndim):
    """ image gradient """
    rstate = xp.random.RandomState(1234)
    r = rstate.randn(*([4] * ndim))
    g = gradient_periodic(r, deltas=None, direction="forward", grad_axis=axis)
    assert_equal(g.ndim, r.ndim + 1)
    assert_equal(g.shape[axis], r.ndim)

    d = divergence_periodic(g, deltas=None, direction="forward", grad_axis=axis)
    assert_equal(d.ndim, r.ndim)


@pytest.mark.parametrize("xp", all_xp)
def test_gradient_periodic(xp):
    a = xp.arange(5)
    """
    1D Backward Finite Differencing matrix of size 5 is given by:
        d[x] = f[x] - f[x-1]
    """
    FDb = xp.asarray(
        [
            [1, 0, 0, 0, -1],
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
        ]
    )
    """
    1D Forward Finite Differencing matrix of size 5 is given by:
        d[x] = f[x+1] - f[x]
    """
    FDf = xp.asarray(
        [
            [-1, 1, 0, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 0, -1, 1],
            [1, 0, 0, 0, -1],
        ]
    )

    a_forward = gradient_periodic(a, direction="forward")
    a_forward = a_forward.ravel()
    xp.testing.assert_array_almost_equal(a_forward, xp.dot(FDf, a))

    a = xp.arange(5)
    a_back = gradient_periodic(a, direction="backward")
    a_back = a_back.ravel()
    xp.testing.assert_array_almost_equal(a_back, xp.dot(FDb, a))

    # For 1D case, forward direction divergence is backward direction grad
    a = xp.arange(5)
    a_back = divergence_periodic(a[:, xp.newaxis], direction="forward")
    a_back = a_back.ravel()
    xp.testing.assert_array_almost_equal(a_back, xp.dot(FDb, a))

    # adjoint of the forward operator is the negative of the transposed
    # backward operator
    xp.testing.assert_array_almost_equal(FDf, -FDb.T)
    xp.testing.assert_array_almost_equal(FDb, -FDf.T)

    # 2nd order derivative based on forward/back doesn't matter which order
    xp.testing.assert_array_almost_equal(
        xp.dot(FDf, xp.dot(FDb, a)), xp.dot(FDb, xp.dot(FDf, a))
    )


@pytest.mark.parametrize("xp", all_xp)
def test_div_grad_relation(xp):
    # See Lu et. al. Implementation of high-order variational models made easy
    # for image processing.
    # (also Definition 1.3 of Zhou et. al.)
    # verify the identity xp.sum(-div(p)*a) = xp.sum(r*grad(a))
    a = xp.arange(5)
    r = xp.random.randn(a.size)
    assert_almost_equal(
        xp.sum(-backward_diff(r, 0) * a), xp.sum(r * forward_diff(a, 0))
    )

    # verify the identity xp.sum(-div(p)*a) = xp.sum(r*grad(a)) for a 2D case
    a = xp.arange(16).reshape((4, 4))
    r = xp.random.randn(4, 4, 2)
    div = backward_diff(r[..., 0], 0) + backward_diff(r[..., 1], 1)
    grad = xp.zeros_like(r)
    grad[..., 0] = forward_diff(a, 0)
    grad[..., 1] = forward_diff(a, 1)
    assert_almost_equal(xp.sum(-div * a), xp.sum(r * grad))
