from itertools import product
import time
import warnings

import numpy as np
from numpy.testing import assert_raises, assert_
import pytest

from mrrt.operators import (
    TV_Operator,
    FiniteDifferenceOperator,
    DiagonalOperator,
    IdentityOperator,
    CompositeLinOp,
    BlockDiagLinOp,
    BlockColumnLinOp,
    BlockRowLinOp,
)
from mrrt.operators.LinOp import (
    retrieve_block_out,
    retrieve_block_in,
    split_block_outputs,
    split_block_inputs,
)
from mrrt.utils import config

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


def get_data(xp, shape=(128, 128)):
    # c_cpu = skimage.data.camera().astype(np.float64)
    rstate = xp.random.RandomState(5)
    return rstate.randn(*shape)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_block_diag_identical(xp, order):

    MDWT_Operator = pytest.importorskip("mrrt.operators.MDWT_Operator")
    filters = pytest.importorskip("pyframelets.separable.filters")

    c = get_data(xp)
    Phi = MDWT_Operator(
        c.shape,
        order=order,
        nd_input=False,
        nd_output=False,
        level=3,
        filterbank=filters.pywt_as_filterbank("db2", xp=xp),
        mode="periodization",
        **get_loc(xp),
    )
    nblocks = 3
    Phi_block_op = BlockDiagLinOp([Phi] * nblocks)
    assert_(len(Phi_block_op.blocks) == nblocks)

    # should have references to the same underlying operator (not copies!)
    assert_(Phi_block_op.blocks[0] is Phi_block_op.blocks[1])

    # test round trip
    x = xp.concatenate((c.ravel(order=order),) * nblocks, axis=0)
    y = Phi_block_op * x
    x2 = Phi_block_op.H * y
    xp.testing.assert_allclose(x, x2, rtol=1e-9, atol=1e-9)

    # access linops directly without the blocks attribute
    assert_(Phi_block_op[0] is Phi)

    # access subset via a slice
    Phi_sl = Phi_block_op[slice(1, None)]
    assert_(Phi_sl.nblocks == (Phi_block_op.nblocks - 1))
    assert_(Phi_sl[-1] is Phi)

    # Change one of the linops dynamically
    Phi_sl[0] = 2 * Phi
    assert_(Phi_sl[0] is not Phi)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_block_diag_identical_TV(xp, order):
    c = get_data(xp)
    Phi = TV_Operator(
        c.shape, order=order, nd_input=False, nd_output=False, **get_loc(xp)
    )
    nblocks = 3
    Phi_block_op = BlockDiagLinOp([Phi] * nblocks)
    assert_(len(Phi_block_op.blocks) == nblocks)

    # should have references to the same underlying operator (not copies)
    assert_(Phi_block_op.blocks[0] is Phi_block_op.blocks[1])

    # test round trip
    x = xp.concatenate([c.ravel(order=order)] * nblocks, axis=0)
    y = Phi_block_op * x
    tv_out_ref = Phi * c.ravel(order=order)
    for blk in range(nblocks):
        xp.testing.assert_allclose(y[Phi_block_op.slices_out[0]], tv_out_ref)

    tv_adj_ref = Phi.H * tv_out_ref
    x2 = Phi_block_op.H * y
    for blk in range(nblocks):
        xp.testing.assert_allclose(x2[Phi_block_op.slices_in[0]], tv_adj_ref)

    # test_norm
    xp.testing.assert_allclose(Phi_block_op.norm(x), x2)


@pytest.mark.parametrize(
    "xp, nd_in, nd_out, order1, order2",
    product(
        all_xp,
        [True, False],
        [False],  # MDWT_Operator does not support nd_output = True
        ["C", "F"],
        ["C", "F"],
    ),
)
def test_block_diag(xp, nd_in, nd_out, order1, order2):

    MDWT_Operator = pytest.importorskip("mrrt.operators.MDWT_Operator")
    filters = pytest.importorskip("pyframelets.separable.filters")

    c = get_data(xp)
    Phi = MDWT_Operator(
        c.shape,
        order=order1,
        nd_input=nd_in,
        nd_output=nd_out,
        level=3,
        filterbank=filters.pywt_as_filterbank("db2", xp=xp),
        mode="periodization",
        **get_loc(xp),
    )
    TV = TV_Operator(
        c.shape, order=order2, nd_input=nd_in, nd_output=nd_out, **get_loc(xp)
    )
    if nd_out:
        # non-uniform nd_output shape not allowed
        assert_raises(
            ValueError, BlockDiagLinOp, [Phi, TV], enforce_uniform_order=False
        )
        return

    B = BlockDiagLinOp([Phi, TV], enforce_uniform_order=False, **get_loc(xp))
    assert_(B.blocks[0] is Phi)
    assert_(B.blocks[1] is TV)

    c2 = xp.concatenate((c.ravel(order=order1), c.ravel(order=order2)), axis=0)
    res = B * c2
    dwt_out = res[B.slices_out[0]]
    tv_out = res[B.slices_out[1]]
    if nd_out:
        dwt_out = dwt_out.reshape(B.shapes_out[0], order=B.blocks[0].order)
        tv_out = tv_out.reshape(B.shapes_out[1], order=B.blocks[1].order)
    xp.testing.assert_allclose(dwt_out, Phi * c)
    xp.testing.assert_allclose(tv_out, TV * c)


# @dec.slow  # TODO: mark as slow


@pytest.mark.parametrize(
    "xp, Op",
    product(
        [np],  # only test concurrent blocks with NumPy
        [
            # MDWT_Operator,  # TODO: fix concurrent operation for MDWT_Operator
            TV_Operator,
            FiniteDifferenceOperator,
            DiagonalOperator,
            "composite",
        ],
    ),
)
def test_concurrent_blocks(xp, Op, verbose=False):
    """Test BlockDiagLinOp with concurrent processing."""

    MDWT_Operator = pytest.importorskip("mrrt.operators.MDWT_Operator")
    filters = pytest.importorskip("pyframelets.separable.filters")

    c = get_data(xp)
    dec_level = 2
    dwt_mode = "periodization"
    nd_in = nd_out = False  # True case not working with CompositeLinOp
    order = "F"
    nblocks = 6
    c3d = xp.stack([c] * 16, axis=-1)
    op_kwargs = dict(
        order=order,
        nd_input=nd_in,
        nd_output=nd_out,
        level=dec_level,
        filterbank=filters.pywt_as_filterbank("db2", xp=xp),
        mode=dwt_mode,
    )
    op_kwargs.update(get_loc(xp))

    if Op is FiniteDifferenceOperator:
        op_kwargs["use_corners"] = False

    if Op is DiagonalOperator:
        Phi = Op(diag=xp.arange(c3d.size), **op_kwargs)
    elif Op == "composite":
        # test BlockDiagLinop where each LinOp is a CompositeLinOp
        Phi1 = MDWT_Operator(c3d.shape, **op_kwargs)
        D1 = DiagonalOperator(diag=xp.arange(c3d.size), **op_kwargs)
        Phi = CompositeLinOp([Phi1, D1])

    else:
        Phi = Op(c3d.shape, **op_kwargs)

    # define serial Operator
    Phi_B = BlockDiagLinOp([Phi] * nblocks, concurrent=False, **get_loc(xp))
    # define concurrent Operator
    Phi_Bc = BlockDiagLinOp([Phi] * nblocks, concurrent=True, **get_loc(xp))

    assert_(Phi_B.nd_input == Phi.nd_input)
    assert_(Phi_B.nd_output == Phi.nd_output)
    if Phi.nd_input:
        if order == "F":
            assert_(Phi_B.shape_in == Phi.shape_in + (nblocks,))
        elif order == "C":
            assert_(Phi_B.shape_in == (nblocks,) + Phi.shape_in)
    else:
        assert_(Phi_B.shape_in == (Phi.nargin * nblocks,))

    if Phi.nd_output:
        if order == "F":
            assert_(Phi_B.shape_out == Phi.shape_out + (nblocks,))
        elif order == "C":
            assert_(Phi_B.shape_out == (nblocks,) + Phi.shape_out)
    else:
        assert_(Phi_B.shape_out == (Phi.nargout * nblocks,))

    """
    test forward transform
    """
    allc = xp.asfortranarray(xp.stack([c3d] * nblocks, axis=-1))

    # run serial Operator
    tstart_serial = time.time()
    tmp = Phi_B.H * (Phi_B * allc)
    t_serial = time.time() - tstart_serial

    # run concurrent Operator
    tstart = time.time()
    tmpc = Phi_Bc.H * (Phi_Bc * allc)
    t = time.time() - tstart

    # verify equivalent result
    xp.testing.assert_allclose(tmp, tmpc)

    # only  if nblocks is large enough and c is big enough
    if not (t < t_serial) and (Op is not DiagonalOperator):
        import os

        ncpus = os.cpu_count()
        if ncpus > 1 and nblocks > 1:
            warnings.warn(
                "test_concurrent_blocks: concurrent case unexpectedly "
                "slower on a machine with {} cpus".format(ncpus)
            )

    if verbose:
        print("Operator: {}".format(Op))
        print("    time (serial) = {}".format(t_serial))
        print("    time (concurrent) = {}".format(t))
        time_ratio = t_serial / t
        print("    speedup factor = {}\n".format(time_ratio))


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_block_column_linop(xp, order):
    """Test BlockColumnLinOp."""
    c = get_data(xp)
    kwargs = dict(squeeze_reps=True, order=order)
    kwargs.update(get_loc(xp))
    D = IdentityOperator(c.size, **kwargs)
    D2 = DiagonalOperator(xp.full(c.size, 2.0, dtype=c.dtype), **kwargs)
    D3 = DiagonalOperator(xp.full(c.size, 3.0, dtype=c.dtype), **kwargs)
    ColOp = BlockColumnLinOp([D, D2, D3])

    assert_(ColOp.nargin == D.nargin == D2.nargin == D3.nargin)
    assert_(ColOp.nargout == D.nargout + D2.nargout + D3.nargout)
    c3 = ColOp * c

    # retrieve single blocks
    c3_0 = retrieve_block_out(c3, ColOp, 0).reshape(c.shape, order=order)
    c3_1 = retrieve_block_out(c3, ColOp, 1).reshape(c.shape, order=order)
    c3_2 = retrieve_block_out(c3, ColOp, 2).reshape(c.shape, order=order)

    # retrieve all blocks at once
    allc = split_block_outputs(c3, ColOp, reshape=False)
    xp.testing.assert_array_equal(allc[0], c3_0.ravel(order=order))
    xp.testing.assert_array_equal(allc[1], c3_1.ravel(order=order))
    xp.testing.assert_array_equal(allc[2], c3_2.ravel(order=order))

    # reshape and compare to expected result of running each individually
    xp.testing.assert_allclose(D * c, allc[0])
    xp.testing.assert_allclose(D2 * c, allc[1])
    xp.testing.assert_allclose(D3 * c, allc[2])

    # only retrieve the last 2 outputs
    c3_sl = split_block_outputs(c3, ColOp, sl=slice(1, 3), reshape=False)
    assert_(len(c3_sl) == 2)

    # Block column operator can't split inputs
    assert_raises(ValueError, split_block_inputs, c, ColOp)
    assert_raises(ValueError, retrieve_block_in, c, ColOp, 0)

    # test adjoint
    c_adj = ColOp.H * c3
    c_adj_expected = 0
    for n in range(ColOp.nblocks):
        c_adj_expected += ColOp.blocks[n].H * retrieve_block_out(c3, ColOp, n)
    xp.testing.assert_allclose(c_adj, c_adj_expected)

    # test norm()
    xp.testing.assert_allclose(ColOp.norm(c), c_adj)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_block_row_linop(xp, order):
    """Test BlockRowLinOp."""
    c = get_data(xp)
    c3 = xp.concatenate((c.ravel(order=order),) * 3)
    kwargs = dict(squeeze_reps=True, order=order)
    kwargs.update(get_loc(xp))
    D = IdentityOperator(c.size, **kwargs)
    D2 = DiagonalOperator(xp.full(c.size, 2.0, dtype=c.dtype), **kwargs)
    D3 = DiagonalOperator(xp.full(c.size, 3.0, dtype=c.dtype), **kwargs)
    RowOp = BlockRowLinOp([D, D2, D3])

    assert_(RowOp.nargout == D.nargin == D2.nargin == D3.nargin)
    assert_(RowOp.nargin == D.nargin + D2.nargin + D3.nargin)
    c3sum = RowOp * c3
    xp.testing.assert_array_equal(c3sum.reshape(c.shape, order=order), c * 6)

    # Block row operator can't split outputs
    assert_raises(ValueError, split_block_outputs, c3sum, RowOp)
    assert_raises(ValueError, retrieve_block_out, c3sum, RowOp, 0)

    # test adjoint
    c3_adj = RowOp.H * c3sum
    c3_adj_expected = xp.concatenate(
        [xp.ravel(B.H * c3sum, order=order) for B in RowOp.blocks]
    )
    xp.testing.assert_allclose(c3_adj, c3_adj_expected)

    # test norm()
    xp.testing.assert_allclose(RowOp.norm(c3), c3_adj)
