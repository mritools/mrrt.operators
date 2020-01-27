from itertools import product

import numpy as np
from numpy.testing import assert_
import pytest

from mrrt.operators import FFT_Operator, TV_Operator
from mrrt.operators import BlockDiagLinOp, CompositeLinOp, DiagonalOperator

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


def _get_data(xp, shape=(128, 128)):
    rstate = np.random.RandomState(5)
    return rstate.randn(*shape)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_composite_roundtrip(xp, order):
    x = _get_data(xp)
    Phi = FFT_Operator(
        x.shape,
        order=order,
        nd_input=False,
        nd_output=False,
        squeeze_reps=True,
        **get_loc(xp),
    )
    # Create an intentional mismatch on squeeze_reps_in
    D = DiagonalOperator(
        np.full(x.size, 2, dtype=x.dtype),
        order=order,
        squeeze_reps=False,
        **get_loc(xp),
    )
    C = CompositeLinOp([Phi.H, D, Phi])
    assert_(len(C.blocks) == 3)
    assert_(C.blocks[-1] is Phi)
    assert_(C.blocks[0] is Phi.H)
    r1 = Phi.H * (D * (Phi * x))
    r2 = C * x
    xp.testing.assert_allclose(r1, r2)

    # test adjoint
    x2 = C.H * r2
    x1 = Phi.H * (D * (Phi * r1))
    x1v2 = (Phi.H * D * Phi).H * r1
    xp.testing.assert_allclose(x2, x1)
    xp.testing.assert_allclose(x2, x1v2)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_composite_different_shapes(xp, order):
    # test case where all operators are not the same nargin/nargout
    # This makes sure these variables are being handled correctly when
    # chaining the operators
    x = _get_data(xp)
    TV = TV_Operator(
        x.shape,
        order=order,
        nd_input=False,
        nd_output=False,
        squeeze_reps=True,
        **get_loc(xp),
    )

    # Create an intentional mismatch on squeeze_reps_in
    D = DiagonalOperator(
        np.full(x.size, 2, dtype=x.dtype),
        order=order,
        squeeze_reps=False,
        **get_loc(xp),
    )
    C = CompositeLinOp([TV.H, D, TV])

    # apply composite operator
    tmp = C * x

    # apply separate operators in sequence
    tmp2 = TV.H * (D * (TV * x))

    # verify identical result
    xp.testing.assert_array_equal(tmp, tmp2)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_composite_roundtrip2(xp, order):
    # This one uses the TV_Operator as an example where input/output shapes
    # differ
    x = _get_data(xp)
    Phi = TV_Operator(
        x.shape, order=order, nd_input=False, nd_output=False, **get_loc(xp)
    )

    # Create an intentional mismatch on squeeze_reps_in
    # Also set this to allow repetitions
    D = DiagonalOperator(
        np.full(x.size, 2, dtype=x.dtype),
        order=order,
        matvec_allows_repetitions=True,
        squeeze_reps=False,
        **get_loc(xp),
    )
    # TV Op returns an output that is double the size of the input
    # so make a Block Diagonal LinOp for D
    D2 = BlockDiagLinOp([D, D], **get_loc(xp))
    C = CompositeLinOp([Phi.H, D, Phi])  # D is set to allow repetitions
    C2 = CompositeLinOp([Phi.H, D2, Phi])
    assert_(len(C.blocks) == 3)
    assert_(C.blocks[-1] is Phi)
    assert_(C.blocks[0] is Phi.H)
    r1 = Phi.H * (D * (Phi * x))
    r2 = C * x
    r3 = C2 * x
    xp.testing.assert_allclose(r1, r2)
    xp.testing.assert_allclose(r1, r3)

    # test adjoint
    x1 = Phi.H * (D * (Phi * r1))
    x2 = C.H * r2
    x3 = C2.H * r2
    xp.testing.assert_allclose(x2, x1)
    xp.testing.assert_allclose(x3, x1)
    # The following would raise a ShapeError
    if False:
        x1v2 = (Phi.H * D * Phi).H * r1
        xp.testing.assert_allclose(x2, x1v2)

    # access linops directly without the blocks attribute
    assert_(C[-1] is Phi)
    assert_(C[0] is Phi.H)

    # access subset via a slice
    Csl = C[slice(1, None)]
    assert_(Csl.nblocks == 2)
    assert_(Csl[0] is D)
    assert_(Csl[1] is Phi)
    r1 = D * (Phi * x)
    r2 = Csl * x
    xp.testing.assert_allclose(r1, r2.reshape(r1.shape, order=Csl.order))

    # Change one of the linops dynamically
    assert_(C2.nargins[1] == D2.nargin)
    assert_(C2.nargouts[1] == D2.nargout)
    C2[1] = D
    assert_(C2.nargins[1] == D.nargin)
    assert_(C2.nargouts[1] == D.nargout)
    assert_(C2[1] is D)


@pytest.mark.parametrize("xp, order", product(all_xp, ["C", "F"]))
def test_block_composite_op(xp, order):
    # Create a BlockDiagLinOp where each block is a CompositeLinOp
    x = _get_data(xp)
    Phi = TV_Operator(
        x.shape, order=order, nd_input=False, nd_output=False, **get_loc(xp)
    )

    # Create an intentional mismatch on squeeze_reps_in
    # Also set this to allow repetitions
    D = DiagonalOperator(
        np.full(x.size, 2, dtype=x.dtype),
        order=order,
        matvec_allows_repetitions=True,
        squeeze_reps=False,
        **get_loc(xp),
    )
    C = CompositeLinOp([Phi.H, D, Phi])
    B = BlockDiagLinOp([C] * 8)
    assert_(B[5][0] is Phi.H)
    assert_(B[7][1] is D)
    assert_(B[0][2] is Phi)
