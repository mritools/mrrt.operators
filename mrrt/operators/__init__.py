"""
Package containing various system matrix objects for NUFFT, MRI, etc.
"""
from __future__ import absolute_import

from numpy.testing import Tester

# rename basic operators for convenience
from .version import __version__  # noqa
from .LinOp import IdentityOperatorMulti as IdentityOperator  # noqa
from .LinOp import ZeroOperatorMulti as ZeroOperator  # noqa
from .LinOp import DiagonalOperatorMulti as DiagonalOperator  # noqa
from .LinOp import LinearOperatorMulti as LinearOperator  # noqa
from .LinOp import (  # noqa
    LinearOperatorMulti,
    RDiagOperator,
    IDiagOperator,
    BlockDiagLinOp,
    BlockColumnLinOp,
    BlockRowLinOp,
    ArrayOp,
    MaskingOperator,
    linop_shape_args,
    CompositeLinOp,
)

from ._DCT import DCT_Operator  # noqa

# (partial) FFT Operator
from ._FFT import FFT_Operator  # noqa

# Total-Variation operator
from ._TV import TV_Operator  # noqa

# Finite Differences Operator
from ._FiniteDifference import FiniteDifferenceOperator  # noqa

# Orthogonal Matrix Operator
try:
    # requires PyFramelets
    from ._OrthoMatrix import OrthoMatrixOperator  # noqa
except ImportError:
    pass

# Multilevel discrete Framelet operator (requires PyFramelets)
try:
    from ._DWT import MDWT_Operator, FSDWT_Operator  # noqa
except ImportError:
    pass

try:
    # NonUniform FFT, Higher-order TV, NFFT+SENSE+FIELDMAP, Warping
    from mrrt.mri.operators import MRI_Operator, NUFFT_Operator  # noqa
except ImportError:
    pass

test = Tester().test
bench = Tester().bench
