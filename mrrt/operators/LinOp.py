"""
Extended version of the LinearOperator class from pykrylov

class `LinearOperatorMulti` supports multiplication with n-dimensional vectors
and has flexible input/output reshaping options.

A few basic `LinearOperatorMulti` subclasses are also defined here:
`ZeroOperatorMulti`, `IdentityOperatorMulti`, `DiagonalOperatorMulti`,
`ArrayOp`

An alternative Operator toolbox in Python
https://github.com/PythonOptimizers/pykrylov
https://github.com/ghisvail/linop
https://github.com/pchanial/pyoperators

TODO:
Should technically be using the ``matmat`` and ``rmatmat`` methods as in more
recent scipy LinearOperator to indicate that these operators support batch
operation.

"""
from concurrent import futures
from functools import reduce
import logging
import os
import warnings
import weakref

import numpy as np
import scipy.sparse

from mrrt.operators.linop import BaseLinearOperator, ShapeError
from mrrt.operators.linop._types import integer_types, real_types, complex_types
from mrrt.operators.mixins import GpuCheckerMixin
from mrrt.utils import config, embed, masker, prod, profile

if config.have_cupy:
    import cupy
    import cupyx.scipy.sparse

    cupy_ndarray_type = cupy.ndarray
    cupy_spmatrix_type = cupyx.scipy.sparse.spmatrix
else:
    cupy = None
    cupy_ndarray_type = ()
    cupy_spmatrix_type = ()


# TODO: move _same_loc elsewhere?
def _same_loc(y, y_ref):
    """Copy y to the same device (CPU or GPU) as y_ref

    """
    if type(y) == type(y_ref):
        if hasattr(y_ref, "__cuda_array_interface__") and hasattr(
            y_ref, "device"
        ):
            if y.device != y_ref.device:
                with cupy.cuda.Device(y_ref.device):
                    y = cupy.asarray(y)
        return y
    else:
        if isinstance(y, np.ndarray):
            return cupy.asarray(y)
        else:
            return y.get()


__docformat__ = "restructuredtext"

# Default (null) logger.
null_log = logging.getLogger("linop")
# null_log.setLevel(logging.INFO)
null_log.setLevel(logging.WARNING)
null_log.addHandler(logging.NullHandler())

# TODO 1: allow use of a mask to be applied to the input
# TODO 2: allow an option to force input or output to be contiguous

# TODO:  __all__

__all__ = [
    "DiagonalOperatorMulti",
    "RDiagOperator",  # TODO: rename to RealOperator
    "IDiagOperator",  # TODO: rename to ImagOperator
    "IdentityOperatorMulti",
    "BlockDiagLinOp",
    "BlockColumnLinOp",
    "BlockRowLinOp",
    "ArrayOp",
    "LinearOperatorMulti",
    "is_multiarray",
    "linop_shape_args",
    "retrieve_block_out",
    "retrieve_block_in",
    "ZeroOperatorMulti",
    "MaskingOperator",
    "CompositeLinOp",
]


# @profile
def _reshape_input_to_nd(
    x, mask_in, shape_in, nargin, Nrepetitions, order, squeeze_reps, xp=np
):
    x = xp.asarray(x)
    if order == "C":
        if mask_in is None:
            if Nrepetitions == 1 and squeeze_reps:
                x = x.reshape(shape_in, order=order)
            else:
                x = x.reshape((Nrepetitions,) + shape_in, order=order)
        else:
            if x.shape != (Nrepetitions, nargin):
                x = x.reshape((Nrepetitions, nargin), order=order)
            x = embed(x, mask=mask_in, order=order, squeeze_output=squeeze_reps)
    else:
        if mask_in is None:
            if Nrepetitions == 1 and squeeze_reps:
                x = x.reshape(shape_in, order=order)
            else:
                x = x.reshape(shape_in + (Nrepetitions,), order=order)
        else:
            if x.shape != (nargin, Nrepetitions):
                x = x.reshape((nargin, Nrepetitions), order=order)
            x = embed(x, mask=mask_in, order=order, squeeze_output=squeeze_reps)
    return x


def is_multiarray(arr, xp=np):
    """Check if arr is a list of arrays or array of arrays.

    xp is the array_module to use. It can be either numpy.ndarray or
    cupy.ndarray.
    """

    err_msg = "input is not an array, array of arrays or list of arrays"
    if isinstance(arr, xp.ndarray):
        if arr.dtype == "object" and len(arr) > 0:
            # for now, just check based on the first element
            if isinstance(arr[0], xp.ndarray):
                # the elements are arrays
                return True
            else:
                raise ValueError(err_msg)
        else:
            return False
    elif isinstance(arr, (list, tuple)) and len(arr) > 0:
        if isinstance(arr[0], xp.ndarray):
            return True
        else:
            raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)


class LinearOperatorMulti(BaseLinearOperator, GpuCheckerMixin):
    """
    Variant of pykrylov.linop.LinearOperator that supports multiplication with
    n-dimensional vectors and has flexible input/output reshaping options

    A linear operator constructed from a `matvec` and (possibly) a
    `matvec_transp` function. If `symmetric` is `True`, `matvec_transp` is
    ignored. All other keyword arguments are passed directly to the superclass.

    Parameters
    ----------
    nargin : int
    nargout : int
    matvec : callable
    matvec_transp : callable, optional
    matvec_adj : callable, optional
    order : {'C', 'F'}
        Data ordering of multidimensional input arrays
    matvec_allows_repetitions : bool, optional
        If True, multiple repetitions of the input are allowed.  Repetitions
        are currently assumed to occur along the final dimension
    nd_input : bool, optional
        If True, allows inputs to be ndimensional [`shape_in` x repetitions].
        Otherwise inputs must be [nargin x repetitions]
    mask_in : array_like, optional
        if nd_input = True, embed the input into the logical mask prior to the
        operation
    shape_in : array_like, optional
        Expected shape of the input when nd_input = True.  numpy.prod of
        shape_in must match nargin
    squeeze_reps : bool, optional
        if True, remove the repetition dimension when there is only 1
        repetition.
    nd_output : bool, optional
        If True, allows outputs to be ndimensional [`shape_out` x repetitions].
        Otherwise inputs must be [nargout x repetitions]
    mask_out : array_like, optional
        if nd_output=True, only values of the output corresponding to the
        logical mask are returned
    shape_out : array_like, optional
        shape of the output when nd_output = True.  numpy.prod of shape_out
        must match nargout
    squeeze_reps : bool, optional
        if True, remove the output repetition dimension when there is only 1
        repetition

    """

    def __init__(
        self,
        nargin,
        nargout,
        matvec,
        matvec_transp=None,
        matvec_adj=None,
        order="F",
        nd_input=False,
        nd_output=False,
        shape_in=None,
        shape_out=None,
        matvec_allows_repetitions=True,
        squeeze_reps=True,
        mask_in=None,
        mask_out=None,
        debug=False,
        loc_in="cpu",
        loc_out="cpu",
        **kwargs,
    ):

        if isinstance(nargout, cupy_ndarray_type):
            nargout = int(nargout.get())
        if isinstance(nargin, cupy_ndarray_type):
            nargin = int(nargin.get())

        if debug:
            print("Entering LinearOperatorMulti init")
        super(LinearOperatorMulti, self).__init__(nargin, nargout, **kwargs)

        transpose_of = kwargs.pop("transpose_of", None)
        adjoint_of = kwargs.pop("adjoint_of", None)
        conjugate_of = kwargs.pop("conjugate_of", None)

        self._on_gpu = loc_in == "gpu"
        self._check_gpu(self.xp)

        """ Extended attributes for reshaping / matrix multiplication """
        if shape_in is None:
            self.shape_in = (nargin,)
        else:
            self.shape_in = tuple(shape_in)

        if shape_out is None:
            self.shape_out = (nargout,)
        else:
            self.shape_out = tuple(shape_out)

        if shape_in is None:
            if squeeze_reps:
                self.ndim_in = 1
            else:
                self.ndim_in = 2
        else:
            self.ndim_in = len(shape_in)

        if shape_out is None:
            if squeeze_reps:
                self.ndim_out = 1
            else:
                self.ndim_out = 2
        else:
            self.ndim_out = len(shape_out)

        # GRL: adding experimental parameters loc_in, loc_out
        loc_in = loc_in.lower()
        loc_out = loc_out.lower()
        if loc_in not in ["cpu", "gpu"] or loc_out not in ["cpu", "gpu"]:
            raise ValueError("loc_in, loc_out must be either 'cpu' or 'gpu'")
        if loc_in == "gpu":
            if not config.have_cupy:
                raise ValueError(
                    "creating a GPU-based operators requires " "cupy"
                )
            self.xp_in = cupy
        else:
            self.xp_in = np

        if loc_out == "gpu":
            self.xp_out = cupy
        else:
            self.xp_out = np

        self.loc_in = loc_in
        self.loc_out = loc_out

        # are input arrays are stored in C or Fortran order?
        self.order = order

        self.nd_output = nd_output
        self.nd_input = nd_input
        if self.nd_output and (shape_out is None):
            raise ValueError("nd_output requires shape_out be specified")
        if self.nd_input and (shape_in is None):
            raise ValueError("nd_input requires shape_in be specified")

        self.matvec_allows_repetitions = matvec_allows_repetitions
        self.squeeze_reps = squeeze_reps

        if mask_in is not None:
            # TODO: raise error if nd_input is False?
            if loc_in == "cpu" and isinstance(mask_in, cupy_ndarray_type):
                # transfer from gpu to cpu required
                mask_in = mask_in.get()
            mask_in = self.xp.asanyarray(mask_in, order=self.order)
            if mask_in.dtype != self.xp.bool:
                raise ValueError("dtype bool required for mask_in")
            if nargin != mask_in.sum():
                raise ValueError("mask_in size mismatch")
        #            if mask_in.size == mask_in.sum():
        #                # if true everywhere, set the mask to None
        #                mask_in = None
        else:
            if shape_in is not None:
                if nargin != prod(shape_in):
                    raise ValueError("Incompatible nargin, shape_in")
        self.mask_in = mask_in

        if mask_out is not None:
            mask_out = self.xp.asanyarray(mask_out, order=self.order)
            if mask_out.dtype != np.bool:
                raise ValueError("dtype bool required for mask_out")
            if nargout != mask_out.sum():
                raise ValueError("mask_out size mismatch")
        #            if mask_out.size == mask_out.sum():
        #                # if true everywhere, set the mask to None
        #                mask_in = None
        else:
            if shape_out is not None:
                if nargout != prod(shape_out):
                    raise ValueError("Incompatible nargin, shape_out")
        self.mask_out = mask_out
        """ End of extended attributes """

        self.__matvec = matvec
        self.__set_transpose(matvec, transpose_of, matvec_transp, **kwargs)
        self.__set_adjoint(matvec, adjoint_of, matvec_adj, **kwargs)

        # For non-complex operators, transpose = adjoint.
        if self.dtype in integer_types + real_types:
            if self.__T is not None and self.__H is None:
                self.__H = self.__T
            elif self.__T is None and self.__H is not None:
                self.__T = self.__H
        else:
            if (
                transpose_of is None
                and adjoint_of is None
                and conjugate_of is None
            ):
                # We're not in a recursive instantiation.
                # Try to infer missing operators.
                __conj = self.conjugate()
                if self.T is not None:
                    self.__T.__H = __conj
                    if self.H is None and __conj is not None:
                        self.logger.debug("Inferring .H")
                        self.__H = __conj.T
                if self.H is not None:
                    self.__H.__T = __conj
                    if self.T is None and __conj is not None:
                        self.logger.debug("Inferring .T")
                        self.__T = __conj.H
        if debug:
            print("Exiting LinearOperatorMulti init")

    def __set_transpose(
        self, matvec, transpose_of=None, matvec_transp=None, **kwargs
    ):
        self.__T = None
        if self.symmetric:
            self.__T = self
            return

        if transpose_of is None:
            if matvec_transp is not None:
                # Create 'pointer' to transpose operator.
                self.__T = LinearOperatorMulti(
                    nargin=self.nargout,
                    nargout=self.nargin,
                    matvec=matvec_transp,
                    matvec_transp=matvec,
                    transpose_of=self,
                    order=self.order,
                    shape_in=self.shape_out,
                    shape_out=self.shape_in,
                    nd_input=self.nd_output,
                    nd_output=self.nd_input,
                    squeeze_reps=self.squeeze_reps,
                    mask_in=self.mask_out,
                    mask_out=self.mask_in,
                    matvec_allows_repetitions=self.matvec_allows_repetitions,
                    loc_in=self.loc_out,
                    loc_out=self.loc_in,
                    **kwargs,
                )
        else:
            # Use operator supplied as transpose operator.
            if isinstance(transpose_of, BaseLinearOperator):
                self.__T = transpose_of
            else:
                msg = "kwarg transpose_of must be a BaseLinearOperator."
                msg += " Got " + str(transpose_of.__class__)
                raise ValueError(msg)

    def __set_adjoint(self, matvec, adjoint_of=None, matvec_adj=None, **kwargs):

        self.__H = None
        if self.hermitian:
            self.__H = self
            return

        if adjoint_of is None:
            if matvec_adj is not None:
                # Create 'pointer' to adjoint operator.
                self.__H = LinearOperatorMulti(
                    nargin=self.nargout,
                    nargout=self.nargin,
                    matvec=matvec_adj,
                    matvec_adj=matvec,
                    adjoint_of=self,
                    order=self.order,
                    shape_in=self.shape_out,
                    shape_out=self.shape_in,
                    nd_input=self.nd_output,
                    nd_output=self.nd_input,
                    squeeze_reps=self.squeeze_reps,
                    mask_in=self.mask_out,
                    mask_out=self.mask_in,
                    loc_in=self.loc_out,
                    loc_out=self.loc_in,
                    matvec_allows_repetitions=self.matvec_allows_repetitions,
                    **kwargs,
                )
        else:
            # Use operator supplied as adjoint operator.
            if isinstance(adjoint_of, BaseLinearOperator):
                self.__H = adjoint_of
            else:
                msg = "kwarg adjoint_of must be a BaseLinearOperator."
                msg += " Got " + str(adjoint_of.__class__)
                raise ValueError(msg)

    @property
    def T(self):
        "The transpose operator."
        return self.__T

    @property
    def H(self):
        "The adjoint operator."
        return self.__H

    @property
    def bar(self):
        "The complex conjugate operator."
        return self.conjugate()

    def conjugate(self):
        "Return the complex conjugate operator."
        if self.dtype not in complex_types:
            return self

        # conj(A) * x = conj(A * conj(x))
        def matvec(x):
            if x.dtype not in complex_types:
                return (self * x).conj()
            return (self * x.conj()).conj()

        # conj(A).T * x = A.H * x = conj(A.T * conj(x))
        # conj(A).H * x = A.T * x = conj(A.H * conj(x))
        if self.H is not None:
            matvec_transp = self.H.__matvec

            if self.T is not None:
                matvec_adj = self.T.__matvec
            else:

                def matvec_adj(x):
                    if x.dtype not in complex_types:
                        return (self.H * x).conj()
                    return (self.H * x.conj()).conj()

        elif self.T is not None:
            matvec_adj = self.T.__matvec

            def matvec_transp(x):
                if x.dtype not in complex_types:
                    return (self.T * x).conj()
                return (self.T * x.conj()).conj()

        else:
            # Cannot infer transpose or adjoint of conjugate operator.
            matvec_transp = matvec_adj = None

        return LinearOperatorMulti(
            nargin=self.nargin,
            nargout=self.nargout,
            matvec=matvec,
            matvec_transp=matvec_transp,
            matvec_adj=matvec_adj,
            transpose_of=self.H,
            adjoint_of=self.T,
            conjugate_of=self,
            order=self.order,
            shape_in=self.shape_in,
            shape_out=self.shape_out,
            nd_input=self.nd_input,
            nd_output=self.nd_output,
            squeeze_reps=self.squeeze_reps,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            loc_in=self.loc_in,
            loc_out=self.loc_out,
            matvec_allows_repetitions=self.matvec_allows_repetitions,
            dtype=self.dtype,
        )

    def to_array(self):
        "Convert operator to a dense matrix. This is the same as `full`."
        n, m = self.shape
        xp = self.xp_in
        # have to set nd_output False and squeeze_reps True
        sro = self.squeeze_reps
        ndo = self.nd_output
        self.squeeze_reps = True
        self.nd_output = False
        H = xp.empty((n, m), dtype=self.dtype)
        if self.xp_in == "gpu" and self.xp_out == "cpu":
            H = xp.asnumpy(H)
        elif self.xp_in == "cpu" and self.xp_out == "gpu":
            import cupy

            H = cupy.asarray(H)
        e = xp.zeros(m, dtype=self.dtype)
        for j in range(m):
            e[j] = 1
            H[:, j] = self * e
            e[j] = 0
        # restore original operator settings
        self.squeeze_reps = sro
        self.nd_output = ndo
        return H

    def full(self):
        "Convert operator to a dense matrix. This is the same as `to_array`."
        return self.to_array()

    # @profile
    def _matvec(self, x):
        """
        Matrix-vector multiplication.

        Encapsulates the matvec routine specified at
        construct time, to ensure the consistency of the input and output
        arrays with the operator's shape.
        """
        x, Nrepetitions = self.reshape_input_contig(x)

        y = self.__matvec(x)

        return self.reshape_output_contig(y, Nrepetitions)

    @profile
    def reshape_input_contig(self, x, squeeze=None):
        """ reshape input, x, prior to calling `matvec`.

        behavior determined by attributes:  `order`, `nd_input`, `mask_in`,
        `shape_in`, `squeeze_reps_in`."""
        xp = self.xp_in
        if self.loc_in == "cpu" and isinstance(x, cupy_ndarray_type):
            x = x.get()
        else:
            x = xp.asanyarray(x)
        nargout, nargin = self.shape

        if squeeze is None:
            squeeze = self.squeeze_reps

        # check input data consistency
        if (x.size % nargin) == 0:
            Nrepetitions = int(x.size / nargin)
        else:
            msg = "input array size incompatible with operator dimensions"
            msg += "\nx.size={}, nargin={}, nargout={}".format(
                x.size, nargin, nargout
            )
            raise ValueError(msg)

        if (Nrepetitions > 1) and (not self.matvec_allows_repetitions):
            raise ValueError(
                "Multiple repetitions not supported by this " "operator"
            )

        # if 1D, embed back to size ND
        if self.nd_input:
            x = _reshape_input_to_nd(
                x,
                mask_in=self.mask_in,
                shape_in=self.shape_in,
                nargin=self.nargin,
                Nrepetitions=Nrepetitions,
                order=self.order,
                squeeze_reps=squeeze,
                xp=xp,
            )
        else:
            if Nrepetitions == 1 and squeeze:
                x = x.reshape((self.nargin,), order=self.order)
            else:
                if self.order == "C":
                    x = x.reshape((Nrepetitions, self.nargin), order=self.order)
                else:
                    x = x.reshape((self.nargin, Nrepetitions), order=self.order)

        if (self.order == "F") and (not x.flags.f_contiguous):
            x = xp.asfortranarray(x)
        elif (self.order == "C") and (not x.flags.c_contiguous):
            x = xp.ascontiguousarray(x)

        return x, Nrepetitions

    @profile
    def reshape_output_contig(self, y, Nrepetitions, squeeze=None):
        """ reshape output, y, after call to `matvec`.

        behavior determined by attributes:  `order`, `nd_output`,
        `mask_out`, `shape_out`, `squeeze_reps`."""
        nargout, nargin = self.shape
        if squeeze is None:
            squeeze = self.squeeze_reps

        xp = self.xp_out
        if self.loc_out == "cpu" and isinstance(y, cupy_ndarray_type):
            y = y.get()
        else:
            y = xp.asanyarray(y)

        # check output data consistency
        if (not self.nd_output) or (self.mask_out is None):
            if (y.size % nargout) != 0:
                print(
                    "y.shape = {}, y.size = {}, nargout = {}".format(
                        y.shape, y.size, nargout
                    )
                )
                msg = "y size incompatible with operator dimensions"
                raise ValueError(msg)
        else:
            if (y.size % self.mask_out.size) != 0:
                print(
                    "y.shape = {}, y.size = {}, nargout = {}".format(
                        y.shape, y.size, nargout
                    )
                )
                msg = "y size incompatible with mask dimensions"
                raise ValueError(msg)

        if self.nd_output:
            if Nrepetitions == 1 and squeeze:
                shape_out = self.shape_out
            else:
                if self.order == "C":
                    shape_out = (Nrepetitions,) + self.shape_out
                else:
                    shape_out = self.shape_out + (Nrepetitions,)
            y = y.reshape(shape_out, order=self.order)
            if self.mask_out is not None:
                # raise ValueError("unexpected mask")
                # TODO: add attribute to enable mask_idx_ravel argument in
                #       masker
                y = masker(y, self.mask_out, order=self.order)
        else:
            if Nrepetitions == 1 and squeeze:
                shape_out = (nargout,)
            else:
                if self.order == "C":
                    shape_out = (Nrepetitions, nargout)
                else:
                    shape_out = (nargout, Nrepetitions)
            y = y.reshape(shape_out, order=self.order)

        if (self.order == "F") and (not y.flags.f_contiguous):
            y = xp.asfortranarray(y)
        elif (self.order == "C") and (not y.flags.c_contiguous):
            y = xp.ascontiguousarray(y)

        return y

    def rmatvec(self, x):
        """
        Product with the conjugate transpose. This method is included for
        compatibility with Scipy only. Please use the ``H`` attribute instead.
        """
        return self.__H.__mul__(x)

    def matvec(self, x):
        """
        Product. This method is included for compatibility with Scipy only.
        """
        return self.__mul__(x)

    def rmatmat(self, x):
        """
        Matrix (batch) product with the conjugate transpose. This method is
        included for compatibility with Scipy only. Please use the ``H``
        attribute instead.
        """
        return self.__H.__mul__(x)

    def matmat(self, x):
        """
        Matrix (batch) product. This method is included for compatibility with
        Scipy only.
        """
        return self.__mul__(x)

    def print_shape_info(self):
        """ print information corresponding to the input/output reshaping
        attributes. """
        print("matvec = {}".format(self.__matvec))
        print("matvec_adj = {}".format(self.H.__matvec))
        print("matvec_transp = {}".format(self.T.__matvec))
        print("shape_in = {}".format(self.shape_in))
        print("shape_out = {}".format(self.shape_out))
        print("order = {}".format(self.order))
        print("nargin = {}".format(self.shape[0]))
        print("nargout = {}".format(self.shape[1]))
        print("nd_input = {}".format(self.nd_input))
        print("nd_output = {}".format(self.nd_output))
        print(
            "matvec_allows_repetitions = {}".format(
                self.matvec_allows_repetitions
            )
        )
        print("squeeze_reps = {}".format(self.squeeze_reps))

    def __mul_scalar(self, x):
        "Product between a linear operator and a scalar."
        result_type = np.result_type(self.dtype, type(x))

        if x == 0:
            return ZeroOperatorMulti(
                self.nargin,
                self.nargout,
                dtype=result_type,
                order=self.order,
                shape_in=self.shape_in,
                shape_out=self.shape_out,
                nd_input=self.nd_input,
                nd_output=self.nd_output,
                squeeze_reps=self.squeeze_reps,
                mask_in=self.mask_in,
                mask_out=self.mask_out,
                loc_in=self.loc_in,
                loc_out=self.loc_out,
                matvec_allows_repetitions=self.matvec_allows_repetitions,
            )

        def matvec(y):
            return x * (self(y))

        def matvec_transp(y):
            return x * (self.T(y))

        def matvec_adj(y):
            return x.conjugate() * (self.H(y))

        return LinearOperatorMulti(
            nargin=self.nargin,
            nargout=self.nargout,
            symmetric=self.symmetric,
            hermitian=(result_type not in complex_types) and self.hermitian,
            matvec=matvec,
            matvec_transp=matvec_transp,
            matvec_adj=matvec_adj,
            order=self.order,
            shape_in=self.shape_in,
            shape_out=self.shape_out,
            nd_input=self.nd_input,
            nd_output=self.nd_output,
            squeeze_reps=self.squeeze_reps,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            loc_in=self.loc_in,
            loc_out=self.loc_out,
            matvec_allows_repetitions=self.matvec_allows_repetitions,
            dtype=result_type,
        )

    def __mul_linop(self, op):
        "Product between two linear operators."
        if self.nargin != op.nargout:
            raise ShapeError("Cannot multiply operators together")

        # Properly handle shapes for the composite operator
        nd_output = self.nd_output
        squeeze_reps = self.squeeze_reps
        shape_out = self.shape_out
        mask_out = self.mask_out
        loc_out = self.loc_out

        if isinstance(op, LinearOperatorMulti):
            # TODO: check dimensions are compatible
            shape_in = op.shape_in
            order = op.order
            nd_input = op.nd_input
            mask_in = op.mask_in
            loc_in = op.loc_in
            matvec_allows_repetitions = (
                self.matvec_allows_repetitions and op.matvec_allows_repetitions
            )
        else:
            warnings.warn(
                "__mul_linop may be problematic when other operator"
                " is not of type LinearOperatorMulti"
            )

            # TODO: may need to reshape output of self.T or self.H below to
            # be compatible with normal LinearOperator, etc.
            if self.nd_input:
                raise ValueError(
                    "nd_input not supported when other operator"
                    " is not a LinearOperatorMulti instance"
                )
            nd_input = False
            squeeze_reps = True
            order = "C"
            matvec_allows_repetitions = False
            mask_in = None
            shape_in = (op.nargin,)
            loc_in = loc_out

        def matvec(x):
            return self(op(x))

        def matvec_transp(x):
            return op.T(self.T(x))

        def matvec_adj(x):
            return op.H(self.H(x))

        result_type = np.result_type(self.dtype, op.dtype)

        return LinearOperatorMulti(
            nargin=op.nargin,
            nargout=self.nargout,
            symmetric=False,  # Generally.
            hermitian=False,  # Generally.
            matvec=matvec,
            matvec_transp=matvec_transp,
            matvec_adj=matvec_adj,
            order=order,
            shape_in=shape_in,
            shape_out=shape_out,
            nd_input=nd_input,
            nd_output=nd_output,
            squeeze_reps=squeeze_reps,
            matvec_allows_repetitions=matvec_allows_repetitions,
            mask_in=mask_in,
            mask_out=mask_out,
            loc_in=loc_in,
            loc_out=loc_out,
            dtype=result_type,
        )

    # @profile
    def __mul_vector(self, x):
        "Product between a linear operator and a vector."
        self._nMatvec += 1
        result_type = self.xp_in.result_type(self.dtype, x.dtype)
        return self._matvec(x).astype(result_type, copy=False)

    # @profile
    def __mul__(self, x):
        if np.isscalar(x):
            return self.__mul_scalar(x)
        if isinstance(x, BaseLinearOperator):
            return self.__mul_linop(x)
        if isinstance(x, self.xp_in.ndarray):
            return self.__mul_vector(x)
        elif self.loc_in == "gpu":
            # transfer from cpu to gpu required
            return self.__mul_vector(cupy.asarray(x))
        elif self.loc_in == "cpu" and isinstance(x, cupy_ndarray_type):
            # transfer from gpu to cpu required
            return self.__mul_vector(x.get())
        raise ValueError("Cannot multiply")

    # @profile
    def __rmul__(self, x):
        if np.isscalar(x):
            return self.__mul_scalar(x)
        if isinstance(x, BaseLinearOperator):
            return x.__mul_linop(self)
        raise ValueError("Cannot multiply")

    def __add__(self, other):
        if not isinstance(other, BaseLinearOperator):
            raise ValueError("Cannot add")
        if self.shape != other.shape:
            raise ShapeError("Cannot add: different shape")

        # Properly handle shapes for the composite operator
        if isinstance(other, LinearOperatorMulti):
            if self.squeeze_reps != other.squeeze_reps:
                raise ShapeError("Cannot add: different squeeze_reps")
            if self.mask_in != other.mask_in:
                raise ShapeError("Cannot add: different mask_in")
            if self.mask_out != other.mask_out:
                raise ShapeError("Cannot add: different mask_out")
            if self.nd_output != other.nd_output:
                raise ShapeError("Cannot add: different nd_output")
            if self.nd_input != other.nd_input:
                raise ShapeError("Cannot add: different nd_input")
            if self.nd_output:
                if self.shape_out != other.shape_out:
                    raise ShapeError("Cannot add: different shape_out")
            if self.nd_input:
                if self.shape_in != other.shape_in:
                    raise ShapeError("Cannot add: different shape_in")
                if self.order != other.order:
                    raise ShapeError("Cannot add: different order")

            matvec_allows_repetitions = (
                self.matvec_allows_repetitions
                and other.matvec_allows_repetitions
            )
        else:
            matvec_allows_repetitions = False
            if self.nd_input or self.nd_output or self.mask_out or self.mask_in:
                raise ValueError(
                    "nd_input/output and masks only supported if"
                    "both operators are LinearOperatorMulti"
                )
            else:
                warnings.warn(
                    "adding types LinearOperatorMulti and"
                    " {} not tested".format(type(other))
                )

        def matvec(x):
            return self(x) + other(x)

        def matvec_transp(x):
            return self.T(x) + other.T(x)

        def matvec_adj(x):
            return self.H(x) + other.H(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperatorMulti(
            nargin=self.nargin,
            nargout=self.nargout,
            symmetric=self.symmetric and other.symmetric,
            hermitian=self.hermitian and other.hermitian,
            matvec=matvec,
            matvec_transp=matvec_transp,
            matvec_adj=matvec_adj,
            order=self.order,
            shape_in=self.shape_in,
            shape_out=self.shape_out,
            nd_input=self.nd_input,
            nd_output=self.nd_output,
            squeeze_reps=self.squeeze_reps,
            matvec_allows_repetitions=matvec_allows_repetitions,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            loc_in=self.loc_in,
            loc_out=self.loc_out,
            dtype=result_type,
        )

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        if not isinstance(other, BaseLinearOperator):
            raise ValueError("Cannot add")
        if self.shape != other.shape:
            raise ShapeError("Cannot add")

        # Properly handle shapes for the composite operator
        if isinstance(other, LinearOperatorMulti):
            if self.squeeze_reps != other.squeeze_reps:
                raise ShapeError("Cannot add: different squeeze_reps")
            if self.mask_in != other.mask_in:
                raise ShapeError("Cannot add: different mask_in")
            if self.mask_out != other.mask_out:
                raise ShapeError("Cannot add: different mask_out")
            if self.loc_in != other.loc_in:
                raise ShapeError("Cannot add: different loc_in")
            if self.loc_out != other.loc_out:
                raise ShapeError("Cannot add: different loc_out")
            if self.nd_output != other.nd_output:
                raise ShapeError("Cannot add: different nd_output")
            if self.nd_input != other.nd_input:
                raise ShapeError("Cannot add: different nd_input")
            if self.nd_output:
                if self.shape_out != other.shape_out:
                    raise ShapeError("Cannot add: different shape_out")
            if self.nd_input:
                if self.shape_in != other.shape_in:
                    raise ShapeError("Cannot add: different shape_in")
                if self.order != other.order:
                    raise ShapeError("Cannot add: different order")

            matvec_allows_repetitions = (
                self.matvec_allows_repetitions
                and other.matvec_allows_repetitions
            )
        else:
            matvec_allows_repetitions = False
            if self.nd_input or self.nd_output or self.mask_out or self.mask_in:
                raise ValueError(
                    "nd_input/output and masks only supported if"
                    "both operators are LinearOperatorMulti"
                )
            else:
                warnings.warn(
                    "adding types LinearOperatorMulti and"
                    " {} not tested".format(type(other))
                )

        def matvec(x):
            return self(x) - other(x)

        def matvec_transp(x):
            return self.T(x) - other.T(x)

        def matvec_adj(x):
            return self.H(x) - other.H(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperatorMulti(
            nargin=self.nargin,
            nargout=self.nargout,
            symmetric=self.symmetric and other.symmetric,
            hermitian=self.hermitian and other.hermitian,
            matvec=matvec,
            matvec_transp=matvec_transp,
            matvec_adj=matvec_adj,
            order=self.order,
            shape_in=self.shape_in,
            shape_out=self.shape_out,
            nd_input=self.nd_input,
            nd_output=self.nd_output,
            squeeze_reps=self.squeeze_reps,
            matvec_allows_repetitions=matvec_allows_repetitions,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            loc_in=self.loc_in,
            loc_out=self.loc_out,
            dtype=result_type,
        )

    def __div__(self, other):
        if not np.isscalar(other):
            raise ValueError("Cannot divide")
        return self * (1 / other)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise ValueError("Cannot divide")
        return self * (1.0 / other)

    def __pow__(self, other):
        if not isinstance(other, int):
            raise ValueError("Can only raise to integer power")
        if other < 0:
            raise ValueError("Can only raise to nonnegative power")
        if self.nargin != self.nargout:
            raise ShapeError("Can only raise square operators to a power")
        if other == 0:
            return IdentityOperatorMulti(self.nargin)
        if other == 1:
            return self
        return self * self ** (other - 1)


def linop_shape_args(Op):
    """Return a dictionary containing an operator's shape-related kwargs."""
    if not isinstance(Op, LinearOperatorMulti):
        raise TypeError("Expected Op to be of type LinearOperatorMulti")
    shape_args = dict(
        shape_in=Op.shape_in,
        shape_out=Op.shape_out,
        squeeze_reps=Op.squeeze_reps,
        nd_input=Op.nd_input,
        nd_output=Op.nd_output,
    )
    return shape_args


class IdentityOperatorMulti(LinearOperatorMulti):
    """
    A linear operator representing the identity operator of size `nargin`.
    """

    def __init__(self, nargin, shape=None, **kwargs):
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "matvec" in kwargs:
            kwargs.pop("matvec")

        if shape is not None:
            kwargs["shape_in"] = shape
            kwargs["shape_out"] = shape
            kwargs["nd_input"] = True
            kwargs["nd_output"] = True
            kwargs["squeeze_reps"] = True

        mask_in = kwargs.get("mask_in", None)
        if mask_in is not None:
            raise ValueError("masks not supported by IdentityOperatorMulti")
        mask_out = kwargs.get("mask_out", None)
        if mask_out is not None:
            raise ValueError("masks not supported by IdentityOperatorMulti")

        super(IdentityOperatorMulti, self).__init__(
            nargin,
            nargin,
            symmetric=True,
            hermitian=True,
            matvec=lambda x: x,
            **kwargs,
        )


class MaskingOperator(LinearOperatorMulti):
    """
    A linear operator representing the identity operator of size `nargin`.
    """

    def __init__(
        self,
        nargin=None,
        mask_in=None,
        nargout=None,
        mask_out=None,
        shape=None,
        symmetric=False,
        hermitian=False,
        **kwargs,
    ):
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "matvec" in kwargs:
            kwargs.pop("matvec")

        if shape is not None:
            kwargs["shape_in"] = shape
            kwargs["shape_out"] = shape
            kwargs["nd_input"] = True
            kwargs["nd_output"] = True
            kwargs["squeeze_reps"] = True

        if mask_in is None and mask_out is None:
            if nargin is None:
                raise ValueError(
                    "must provide at least one of nargin, " "mask_in, mask_out"
                )
            nargout = nargin
        if mask_in is None:
            if mask_out is None:
                nargout = nargin
            else:
                nargout = mask_out.sum()
                if nargin is None:
                    nargin = mask_out.size
        else:
            nargin = mask_in.sum()
            if mask_out is None:
                nargout = mask_in.size
            else:
                nargout = mask_out.sum()

        if isinstance(nargout, cupy_ndarray_type):
            nargout = int(nargout.get())
        if isinstance(nargin, cupy_ndarray_type):
            nargin = int(nargin.get())

        shape_in = (nargin,)
        shape_out = (nargout,)

        if ("nd_input" in kwargs) or ("nd_output" in kwargs):
            raise ValueError(
                "MaskingOperator doesn't support nd_input or nd_output"
            )

        super(MaskingOperator, self).__init__(
            nargin,
            nargout,
            shape_in=shape_in,
            shape_out=shape_out,
            symmetric=symmetric,
            hermitian=hermitian,
            matvec=self.forward,
            matvec_adj=self.adjoint,
            mask_out=mask_out,
            mask_in=mask_in,
            nd_input=False,
            nd_output=False,
            **kwargs,
        )

    def forward(self, x):
        if (self.mask_in is not None) and (not self.nd_input):
            x = embed(x, self.mask_in, order=self.order)
        y = x
        if (self.mask_out is not None) and (not self.nd_output):
            y = masker(y, self.mask_out, order=self.order)
        return y

    def adjoint(self, y):
        if (self.mask_out is not None) and (not self.nd_output):
            y = embed(y, self.mask_out, order=self.order)
        x = y
        if (self.mask_in is not None) and (not self.nd_input):
            x = masker(x, self.mask_in, order=self.order)
        return x


class RDiagOperator(LinearOperatorMulti):
    """
    A linear operator representing the identitity for the real component and
    zeroing of the imaginary component.
    """

    def __init__(self, nargin, shape=None, dtype=np.complex64, **kwargs):
        # remove any hardcoded LinOp kwargs
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "hermitian" in kwargs:
            kwargs.pop("hermitian")
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "matvec_adj" in kwargs:
            kwargs.pop("matvec_adj")
        if "matvec_transp" in kwargs:
            kwargs.pop("matvec_transp")
        if "nargout" in kwargs:
            kwargs.pop("nargout")

        if shape is not None:
            kwargs["shape_in"] = shape
            kwargs["shape_out"] = shape
            kwargs["nd_input"] = True
            kwargs["nd_output"] = True
            kwargs["squeeze_reps"] = True

        super(RDiagOperator, self).__init__(
            nargin,
            nargin,
            symmetric=True,
            hermitian=True,
            matvec=self.rdiag_matvec,
            dtype=dtype,
            **kwargs,
        )

    def rdiag_matvec(self, x):
        return x.real + 0j


class IDiagOperator(LinearOperatorMulti):
    """
    A linear operator representing the identitity for the imaginary component
    and zeroing of the real component.
    """

    def __init__(self, nargin, shape=None, dtype=np.complex64, **kwargs):
        # remove any hardcoded LinOp kwargs
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "hermitian" in kwargs:
            kwargs.pop("hermitian")
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "matvec_adj" in kwargs:
            kwargs.pop("matvec_adj")
        if "matvec_transp" in kwargs:
            kwargs.pop("matvec_transp")
        if "nargout" in kwargs:
            kwargs.pop("nargout")

        if shape is not None:
            kwargs["shape_in"] = shape
            kwargs["shape_out"] = shape
            kwargs["nd_input"] = True
            kwargs["nd_output"] = True
            kwargs["squeeze_reps"] = True

        super(IDiagOperator, self).__init__(
            nargin,
            nargin,
            symmetric=True,
            hermitian=True,
            matvec=self.idiag_matvec,
            dtype=dtype,
            **kwargs,
        )

    def idiag_matvec(self, x):
        return 1j * x.imag


class DiagonalOperatorMulti(LinearOperatorMulti):
    """
    A diagonal linear operator defined by its diagonal `diag` (a Numpy array.)
    The type must be specified in the `diag` argument, e.g.,
    `np.ones(5, dtype=np.complex)` or `np.ones(5).astype(np.complex)`.

    If ``inplace=True``, the multiplication is done in-place (the input is
    modified).
    """

    def __init__(
        self, diag, nd_input=False, nd_output=False, inplace=False, **kwargs
    ):
        if "symmetric" in kwargs:
            kwargs.pop("symmetric")
        if "hermitian" in kwargs:
            kwargs.pop("hermitian")
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "matvec_adj" in kwargs:
            kwargs.pop("matvec_adj")
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        order = kwargs.pop("order", "F")

        if kwargs.get("nd_input", False):
            warnings.warn("nd_input not supported.  setting to False")
            kwargs["nd_input"] = False
        if kwargs.get("nd_output", False):
            warnings.warn("nd_output not supported.  setting to False")
            kwargs["nd_output"] = False

        self.inplace = inplace

        loc_in = kwargs.get("loc_in", "cpu")
        if loc_in == "gpu":
            xp = cupy
        else:
            xp = np

        diag = xp.squeeze(xp.asarray(diag))
        if diag.ndim != 1:
            raise ValueError("Input must be 1-d array")

        # Fallback implementation if scipy was not found
        self.__diag = diag.copy()

        def forward(x):
            if x.ndim > 1:
                if x.ndim > 2:
                    raise ValueError("Incompatible dimension")
                if self.order == "C":
                    xsize = x.shape[-1]
                    diag_slice = (np.newaxis, Ellipsis)
                else:
                    xsize = x.shape[0]
                    diag_slice = (Ellipsis, np.newaxis)
                if xsize != diag.size:
                    raise ValueError("Incompatible size")
                if self.inplace:
                    x *= diag[diag_slice]
                else:
                    x = x * diag[diag_slice]
            else:
                if self.inplace:
                    x *= diag
                else:
                    x = x * diag

            return x

        def adjoint(x):
            if x.ndim > 1:
                if x.ndim > 2:
                    raise ValueError("Incompatible dimension")
                if self.order == "C":
                    xsize = x.shape[-1]
                    diag_slice = (np.newaxis, Ellipsis)
                else:
                    xsize = x.shape[0]
                    diag_slice = (Ellipsis, np.newaxis)
                if xsize != diag.size:
                    raise ValueError("Incompatible size")
                if self.inplace:
                    if diag.dtype.kind == "c":
                        x *= xp.conj(diag)[diag_slice]
                    else:
                        x *= diag[diag_slice]
                else:
                    if diag.dtype.kind == "c":
                        x = x * xp.conj(diag)[diag_slice]
                    else:
                        x = x * diag[diag_slice]
            else:
                if self.inplace:
                    if diag.dtype.kind == "c":
                        x *= xp.conj(diag)
                    else:
                        x *= diag
                else:
                    if diag.dtype.kind == "c":
                        x = x * xp.conj(diag)
                    else:
                        x = x * diag
            return x

        matvec = forward
        matvec_transp = forward
        if diag.dtype in complex_types:
            matvec_adj = adjoint
        else:
            matvec_adj = matvec_transp

        super(DiagonalOperatorMulti, self).__init__(
            diag.shape[0],
            diag.shape[0],
            symmetric=True,
            hermitian=(diag.dtype not in complex_types),
            matvec=matvec,
            # matvec_adj=lambda x: diag.conjugate()*x,
            matvec_adj=matvec_adj,
            matvec_transp=matvec_transp,
            dtype=diag.dtype,
            order=order,
            **kwargs,
        )

    @property
    def diag(self):
        "Return a reference to the diagonal of the operator."
        return self.__diag

    def __abs__(self):
        return DiagonalOperatorMulti(self.xp_in.abs(self.__diag))

    def _sqrt(self):
        xp = self.xp_in
        if self.dtype not in complex_types and xp.any(self.__diag < 0):
            raise ValueError("Math domain error")
        return DiagonalOperatorMulti(xp.sqrt(self.__diag))


class ZeroOperatorMulti(LinearOperatorMulti):
    """
    The zero linear operator of shape `nargout`-by-`nargin`.
    """

    def __init__(self, nargin, nargout, **kwargs):
        if "matvec" in kwargs:
            kwargs.pop("matvec")
        if "matvec_transp" in kwargs:
            kwargs.pop("matvec_transp")
        order = kwargs.pop("order", "F")

        loc_in = kwargs.get("loc_in", "cpu")
        if loc_in == "gpu":
            xp = cupy
        else:
            xp = np

        def matvec(x):
            result_type = np.result_type(self.dtype, x.dtype)
            return xp.zeros(nargout, dtype=result_type)

        def matvec_transp(x):
            result_type = np.result_type(self.dtype, x.dtype)
            return xp.zeros(nargin, dtype=result_type)

        super(ZeroOperatorMulti, self).__init__(
            nargin,
            nargout,
            symmetric=(nargin == nargout),
            matvec=matvec,
            matvec_transp=matvec_transp,
            matvec_adj=matvec_transp,
            order=order,
            **kwargs,
        )

    def __abs__(self):
        return self

    def _sqrt(self):
        return self


class ArrayOp(LinearOperatorMulti):
    def __init__(
        self, A, symmetric=False, hermitian=False, order="F", **kwargs
    ):
        # self.A = A
        if order != "F":
            raise ValueError("ArrayOp requires order='F'")

        self.loc_in = kwargs.get("loc_in", "cpu")
        if self.loc_in == "gpu":
            xp = cupy
            on_gpu = True
        else:
            xp = np
            on_gpu = False

        if self.loc_in == "cpu" and (
            isinstance(A, cupy_ndarray_type)
            or isinstance(A, cupy_spmatrix_type)
        ):
            A = A.get()
            raise ValueError(
                "A must be a numpy array for weakref below to work correctly."
            )

        if self.loc_in == "gpu" and not (
            isinstance(A, cupy_ndarray_type)
            or isinstance(A, cupy_spmatrix_type)
        ):
            raise ValueError(
                "A must be a cupy array or sparse matrix for weakref below to work correctly."
            )

        self.Aref = weakref.ref(
            A
        )  # TODO!!: find a better way to avoid memory leaks
        if (not on_gpu) and (
            isinstance(A, scipy.sparse.spmatrix)
            or isinstance(A, np.matrixlib.defmatrix.matrix)
        ):
            self.is_matrix = True
        elif on_gpu and isinstance(A, cupy_spmatrix_type):
            self.is_matrix = True
        else:
            self.is_matrix = False
            # A = self._get_if_needed(A)
            A = xp.asarray(A)
            if A.ndim != 2:
                raise ValueError("A must be a 2D array (matrix)")

        nargout, nargin = A.shape

        super(ArrayOp, self).__init__(
            nargin,
            nargout,
            matvec=self.matvec,
            matvec_transp=self.matvec_transp,
            matvec_adj=self.matvec_adj,
            symmetric=symmetric,
            hermetian=hermitian,
            dtype=A.dtype,
            order=order,
            **kwargs,
        )

    def _get_if_needed(self, x):
        if self.loc_in == "cpu" and isinstance(x, cupy_ndarray_type):
            return x.get()
        else:
            return x

    def matvec(self, x):
        xp = self.xp_in
        if x.shape[0] != self.nargin:
            msg = "Input has leading dimension " + str(x.shape[0])
            msg += " instead of %d" % self.nargin
            raise ShapeError(msg)
        x = self._get_if_needed(x)
        if self.is_matrix:
            Ax = self.Aref() * x
        else:
            Ax = xp.dot(self.Aref(), x)
        return Ax

    def matvec_transp(self, y):
        xp = self.xp_in  # TODO: xp_out?
        if y.shape[0] != self.nargout:
            msg = "Input has leading dimension " + str(y.shape[0])
            msg += " instead of %d" % self.nargout
            raise ShapeError(msg)
        y = self._get_if_needed(y)
        if self.is_matrix:
            Ay = self.Aref().T * y
        else:
            Ay = xp.dot(self.Aref().T, y)
        return Ay

    def matvec_adj(self, y):
        xp = self.xp_in  # TODO: xp_out?
        if y.shape[0] != self.nargout:
            msg = "Input has leading dimension " + str(y.shape[0])
            msg += " instead of %d" % self.nargout
            raise ShapeError(msg)
        y = self._get_if_needed(y)
        if self.is_matrix:
            Ay = self.Aref().H * y
        else:
            Ay = xp.dot(xp.conj(self.Aref()).T, y)
        return Ay


class BlockDiagLinOp(LinearOperatorMulti):
    """
    A block diagonal linear operator. Each block must be a linear operator.
    The blocks may be specified as one list, e.g., `[A, B, C]`.

    The input should be a single, contiguous vector corresponding to the
    stacked inputs to [A, B, C].  If operating upon a matrix, columns are
    assumed to correspond to repetitions.

    If 'order' is not specified, the order attribute of the block operator
    will match the 'order' attribute of the first block.
    """

    def __init__(
        self,
        blocks,
        order=None,
        enforce_uniform_order=True,
        concurrent=False,
        **kwargs,
    ):

        self._blocks = blocks

        symmetric = reduce(
            lambda x, y: x and y, [blk.symmetric for blk in blocks]
        )
        hermitian = reduce(
            lambda x, y: x and y, [blk.hermitian for blk in blocks]
        )

        # copy attributes from individual operators
        matvec_allows_repetitions = reduce(
            lambda x, y: x and y,
            [blk.matvec_allows_repetitions for blk in blocks],
        )
        if "squeeze_reps" not in kwargs:
            squeeze_reps = reduce(
                lambda x, y: x and y, [blk.squeeze_reps for blk in blocks]
            )
        else:
            squeeze_reps = kwargs.pop("squeeze_reps")

        if "nd_input" not in kwargs:
            nd_input = reduce(
                lambda x, y: x and y, [blk.nd_input for blk in blocks]
            )
        else:
            nd_input = kwargs.pop("nd_input")

        if "nd_output" not in kwargs:
            nd_output = reduce(
                lambda x, y: x and y, [blk.nd_output for blk in blocks]
            )
        else:
            nd_output = kwargs.pop("nd_output")

        if "loc_in" not in kwargs:
            loc_in = reduce(
                lambda x, y: x and y, [blk.loc_in for blk in blocks]
            )

        else:
            loc_in = kwargs.pop("loc_in")

        if "loc_out" not in kwargs:
            loc_out = reduce(
                lambda x, y: x and y, [blk.loc_out for blk in blocks]
            )
        else:
            loc_out = kwargs.pop("loc_out")

        if loc_in == "gpu":
            if concurrent:
                raise ValueError("concurrent blocks untested with cupy")
            import cupy

            xp = cupy
        else:
            xp = np

        if loc_in != loc_out:
            raise ValueError("loc_in and loc_out must match")

        orders = [blk.order for blk in blocks]
        if enforce_uniform_order and (len(np.unique(orders)) > 1):
            raise ValueError(
                "all operators must have the same 'order' attribute"
            )

        order = orders[0]

        shapes_in = self.shapes_in
        shapes_out = self.shapes_out

        if len(set(shapes_in)) > 1 and nd_input:
            raise ValueError(
                "nd_input=True not supported for non-uniform " "input shape"
            )

        if nd_input:
            if order == "F":
                shape_in = list(set(shapes_in))[0] + (self.nblocks,)
            elif order == "C":
                shape_in = (self.nblocks,) + list(set(shapes_in))[0]
            else:
                raise ValueError("invalid order")
        else:
            shape_in = None

        if len(set(shapes_out)) > 1 and nd_output:
            raise ValueError(
                "nd_output=True not supported for non-uniform " "output shape"
            )
        if nd_output:
            if order == "F":
                shape_out = list(set(shapes_out))[0] + (self.nblocks,)
            elif order == "C":
                shape_out = (self.nblocks,) + list(set(shapes_out))[0]
            else:
                raise ValueError("invalid order")
        else:
            shape_out = None

        self.norm_available = reduce(
            lambda x, y: x and y, [hasattr(blk, "norm") for blk in blocks]
        )

        # shape_in=None, shape_out=None, mask_in=None, mask_out=None

        log = kwargs.get("logger", null_log)
        log.debug("Building new BlockDiagLinOp")

        nargins = [blk.shape[-1] for blk in blocks]
        log.debug("nargins = " + repr(nargins))

        nargouts = [blk.shape[0] for blk in blocks]
        log.debug("nargouts = " + repr(nargouts))

        nargin = sum(nargins)
        nargout = sum(nargouts)

        # Create blocks of transpose and adjoint operators.
        self._blocksT = [blk.T for blk in blocks]
        self._blocksH = [blk.H for blk in blocks]

        #        self.nargins = [blk.shape[-1] for blk in blocks]
        #        self.nargouts = [blk.shape[0] for blk in blocks]
        #        self.shapes_in = [blk.shape_in for blk in blocks]
        #        self.shapes_out = [blk.shape_out for blk in blocks]

        self.slices_in = []
        self.slices_out = []
        row_start = col_start = 0
        nblks = len(self._blocks)
        for blk in range(nblks):
            row_end = row_start + self.nargouts[blk]
            col_end = col_start + self.nargins[blk]
            self.slices_in.append(slice(col_start, col_end))
            self.slices_out.append(slice(row_start, row_end))
            col_start = col_end
            row_start = row_end

        self.concurrent = concurrent
        self.max_workers = kwargs.get("max_workers", None)

        # @profile
        def blk_matvec(
            x,
            blks,
            nargins,
            nargouts,
            slices_in,
            slices_out,
            order=order,
            concurrent=self.concurrent,
            max_workers=self.max_workers,
        ):
            nargin = sum(nargins)
            nargout = sum(nargouts)
            Nrepetitions = self._check_input(x, nargin, nargout)

            # reshape to [nargin, Nrepetitions]
            if order == "C":
                x = x.reshape((Nrepetitions, nargin), order=order)
                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (Nrepetitions, nargout), dtype=result_type, order=order
                )
            else:
                x = x.reshape((nargin, Nrepetitions), order=order)
                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (nargout, Nrepetitions), dtype=result_type, order=order
                )
            nblks = len(blks)

            single_rep = Nrepetitions == 1

            # def _run_block(B, xin):
            #    return B * xin

            # @profile
            def _run_block2(B, xin, sl_out):
                tmp = B * xin
                if B.order == "C":
                    tmp = tmp.reshape((-1, B.nargout), order=B.order)
                else:
                    tmp = tmp.reshape((B.nargout, -1), order=B.order)
                if order == "C":
                    y[:, sl_out] = tmp
                else:
                    y[sl_out, :] = tmp

            if concurrent and (nblks > 1):
                if max_workers is None:
                    # default to as many workers as available cpus
                    max_workers = min(nblks, os.cpu_count())

                # data generator
                if order == "C":
                    data_blocks = (x[:, sl_in] for sl_in in slices_in)
                else:
                    data_blocks = (x[sl_in, :] for sl_in in slices_in)

                # process blocks concurrently
                with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for result in zip(
                        ex.map(_run_block2, blks, data_blocks, slices_out)
                    ):
                        pass

            else:
                for blk in range(nblks):
                    if order == "C":
                        xin = x[:, slices_in[blk]]
                    else:
                        xin = x[slices_in[blk], :]
                    B = blks[blk]

                    ytmp = B * xin

                    # in GPU memory-limited cases may be using Numpy for the
                    # block operator, but CuPy for individual blocks. In that
                    # case need to make sure ytmp is on the correct device.
                    ytmp = _same_loc(ytmp, y)

                    if B.nd_output:
                        if single_rep and B.squeeze_reps:
                            ytmp = ytmp.ravel(B.order)
                        else:
                            if B.order == "C":
                                ytmp = ytmp.reshape(
                                    (-1, B.nargout), order=B.order
                                )
                            else:
                                ytmp = ytmp.reshape(
                                    (B.nargout, -1), order=B.order
                                )
                    if order == "C":
                        if single_rep and B.squeeze_reps:
                            y[0, slices_out[blk]] = ytmp
                        else:
                            # y[row_start:row_end, :] = B * xin
                            y[:, slices_out[blk]] = ytmp
                    else:
                        if single_rep and B.squeeze_reps:
                            y[slices_out[blk], 0] = ytmp
                        else:
                            # y[row_start:row_end, :] = B * xin
                            y[slices_out[blk], :] = ytmp

            if self.squeeze_reps and Nrepetitions == 1:
                y = xp.squeeze(y)
            return y

        blk_dtypes = [blk.dtype for blk in blocks]
        op_dtype = xp.result_type(*blk_dtypes)

        super(BlockDiagLinOp, self).__init__(
            nargin,
            nargout,
            symmetric=symmetric,
            hermitian=hermitian,
            matvec=lambda x: blk_matvec(
                x,
                self._blocks,
                self.nargins,
                self.nargouts,
                self.slices_in,
                self.slices_out,
                order,
            ),
            matvec_transp=lambda x: blk_matvec(
                x,
                self._blocksT,
                self.nargouts,
                self.nargins,
                self.slices_out,
                self.slices_in,
                order,
            ),
            matvec_adj=lambda x: blk_matvec(
                x,
                self._blocksH,
                self.nargouts,
                self.nargins,
                self.slices_out,
                self.slices_in,
                order,
            ),
            dtype=op_dtype,
            order=order,
            nd_output=nd_output,
            nd_input=nd_input,
            shape_in=shape_in,
            shape_out=shape_out,
            squeeze_reps=squeeze_reps,
            matvec_allows_repetitions=matvec_allows_repetitions,
            loc_in=loc_in,
            loc_out=loc_out,
        )

        self.T._blocks = self._blocksT
        self.H._blocks = self._blocksH

    @property
    def blocks(self):
        "The list of blocks defining the block diagonal operator."
        return self._blocks

    @property
    def nblocks(self):
        "The number of blocks contained in the composite operator."
        return len(self._blocks)

    @property
    def nargins(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape[-1] for blk in self._blocks]

    @property
    def nargouts(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape[0] for blk in self._blocks]

    @property
    def shapes_in(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape_in for blk in self._blocks]

    @property
    def shapes_out(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape_out for blk in self._blocks]

    def __getitem__(self, idx):
        blks = self._blocks.__getitem__(idx)
        if isinstance(idx, slice):
            return BlockDiagLinOp(
                blks, symmetric=self.symmetric, hermitian=self.hermitian
            )
        return blks

    def __setitem__(self, idx, ops):
        if not isinstance(ops, BaseLinearOperator):
            if isinstance(ops, list) or isinstance(ops, tuple):
                for op in ops:
                    if not isinstance(op, BaseLinearOperator):
                        msg = (
                            "Block operators can only contain linear operators"
                        )
                        raise ValueError(msg)
        self._blocks.__setitem__(idx, ops)
        if not self.symmetric:
            self._blocksT = [blk.T for blk in self._blocks]
        if not self.hermitian:
            self._blocksH = [blk.H for blk in self._blocks]
        # TODO: norm_available, etc aren't currently auto-updated
        self.__nargin = np.sum(self.nargins)
        self.__nargout = np.sum(self.nargouts)

    def _check_input(self, x, nargin, nargout):
        """Input prep code shared across norm and blk_matvec."""
        self.logger.debug(
            "Multiplying with a vector of size {}".format(x.shape)
        )
        self.logger.debug("nargin=%d, nargout=%d" % (nargin, nargout))

        # check input data consistency
        if np.remainder(x.size, nargin) != 0:
            msg = "input array size incompatible with operator dimensions"
            msg += "\nx.size={}, nargin={}, nargout={}".format(
                x.size, nargin, nargout
            )
            raise ValueError(msg)
        else:
            Nrepetitions = x.size // nargin

        if (not self.matvec_allows_repetitions) and (Nrepetitions > 1):
            raise ValueError(
                "Multiple repetitions not supported by this " "operator"
            )
        return Nrepetitions

    def norm(self, x, concurrent=False, max_workers=None):
        xp = self.xp_in
        if not self.norm_available:
            return self.H * (self * x)
        else:
            nargin = sum(self.nargins)
            nargout = sum(self.nargouts)
            blks = self.blocks

            Nrepetitions = self._check_input(x, nargin, nargout)

            if self.order == "C":
                x = x.reshape((Nrepetitions, nargin), order=self.order)
            else:
                x = x.reshape((nargin, Nrepetitions), order=self.order)

            result_type = xp.result_type(self.dtype, x.dtype)
            nblks = len(blks)

            y = xp.empty(x.shape, dtype=result_type, order=self.order)

            single_rep = Nrepetitions == 1

            def _block_norm(B, xin):
                ytmp = B.norm(xin)
                if B.nd_input:
                    if single_rep and B.squeeze_reps:
                        ytmp = ytmp.ravel(B.order)
                    else:
                        if B.order == "C":
                            ytmp = ytmp.reshape((-1, B.nargout), order=B.order)
                        else:
                            ytmp = ytmp.reshape((B.nargout, -1), order=B.order)
                return ytmp

            if concurrent and (nblks > 1):
                if max_workers is None:
                    # default to as many workers as available cpus
                    max_workers = min(nblks, os.cpu_count())

                # data generator
                if self.order == "C":
                    data_blocks = (x[:, sl_in] for sl_in in self.slices_in)
                else:
                    data_blocks = (x[sl_in, :] for sl_in in self.slices_in)

                # process blocks concurrently
                with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for result, sl_out, blk in zip(
                        ex.map(_block_norm, blks, data_blocks),
                        self.slices_out,
                        blks,
                    ):
                        if result.ndim == 1:
                            y[sl_out, 0] = result
                        else:
                            y[sl_out, :] = result
            else:
                # process blocks sequentially
                if self.order == "C":
                    for blk in range(nblks):
                        xin = x[:, self.slices_in[blk]]
                        B = blks[blk]

                        ytmp = _block_norm(B, xin)

                        if ytmp.ndim == 1:
                            y[0, self.slices_in[blk]] = ytmp
                        else:
                            # y[row_start:row_end, :] = B * xin
                            y[:, self.slices_in[blk]] = ytmp
                else:
                    for blk in range(nblks):
                        xin = x[self.slices_in[blk], :]
                        B = blks[blk]

                        ytmp = _block_norm(B, xin)

                        if ytmp.ndim == 1:
                            y[self.slices_in[blk], 0] = ytmp
                        else:
                            # y[row_start:row_end, :] = B * xin
                            y[self.slices_in[blk], :] = ytmp

            if self.squeeze_reps and Nrepetitions == 1:
                y = xp.squeeze(y)

            return y


def retrieve_block_out(arr, block_op, n, reshape=True, xp=np):
    """Select output from block n from the full output array."""
    if not isinstance(
        block_op, (BlockDiagLinOp, BlockColumnLinOp, BlockRowLinOp)
    ):
        raise ValueError("Expected a Block Operator of Operators")
    xp = block_op.xp_out
    if not isinstance(arr, xp.ndarray):
        raise ValueError("Expected arr to be a {}".format(xp.ndarray))
    if block_op.slices_out is None:
        raise ValueError(
            "Operator of type({}) doesn't support splitting inputs".format(
                type(block_op)
            )
        )
    arr_sub = arr[block_op.slices_out[n]]
    if reshape:
        B = block_op.blocks[n]
        arr_sub = arr_sub.reshape(block_op.shapes_out[n], order=B.order)
    return arr_sub


def split_block_outputs(arr, block_op, sl=None, reshape=True, xp=np):
    # which blocks to retrieve
    block_list = np.arange(block_op.nblocks, dtype=int)
    if sl is not None:
        block_list = block_list[sl]
    out = []
    for blk in block_list:
        sub_array = retrieve_block_out(
            arr, block_op, blk, reshape=reshape, xp=xp
        )
        out.append(sub_array)
    return out


def retrieve_block_in(arr, block_op, n, reshape=True, xp=np):
    """Select output from block n from the full input array."""
    if not isinstance(
        block_op, (BlockDiagLinOp, BlockColumnLinOp, BlockRowLinOp)
    ):
        raise ValueError("Expected a Block Operator of Operators")
    if not isinstance(arr, xp.ndarray):
        raise ValueError("Expected arr to be a numpy ndarray")
    if block_op.slices_in is None:
        raise ValueError(
            "Operator of type({}) doesn't support splitting inputs".format(
                type(block_op)
            )
        )
    arr_sub = arr[block_op.slices_in[n]]
    if reshape:
        B = block_op.blocks[n]
        arr_sub = arr_sub.reshape(block_op.shapes_in[n], order=B.order)
    return arr_sub


def split_block_inputs(arr, block_op, sl=None, reshape=True, xp=np):
    # which blocks to retrieve
    block_list = xp.arange(block_op.nblocks, dtype=int)
    if sl is not None:
        block_list = block_list[sl]
    out = []
    for blk in block_list:
        sub_array = retrieve_block_in(
            arr, block_op, blk, reshape=reshape, xp=xp
        )
        out.append(sub_array)
    return out


class BlockColumnLinOp(LinearOperatorMulti):
    """
    Vertical stack of several linear operators. Each block must be a linear
    operator. The blocks may be specified as one list, e.g., `[A, B, C]`.

    The input should be a single, contiguous vector corresponding to the
    stacked inputs to [A, B, C].  If operating upon a matrix, columns are
    assumed to correspond to repetitions.

    If 'order' is not specified, the order attribute of the block operator
    will match the 'order' attribute of the first block.
    """

    def __init__(
        self,
        blocks,
        order=None,
        enforce_uniform_order=True,
        concurrent=False,
        row_operator=False,
        **kwargs,
    ):

        self._blocks = blocks

        symmetric = False
        hermitian = False

        # copy attributes from individual operators
        matvec_allows_repetitions = reduce(
            lambda x, y: x and y,
            [blk.matvec_allows_repetitions for blk in blocks],
        )

        if "squeeze_reps" not in kwargs:
            squeeze_reps = reduce(
                lambda x, y: x and y, [blk.squeeze_reps for blk in blocks]
            )
        else:
            squeeze_reps = kwargs.pop("squeeze_reps")

        if "loc_in" not in kwargs:
            loc_in = reduce(
                lambda x, y: x and y, [blk.loc_in for blk in blocks]
            )

        else:
            loc_in = kwargs.pop("loc_in")

        if "loc_out" not in kwargs:
            loc_out = reduce(
                lambda x, y: x and y, [blk.loc_out for blk in blocks]
            )
        else:
            loc_out = kwargs.pop("loc_out")

        if loc_in == "gpu":
            if concurrent:
                raise ValueError("concurrent blocks untested with cupy")
            xp = cupy
        else:
            xp = np

        if loc_out != loc_in:
            raise ValueError("loc_in and loc_out must match.")

        if "nd_input" not in kwargs:
            nd_input = reduce(
                lambda x, y: x and y, [blk.nd_input for blk in blocks]
            )
        else:
            nd_input = kwargs.pop("nd_input")

        if "nd_output" not in kwargs:
            nd_output = reduce(
                lambda x, y: x and y, [blk.nd_output for blk in blocks]
            )
        else:
            nd_output = kwargs.pop("nd_output")

        orders = [blk.order for blk in blocks]
        if enforce_uniform_order and (len(np.unique(orders)) > 1):
            raise ValueError(
                "all operators must have the same 'order' attribute"
            )

        order = orders[0]

        shapes_in = self.shapes_in
        shapes_out = self.shapes_out

        if len(set(shapes_in)) > 1 and nd_input:
            raise ValueError(
                "nd_input=True not supported for non-uniform " "input shape"
            )

        if nd_input:
            if order == "F":
                shape_in = list(set(shapes_in))[0] + (self.nblocks,)
            elif order == "C":
                shape_in = (self.nblocks,) + list(set(shapes_in))[0]
            else:
                raise ValueError("invalid order")
        else:
            shape_in = None

        if len(set(shapes_out)) > 1 and nd_output:
            raise ValueError(
                "nd_output=True not supported for non-uniform " "output shape"
            )
        if nd_output:
            if order == "F":
                shape_out = list(set(shapes_out))[0] + (self.nblocks,)
            elif order == "C":
                shape_out = (self.nblocks,) + list(set(shapes_out))[0]
            else:
                raise ValueError("invalid order")
        else:
            shape_out = None

        # self.norm() can only be called if
        self.norm_available = reduce(
            lambda x, y: x and y, [hasattr(blk, "norm") for blk in blocks]
        )

        # shape_in=None, shape_out=None, mask_in=None, mask_out=None

        log = kwargs.get("logger", null_log)
        log.debug("Building new BlockDiagLinOp")

        nargins = [blk.shape[-1] for blk in blocks]
        log.debug("nargins = " + repr(nargins))

        nargouts = [blk.shape[0] for blk in blocks]
        log.debug("nargouts = " + repr(nargouts))

        # nargin behaviour differs from BlockDiagLinOp
        if len(np.unique(self.nargins)) > 1:
            raise ValueError(
                "For a column operator, all blocks must have the same "
                ".nargin (i.e. same number of columns)"
            )
        nargin = nargins[0]

        nargout = sum(nargouts)

        # Create blocks of transpose and adjoint operators.
        self._blocksT = [blk.T for blk in blocks]
        self._blocksH = [blk.H for blk in blocks]

        #        self.nargins = [blk.shape[-1] for blk in blocks]
        #        self.nargouts = [blk.shape[0] for blk in blocks]
        #        self.shapes_in = [blk.shape_in for blk in blocks]
        #        self.shapes_out = [blk.shape_out for blk in blocks]

        # all blocks operate on the same input vector

        if row_operator:
            self.slices_in = []
            self.slices_out = None
            col_start = 0
            nblks = len(self._blocks)
            for blk in range(nblks):
                col_end = col_start + self.nargins[blk]
                self.slices_in.append(slice(col_start, col_end))
                col_start = col_end

        else:
            self.slices_in = None
            self.slices_out = []
            row_start = 0
            nblks = len(self._blocks)
            for blk in range(nblks):
                row_end = row_start + self.nargouts[blk]
                self.slices_out.append(slice(row_start, row_end))
                row_start = row_end

        self.concurrent = concurrent
        self.max_workers = kwargs.get("max_workers", None)

        # @profile
        def blk_matvec(
            x,
            blks,
            nargin,
            nargouts,
            slices_out,
            order,
            concurrent=self.concurrent,
            max_workers=self.max_workers,
        ):
            nargout = sum(nargouts)
            Nrepetitions = self._check_input(x, nargin, nargout)

            # reshape to [nargin, Nrepetitions]
            if order == "C":
                x = x.reshape((Nrepetitions, nargin), order=order)
                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (Nrepetitions, nargout), dtype=result_type, order=order
                )
            else:
                x = x.reshape((nargin, Nrepetitions), order=order)
                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (nargout, Nrepetitions), dtype=result_type, order=order
                )
            nblks = len(blks)

            single_rep = Nrepetitions == 1

            # @profile
            def _run_block2(B, xin, sl_out):
                tmp = B * xin
                if B.order == "C":
                    tmp = tmp.reshape((-1, B.nargout), order=B.order)
                else:
                    tmp = tmp.reshape((B.nargout, -1), order=B.order)
                if order == "C":
                    y[:, sl_out] = tmp
                else:
                    y[sl_out, :] = tmp

            if concurrent and (nblks > 1):
                if max_workers is None:
                    # default to as many workers as available cpus
                    max_workers = min(nblks, os.cpu_count())

                # send the same data to each block
                data_blocks = [x] * nblks

                # process blocks concurrently
                with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for result in zip(
                        ex.map(_run_block2, blks, data_blocks, slices_out)
                    ):
                        pass

            else:
                for blk in range(nblks):
                    xin = x
                    B = blks[blk]

                    ytmp = B * xin

                    if B.nd_output:
                        if single_rep and B.squeeze_reps:
                            ytmp = ytmp.ravel(B.order)
                        else:
                            if B.order == "C":
                                ytmp = ytmp.reshape(
                                    (-1, B.nargout), order=B.order
                                )
                            else:
                                ytmp = ytmp.reshape(
                                    (B.nargout, -1), order=B.order
                                )

                    if order == "C":
                        if ytmp.ndim == 1:
                            y[0, slices_out[blk]] = ytmp
                        else:
                            # y[row_start:row_end, :] = B * xin
                            y[:, slices_out[blk]] = ytmp
                    else:
                        if ytmp.ndim == 1:
                            y[slices_out[blk], 0] = ytmp
                        else:
                            # y[row_start:row_end, :] = B * xin
                            y[slices_out[blk], :] = ytmp

            if self.squeeze_reps and Nrepetitions == 1:
                y = xp.squeeze(y)
            return y

        # @profile
        def blk_matvec_adj(x, blks, nargins, nargout, slices_in, order):
            # TODO?: concurrency not implemented for adjoint case
            #        could implement, but would have some memory overhead
            nargin = sum(nargins)
            Nrepetitions = self._check_input(x, nargin, nargout)

            # reshape to [nargin, Nrepetitions]
            if self.order == "C":
                x = x.reshape((Nrepetitions, nargin), order=order)

                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (Nrepetitions, nargout), dtype=result_type, order=order
                )

            else:
                x = x.reshape((nargin, Nrepetitions), order=order)

                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (nargout, Nrepetitions), dtype=result_type, order=order
                )
            nblks = len(blks)

            single_rep = Nrepetitions == 1

            for blk in range(nblks):
                if self.order == "C":
                    xin = x[:, slices_in[blk]]
                else:
                    xin = x[slices_in[blk], :]
                B = blks[blk]

                ytmp = B * xin
                if self.order == "C":
                    if B.nd_output:
                        if single_rep and B.squeeze_reps:
                            ytmp = ytmp.ravel(B.order)
                        else:
                            ytmp = ytmp.reshape((-1, B.nargout), order=B.order)
                    if single_rep and B.squeeze_reps:
                        if blk == 0:
                            y[0, :] = ytmp
                        else:
                            y[0, :] += ytmp
                    else:
                        if blk == 0:
                            y = ytmp
                        else:
                            y += ytmp
                else:
                    if B.nd_output:
                        if single_rep and B.squeeze_reps:
                            ytmp = ytmp.ravel(B.order)
                        else:
                            ytmp = ytmp.reshape((B.nargout, -1), order=B.order)
                    if single_rep and B.squeeze_reps:
                        if blk == 0:
                            y[:, 0] = ytmp
                        else:
                            y[:, 0] += ytmp
                    else:
                        if blk == 0:
                            y = ytmp
                        else:
                            y += ytmp

            if self.squeeze_reps and Nrepetitions == 1:
                y = xp.squeeze(y)
            return y

        blk_dtypes = [blk.dtype for blk in blocks]
        op_dtype = xp.result_type(*blk_dtypes)

        self.is_row_operator = row_operator
        if row_operator:
            # swap in/out of column operator to create a row operator

            super(BlockColumnLinOp, self).__init__(
                sum(nargins),
                nargouts[0],
                symmetric=symmetric,
                hermitian=hermitian,
                matvec=lambda x: blk_matvec_adj(
                    x,
                    self._blocks,
                    self.nargins,
                    nargouts[0],
                    self.slices_in,
                    order,
                ),
                matvec_transp=lambda x: blk_matvec(
                    x,
                    self._blocksT,
                    nargouts[0],
                    self.nargins,
                    self.slices_in,
                    order,
                ),
                matvec_adj=lambda x: blk_matvec(
                    x,
                    self._blocksH,
                    nargouts[0],
                    self.nargins,
                    self.slices_in,
                    order,
                ),
                dtype=op_dtype,
                order=order,
                nd_output=nd_output,
                nd_input=nd_input,
                shape_in=shape_out,
                shape_out=shape_in,
                squeeze_reps=squeeze_reps,
                matvec_allows_repetitions=matvec_allows_repetitions,
                loc_in=loc_in,
                loc_out=loc_out,
            )
        else:
            super(BlockColumnLinOp, self).__init__(
                nargin,
                nargout,
                symmetric=symmetric,
                hermitian=hermitian,
                matvec=lambda x: blk_matvec(
                    x,
                    self._blocks,
                    nargin,
                    self.nargouts,
                    self.slices_out,
                    order,
                ),
                matvec_transp=lambda x: blk_matvec_adj(
                    x,
                    self._blocksT,
                    self.nargouts,
                    nargin,
                    self.slices_out,
                    order,
                ),
                matvec_adj=lambda x: blk_matvec_adj(
                    x,
                    self._blocksH,
                    self.nargouts,
                    nargin,
                    self.slices_out,
                    order,
                ),
                dtype=op_dtype,
                order=order,
                nd_output=nd_output,
                nd_input=nd_input,
                shape_in=shape_in,
                shape_out=shape_out,
                squeeze_reps=squeeze_reps,
                matvec_allows_repetitions=matvec_allows_repetitions,
                loc_in=loc_in,
                loc_out=loc_out,
            )

        self.T._blocks = self._blocksT
        self.H._blocks = self._blocksH

    @property
    def blocks(self):
        "The list of blocks defining the block diagonal operator."
        return self._blocks

    @property
    def nblocks(self):
        "The number of blocks contained in the composite operator."
        return len(self._blocks)

    @property
    def nargins(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape[-1] for blk in self._blocks]

    @property
    def nargouts(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape[0] for blk in self._blocks]

    @property
    def shapes_in(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape_in for blk in self._blocks]

    @property
    def shapes_out(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape_out for blk in self._blocks]

    def __getitem__(self, idx):
        blks = self._blocks.__getitem__(idx)
        if isinstance(idx, slice):
            return BlockDiagLinOp(
                blks, symmetric=self.symmetric, hermitian=self.hermitian
            )
        return blks

    def __setitem__(self, idx, ops):
        if not isinstance(ops, BaseLinearOperator):
            if isinstance(ops, list) or isinstance(ops, tuple):
                for op in ops:
                    if not isinstance(op, BaseLinearOperator):
                        msg = (
                            "Block operators can only contain linear "
                            "operators"
                        )
                        raise ValueError(msg)
        self._blocks.__setitem__(idx, ops)
        if not self.symmetric:
            self._blocksT = [blk.T for blk in self._blocks]
        if not self.hermitian:
            self._blocksH = [blk.H for blk in self._blocks]
        # TODO: norm_available, etc aren't currently auto-updated
        self.__nargin = np.sum(self.nargins)
        self.__nargout = np.sum(self.nargouts)

    def _check_input(self, x, nargin, nargout):
        """Input prep code shared across norm and blk_matvec."""
        self.logger.debug(
            "Multiplying with a vector of size {}".format(x.shape)
        )
        self.logger.debug("nargin=%d, nargout=%d" % (nargin, nargout))

        # check input data consistency
        if np.remainder(x.size, nargin) != 0:
            msg = "input array size incompatible with operator dimensions"
            msg += "\nx.size={}, nargin={}, nargout={}".format(
                x.size, nargin, nargout
            )
            raise ValueError(msg)
        else:
            Nrepetitions = x.size // nargin

        if (not self.matvec_allows_repetitions) and (Nrepetitions > 1):
            raise ValueError(
                "Multiple repetitions not supported by this " "operator"
            )
        return Nrepetitions

    def norm(self, x, concurrent=False, max_workers=None):
        if not self.norm_available:
            return self.H * (self * x)
        else:
            xp = self.xp_in

            if self.is_row_operator:
                nargin = nargout = xp.sum(self.nargins)
            else:
                nargin = nargout = self.nargins[0]
            if self.loc_in == "gpu":
                nargin = nargin.get()
                nargout = nargout.get()

            blks = self.blocks

            Nrepetitions = self._check_input(x, nargin, nargout)

            if self.order == "C":
                x = x.reshape((Nrepetitions, nargin), order=self.order)
            else:
                x = x.reshape((nargin, Nrepetitions), order=self.order)

            result_type = xp.result_type(self.dtype, x.dtype)
            nblks = len(blks)

            y = xp.empty(x.shape, dtype=result_type, order=self.order)

            single_rep = Nrepetitions == 1

            def _block_norm(B, xin):
                ytmp = B.norm(xin)
                if B.nd_input:
                    if single_rep and B.squeeze_reps:
                        ytmp = ytmp.ravel(B.order)
                    else:
                        if B.order == "C":
                            ytmp = ytmp.reshape((-1, B.nargout), order=B.order)
                        else:
                            ytmp = ytmp.reshape((B.nargout, -1), order=B.order)
                return ytmp

            # process blocks sequentially
            ytmp = _block_norm(blks[0], x)
            if single_rep and blks[0].squeeze_reps:
                if self.order == "C":
                    y[0, :] = ytmp
                else:
                    y[:, 0] = ytmp
            else:
                y = ytmp
            for blk in range(1, nblks):
                B = blks[blk]
                ytmp = _block_norm(B, x)

                if single_rep and B.squeeze_reps:
                    if self.order == "C":
                        y[0, :] += ytmp
                    else:
                        y[:, 0] += ytmp
                else:
                    y += ytmp

            if self.squeeze_reps and Nrepetitions == 1:
                y = xp.squeeze(y)

            return y


class BlockRowLinOp(BlockColumnLinOp):
    """
    Horizontal stack of several linear operators. Each block must be a linear
    operator. The blocks may be specified as one list, e.g., `[A, B, C]`.

    The input should be a single, contiguous vector corresponding to the
    stacked inputs to [A, B, C].  If operating upon a matrix, columns are
    assumed to correspond to repetitions.

    If 'order' is not specified, the order attribute of the block operator
    will match the 'order' attribute of the first block.
    """

    def __init__(
        self,
        blocks,
        order=None,
        enforce_uniform_order=True,
        concurrent=False,
        **kwargs,
    ):

        # re-use ColumnLinOp
        super(BlockRowLinOp, self).__init__(
            blocks,
            order=order,
            enforce_uniform_order=enforce_uniform_order,
            concurrent=concurrent,
            row_operator=True,
            **kwargs,
        )


# TODO: GPU Support
class CompositeLinOp(LinearOperatorMulti):
    """A composite linear operator.

    CompositeLinop([A, B, ..., C]) creates a linop that is equivalent to
    A*B*...*C. Each block must be a linear operator.  The blocks may be
    specified as one list, e.g., `[A, B, C]`.

    The input should be a single, contiguous vector corresponding to the
    input to C.  If operating upon a matrix, columns are
    assumed to correspond to repetitions.

    Note
    ----
    blocks[-1] is applied first.  blocks[0] is applied last.
    i.e. CompositeLinop([A, B, C]) * x ->  A * B * C * x
    """

    def __init__(self, blocks, order=None, **kwargs):

        symmetric = reduce(
            lambda x, y: x and y, [blk.symmetric for blk in blocks]
        )
        hermitian = reduce(
            lambda x, y: x and y, [blk.hermitian for blk in blocks]
        )

        # copy attributes from individual operators
        matvec_allows_repetitions = reduce(
            lambda x, y: x and y,
            [blk.matvec_allows_repetitions for blk in blocks],
        )

        squeeze_reps = blocks[0].squeeze_reps

        loc_in = blocks[-1].loc_in
        loc_out = blocks[0].loc_out
        if loc_in != loc_out:
            raise ValueError("loc_in and loc_out must match")
        if loc_in == "gpu":
            import cupy

            xp = cupy
        else:
            xp = np

        # TODO: add support for nd_input, nd_output?
        nd_input = nd_output = False
        if any([blk.nd_input for blk in blocks]):
            raise ValueError(
                "CompositeLinOp only supports operators with nd_input=False"
            )
        if any([blk.nd_output for blk in blocks]):
            raise ValueError(
                "CompositeLinOp only supports operators with nd_input=False"
            )

        orders = [blk.order for blk in blocks]
        if len(np.unique(orders)) > 1:
            raise ValueError(
                "all operators must have the same 'order' attribute"
            )

        order = orders[0]

        # shape_in=None, shape_out=None, mask_in=None, mask_out=None
        self._blocks = blocks

        log = kwargs.get("logger", null_log)
        log.debug("Building new BlockDiagLinOp")

        log.debug("nargins = " + repr(self.nargins))
        log.debug("nargouts = " + repr(self.nargouts))

        nargin = self.nargins[-1]
        nargout = self.nargouts[0] * self.nreps_internal

        # Create blocks of transpose and adjoint operators.
        self._blocksT = [blk.T for blk in blocks[::-1]]
        self._blocksH = [blk.H for blk in blocks[::-1]]

        additional_validation = True
        if additional_validation:
            # additional sanity checks that operators have compatible shapes

            # ratio between output size of one operator and input of the next
            ratios = self.size_ratios

            # if the cumprod of these (in reversed order) are not integers
            # there is a size mismatch
            if np.any((np.cumprod(ratios[::-1]) % 1) != 0):
                raise ValueError("Incompatible composite operator shapes")

            # any place where ratios > 1, the operator must allow repetitions
            ops_need_reps = np.where(np.asarray(ratios) > 1)[0]
            if np.any(
                [
                    (not self._blocks[k].matvec_allows_repetitions)
                    for k in ops_need_reps
                ]
            ):
                raise ValueError(
                    (
                        "Operators {} of {} would need to allow repetitions"
                        "".format(ops_need_reps, self.nblocks)
                    )
                )

        def blk_matvec(x, blks):
            nargin = blks[-1].nargin
            nargout = blks[0].nargout

            self.logger.debug(
                "Multiplying with a vector of size {}".format(x.shape)
            )
            self.logger.debug("nargin=%d, nargout=%d" % (nargin, nargout))

            # check input data consistency
            if np.remainder(x.size, nargin) != 0:
                msg = "input array size incompatible with operator dimensions"
                msg += "\nx.size={}, nargin={}, nargout={}".format(
                    x.size, nargin, nargout
                )
                raise ValueError(msg)
            else:
                Nrepetitions = x.size // nargin

            if (not self.matvec_allows_repetitions) and (Nrepetitions > 1):
                raise ValueError(
                    "Multiple repetitions not supported by this " "operator"
                )

            # reshape to [nargin, Nrepetitions]
            if self.order == "C":
                x = x.reshape((Nrepetitions, nargin), order=self.order)

                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (Nrepetitions * self.nreps_internal, nargout),
                    dtype=result_type,
                    order=self.order,
                )
            else:
                x = x.reshape((nargin, Nrepetitions), order=self.order)

                result_type = xp.result_type(self.dtype, x.dtype)
                y = xp.empty(
                    (nargout, Nrepetitions * self.nreps_internal),
                    dtype=result_type,
                    order=self.order,
                )
            nblks = len(blks)

            single_rep = y.shape[-1] == 1
            for blk in range(nblks):
                B = blks[-1 - blk]

                x = B * x

            if single_rep and B.squeeze_reps:
                if self.order == "C":
                    y[0, :] = x
                else:
                    y[:, 0] = x
            else:
                # y[row_start:row_end, :] = B * xin
                y[:, :] = x

            if self.squeeze_reps and Nrepetitions == 1:
                # if nrep_final_op > 1:
                #    y = y.reshape((-1, Nrepetitions), order=self.order)
                y = xp.squeeze(y)
            return y

        blk_dtypes = [blk.dtype for blk in blocks]
        op_dtype = xp.result_type(*blk_dtypes)

        super(CompositeLinOp, self).__init__(
            nargin,
            nargout,
            symmetric=symmetric,
            hermitian=hermitian,
            matvec=lambda x: blk_matvec(x, self._blocks),
            atvec_transp=lambda x: blk_matvec(x, self._blocksT),
            matvec_adj=lambda x: blk_matvec(x, self._blocksH),
            dtype=op_dtype,
            order=order,
            nd_output=nd_output,
            nd_input=nd_input,
            squeeze_reps=squeeze_reps,
            matvec_allows_repetitions=matvec_allows_repetitions,
            loc_in=loc_in,
            loc_out=loc_out,
        )

        self.T._blocks = self._blocksT
        self.H._blocks = self._blocksH

    @property
    def blocks(self):
        "The list of blocks defining the composite operator."
        return self._blocks

    @property
    def nblocks(self):
        "The number of blocks contained in the composite operator."
        return len(self._blocks)

    @property
    def nargins(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape[-1] for blk in self._blocks]

    @property
    def nargouts(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape[0] for blk in self._blocks]

    @property
    def shapes_in(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape_in for blk in self._blocks]

    @property
    def shapes_out(self):
        "The number of blocks contained in the composite operator."
        return [blk.shape_out for blk in self._blocks]

    @property
    def size_ratios(self):
        return [
            self._blocks[k + 1].nargout / self._blocks[k].nargin
            for k in range(self.nblocks - 1)
        ] + [1]

    @property
    def nreps_internal(self):
        "Net multiple of the input size, present at the output"
        nreps_in = np.cumprod(self.size_ratios[::-1])[-1]
        if (nreps_in % 1) != 0:
            raise ValueError("size error")
        return int(nreps_in)

    def __getitem__(self, idx):
        blks = self._blocks.__getitem__(idx)
        if isinstance(idx, slice):
            return CompositeLinOp(blks)
        return blks

    def __setitem__(self, idx, ops):
        if not isinstance(ops, BaseLinearOperator):
            if isinstance(ops, list) or isinstance(ops, tuple):
                for op in ops:
                    if not isinstance(op, BaseLinearOperator):
                        msg = (
                            "Block operators can only contain linear "
                            "operators"
                        )
                        raise ValueError(msg)
        self._blocks.__setitem__(idx, ops)
        self.__symmetric = reduce(
            lambda x, y: x and y, [blk.symmetric for blk in self._blocks]
        )
        self.__hermitian = reduce(
            lambda x, y: x and y, [blk.hermitian for blk in self._blocks]
        )
        if not self.symmetric:
            self._blocksT = [blk.T for blk in self._blocks]
        if not self.hermitian:
            self._blocksH = [blk.H for blk in self._blocks]
        self.__nargin = self.nargins[-1]
        self.__nargout = self.nargouts[0] * self.nreps_internal
