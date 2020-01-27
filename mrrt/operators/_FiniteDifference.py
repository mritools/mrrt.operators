import functools

import numpy as np
from mrrt.operators import LinearOperatorMulti
from mrrt.operators.mixins import PriorMixin
from mrrt.utils import get_array_module, prod

# Note:  _prior_add and _prior_subtract methods provided by PriorMixin


# Could use routines from ODL that support more boundary modes, etc, but for
# the simple diff routines here seem sufficient for now.
# try:
#    # Note odl is under Mozilla Public License v2.0, but odlcuda is GPLv3
#    from odl.discr.diff_ops import finite_diff
#    from functools import partial
#    # the following function behaves similarly to gradient_periodic
#
#    odl_finite_diff_periodic = partial(finite_diff, method='forward',
#                                       pad_mode='periodic')
#    has_odl = True
# except ImportError():
#    has_odl = False


__all__ = [
    "FiniteDifferenceOperator",
    "gradient_periodic",
    "divergence_periodic",
    "gradient_ravel_offsets",
    "divergence_ravel_offsets",
    "gradient",
    "compute_offsets",
]

"""
Note:  See Lysaker2005 on higher order TV minimization
       Lysaker2006 - Iterative Image Restoration Combining Total Variation
       Minimization and a Second-Order Functional
       See also Papafitsoros2013

       Very detailed technical report:
       Wu2010 - cam09-76 - Augmented Lagrangian Method, Dual Methods, and Split
       Bregman Iteration for ROF, Vectorial TV, and High Order Models

       Chambolle2009 - An introduction to Total Variation for Image Analysis
       particularly, section 3, eq. 37

       See Lou2010, Beck2009 for definition of anisotropic case
       Choski2011 is explicitly about denoising & deblurring in the
       anisotropic case

See also the TVReg software accompanying the following paper:
Jensen2011 - Implementation of an optimal first-order method for strongly
convex total variation regularization
And the mxTV Software for Total Variation Image Reconstruction by the same
group.


Original reference for the isotropic TV model:
[1] L. Rudin, S. Osher, and E. Fatemi, Nonlinear total variation based noise
removal algorithms, Physica D, 60 (1992), pp. 259–268.

References for anisotropic TV:
[2] S. Esedoglu and S. J. Osher, Decomposition of images by the anisotropic
rudin-osher-fatemi model, Comm. Pure Appl. Math, 57 (2004), pp. 1609–1626.

[3] R. Choksi, Y. van Gennip, and A. Oberman, Anisotropic total variation
regularized l1-approximation and denoising/deblurring of 2d bar codes,
Inverse Probl. and Imaging, 3 (2011), pp. 591–617.



        Weiss2009 - REVIEW - EFFICIENT SCHEMES FOR TOTAL VARIATION MINIMIZATION
        UNDER CONSTRAINTS IN IMAGE PROCESSING.pdf



Note: TODO:
    for periodic boundary conditions, the discrete differentiation operators
    can be represented as a circular convolution, allowing use of the FFT

TODO:
    see Lou et. al.
    A WEIGHTED DIFFERENCE OF ANISOTROPIC AND ISOTROPIC TOTAL VARIATION MODEL
    FOR IMAGE PROCESSING
    ftp://ftp.math.ucla.edu/pub/camreport/cam14-69.pdf
"""


def _slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    This function is copied from numpy's arraypad.py

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> _slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), (...,))
    """
    return (slice(None),) * axis + (sl,) + (...,)


def compute_offsets(shape, use_corners=False):
    ndim = len(shape)
    if ndim == 1:
        offsets = [1]
    elif ndim == 2:
        nx = shape[0]
        strides = [1, nx]
        if use_corners:
            offsets = [1, nx, nx - 1, nx + 1]
        else:
            offsets = strides
    elif ndim == 3:
        nx, ny = shape[:2]
        strides = [1, nx, nx * ny]
        if not use_corners:
            offsets = strides
        else:
            offsets = []
            for xoff in [-1, 0, 1]:
                for yoff in [-1, 0, 1]:
                    for zoff in [-1, 0, 1]:
                        if xoff == 0 and yoff == 0 and zoff == 0:
                            continue
                        offset = (
                            xoff * strides[0]
                            + yoff * strides[1]
                            + zoff * strides[2]
                        )
                        if offset < 0:
                            # can skip all negative offsets by symmetry
                            continue
                        offsets.append(offset)
            offsets = np.sort(offsets).tolist()
    elif ndim == 4:
        nx, ny, nz = shape[:3]
        strides = [1, nx, nx * ny, nx * ny * nz]
        if not use_corners:
            offsets = strides
        else:
            offsets = []
            for xoff in [-1, 0, 1]:
                for yoff in [-1, 0, 1]:
                    for zoff in [-1, 0, 1]:
                        for toff in [-1, 0, 1]:
                            if (
                                xoff == 0
                                and yoff == 0
                                and zoff == 0
                                and toff == 0
                            ):
                                continue
                            offset = (
                                xoff * strides[0]
                                + yoff * strides[1]
                                + zoff * strides[2]
                                + toff * strides[3]
                            )
                            if offset < 0:
                                # can skip all negative offsets by symmetry
                                continue
                            offsets.append(offset)
            offsets = np.sort(offsets).tolist()
    else:
        if not use_corners:
            offsets = [1]
            for d in range(1, len(shape)):
                offsets.append(prod(shape[:d]))
        else:
            raise ValueError(">4D currently unsupported for use_corners case")
    return offsets


class FiniteDifferenceOperator(PriorMixin, LinearOperatorMulti):
    """ Total variation operator. """

    def __init__(
        self,
        arr_shape,
        norm=1,
        prior=None,
        order="F",
        arr_dtype=np.float32,
        grid_size=None,
        nd_input=True,
        nd_output=True,
        squeeze_reps=True,
        use_corners=False,
        custom_offsets=None,
        axes=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        arr_shape : int
            shape of the array to filter
        degree : int
            degree of the directional derivatives to apply
        order : {'C','F'}
            array ordering that will be assumed if inputs/outputs need to be
            reshaped
        prior : array_like
            subtract this prior before computing the finite difference
        arr_dtype : numpy.dtype
            dtype for the filter coefficients
        grid_size : array, optional
            size of the grid along each dimension of the image.  defaults to
            ones.  (e.g. needed if the voxels are anisotropic.)
        use_corners : bool, optional
            If True, use additional "diagonal" directions as opposed to only
            differences along the primary axes.  When True, the number of
            directions will be (3**ndim - 1)/2.  When False, ndim directions
            are used.
        custom_offsets : list of int or None
            can be used to override the default offsets
        axes : list of int or None
            Can be used to specify a subset of axes over which to take the
            finite difference.  If None, all axes are used.  This option is
            only compatible with `use_corners = False`.  If `custom_offsets`
            is provided, this argument is ignored.
        """
        if isinstance(arr_shape, np.ndarray):
            # retrieve shape from array
            arr_shape = arr_shape.shape

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        if axes is not None:
            if np.isscalar(axes):
                axes = (axes,)
            self.axes = axes
        else:
            self.axes = None

        if custom_offsets is None:
            self.offsets = compute_offsets(
                tuple(arr_shape), use_corners=use_corners
            )
            if use_corners and axes is not None:
                raise ValueError(
                    "use_corners not supported with customized axes"
                )
            elif axes is not None:
                self.offsets = [self.offsets[ax] for ax in axes]
        else:
            self.offsets = custom_offsets
        self.num_offsets = len(self.offsets)

        self.grid_size = grid_size
        if self.grid_size is not None:
            if len(self.grid_size) != self.num_offsets:
                raise ValueError(
                    "grid_size array must match the number of offsets"
                )
        #           else:
        #           if len(self.grid_size) != len(self.axes):
        #               raise ValueError(
        #                   "grid_size array must match the number of "
        #                   "axes specified")
        if self.order == "C":
            self.grad_shape = (self.num_offsets,) + tuple(arr_shape)
        else:
            self.grad_shape = tuple(arr_shape) + (self.num_offsets,)

        shape_out = self.grad_shape
        shape_in = self.arr_shape
        nargin = prod(self.arr_shape)
        nargout = prod(self.grad_shape)

        # output of FFTs will be complex, regardless of input type
        self.result_dtype = np.result_type(arr_dtype, np.float32)

        self.prior = prior
        self.norm = norm
        self._limit = 1e-6

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            import cupy

            xp = cupy
        else:
            xp = np

        self.mask_in = kwargs.pop("mask_in", None)
        if self.mask_in is not None:
            nargin = self.mask_in.sum()
            nd_input = True  # TODO: fix LinOp to remove need for this.  why does DWT case not need it?
        self.mask_out = kwargs.pop("mask_out", None)
        if self.mask_out is not None:
            # TODO: probably wrong if order = 'C'
            if self.order == "C":
                stack_axis = 0
            else:
                stack_axis = -1
            self.mask_out = xp.stack(
                [self.mask_out] * self.num_offsets, stack_axis
            )
            nargout = self.mask_out.sum()
            nd_output = True

        self.grad_func = gradient_periodic
        self.div_func = divergence_periodic

        if self.order == "C":
            grad_axis = 0
        else:
            grad_axis = -1
        self.grad_axis = grad_axis

        if custom_offsets is None and (not use_corners):
            # standard grad/div along all axes
            # This verion uses periodic boundary conditions
            self.grad_func = functools.partial(
                gradient_periodic, axes=self.axes
            )
            self.div_func = functools.partial(
                divergence_periodic, axes=self.axes
            )
        else:
            # customized offsets
            self.grad_func = functools.partial(
                gradient_ravel_offsets, offsets=self.offsets
            )
            self.div_func = functools.partial(
                divergence_ravel_offsets, offsets=self.offsets
            )

        super(FiniteDifferenceOperator, self).__init__(
            nargin=nargin,
            nargout=nargout,
            matvec=self.forward,
            matvec_transp=self.adjoint,
            matvec_adj=self.adjoint,
            nd_input=nd_input,
            nd_output=nd_output,
            shape_in=shape_in,
            shape_out=shape_out,
            order=self.order,
            matvec_allows_repetitions=True,
            squeeze_reps=squeeze_reps,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            symmetric=False,
            hermetian=False,
            dtype=self.result_dtype,
            **kwargs,
        )

    def forward(self, x):
        """ image gradient """
        xp = self.xp_in
        nreps = int(x.size / prod(self.arr_shape))
        if nreps == 1:
            x = x.reshape(self.arr_shape, order=self.order)
            g = self.grad_func(
                self._prior_subtract(x),
                deltas=self.grid_size,
                direction="forward",
                grad_axis=self.grad_axis,
            )
        else:
            if self.order == "C":
                g = xp.zeros((nreps,) + self.grad_shape, dtype=x.dtype)
                for r in range(nreps):
                    xr = x[r, ...].reshape(self.arr_shape, order=self.order)
                    g[r, ...] = self.grad_func(
                        self._prior_subtract(xr),
                        deltas=self.grid_size,
                        direction="forward",
                        grad_axis=self.grad_axis,
                    )
            else:
                g = xp.zeros(self.grad_shape + (nreps,), dtype=x.dtype)
                for r in range(nreps):
                    xr = x[..., r].reshape(self.arr_shape, order=self.order)
                    g[..., r] = self.grad_func(
                        self._prior_subtract(xr),
                        deltas=self.grid_size,
                        direction="forward",
                        grad_axis=self.grad_axis,
                    )

        return g

    def adjoint(self, g):
        """ image divergence """
        xp = self.xp_in
        nreps = int(g.size / prod(self.grad_shape))
        # TODO: fix gradient_adjoint for N-dimensional case
        # TODO: prior case-> add prior back?
        # return gradient_adjoint(g, grad_axis='last')
        if nreps == 1:
            g = g.reshape(self.grad_shape, order=self.order)
            d = self.div_func(
                g,
                deltas=self.grid_size,
                direction="forward",
                grad_axis=self.grad_axis,
            )
        else:
            if self.order == "C":
                d = xp.zeros((nreps,) + self.arr_shape, dtype=g.dtype)
                for r in range(nreps):
                    d[r, ...] = self.div_func(
                        g[r, ...],
                        deltas=self.grid_size,
                        direction="forward",
                        grad_axis=self.grad_axis,
                    )
            else:
                d = xp.zeros(self.arr_shape + (nreps,), dtype=g.dtype)
                for r in range(nreps):
                    d[..., r] = self.div_func(
                        g[..., r],
                        deltas=self.grid_size,
                        direction="forward",
                        grad_axis=self.grad_axis,
                    )
        return -d  # adjoint of grad is -div


def forward_diff(f, axis, mode="periodic", xp=None):
    # (periodic) forward difference of f along the specified axis
    if xp is None:
        xp, on_gpu = get_array_module(f)
    if mode == "periodic":
        return xp.roll(f, -1, axis=axis) - f
    elif mode == "edge":
        tmp = xp.roll(f, -1, axis=axis) - f
        sl = [slice(None)] * tmp.ndim
        sl[axis] = slice(-1, None)
        tmp[tuple(sl)] = 0
        return tmp
    else:
        raise NotImplementedError(
            "Only periodic boundary currently implemented."
        )


def backward_diff(f, axis, mode="periodic", xp=None):
    # (periodic) backward difference of f along the specified axis
    if xp is None:
        xp, on_gpu = get_array_module(f)
    if mode == "periodic":
        return f - xp.roll(f, 1, axis=axis)
    elif mode == "edge":
        tmp = f - xp.roll(f, 1, axis=axis)
        sl = [slice(None)] * tmp.ndim
        sl[axis] = slice(0, 1)
        tmp[tuple(sl)] = 0
        return tmp
    else:
        raise NotImplementedError(
            "Only periodic boundary currently implemented."
        )


# from itertools import combinations
# list(combinations(range(ndim), 2))


def gradient_periodic(
    f,
    direction="forward",
    axes=None,
    deltas=None,
    grad_axis="last",
    mode="periodic",
):
    """ This version based on np.roll is very simple, but a bit slower than
    the gradient defined in numpy.gradient and it's variant below

    Parameters
    ----------
    f : array
        n-dimensional array over which to compute the gradient

    direction : {'forward', 'backward'}
        whether to use forward or backward differencing

    axes : list or array, optional
        list of axes along which to compute the gradient (default = all)

    deltas : list or array, optional
        grid spacing along each dimension (defaults to 1 on all dimensions)

    grad_axis : {'last', 'first', 0, -1}
        output array has dimensions f.ndim + 1.  `grad_axis` controls whether
        the extra dimension is added in the first or last position

    """
    xp, on_gpu = get_array_module(f)

    f = xp.asanyarray(f)
    N = f.ndim  # number of dimensions

    if direction.lower() in ["forward", "forw", "f"]:
        diff_func = functools.partial(forward_diff, xp=xp, mode=mode)
    elif direction.lower() in ["backward", "back", "b"]:
        if mode != "periodic":
            raise NotImplementedError("untested")
        diff_func = functools.partial(backward_diff, xp=xp)
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if axes is None:
        axes = np.arange(N)
    else:
        if np.isscalar(axes):
            axes = (axes,)
        axes = np.asanyarray(axes)
        if axes.max() > (N - 1):
            raise ValueError(
                "maximum axis = {}, but f.ndim only {}".format(axes.max(), N)
            )

    if deltas is not None:
        try:
            if len(deltas) != len(axes):
                raise ValueError("deltas array length must match f.ndim")
        except TypeError:
            raise TypeError("deltas should be a sequence")

    otype = f.dtype.char
    if otype not in ["f", "d", "F", "D", "m", "M"]:
        otype = "d"

    if grad_axis in [0, "first"]:
        grad_axis = 0
        g = xp.empty((len(axes),) + f.shape, dtype=otype)
    elif grad_axis in [-1, "last"]:
        grad_axis = -1
        g = xp.empty(f.shape + (len(axes),), dtype=otype)
    else:
        raise ValueError("Unsupported grad_axis: {}".format(grad_axis))
    for n, axis in enumerate(axes):
        if grad_axis == 0:
            g[n, ...] = diff_func(f, axis=axis)
            if deltas is not None:
                g[n, ...] /= deltas[n]
        elif grad_axis == -1:
            g[..., n] = diff_func(f, axis=axis)
            if deltas is not None:
                g[..., n] /= deltas[n]

    return g


def gradient_ravel_offsets(
    f,
    direction="forward",
    axes=None,
    deltas=None,
    grad_axis="last",
    offsets=None,
    use_corners=False,
):
    """ Can use this version for various offsets as in Fessler's CDiff objects.
    Note: boundary conditions won't exactly match.  problematic?

    Parameters
    ----------
    f : array
        n-dimensional array over which to compute the gradient

    direction : {'forward', 'backward'}
        whether to use forward or backward differencing

    axes : list or array, optional
        list of axes along which to compute the gradient (default = all)

    deltas : list or array, optional
        grid spacing along each dimension (defaults to 1 on all dimensions)

    grad_axis : {'last', 'first', 0, -1}
        output array has dimensions f.ndim + 1.  `grad_axis` controls whether
        the extra dimension is added in the first or last position

    """
    xp, on_gpu = get_array_module(f)
    f = xp.asanyarray(f)
    if offsets is None:
        offsets = compute_offsets(f.shape, use_corners)
    num_offsets = len(offsets)

    ndim = f.ndim
    if direction.lower() in ["forward", "forw", "f"]:
        n_roll = -1
    elif direction.lower() in ["backward", "back", "b"]:
        n_roll = 1
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if deltas is not None:
        deltas = np.asanyarray(deltas)
        if len(deltas) != num_offsets:
            raise ValueError(
                "deltas array length must match the number of " "offsets"
            )

    if axes is None:
        axes = np.arange(ndim)
    else:
        axes = np.asanyarray(axes)
        if axes.max() > (ndim - 1):
            raise ValueError(
                "maximum axis = {}, but f.ndim only {}".format(axes.max(), ndim)
            )

    otype = f.dtype.char
    if otype not in ["f", "d", "F", "D", "m", "M"]:
        otype = "d"

    fshape = f.shape
    f = f.ravel(order="F")
    if grad_axis in [0, "first"]:
        grad_axis = 0
        g = xp.empty((num_offsets,) + f.shape, dtype=otype)
    elif grad_axis in [-1, "last"]:
        grad_axis = -1
        g = xp.empty(f.shape + (num_offsets,), dtype=otype)
    else:
        raise ValueError("Unsupported grad_axis: {}".format(grad_axis))
    for n, off in enumerate(offsets):
        if grad_axis == 0:
            g[n, ...] = xp.roll(f, n_roll * off, axis=0) - f
            if deltas is not None:
                g[n, ...] /= deltas[n]
        elif grad_axis == -1:
            g[..., n] = xp.roll(f, n_roll * off, axis=0) - f
            if deltas is not None:
                g[..., n] /= deltas[n]
    if grad_axis in [0, "first"]:
        g = g.reshape((num_offsets,) + fshape, order="F")
    else:
        g = g.reshape(fshape + (num_offsets,), order="F")
    return g


def divergence_ravel_offsets(
    g,
    direction="forward",
    deltas=None,
    grad_axis="last",
    offsets=None,
    use_corners=False,
):
    xp, on_gpu = get_array_module(g)
    g = xp.asanyarray(g)

    # Note: direction here is opposite of the corresponding gradient_periodic
    if direction.lower() == "forward":
        n_roll = 1
    elif direction.lower() == "backward":
        n_roll = -1
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    otype = g.dtype.char
    if otype not in ["f", "d", "F", "D", "m", "M"]:
        otype = "d"

    if grad_axis in [0, "first"]:
        grad_axis = 0
        fshape = g.shape[1:]
    elif grad_axis in [-1, "last"]:
        grad_axis = -1
        fshape = g.shape[:-1]

    else:
        raise ValueError(
            "Unsupported grad_axis: {}".format(grad_axis)
            + "... must be first or last axis"
        )

    if offsets is None:
        offsets = compute_offsets(fshape, use_corners)
    n = len(offsets)

    if grad_axis in [0, "first"]:
        g = g.reshape((n, prod(fshape)), order="F")
    elif grad_axis in [-1, "last"]:
        g = g.reshape((prod(fshape), n), order="F")
    f = xp.empty(g.shape, dtype=otype)

    if deltas is not None:
        deltas = np.asanyarray(deltas)
        if len(deltas) != n:
            raise ValueError("deltas array length must match f.ndim")

    for n, off in enumerate(offsets):
        if grad_axis == 0:
            f[n, ...] = xp.roll(g[n, ...], n_roll * off, axis=0) - g[n, ...]
            if deltas is not None:
                f[n, ...] /= deltas[n]
        elif grad_axis == -1:
            f[..., n] = xp.roll(g[..., n], n_roll * off, axis=0) - g[..., n]
            if deltas is not None:
                f[..., n] /= deltas[n]
    div = -f.sum(axis=grad_axis)
    return div.reshape(fshape, order="F")


def divergence_periodic(
    g,
    direction="forward",
    axes=None,
    deltas=None,
    grad_axis="last",
    mode="periodic",
):
    xp, on_gpu = get_array_module(g)
    g = xp.asanyarray(g)
    n = g.ndim - 1  # number of dimensions

    # Note: direction here is opposite of the corresponding gradient_periodic
    if direction.lower() == "forward":
        diff_func = functools.partial(backward_diff, xp=xp)
    elif direction.lower() == "backward":
        diff_func = functools.partial(forward_diff, xp=xp)
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    if axes is None:
        axes = np.arange(n)
    else:
        if np.isscalar(axes):
            axes = (axes,)
        axes = np.asanyarray(axes)
        if axes.max() > (n - 1):
            raise ValueError(
                "maximum axis = {}, but f.ndim only {}".format(axes.max(), n)
            )

    if deltas is not None:
        try:
            if len(deltas) != len(axes):
                raise ValueError("deltas array length must match f.ndim")
        except TypeError:
            raise TypeError("deltas should be a sequence")

    otype = g.dtype.char
    if otype not in ["f", "d", "F", "D", "m", "M"]:
        otype = "d"

    f = xp.empty(g.shape, dtype=otype)

    if grad_axis in [0, "first"]:
        grad_axis = 0
    elif grad_axis in [-1, "last"]:
        grad_axis = -1
    else:
        raise ValueError("Unsupported grad_axis: {}".format(grad_axis))

    for n, axis in enumerate(axes):
        if grad_axis == 0:
            f[n, ...] = diff_func(g[n, ...], axis=axis)
            if deltas is not None:
                f[n, ...] /= deltas[n]
        elif grad_axis == -1:
            f[..., n] = diff_func(g[..., n], axis=axis)
            if deltas is not None:
                f[..., n] /= deltas[n]
        if mode == "edge":
            if grad_axis == 0:
                sl_axis = axis + 1
            else:
                sl_axis = axis
            sl_beg = _slice_at_axis(slice(0, 1), sl_axis)
            sl_end = _slice_at_axis(slice(-1, None), sl_axis)
            f[sl_beg] = -g[sl_end]
            f[sl_end] = g[sl_beg]

    div = f.sum(axis=grad_axis)
    return div


def gradient(f, dx=None, order=1, grad_axis="last"):
    """
    modified version of numpy's gradient function.  This differs in two ways
    1.) returns ndim+1 dimensional array instead of a list
          the extra dimension can be appended at either the start or end
    2.) if order=1, first-order differences are performed

    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior and second order accurate one-sides (forward or backwards)
    differences at the boundaries. The returned gradient hence has the same
    shape as the input array.

    Parameters
    ----------
    f : array_like
      An N-dimensional array containing samples of a scalar function.

    Returns
    -------
    gradient : ndarray
      N arrays of the same shape as `f` giving the derivative of `f` with
      respect to each dimension.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)
    >>> np.gradient(x)
    array([ 1. ,  1.5,  2.5,  3.5,  4.5,  5. ])
    >>> np.gradient(x, 2)
    array([ 0.5 ,  0.75,  1.25,  1.75,  2.25,  2.5 ])

    >>> np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=np.float))
    [array([[ 2.,  2., -1.],
           [ 2.,  2., -1.]]),
    array([[ 1. ,  2.5,  4. ],
           [ 1. ,  1. ,  1. ]])]

    >>> x = np.array([0,1,2,3,4])
    >>> dx = gradient(x)
    >>> y = x**2
    >>> gradient(y,dx)
    array([0.,  2.,  4.,  6.,  8.])
    """
    xp, on_gpu = get_array_module(f)
    f = xp.asanyarray(f)
    n = len(f.shape)  # number of dimensions
    if dx is not None:
        if np.isscalar(dx):
            dx = [dx] * n
        else:
            dx = list(dx)

    if order < 1 or order > 2:
        raise ValueError("Only first or second order differences supported")
    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    # create slice objects --- initially all are [:, :, ..., :]
    # output has additional dimension for the differences
    slice1 = [slice(None)] * (n + 1)
    slice2 = [slice(None)] * (n + 1)
    slice3 = [slice(None)] * (n + 1)
    slice4 = [slice(None)] * (n + 1)

    otype = f.dtype.char
    if otype not in ["f", "d", "F", "D", "m", "M"]:
        otype = "d"

    # Difference of datetime64 elements results in timedelta64
    if otype == "M":
        # Need to use the full dtype name because it contains unit information
        otype = f.dtype.name.replace("datetime", "timedelta")
    elif otype == "m":
        # Needs to keep the specific units, can't be a general unit
        otype = f.dtype

    # Convert datetime64 data into ints. Make dummy variable `y`
    # that is a view of ints if the data is datetime64, otherwise
    # just set y equal to the the array `f`.
    if f.dtype.char in ["M", "m"]:
        y = f.view("int64")
    else:
        y = f

    shape_orig = y.shape
    if grad_axis in [0, "first"]:
        grad_axis = 0
        out = xp.empty((n,) + y.shape, dtype=otype)
        y = y[np.newaxis, ...]
    elif grad_axis in [-1, "last"]:
        grad_axis = -1
        out = xp.empty(y.shape + (n,), dtype=otype)
        y = y[..., np.newaxis]
    else:
        raise ValueError("Unsupported grad_axis: {}".format(grad_axis))
    slice2[grad_axis] = 0
    slice3[grad_axis] = 0
    slice4[grad_axis] = 0

    for axis in range(n):
        slice1[grad_axis] = axis
        if grad_axis == 0:
            out_axis = axis + 1
        else:
            out_axis = axis
        if y.shape[out_axis] < 2:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least two elements are required."
            )

        if order == 1:
            slice1[out_axis] = slice(0, -1)
            slice2[out_axis] = slice(1, None)
            slice3[out_axis] = slice(None, -1)
            # 1D equivalent -- out[0:-1] = y[1:] - y[:-1]
            out[slice1] = y[slice2] - y[slice3]

            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            slice1[out_axis] = -1
            slice2[out_axis] = -1
            slice3[out_axis] = -2
            out[slice1] = y[slice2] - y[slice3]

        elif order == 2:  # numpy.gradient's usual case
            # Numerical differentiation: 1st order edges, 2nd order interior
            if y.shape[out_axis] == 2:
                # Use first order differences for time data

                slice1[out_axis] = slice(1, -1)
                slice2[out_axis] = slice(2, None)
                slice3[out_axis] = slice(None, -2)
                # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
                out[slice1] = (y[slice2] - y[slice3]) / 2.0

                slice1[out_axis] = 0
                slice2[out_axis] = 1
                slice3[out_axis] = 0
                # 1D equivalent -- out[0] = (y[1] - y[0])
                out[slice1] = y[slice2] - y[slice3]

                slice1[out_axis] = -1
                slice2[out_axis] = -1
                slice3[out_axis] = -2
                # 1D equivalent -- out[-1] = (y[-1] - y[-2])
                out[slice1] = y[slice2] - y[slice3]

            # Numerical differentiation: 2st order edges, 2nd order interior
            else:
                # Use second order differences where possible

                slice1[out_axis] = slice(1, -1)
                slice2[out_axis] = slice(2, None)
                slice3[out_axis] = slice(None, -2)
                # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
                out[slice1] = (y[slice2] - y[slice3]) / 2.0

                slice1[out_axis] = 0
                slice2[out_axis] = 0
                slice3[out_axis] = 1
                slice4[out_axis] = 2
                # 1D equivalent -- out[0] = -(3*y[0] - 4*y[1] + y[2]) / 2.0
                out[slice1] = (
                    -(3.0 * y[slice2] - 4.0 * y[slice3] + y[slice4]) / 2.0
                )

                slice1[out_axis] = -1
                slice2[out_axis] = -1
                slice3[out_axis] = -2
                slice4[out_axis] = -3
                # 1D equivalent -- out[-1] = (3*y[-1] - 4*y[-2] + y[-3])
                out[slice1] = (
                    3.0 * y[slice2] - 4.0 * y[slice3] + y[slice4]
                ) / 2.0
        else:
            raise ValueError("Unsupported Order: {}".format(order))
        # divide by step size
        # outvals.append(out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice1[out_axis] = slice(None)
        slice2[out_axis] = slice(None)
        slice3[out_axis] = slice(None)
        slice4[out_axis] = slice(None)

        if dx is not None:
            slice1[grad_axis] = slice(axis, axis + 1)
            out[slice1] = out[slice1] / dx[axis]
            slice1[grad_axis] = slice(None)

    y.shape = shape_orig  # restore original shape
    return out
