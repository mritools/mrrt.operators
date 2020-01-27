"""
Total Variation Operator.

The operator is discretized using finite differinces with periodic boundary
conditions.

see:
Discrete Regularization by Dengyong Zhou and Bernhard Scholkopf
http://www.msr-waypoint.com/en-us/um/people/denzho/papers/DR.pdf



"""
import functools

import numpy as np
from mrrt.operators import LinearOperatorMulti
from mrrt.operators.mixins import PriorMixin
from mrrt.operators._FiniteDifference import (
    gradient_periodic,
    divergence_periodic,
    gradient_ravel_offsets,
    divergence_ravel_offsets,
    compute_offsets,
)
from mrrt.operators.LinOp import cupy_ndarray_type
from mrrt.utils import get_array_module, prod


# Note:  _prior_add and _prior_subtract methods provided by PriorMixin

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


def TVnorm(grad, grad_axis=-1, alpha=0):
    """ TVnorm.  This is the isotropic TV norm (L2 norm) """
    xp, on_gpu = get_array_module(grad)
    if xp.iscomplexobj(grad):
        gradsq = xp.real(grad * xp.conj(grad))
    else:
        gradsq = grad * grad
    if alpha == 0:
        return xp.sqrt(gradsq.sum(axis=grad_axis))
    else:
        return xp.sqrt(gradsq.sum(axis=grad_axis) + alpha)


class TV_Operator(PriorMixin, LinearOperatorMulti):
    """ Total variation operator. """

    def __init__(
        self,
        arr_shape,
        weight=1,
        norm=1,
        prior=None,
        order="F",
        arr_dtype=np.float32,
        tv_type="iso",
        grid_size=None,
        nd_input=True,
        nd_output=True,
        squeeze_reps=True,
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
            If provided, subtract the prior before computing the TV.
        arr_dtype : numpy.dtype
            dtype for the filter coefficients
        tv_type : {'iso', 'aniso'}
            Select whether isotropic or anisotropic (l1) TV is used
        grid_size : array, optional
            size of the grid along each dimension of the image.  defaults to
            ones.  (e.g. needed if the voxels are anisotropic.)
        """
        if isinstance(arr_shape, np.ndarray):
            # retrieve shape from array
            arr_shape = arr_shape.shape

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        self.grid_size = grid_size
        self.tv_type = tv_type
        if tv_type == "aniso":
            # TODO
            raise ValueError("Not implemented")

        self.offsets = compute_offsets(tuple(arr_shape), use_corners=False)
        if axes is not None:
            if np.isscalar(axes):
                axes = (axes,)
            axes = np.asarray(axes)
            if len(axes) > self.ndim:
                raise ValueError("invalid number of axes")
            if np.any(np.abs(axes) > self.ndim):
                raise ValueError("invalid axis")
            axes = axes % self.ndim
            self.offsets = np.asarray(self.offsets)[axes].tolist()

        self.num_offsets = len(self.offsets)

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

        self.weight = weight
        self.prior = prior
        self.norm_p = norm
        self._limit = 1e-6

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            import cupy

            xp = cupy
        else:
            xp = np

        if self.order == "C":
            grad_axis = 0
        else:
            grad_axis = -1
        self.grad_axis = grad_axis

        if axes is not None:
            # customized offsets
            if True:
                # non-periodic, slightly faster
                self.grad_func = functools.partial(
                    gradient_ravel_offsets, offsets=self.offsets
                )
                self.div_func = functools.partial(
                    divergence_ravel_offsets, offsets=self.offsets
                )
            else:
                # periodic
                self.grad_func = functools.partial(gradient_periodic, axes=axes)
                self.div_func = functools.partial(
                    divergence_periodic, axes=axes
                )
        else:
            self.grad_func = functools.partial(gradient_periodic)
            self.div_func = functools.partial(divergence_periodic)

        self.mask_in = kwargs.pop("mask_in", None)
        if self.mask_in is not None:
            nargin = self.mask_in.sum()
            if isinstance(nargin, cupy_ndarray_type):
                nargin = nargin.get()
            nd_input = True  # TODO: fix LinOp to remove need for this.  why does DWT case not need it?
        self.mask_out = kwargs.pop("mask_out", None)
        if self.mask_out is not None:
            # probably wrong if order = 'C'
            if order == "C":
                stack_axis = 0
            else:
                stack_axis = -1
            self.mask_out = xp.stack(
                [self.mask_out] * self.num_offsets, stack_axis
            )
            nargout = self.mask_out.sum()
            if isinstance(nargout, cupy_ndarray_type):
                nargout = nargout.get()
            nd_output = True

        super(TV_Operator, self).__init__(
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

    @property
    def norm_p(self):
        "weight to be applied to the gradient"
        return self._norm_p

    @norm_p.setter
    def norm_p(self, norm_p):
        if norm_p < 1 or norm_p > 2:
            raise ValueError("norm must be in range: [1 2]")
        self._norm_p = norm_p

    def forward(self, x):
        """ image gradient """
        xp, on_gpu = get_array_module(x)
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
        xp, on_gpu = get_array_module(g)
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
        # d = self._prior_add(d)  # don't think prior should be added back here
        return -d  # adjoint of grad is -div

    def gradient(self, x):
        """ gradient of the TV norm """
        xp, on_gpu = get_array_module(x)
        Tf = self * x  # self.forward(x)
        term1 = self.norm_p * Tf
        if self.norm_p == 1:
            term2 = 1 / xp.sqrt(Tf * xp.conj(Tf) + self._limit)
        elif self.norm_p == 2:
            term2 = 1
        else:
            p = self.norm_p / 2 - 1
            term2 = (Tf * xp.conj(Tf) + self._limit) ** p

        return self.weight * (self.H * (term1 * term2))

    def opnorm(self, x, aniso=False, alpha=0):
        """ L1 TV norm """
        xp, on_gpu = get_array_module(x)
        if aniso:
            # TODO : check this
            nrm = self.weight * xp.linalg.norm(self.forward(x), ord=1, axis=-1)
        else:
            nrm = self.weight * TVnorm(
                self.forward(x), alpha=alpha, grad_axis=self.grad_axis
            )
        return nrm

    def magnitude(self, x, aniso=False, alpha=0):
        xp, on_gpu = get_array_module(x)
        """ summed TV norm """
        return self.weight * xp.sum(self.opnorm(x, aniso=aniso, alpha=alpha))
