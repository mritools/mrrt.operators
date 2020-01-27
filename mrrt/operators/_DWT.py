import abc
import warnings

import numpy as np

try:
    from pyframelets import separable as sep

    # from pywt._utils import _wavelets_per_axis, _modes_per_axis
    from pyframelets.separable.filterbanks.analysis_and_synthesis import (
        _prep_filterbank,
        _modes_per_axis,
    )
    from pyframelets._utils import _prep_axes_nd, is_nonstring_sequence
except ImportError:
    # don't raise error if PyFramelets is not available
    pass

from mrrt.operators import LinearOperatorMulti
from mrrt.operators.mixins import PriorMixin

# Note:  _prior_add and _prior_subtract methods provided by PriorMixin

from mrrt.utils import embed, masker, prod, profile

from mrrt.utils import config

if config.have_cupy:
    import cupy


"""
   Random shifts are to achieve translation invariance as described in:

   Figueiredo, M.A.T. and Nowak, R.D. An EM Algorithm for Wavelet-Based Image
   Restoration. IEEE Trans. Image Process. 2003; 12(8):906-916
"""


def _get_shifts(xshape, shift_range=None, rstate=None):
    ndim = len(xshape)
    if shift_range is None:
        shift_range = np.min(xshape)
    if rstate is None:
        shifts = shift_range * (np.random.rand(ndim) - 0.5)
    else:
        shifts = shift_range * (rstate.rand(ndim) - 0.5)
    shifts = shifts.astype(np.int)
    return shifts


def _circ_shift(x, shifts, xp):
    for d in range(x.ndim):
        if shifts[d] != 0:
            x = xp.roll(x, shifts[d], axis=d)
    return x


def _circ_unshift(x, shifts, xp):
    for d in range(x.ndim):
        if shifts[d] != 0:
            x = xp.roll(x, -shifts[d], axis=d)
    return x


class Framelet_Operator(abc.ABC, PriorMixin, LinearOperatorMulti):
    def __init__():
        raise NotImplementedError()

    def new_random_shift(self):
        # shifts for the transformed axes
        tmp_shifts = _get_shifts(self.axes_shape, self.shift_range)
        # shift of zero on all axes not transformed
        self.shifts = np.zeros(self.ndim, dtype=int)
        self.shifts[np.asarray(self.axes)] = tmp_shifts

    @abc.abstractmethod
    def _forward1(self, x):
        """Forward framelet transform of a single input."""
        pass

    @abc.abstractmethod
    def _adjoint1(self, coeffs):
        """Adjoint framelet transform of a single input."""
        pass

    def forward(self, x):
        xp = self.xp
        if x.size % self.nargin != 0:
            raise ValueError("shape mismatch for forward DWT")
        if (self.mask_in is not None) and (not self.nd_input):
            x = embed(x, self.mask_in, order=self.order)
        nrepetitions = x.size // self.nargin
        if nrepetitions > 1:

            if self.order == "C":
                y = xp.zeros(
                    (nrepetitions,) + self.coeff_arr_shape, dtype=x.dtype
                )
                for rep in range(nrepetitions):
                    y[rep, ...] = self._forward1(x[rep, ...])
            else:
                y = xp.zeros(
                    self.coeff_arr_shape + (nrepetitions,), dtype=x.dtype
                )
                for rep in range(nrepetitions):
                    y[..., rep] = self._forward1(x[..., rep])
        else:
            y = self._forward1(x)
        if (self.mask_out is not None) and (not self.nd_output):
            y = masker(y, self.mask_out, order=self.order)
        return y

    def adjoint(self, coeffs):
        xp = self.xp
        if coeffs.size % self.nargout != 0:
            raise ValueError("shape mismatch for adjoint DWT")
        nrepetitions = coeffs.size // self.nargout
        if (self.mask_out is not None) and (not self.nd_output):
            coeffs = embed(coeffs, self.mask_out, order=self.order)
            coeffs = coeffs.ravel(order=self.order)
        if nrepetitions > 1:
            if self.order == "C":
                x = xp.zeros(
                    (nrepetitions,) + self.arr_shape, dtype=coeffs.dtype
                )
                for rep in range(nrepetitions):
                    x[rep, ...] = self._adjoint1(coeffs[rep, ...])
            else:
                x = xp.zeros(
                    self.arr_shape + (nrepetitions,), dtype=coeffs.dtype
                )
                for rep in range(nrepetitions):
                    x[..., rep] = self._adjoint1(coeffs[..., rep])
        else:
            x = self._adjoint1(coeffs)
        if (self.mask_in is not None) and (not self.nd_input):
            x = masker(x, self.mask_in, order=self.order)
        return x


class MDWT_Operator(Framelet_Operator):
    """ multilevel discrete wavelet transform operators

    Works for both real and complex inputs
    """

    def __init__(
        self,
        arr_shape,
        order="F",
        arr_dtype=np.float32,
        filterbank=None,
        mode="symmetric",
        level=None,
        prior=None,
        force_real=False,
        random_shift=False,
        autopad=False,
        autopad_mode="symmetric",
        axes=None,
        **kwargs,
    ):
        """ TODO: fix docstring

        Parameters
        ----------
        arr_shape : int
            shape of the array to filter
        degree : int
            degree of the directional derivatives to apply
        order : {'C','F'}
            array ordering that will be assumed if inputs/outputs need to be
            reshaped
        arr_dtype : numpy.dtype
            dtype for the filter coefficients
        prior : array_like
            subtract this prior before computing the DWT
        force_real : bool
            subtract this prior before computing the DWT
        random_shift : bool
            Random shifts can be introduced to achieve translation invariance
            as described in [1]_
        autopad : bool
            If True, the array will be padded from ``arr_shape`` up to the
            nearest integer multiple of 2**``level`` prior to the transform.
            The padding will be removed upon the adjoint transform.
        autopad_mode : str
            The mode for `numpy.pad` to use when ``autopad`` is True.

        References
        ----------
        ..[1] MAT Figueiredo and RD Nowak.
              An EM Algorithm for Wavelet-Based Image Restoration.
              IEEE Trans. Image Process. 2003; 12(8):906-916

        """

        if isinstance(arr_shape, np.ndarray):
            # retrieve shape from array
            arr_shape = arr_shape.shape

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        self.prior = prior
        self.force_real = force_real
        self.random_shift = random_shift

        # determine axes and the shape of the axes to be transformed
        self.axes, self.axes_shape, self.ndim_transform = _prep_axes_nd(
            self.arr_shape, axes
        )

        if self.random_shift:
            self.shifts = np.zeros(self.ndim, dtype=np.int)

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            xp = cupy
            on_gpu = True
        else:
            xp = np
            on_gpu = False
        self._on_gpu = on_gpu

        self.autopad = autopad
        self.autopad_mode = autopad_mode
        if not is_nonstring_sequence(filterbank):
            filterbank = [filterbank] * len(self.axes)
        self.filterbank = _prep_filterbank(filterbank, self.axes, xp=xp)

        max_level = sep.dwt_max_level(
            self.arr_shape, self.filterbank, self.axes
        )
        # default to 4 levels unless otherwise specified or array is too small
        if level is None:
            level = min(4, max_level)
        elif not self.autopad:
            if level > max_level:
                level = max_level
                warnings.warn(
                    "level exceeds max level for the size of the "
                    "input.  reducing level to {}".format(level)
                )
        self.level = level
        mode = _modes_per_axis(mode, self.axes)

        if self.autopad:
            min_factors = np.asarray(
                [
                    fb.sizes[0][0] * fb.bw_decimation[0] ** level
                    for fb in self.filterbank
                ]
            )
            arr_shape = np.asarray(arr_shape)
            pad_shape = arr_shape.copy()  # includes non-transformed axes
            pad_shape[np.asarray(self.axes)] = min_factors * np.ceil(
                np.asarray(self.axes_shape) / min_factors
            )
            pad_shape = pad_shape.astype(int)
            self.pad_width = [(0, s) for s in (pad_shape - arr_shape)]
            self.pad_shape = tuple(pad_shape)
        else:
            self.pad_shape = self.arr_shape

        self.mode = mode
        self.truncate_coefficients = False

        self.shift_range = 1 << self.level

        if True:
            # determine the shape of the coefficient arrays
            # TODO: determine without running the transform
            coeffs_tmp = sep.wavedecn(
                xp.ones(self.pad_shape, dtype=np.float32),
                filterbank=self.filterbank,
                mode=self.mode,
                level=self.level,
                axes=self.axes,
            )
            (
                coeff_arr,
                self.coeff_arr_slices,
                self.coeff_arr_shapes,
            ) = sep.ravel_coeffs(coeffs_tmp, axes=self.axes, xp=xp)
            self.coeff_arr_shape = coeff_arr.shape

        nargin = prod(self.arr_shape)
        nargout = prod(self.coeff_arr_shape)

        # output of DWT may be complex, depending on input type
        self.result_dtype = np.result_type(arr_dtype, np.float32)
        self.coeffs = None

        matvec_allows_repetitions = kwargs.pop(
            "matvec_allows_repetitions", True
        )
        squeeze_reps = kwargs.pop("squeeze_reps", True)
        nd_input = kwargs.pop("nd_input", False)
        nd_output = kwargs.pop("nd_output", False)
        if nd_output:
            raise ValueError(
                "nd_output = True is not supported. All framelet coefficients "
                "will be stacked into a 1d array"
            )
        self.mask_in = kwargs.pop("mask_in", None)
        if self.mask_in is not None:
            nargin = self.mask_in.sum()
        self.mask_out = kwargs.pop("mask_out", None)
        if self.mask_out is not None:
            nargout = self.mask_out.sum()

        super(Framelet_Operator, self).__init__(
            nargin=nargin,
            nargout=nargout,
            matvec=self.forward,
            matvec_transp=self.adjoint,
            matvec_adj=self.adjoint,
            nd_input=nd_input,
            nd_output=nd_output,
            shape_in=self.arr_shape,
            shape_out=self.coeff_arr_shape,
            order=self.order,
            matvec_allows_repetitions=matvec_allows_repetitions,
            squeeze_reps=squeeze_reps,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            symmetric=False,  # TODO: set properly
            hermetian=False,  # TODO: set properly
            dtype=self.result_dtype,
            **kwargs,
        )

    #    def thresh(self, x, mu):
    #        if self.random_shift:
    #            self.new_random_shift()
    #        x = _forward1(x)
    #        x = soft_thresh(x, mu)
    #        x = _adjoint1(x)

    @profile
    def _forward1(self, x):
        """PyWavelets version of forward DWT."""
        xp = self.xp
        if (x.ndim > 1) and ((x.shape != self.shape_in) and (x.shape[1] > 1)):
            raise ValueError("multiple reps not currently supported")
        # regardless of whether operator returns 1D or nD, must reshape to nD
        # internally
        # have to reshape to nD before calling wavedecn
        x = x.reshape(self.shape_in, order=self.order)
        x = self._prior_subtract(x)

        if self.random_shift:
            # For now, make user manually call new_random_shift to change the
            # shifts
            # self.new_random_shift()
            x = _circ_shift(x, self.shifts, self.xp)

        if self.autopad:
            # pad up to appropriate size
            x = xp.pad(x, self.pad_width, mode=self.autopad_mode)

        coeffs = sep.wavedecn(
            x,
            filterbank=self.filterbank,
            mode=self.mode,
            level=self.level,
            axes=self.axes,
        )

        (
            coeffs_arr,
            self.coeff_arr_slices,
            self.coeff_arr_shapes,
        ) = sep.ravel_coeffs(coeffs, axes=self.axes, xp=self.xp)
        self.coeff_arr_shape = coeffs_arr.shape

        # mask of the coarest level approximation coefficients
        self.approx_mask = xp.zeros(coeffs_arr.shape, dtype=np.bool)
        self.approx_mask[self.coeff_arr_slices[0]] = 1
        # # mask of all detail coefficient locations
        self.detail_mask = ~self.approx_mask

        if self.force_real and np.iscomplexobj(coeffs_arr):
            return coeffs_arr.real

        return coeffs_arr

    @profile
    def _adjoint1(self, coeffs):
        """PyWavelets version of adjoint DWT."""
        xp = self.xp
        if isinstance(coeffs, xp.ndarray):
            if self.force_real:
                coeffs = coeffs.real

            if (coeffs.ndim > 1) and (
                (coeffs.shape != self.shape_in) and (coeffs.shape[1] > 1)
            ):
                raise ValueError("multiple reps not currently supported")

            # have to reshape to nD before calling wavedecn
            # coeffs = coeffs.reshape(self.shape_out, order=self.order)
            coeffs = sep.unravel_coeffs(
                coeffs,
                self.coeff_arr_slices,
                self.coeff_arr_shapes,
                output_format="wavedecn",
                xp=self.xp,
            )
        x = sep.waverecn(
            coeffs, filterbank=self.filterbank, mode=self.mode, axes=self.axes
        )

        # trim excess boundary from reconstructed signal if necessary
        # this also takes care of removing any padding that was added
        x = x[tuple([slice(sz) for sz in self.arr_shape])]

        if self.random_shift:
            x = _circ_unshift(x, shifts=self.shifts, xp=self.xp)
        x = self._prior_add(x)
        return x


class FSDWT_Operator(Framelet_Operator):
    """ multilevel discrete wavelet transform operators

    Works for both real and complex inputs
    """

    def __init__(
        self,
        arr_shape,
        order="F",
        arr_dtype=np.float32,
        filterbank=None,
        mode="symmetric",
        level=None,
        prior=None,
        force_real=False,
        random_shift=False,
        autopad=False,
        autopad_mode="symmetric",
        axes=None,
        **kwargs,
    ):
        """ TODO: fix docstring

        Parameters
        ----------
        arr_shape : int
            shape of the array to filter
        degree : int
            degree of the directional derivatives to apply
        order : {'C','F'}
            array ordering that will be assumed if inputs/outputs need to be
            reshaped
        arr_dtype : numpy.dtype
            dtype for the filter coefficients
        prior : array_like
            subtract this prior before computing the DWT
        force_real : bool
            subtract this prior before computing the DWT
        random_shift : bool
            Random shifts can be introduced to achieve translation invariance
            as described in [1]_
        autopad : bool
            If True, the array will be padded from ``arr_shape`` up to the
            nearest integer multiple of 2**``level`` prior to the transform.
            The padding will be removed upon the adjoint transform.
        autopad_mode : str
            The mode for `numpy.pad` to use when ``autopad`` is True.

        References
        ----------
        ..[1] MAT Figueiredo and RD Nowak.
              An EM Algorithm for Wavelet-Based Image Restoration.
              IEEE Trans. Image Process. 2003; 12(8):906-916

        """
        if isinstance(arr_shape, np.ndarray):
            # retrieve shape from array
            arr_shape = arr_shape.shape

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        self.prior = prior
        self.force_real = force_real
        self.random_shift = random_shift

        # determine axes and the shape of the axes to be transformed
        self.axes, self.axes_shape, self.ndim_transform = _prep_axes_nd(
            self.arr_shape, axes
        )

        if self.random_shift:
            self.shifts = np.zeros(self.ndim, dtype=np.int)

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            xp = cupy
            on_gpu = True
        else:
            xp = np
            on_gpu = False
        self._on_gpu = on_gpu

        self.autopad = autopad
        self.autopad_mode = autopad_mode
        if not is_nonstring_sequence(filterbank):
            filterbank = [filterbank] * len(self.axes)
        self.filterbank = _prep_filterbank(filterbank, self.axes, xp=xp)

        max_levels = []
        for ax, fb in zip(self.axes, self.filterbank):
            max_levels.append(sep.dwt_max_level(self.arr_shape, fb, axes=ax))

        if level is None or xp.isscalar(level):
            level = [level] * len(self.axes)
        if len(level) != len(self.axes):
            raise ValueError("level must match the length of the axes list")

        for n in range(len(level)):
            if level[n] is None:
                # default to 4 levels unless otherwise specified
                level[n] = min(4, max_levels[n])
            elif not self.autopad:
                if level[n] > max_levels[n]:
                    level[n] = max_levels[n]
                    warnings.warn(
                        "level exceeds max level for the size of the "
                        "input along axis {}. Reducing level to {} for this "
                        "axis".format(self.axes[n], level[n])
                    )
        self.level = level
        mode = _modes_per_axis(mode, self.axes)

        if self.autopad:
            min_factors = np.asarray(
                [
                    fb.sizes[0][0] * fb.bw_decimation[0] ** level
                    for fb in self.filterbank
                ]
            )
            arr_shape = np.asarray(arr_shape)
            pad_shape = arr_shape.copy()  # includes non-transformed axes
            pad_shape[np.asarray(self.axes)] = min_factors * np.ceil(
                np.asarray(self.axes_shape) / min_factors
            )
            pad_shape = pad_shape.astype(int)
            self.pad_width = [(0, s) for s in (pad_shape - arr_shape)]
            self.pad_shape = tuple(pad_shape)
        else:
            self.pad_shape = self.arr_shape

        self.mode = mode
        self.truncate_coefficients = False

        # TODO: define per-axis shift_range instead
        self.shift_range = 1 << np.min(self.level)

        if True:
            # determine the shape of the coefficient arrays
            # TODO: determine without running the transform
            coeffs_tmp = sep.fswavedecn(
                xp.ones(self.pad_shape, dtype=np.float32),
                filterbank=self.filterbank,
                mode=self.mode,
                level=self.level,
                axes=self.axes,
            )
            self.coeff_arr_shape = coeffs_tmp._coeffs.shape

        nargin = prod(self.arr_shape)
        nargout = prod(self.coeff_arr_shape)

        # output of DWT may be complex, depending on input type
        self.result_dtype = np.result_type(arr_dtype, np.float32)
        self.coeffs = None

        matvec_allows_repetitions = kwargs.pop(
            "matvec_allows_repetitions", True
        )
        squeeze_reps = kwargs.pop("squeeze_reps", True)
        nd_input = kwargs.pop("nd_input", False)
        nd_output = kwargs.pop("nd_output", False)
        if nd_output:
            raise ValueError(
                "nd_output = True is not supported. All framelet coefficients "
                "will be stacked into a 1d array"
            )
        self.mask_in = kwargs.pop("mask_in", None)
        if self.mask_in is not None:
            nargin = self.mask_in.sum()
        self.mask_out = kwargs.pop("mask_out", None)
        if self.mask_out is not None:
            nargout = self.mask_out.sum()

        super(Framelet_Operator, self).__init__(
            nargin=nargin,
            nargout=nargout,
            matvec=self.forward,
            matvec_transp=self.adjoint,
            matvec_adj=self.adjoint,
            nd_input=nd_input,
            nd_output=nd_output,
            shape_in=self.arr_shape,
            shape_out=self.coeff_arr_shape,
            order=self.order,
            matvec_allows_repetitions=matvec_allows_repetitions,
            squeeze_reps=squeeze_reps,
            mask_in=self.mask_in,
            mask_out=self.mask_out,
            symmetric=False,  # TODO: set properly
            hermetian=False,  # TODO: set properly
            dtype=self.result_dtype,
            **kwargs,
        )

    #    def thresh(self, x, mu):
    #        if self.random_shift:
    #            self.new_random_shift()
    #        x = _forward1(x)
    #        x = soft_thresh(x, mu)
    #        x = _adjoint1(x)

    @profile
    def _forward1(self, x):
        """PyWavelets version of forward DWT."""
        xp = self.xp
        if (x.ndim > 1) and ((x.shape != self.shape_in) and (x.shape[1] > 1)):
            raise ValueError("multiple reps not currently supported")
        # regardless of whether operator returns 1D or nD, must reshape to nD
        # internally
        # have to reshape to nD before calling wavedecn
        x = x.reshape(self.shape_in, order=self.order)
        x = self._prior_subtract(x)

        if self.random_shift:
            # For now, make user manually call new_random_shift to change the
            # shifts
            # self.new_random_shift()
            x = _circ_shift(x, self.shifts, self.xp)

        if self.autopad:
            # pad up to appropriate size
            x = xp.pad(x, self.pad_width, mode=self.autopad_mode)

        coeffs = sep.fswavedecn(
            x,
            filterbank=self.filterbank,
            mode=self.mode,
            level=self.level,
            axes=self.axes,
        )

        self.transform_results = coeffs
        self.coeffs_arr_shape = self.transform_results._coeffs.shape

        copy_coeffs = False
        if copy_coeffs:
            coeffs_arr = self.transform_results._coeffs.copy()
        else:
            # extract the coefficients array from the transform_results class
            coeffs_arr = self.transform_results._coeffs
            self.transform_results._coeffs = None

        # mask of the coarest level approximation coefficients
        self.approx_mask = xp.zeros(coeffs_arr.shape, dtype=np.bool)
        self.approx_mask[
            tuple(sl[0] for sl in self.transform_results.coeff_slices)
        ] = 1
        # # mask of all detail coefficient locations
        self.detail_mask = ~self.approx_mask

        if self.force_real and np.iscomplexobj(coeffs_arr):
            return coeffs_arr.real

        return coeffs_arr

    @profile
    def _adjoint1(self, coeffs):
        """PyWavelets version of adjoint DWT."""
        xp = self.xp
        if isinstance(coeffs, xp.ndarray):
            if self.force_real:
                coeffs = coeffs.real

            if (coeffs.ndim > 1) and (
                (coeffs.shape != self.shape_in) and (coeffs.shape[1] > 1)
            ):
                raise ValueError("multiple reps not currently supported")

            coeffs = coeffs.reshape(self.coeffs_arr_shape, order=self.order)

            # restore coefficients to the transform_results class
            self.transform_results._coeffs = coeffs
        x = sep.fswaverecn(self.transform_results)

        # trim excess boundary from reconstructed signal if necessary
        # this also takes care of removing any padding that was added
        x = x[tuple([slice(sz) for sz in self.arr_shape])]

        if self.random_shift:
            x = _circ_unshift(x, shifts=self.shifts, xp=self.xp)
        x = self._prior_add(x)
        return x
