from functools import partial
from math import sqrt

import numpy as np

from mrrt.operators import LinearOperatorMulti
from mrrt.utils import config, masker, embed, prod, profile
from mrrt.utils import fftn, ifftn

if config.have_pyfftw:
    from mrrt.utils import build_fftn, build_ifftn  # TODO: change import

if config.have_cupy:
    import cupy


class FFT_Operator(LinearOperatorMulti):
    """ n-dimensional Fourier Transform operator with optional sampling mask.
    """

    def __init__(
        self,
        arr_shape,
        order="F",
        arr_dtype=np.float32,
        use_fft_shifts=True,
        sample_mask=None,
        ortho=False,
        force_real_image=False,
        debug=False,
        preplan_pyfftw=True,
        pyfftw_threads=None,
        fft_axes=None,
        fftshift_axes=None,
        planner_effort="FFTW_ESTIMATE",
        disable_warnings=False,
        im_mask=None,
        rel_fov=None,
        **kwargs,
    ):
        """Cartesian MRI Operator  (with partial FFT and coil maps).

        Parameters
        ----------
        arr_shape : int
            shape of the array
        order : {'C','F'}, optional
            array ordering that will be assumed if inputs/outputs need to be
            reshaped
        arr_dtype : numpy.dtype, optional
            dtype for the array
        sample_mask : array_like, optional
            boolean mask of which FFT coefficients to keep
        ortho : bool, optional
            if True, change the normalizeation to the orthogonal case
        preplan_pyfftw : bool, optional
            if True, precompute the pyFFTW plan upon object creation
        pyfftw_threads : int, optional
            number of threads to be used by pyFFTW.  defaults to
            multiprocessing.cpu_count() // 2.
        use_fft_shifts : bool, optional
            If False, do not apply any FFT shifts
        fft_axes : tuple or None, optional
            Specify a subset of the axes to transform.  The default is to
            transform all axes.
        fftshift_axes : tuple or None, optional
            Specify a subset of the axes to fftshift.  The default is to
            shift all axes.
        im_mask : ndarray or None, optional
            Image domain mask
        force_real_image : bool, optional
        debug : bool, optional

        Additional Parameters
        ---------------------
        nd_input : bool, optional
        nd_output : bool, optional

        """
        if isinstance(arr_shape, (np.ndarray, list)):
            # retrieve shape from array
            arr_shape = tuple(arr_shape)
        if not isinstance(arr_shape, tuple):
            raise ValueError("expected array_shape to be a tuple or list")

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        self.use_fft_shifts = use_fft_shifts
        self.disable_warnings = disable_warnings

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            xp = cupy
            on_gpu = True
        else:
            xp = np
            on_gpu = False
        self._on_gpu = on_gpu

        if sample_mask is not None:
            # masking faster if continguity of mask matches
            if self.order == "F":
                sample_mask = xp.asfortranarray(sample_mask)
            elif self.order == "C":
                sample_mask = xp.ascontiguousarray(sample_mask)
            else:
                raise ValueError("order must be C or F")
        self.sample_mask = sample_mask

        self.force_real_image = force_real_image
        self.debug = debug
        if self.sample_mask is not None:
            if sample_mask.shape != arr_shape:
                raise ValueError("sample mask shape must match arr_shape")
            # make sure it is boolean
            self.sample_mask = self.sample_mask > 0
            # prestore raveled mask indices to save time later during masking
            # self.sample_mask_idx = xp.where(
            #     self.sample_mask.ravel(order=self.order)
            # )

        # can specify a subset of the axes to perform the FFT/FFTshifts over
        self.fft_axes = fft_axes
        if self.fft_axes is None:
            self.fft_axes = tuple(range(self.ndim))

        if fftshift_axes is None:
            self.fftshift_axes = self.fft_axes
        else:
            self.fftshift_axes = fftshift_axes

        # configure scaling  (e.g. unitary operator or not)
        self.ortho = ortho
        if self.fft_axes is None:
            Ntrans = prod(self.arr_shape)
        else:
            Ntrans = prod(np.asarray(self.arr_shape)[np.asarray(self.fft_axes)])
        if self.ortho:
            # sqrt of product of shape along axes where FFT is performed
            self.scale_ortho = sqrt(Ntrans)
            self.gpu_scale_inverse = 1  # self.scale_ortho / Ntrans
            # self.gpu_scale_forward = self.scale_ortho
        else:
            self.scale_ortho = None
            self.gpu_scale_inverse = 1 / Ntrans

        if "mask_out" in kwargs:
            raise ValueError(
                "This operator specifies `mask_out` via the "
                "parameter `sample_mask"
            )

        if ("mask_in" in kwargs) or ("mask_out" in kwargs):
            raise ValueError(
                "This operator specifies `mask_in` via the "
                "parameter `im_mask"
            )

        if im_mask is not None:
            if im_mask.shape != arr_shape:
                raise ValueError("im_mask shape mismatch")
            if order != "F":
                raise ValueError("only order='F' supported for im_mask case")
            nargin = im_mask.sum()
            self.im_mask = im_mask
        else:
            nargin = prod(arr_shape)
            self.im_mask = None

        if sample_mask is not None:
            nargout = sample_mask.sum()
        else:
            nargout = nargin
        nargout = int(nargout)

        # output of FFTs will be complex, regardless of input type
        self.result_dtype = np.result_type(arr_dtype, np.complex64)

        matvec_allows_repetitions = kwargs.pop(
            "matvec_allows_repetitions", True
        )
        squeeze_reps = kwargs.pop("squeeze_reps", True)
        nd_input = kwargs.pop("nd_input", False)
        nd_output = kwargs.pop("nd_output", False)

        if (self.sample_mask is not None) and nd_output:
            raise ValueError("cannot have both nd_output and sample_mask")
        if nd_output:
            shape_out = self.arr_shape
        else:
            shape_out = (nargout, 1)

        shape_in = self.arr_shape

        self.have_pyfftw = config.have_pyfftw
        if self.on_gpu:
            self.preplan_pyfftw = False
        else:
            self.preplan_pyfftw = preplan_pyfftw if self.have_pyfftw else False
            if self.preplan_pyfftw:
                self._preplan_fft(pyfftw_threads, planner_effort)
                # raise ValueError("Implementation Incomplete")

        if self.on_gpu:
            self.fftn = partial(fftn, xp=cupy)
            self.ifftn = partial(ifftn, xp=cupy)
        else:
            if self.preplan_pyfftw:
                self._preplan_fft(pyfftw_threads, planner_effort)
            else:
                self.fftn = partial(fftn, xp=np)
                self.ifftn = partial(ifftn, xp=np)
        self.fftshift = xp.fft.fftshift
        self.ifftshift = xp.fft.ifftshift

        self.rel_fov = rel_fov

        self.mask = None  # TODO: implement or remove (expected by CUDA code)
        matvec = self.forward
        matvec_adj = self.adjoint
        self.norm_available = True
        self.norm = self._norm

        super(FFT_Operator, self).__init__(
            nargin=nargin,
            nargout=nargout,
            matvec=matvec,
            matvec_transp=matvec_adj,
            matvec_adj=matvec_adj,
            nd_input=nd_input or (im_mask is not None),
            nd_output=nd_output,
            shape_in=shape_in,
            shape_out=shape_out,
            order=self.order,
            matvec_allows_repetitions=matvec_allows_repetitions,
            squeeze_reps=squeeze_reps,
            mask_in=im_mask,
            mask_out=None,  # mask_out,
            symmetric=False,  # TODO: set properly
            hermetian=False,  # TODO: set properly
            dtype=self.result_dtype,
            **kwargs,
        )

    def _preplan_fft(self, pyfftw_threads=None, planner_effort="FFTW_MEASURE"):
        """ Use FFTW builders to pre-plan the FFT for faster repeated
        calls. """
        if pyfftw_threads is None:
            import multiprocessing

            pyfftw_threads = max(1, multiprocessing.cpu_count())
        self.pyfftw_threads = pyfftw_threads
        a_b = np.empty(self.arr_shape, dtype=self.result_dtype)
        self.fftn = build_fftn(
            a_b,
            axes=self.fft_axes,
            threads=pyfftw_threads,
            overwrite_input=False,
            planner_effort=planner_effort,
        )
        self.ifftn = build_ifftn(
            a_b,
            axes=self.fft_axes,
            threads=pyfftw_threads,
            overwrite_input=False,
            planner_effort=planner_effort,
        )
        del a_b

    @profile
    def _adjoint_single_rep(self, y):
        if self.use_fft_shifts:
            y = self.ifftshift(y, axes=self.fftshift_axes)
        if self.preplan_pyfftw:
            x = self.ifftn(y)
        else:
            x = self.ifftn(y, axes=self.fft_axes)
        if self.use_fft_shifts:
            x = self.fftshift(x, axes=self.fftshift_axes)
        return x

    @profile
    def adjoint(self, y):
        # TODO: add test for this case and the coils + sample_mask case
        # if y.ndim == 1 or y.shape[-1] == 1:
        xp = self.xp
        nreps = int(y.size / self.shape[0])
        if self.sample_mask is not None:
            if y.ndim == 1 and self.ndim > 1:
                y = y[:, np.newaxis]
            nmask = self.sample_mask.sum()
            if y.shape[0] != nmask:
                if self.order == "C":
                    y = y.reshape((-1, nmask), order=self.order)
                else:
                    y = y.reshape((nmask, -1), order=self.order)
            y = embed(y, mask=self.sample_mask, order=self.order)
        if nreps == 1:
            # 1D or single repetition or single coil nD
            y = y.reshape(self.shape_in, order=self.order)

            x = self._adjoint_single_rep(y)
        else:
            if self.order == "C":
                shape_tmp = (nreps,) + self.shape_in
            else:
                shape_tmp = self.shape_in + (nreps,)

            y = y.reshape(shape_tmp, order=self.order)
            x = xp.zeros(shape_tmp, dtype=xp.result_type(y, np.complex64))
            if self.order == "C":
                for rep in range(nreps):
                    x[rep, ...] = self._adjoint_single_rep(y[rep, ...])
            else:
                for rep in range(nreps):
                    x[..., rep] = self._adjoint_single_rep(y[..., rep])

        #        if self.im_mask:
        #            x = masker(x, self.im_mask, order=self.order)

        if self.ortho:
            x *= self.scale_ortho
        if x.dtype != self.result_dtype:
            x = x.astype(self.result_dtype)
        if self.force_real_image:
            x = x.real.astype(self.result_dtype)
        return x

    @profile
    def _forward_single_rep(self, x):
        xp = self.xp
        y = xp.zeros(
            self.shape_in,
            dtype=xp.result_type(x, xp.complex64),
            order=self.order,
        )
        if self.use_fft_shifts:
            x = self.ifftshift(x, axes=self.fftshift_axes)
        if self.preplan_pyfftw:
            y = self.fftn(x)
        else:
            y = self.fftn(x, axes=self.fft_axes)
        if self.use_fft_shifts:
            y = self.fftshift(y, axes=self.fftshift_axes)

        if self.sample_mask is not None:
            # y = y[self.sample_mask]
            y = masker(
                y,
                mask=self.sample_mask,
                order=self.order,
                # mask_idx_ravel=self.sample_mask_idx,
            )
        return y

    @profile
    def forward(self, x):
        xp = self.xp
        if self.force_real_image:
            x = x.real
        if self.im_mask is None:
            size_1rep = self.nargin
        else:
            size_1rep = prod(self.shape_in)
        if x.size < size_1rep:
            raise ValueError("data, x, too small to transform.")
        elif x.size == size_1rep:
            nreps = 1
            # 1D or single repetition nD
            x = x.reshape(self.shape_in, order=self.order)
            y = self._forward_single_rep(x)
        else:
            if self.order == "C":
                nreps = x.shape[0]
                shape_tmp = (nreps,) + self.shape_in
            else:
                nreps = x.shape[-1]
                shape_tmp = self.shape_in + (nreps,)
            x = x.reshape(shape_tmp, order=self.order)
            if self.sample_mask is not None:
                # number of samples by number of repetitions
                if self.order == "C":
                    shape_y = (nreps, self.nargout)
                else:
                    shape_y = (self.nargout, nreps)
                y = xp.zeros(
                    shape_y,
                    dtype=xp.result_type(x, np.complex64),
                    order=self.order,
                )
            else:
                if self.order == "C":
                    shape_y = (nreps,) + self.shape_in
                else:
                    shape_y = self.shape_in + (nreps,)
                y = xp.zeros(
                    shape_y,
                    dtype=xp.result_type(x, np.complex64),
                    order=self.order,
                )
            if self.order == "C":
                for rep in range(nreps):
                    y[rep, ...] = self._forward_single_rep(x[rep, ...]).reshape(
                        y.shape[1:], order=self.order
                    )
            else:
                for rep in range(nreps):
                    y[..., rep] = self._forward_single_rep(x[..., rep]).reshape(
                        y.shape[:-1], order=self.order
                    )

        if self.ortho:
            y /= self.scale_ortho

        if y.dtype != self.result_dtype:
            y = y.astype(self.result_dtype)

        return y

    def _norm_single_rep(self, x):
        # chain forward and adjoint together for a single repetition
        y = self._forward_single_rep(x)
        if self.sample_mask is not None:
            y = embed(y, mask=self.sample_mask, order=self.order, xp=self.xp)
        y = y.reshape(self.shape_in, order=self.order)
        x = self._adjoint_single_rep(y)
        return x

    @profile
    def _norm(self, x):
        # forward transform, immediately followed by inverse transform
        # slightly faster than calling self.adjoint(self.forward(x))
        xp = self.xp
        if self.force_real_image:
            x = x.real
        if self.im_mask is None:
            size_1rep = self.nargin
        else:
            size_1rep = prod(self.shape_in)
        if x.size < size_1rep:
            raise ValueError("data, x, too small to transform.")
        elif x.size == size_1rep:
            nreps = 1
            # 1D or single repetition nD
            x = x.reshape(self.shape_in, order=self.order)
            y = self._norm_single_rep(x)
        else:
            nreps = x.size // size_1rep
            if self.order == "C":
                x = x.reshape((nreps,) + self.shape_in, order=self.order)
                y = xp.zeros_like(x)
                for rep in range(nreps):
                    y[rep, ...] = self._norm_single_rep(x[rep, ...]).reshape(
                        y.shape[1:], order=self.order
                    )
            else:
                x = x.reshape(self.shape_in + (nreps,), order=self.order)
                y = xp.zeros_like(x)
                for rep in range(nreps):
                    y[..., rep] = self._norm_single_rep(x[..., rep]).reshape(
                        y.shape[:-1], order=self.order
                    )

        if y.dtype != self.result_dtype:
            y = y.astype(self.result_dtype)

        if not self.nd_input:
            if self.squeeze_reps_in and (nreps == 1):
                y = xp.ravel(y, order=self.order)
            else:
                if self.order == "C":
                    y = y.reshape((nreps, -1), order=self.order)
                else:
                    y = y.reshape((-1, nreps), order=self.order)

        return y
