from math import sqrt

import numpy as np
from scipy.fftpack import dctn, idctn

from mrrt.operators import LinearOperatorMulti
from mrrt.utils import fftshift, ifftshift, prod


class DCT_Operator(LinearOperatorMulti):
    """ n-dimensional Discrete Cosine Transform operator with optional sampling
    mask.
    """

    def __init__(
        self,
        arr_shape,
        order="F",
        arr_dtype=np.float32,
        use_FFT_shifts=True,
        ortho=False,
        debug=False,
        dct_axes=None,
        fftshift_axes=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        arr_shape : int
            shape of the array
        order : {'C','F'}, optional
            array ordering that will be assumed if inputs/outputs need to be
            reshaped
        arr_dtype : numpy.dtype, optional
            dtype for the array
        use_FFT_shifts : bool, optional
            If False, do not apply any FFT shifts
        dct_axes : tuple or None, optional
            Specify a subset of the axes to transform.  The default is to
            transform all axes.
        fftshift_axes : tuple or None, optional
            Specify a subset of the axes to fftshift.  The default is to
            shift all axes.
        debug : bool, optional

        Additional Parameters
        ---------------------
        nd_input : bool, optional
        nd_output : bool, optional

        """
        if isinstance(arr_shape, (np.ndarray, list)):
            # retrieve shape from array
            arr_shape = tuple(arr_shape)

        self.arr_shape = arr_shape
        self.ndim = len(arr_shape)
        self.order = order
        self.use_FFT_shifts = use_FFT_shifts
        self.debug = debug

        nargin = prod(arr_shape)
        nargout = nargin

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            raise NotImplementedError("GPU version not implemented")

        # can specify a subset of the axes to perform the FFT/FFTshifts over
        self.dct_axes = dct_axes
        if self.dct_axes is None:
            self.dct_axes = tuple(np.arange(self.ndim))
        if fftshift_axes is None:
            self.fftshift_axes = self.dct_axes
        else:
            self.fftshift_axes = fftshift_axes

        # output of FFTs will be complex, regardless of input type
        self.result_dtype = np.result_type(arr_dtype, np.complex64)
        self.ortho = ortho
        if self.ortho:
            # sqrt of product of shape along axes where FFT is performed
            if self.dct_axes is None:
                self.scale_ortho = sqrt(nargin)
            else:
                self.scale_ortho = sqrt(
                    prod(np.asarray(self.arr_shape)[np.asarray(self.fft_axes)])
                )
        else:
            self.scale_ortho = None

        matvec_allows_repetitions = kwargs.pop(
            "matvec_allows_repetitions", True
        )
        squeeze_reps = kwargs.pop("squeeze_reps", True)
        nd_input = kwargs.pop("nd_input", False)
        nd_output = kwargs.pop("nd_output", False)
        if nd_output:
            shape_out = self.arr_shape
        else:
            shape_out = (nargout, 1)

        shape_in = self.arr_shape

        # mask_out = self.sample_mask
        mask_in = kwargs.pop("mask_in", None)
        mask_out = kwargs.pop("mask_out", None)

        self.dctn = dctn
        self.idctn = idctn

        matvec = self.forward
        matvec_adj = self.adjoint

        super(DCT_Operator, self).__init__(
            nargin=nargin,
            nargout=nargout,
            matvec=matvec,
            matvec_transp=matvec_adj,
            matvec_adj=matvec_adj,
            nd_input=nd_input,
            nd_output=nd_output,
            shape_in=shape_in,
            shape_out=shape_out,
            order=self.order,
            matvec_allows_repetitions=matvec_allows_repetitions,
            squeeze_reps=squeeze_reps,
            mask_in=mask_in,
            mask_out=mask_out,
            symmetric=False,  # TODO: set properly
            hermetian=False,  # TODO: set properly
            dtype=self.result_dtype,
            **kwargs,
        )

    # @profile
    def _adjoint_single_rep(self, y):
        if self.use_FFT_shifts:
            y = ifftshift(y, axes=self.fftshift_axes)
        x = self.idctn(y, axes=self.dct_axes)
        if self.use_FFT_shifts:
            x = fftshift(x, axes=self.fftshift_axes)
        return x

    def adjoint(self, y):
        nreps = int(y.size / self.shape[0])
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
            x = np.zeros(shape_tmp, dtype=np.result_type(y, np.complex64))
            if self.order == "C":
                for rep in range(nreps):
                    x[rep, ...] = self._adjoint_single_rep(y[rep, ...])
            else:
                for rep in range(nreps):
                    x[..., rep] = self._adjoint_single_rep(y[..., rep])
        if self.ortho:
            x *= self.scale_ortho
        if x.dtype != self.result_dtype:
            x = x.astype(self.result_dtype)
        return x

    # @profile
    def _forward_single_rep(self, x):
        y = np.zeros(
            self.shape_in,
            dtype=np.result_type(x, np.complex64),
            order=self.order,
        )
        if self.use_FFT_shifts:
            x = ifftshift(x, axes=self.fftshift_axes)
        y = self.dctn(x, axes=self.dct_axes)
        if self.use_FFT_shifts:
            y = fftshift(y, axes=self.fftshift_axes)
        return y

    def forward(self, x):
        if x.size < self.nargin:
            raise ValueError("data, x, too small to transform.")
        elif x.size == self.nargin:
            # 1D or single repetition nD
            if x.shape != self.shape_in:
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
            y = np.zeros(
                shape_tmp,
                dtype=np.result_type(x, np.complex64),
                order=self.order,
            )
            for rep in range(nreps):
                if self.order == "C":
                    y[rep, ...] = self._forward_single_rep(x[rep, ...]).reshape(
                        y.shape[1:], order=self.order
                    )
                else:
                    y[..., rep] = self._forward_single_rep(x[..., rep]).reshape(
                        y.shape[:-1], order=self.order
                    )
        if self.ortho:
            y /= self.scale_ortho

        if y.dtype != self.result_dtype:
            y = y.astype(self.result_dtype)

        return y
