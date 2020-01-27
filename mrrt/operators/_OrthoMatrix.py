import numpy as np

from mrrt.operators import LinearOperatorMulti
from mrrt.utils import config, prod

from pyframelets.restoration._utils import dct_matrix, transform_channels

if config.have_cupy:
    import cupy

"""
TODO:
    Lingala2011 kt-SLR
        http://user.engineering.uiowa.edu/~jcb/Software/ktslr_matlab/Software.html
    Candes2013 et. al.  Unbiased Risk Estimates for Singular Value Thresholding and Spectral Estimators
        http://statweb.stanford.edu/~candes/SURE/index.html
"""


class OrthoMatrixOperator(LinearOperatorMulti):
    """ n-dimensional Discrete Cosine Transform operator with optional sampling
    mask.
    """

    def __init__(
        self,
        arr_shape,
        axis,
        m,
        order="F",
        arr_dtype=np.float32,
        debug=False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        arr_shape : int
            shape of the array
        axis : int
            The axis to transform
        m : np.ndarray
            Orthogonal matrix representing the transform.
        order : {'F', 'C'}
            The memory layout ordering of the data.
        arr_dtype: np.dtype
            The dtype of the array to be transformed.
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

        if axis > self.ndim or axis < (-self.ndim + 1):
            raise ValueError("invalid axis")
        else:
            axis = axis % self.ndim
        self.axis = axis

        self.debug = debug

        if "loc_in" in kwargs and kwargs["loc_in"] == "gpu":
            if not config.have_cupy:
                raise ImportError("CuPy not found")
            xp = cupy
        else:
            xp = np

        if m == "dct":
            # default is a DCT transform
            m = dct_matrix(self.arr_shape[self.axis], xp=xp)
        else:
            m = xp.asarray(m)
        self.m = m

        if self.m.dtype.kind == "c":
            m_inv = xp.conj(self.m).H
        else:
            m_inv = m.T
        self.m_inv = m_inv

        nargin = prod(arr_shape)
        nargout = nargin

        self.result_dtype = xp.result_type(arr_dtype, m.dtype)

        # self.ortho = True
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

        matvec = self.forward
        matvec_adj = self.adjoint

        super(OrthoMatrixOperator, self).__init__(
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
        return transform_channels(
            y, self.m_inv, channel_axis=self.axis, xp=self.xp
        )

    def adjoint(self, y):
        xp = self.xp
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
            x = xp.zeros(shape_tmp, dtype=xp.result_type(y, np.float32))
            if self.order == "C":
                for rep in range(nreps):
                    x[rep, ...] = self._adjoint_single_rep(y[rep, ...])
            else:
                for rep in range(nreps):
                    x[..., rep] = self._adjoint_single_rep(y[..., rep])
        if x.dtype != self.result_dtype:
            x = x.astype(self.result_dtype)
        return x

    # @profile
    def _forward_single_rep(self, x):
        return transform_channels(x, self.m, channel_axis=self.axis, xp=self.xp)

    def forward(self, x):
        xp = self.xp
        if x.size < self.nargin:
            raise ValueError("data, x, too small to transform.")
        elif x.size == self.nargin:
            # 1D or single repetition nD
            if x.shape != self.shape_in:
                x = x.reshape(self.shape_in, order=self.order)
            y = self._forward_single_rep(x)
        else:
            # could just do in one call instead, but this way should require
            # less memory
            if self.order == "C":
                nreps = x.shape[0]
                shape_tmp = (nreps,) + self.shape_in
            else:
                nreps = x.shape[-1]
                shape_tmp = self.shape_in + (nreps,)

            x = x.reshape(shape_tmp, order=self.order)
            y = xp.zeros(
                shape_tmp,
                dtype=xp.result_type(x, xp.float32),
                order=self.order,
            )
            if self.order == "C":
                for rep in range(nreps):
                    y[rep, ...] = self._forward_single_rep(x[rep, ...])
            else:
                for rep in range(nreps):
                    y[..., rep] = self._forward_single_rep(x[..., rep])

        if y.dtype != self.result_dtype:
            y = y.astype(self.result_dtype)

        return y
