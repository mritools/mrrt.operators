import numpy as np
from types import ModuleType
from mrrt.utils import config, get_array_module

if config.have_cupy:
    import cupy


class PriorMixin(object):
    prior = None

    def _prior_subtract(self, x):
        """Subtract a prior from x."""
        if self.prior is None:
            return x
        else:
            if self.prior.shape != x.shape:
                self.prior = self.prior.reshape(x.shape, self.order)
            return x - self.prior

    def _prior_add(self, x):
        """Add a prior to x."""
        if self.prior is None:
            return x
        else:
            if self.prior.shape != x.shape:
                self.prior = self.prior.reshape(x.shape, self.order)
            return x + self.prior

    # TODO: remove
    def _prior_from_mean(self, x, axis):
        """Create a prior based on the mean of x along axis."""
        if hasattr(self, "xp"):
            xp = self.xp
        else:
            xp = np
        self.prior = xp.mean(x, axis=axis, keepdims=True)


class GpuCheckerMixin:
    """Mixin to determine CPU vs GPU backend.

    This mix-in adds a method, ``_check_gpu`` that is meant to be used in
    the constructor of a class todetermine if the object operates on the CPU
    or GPU. ``_check_gpu`` sets the ``on_gpu`` property which is a boolean
    indicating if the object operates on the GPU.

    """

    def _check_gpu(self, xp):
        """Initialize ``on_gpu`` class property.

        Parameters
        ----------
        xp : module, numpy.ndarray, cupy.ndarray or str
            ``xp`` should be the numpy or cupy module. It can also be a
            numpy or cupy ndarray. Finally, one can specify the string
            "numpy" or "cupy" as well.
        """
        self._on_gpu = False
        if isinstance(xp, ModuleType):
            if xp != np:
                self._on_gpu = True
        elif hasattr(xp, "__array_interface__") or hasattr(
            xp, "__cuda_array_interface__"
        ):
            xp, self._on_gpu = get_array_module(xp)
        elif xp == "numpy":
            pass
        elif xp == "cupy":
            if not config.have_cupy:
                raise ValueError("Cannot select cupy: CuPy is unavailable.")
            self._on_gpu = True
        else:
            raise ValueError("xp must be a module, 'numpy', or 'cupy'")

    @property
    def on_gpu(self):
        """Boolean indicating whether the filterbank is a GPU filterbank."""
        return self._on_gpu

    @on_gpu.setter
    def on_gpu(self, on_gpu):
        if on_gpu in [True, False]:
            self._on_gpu = on_gpu
        else:
            raise ValueError("on_gpu must be True or False")

    @property
    def xp(self):
        """The ndarray module being used (module: cupy or numpy)."""
        if self._on_gpu:
            return cupy
        else:
            return np
