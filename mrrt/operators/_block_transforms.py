from itertools import product

import numpy as np

from scipy.fftpack import dctn, idctn
from mrrt.utils import fftn, ifftn
from mrrt.operators._cycle_spinning import cycle_spin
from mrrt.utils import prod


def block_dctn(
    x, block_size=(8, 8), reshape_output=False, dct_kwargs=dict(norm="ortho")
):
    """ block-wise dctn
    x.shape must be an integer multiple of by block_size
    any axis where block_size = 1 will not be transformed
    if reshape_output is false, the output will be shape
    # x.shape//block_size + block_size
    otherwise, the output is reshaped to match the shape of the input
    """
    return block_transform(
        x,
        dctn,
        block_size=block_size,
        reshape_output=reshape_output,
        transform_kwargs=dct_kwargs,
    )


def block_fftn(
    x, block_size=(8, 8), reshape_output=False, fft_kwargs=dict(norm="ortho")
):
    """ block-wise dctn
    x.shape must be an integer multiple of by block_size
    any axis where block_size = 1 will not be transformed
    if reshape_output is false, the output will be shape
    # x.shape//block_size + block_size
    otherwise, the output is reshaped to match the shape of the input
    """
    return block_transform(
        x,
        fftn,
        block_size=block_size,
        reshape_output=reshape_output,
        transform_kwargs=fft_kwargs,
    )


def block_transform_prepad(
    x, block_size=(8, 8, 1), pad_mode="wrap", **pad_kwargs
):
    """Pad x up to the largest integer multiple of block_size

    Returns
    -------
    xpad : np.ndarray
        The padded array.
    npads : tuple
        The number of samples appended to each axis in ``x``.
    """
    if x.ndim != len(block_size):
        raise ValueError("block_size must match size of x")
    xshape = np.asarray(x.shape)
    new_shape = [int(b * np.ceil(s / b)) for s, b in zip(x.shape, block_size)]
    new_shape = np.asarray(new_shape)
    npads = new_shape - xshape
    if np.sum(npads) == 0:
        return x, npads
    xp = np.pad(
        x, pad_width=[(0, p) for p in npads], mode=pad_mode, **pad_kwargs
    )
    return xp, tuple(npads)


def block_transform(
    x, transform, block_size=(8, 8), reshape_output=False, transform_kwargs={}
):
    """ block-wise dctn
    x.shape must be an integer multiple of by block_size
    any axis where block_size = 1 will not be transformed
    if reshape_output is false, the output will be shape
    # x.shape//block_size + block_size
    otherwise, the output is reshaped to match the shape of the input
    """
    from skimage.util.shape import view_as_blocks

    if x.ndim != len(block_size):
        raise ValueError("block_size must match size of x")
    ndim = x.ndim
    block_axes = np.where(np.asarray(block_size) > 1)[0]
    dct_axes = block_axes + ndim
    xview = view_as_blocks(x, block_size)
    y = transform(xview, axes=dct_axes, **transform_kwargs)
    if not reshape_output:
        return y
    else:
        # TODO: implement faster inverse of view_as_blocks
        # tmp2 = idctn(tmp, axes=(0, 1))
        out = np.zeros(x.shape)
        y_axes = np.arange(ndim)
        y_slices = [slice(None)] * (y.ndim)
        out_slices = [slice(None)] * ndim
        if prod(block_size) > prod(y.shape[:ndim]):
            # size of each block exceeds the number of blocks
            for ells in product(*([range(s) for s in y.shape[:ndim]])):
                for n, ax, b in zip(ells, y_axes, block_size):
                    out_slices[ax] = slice(n * b, (n + 1) * b)
                    y_slices[ax] = slice(n, n + 1)
                out[tuple(out_slices)] = y[tuple(y_slices)]
        else:
            # number of blocks exceeds the size of each block
            for ells in product(*([range(s) for s in block_size])):
                for n, ax, b in zip(ells, y_axes, block_size):
                    out_slices[ax] = slice(n, None, b)
                    y_slices[ndim + ax] = slice(n, n + 1)
                out[tuple(out_slices)] = np.squeeze(y[tuple(y_slices)])
    return out


def block_idctn(
    x, block_size=(8, 8), reshape_input=False, dct_kwargs=dict(norm="ortho")
):
    """ block-wise dctn
    x.shape must be an integer multiple of by block_size
    any axis where block_size = 1 will not be transformed

    if reshape_output is false, it assumes the input is from block_dctn with
    reshape_output=False.
    """
    return block_inverse_transform(
        x,
        idctn,
        block_size=block_size,
        reshape_input=reshape_input,
        transform_kwargs=dct_kwargs,
    )


def block_ifftn(
    x, block_size=(8, 8), reshape_input=False, fft_kwargs=dict(norm="ortho")
):
    """ block-wise dctn
    x.shape must be an integer multiple of by block_size
    any axis where block_size = 1 will not be transformed

    if reshape_output is false, it assumes the input is from block_dctn with
    reshape_output=False.
    """
    return block_inverse_transform(
        x,
        ifftn,
        block_size=block_size,
        reshape_input=reshape_input,
        transform_kwargs=fft_kwargs,
    )


def block_inverse_transform(
    x,
    inverse_transform,
    block_size=(8, 8),
    reshape_input=False,
    transform_kwargs={},
):
    """ block-wise dctn
    x.shape must be an integer multiple of by block_size
    any axis where block_size = 1 will not be transformed

    if reshape_output is false, it assumes the input is from block_dctn with
    reshape_output=False.
    """
    from skimage.util.shape import view_as_blocks

    axes = np.where(np.asarray(block_size) > 1)[0]
    if reshape_input:
        ndim = x.ndim
        if ndim != len(block_size):
            raise ValueError("block_size must match size of x")
        xview = view_as_blocks(x, block_size)
        out_shape = x.shape
    else:
        ndim = x.ndim // 2
        if ndim != len(block_size):
            raise ValueError("block_size must match size of x")
        xview = x
        out_shape = np.asarray(x.shape[:ndim]) * np.asarray(x.shape[ndim:])
    dct_axes = axes + ndim
    y = inverse_transform(xview, axes=dct_axes, **transform_kwargs)
    if not np.iscomplexobj(y):
        out = np.zeros(out_shape)
    else:
        out = np.zeros(out_shape, np.result_type(y.dtype, np.complex64))

    y_slices = [slice(None)] * (2 * ndim)
    out_slices = [slice(None)] * ndim
    y_axes = np.arange(ndim)
    if prod(block_size) > prod(y.shape[:ndim]):
        # size of each block exceeds the number of blocks
        for ells in product(*([range(s) for s in y.shape[:ndim]])):
            for n, ax, b in zip(ells, y_axes, block_size):
                out_slices[ax] = slice(n * b, (n + 1) * b)
                y_slices[ax] = slice(n, n + 1)
            out[tuple(out_slices)] = y[tuple(y_slices)]
    else:
        # number of blocks exceeds the size of each block
        for ells in product(*([range(s) for s in block_size])):
            for n, ax, b in zip(ells, y_axes, block_size):
                out_slices[ax] = slice(n, None, b)
                y_slices[ndim + ax] = slice(n, n + 1)
            out[tuple(out_slices)] = np.squeeze(y[tuple(y_slices)])
    return out


def block_dct_thresh(
    x, thresh, block_size=(16, 16), dct_kwargs=dict(norm="ortho")
):
    out = block_dctn(
        x, block_size=block_size, reshape_output=False, dct_kwargs=dct_kwargs
    )
    # TODO: allow choice of threshold.  It is hard thresholding for now.
    out[np.abs(out) < thresh] = 0
    return block_idctn(
        out, block_size=block_size, reshape_input=False, dct_kwargs=dct_kwargs
    )


def block_fft_thresh(x, thresh, block_size=(16, 16)):
    # TODO: allow choice of threshold.  It is hard thresholding for now.
    fft_kwargs = dict(norm="ortho")
    out = block_fftn(
        x, block_size=block_size, reshape_output=False, fft_kwargs=fft_kwargs
    )
    out[np.abs(out) < thresh] = 0
    if np.iscomplexobj(x):
        return block_ifftn(
            out,
            block_size=block_size,
            reshape_input=False,
            fft_kwargs=fft_kwargs,
        )
    else:
        return block_ifftn(
            out,
            block_size=block_size,
            reshape_input=False,
            fft_kwargs=fft_kwargs,
        ).real


def denoise_dctn(
    x,
    thresh,
    block_size=(16, 16),
    max_shifts=None,
    shift_steps=1,
    prepad=True,
    num_workers=1,
):
    # TODO: currently minimal benefit with num_workers > 1 because the scipy
    #       fftpack routines (dct/idct) do not release the GIL.
    orig_shape = x.shape
    if max_shifts is None:
        max_shifts = [(b - 1) for b in block_size]
        print(max_shifts)
    if prepad:
        # pads = [(b, b) for b in block_size]
        pads = [(b, 0) for b in block_size]
        x = np.pad(x, pads, mode="symmetric")
    y = cycle_spin(
        x,
        block_dct_thresh,
        max_shifts=max_shifts,
        shift_steps=shift_steps,
        block_size=block_size,
        thresh=thresh,
        num_workers=num_workers,
    )
    if prepad:
        # slices = [slice(s) for s in orig_shape]
        slices = [slice(b, b + s) for s, b in zip(orig_shape, block_size)]
        y = y[slices]
    return y


def denoise_fftn(
    x,
    thresh,
    block_size=(16, 16),
    max_shifts=None,
    shift_steps=1,
    prepad=True,
    num_workers=1,
):
    # TODO: currently minimal benefit with num_workers > 1 because the scipy
    #       fftpack routines (dct/idct) do not release the GIL.
    orig_shape = x.shape
    if max_shifts is None:
        max_shifts = [(b - 1) for b in block_size]
        print(max_shifts)
    if prepad:
        # pads = [(b, b) for b in block_size]
        pads = [(b, 0) for b in block_size]
        x = np.pad(x, pads, mode="symmetric")
    y = cycle_spin(
        x,
        block_fft_thresh,
        max_shifts=max_shifts,
        shift_steps=shift_steps,
        num_workers=num_workers,
        func_kw=dict(block_size=block_size, thresh=thresh),
    )
    if prepad:
        # slices = [slice(s) for s in orig_shape]
        slices = [slice(b, b + s) for s, b in zip(orig_shape, block_size)]
        y = y[slices]
    return y
