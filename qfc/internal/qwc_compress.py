from typing import Union
import numpy as np


def qwc_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    pywt_wavelet: str,
    pywt_level: Union[int, None],
    pywt_mode: str
):
    """
    Prepares an array for compression using the QWC algorithm This is an
    internal function, called by the codec, and should not be called directly.

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization, obtained from
        qwc_estimate_quant_scale_factor
    pywt_wavelet : str
        The wavelet parameter for PyWavelets
    pywt_level : int or None
        The level parameter for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets

    Returns
    -------
    np.ndarray
        The quantized wavelet coefficients
    list
        The sizes of the wavelet coefficients
    """
    coeffs, coeff_sizes = pywt_compress_timeseries(x, pywt_wavelet=pywt_wavelet, pywt_level=pywt_level, pywt_mode=pywt_mode)
    qs = quant_scale_factor
    coeffs_quantized = np.round(coeffs * qs).astype(
        np.int16
    )
    return coeffs_quantized, coeff_sizes


def qwc_inv_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    coeff_sizes: list,
    dtype: str,
    pywt_wavelet: str,
    pywt_mode: str
):
    """
    Inverts the preparation of an array for compression using the QWC algorithm.
    This is an internal function, called by the codec, and should not be called
    directly.

    Parameters
    ----------
    x : np.ndarray
        The prepared array
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qwc_estimate_quant_scale_factor
    coeff_sizes : list
        The sizes of the wavelet coefficients
    dtype : str
        The data type string for the reconstructed array
    pywt_wavelet: str
        The wavelet type for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    """
    qs = quant_scale_factor
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    coeffs_unquantized = x.astype(np.float32) / qs
    timeseries = pywt_decompress_timeseries(coeffs_unquantized, coeff_sizes=coeff_sizes, pywt_wavelet=pywt_wavelet, pywt_mode=pywt_mode)
    return timeseries.astype(dtype)


def pywt_compress_timeseries(x: np.ndarray, *, pywt_wavelet: str, pywt_level: Union[int, None], pywt_mode: str):
    """
    Compresses a timeseries using PyWavelets

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    pywt_wavelet : str
        The wavelet parameter for PyWavelets
    pywt_level : int or None
        The level parameter for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    """
    import pywt
    cc = pywt.wavedec(x, wavelet=pywt_wavelet, level=pywt_level, mode=pywt_mode, axis=0)
    coeff_sizes = [c.shape[0] for c in cc]
    coeffs_concat = np.concatenate(cc, axis=0)

    return coeffs_concat, coeff_sizes


def pywt_decompress_timeseries(coeffs_concat: np.ndarray, *, coeff_sizes, pywt_wavelet: str, pywt_mode: str):
    """
    Decompresses a timeseries using PyWavelets

    Parameters
    ----------
    coeffs_concat : np.ndarray
        The concatenated wavelet coefficients
    coeff_sizes : list
        The sizes of the wavelet coefficients
    pywt_wavelet: str
        The wavelet type for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    """
    import pywt
    coeffs = []
    i = 0
    for size in coeff_sizes:
        coeffs.append(coeffs_concat[i:i + size])
        i += size
    return pywt.waverec(
        coeffs, wavelet=pywt_wavelet, mode=pywt_mode, axis=0
    )
