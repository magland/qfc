from typing import Callable, Union, Literal
import numpy as np

int_dtype = np.int16


def qfc_pre_compress(x: np.ndarray, *, quant_scale_factor: float):
    """
    Prepares an array for compression using the QFC algorithm This is an
    internal function, called by the codec, and should not be called directly.

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization, obtained from
        qfc_estimate_quant_scale_factor

    Returns
    -------
    np.ndarray
        The prepared array.
    """
    N = x.shape[0]
    N_is_even = N % 2 == 0
    qs = quant_scale_factor
    x_fft = np.fft.rfft(x, axis=0) / np.sqrt(
        x.shape[0]
    )  # we divide by the sqrt of the number of samples so that quant scale factor does not depend on the number of samples
    x_fft = np.ascontiguousarray(
        x_fft
    )  # This is important so the codec will behave properly
    x_fft_re = np.real(x_fft)
    x_fft_im = np.imag(x_fft)
    if N_is_even:
        x_fft_im = x_fft_im[1:-1]  # the first and last values are always zero
    else:
        x_fft_im = x_fft_im[1:]  # the first value is always zero
    x_fft_concat = np.concatenate([x_fft_re, x_fft_im], axis=0)
    x_fft_concat_quantized = np.round(x_fft_concat * qs).astype(int_dtype)
    assert x_fft_concat_quantized.shape[0] == N
    return x_fft_concat_quantized


def qfc_inv_pre_compress(x: np.ndarray, *, quant_scale_factor: float, dtype: str):
    """
    Inverts the preparation of an array for compression using the QFC algorithm.
    This is an internal function, called by the codec, and should not be called
    directly.

    Parameters
    ----------
    x : np.ndarray
        The prepared array
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qfc_estimate_quant_scale_factor
    dtype : str
        The data type string for the reconstructed array
    """
    qs = quant_scale_factor
    num_samples = x.shape[0]
    num_channels = x.shape[1] if len(x.shape) > 1 else 1
    x_fft_re = x[: (num_samples // 2 + 1), :] / qs
    x_fft_im = x[(num_samples // 2 + 1):, :] / qs
    if num_samples % 2 == 0:
        x_fft_im = np.concatenate(
            [np.zeros((1, num_channels)), x_fft_im, np.zeros((1, num_channels))], axis=0
        )
    else:
        x_fft_im = np.concatenate([np.zeros((1, num_channels)), x_fft_im], axis=0)
    x_fft = x_fft_re + 1j * x_fft_im
    # it's important to specify num_samples here in the case where it is odd
    # see https://stackoverflow.com/questions/52594392/numpy-fft-irfft-why-is-lena-necessary
    x = np.fft.irfft(x_fft, num_samples, axis=0) * np.sqrt(num_samples)
    x = np.ascontiguousarray(
        x
    )  # This is important so that the codec will behave properly!
    assert x.shape[0] == num_samples, f"{x.shape[0]} != {num_samples}"
    return x.astype(dtype)


def qfc_estimate_quant_scale_factor(
    x: np.ndarray,
    target_compression_ratio: Union[float, None] = None,
    target_residual_stdev: Union[float, None] = None,
    max_num_samples: int = 30000 * 3,
    compression_method: Literal["zstd", "zlib"] = "zlib",
    zstd_level: int = 3,
    zlib_level: int = 3,
):
    """
    Estimates the quantization scale factor for the QFC algorithm for a given
    target compression ratio or target residual standard deviation

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    target_compression_ratio : float or None
        The target compression ratio
        Exactly one of target_compression_ratio and target_residual_std must be specified
    target_residual_stdev : float or None
        The target residual standard deviation
        Exactly one of target_compression_ratio and target_residual_std must be specified
    max_num_samples : int
        The maximum number of samples to use for the estimation
    compression_method : str
        The compression method to use for the codec
    zstd_level : int
        The zstd compression level to use
    zlib_level : int
        The zlib compression level to use

    Returns
    -------
    float
        The quantization scale factor
    """
    from ..codecs.QFCCodec import QFCCodec

    if x.shape[0] > max_num_samples:
        x = x[:max_num_samples]

    def get_resid_stdev_for_quant_scale_factor(qs: float):
        codec = QFCCodec(
            quant_scale_factor=qs,
            dtype=x.dtype,
            segment_length=x.shape[0],
            compression_method=compression_method,
            zstd_level=zstd_level,
            zlib_level=zlib_level
        )
        compressed = codec.encode(x)
        y_reconstructed = codec.decode(compressed)
        y_resid = x - y_reconstructed
        return float(np.std(y_resid))

    def get_compression_ratio_for_quant_scale_factor(qs: float):
        codec = QFCCodec(
            quant_scale_factor=qs,
            dtype=x.dtype,
            segment_length=x.shape[0],
            compression_method=compression_method,
            zstd_level=zstd_level,
            zlib_level=zlib_level
        )
        compressed = codec.encode(x)
        return (x.size * 2) / len(compressed)

    if target_residual_stdev is not None:
        qs = _monotonic_binary_search(
            get_resid_stdev_for_quant_scale_factor,
            target_value=target_residual_stdev,
            max_iterations=100,
            tolerance=0.001,
            ascending=False
        )
    elif target_compression_ratio is not None:
        qs = _monotonic_binary_search(
            get_compression_ratio_for_quant_scale_factor,
            target_value=target_compression_ratio,
            max_iterations=100,
            tolerance=0.001,
            ascending=False
        )
    else:
        raise ValueError("Exactly one of target_compression_ratio and target_residual_std must be specified")

    return qs


def _monotonic_binary_search(
    func: Callable[[float], float],
    target_value: float,
    max_iterations: int,
    tolerance: float,
    ascending: bool,
):
    """
    Performs a binary search to find the value that minimizes the difference
    between the output of a function and a target value

    Parameters
    ----------
    func : callable
        The function to minimize
        assumed to be a monotonically increasing function
    target_value : float
        The target value
    max_iterations : int
        The maximum number of iterations
    tolerance : float
        The tolerance for the difference between the output of the function
        and the target value
    ascending : bool
        Whether the function is monotonically increasing or decreasing

    Returns
    -------
    float
        The value that minimizes the difference between the output of the
        function and the target value
    """
    effective_target_value = target_value
    if not ascending:
        effective_target_value = -target_value
    num_iterations = 0
    # first find an upper bound
    upper_bound = 1
    last_val = None
    while True:
        new_val = func(upper_bound)
        if not ascending:
            new_val = -new_val
        if new_val > effective_target_value:
            break
        upper_bound *= 2
        num_iterations += 1
        if num_iterations > max_iterations:
            return upper_bound
        if last_val is not None:
            if new_val < last_val:
                # fails to be monotonically increasing
                break
        last_val = new_val
    # then do a binary search
    lower_bound = 0
    while upper_bound - lower_bound > tolerance:
        candidate = (upper_bound + lower_bound) / 2
        candidate_value = func(candidate)
        if not ascending:
            candidate_value = -candidate_value
        if candidate_value < effective_target_value:
            lower_bound = candidate
        else:
            upper_bound = candidate
        num_iterations += 1
        if num_iterations > max_iterations:
            break
    return (upper_bound + lower_bound) / 2
