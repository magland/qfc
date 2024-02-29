from typing import Callable, Union, Literal
import numpy as np
import zlib

def qfc_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float
):
    """
    Prepares an array for compression using the QFC algorithm

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qfc_estimate_quant_scale_factor

    Returns
    -------
    np.ndarray
        The prepared array
    """
    qs = quant_scale_factor
    x_fft = np.fft.rfft(x, axis=0) / np.sqrt(x.shape[0])  # we divide by the sqrt of the number of samples so that quant scale factor does not depend on the number of samples
    x_fft = np.ascontiguousarray(x_fft)  # This is important so the codec will behave properly
    x_fft_re = np.real(x_fft)
    x_fft_im = np.imag(x_fft)
    x_fft_im = x_fft_im[1:-1]  # the first and last values are always zero
    x_fft_concat = np.concatenate([x_fft_re, x_fft_im], axis=0)
    x_fft_concat_quantized = np.round(x_fft_concat * qs).astype(
        np.int16
    )
    return x_fft_concat_quantized

def qfc_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    compression_method: Literal["zlib", "zstd"] = "zlib",
    zstd_level: int = 3,
    zlib_level: int = 3
):
    """
    Compresses an array using the QFC algorithm

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qfc_estimate_quant_scale_factor
    compression_method : str
        The compression method to use, either "zlib" or "zstd"
    zstd_level : int
        The compression level to use if compression_method is "zstd"
    zlib_level : int
        The compression level to use if compression_method is "zlib"

    Returns
    -------
    bytes
        The compressed array as bytes
    """
    x_fft_concat_quantized = qfc_pre_compress(x, quant_scale_factor=quant_scale_factor)
    if compression_method == "zlib":
        compressed_bytes = zlib.compress(x_fft_concat_quantized.tobytes(), level=zlib_level)
    elif compression_method == "zstd":
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(
            level=zstd_level
        )
        compressed_bytes = cctx.compress(x_fft_concat_quantized.tobytes())
    else:
        raise ValueError("compression_method must be 'zlib' or 'zstd'")
    return compressed_bytes


def qfc_inv_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    dtype: str
):
    """
    Inverts the preparation of an array for compression using the QFC algorithm

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
    x_fft_im = np.concatenate(
        [np.zeros((1, num_channels)), x_fft_im, np.zeros((1, num_channels))],
        axis=0
    )
    x_fft = x_fft_re + 1j * x_fft_im
    x = np.fft.irfft(x_fft, axis=0) * np.sqrt(num_samples)
    x = np.ascontiguousarray(x)  # This is important so that the codec will behave properly!
    return x.astype(dtype)


def qfc_decompress(
    compressed_bytes: bytes, *,
    quant_scale_factor: float,
    num_channels: int,
    dtype: str,
    compression_method: Literal["zlib", "zstd"] = "zlib"
):
    """
    Decompresses an array using the QFC algorithm

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed array
    quant_scale_factor : float
        The quantization scale factor used during compression
    num_channels : int
        The number of channels
    dtype : str
        Data type string for the reconstructed array
    compression_method : str
        The compression method used, either "zlib" or "zstd"

    Returns
    -------
    np.ndarray
        The decompressed array
    """
    if compression_method == "zlib":
        decompressed_array = np.frombuffer(
            zlib.decompress(compressed_bytes), dtype=np.int16
        )
    elif compression_method == "zstd":
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        decompressed_array = np.frombuffer(dctx.decompress(compressed_bytes), dtype=np.int16)
    else:
        raise ValueError("compression_method must be 'zlib' or 'zstd'")
    decompressed_array = decompressed_array.reshape(-1, num_channels)
    x = qfc_inv_pre_compress(decompressed_array, quant_scale_factor=quant_scale_factor, dtype=dtype)
    return x


def qfc_estimate_quant_scale_factor(
    x: np.ndarray,
    target_compression_ratio: Union[float, None] = None,
    target_residual_std: Union[float, None] = None
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
    target_residual_std : float or None
        The target residual standard deviation
        Exactly one of target_compression_ratio and target_residual_std must be specified

    Returns
    -------
    float
        The quantization scale factor
    """
    x_fft = np.fft.rfft(x, axis=0) / np.sqrt(x.shape[0])  # we divide by the sqrt of the number of samples so that quantization scale factor does not depend on the number of samples
    x_fft = np.ascontiguousarray(x_fft)  # This is important so that the codec will behave properly!
    x_fft_re = np.real(x_fft)
    x_fft_im = np.imag(x_fft)
    x_fft_im = x_fft_im[1:-1]  # the first and last values are always zero

    if target_compression_ratio is not None:
        if target_residual_std is not None:
            raise ValueError(
                "Only one of target_compression_ratio and target_residual_std can be specified"
            )
        values = np.concatenate([x_fft_re, x_fft_im], axis=0).ravel()

        # sample at most 5000 values to estimate the quantization scale factor
        # do it deterministically to avoid randomness in the results
        if values.size > 5000:
            indices = np.linspace(0, values.size - 1, 5000).astype(np.int32)
            values = values[indices]

        def _estimate_entropy(values: np.ndarray):
            """
            Computes the entropy of an array

            Parameters
            ----------
            values : np.ndarray
                The values to compute the entropy of

            Returns
            -------
            float
                The entropy of the array
            """
            unique_values, counts = np.unique(values, return_counts=True)
            probabilities = counts / values.size
            entropy = -np.sum(probabilities * np.log2(probabilities)).astype(np.float32)
            return float(entropy)

        def entropy_for_quant_scale_factor(qs: float):
            return _estimate_entropy(np.round(values * qs).astype(np.int16))

        num_bits_per_value_in_original_array = np.dtype(x.dtype).itemsize * 8
        target_entropy = float(num_bits_per_value_in_original_array) / target_compression_ratio

        qs = _monotonic_binary_search(
            entropy_for_quant_scale_factor,
            target_value=target_entropy,
            max_iterations=100,
            tolerance=0.001,
            ascending=True
        )
        return qs
    elif target_residual_std is not None:
        if target_compression_ratio is not None:
            raise ValueError(
                "Only one of target_compression_ratio and target_residual_std can be specified"
            )
        target_sumsqr_in_fourier_domain = (target_residual_std**2 / 2) * x.shape[0]
        def resid_sumsqr_in_fourier_domain_for_quant_scale_factor(qs: float):
            x_re_quantized = np.round(x_fft_re * qs).astype(np.int16) / qs
            x_im_quantized = np.round(x_fft_im * qs).astype(np.int16) / qs
            diffs = np.concatenate([x_re_quantized - x_fft_re, x_im_quantized - x_fft_im], axis=0)
            return np.sum(np.square(diffs)) / x.shape[1]
        qs = _monotonic_binary_search(
            resid_sumsqr_in_fourier_domain_for_quant_scale_factor,
            target_value=target_sumsqr_in_fourier_domain,
            max_iterations=100,
            tolerance=0.001,
            ascending=False
        )
        return qs
    else:
        raise ValueError(
            "You must either specify target_compression_ratio or target_residual_std"
        )


def _monotonic_binary_search(
    func: Callable[[float], float],
    target_value: float,
    max_iterations: int,
    tolerance: float,
    ascending: bool
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
