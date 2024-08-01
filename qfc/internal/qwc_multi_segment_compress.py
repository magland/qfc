from typing import Literal, Union
import concurrent.futures
import zlib
import numpy as np
from .qwc_compress import qwc_pre_compress, qwc_inv_pre_compress
from .qfc_multi_segment_compress import _get_segment_ranges
from .qfc_compress import int_dtype
from .qfc_compress import _monotonic_binary_search


def qwc_multi_segment_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    segment_length: int,
    pywt_wavelet: str,
    pywt_level: Union[int, None],
    pywt_mode: str
):
    """
    Prepares an array for compression using the QWC algorithm with multiple segments

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qwc_estimate_quant_scale_factor
    segment_length : int
        The length of each segment
    pywt_wavelet : str
        The wavelet parameter for PyWavelets
    pywt_level : int or None
        The level parameter for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets

    Returns
    -------
    np.ndarray
        The prepared array
    list
        A list of the sizes of the wavelet coefficients
    """
    if segment_length > 0 and segment_length < x.shape[0]:
        segment_ranges = _get_segment_ranges(total_length=x.shape[0], segment_length=segment_length)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prepared_segments = list(executor.map(
                lambda segment_range: qwc_pre_compress(
                    x[segment_range[0]:segment_range[1], :],
                    quant_scale_factor=quant_scale_factor,
                    pywt_wavelet=pywt_wavelet,
                    pywt_level=pywt_level,
                    pywt_mode=pywt_mode
                ),
                segment_ranges
            ))
        arrays = [p[0] for p in prepared_segments]
        coeff_sizes_list = [p[1] for p in prepared_segments]
        return np.concatenate(arrays, axis=0), coeff_sizes_list
    else:
        array, coeff_sizes = qwc_pre_compress(x, quant_scale_factor=quant_scale_factor, pywt_wavelet=pywt_wavelet, pywt_level=pywt_level, pywt_mode=pywt_mode)
        return array, [coeff_sizes]


def qwc_multi_segment_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    segment_length: int,
    pywt_wavelet: str,
    pywt_level: Union[int, None],
    pywt_mode: str,
    compression_method: Literal["zlib", "zstd"] = "zlib",
    zstd_level: int = 3,
    zlib_level: int = 3
):
    """
    Compresses an array using the QWC algorithm with multiple segments

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qwc_estimate_quant_scale_factor
    pywt_wavelet : str
        The wavelet parameter for PyWavelets
    pywt_level : int or None
        The level parameter for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    segment_length : int
        The length of each segment
    compression_method : Literal["zlib", "zstd"]
        The compression method to use
    zstd_level : int
        The compression level to use for zstd
    zlib_level : int
        The compression level to use for zlib

    Returns
    -------
    bytes
        The compressed array
    """
    coeffs_quantized, coeff_sizes_list = qwc_multi_segment_pre_compress(
        x,
        quant_scale_factor=quant_scale_factor,
        segment_length=segment_length,
        pywt_wavelet=pywt_wavelet,
        pywt_level=pywt_level,
        pywt_mode=pywt_mode
    )
    if compression_method == "zlib":
        compressed_bytes = zlib.compress(coeffs_quantized.tobytes(), level=zlib_level)
    elif compression_method == "zstd":
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(
            level=zstd_level
        )
        compressed_bytes = cctx.compress(coeffs_quantized.tobytes())
    else:
        raise ValueError("compression_method must be 'zlib' or 'zstd'")
    header0 = _create_qwc_header_multi(coeff_sizes_list)
    return header0 + compressed_bytes


def _create_qwc_header_multi(coeff_sizes):
    num_segments = len(coeff_sizes)
    header = np.array([num_segments]).astype(np.int32).tobytes()
    for s in coeff_sizes:
        header = header + _create_qwc_header(s)
    return header


def _create_qwc_header(coeff_sizes):
    num_coeff_sizes = len(coeff_sizes)
    header1 = np.array([num_coeff_sizes]).astype(np.int32)
    header2 = np.array(coeff_sizes).astype(np.int32)
    return header1.tobytes() + header2.tobytes()


def qwc_multi_segment_inv_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    coeff_sizes_list: list,
    segment_length: int,
    dtype: str,
    pywt_wavelet: str,
    pywt_mode: str
):
    """
    Inverts the preparation of an array for compression using the QWC algorithm with multiple segments

    Parameters
    ----------
    x : np.ndarray
        The prepared array
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qwc_estimate_quant_scale_factor
    coeff_sizes : list
        A list of lists of the sizes of the wavelet coefficients, one for each segment
    segment_length : int
        The length of each segment
    dtype : str
        The data type string for the reconstructed array
    pywt_wavelet: str
        The wavelet type for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    """
    if segment_length > 0 and segment_length < x.shape[0]:
        segment_ranges = []
        ii = 0
        for coeff_sizes in coeff_sizes_list:
            segment_ranges.append((ii, ii + np.sum(coeff_sizes)))
            ii += np.sum(coeff_sizes)
        prepared_segments = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prepared_segments = list(executor.map(
                lambda aa: qwc_inv_pre_compress(
                    x[aa[0][0]:aa[0][1], :],
                    quant_scale_factor=quant_scale_factor,
                    coeff_sizes=coeff_sizes_list[aa[1]],
                    dtype=dtype,
                    pywt_wavelet=pywt_wavelet,
                    pywt_mode=pywt_mode
                ),
                [(segment_range, segment_range_index) for segment_range_index, segment_range in enumerate(segment_ranges)]
            ))
        return np.concatenate(prepared_segments, axis=0)
    else:
        return qwc_inv_pre_compress(x, quant_scale_factor=quant_scale_factor, dtype=dtype, coeff_sizes=coeff_sizes_list[0], pywt_wavelet=pywt_wavelet, pywt_mode=pywt_mode)


def qwc_multi_segment_decompress(
    compressed_bytes: bytes,
    quant_scale_factor: float,
    num_channels: int,
    dtype: str,
    segment_length: int,
    pywt_wavelet: str,
    pywt_mode: str,
    compression_method: Literal["zlib", "zstd"] = "zlib"
):
    """
    Decompresses an array using the QWC algorithm with multiple segments

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed array
    quant_scale_factor : float
        The quantization scale factor used during compression
    num_channels : int
        The number of channels
    dtype : int
        The data type string for the reconstructed array
    segment_length : int
        The length of each segment
    pywt_wavelet: str
        The wavelet type for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    compression_method : Literal["zlib", "zstd"]
        The compression method used during compression

    Returns
    -------
    np.ndarray
        The decompressed array
    """
    coeff_sizes_list, X = _parse_qwc_compressed_bytes_multi(compressed_bytes)

    if compression_method == "zlib":
        decompressed_coeffs = np.frombuffer(
            zlib.decompress(X), dtype=int_dtype
        )
    elif compression_method == "zstd":
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        decompressed_coeffs = np.frombuffer(dctx.decompress(X), dtype=int_dtype)
    else:
        raise ValueError("compression_method must be 'zlib' or 'zstd'")
    decompressed_coeffs = decompressed_coeffs.reshape(-1, num_channels)
    x = qwc_multi_segment_inv_pre_compress(
        decompressed_coeffs,
        quant_scale_factor=quant_scale_factor,
        coeff_sizes_list=coeff_sizes_list,
        segment_length=segment_length,
        pywt_wavelet=pywt_wavelet,
        pywt_mode=pywt_mode,
        dtype=dtype
    )
    return x


def _parse_qwc_compressed_bytes_multi(compressed_bytes):
    num_segments = np.frombuffer(compressed_bytes[:4], dtype=np.int32)[0]
    coeff_sizes_list = []
    offset = 4
    for i in range(num_segments):
        num_coeff_sizes = np.frombuffer(compressed_bytes[offset:offset + 4], dtype=np.int32)[0]
        coeff_sizes = np.frombuffer(compressed_bytes[offset + 4:offset + 4 + num_coeff_sizes * 4], dtype=np.int32).tolist()
        coeff_sizes_list.append(coeff_sizes)
        offset += 4 + num_coeff_sizes * 4
    X = compressed_bytes[offset:]
    return coeff_sizes_list, X


def qwc_estimate_quant_scale_factor(
    x: np.ndarray,
    target_residual_stdev: float,
    pywt_wavelet: str,
    pywt_level: Union[int, None],
    pywt_mode: str,
    segment_length: int,
    max_num_samples: int = 30000 * 3,
):
    """
    Estimates the quantization scale factor for the QWC algorithm for a given
    target compression ratio or target residual standard deviation

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    target_residual_stdev : float
        The target residual standard deviation
    pywt_wavelet : str
        The wavelet parameter for PyWavelets
    pywt_level : int or None
        The level parameter for PyWavelets
    pywt_mode : str
        The signal extension mode parameter for PyWavelets
    segment_length : int
        The size of each segment
    max_num_samples : int
        The maximum number of samples to use for the estimation

    Returns
    -------
    float
        The quantization scale factor
    """
    if x.shape[0] > max_num_samples:
        x = x[:max_num_samples]

    def get_resid_stdev_for_quant_scale_factor(qs: float):
        compressed = qwc_multi_segment_compress(
            x,
            quant_scale_factor=qs,
            pywt_wavelet=pywt_wavelet,
            pywt_level=pywt_level,
            pywt_mode=pywt_mode,
            segment_length=segment_length,
        )
        y_reconstructed = qwc_multi_segment_decompress(
            compressed,
            quant_scale_factor=qs,
            num_channels=x.shape[1],
            dtype=x.dtype,
            pywt_wavelet=pywt_wavelet,
            pywt_mode=pywt_mode,
            segment_length=segment_length
        )
        y_resid = x - y_reconstructed
        return float(np.std(y_resid))

    qs = _monotonic_binary_search(
        get_resid_stdev_for_quant_scale_factor,
        target_value=target_residual_stdev,
        max_iterations=100,
        tolerance=0.001,
        ascending=False
    )
    return qs