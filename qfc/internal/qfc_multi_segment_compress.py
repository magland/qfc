from typing import Literal
import concurrent.futures
import zlib
import numpy as np
from .qfc_compress import qfc_pre_compress, qfc_inv_pre_compress, int_dtype


def qfc_multi_segment_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    segment_length: int
):
    """
    Prepares an array for compression using the QFC algorithm with multiple segments

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qfc_estimate_quant_scale_factor
    segment_length : int
        The length of each segment

    Returns
    -------
    np.ndarray
        The prepared array
    """
    if segment_length > 0 and segment_length < x.shape[0]:
        segment_ranges = _get_segment_ranges(total_length=x.shape[0], segment_length=segment_length)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prepared_segments = list(executor.map(
                lambda segment_range: qfc_pre_compress(
                    x[segment_range[0]:segment_range[1], :],
                    quant_scale_factor=quant_scale_factor
                ),
                segment_ranges
            ))
        return np.concatenate(prepared_segments, axis=0)
    else:
        return qfc_pre_compress(x, quant_scale_factor=quant_scale_factor)


def qfc_multi_segment_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    segment_length: int,
    compression_method: Literal["zlib", "zstd"] = "zlib",
    zstd_level: int = 3,
    zlib_level: int = 3
):
    """
    Compresses an array using the QFC algorithm with multiple segments

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qfc_estimate_quant_scale_factor
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
        The compressed array as bytes
    """
    x_fft_concat_quantized = qfc_multi_segment_pre_compress(
        x,
        quant_scale_factor=quant_scale_factor,
        segment_length=segment_length
    )
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


def qfc_multi_segment_inv_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    segment_length: int,
    dtype: str
):
    """
    Inverts the preparation of an array for compression using the QFC algorithm with multiple segments

    Parameters
    ----------
    x : np.ndarray
        The prepared array
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qfc_estimate_quant_scale_factor
    segment_length : int
        The length of each segment
    dtype : str
        The data type string for the reconstructed array
    """
    if segment_length > 0 and segment_length < x.shape[0]:
        segment_ranges = _get_segment_ranges(total_length=x.shape[0], segment_length=segment_length)
        prepared_segments = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            prepared_segments = list(executor.map(
                lambda segment_range: qfc_inv_pre_compress(
                    x[segment_range[0]:segment_range[1], :],
                    quant_scale_factor=quant_scale_factor,
                    dtype=dtype
                ),
                segment_ranges
            ))
        return np.concatenate(prepared_segments, axis=0)
    else:
        return qfc_inv_pre_compress(x, quant_scale_factor=quant_scale_factor, dtype=dtype)


def _get_segment_ranges(*, total_length: int, segment_length: int):
    segment_ranges = []
    for start_index in range(0, total_length, segment_length):
        segment_ranges.append((start_index, min(start_index + segment_length, total_length)))
    size_of_final_segment = segment_ranges[-1][1] - segment_ranges[-1][0]
    half_segment_length = int(segment_length / 2)
    if size_of_final_segment < half_segment_length and len(segment_ranges) > 1:
        adjustment = half_segment_length - size_of_final_segment
        segment_ranges[-2] = (segment_ranges[-2][0], segment_ranges[-2][1] - adjustment)
        segment_ranges[-1] = (segment_ranges[-1][0] - adjustment, segment_ranges[-1][1])
    return segment_ranges


def qfc_multi_segment_decompress(
    compressed_bytes: bytes,
    quant_scale_factor: float,
    num_channels: int,
    dtype: str,
    segment_length: int,
    compression_method: Literal["zlib", "zstd"] = "zlib"
):
    """
    Decompresses an array using the QFC algorithm with multiple segments

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
    compression_method : Literal["zlib", "zstd"]
        The compression method used during compression

    Returns
    -------
    np.ndarray
        The decompressed array
    """
    if compression_method == "zlib":
        decompressed_array = np.frombuffer(
            zlib.decompress(compressed_bytes), dtype=int_dtype
        )
    elif compression_method == "zstd":
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        decompressed_array = np.frombuffer(dctx.decompress(compressed_bytes), dtype=int_dtype)
    else:
        raise ValueError("compression_method must be 'zlib' or 'zstd'")
    decompressed_array = decompressed_array.reshape(-1, num_channels)
    x = qfc_multi_segment_inv_pre_compress(
        decompressed_array,
        quant_scale_factor=quant_scale_factor,
        segment_length=segment_length,
        dtype=dtype
    )
    return x
