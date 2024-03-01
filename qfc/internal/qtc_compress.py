from typing import Literal
import numpy as np
import zlib
from .qfc_compress import _monotonic_binary_search


def qtc_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float
):
    """
    Prepares an array for compression using the QTC algorithm This is an
    internal function, called by the codec, and should not be called directly.

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization, obtained from
        qtc_estimate_quant_scale_factor

    Returns
    -------
    np.ndarray
        The prepared array.
    """
    qs = quant_scale_factor
    x_quantized = np.round(x * qs).astype(
        np.int16
    )
    return x_quantized


def qtc_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    compression_method: Literal["zlib", "zstd"] = "zlib",
    zstd_level: int = 3,
    zlib_level: int = 3
):
    """
    Compresses an array using the QTC algorithm. This is an internal function,
    called by the codec, and should not be called directly, because the output
    will not include the header.

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    quant_scale_factor : float
        The scale factor to use during quantization, obtained from
        qtc_estimate_quant_scale_factor
    compression_method : str
        The compression method to use, either "zlib" or "zstd"
    zstd_level : int
        The compression level to use if compression_method is "zstd"
    zlib_level : int
        The compression level to use if compression_method is "zlib"

    Returns
    -------
    bytes
        The compressed array as bytes. Will not contain the header.
    """
    x_quantized = qtc_pre_compress(x, quant_scale_factor=quant_scale_factor)
    if compression_method == "zlib":
        compressed_bytes = zlib.compress(x_quantized.tobytes(), level=zlib_level)
    elif compression_method == "zstd":
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(
            level=zstd_level
        )
        compressed_bytes = cctx.compress(x_quantized.tobytes())
    else:
        raise ValueError("compression_method must be 'zlib' or 'zstd'")
    return compressed_bytes


def qtc_inv_pre_compress(
    x: np.ndarray, *,
    quant_scale_factor: float,
    dtype: str
):
    """
    Inverts the preparation of an array for compression using the QTC algorithm.
    This is an internal function, called by the codec, and should not be called
    directly.

    Parameters
    ----------
    x : np.ndarray
        The prepared array
    quant_scale_factor : float
        The scale factor to use during quantization,
        obtained from qtc_estimate_quant_scale_factor
    dtype : str
        The data type string for the reconstructed array
    """
    qs = quant_scale_factor
    x_unquantized = x.astype(np.float32) / qs
    return x_unquantized.astype(dtype)


def qtc_decompress(
    compressed_bytes: bytes, *,
    quant_scale_factor: float,
    num_channels: int,
    dtype: str,
    compression_method: Literal["zlib", "zstd"] = "zlib"
):
    """
    Decompresses an array using the QTC algorithm.
    This is an internal function, called by the codec, and should not be called
    directly, because the input is expected to be the compressed array only, not
    including the header.

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed array, not including the header
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
    x = qtc_inv_pre_compress(decompressed_array, quant_scale_factor=quant_scale_factor, dtype=dtype)
    return x


def qtc_estimate_quant_scale_factor(
    x: np.ndarray,
    target_residual_stdev: float,
    max_num_samples: int = 30000 * 3
):
    """
    Estimates the quantization scale factor for the QTC algorithm for a given
    target compression ratio or target residual standard deviation

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    target_residual_stdev : float
        The target residual standard deviation
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
        x_quantized = np.round(x * qs).astype(np.int16)
        x_reconstructed = x_quantized.astype(np.float32) / qs
        x_resid = x - x_reconstructed
        return float(np.std(x_resid))

    qs = _monotonic_binary_search(
        get_resid_stdev_for_quant_scale_factor,
        target_value=target_residual_stdev,
        max_iterations=100,
        tolerance=0.001,
        ascending=False
    )
    return qs
