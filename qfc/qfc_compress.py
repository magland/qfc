import numpy as np
import zlib


def qfc_compress(
    x: np.ndarray,
    normalization_factor: float
):
    """
    Compresses an array using the QFC algorithm

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    normalization_factor : float
        The normalization factor to use during compression,
        obtained from qfc_estimate_normalization_factor

    Returns
    -------
    np.ndarray
        The compressed array
    """
    nf = normalization_factor
    x_fft = np.fft.rfft(x, axis=0)
    x_fft_re = np.real(x_fft)
    x_fft_im = np.imag(x_fft)
    x_fft_im = x_fft_im[1:-1]  # the first and last values are always zero
    x_fft_concat = np.concatenate([x_fft_re, x_fft_im], axis=0)
    x_fft_concat_quantized = np.round(x_fft_concat * nf).astype(
        np.int16
    )
    compressed_bytes = zlib.compress(x_fft_concat_quantized.tobytes())
    return compressed_bytes


def qfc_decompress(
    compressed_bytes: bytes,
    normalization_factor: float,
    original_shape: tuple
):
    """
    Decompresses an array using the QFC algorithm

    Parameters
    ----------
    compressed_bytes : bytes
        The compressed array
    normalization_factor : float
        The normalization factor used during compression
    original_shape : tuple
        The original shape of the array

    Returns
    -------
    np.ndarray
        The decompressed array
    """
    nf = normalization_factor
    num_samples = original_shape[0]
    num_channels = original_shape[1] if len(original_shape) > 1 else 1
    decompressed_array = np.frombuffer(
        zlib.decompress(compressed_bytes), dtype=np.int16
    )
    decompressed_array = decompressed_array.reshape(num_samples, num_channels)
    x_fft_re = decompressed_array[: (num_samples // 2 + 1), :] / nf
    x_fft_im = decompressed_array[(num_samples // 2 + 1):, :] / nf
    x_fft_im = np.concatenate(
        [np.zeros((1, num_channels)), x_fft_im, np.zeros((1, num_channels))],
        axis=0
    )
    x_fft = x_fft_re + 1j * x_fft_im
    x = np.fft.irfft(x_fft, axis=0)
    if len(original_shape) == 1:
        x = x.ravel()
    return x


def qfc_estimate_normalization_factor(
    x: np.ndarray,
    target_compression_ratio: float
):
    """
    Estimates the normalization factor for the QFC algorithm for a given
    target compression ratio

    Parameters
    ----------
    x : np.ndarray
        The input array to be compressed
    target_compression_ratio : float
        The target compression ratio

    Returns
    -------
    float
        The normalization factor
    """
    x_fft = np.fft.rfft(x, axis=0)
    x_fft_re = np.real(x_fft)
    x_fft_im = np.imag(x_fft)
    x_fft_im = x_fft_im[1:-1]  # the first and last values are always zero
    values = np.concatenate([x_fft_re, x_fft_im], axis=0).ravel()

    # sample at most 5000 values to estimate the normalization factor
    # do it deterministically to avoid randomness in the results
    if values.size > 5000:
        indices = np.linspace(0, values.size - 1, 5000).astype(np.int32)
        values = values[indices]

    max_abs_value = np.max(np.abs(values))

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
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    candidates = np.arange(1, 1000, 1) * 1 / max_abs_value
    entropies = np.array(
        [
            _estimate_entropy(np.round(values * candidate).astype(np.int16))
            for candidate in candidates
        ]
    )
    num_bits_per_value_in_original_array = np.dtype(x.dtype).itemsize * 8
    target_entropy = (
        float(num_bits_per_value_in_original_array) / target_compression_ratio
    )
    best_ind = np.argmin(np.abs(entropies - target_entropy))
    normalization_factor = candidates[best_ind]
    return normalization_factor
