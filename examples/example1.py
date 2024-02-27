from matplotlib import pyplot as plt
import numpy as np
from qfc import qfc_compress, qfc_decompress, qfc_estimate_quant_scale_factor


def main1():
    num_samples = 5000
    sampling_frequency = 30000
    y = np.random.randn(num_samples, 10) * 50
    y = lowpass_filter(y, sampling_frequency, 6000)
    y = y.astype(np.int16)
    target_compression_ratio = 15

    ############################################################
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        y,
        target_compression_ratio=target_compression_ratio
    )
    print(f'Using quant scale factor: {quant_scale_factor}')
    compressed_bytes = qfc_compress(
        y,
        quant_scale_factor=quant_scale_factor
    )
    y_decompressed = qfc_decompress(
        compressed_bytes,
        quant_scale_factor=quant_scale_factor,
        original_shape=y.shape
    )
    ############################################################

    y_resid = y - y_decompressed
    original_size = y.nbytes
    compressed_size = len(compressed_bytes)
    compression_ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f'Target compression ratio: {target_compression_ratio}')
    print(f"Actual compression ratio: {compression_ratio}")
    print(f'Std. dev. of residual: {np.std(y_resid):.2f}')

    xgrid = np.arange(y.shape[0]) / sampling_frequency
    ch = 3  # select a channel to plot
    plt.figure()
    plt.plot(xgrid, y[:, ch], label="Original")
    plt.plot(xgrid, y_decompressed[:, ch], label="Decompressed")
    plt.plot(xgrid, y_resid[:, ch], label="Residual")
    plt.xlabel("Time")
    plt.title(f'QFC compression ratio: {compression_ratio:.2f}')
    plt.legend()
    plt.show()


def main2():
    num_samples = 50000
    sampling_frequency = 30000
    y = np.random.randn(num_samples, 10) * 50
    y = lowpass_filter(y, sampling_frequency, 6000)
    y = y.astype(np.int16)
    target_residual_std = 3

    ############################################################
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        y,
        target_residual_std=target_residual_std
    )
    print(f'Using quant scale factor: {quant_scale_factor}')
    compressed_bytes = qfc_compress(
        y,
        quant_scale_factor=quant_scale_factor
    )
    y_decompressed = qfc_decompress(
        compressed_bytes,
        quant_scale_factor=quant_scale_factor,
        original_shape=y.shape
    )
    ############################################################

    y_resid = y - y_decompressed
    original_size = y.nbytes
    compressed_size = len(compressed_bytes)
    compression_ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Actual compression ratio: {compression_ratio}")
    print(f'Target residual std: {target_residual_std}')
    print(f'Std. dev. of residual: {np.std(y_resid):.2f}')

    xgrid = np.arange(y.shape[0]) / sampling_frequency
    ch = 3  # select a channel to plot
    plt.figure()
    plt.plot(xgrid, y[:, ch], label="Original")
    plt.plot(xgrid, y_decompressed[:, ch], label="Decompressed")
    plt.plot(xgrid, y_resid[:, ch], label="Residual")
    plt.xlabel("Time")
    plt.title(f'QFC compression ratio: {compression_ratio:.2f}')
    plt.legend()
    plt.show()


def lowpass_filter(input_array, sampling_frequency, cutoff_frequency):
    F = np.fft.fft(input_array, axis=0)
    N = input_array.shape[0]
    freqs = np.fft.fftfreq(N, d=1/sampling_frequency)
    sigma = cutoff_frequency / 3
    window = np.exp(-np.square(freqs) / (2 * sigma**2))
    F_filtered = F * window[:, None]
    filtered_array = np.fft.ifft(F_filtered, axis=0)
    return np.real(filtered_array)


if __name__ == "__main__":
    main1()
    main2()
