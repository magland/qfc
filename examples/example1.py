from matplotlib import pyplot as plt
import numpy as np
from qfc import qfc_estimate_quant_scale_factor
from qfc.codecs import QFCCodec


def main():
    sampling_frequency = 30000
    duration = 2
    num_channels = 10
    num_samples = int(sampling_frequency * duration)
    y = np.random.randn(num_samples, num_channels) * 50
    y = lowpass_filter(y, sampling_frequency, 6000)
    y = np.ascontiguousarray(y)  # compressor requires C-order arrays
    y = y.astype(np.int16)
    target_residual_stdev = 5

    ############################################################
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        y,
        target_residual_stdev=target_residual_stdev
    )
    codec = QFCCodec(
        quant_scale_factor=quant_scale_factor,
        dtype="int16",
        segment_length=10000,
        compression_method="zstd",
        zstd_level=3
    )
    compressed_bytes = codec.encode(y)
    y_reconstructed = codec.decode(compressed_bytes)
    ############################################################

    y_resid = y - y_reconstructed
    original_size = y.nbytes
    compressed_size = len(compressed_bytes)
    compression_ratio = original_size / compressed_size
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Actual compression ratio: {compression_ratio}")
    print(f'Target residual std. dev.: {target_residual_stdev:.2f}')
    print(f'Actual Std. dev. of residual: {np.std(y_resid):.2f}')

    xgrid = np.arange(y.shape[0]) / sampling_frequency
    ch = 3  # select a channel to plot
    n = 1000  # number of samples to plot
    plt.figure()
    plt.plot(xgrid[:n], y[:n, ch], label="Original")
    plt.plot(xgrid[:n], y_reconstructed[:n, ch], label="Decompressed")
    plt.plot(xgrid[:n], y_resid[:n, ch], label="Residual")
    plt.xlabel("Time")
    plt.title(f'QFC compression ratio: {compression_ratio:.2f}')
    plt.legend()
    plt.show()


def lowpass_filter(input_array, sampling_frequency, cutoff_frequency):
    F = np.fft.fft(input_array, axis=0)
    N = input_array.shape[0]
    freqs = np.fft.fftfreq(N, d=1 / sampling_frequency)
    sigma = cutoff_frequency / 3
    window = np.exp(-np.square(freqs) / (2 * sigma**2))
    F_filtered = F * window[:, None]
    filtered_array = np.fft.ifft(F_filtered, axis=0)
    return np.real(filtered_array)


if __name__ == "__main__":
    main()
