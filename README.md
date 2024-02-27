# QFC - Quantized Fourier Compression of Timeseries Data with Application to Electrophysiology

## Overview

With the increasing sizes of data for extracellular electrophysiology, it is crucial to develop efficient methods for compressing multi-channel time series data. While lossless methods are desirable for perfectly preserving the original signal, the compression ratios for these methods usually range only from 2-4x. What is needed are ratios on the order of 10-30x, leading us to consider lossy methods.

Here, we introduce a simple lossy compression method, inspired by the Discrete Cosine Transform (DCT) and the quantization steps of JPEG compression for images. The method comprises the following steps:
* Compute the Discrete Fourier Transform (DFT) of the time series data in the time domain.
* Quantize the Fourier coefficients to achieve a target entropy (the entropy determines the theoretically achievable compression ratio). This is done by multiplying by a normalization factor and then rounding to the nearest integer.
* Compress the reduced-entropy quantized Fourier coefficients using GZIP (other methods could also be used).

To decompress:
* Unzip the quantized Fourier coefficients.
* Divide by the normalization factor.
* Compute the Inverse Discrete Fourier Transform (IDFT) to obtain the reconstructed time series data.

This method is particularly well-suited for data that has been bandpass-filtered, as the suppressed Fourier coefficients yield an especially low entropy of the quantized signal.

For a comparison of various lossy and lossless compression schemes, see [Compression strategies for large-scale electrophysiology data, Buccino et al.](https://www.biorxiv.org/content/10.1101/2023.05.22.541700v2.full.pdf).

For application to real data, [see this notebook](https://github.com/magland/qfc/blob/main/qfc.ipynb).

## Installation

```bash
pip install qfc
```

## Usage

```python
# See the examples directory

from matplotlib import pyplot as plt
import numpy as np
from qfc import qfc_compress, qfc_decompress, qfc_estimate_normalization_factor


def main():
    sampling_frequency = 30000
    y = np.random.randn(5000, 10) * 50
    y = lowpass_filter(y, sampling_frequency, 6000)
    y = y.astype(np.int16)
    target_compression_ratio = 15

    ############################################################
    normalization_factor = qfc_estimate_normalization_factor(
        y,
        target_compression_ratio=target_compression_ratio
    )
    compressed_bytes = qfc_compress(
        y,
        normalization_factor=normalization_factor
    )
    y_decompressed = qfc_decompress(
        compressed_bytes,
        normalization_factor=normalization_factor,
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
    main()
```

## License

This code is provided under the Apache License, Version 2.0.


## Author

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute