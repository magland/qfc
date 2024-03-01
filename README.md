# QFC - Quantized Fourier Compression of Timeseries Data with Application to Electrophysiology

## Overview

With the increasing sizes of data for extracellular electrophysiology, it is crucial to develop efficient methods for compressing multi-channel time series data. While lossless methods are desirable for perfectly preserving the original signal, the compression ratios for these methods usually range only from 2-4x. What is needed are ratios on the order of 10-30x, leading us to consider lossy methods.

Here, we implement a simple lossy compression method, inspired by the Discrete Cosine Transform (DCT) and the quantization steps of JPEG compression for images. The method comprises the following steps:
* Compute the Discrete Fourier Transform (DFT) of the time series data in the time domain.
* Quantize the Fourier coefficients to achieve a target entropy (the entropy determines the theoretically achievable compression ratio). This is done by multiplying by a normalization factor and then rounding to the nearest integer.
* Compress the reduced-entropy quantized Fourier coefficients using zlib or zstd (other methods could be used instead).

To decompress:
* Decompress the quantized Fourier coefficients.
* Divide by the normalization factor.
* Compute the Inverse Discrete Fourier Transform (IDFT) to obtain the reconstructed time series data.

This method is particularly well-suited for data that has been bandpass-filtered, as the suppressed Fourier coefficients yield an especially low entropy of the quantized signal.

For a comparison of various lossy and lossless compression schemes, see [Compression strategies for large-scale electrophysiology data, Buccino et al.](https://www.biorxiv.org/content/10.1101/2023.05.22.541700v2.full.pdf).

## Installation

```bash
pip install qfc
```

## Example usage

```python
# See examples/example1.py

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
```

## Zarr example

See [examples/zarr_example.py](./examples/zarr_example.py)

## Benchmarks

I have put together some preliminary systematic benchmarks on real and synthetic data. See [./benchmarks](./benchmarks) and [./benchmarks/results](./benchmarks/results).

As can be seen:
- Quantizing in the Fourier domain (QFC) is a lot better than quantizing in the time domain (call it QTC) for real data or for bandpass-filtered data.
- The compression ratio is a lot better for bandpass-filtered data compared with unfiltered raw.
- For the lossless part of the method, zstd is better than zlib, both in terms of all three of these factors: compression ratio, compression speed, and decompression speed.
- Obviously, the compression ratio is going to depend heavily on the target residual std. dev.

## License

This code is provided under the Apache License, Version 2.0.


## Author

Jeremy Magland, Center for Computational Mathematics, Flatiron Institute
