import tempfile
import os
import time
import numpy as np
import zarr
from models import BenchmarkTestParams, BenchmarkTestResult
from qfc.codecs import QFCCodec, QTCCodec
from qfc import qfc_estimate_quant_scale_factor, qtc_estimate_quant_scale_factor


def run_benchmark_test(params: BenchmarkTestParams, *, show_plots=False) -> BenchmarkTestResult:
    """
    Run a benchmark test with the given parameters and return the result.
    """

    sampling_frequency = params.sampling_frequency
    duration_sec = params.duration_sec
    num_samples = int(sampling_frequency * duration_sec)
    num_channels = params.num_channels
    segment_length = int(params.segment_length_sec * sampling_frequency)
    chunks = (int(params.chunk_size_sec * sampling_frequency), num_channels)
    compression_method = params.compression_method
    zstd_level = params.compression_level
    zlib_level = params.compression_level

    if params.data_path:
        array = np.load(params.data_path)
        array = array[:num_samples, :num_channels]
        if array.shape[0] != num_samples or array.shape[1] != num_channels:
            raise Exception(f"Data shape mismatch: {array.shape} vs. {num_samples} x {num_channels}")
        if str(array.dtype) != "int16":
            raise Exception(f"Data type mismatch: {array.dtype} vs. int16")
    else:
        array = (np.random.randn(num_samples, num_channels) * 20)
    if params.bandpass_filter:
        array = bandpass_filter(array, sampling_frequency, 300, 6000)
    array = np.ascontiguousarray(array)  # compressor requires C-order arrays
    array = array.astype("int16")
    noise_level = estimate_noise_level(array, sampling_frequency)
    print(f"Estimated noise level: {noise_level}")
    target_residual_stdev = noise_level * params.relative_target_residual_stdev

    if params.method == 'qfc':
        quant_scale_factor = qfc_estimate_quant_scale_factor(
            array, target_residual_stdev=target_residual_stdev
        )
        print(f'Using quant_scale_factor: {quant_scale_factor}')
        codec = QFCCodec(
            quant_scale_factor=quant_scale_factor,
            dtype="int16",
            compression_method=compression_method,  # type: ignore
            segment_length=segment_length,
            zstd_level=zstd_level,
            zlib_level=zlib_level,
        )
    elif params.method == 'qtc':
        quant_scale_factor = qtc_estimate_quant_scale_factor(
            array, target_residual_stdev=target_residual_stdev
        )
        print(f'Using quant_scale_factor: {quant_scale_factor}')
        codec = QTCCodec(
            quant_scale_factor=quant_scale_factor,
            dtype="int16",
            compression_method=compression_method,  # type: ignore
            zstd_level=zstd_level,
            zlib_level=zlib_level,
        )
    else:
        raise Exception(f"Unknown method: {params.method}")

    with tempfile.TemporaryDirectory() as tmpdir:
        g = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="w")

        timer = time.time()
        g.create_dataset("array", data=array, chunks=chunks, compressor=codec)
        elapsed_compress = time.time() - timer

        compressed_size = measure_total_size(tmpdir)
        uncompressed_size = array.nbytes

        timer = time.time()
        g2 = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="r")
        loaded_z = g2["array"][:]
        elapsed_decompress = time.time() - timer

        resid_stdev = np.sqrt(np.var(array - loaded_z))

        if show_plots:
            import matplotlib.pyplot as plt
            resid = array - loaded_z
            # zoom in on a single spike that has half the amplitude of the largest spike
            channel = 120
            aa = bandpass_filter(array[:, channel], sampling_frequency, 300, 6000)
            minval = np.min(aa)
            targetval = minval * 0.5
            ind = np.argmin(np.abs(aa - targetval))
            rng = [ind - 300, ind + 300]
            plt.figure()
            plt.plot(array[rng[0]:rng[1], channel], label="original")
            plt.plot(loaded_z[rng[0]:rng[1], channel], label="loaded")  # type: ignore
            plt.plot(resid[rng[0]:rng[1], channel], label="residual")
            plt.legend()
            plt.title(f"{params.name} (CR = {uncompressed_size / compressed_size:.2f})")
            plt.show()

    return BenchmarkTestResult(
        params=params,
        uncompressed_size=uncompressed_size,
        compressed_size=compressed_size,
        compression_ratio=uncompressed_size / compressed_size,
        elapsed_compression_sec=elapsed_compress,
        elapsed_decompression_sec=elapsed_decompress,
        residual_stdev=resid_stdev,
        relative_residual_stdev=resid_stdev / noise_level
    )


def lowpass_filter(array, sampling_frequency, highcut) -> np.ndarray:
    from scipy.signal import butter, lfilter

    nyquist = 0.5 * sampling_frequency
    high = highcut / nyquist
    b, a = butter(5, high, btype="low")
    return lfilter(b, a, array, axis=0)  # type: ignore


def highpass_filter(array, sampling_frequency, lowcut) -> np.ndarray:
    from scipy.signal import butter, lfilter

    nyquist = 0.5 * sampling_frequency
    low = lowcut / nyquist
    b, a = butter(5, low, btype="high")
    return lfilter(b, a, array, axis=0)  # type: ignore


def bandpass_filter(array, sampling_frequency, lowcut, highcut) -> np.ndarray:
    from scipy.signal import butter, lfilter

    nyquist = 0.5 * sampling_frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype="band")
    return lfilter(b, a, array, axis=0)  # type: ignore


def measure_total_size(tmpdir):
    import os

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(tmpdir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def estimate_noise_level(array, sampling_frequency: float):
    x = highpass_filter(array, sampling_frequency, 300)
    # use median absolute deviation as a robust estimator of noise level
    return float(1.4826 * np.median(np.abs(x - np.median(x))))
