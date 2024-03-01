import time
import numpy as np
from qfc.codecs import QFCCodec
from qfc import qfc_estimate_quant_scale_factor

QFCCodec.register_codec()


def zarr_example():
    import zarr
    import tempfile
    import os

    sampling_frequency = 30000
    duration_sec = 6
    num_samples = int(sampling_frequency * duration_sec)
    segment_length = 10000
    chunks = (sampling_frequency, 64)
    num_channels = 384
    compression_method = "zstd"
    zstd_level = 3
    zlib_level = 3

    print(f"Duration = {duration_sec} sec; num_channels = {num_channels}")

    print("Defining array")
    array = (np.random.randn(num_samples, num_channels) * 20)
    array = lowpass_filter(array, sampling_frequency, 6000)
    array = np.ascontiguousarray(array)  # compressor requires C-order arrays
    array = array.astype("int16")
    noise_level = estimate_noise_level(array)
    target_residual_stdev = noise_level / 3
    print("Estimating quant scale factor")
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        array, target_residual_stdev=target_residual_stdev
    )

    print("Defining codec")
    codec = QFCCodec(
        quant_scale_factor=quant_scale_factor,
        dtype="int16",
        compression_method=compression_method,
        segment_length=segment_length,
        zstd_level=zstd_level,
        zlib_level=zlib_level,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        g = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="w")

        print("Writing compressed array")
        timer = time.time()
        g.create_dataset("array", data=array, chunks=chunks, compressor=codec)
        elapsed_compress = time.time() - timer

        print("Measuring compressed size")
        compressed_size = measure_total_size(tmpdir)
        uncompressed_size = array.nbytes

        print("Uncompressed size (MB):", uncompressed_size / 1024 / 1024)
        print("Compressed size (MB):", compressed_size / 1024 / 1024)

        print("Reading compressed array")
        timer = time.time()
        g2 = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="r")
        loaded_z = g2["array"][:]
        elapsed_decompress = time.time() - timer

        print("Checking residual stdev")
        resid_stdev = np.sqrt(np.var(array - loaded_z))

        print(f"Target residual stdev: {target_residual_stdev}")
        print(f"Residual stdev: {resid_stdev}")
        print('')
        print(f'Elapsed time (compress): {elapsed_compress:.2f} sec')
        print(f'Elapsed time (decompress): {elapsed_decompress:.2f} sec')
        print("Compression ratio:", uncompressed_size / compressed_size)

        assert (
            target_residual_stdev - 0.5 < resid_stdev
            and resid_stdev < target_residual_stdev + 0.5
        )


def lowpass_filter(input_array, sampling_frequency, cutoff_frequency):
    F = np.fft.fft(input_array, axis=0)
    N = input_array.shape[0]
    freqs = np.fft.fftfreq(N, d=1 / sampling_frequency)
    sigma = cutoff_frequency / 3
    window = np.exp(-np.square(freqs) / (2 * sigma**2))
    F_filtered = F * window[:, None]
    filtered_array = np.fft.ifft(F_filtered, axis=0)
    return np.real(filtered_array)


def measure_total_size(tmpdir):
    import os

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(tmpdir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def estimate_noise_level(array):
    # use median absolute deviation as a robust estimator of noise level
    return 1.4826 * np.median(np.abs(array - np.median(array)))


if __name__ == "__main__":
    zarr_example()
