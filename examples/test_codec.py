import numpy as np
from qfc.codecs import QFCCodec
from qfc import qfc_estimate_quant_scale_factor

QFCCodec.register_codec()


def zarr_codec_test():
    import zarr
    import tempfile
    import os

    duration_sec = 6
    num_samples = 30000 * duration_sec
    segment_length = 10000
    chunks = (30000, 64)
    num_channels = 384
    target_residual_std = 4

    print(f'Duration = {duration_sec} sec; num_channels = {num_channels}')

    print('Defining array')
    array = (np.random.randn(num_samples, num_channels) * 5).astype("int16")
    print('Estimating quant scale factor')
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        array[:10000], target_residual_std=target_residual_std
    )

    print('Defining codec')
    codec = QFCCodec(
        quant_scale_factor=quant_scale_factor,
        dtype="int16",
        compression_method="zlib",
        segment_length=segment_length,
        zstd_level=3,
        zlib_level=3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        g = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="w")

        print('Writing compressed array')
        g.create_dataset(
            "array",
            data=array,
            chunks=chunks,
            compressor=codec
        )

        print('Measuring compressed size')
        compressed_size = measure_total_size(tmpdir)
        uncompressed_size = array.nbytes

        print('Uncompressed size (MB):', uncompressed_size / 1024 / 1024)
        print('Compressed size (MB):', compressed_size / 1024 / 1024)
        print('Compression ratio:', uncompressed_size / compressed_size)
        
        print('Reading compressed array')
        g2 = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="r")
        loaded_z = g2["array"][:]

        print('Checking residual stdev')
        resid_stdev = np.sqrt(np.var(array - loaded_z))

        print(f'Target residual stdev: {target_residual_std}')
        print(f"Residual stdev: {resid_stdev}")

        assert (
            target_residual_std - 0.5 < resid_stdev
            and resid_stdev < target_residual_std + 0.5
        )

def measure_total_size(tmpdir):
    import os
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(tmpdir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


if __name__ == "__main__":
    zarr_codec_test()
