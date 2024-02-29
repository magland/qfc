import time
import numpy as np
from qfc.codecs import QFCCodec
from qfc import qfc_estimate_quant_scale_factor
from numcodecs import register_codec

register_codec(QFCCodec)


def codec_test():
    num_samples = 20000
    segment_length = 9000
    num_channels = 3
    target_residual_std = 2
    array = (np.random.randn(num_samples, num_channels) * 50).astype("float32")
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        array, target_residual_std=target_residual_std
    )
    codec = QFCCodec(
        quant_scale_factor=quant_scale_factor,
        dtype="float32",
        compression_method="zlib",
        segment_length=segment_length,
        zstd_level=3,
        zlib_level=3,
    )
    encoded = codec.encode(array)
    decoded = codec.decode(encoded)
    residual_stdev = np.sqrt(np.var(decoded - array))
    print(residual_stdev)
    assert (
        target_residual_std - 0.5 < residual_stdev
        and residual_stdev < target_residual_std + 0.5
    )


def zarr_codec_test():
    import zarr
    import tempfile
    import os

    num_samples = 4000
    num_channels = 3
    target_residual_std = 2
    array = (np.random.randn(num_samples, num_channels) * 50).astype("float32")
    quant_scale_factor = qfc_estimate_quant_scale_factor(
        array, target_residual_std=target_residual_std
    )

    codec = QFCCodec(
        quant_scale_factor=quant_scale_factor,
        dtype="float32",
        compression_method="zlib",
        segment_length=9000,
        zstd_level=3,
        zlib_level=3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        g = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="w")

        g.create_dataset(
            "array",
            data=array,
            chunks=(1000, 1),
            compressor=codec
        )

        g2 = zarr.open_group(store=os.path.join(tmpdir, "array.zarr"), mode="r")
        loaded_z = g2["array"][:]

        resid_stdev = np.sqrt(np.var(array - loaded_z))

        print(resid_stdev)

        print(array[:2, :])
        print(loaded_z[:2, :])

        assert (
            target_residual_std - 0.5 < resid_stdev
            and resid_stdev < target_residual_std + 0.5
        )


if __name__ == "__main__":
    codec_test()
    zarr_codec_test()
