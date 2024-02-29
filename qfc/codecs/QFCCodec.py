from typing import Literal
import numpy as np
from numcodecs.abc import Codec
from ..qfc_multi_segment_compress import qfc_multi_segment_compress, qfc_multi_segment_decompress


# flake8: noqa: E501


class QFCCodec(Codec):
    codec_id = "qfc"

    def __init__(
        self,
        quant_scale_factor: float,
        dtype: str,
        segment_length: int,
        compression_method: Literal["zlib", "zstd"] = "zlib",
        zstd_level: int = 3,
        zlib_level: int = 3,
    ):
        self.quant_scale_factor = quant_scale_factor
        self.dtype = dtype
        self.segment_length = segment_length
        self.compression_method: Literal["zlib", "zstd"] = compression_method
        self.zstd_level = zstd_level
        self.zlib_level = zlib_level


    def encode(self, buf: np.ndarray):
        print('----- encode', buf.shape)
        num_channels = buf.shape[1] if buf.ndim > 1 else 1
        if buf.ndim > 2:
            raise Exception(f'qfc: expected ndims <= 2, got {buf.ndim}')
        if self.dtype != str(buf.dtype):
            raise Exception(f'qfc: expected dtype = {self.dtype}, got {buf.dtype}')
        # check if buf is C order
        if not buf.flags['C_CONTIGUOUS']:
            raise Exception('qfc: expected buf to be in C order')
        ret = qfc_multi_segment_compress(
            buf,
            quant_scale_factor=self.quant_scale_factor,
            compression_method=self.compression_method,
            segment_length=self.segment_length,
            zstd_level=self.zstd_level,
            zlib_level=self.zlib_level
        )
        # We need a small header to store the num_samples and num_channels
        num_samples = buf.shape[0]
        header = np.array([1, num_samples, num_channels]).astype(np.int32)
        return header.tobytes() + ret


    def decode(self, buf: bytes, out = None):
        header = np.frombuffer(buf, dtype=np.int32, count=3)
        if header[0] != 1:
            raise Exception(f'qfc: invalid header: {header[0]}')
        num_samples = header[1]
        num_channels = header[2]
        print('--- decode', num_samples, num_channels)
        if out is not None:
            if out.shape[1] != num_channels:
                raise Exception(f'Unexpected num. channels in out: expected {num_channels}, got {out.shape[1]}')
        decompressed_array = qfc_multi_segment_decompress(
            buf[12:],
            dtype=self.dtype,
            num_channels=num_channels,
            segment_length=self.segment_length,
            quant_scale_factor=self.quant_scale_factor,
            compression_method=self.compression_method,
        )
        if decompressed_array.shape[0] != num_samples:
            raise Exception(f'qfc: unexpected num samples in decompressed array. Expected {num_samples}, got {decompressed_array.shape[0]}')
        if out is not None:
            np.copyto(out, decompressed_array)
        else:
            out = decompressed_array
        return out


    def __repr__(self):
        return f"QFCCodec(quant_scale_factor={self.quant_scale_factor}, compression_method={self.compression_method}, zstd_level={self.zstd_level}, zlib_level={self.zlib_level})"
