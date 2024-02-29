from atexit import register
from typing import Literal
import numpy as np
from numcodecs.abc import Codec
from numcodecs import register_codec
from ..qfc_multi_segment_compress import qfc_multi_segment_compress, qfc_multi_segment_decompress

_special_int32_for_header = 7364182

class QFCCodec(Codec):
    codec_id = "qfc"

    def __init__(
        self,
        quant_scale_factor: float,
        dtype: str,
        segment_length: int,
        compression_method: Literal["zlib", "zstd"] = "zstd",
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
        # We need a small header to store the info needed to decompress
        num_samples = buf.shape[0]
        header = np.array([_special_int32_for_header, 1, num_samples, num_channels, self.segment_length]).astype(np.int32)
        return header.tobytes() + ret


    def decode(self, buf: bytes, out = None):
        header = np.frombuffer(buf, dtype=np.int32, count=5)
        if header[0] != _special_int32_for_header:
            raise Exception(f'qfc: invalid header[0]: {header[0]}')
        if header[1] != 1:
            raise Exception(f'qfc: invalid header[1]: {header[1]}')
        num_samples = header[2]
        num_channels = header[3]
        if header[4] != self.segment_length:
            raise Exception(f'qfc: unexpected segment length in header. Expected {self.segment_length}, got {header[3]}')
        if out is not None:
            if out.shape[1] != num_channels:
                raise Exception(f'Unexpected num. channels in out: expected {num_channels}, got {out.shape[1]}')
            if out.shape[0] != num_samples:
                raise Exception(f'Unexpected num. samples in out: expected {num_samples}, got {out.shape[0]}')
            if str(out.dtype) != self.dtype:
                raise Exception(f'Unexpected dtype in out: expected {self.dtype}, got {out.dtype}')
            if not out.flags['C_CONTIGUOUS']:
                raise Exception('qfc: expected out to be in C order')
        decompressed_array = qfc_multi_segment_decompress(
            buf[4 * 5:],  # skip the header
            dtype=self.dtype,
            num_channels=num_channels,
            segment_length=self.segment_length,
            quant_scale_factor=self.quant_scale_factor,
            compression_method=self.compression_method,
        )
        if decompressed_array.shape[0] != num_samples:
            raise Exception(f'qfc: unexpected num samples in decompressed array. Expected {num_samples}, got {decompressed_array.shape[0]}')
        # check if buf is C order
        if not decompressed_array.flags['C_CONTIGUOUS']:
            raise Exception('qfc: expected decompressed_array to be in C order')
        if out is not None:
            np.copyto(out, decompressed_array)
        else:
            out = decompressed_array
        return out

    def __repr__(self):
        return (
            f'QFCCodec (quant_scale_factor={self.quant_scale_factor}, dtype={self.dtype}, segment_length={self.segment_length}, compression_method={self.compression_method}, zstd_level={self.zstd_level}, zlib_level={self.zlib_level})'
        )

    @staticmethod
    def register_codec():
        register_codec(QFCCodec)
