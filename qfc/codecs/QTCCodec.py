from typing import Literal
import numpy as np
from numcodecs.abc import Codec
from numcodecs import register_codec
from ..internal.qtc_compress import qtc_compress, qtc_decompress

_special_qtc_int32_for_header = 8364182


class QTCCodec(Codec):
    """
    Codec providing compression using the quantization in the time domain
    """
    codec_id = "qfc_qtc"

    def __init__(
        self,
        quant_scale_factor: float,
        dtype: str,  # e.g. "float32" or "int16"
        compression_method: Literal["zlib", "zstd"] = "zstd",
        zstd_level: int = 3,
        zlib_level: int = 3,
    ):
        """
        Parameters
        ----------
        quant_scale_factor : float
            The scale factor to use during quantization, obtained from
            qtc_estimate_quant_scale_factor
        dtype : str
            The dtype of the input array as a string (e.g. "float32" or "int16")
        compression_method : Literal["zlib", "zstd"]
            The compression method to use
        zstd_level : int
            The compression level to use for zstd
        zlib_level : int
            The compression level to use for zlib
        """
        self.quant_scale_factor = quant_scale_factor
        self.dtype = dtype
        self.compression_method: Literal["zlib", "zstd"] = compression_method
        self.zstd_level = zstd_level
        self.zlib_level = zlib_level

    def encode(self, array: np.ndarray):  # type: ignore
        """
        Compress the input array using quantization in the time domain

        Parameters
        ----------
        array : np.ndarray
            The input array to be compressed
            The dimensions must be (num_samples, num_channels)
            It must have a dtype matching the dtype parameter of this codec
            This array must be C-contiguous
        """
        num_channels = array.shape[1] if array.ndim > 1 else 1
        if array.ndim > 2:
            raise Exception(f'qtc: expected ndims <= 2, got {array.ndim}')
        if self.dtype != str(array.dtype):
            raise Exception(f'qtc: expected dtype = {self.dtype}, got {array.dtype}')
        # check if buf is C order
        if not array.flags['C_CONTIGUOUS']:
            raise Exception('qtc: expected buf to be in C order')
        ret = qtc_compress(
            array,
            quant_scale_factor=self.quant_scale_factor,
            compression_method=self.compression_method,
            zstd_level=self.zstd_level,
            zlib_level=self.zlib_level
        )
        # We need a small header to store the info needed to decompress
        num_samples = array.shape[0]
        header = np.array([_special_qtc_int32_for_header, 1, num_samples, num_channels]).astype(np.int32)
        return header.tobytes() + ret

    def decode(self, buf: bytes, out=None):  # type: ignore
        """
        Decompress the input buffer using quantization in the time domain

        Parameters
        ----------
        buf : bytes
            The input buffer to be decompressed
        out : np.ndarray, optional
            If provided, the decompressed array will be written into this array
            If not provided, a new array will be allocated
        """
        header = np.frombuffer(buf, dtype=np.int32, count=4)
        if header[0] != _special_qtc_int32_for_header:
            raise Exception(f'qtc: invalid header[0]: {header[0]}')
        if header[1] != 1:
            raise Exception(f'qtc: invalid header[1]: {header[1]}')
        num_samples = header[2]
        num_channels = header[3]
        if out is not None:
            if out.shape[1] != num_channels:
                raise Exception(f'Unexpected num. channels in out: expected {num_channels}, got {out.shape[1]}')
            if out.shape[0] != num_samples:
                raise Exception(f'Unexpected num. samples in out: expected {num_samples}, got {out.shape[0]}')
            if str(out.dtype) != self.dtype:
                raise Exception(f'Unexpected dtype in out: expected {self.dtype}, got {out.dtype}')
            if not out.flags['C_CONTIGUOUS']:
                raise Exception('qtc: expected out to be in C order')
        if out is not None:
            if out.shape[1] != num_channels:
                raise Exception(f'qtc: Unexpected num. channels in out: expected {num_channels}, got {out.shape[1]}')
            if out.shape[0] != num_samples:
                raise Exception(f'qtc: Unexpected num. samples in out: expected {num_samples}, got {out.shape[0]}')
            if str(out.dtype) != self.dtype:
                raise Exception(f'qtc: Unexpected dtype in out: expected {self.dtype}, got {out.dtype}')
            if not out.flags['C_CONTIGUOUS']:
                raise Exception('qtc: expected out to be in C order')
        decompressed_array = qtc_decompress(
            buf[4 * 4:],  # skip the header
            dtype=self.dtype,
            num_channels=num_channels,
            quant_scale_factor=self.quant_scale_factor,
            compression_method=self.compression_method
        )
        if decompressed_array.shape[0] != num_samples:
            raise Exception(f'qtc: unexpected num samples in decompressed array. Expected {num_samples}, got {decompressed_array.shape[0]}')
        if not decompressed_array.flags['C_CONTIGUOUS']:
            raise Exception('qtc: expected decompressed_array to be in C order')
        if out is not None:
            out[:] = decompressed_array
            return out
        else:
            return decompressed_array

    def __repr__(self):
        return (
            f'QTCCodec (quant_scale_factor={self.quant_scale_factor}, dtype={self.dtype}, compression_method={self.compression_method}, zstd_level={self.zstd_level}, zlib_level={self.zlib_level})'
        )

    @staticmethod
    def register_codec():
        register_codec(QTCCodec)
