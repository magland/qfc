from typing import Union, Literal
import numpy as np
from numcodecs.abc import Codec
from numcodecs import register_codec
from ..internal.qwc_multi_segment_compress import qwc_multi_segment_compress, qwc_multi_segment_decompress

_special_qwc_int32_for_header = 9364182


class QWCCodec(Codec):
    """
    Codec providing timeseries compression using quantized wavelet coefficients
    """
    codec_id = "qfc_qwc"

    def __init__(
        self,
        *,
        quant_scale_factor: float,
        dtype: str,  # e.g. "float32" or "int16"
        segment_length: int,
        pywt_wavelet: str,
        pywt_level: Union[int, None] = None,
        pywt_mode: str = "symmetric",
        compression_method: Literal["zlib", "zstd"] = "zstd",
        zstd_level: int = 3,
        zlib_level: int = 3,
    ):
        """
        Parameters
        ----------
        quant_scale_factor : float
            The scale factor to use during quantization, obtained from
            qwc_estimate_quant_scale_factor
        dtype : str
            The dtype of the input array as a string (e.g. "float32" or "int16")
        segment_length : int
            See below. Setting to zero turns off segmentation.
        pywt_wavelet : str
            The wavelet parameter for PyWavelets
        pywt_level : int or None
            The level parameter for PyWavelets
        pywt_mode : str
            The signal extension mode parameter for PyWavelets
        compression_method : Literal["zlib", "zstd"]
            The compression method to use
        zstd_level : int
            The compression level to use for zstd
        zlib_level : int
            The compression level to use for zlib

        How segment_length affects compression: When segment_length is non-zero,
        the input array is split into segments of length segment_length. If the
        segment_length doesn't divide evenly into the length of the input array,
        the last segment will be shorter. If the last segment is to be shorter
        than segment_length/2, an adjustment is made to the previous segment to
        make the last segment length equal to segment_length / 2.
        """
        self.quant_scale_factor = quant_scale_factor
        self.dtype = dtype
        self.segment_length = segment_length
        self.pywt_wavelet = pywt_wavelet
        self.pywt_level = pywt_level
        self.pywt_mode = pywt_mode
        self.compression_method: Literal["zlib", "zstd"] = compression_method
        self.zstd_level = zstd_level
        self.zlib_level = zlib_level

    def encode(self, array: np.ndarray):  # type: ignore
        """
        Compress the input timeseries using quantization of wavelet coefficients

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
            raise Exception(f'qwc: expected ndims <= 2, got {array.ndim}')
        if self.dtype != str(array.dtype):
            raise Exception(f'qwc: expected dtype = {self.dtype}, got {array.dtype}')
        # check if buf is C order
        if not array.flags['C_CONTIGUOUS']:
            raise Exception('qwc: expected buf to be in C order')
        ret = qwc_multi_segment_compress(
            array,
            segment_length=self.segment_length,
            quant_scale_factor=self.quant_scale_factor,
            pywt_wavelet=self.pywt_wavelet,
            pywt_level=self.pywt_level,
            pywt_mode=self.pywt_mode,
            compression_method=self.compression_method,
            zstd_level=self.zstd_level,
            zlib_level=self.zlib_level
        )
        # We need a small header to store the info needed to decompress
        num_samples = array.shape[0]
        header = np.array([_special_qwc_int32_for_header, 1, num_samples, num_channels, self.segment_length]).astype(np.int32)
        return header.tobytes() + ret

    def decode(self, buf: bytes, out=None):  # type: ignore
        """
        Decompress the input buffer using quantization of wavelet coefficients

        Parameters
        ----------
        buf : bytes
            The input buffer to be decompressed
        out : np.ndarray, optional
            If provided, the decompressed array will be written into this array
            If not provided, a new array will be allocated
        """
        header = np.frombuffer(buf, dtype=np.int32, count=5)
        if header[0] != _special_qwc_int32_for_header:
            raise Exception(f'qwc: invalid header[0]: {header[0]}')
        if header[1] != 1:
            raise Exception(f'qwc: invalid header[1]: {header[1]}')
        num_samples = header[2]
        num_channels = header[3]
        if header[4] != self.segment_length:
            raise Exception(f'qfc: unexpected segment length in header. Expected {self.segment_length}, got {header[4]}')
        if out is not None:
            if out.shape[1] != num_channels:
                raise Exception(f'Unexpected num. channels in out: expected {num_channels}, got {out.shape[1]}')
            if out.shape[0] != num_samples:
                raise Exception(f'Unexpected num. samples in out: expected {num_samples}, got {out.shape[0]}')
            if str(out.dtype) != self.dtype:
                raise Exception(f'Unexpected dtype in out: expected {self.dtype}, got {out.dtype}')
            if not out.flags['C_CONTIGUOUS']:
                raise Exception('qwc: expected out to be in C order')
        if out is not None:
            if out.shape[1] != num_channels:
                raise Exception(f'qwc: Unexpected num. channels in out: expected {num_channels}, got {out.shape[1]}')
            if out.shape[0] != num_samples:
                raise Exception(f'qwc: Unexpected num. samples in out: expected {num_samples}, got {out.shape[0]}')
            if str(out.dtype) != self.dtype:
                raise Exception(f'qwc: Unexpected dtype in out: expected {self.dtype}, got {out.dtype}')
            if not out.flags['C_CONTIGUOUS']:
                raise Exception('qwc: expected out to be in C order')
        decompressed_array = qwc_multi_segment_decompress(
            buf[4 * 5:],  # skip the header
            segment_length=self.segment_length,
            dtype=self.dtype,
            quant_scale_factor=self.quant_scale_factor,
            num_channels=num_channels,
            pywt_wavelet=self.pywt_wavelet,
            pywt_mode=self.pywt_mode,
            compression_method=self.compression_method
        )
        if decompressed_array.shape[0] != num_samples:
            raise Exception(f'qwc: unexpected num samples in decompressed array. Expected {num_samples}, got {decompressed_array.shape[0]}')
        if not decompressed_array.flags['C_CONTIGUOUS']:
            raise Exception('qwc: expected decompressed_array to be in C order')
        if out is not None:
            out[:] = decompressed_array
            return out
        else:
            return decompressed_array

    def __repr__(self):
        return (
            f'QWCCodec (quant_scale_factor={self.quant_scale_factor}, dtype={self.dtype}, wavelet={self.pywt_wavelet}, level={self.pywt_level}, mode={self.pywt_mode}, compression_method={self.compression_method}, zstd_level={self.zstd_level}, zlib_level={self.zlib_level})'
        )

    @staticmethod
    def register_codec():
        register_codec(QWCCodec)
