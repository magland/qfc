from pydantic import BaseModel, Field


class BenchmarkTestParams(BaseModel):
    name: str = Field(..., title="Name of the benchmark")
    sampling_frequency: int = Field(..., title="Sampling frequency in Hz")
    duration_sec: float = Field(..., title="Duration of the data in seconds")
    data_path: str = Field(..., title="Path to the raw data file")
    num_channels: int = Field(..., title="Number of channels in the data")
    chunk_size_sec: int = Field(..., title="Chunk size in num. samples used for zarr storage")
    bandpass_filter: bool = Field(..., title="Whether to apply a bandpass filter to the data")
    method: str = Field(..., title="Method used to perform the benchmark, 'qfc' or 'qtc'")
    compression_method: str = Field(..., title="Compression method used, 'zstd' or 'zlib'")
    compression_level: int = Field(..., title="Compression level used")
    segment_length_sec: float = Field(..., title="Segment length used for zstd method")
    relative_target_residual_stdev: float = Field(..., title="Target residual standard deviation in the data relative to estimated noise level")


class BenchmarkTestResult(BaseModel):
    params: BenchmarkTestParams = Field(..., title="Parameters used for the benchmark")
    uncompressed_size: int = Field(..., title="Original size of the data in bytes")
    compressed_size: int = Field(..., title="Compressed size of the data in bytes")
    compression_ratio: float = Field(..., title="Compression ratio")
    elapsed_compression_sec: float = Field(..., title="Elapsed time to compress the data in seconds")
    elapsed_decompression_sec: float = Field(..., title="Elapsed time to decompress the data in seconds")
    residual_stdev: float = Field(..., title="Actual residual standard deviation in the data")
    relative_residual_stdev: float = Field(..., title="Actual relative residual standard deviation in the data compared to estimated noise level")
