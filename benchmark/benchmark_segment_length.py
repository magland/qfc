from typing import List
from models import BenchmarkTestParams, BenchmarkTestResult
from run_benchmark_test import run_benchmark_test
from qfc.codecs import QFCCodec, QTCCodec

QFCCodec.register_codec()
QTCCodec.register_codec()

show_plots = False
duration_sec = 2
relative_target_residual_stdev = 0.3
method = 'qfc'
compression_level = 3


def main():
    test_params: List[BenchmarkTestParams] = []
    for data_path in ['real_example_traces.npy']:
        for bandpass_filter in [True, False]:
            for segment_length_sec in [0.02, 0.1, 1, 2]:
                name = method
                if data_path:
                    name += '_real'
                else:
                    name += '_gaussian'
                if bandpass_filter:
                    name += '_bandpass'
                else:
                    name += '_nofilt'
                name += f'_seg{segment_length_sec}'
                test_params.append(BenchmarkTestParams(
                    name=name,
                    sampling_frequency=30000,
                    duration_sec=duration_sec,
                    num_channels=384,
                    data_path=data_path,
                    chunk_size_sec=2,
                    bandpass_filter=bandpass_filter,
                    method=method,
                    compression_method="zstd",
                    compression_level=compression_level,
                    segment_length_sec=segment_length_sec,
                    relative_target_residual_stdev=relative_target_residual_stdev
                ))

    results: List[BenchmarkTestResult] = []
    for t in test_params:
        print(f'RUNNING TEST: {t.name}')
        r = run_benchmark_test(t, show_plots=show_plots)
        results.append(r)
        print(r)
        print('')
        print(r.params.name)
        print(f'  Compression ratio: {r.compression_ratio}')
        print(f'  Elapsed compression time: {r.elapsed_compression_sec}')
        print(f'  Elapsed decompression time: {r.elapsed_decompression_sec}')
        print(f'  Residual stdev: {r.residual_stdev}')
        print('')

    print('RESULTS:')
    for r in results:
        print(r.params.name)
        print(f'  Compression ratio: {r.compression_ratio}')
        print(f'  Elapsed compression time: {r.elapsed_compression_sec}')
        print(f'  Elapsed decompression time: {r.elapsed_decompression_sec}')
        print(f'  Residual stdev: {r.residual_stdev}')
        print('')

    for r in results:
        print(f'{r.params.name} | cr: {r.compression_ratio} | comp: {r.elapsed_compression_sec} | decomp: {r.elapsed_decompression_sec} | stdev: {r.residual_stdev}')


if __name__ == "__main__":
    main()
