import time
from matplotlib import pyplot as plt
import numpy as np
from qfc import qwc_estimate_quant_scale_factor, qfc_estimate_quant_scale_factor, qtc_estimate_quant_scale_factor
from qfc.codecs import QFCCodec, QWCCodec, QTCCodec


def main():
    y, sampling_frequency = load_traces()
    for method in ['qfc', 'qtc', 'qwc']:
        # for method in ['qwc']:

        target_residual_stdev = 7
        segment_length = 10000

        ############################################################
        if method == 'qfc':
            quant_scale_factor = qfc_estimate_quant_scale_factor(
                y,
                target_residual_stdev=target_residual_stdev
            )
            codec = QFCCodec(
                quant_scale_factor=quant_scale_factor,
                dtype="int16",
                segment_length=segment_length,
                compression_method="zstd",
                zstd_level=3
            )
        elif method == 'qtc':
            quant_scale_factor = qtc_estimate_quant_scale_factor(
                y,
                target_residual_stdev=target_residual_stdev
            )
            codec = QTCCodec(
                quant_scale_factor=quant_scale_factor,
                dtype="int16",
                compression_method="zstd",
                zstd_level=3
            )
        elif method == 'qwc':
            quant_scale_factor = qwc_estimate_quant_scale_factor(
                y,
                target_residual_stdev=target_residual_stdev,
                pywt_wavelet='db4',
                pywt_level=4,
                pywt_mode='symmetric',
                segment_length=segment_length
            )
            codec = QWCCodec(
                quant_scale_factor=quant_scale_factor,
                dtype="int16",
                segment_length=segment_length,
                pywt_wavelet='db4',
                pywt_level=None,
                pywt_mode='symmetric',
                compression_method="zstd",
                zstd_level=3
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        timer = time.time()
        compressed_bytes = codec.encode(y)
        elapsed_compress = time.time() - timer

        timer = time.time()
        y_reconstructed = codec.decode(compressed_bytes)
        elapsed_decompress = time.time() - timer
        ############################################################

        y_resid = y - y_reconstructed
        original_size = y.nbytes
        compressed_size = len(compressed_bytes)
        compression_ratio = original_size / compressed_size
        print(method)
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Actual compression ratio: {compression_ratio}")
        print(f'Target residual std. dev.: {target_residual_stdev:.2f}')
        print(f'Actual Std. dev. of residual: {np.std(y_resid):.2f}')
        print(f'Compression time: {elapsed_compress:.2f} s')
        print(f'Decompression time: {elapsed_decompress:.2f} s')
        print('')

        xgrid = np.arange(y.shape[0]) / sampling_frequency
        ch = 23  # select a channel to plot
        n = 5000  # number of samples to plot
        plt.figure()
        plt.plot(xgrid[:n], y[:n, ch], label="Original")
        plt.plot(xgrid[:n], y_reconstructed[:n, ch], label="Decompressed")
        plt.plot(xgrid[:n], y_resid[:n, ch], label="Residual")
        plt.xlabel("Time")
        plt.title(f'{method} compression ratio: {compression_ratio:.2f}')
        plt.legend()
        plt.show()


def load_traces():
    import lindi

    cache = lindi.LocalCache()

    # url = 'https://lindi.neurosift.org/dandi/dandisets/000463/assets/2e6b590a-a2a4-4455-bb9b-45cc3d7d7cc0/nwb.lindi.json'
    # electrical_series_path = '/acquisition/ElectricalSeries'
    # do_normalize = True
    # do_filter = False
    # sampling_frequency = 24414

    url = 'https://lindi.neurosift.org/dandi/dandisets/000409/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/nwb.lindi.json'
    electrical_series_path = '/acquisition/ElectricalSeriesAp'
    do_filter = True
    do_normalize = True
    sampling_frequency = 30000

    # Load the remote file
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=cache)

    # load the neurodata object
    X = f[electrical_series_path]
    assert isinstance(X, lindi.LindiH5pyGroup)
    data = X['data']
    assert isinstance(data, lindi.LindiH5pyDataset)

    x = data[:90000, :]
    assert isinstance(x, np.ndarray)

    if do_filter:
        x = bandpass_filter(x, sampling_frequency, 300, 6000)
        x = x.astype(np.int16)

    if do_normalize:
        stdev = float(np.std(x))
        x = x * (20 / stdev)
        x = x.astype(np.int16)

    return x, sampling_frequency


def bandpass_filter(array, sampling_frequency, lowcut, highcut) -> np.ndarray:
    from scipy.signal import butter, lfilter

    nyquist = 0.5 * sampling_frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype="band")
    return lfilter(b, a, array, axis=0)  # type: ignore


if __name__ == "__main__":
    main()
