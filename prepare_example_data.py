import os
import json
import tempfile
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from matplotlib import pyplot as plt


# imported/000784/sub-F/sub-F_ses-20230917_obj-34uga6_ecephys.nwb
# neurosift: https://flatironinstitute.github.io/neurosift/?p=/nwb&url=https://api.dandiarchive.org/api/assets/a04169c9-3f75-4dfa-b870-992cfccbde9a/download/&dandisetId=000784&dandisetVersion=draft&dandiAssetPath=sub-F%2Fsub-F_ses-20230917_obj-34uga6_ecephys.nwb
nwb_url = "https://api.dandiarchive.org/api/assets/a04169c9-3f75-4dfa-b870-992cfccbde9a/download/"
electrical_series_path = "/acquisition/ElectricalSeriesAP"
duration_sec = 1


def main():
    print("Loading recording")
    recording_full = se.NwbRecordingExtractor(
        nwb_url,
        electrical_series_path=electrical_series_path,
        stream_mode="remfile",
    )
    sampling_frequency = recording_full.get_sampling_frequency()
    recording = recording_full.frame_slice(
        start_frame=0, end_frame=int(sampling_frequency * duration_sec)
    )

    print("Extracting traces")
    X: np.ndarray = recording.get_traces()
    # traces = traces[:, channel_order]

    print("Writing traces to example_traces.npy")
    np.save("example_traces.npy", X)

    print("Filtering recording")
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    X_filtered: np.ndarray = recording_filtered.get_traces()

    print("Writing filtered traces to example_traces_filtered.npy")
    np.save("example_traces_filtered.npy", X_filtered)


if __name__ == "__main__":
    main()
