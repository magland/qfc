import numpy as np
import spikeinterface.extractors as se


# imported/000784/sub-F/sub-F_ses-20230917_obj-34uga6_ecephys.nwb
# neurosift: https://flatironinstitute.github.io/neurosift/?p=/nwb&url=https://api.dandiarchive.org/api/assets/a04169c9-3f75-4dfa-b870-992cfccbde9a/download/&dandisetId=000784&dandisetVersion=draft&dandiAssetPath=sub-F%2Fsub-F_ses-20230917_obj-34uga6_ecephys.nwb
nwb_url = "https://api.dandiarchive.org/api/assets/a04169c9-3f75-4dfa-b870-992cfccbde9a/download/"
electrical_series_path = "/acquisition/ElectricalSeriesAP"
duration_sec = 6


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

    print("Writing traces to real_example_traces.npy")
    np.save("real_example_traces.npy", X)


if __name__ == "__main__":
    main()
