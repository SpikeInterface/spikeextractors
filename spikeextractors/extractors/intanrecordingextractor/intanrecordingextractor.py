from spikeextractors import RecordingExtractor, SortingExtractor
import numpy as np
from pathlib import Path


class IntanRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_file, *, experiment_id=0, recording_id=0):
        try:
            import pyintan
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Intan extractor, install pyintan: \n\n"
                                      "pip install pyintan\n\n")
        RecordingExtractor.__init__(self)
        assert Path(recording_file).suffix == '.rhs' or Path(recording_file).suffix == '.rhd', \
            "Only '.rhd' and '.rhs' files are supported"
        self._recording_file = recording_file
        self._recording = pyintan.File(recording_file)

    def get_channel_ids(self):
        return list(range(self._recording.analog_signals[0].signal.shape[0]))

    def get_num_frames(self):
        return self._recording.analog_signals[0].signal.shape[1]

    def get_sampling_frequency(self):
        return float(self._recording.sample_rate.rescale('Hz').magnitude)

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()

        return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame]
