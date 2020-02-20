from spikeextractors import RecordingExtractor, SortingExtractor
import numpy as np
from pathlib import Path

try:
    import pyintan
    HAVE_INTAN = True
except ImportError:
    HAVE_INTAN = False

class IntanRecordingExtractor(RecordingExtractor):

    extractor_name = 'IntanRecordingExtractor'
    has_default_locations = False
    is_writable = False
    mode = 'file'
    installed = HAVE_INTAN  # check at class level if installed or not
    extractor_gui_params = [
        {'name': 'file_path', 'type': 'file', 'title': "Path to file (.rhs or .rhd)"},
    ]
    installation_mesg = "To use the Intan extractor, install pyintan: \n\n pip install pyintan\n\n"  # error message when not installed

    def __init__(self, file_path, verbose=False):
        assert HAVE_INTAN, "To use the Intan extractor, install pyintan: \n\n pip install pyintan\n\n"
        RecordingExtractor.__init__(self)
        assert Path(file_path).suffix == '.rhs' or Path(file_path).suffix == '.rhd', \
            "Only '.rhd' and '.rhs' files are supported"
        self._recording_file = file_path
        self._recording = pyintan.File(file_path, verbose)

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
