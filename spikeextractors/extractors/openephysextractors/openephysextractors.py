from spikeextractors import RecordingExtractor, SortingExtractor
import numpy as np


try:
    import pyopenephys
    HAVE_OE = True
except ImportError:
    HAVE_OE = False

class OpenEphysRecordingExtractor(RecordingExtractor):

    extractor_name = 'OpenEphysRecordingExtractor'
    installed = HAVE_OE  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording_file', 'type': 'path', 'title': "str, Path to file"},
        {'name': 'experiment_id', 'type': 'int', 'value':0, 'default':0, 'title': "Experiment ID"},
        {'name': 'recording_id', 'type': 'int', 'value':0, 'default':0, 'title': "Recording ID"},
        {'name': 'dtype', 'type': 'str',  'value':'float', 'default':'float', 'title':"dtype ('float' or 'int')"},
        {'name': 'probe_path', 'type': 'path', 'title': "Path to probe file (.csv or .prb)"}
    ]

    installation_mesg = "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"  # error message when not installed

    def __init__(self, recording_file, *, experiment_id=0, recording_id=0, dtype='float'):
        assert HAVE_OE, "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"
        assert dtype == 'int16' or 'float' in dtype, "'dtype' can be int16 (memory map) or 'float' (load into memory)"
        RecordingExtractor.__init__(self)
        self._recording_file = recording_file
        self._recording = pyopenephys.File(recording_file).experiments[experiment_id].recordings[recording_id]
        self._dtype = dtype

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
        if self._dtype == 'int16':
            return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame]
        elif self._dtype == 'float':
            return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame] * \
                   self._recording.analog_signals[0].gain


class OpenEphysSortingExtractor(SortingExtractor):

    extractor_name = 'OpenEphysSortingExtractor'
    installed = HAVE_OE  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording_file', 'type': 'file_path', 'title': "str, Path to file"},
        {'name': 'experiment_id', 'type': 'int', 'value':0, 'default':0, 'title': "Experiment ID"},
        {'name': 'recording_id', 'type': 'int', 'value':0, 'default':0, 'title': "Recording ID"},
    ]

    installation_mesg = "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"  # error message when not installed

    def __init__(self, recording_file, *, experiment_id=0, recording_id=0):
        assert HAVE_OE, "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"
        SortingExtractor.__init__(self)
        self._recording_file = recording_file
        self._recording = pyopenephys.File(recording_file).experiments[experiment_id].recordings[recording_id]
        self._spiketrains = self._recording.spiketrains
        self._unit_ids = list([np.unique(st.clusters)[0] for st in self._spiketrains])
        self._sampling_frequency = float(self._recording.sample_rate.rescale('Hz').magnitude)

    def get_unit_ids(self):
        return self._unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        st = self._spiketrains[unit_id]
        inds = np.where((start_frame <= (st.times * self._recording.sample_rate)) &
                        ((st.times * self._recording.sample_rate) < end_frame))
        return (st.times[inds] * self._recording.sample_rate).magnitude
