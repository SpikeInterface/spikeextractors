from spikeextractors import RecordingExtractor, SortingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args, check_get_unit_spike_train, check_get_ttl_args

try:
    import pyopenephys

    HAVE_OE = True
except ImportError:
    HAVE_OE = False


class OpenEphysRecordingExtractor(RecordingExtractor):
    extractor_name = 'OpenEphysRecording'
    has_default_locations = False
    installed = HAVE_OE  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"  # error message when not installed

    def __init__(self, folder_path, *, experiment_id=0, recording_id=0):
        assert HAVE_OE, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._recording_file = folder_path

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'experiment_id': experiment_id,
                        'recording_id': recording_id}
        
        self._file_obj = pyopenephys.File(folder_path)
        self._recording = self._file_obj.experiments[experiment_id].recordings[recording_id]
        
        # Set gains: int16 to uV
        self.set_channel_gains(gains=self._recording.analog_signals[0].gains)

    def get_channel_ids(self):
        return list(range(self._recording.analog_signals[0].signal.shape[0]))

    def get_num_frames(self):
        return self._recording.analog_signals[0].signal.shape[1]

    def get_sampling_frequency(self):
        return float(self._recording.sample_rate.rescale('Hz').magnitude)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        if return_scaled:  # Returns traces as uV
            return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame] * np.array([self.get_channel_gains(channel_ids=channel_ids)]).T
        else:   # Returns traces as int16 
            return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame]

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        channels = [np.unique(ev.channels)[0] for ev in self._recording.events]
        assert channel_id in channels, f"Specified 'channel' not found. Available channels are {channels}"
        ev = self._recording.events[channels.index(channel_id)]

        ttl_frames = (ev.times.rescale("s") * self.get_sampling_frequency()).magnitude.astype(int)
        ttl_states = np.sign(ev.channel_states)
        ttl_valid_idxs = np.where((ttl_frames >= start_frame) & (ttl_frames < end_frame))[0]
        return ttl_frames[ttl_valid_idxs], ttl_states[ttl_valid_idxs]


class OpenEphysSortingExtractor(SortingExtractor):
    extractor_name = 'OpenEphysSortingExtractor'
    installed = HAVE_OE  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"  # error message when not installed

    def __init__(self, folder_path, *, experiment_id=0, recording_id=0):
        assert HAVE_OE, self.installation_mesg
        SortingExtractor.__init__(self)
        self._recording_file = folder_path
        self._recording = pyopenephys.File(folder_path).experiments[experiment_id].recordings[recording_id]
        self._spiketrains = self._recording.spiketrains
        self._unit_ids = list([np.unique(st.clusters)[0] for st in self._spiketrains])
        self._sampling_frequency = float(self._recording.sample_rate.rescale('Hz').magnitude)

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'experiment_id': experiment_id,
                        'recording_id': recording_id}

    def get_unit_ids(self):
        return self._unit_ids

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

        st = self._spiketrains[unit_id]
        inds = np.where((start_frame <= (st.times * self._recording.sample_rate)) &
                        ((st.times * self._recording.sample_rate) < end_frame))
        return (st.times[inds] * self._recording.sample_rate).magnitude
