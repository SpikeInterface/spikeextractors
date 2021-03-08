from spikeextractors import RecordingExtractor, SortingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args, check_get_unit_spike_train, check_get_ttl_args
from distutils.version import StrictVersion
import warnings


try:
    import pyopenephys
    HAVE_OE = True

    if pyopenephys.__version__ >= StrictVersion("1.1.2"):
        HAVE_OE_11 = True
    else:
        warnings.warn("pyopenephys>=1.1.2 should be installed. Support for older versions will be removed in "
                      "future releases. Install with:\n\n pip install --upgrade pyopenephys\n\n")
        HAVE_OE_11 = False
except ImportError:
    HAVE_OE = False
    HAVE_OE_11 = False

extractors_dir = Path(__file__).parent.parent


class OpenEphysRecordingExtractor(RecordingExtractor):
    extractor_name = 'OpenEphysRecording'
    has_default_locations = False
    has_unscaled = True
    installed = HAVE_OE  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"

    def __init__(self, folder_path, experiment_id=0, recording_id=0):
        assert self.installed, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._recording_file = folder_path

        self._fileobj = pyopenephys.File(folder_path)
        self._recording = self._fileobj.experiments[experiment_id].recordings[recording_id]
        self._set_analogsignal(self._recording.analog_signals[0])
        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'experiment_id': experiment_id,
                        'recording_id': recording_id}

    def _set_analogsignal(self, analogsignals):
        self._analogsignals = analogsignals
        # Set gains: int16 to uV
        if HAVE_OE_11:
            self.set_channel_gains(gains=self._analogsignals.gains)
        else:
            self.set_channel_gains(gains=self._analogsignals.gain)

    def get_channel_ids(self):
        if HAVE_OE_11:
            return list(self._analogsignals.channel_ids)
        else:
            return list(range(self._analogsignals.signal.shape[0]))

    def get_num_frames(self):
        return self._analogsignals.signal.shape[1]

    def get_sampling_frequency(self):
        if HAVE_OE_11:
            return self._analogsignals.sample_rate
        else:
            return float(self._recording.sample_rate.rescale('Hz').magnitude)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        return self._analogsignals.signal[channel_ids, start_frame:end_frame]

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        channels = [np.unique(ev.channels)[0] for ev in self._recording.events]
        assert channel_id in channels, f"Specified 'channel' not found. Available channels are {channels}"
        ev = self._recording.events[channels.index(channel_id)]

        ttl_frames = (ev.times.rescale("s") * self.get_sampling_frequency()).magnitude.astype(int)
        ttl_states = np.sign(ev.channel_states)
        ttl_valid_idxs = np.where((ttl_frames >= start_frame) & (ttl_frames < end_frame))[0]
        return ttl_frames[ttl_valid_idxs], ttl_states[ttl_valid_idxs]


class OpenEphysNPIXRecordingExtractor(OpenEphysRecordingExtractor):
    extractor_name = 'OpenEphysNPIXRecording'
    has_default_locations = False
    installed = HAVE_OE_11  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = "To use the OpenEphys extractor, " \
                        "install pyopenephys >= 1.1: \n\n pip install pyopenephys>=1.1\n\n"

    def __init__(self, folder_path, experiment_id=0, recording_id=0, stream="AP"):
        assert self.installed, self.installation_mesg
        assert stream.upper() in ["AP", "LFP"]
        OpenEphysRecordingExtractor.__init__(self, folder_path, experiment_id, recording_id)

        analogsignals = self._recording.analog_signals
        for analog in analogsignals:
            channel_names = analog.channel_names
            if np.all([stream.upper() in chan for chan in channel_names]):
                self._set_analogsignal(analog)
                # load neuropixels locations
                channel_locations = np.loadtxt(extractors_dir / 'neuropixelsdatrecordingextractor' /
                                               'channel_positions_neuropixels.txt').T
                # get correct channel ID from channel name (e.g. AP32 --> 32)
                channel_ids = [int(chan_name[chan_name.find(stream.upper())+len(stream):]) - 1
                               for chan_name in channel_names]
                locations = channel_locations[channel_ids]
                self.set_channel_locations(locations)
                for i, ch in enumerate(self.get_channel_ids()):
                    self.set_channel_property(ch, "channel_name", channel_names[i])
                break

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'experiment_id': experiment_id,
                        'recording_id': recording_id, 'stream': stream}


class OpenEphysSortingExtractor(SortingExtractor):
    extractor_name = 'OpenEphysSorting'
    installed = HAVE_OE  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the OpenEphys extractor, install pyopenephys: \n\n pip install pyopenephys\n\n"  # error message when not installed

    def __init__(self, folder_path, experiment_id=0, recording_id=0):
        assert self.installed, self.installation_mesg
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
