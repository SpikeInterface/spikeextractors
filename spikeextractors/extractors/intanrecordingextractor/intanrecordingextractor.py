import numpy as np
from pathlib import Path
from distutils.version import StrictVersion
from typing import Optional

from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args

try:
    import pyintan
    if StrictVersion(pyintan.__version__) >= '0.2.1':
        HAVE_INTAN = True
    else:
        print("pyintan version requires an update (>=0.2.1). Please upgrade with 'pip install --upgrade pyintan'")
        HAVE_INTAN = False
except ImportError:
    HAVE_INTAN = False


class IntanRecordingExtractor(RecordingExtractor):
    extractor_name = 'IntanRecording'
    has_default_locations = False
    is_writable = False
    mode = 'file'
    installed = HAVE_INTAN  # check at class level if installed or not
    installation_mesg = "To use the Intan extractor, install pyintan: \n\n pip install pyintan\n\n"  # error message when not installed

    def __init__(self, file_path: str, dtype: str = 'float', verbose: bool = False):
        assert HAVE_INTAN, self.installation_mesg
        RecordingExtractor.__init__(self)
        assert Path(file_path).suffix == '.rhs' or Path(file_path).suffix == '.rhd', \
            "Only '.rhd' and '.rhs' files are supported"
        self._recording_file = file_path
        self._recording = pyintan.File(file_path, verbose)
        self._num_frames = len(self._recording.times)
        self._analog_channels = np.array([ch for ch in self._recording._anas_chan if all([other_ch not in ch['name']
                                                                                          for other_ch in
                                                                                          ['ADC', 'VDD', 'AUX']])])
        self._num_channels = len(self._analog_channels)
        self._channel_ids = list(range(self._num_channels))
        self._fs = float(self._recording.sample_rate.rescale('Hz').magnitude)

        assert dtype in ['float', 'int16', 'uint16'], "'dtype' can be either 'float', 'int16', or 'uint16'"
        self._dtype = dtype

        if self._dtype in ['int16', 'uint16']:
            for i, ch in enumerate(self._analog_channels):
                self.set_channel_property(i, 'gain', ch['gain'])

        if self._dtype == 'uint16':
            for i, ch in enumerate(self._analog_channels):
                self.set_channel_property(i, 'offset', ch['offset'])

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'dtype': dtype, 'verbose': verbose}

    def get_channel_ids(self):
        return self._channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled: Optional[bool] = None,
                   dtype: Optional[str] = None):
        if return_scaled is not None and dtype is None:
            if return_scaled:
                dtype = 'float'
            else:
                raise ValueError("If attempting to return unscaled traces for the IntanRecordingExtractor, "
                                 "set dtype to 'int16' or 'uint16'!")
        if dtype is not None:
            assert dtype in ['float', 'int16', 'uint16'], "'dtype' must be either 'float', 'int16', or 'uint16'!"
            return_scaled = dtype == 'float'
        if return_scaled is None and dtype is None:
            dtype = self._dtype
        if return_scaled is False and dtype == 'float':  # User explicitly passed both conflicting arguments
            raise ValueError("Unable to return unscaled traces with dtype set to 'float'! "
                             "To return unscaled traces, set dtype to 'int16' or 'uint16'.")
        if return_scaled is True and dtype != 'float':  # User explicitly passed both conflicting arguments
            raise ValueError("Unable to return scaled traces with dtype set to 'int16' or 'uint16'!")

        channel_idxs = np.array([self._channel_ids.index(ch) for ch in channel_ids])
        analog_chans = self._analog_channels[channel_idxs]
        return self._recording._read_analog(channels=analog_chans, i_start=start_frame, i_stop=end_frame,
                                            dtype=dtype).T

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        channels = [np.unique(ev.channels)[0] for ev in self._recording.digital_in_events]
        assert channel_id in channels, f"Specified 'channel' not found. Available channels are {channels}"
        ev = self._recording.events[channels.index(channel_id)]

        ttl_frames = (ev.times.rescale("s") * self.get_sampling_frequency()).magnitude.astype(int)
        ttl_states = np.sign(ev.channel_states)
        ttl_valid_idxs = np.where((ttl_frames >= start_frame) & (ttl_frames < end_frame))[0]
        return ttl_frames[ttl_valid_idxs], ttl_states[ttl_valid_idxs]
