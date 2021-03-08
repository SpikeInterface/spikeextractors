import numpy as np
from pathlib import Path
from distutils.version import StrictVersion
from typing import Union, Optional

from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args

DtypeType = Union[str, np.dtype]
OptionalArrayType = Optional[Union[np.ndarray, list]]

try:
    import pyintan
    if StrictVersion(pyintan.__version__) >= '0.3.0':
        HAVE_INTAN = True
    else:
        print("pyintan version requires an update (>=0.3.0). Please upgrade with 'pip install --upgrade pyintan'")
        HAVE_INTAN = False
except ImportError:
    HAVE_INTAN = False


class IntanRecordingExtractor(RecordingExtractor):
    """
    Extracts raw neural recordings from the Intan file format.

    The recording extractor always returns channel IDs starting from 0.

    The recording data will always be returned in the shape of (num_channels, num_frames).

    Parameters
    ----------
    file_path : str
        Path to the .dat file to be extracted.
    dtype : dtype
        The data type used in the binary file.
    verbose : bool, optional
        Print output during pyintan file read.
    """

    extractor_name = 'IntanRecording'
    has_default_locations = False
    has_unscaled = True
    is_writable = False
    mode = "file"
    installed = HAVE_INTAN
    installation_mesg = "To use the Intan extractor, install pyintan: \n\n pip install pyintan\n\n"

    def __init__(self, file_path: str, verbose: bool = False):
        assert self.installed, self.installation_mesg
        RecordingExtractor.__init__(self)
        assert Path(file_path).suffix == '.rhs' or Path(file_path).suffix == '.rhd', \
            "Only '.rhd' and '.rhs' files are supported"
        self._recording_file = file_path
        self._recording = pyintan.File(file_path, verbose)
        self._num_frames = len(self._recording.times)
        self._analog_channels = np.array([
            ch for ch in self._recording._anas_chan
            if all([other_ch not in ch['name'] for other_ch in ['ADC', 'VDD', 'AUX']])
        ])
        self._num_channels = len(self._analog_channels)
        self._channel_ids = list(range(self._num_channels))
        self._fs = float(self._recording.sample_rate.rescale('Hz').magnitude)

        for i, ch in enumerate(self._analog_channels):
            self.set_channel_gains(channel_ids=i, gains=ch['gain'])
            self.set_channel_offsets(channel_ids=i, offsets=ch['offset'])

        self._kwargs = dict(file_path=str(Path(file_path).absolute()), verbose=verbose)

    def get_channel_ids(self):
        return self._channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    @check_get_traces_args
    def get_traces(
        self,
        channel_ids: OptionalArrayType = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        return_scaled: bool = True,
    ):
        """
        This function extracts and returns a trace from the recorded data from the
        given channels ids and the given start and end frame. It will return
        traces from within three ranges:

            [start_frame, start_frame+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_recording_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_recording_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Traces are returned in a 2D array that
        contains all of the traces from each channel with dimensions
        (num_channels x num_frames). In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        start_frame : int, optional
            The starting frame of the trace to be returned (inclusive)
        end_frame : int, optional
            The ending frame of the trace to be returned (exclusive)
        channel_ids : ArrayType, optional
            A list or 1D array of channel ids (ints) from which each trace will be
            extracted
        return_scaled : bool, optional
            If True, traces are returned after scaling (using gain/offset). If False, the raw traces are returned.
            Defaults to True.

        Returns
        ----------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (num_channels x num_frames)
        """
        channel_idxs = np.array([self._channel_ids.index(ch) for ch in channel_ids])
        return self._recording._read_analog(
            channels=self._analog_channels[channel_idxs],
            i_start=start_frame,
            i_stop=end_frame,
            dtype="uint16"
        ).T

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        channels = [np.unique(ev.channels)[0] for ev in self._recording.digital_in_events]
        assert channel_id in channels, f"Specified 'channel' not found. Available channels are {channels}"
        ev = self._recording.events[channels.index(channel_id)]

        ttl_frames = (ev.times.rescale("s") * self.get_sampling_frequency()).magnitude.astype(int)
        ttl_states = np.sign(ev.channel_states)
        ttl_valid_idxs = np.where((ttl_frames >= start_frame) & (ttl_frames < end_frame))[0]
        return ttl_frames[ttl_valid_idxs], ttl_states[ttl_valid_idxs]
