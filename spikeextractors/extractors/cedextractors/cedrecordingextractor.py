from spikeextractors import RecordingExtractor
from .utils import get_channel_info, get_channel_data
from spikeextractors.extraction_tools import check_get_traces_args

import numpy as np
from pathlib import Path
from typing import Union
from copy import deepcopy

try:
    from sonpy import lib as sp

    HAVE_SONPY = True
except ImportError:
    HAVE_SONPY = False

PathType = Union[str, Path, None]
DtypeType = Union[str, np.dtype, None]


class CEDRecordingExtractor(RecordingExtractor):
    """
    Extracts electrophysiology recordings from .smrx files.
    The recording extractor always returns channel IDs starting from 0.
    The recording data will always be returned in the shape of (num_channels,num_frames).

    Parameters
    ----------
    file_path: str
        Path to the .smrx file to be extracted
    smrx_channel_ids: list of int
        List with indexes of valid smrx channels. Does not match necessarily
        with extractor id.
    """

    extractor_name = 'CEDRecording'
    installed = HAVE_SONPY  # check at class level if installed or not
    is_writable = False
    has_default_locations = False
    has_unscaled = False
    mode = 'file'
    installation_mesg = "To use the CED extractor, install sonpy: \n\n pip install sonpy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, smrx_channel_ids: list):
        assert self.installed, self.installation_mesg
        file_path = Path(file_path)
        assert file_path.is_file() and file_path.suffix == '.smrx', 'file_path must lead to a .smrx file!'
        assert len(smrx_channel_ids) > 0, "'smrx_channel_ids' cannot be an empty list!"

        super().__init__()

        # Open smrx file
        self._recording_file_path = file_path
        self._recording_file = sp.SonFile(sName=str(file_path), bReadOnly=True)
        if self._recording_file.GetOpenError() != 0:
            raise ValueError(f'Error opening file:', sp.GetErrorString(self._recording_file.GetOpenError()))

        # Map Recording channel_id to smrx index / test for invalid indexes /
        # get channel info / set channel gains
        self._channelid_to_smrxind = dict()
        self._channel_smrxinfo = dict()
        self._channel_names = []

        gains = []
        for i, ind in enumerate(smrx_channel_ids):
            if self._recording_file.ChannelType(ind) == sp.DataType.Off:
                raise ValueError(f'Channel {ind} is type Off and cannot be used')
            self._channelid_to_smrxind[i] = ind
            self._channel_smrxinfo[i] = get_channel_info(
                f=self._recording_file,
                smrx_ch_ind=ind
            )
            # Set channel gains: http://ced.co.uk/img/Spike10.pdf
            # from 16-bit encoded int / to ADC +-5V input / to measured Volts
            gain = self._channel_smrxinfo[i]['scale'] / 6553.6
            gain *= 1000  # mV --> uV
            gains.append(gain)
            self._channel_names.append(self._channel_smrxinfo[i]['title'])

        # Set gains
        self.set_channel_gains(gains=gains)
        self.has_unscaled = True

        rate0 = self._channel_smrxinfo[0]['rate']
        for chan, info in self._channel_smrxinfo.items():
            assert info['rate'] == rate0, "Inconsistency between 'sampling_frequency' of different channels. The " \
                                          "extractor only supports channels with the same 'rate'"

        # Set self._times
        times = (self._channel_smrxinfo[0]['frame_offset'] + np.arange(self.get_num_frames())) / self.get_sampling_frequency()
        self.set_times(times=times)

        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'smrx_channel_ids': smrx_channel_ids}

    @property
    def channel_names(self):
        return deepcopy(self._channel_names)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        """This function extracts and returns a trace from the recorded data from the
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
        start_frame: int
            The starting frame of the trace to be returned (inclusive)
        end_frame: int
            The ending frame of the trace to be returned (exclusive)
        channel_ids: array_like
            A list or 1D array of channel ids (ints) from which each trace will be
            extracted
        return_scaled: bool
            If True, traces are returned after scaling (using gain/offset). If False, the traces are returned as integers

        Returns
        ----------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (num_channels x num_frames)
        """

        recordings = np.vstack(
            [get_channel_data(
                f=self._recording_file,
                smrx_ch_ind=self._channelid_to_smrxind[i],
                start_frame=start_frame,
                end_frame=end_frame
            ) for i in channel_ids]
        )

        return recordings

    def get_num_frames(self):
        """This function returns the number of frames in the recording

        Returns
        -------
        num_frames: int
            Number of frames in the recording (duration of recording)
        """
        return 1 + int(self._channel_smrxinfo[0]['max_time'] / self._channel_smrxinfo[0]['divide'] - self._channel_smrxinfo[0]['frame_offset'])

    def get_sampling_frequency(self):
        """This function returns the sampling frequency in units of Hz.

        Returns
        -------
        fs: float
            Sampling frequency of the recordings in Hz
        """
        return self._channel_smrxinfo[0]['rate']

    def get_channel_ids(self):
        """Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.

        Returns
        -------
        channel_ids: list
            Channel list

        """
        return list(self._channelid_to_smrxind.keys())
    
    @staticmethod
    def get_all_channels_info(file_path):
        """
        Extract info from all channels in the smrx file. Returns a dictionary with
        valid smrx channel indexes as keys and the respective channel information as
        value.

        Parameters:
        -----------
        f: str
            Path to .smrx file
        """
        f = sp.SonFile(sName=str(file_path), bReadOnly=True)
        n_channels = f.MaxChannels()
        return {
            i: get_channel_info(f, i) for i in range(n_channels)
            if f.ChannelType(i) != sp.DataType.Off
        }
