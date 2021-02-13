from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import read_binary, write_to_binary_dat_format, check_get_traces_args
import shutil
import numpy as np
from pathlib import Path
from typing import Union, Optional

PathType = Union[str, Path]
DtypeType = Union[str, np.dtype]
ArrayType = Union[list, np.ndarray]

class BinDatRecordingExtractor(RecordingExtractor):
    """
    RecordingExtractor for a binary format

    Parameters
    ----------
    file_path: str or Path
        Path to the binary file
    sampling_frequency: float
        The sampling frequncy
    numchan: int
        Number of channels
    dtype: str or dtype
        The dtype of the binary file
    time_axis: int
        The axis of the time dimension (default 0: F order)
    recording_channels: list (optional)
        A list of channel ids
    offset: float (optional)
        The offset to apply to the traces
    geom: array-like (optional)
        A list or array with channel locations
    gain: float (optional)
        The gain to apply to the traces
    gain_first: bool
        If True and gain and offset are provided, traces are scaled as: original_traces * gain - offset
        If False and gain and offset are provided, traces are scaled as: (original_traces - offset) * gain
    is_filtered: bool
        If True, the recording is assumed to be filtered
    """
    extractor_name = 'BinDatRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path: PathType, sampling_frequency: float, numchan: int, dtype: DtypeType,
                 time_axis: int = 0, recording_channels: Optional[list] = None,  geom: Optional[ArrayType] = None,
                 offset: Optional[float] = 0, gain: Optional[float] = None, gain_first: bool = True,
                 is_filtered: Optional[bool] = None):
        RecordingExtractor.__init__(self)
        self._datfile = Path(file_path)
        self._time_axis = time_axis
        self._dtype = str(dtype)
        self._sampling_frequency = float(sampling_frequency)
        self._gain = gain
        self._numchan = numchan
        self._geom = geom
        self._offset = offset
        self._gain_first = gain_first
        self._timeseries = read_binary(self._datfile, numchan, dtype, time_axis, offset)

        # keep track of filter status when dumping
        if is_filtered is not None:
            self.is_filtered = is_filtered
        else:
            self.is_filtered = False

        if recording_channels is not None:
            assert len(recording_channels) <= self._timeseries.shape[0], \
               'Provided recording channels have the wrong length'
            self._channels = recording_channels
        else:
            self._channels = list(range(self._timeseries.shape[0]))

        if len(self._channels) == self._timeseries.shape[0]:
            self._complete_channels = True
        else:
            assert max(self._channels) < self._timeseries.shape[0], "Channel ids exceed the number of " \
                                                                    "available channels"
            self._complete_channels = False

        if geom is not None:
            self.set_channel_locations(self._geom)
        if 'numpy' in str(dtype):
            dtype_str = str(dtype).replace("<class '", "").replace("'>", "")
            # drop 'numpy
            dtype_str = dtype_str.split('.')[1]
        else:
            dtype_str = str(dtype)
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'sampling_frequency': sampling_frequency,
                        'numchan': numchan, 'dtype': dtype_str, 'recording_channels': recording_channels,
                        'time_axis': time_axis, 'geom': geom, 'offset': offset, 'gain': gain,
                        'is_filtered': is_filtered}

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if self._complete_channels:
            if np.all(channel_ids == self.get_channel_ids()):
                recordings = self._timeseries[:, start_frame:end_frame]
            else:
                channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
                if np.all(np.diff(channel_idxs) == 1):
                    recordings = self._timeseries[channel_idxs[0]:channel_idxs[0]+len(channel_idxs),
                                 start_frame:end_frame]
                else:
                    # This block of the execution will return the data as an array, not a memmap
                    recordings = self._timeseries[channel_idxs, start_frame:end_frame]
        else:
            # in this case channel ids are actually indexes
            recordings = self._timeseries[channel_ids, start_frame:end_frame]

        if self._gain is not None:
            # unint needs to be converted to float in this case!
            if self._dtype.startswith('uint'):
                exp_idx = self._dtype.find('int') + 3
                exp = int(self._dtype[exp_idx:])
                recordings = recordings.astype('float32') - 2 ** (exp - 1)
            if self._offset is None:
                offset = 0
            else:
                offset = self._offset
            if self._gain_first:
                recordings = recordings * self._gain - offset
            else:
                recordings = (recordings - offset) * self._gain

        return recordings

    def write_to_binary_dat_format(self, save_path, time_axis=0, dtype=None, chunk_size=None, chunk_mb=500,
                                   n_jobs=1, joblib_backend='loky', verbose=False):
        '''Saves the traces of this recording extractor into binary .dat format.

        Parameters
        ----------
        save_path: str
            The path to the file.
        time_axis: 0 (default) or 1
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype: dtype
            Type of the saved data. Default float32
        chunk_size: None or int
            If not None then the file is saved in chunks.
            This avoid to much memory consumption for big files.
            If 'auto' the file is saved in chunks of ~ 500Mb
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        n_jobs: int
            Number of jobs to use (Default 1)
        joblib_backend: str
            Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
        '''
        if dtype is None or dtype == self.get_dtype():
            try:
                shutil.copy(self._datfile, save_path)
            except Exception as e:
                print('Error occurred while copying:', e)
                print('Writing to binary')
                write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                           chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                           joblib_backend=joblib_backend)
        else:
            write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                       chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                       joblib_backend=joblib_backend)


    @staticmethod
    def write_recording(recording, save_path, time_axis=0, dtype=None, chunk_size=None):
        '''Saves the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor object to be saved in .dat format
        save_path: str
            The path to the file.
        time_axis: 0 (default) or 1
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype: dtype
            Type of the saved data. Default float32.
        chunk_size: None or int
            If not None then the copy done by chunk size.
            This avoid to much memory consumption for big files.
        '''
        write_to_binary_dat_format(recording, save_path, time_axis=time_axis, dtype=dtype, chunk_size=chunk_size)
