from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import read_binary, write_to_binary_dat_format
import os
import shutil
import numpy as np
from pathlib import Path


class BinDatRecordingExtractor(RecordingExtractor):

    extractor_name = 'BinDatRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = True
    is_dumpable = True
    mode = 'file'      
    extractor_gui_params = [
        {'name': 'file_path', 'type': 'file', 'title': "Path to file (.dat)"},
        {'name': 'sampling_frequency', 'type': 'float', 'title': "Sampling rate in HZ"},
        {'name': 'numchan', 'type': 'int', 'title': "Number of channels"},
        {'name': 'dtype', 'type': 'np.dtype', 'title': "The dtype of underlying data (int16, float32, etc.)"},
        {'name': 'recording_channels', 'type': 'int_list', 'value': None, 'default': None, 'title': "List of recording channels"},
        {'name': 'time_axis', 'type': 'int', 'value': 0, 'default': 0, 'title': "If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file. If 1, the traces shape (nb_channel, nb_sample) is kept in the file."},
        {'name': 'offset', 'type': 'int', 'value': 0, 'default': 0, 'title': "Offset in binary file"},
        {'name': 'gain', 'type': 'float', 'value': None, 'default': None, 'title': "gain of the recordings"},
    ]
    installation_mesg = ""  # error message when not installed

    def _init_extractor(self, file_path, sampling_frequency, numchan, dtype, recording_channels=None,
                 time_axis=0, geom=None, offset=0, gain=None):
        RecordingExtractor.__init__(self)
        self._datfile = Path(file_path)
        self._time_axis = time_axis
        self._dtype = str(dtype)
        self._timeseries = read_binary(self._datfile, numchan, dtype, time_axis, offset)
        self._sampling_frequency = float(sampling_frequency)
        self._gain = gain
        self._geom = geom

        if recording_channels is not None:
            assert len(recording_channels) == self._timeseries.shape[0], \
               'Provided recording channels have the wrong length'
            self._channels = recording_channels
        else:
            self._channels = list(range(self._timeseries.shape[0]))

        if geom is not None:
            for idx, channel in enumerate(self._channels):
                self.set_channel_property(channel, 'location', self._geom[idx, :])
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'sampling_frequency': sampling_frequency,
                        'numchan': numchan, 'dtype': str(dtype), 'recording_channels': recording_channels,
                        'time_axis': time_axis, 'geom': geom, 'offset': offset, 'gain': gain}


    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = list(range(self._timeseries.shape[0]))
        else:
            channel_ids = [self._channels.index(ch) for ch in channel_ids]
        recordings = self._timeseries[:, start_frame:end_frame][channel_ids, :]
        if self._dtype.startswith('uint'):
            exp_idx = self._dtype.find('int') + 3
            exp = int(self._dtype[exp_idx:])
            recordings = recordings.astype('float32') - 2**(exp - 1) - 1
        if self._gain is not None:
            recordings = recordings * self._gain
        return recordings

    def write_to_binary_dat_format(self, save_path, time_axis=0, dtype=None, chunk_size=None, chunk_mb=500):
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
        '''
        if dtype is None or dtype == self.get_dtype():
            try:
                shutil.copy(self._datfile, save_path)
            except Exception as e:
                print('Error occurred while copying:', e)
                print('Writing to binary')
                write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                           chunk_size=chunk_size, chunk_mb=chunk_mb)
        else:
            write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                       chunk_size=chunk_size, chunk_mb=chunk_mb)

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
