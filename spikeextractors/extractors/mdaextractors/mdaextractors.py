from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from spikeextractors.extraction_tools import write_to_binary_dat_format, check_get_traces_args

import json
import numpy as np
from pathlib import Path
from .mdaio import DiskReadMda, readmda, writemda64, MdaHeader
import os
import shutil


class MdaRecordingExtractor(RecordingExtractor):
    extractor_name = 'MdaRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path, raw_fname='raw.mda', params_fname='params.json', geom_fname='geom.csv'):
        dataset_directory = Path(folder_path)
        self._dataset_directory = dataset_directory
        timeseries0 = dataset_directory / raw_fname
        self._dataset_params = read_dataset_params(dataset_directory, params_fname)
        self._sampling_frequency = self._dataset_params['samplerate'] * 1.0
        self._timeseries_path = os.path.abspath(timeseries0)
        geom0 = dataset_directory / geom_fname
        self._geom_fname = geom0
        self._geom = np.genfromtxt(self._geom_fname, delimiter=',')
        X = DiskReadMda(self._timeseries_path)
        if self._geom.shape[0] != X.N1():
            raise Exception(
                'Incompatible dimensions between geom.csv and timeseries file {} <> {}'.format(self._geom.shape[0],
                                                                                               X.N1()))
        self._num_channels = X.N1()
        self._num_timepoints = X.N2()
        RecordingExtractor.__init__(self)
        for m in range(self._num_channels):
            self.set_channel_property(m, 'location', self._geom[m, :])
        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_timepoints

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        X = DiskReadMda(self._timeseries_path)
        recordings = X.readChunk(i1=0, i2=start_frame, N1=X.N1(), N2=end_frame - start_frame)
        recordings = recordings[channel_ids, :]
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
        X = DiskReadMda(self._timeseries_path)
        header_size = X._header.header_size
        if dtype is None or dtype == self.get_dtype():
            try:
                with open(self._timeseries_path, 'rb') as src, open(save_path, 'wb') as dst:
                    src.seek(header_size)
                    shutil.copyfileobj(src, dst)
            except Exception as e:
                print('Error occurred while copying:', e)
                print('Writing to binary')
                write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                           chunk_size=chunk_size, chunk_mb=chunk_mb)
        else:
            write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                       chunk_size=chunk_size, chunk_mb=chunk_mb)

    @staticmethod
    def write_recording(recording, save_path, params=dict(), raw_fname='raw.mda', params_fname='params.json',
                        geom_fname='geom.csv', dtype=None, chunk_size=None, chunk_mb=500):
        '''

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be saved
        save_path: str or Path
            The folder in which the Mda files are saved
        params: dictionary
            Dictionary with optional parameters to save metadata. Sampling frequency is appended to this dictionary.
        raw_fname: str
            File name of raw file (default raw.mda)
        params_fname: str
            File name of params file (default params.json)
        geom_fname: str
            File name of geom file (default geom.csv)
        dtype: dtype
            dtype to be used. If None dtype is same as recording traces.
        chunk_size: None or int
            Number of chunks to save the file in. This avoid to much memory consumption for big files.
            If None and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        '''
        save_path = Path(save_path)
        if not save_path.exists():
            if not save_path.is_dir():
                os.makedirs(str(save_path))
        save_file_path = save_path / raw_fname
        parent_dir = save_path
        channel_ids = recording.get_channel_ids()
        num_chan = recording.get_num_channels()
        num_frames = recording.get_num_frames()

        location0 = recording.get_channel_property(channel_ids[0], 'location')
        nd = len(location0)
        geom = np.zeros((num_chan, nd))
        for ii in range(len(channel_ids)):
            location_ii = recording.get_channel_property(channel_ids[ii], 'location')
            geom[ii, :] = list(location_ii)

        if not save_path.is_dir():
            os.mkdir(save_path)

        if dtype is None:
            dtype = recording.get_dtype()

        if dtype == 'float':
            dtype = 'float32'
        if dtype == 'int':
            dtype = 'int16'

        with save_file_path.open('wb') as f:
            header = MdaHeader(dt0=dtype, dims0=(num_chan, num_frames))
            header.write(f)
            # takes care of the chunking
            write_to_binary_dat_format(recording, file_handle=f, dtype=dtype, chunk_size=chunk_size,
                                       chunk_mb=chunk_mb)

        params["samplerate"] = recording.get_sampling_frequency()
        with (parent_dir / params_fname).open('w') as f:
            json.dump(params, f)
        np.savetxt(str(parent_dir / geom_fname), geom, delimiter=',')


class MdaSortingExtractor(SortingExtractor):
    extractor_name = 'MdaSortingExtractor'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path, sampling_frequency=None):

        SortingExtractor.__init__(self)
        self._firings_path = file_path
        self._firings = readmda(self._firings_path)
        self._max_channels = self._firings[0, :]
        self._times = self._firings[1, :]
        self._labels = self._firings[2, :]
        self._unit_ids = np.unique(self._labels).astype(int)
        self._sampling_frequency = sampling_frequency
        for unit_id in self._unit_ids:
            inds = np.where(self._labels == unit_id)
            max_channels = self._max_channels[inds].astype(int)
            self.set_unit_property(unit_id, 'mda_max_channel', max_channels[0])
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'sampling_frequency': sampling_frequency}

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        inds = np.where((self._labels == unit_id) & (start_frame <= self._times) & (self._times < end_frame))
        return np.rint(self._times[inds]).astype(int)

    @staticmethod
    def write_sorting(sorting, save_path, write_primary_channels=False):
        unit_ids = sorting.get_unit_ids()
        times_list = []
        labels_list = []
        primary_channels_list = []
        for unit_id in unit_ids:
            times = sorting.get_unit_spike_train(unit_id=unit_id)
            times_list.append(times)
            labels_list.append(np.ones(times.shape) * unit_id)
            if write_primary_channels:
                if 'max_channel' in sorting.get_unit_property_names(unit_id):
                    primary_channels_list.append([sorting.get_unit_property(unit_id, 'max_channel')]*times.shape[0])
                else:
                    raise ValueError(
                        "Unable to write primary channels because 'max_channel' spike feature not set in unit " + str(
                            unit_id))
            else:
                primary_channels_list.append(np.zeros(times.shape))
        all_times = _concatenate(times_list)
        all_labels = _concatenate(labels_list)
        all_primary_channels = _concatenate(primary_channels_list)
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        all_primary_channels = all_primary_channels[sort_inds]
        L = len(all_times)
        firings = np.zeros((3, L))
        firings[0, :] = all_primary_channels
        firings[1, :] = all_times
        firings[2, :] = all_labels

        writemda64(firings, save_path)


def _concatenate(list):
    if len(list) == 0:
        return np.array([])
    return np.concatenate(list)


def read_dataset_params(dsdir, params_fname):
    fname1 = dsdir / params_fname
    if not os.path.exists(fname1):
        raise Exception('Dataset parameter file does not exist: ' + fname1)
    with open(fname1) as f:
        return json.load(f)
