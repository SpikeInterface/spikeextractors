from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from spikeextractors.extraction_tools import write_to_binary_dat_format, check_get_traces_args, \
    check_get_unit_spike_train

import json
import numpy as np
from pathlib import Path
from .mdaio import DiskReadMda, readmda, writemda64, MdaHeader
import shutil


class MdaRecordingExtractor(RecordingExtractor):
    extractor_name = 'MdaRecording'
    has_default_locations = True
    has_unscaled = False
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
        self._timeseries_path = str(timeseries0.absolute())
        geom0 = dataset_directory / geom_fname
        self._geom_fname = geom0
        self._geom = np.loadtxt(self._geom_fname, delimiter=',', ndmin=2)
        X = DiskReadMda(self._timeseries_path)
        if self._geom.shape[0] != X.N1():
            raise Exception(
                'Incompatible dimensions between geom.csv and timeseries file {} <> {}'.format(self._geom.shape[0],
                                                                                               X.N1()))
        self._num_channels = X.N1()
        self._num_timepoints = X.N2()
        RecordingExtractor.__init__(self)
        self.set_channel_locations(self._geom)
        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_timepoints

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        X = DiskReadMda(self._timeseries_path)
        recordings = X.readChunk(i1=0, i2=start_frame, N1=X.N1(), N2=end_frame - start_frame)
        recordings = recordings[channel_ids, :]
        return recordings

    def write_to_binary_dat_format(self, save_path, time_axis=0, dtype=None, chunk_size=None, chunk_mb=500,
                                   n_jobs=1, joblib_backend='loky', verbose=False):
        """Saves the traces of this recording extractor into binary .dat format.

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
            Size of each chunk in number of frames.
            If None (default) and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        n_jobs: int
            Number of jobs to use (Default 1)
        joblib_backend: str
            Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
        verbose: bool
            If True, output is verbose
        """
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
                                           chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                           joblib_backend=joblib_backend, verbose=verbose)
        else:
            write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype,
                                       chunk_size=chunk_size, chunk_mb=chunk_mb, n_jobs=n_jobs,
                                       joblib_backend=joblib_backend, verbose=verbose)

    @staticmethod
    def write_recording(recording, save_path, params=dict(), raw_fname='raw.mda', params_fname='params.json',
                        geom_fname='geom.csv', dtype=None, chunk_size=None, n_jobs=None, chunk_mb=500, verbose=False):
        """
        Writes recording to file in MDA format.

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
            Size of each chunk in number of frames.
            If None (default) and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
        n_jobs: int
            Number of jobs to use (Default 1)
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        verbose: bool
            If True, output is verbose
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_file_path = save_path / raw_fname
        parent_dir = save_path
        channel_ids = recording.get_channel_ids()
        num_chan = recording.get_num_channels()
        num_frames = recording.get_num_frames()

        geom = recording.get_channel_locations()

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
            write_to_binary_dat_format(recording, file_handle=f, dtype=dtype, n_jobs=n_jobs, chunk_size=chunk_size,
                                       chunk_mb=chunk_mb, verbose=verbose)

        params["samplerate"] = float(recording.get_sampling_frequency())
        with (parent_dir / params_fname).open('w') as f:
            json.dump(params, f)
        np.savetxt(str(parent_dir / geom_fname), geom, delimiter=',')


class MdaSortingExtractor(SortingExtractor):
    extractor_name = 'MdaSorting'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path, sampling_frequency=None):

        SortingExtractor.__init__(self)
        self._firings_path = file_path
        self._firings = readmda(self._firings_path)
        self._max_channels = self._firings[0, :]
        self._spike_times = self._firings[1, :]
        self._labels = self._firings[2, :]
        self._unit_ids = np.unique(self._labels).astype(int)
        self._sampling_frequency = sampling_frequency
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'sampling_frequency': sampling_frequency}

    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

        inds = np.where(
            (self._labels == unit_id) & (start_frame <= self._spike_times) & (self._spike_times < end_frame))
        return np.rint(self._spike_times[inds]).astype(int)

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
                    primary_channels_list.append([sorting.get_unit_property(unit_id, 'max_channel')] * times.shape[0])
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
    if not fname1.is_file():
        raise Exception('Dataset parameter file does not exist: ' + fname1)
    with open(fname1) as f:
        return json.load(f)
