from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import json
import numpy as np
from pathlib import Path
from .mdaio import DiskReadMda, readmda, writemda32, writemda64
import os


class MdaRecordingExtractor(RecordingExtractor):
    extractor_name = 'MdaRecordingExtractor'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'dataset_directory', 'type': 'path', 'title': "Path to folder"},
        {'name': 'probe_path', 'type': 'path', 'value':None, 'default':None, 'title': "Path to probe file (.csv or .prb)"}
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, dataset_directory):
        RecordingExtractor.__init__(self)
        dataset_directory = Path(dataset_directory)
        self._dataset_directory = dataset_directory
        timeseries0 = dataset_directory / 'raw.mda'
        self._dataset_params = read_dataset_params(str(dataset_directory))
        self._samplerate = self._dataset_params['samplerate'] * 1.0
        self._timeseries_path = os.path.abspath(timeseries0)
        geom0 = os.path.join(dataset_directory, 'geom.csv')
        self._geom_fname = geom0
        self._geom = np.genfromtxt(self._geom_fname, delimiter=',')
        X = DiskReadMda(self._timeseries_path)
        if self._geom.shape[0] != X.N1():
            raise Exception(
                'Incompatible dimensions between geom.csv and timeseries file {} <> {}'.format(self._geom.shape[0],
                                                                                               X.N1()))
        self._num_channels = X.N1()
        self._num_timepoints = X.N2()
        for m in range(self._num_channels):
            self.set_channel_property(m, 'location', self._geom[m, :])

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_timepoints

    def get_sampling_frequency(self):
        return self._samplerate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        X = DiskReadMda(self._timeseries_path)
        recordings = X.readChunk(i1=0, i2=start_frame, N1=X.N1(), N2=end_frame - start_frame)
        recordings = recordings[channel_ids, :]
        return recordings

    @staticmethod
    def write_recording(recording, save_path, params=dict()):
        save_path = Path(save_path)
        if not save_path.exists():
            if not save_path.is_dir():
                os.makedirs(str(save_path))
        save_file_path = str(save_path / 'raw.mda')
        parent_dir = save_path

        channel_ids = recording.get_channel_ids()
        M = len(channel_ids)
        raw = recording.get_traces()
        location0 = recording.get_channel_property(channel_ids[0], 'location')
        nd = len(location0)
        geom = np.zeros((M, nd))
        for ii in range(len(channel_ids)):
            location_ii = recording.get_channel_property(channel_ids[ii], 'location')
            geom[ii, :] = list(location_ii)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        writemda32(raw, save_file_path)
        params["samplerate"] = recording.get_sampling_frequency()
        with (parent_dir / 'params.json').open('w') as f:
            json.dump(params, f)
        np.savetxt(str(parent_dir / 'geom.csv'), geom, delimiter=',')


class MdaSortingExtractor(SortingExtractor):
    extractor_name = 'MdaSortingExtractor'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'firings_file', 'type': 'file_path', 'title': "str, Path to file"},
        {'name': 'sampling_frequency', 'type': 'float', 'title': "sampling frequency"}
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, firings_file, sampling_frequency=None):

        SortingExtractor.__init__(self)
        self._firings_path = firings_file
        self._firings = readmda(self._firings_path)
        self._times = self._firings[1, :]
        self._labels = self._firings[2, :]
        self._unit_ids = np.unique(self._labels).astype(int)
        self._sampling_frequency = sampling_frequency

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
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
        for unit in unit_ids:
            times = sorting.get_unit_spike_train(unit_id=unit)
            times_list.append(times)
            labels_list.append(np.ones(times.shape) * unit)
            if write_primary_channels:
                if 'max_channel' in sorting.get_unit_spike_feature_names(unit_id=unit):
                    primary_channels_list.append(sorting.get_unit_spike_features(unit_id=unit,
                                                                                 feature_name='max_channel'))
                else:
                    raise ValueError(
                        "Unable to write primary channels because 'max_channel' spike feature not set in unit " + str(
                            unit))
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


def read_dataset_params(dsdir):
    fname1 = os.path.join(dsdir, 'params.json')
    if not os.path.exists(fname1):
        raise Exception('Dataset parameter file does not exist: ' + fname1)
    with open(fname1) as f:
        return json.load(f)
