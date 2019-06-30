import json
import os
from pathlib import Path

import numpy as np
from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

from .mdaio import DiskReadMda, readmda, writemda32, writemda64


class MdaRecordingExtractor(RecordingExtractor):
    extractor_name = 'MdaRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'dataset_directory', 'type': 'path', 'title': "Path to folder"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, dataset_directory, *, download=True):
        RecordingExtractor.__init__(self)
        dataset_directory = Path(dataset_directory)
        self._dataset_directory = dataset_directory
        timeseries0 = dataset_directory / 'raw.mda'
        self._dataset_params = read_dataset_params(str(dataset_directory))
        self._samplerate = self._dataset_params['samplerate'] * 1.0
        if is_kbucket_url(str(timeseries0)):
            download_needed = is_url(_find_file(path=timeseries0))
            if download and download_needed:
                print('Downloading file: ' + timeseries0)
                self._timeseries_path = _realize_file(path=timeseries0)
                print('Done.')
            else:
                self._timeseries_path = _find_file(path=timeseries0)
        else:
            self._timeseries_path = os.path.abspath(timeseries0)
        geom0 = os.path.join(dataset_directory, 'geom.csv')
        self._geom_fname = _realize_file(path=geom0)
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
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, firings_file):
        SortingExtractor.__init__(self)
        firing_dir = str(Path(firings_file).parent)
        if is_kbucket_url(firings_file):
            try:
                from cairio import client as cairio
            except:
                raise Exception('To use kbucket files, you need to install the cairio client')
            download_needed = is_url(_find_file(path=firings_file))
        else:
            download_needed = is_url(firings_file)
        if download_needed:
            print('Downloading file: ' + firings_file)
            self._firings_path = _realize_file(path=firings_file)
            print('Done.')
        else:
            self._firings_path = _realize_file(path=firings_file)
        self._dataset_params = read_dataset_params(firing_dir, raise_exception=False)
        if self._dataset_params is not None and 'samplerate' in self._dataset_params.keys():
            self._sampling_frequency = self._dataset_params['samplerate'] * 1.0
        self._firings = readmda(self._firings_path)
        self._times = self._firings[1, :]
        self._labels = self._firings[2, :]
        self._unit_ids = np.unique(self._labels).astype(int)

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
    def write_sorting(sorting, save_path, write_primary_channels=False, params=dict()):
        save_path = Path(save_path)
        if not save_path.exists():
            if save_path.suffix == '.mda':
                if not save_path.parent.is_dir():
                    os.makedirs(str(save_path.parent))
            else:
                os.makedirs(str(save_path))
        if save_path.is_dir():
            save_file_path = str(save_path / 'firings.mda')
            parent_dir = save_path
        elif save_path.suffix == '.mda':
            save_file_path = str(save_path)
            parent_dir = save_path.parent
        else:
            raise Exception("'save_path' can be either a directory or a .mda file")

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
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        L = len(all_times)
        firings = np.zeros((3, L))
        firings[1, :] = all_times
        firings[2, :] = all_labels
        writemda64(firings, save_file_path)
        params["samplerate"] = sorting.get_sampling_frequency()
        if params["samplerate"] is not None:
            if not (parent_dir / 'params.json').is_file():
                with (parent_dir / 'params.json').open('w') as f:
                    json.dump(params, f)


def _concatenate(list):
    if len(list) == 0:
        return np.array([])
    return np.concatenate(list)


def is_kbucket_url(path):
    return path.startswith('kbucket://') or path.startswith('sha1://')


def is_url(path):
    return path.startswith('http://') or path.startswith('https://') or path.startswith(
        'kbucket://') or path.startswith('sha1://')


def _realize_file(*, path):
    if is_kbucket_url(path):
        try:
            from cairio import client as cairio
        except:
            raise Exception('To use kbucket files, you need to install the cairio client')
        return _realize_file(path=path)
    else:
        return path


def _find_file(*, path):
    if is_kbucket_url(path):
        try:
            from cairio import client as cairio
        except:
            raise Exception('To use kbucket files, you need to install the cairio client')
        return _find_file(path=path)
    else:
        return path


def read_dataset_params(dsdir, raise_exception=True):
    fname1 = os.path.join(dsdir, 'params.json')
    fname2 = _realize_file(path=fname1)
    if not fname2:
        if raise_exception:
            raise Exception('Unable to find file: ' + fname1)
        else:
            pass
    if not os.path.exists(fname2):
        if raise_exception:
            raise Exception('Dataset parameter file does not exist: ' + fname2)
        else:
            return None
    with open(fname2) as f:
        return json.load(f)
