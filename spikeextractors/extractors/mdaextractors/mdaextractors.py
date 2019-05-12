from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import json
import numpy as np
from .mdaio import DiskReadMda, readmda, writemda32, writemda64
import os

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
        self._dataset_directory = dataset_directory
        timeseries0 = os.path.join(dataset_directory, 'raw.mda')
        self._dataset_params = read_dataset_params(dataset_directory)
        self._samplerate = self._dataset_params['samplerate'] * 1.0
        if is_kbucket_url(timeseries0):
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
        channel_ids = recording.get_channel_ids()
        M = len(channel_ids)
        N = recording.get_num_frames()
        raw = recording.get_traces()
        location0 = recording.get_channel_property(channel_ids[0], 'location')
        nd = len(location0)
        geom = np.zeros((M, nd))
        for ii in range(len(channel_ids)):
            location_ii = recording.get_channel_property(channel_ids[ii], 'location')
            geom[ii, :] = list(location_ii)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        writemda32(raw, os.path.join(save_path, 'raw.mda'))
        params["samplerate"] = recording.get_sampling_frequency()
        with open(os.path.join(save_path, 'params.json'),'w') as f:
            json.dump(params, f)
        np.savetxt(os.path.join(save_path, 'geom.csv'), geom, delimiter=',')


class MdaSortingExtractor(SortingExtractor):

    extractor_name = 'MdaSortingExtractor'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'firings_file', 'type': 'file_path', 'title': "str, Path to file"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, firings_file):

        SortingExtractor.__init__(self)
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
        self._firings = readmda(self._firings_path)
        self._times = self._firings[1, :]
        self._labels = self._firings[2, :]
        self._unit_ids = np.unique(self._labels).astype(int)

    def get_unit_ids(self):
        return self._unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        inds = np.where((self._labels == unit_id) & (start_frame <= self._times) & (self._times < end_frame))
        return np.rint(self._times[inds]).astype(int)

    @staticmethod
    def write_sorting(sorting, save_path):
        unit_ids = sorting.get_unit_ids()
        if len(unit_ids) > 0:
            K = np.max(unit_ids)
        else:
            K = 0
        times_list = []
        labels_list = []
        for i in range(len(unit_ids)):
            unit = unit_ids[i]
            times = sorting.get_unit_spike_train(unit_id=unit)
            times_list.append(times)
            labels_list.append(np.ones(times.shape) * unit)
        all_times = _concatenate(times_list)
        all_labels = _concatenate(labels_list)
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        L = len(all_times)
        firings = np.zeros((3, L))
        firings[1, :] = all_times
        firings[2, :] = all_labels
        writemda64(firings, save_path)


def _concatenate(list):
    if len(list) == 0:
        return np.array([])
    return np.concatenate(list)


def is_kbucket_url(path):
    return path.startswith('kbucket://') or path.startswith('sha1://')


def is_url(path):
    return path.startswith('http://') or path.startswith('https://') or path.startswith(
        'kbucket://') or path.startswith('sha1://')

def _realize_file(*,path):
    if is_kbucket_url(path):
        try:
            from cairio import client as cairio
        except:
            raise Exception('To use kbucket files, you need to install the cairio client')
        return _realize_file(path=path)
    else:
        return path

def _find_file(*,path):
    if is_kbucket_url(path):
        try:
            from cairio import client as cairio
        except:
            raise Exception('To use kbucket files, you need to install the cairio client')
        return _find_file(path=path)
    else:
        return path


def read_dataset_params(dsdir):
    fname1=os.path.join(dsdir, 'params.json')
    fname2=_realize_file(path=fname1)
    if not fname2:
        raise Exception('Unable to find file: '+fname1)
    if not os.path.exists(fname2):
        raise Exception('Dataset parameter file does not exist: ' + fname2)
    with open(fname2) as f:
        return json.load(f)
