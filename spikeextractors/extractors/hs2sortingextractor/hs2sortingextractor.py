from spikeextractors import SortingExtractor

import numpy as np
import h5py


class HS2SortingExtractor(SortingExtractor):
    def __init__(self, recording_file):
        SortingExtractor.__init__(self)
        self._recording_file = recording_file
        self._rf = h5py.File(self._recording_file, mode='r')
        self._unit_ids = set(self._rf['cluster_id'].value)
        if 'centres' in self._rf.keys():
            self._unit_locs = self._rf['centres'].value  # cache for faster access

    def get_unit_indices(self, x):
        return np.where(self._rf['cluster_id'].value == x)[0]

    def getUnitIds(self):
        return list(self._unit_ids)

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._rf['times'].value[self.get_unit_indices(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    def getUnitChannels(self, unit_id):
        return self._rf['ch'].value[self.get_unit_indices(unit_id)]

    def getUnitProperty(self, unit_id, property):
        if property == 'spike_locations':
            return self._rf['data'][:2, self.get_unit_indices(unit_id)].T
        elif property == 'unit_location':
            return self._unit_locs[unit_id]
        else:
            raise Exception("Property does not exist.")

    @staticmethod
    def writeSorting(sorting, save_path):
        unit_ids = sorting.getUnitIds()
        times_list = []
        labels_list = []
        for i in range(len(unit_ids)):
            unit = unit_ids[i]
            times = sorting.getUnitSpikeTrain(unit_id=unit)
            times_list.append(times)
            labels_list.append(np.ones(times.shape, dtype=int) * unit)
        all_times = np.concatenate(times_list)
        all_labels = np.concatenate(labels_list)
        rf = h5py.File(save_path, mode='w')
        # for now only create the entries required by any RecordingExtractor
        rf.create_dataset("times", data=all_times)
        rf.create_dataset("cluster_id", data=all_labels)
        rf.close()
