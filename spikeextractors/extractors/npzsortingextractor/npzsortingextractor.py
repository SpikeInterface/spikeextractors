from spikeextractors import SortingExtractor
from pathlib import Path

import numpy as np


class NpzSortingExtractor(SortingExtractor):
    """
    Dead simple format super light base on the NPZ numpy format.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html#numpy.savez

    It is in fact an arichive of several .npy format.
    All spike are store in two columns maner index+labels


    """
    extractor_name = 'NpzSortingExtractor'
    installed = True # depend only on numpy
    _gui_params = []
    installation_mesg = "Always installed"

    def __init__(self, npz_filename):
        self.npz_filename = npz_filename

        npz = np.load(npz_filename)

        self.unit_ids = npz['unit_ids']
        self.spike_indexes = npz['spike_indexes']
        self.spike_labels = npz['spike_labels']

        if 'sampling_frequency' in npz:
            self._sampling_frequency = float(npz['sampling_frequency'][0])
        else:
            self._sampling_frequency = None

    def get_unit_ids(self):
        return list(self.unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        spike_times = self.spike_indexes[self.spike_labels == unit_id]
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times

    @staticmethod
    def write_sorting(sorting, save_file):
        d = {}
        units_ids = np.array(sorting.get_unit_ids())
        d['unit_ids'] = units_ids
        spike_indexes = []
        spike_labels = []
        for unit_id in units_ids:
            sp_ind = sorting.get_unit_spike_train(unit_id)
            spike_indexes.append(sp_ind)
            spike_labels.append(np.ones(sp_ind.size, dtype='int64')*unit_id)

        # order times
        if len(spike_indexes) > 0:
            spike_indexes = np.concatenate(spike_indexes)
            spike_labels = np.concatenate(spike_labels)
            order = np.argsort(spike_indexes)
            spike_indexes = spike_indexes[order]
            spike_labels = spike_labels[order]
        else:
            spike_indexes = np.array([], dtype='int64')
            spike_labels = np.array([], dtype='int64')

        d['spike_indexes'] = spike_indexes
        d['spike_labels'] = spike_labels

        if sorting.get_sampling_frequency() is not None:
            d['sampling_frequency'] = np.array([sorting.get_sampling_frequency()], dtype='float64')

        np.savez(save_file, **d)
