from spikeinterface import SortingExtractor
from spikeinterface.tools import read_python

import numpy as np
import os
from os.path import join
import neo

class KilosortSortingExtractor(SortingExtractor):
    def __init__(self, kilosort_folder):
        SortingExtractor.__init__(self)
        print('Parsing output files...')
        spike_times = np.load(join(kilosort_folder, 'spike_times.npy'))
        spike_clusters = np.load(join(kilosort_folder, 'spike_clusters.npy'))

        self._spiketrains = []
        clust_id = np.unique(spike_clusters)
        self._unit_ids = list(clust_id)
        spike_times.astype(int)

        self._spiketrais = []
        for clust in self._unit_ids:
            idx = np.where(spike_clusters == clust)[0]
            self._spiketrains.append(spike_times[idx])

    def getUnitIds(self):
        return list(self._unit_ids)

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.getUnitIds().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def writeSorting(self, sorting, save_path):
        pass