from spikeinterface import SortingExtractor

import numpy as np
import os
from os.path import join
import neo

class SpykingCircusSortingExtractor(SortingExtractor):
    def __init__(self, kilosort_folder):
        SortingExtractor.__init__(self)
        print('Parsing output files...')
        spike_times = np.load(join(kilosort_folder, 'spike_times.npy'))
        spike_clusters = np.load(join(kilosort_folder, 'spike_clusters.npy'))
        spike_templates = np.load(join(kilosort_folder, 'templates.npy')).swapaxes(1, 2)
        spike_templates_id = np.load(join(kilosort_folder, 'spike_templates.npy'))

        self._spiketrains = []
        clust_id, n_counts = np.unique(spike_clusters, return_counts=True)
        self._unit_ids = list(clust_id)
        spike_times.astype(int)

        counts = 0
        for clust, count in zip(clust_id, n_counts):
            idx = np.where(spike_clusters == clust)[0]
            # spike_templates.append(kl_templates[clust])
            # counts += len(idx)
            # spike_times = kl_times[idx]
            # spiketrain = neo.SpikeTrain(spike_times, t_start=t_start, t_stop=t_stop)
            # spike_trains.append(spiketrain)

        spike_templates = np.array(spike_templates)
        # set unit properties
        for i_s, spiketrain in enumerate(self._spiketrains):
            for key, val in spiketrain.annotations.items():
                self.setUnitProperty(self.getUnitIds()[i_s], key, val)

    def getUnitIds(self):
        return list(self._unit_ids)

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.getUnitIds().index(unit_id)].rescale('s').magnitude * self._fs
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]