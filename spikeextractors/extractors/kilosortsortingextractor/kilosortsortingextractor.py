from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path

class KiloSortSortingExtractor(SortingExtractor):
    def __init__(self, kilosort_folder):
        SortingExtractor.__init__(self)
        kilosort_folder = Path(kilosort_folder)
        spike_times = np.load(kilosort_folder / 'spike_times.npy')
        spike_templates = np.load(kilosort_folder /'spike_templates.npy')
        if (kilosort_folder / 'spike_clusters.npy').is_file():
            spike_clusters = np.load(kilosort_folder / 'spike_clusters.npy')
        else:
            spike_clusters = spike_templates

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
    def writeSorting(sorting, save_path):
        save_path = Path(save_path)
        spike_times = np.array([])
        spike_templates = np.array([])
        for id in sorting.getUnitIds():
            st = sorting.getUnitSpikeTrain(id)
            cl = [id] * len(sorting.getUnitSpikeTrain(id))
            spike_times = np.concatenate((spike_times, np.array(st)))
            spike_templates = np.concatenate((spike_templates, np.array(cl)))
        sorting_idxs = np.argsort(spike_times)
        spike_times = spike_times[sorting_idxs]
        spike_clusters = spike_templates[sorting_idxs]
        if not save_path.is_dir():
            save_path.mkdir()
        np.save(save_path / 'spike_times.npy', spike_times.astype(int))
        np.save(save_path /'spike_templates.npy', spike_clusters.astype(int))
