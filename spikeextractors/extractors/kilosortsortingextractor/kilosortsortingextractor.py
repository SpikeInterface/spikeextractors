from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path

class KiloSortSortingExtractor(SortingExtractor):

    extractor_name = 'KiloSortSortingExtractor'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'kilosort_folder', 'type': 'path', 'title': "str, Path to folder"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, kilosort_folder):
        SortingExtractor.__init__(self)
        kilosort_folder = Path(kilosort_folder)
        spike_times = np.load(str(kilosort_folder / 'spike_times.npy'))
        spike_templates = np.load(str(kilosort_folder /'spike_templates.npy'))
        if (kilosort_folder / 'spike_clusters.npy').is_file():
            spike_clusters = np.load(str(kilosort_folder / 'spike_clusters.npy'))
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

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def write_sorting(sorting, save_path):
        save_path = Path(save_path)
        spike_times = np.array([])
        spike_templates = np.array([])
        for id in sorting.get_unit_ids():
            st = sorting.get_unit_spike_train(id)
            cl = [id] * len(sorting.get_unit_spike_train(id))
            spike_times = np.concatenate((spike_times, np.array(st)))
            spike_templates = np.concatenate((spike_templates, np.array(cl)))
        sorting_idxs = np.argsort(spike_times)
        spike_times = spike_times[sorting_idxs]
        spike_clusters = spike_templates[sorting_idxs]
        if not save_path.is_dir():
            save_path.mkdir()
        np.save(str(save_path / 'spike_times.npy'), spike_times.astype(int))
        np.save(str(save_path /'spike_templates.npy'), spike_clusters.astype(int))
