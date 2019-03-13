from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path

# written for yass v0.5
# James Jun, 2017 Feb 8
# modified from spykingcircussortingextractor.py
class YassSortingExtractor(SortingExtractor):
    def __init__(self, yass_folder):
        SortingExtractor.__init__(self)
        yass_folder = Path(yass_folder)
        files = yass_folder.iterdir()
        results = None
        for f in files:
            if 'spike_train.npy' in str(f):
                results = f
                break
        if results is None:
            raise Exception(yass_folder, " is not a yass folder")
        np_results = np.load(results)
        spike_times = np_results[:,0].astype(int)

        self._spiketrains = []
        clust_id = np.unique(np_results[:,1])
        self._unit_ids = list(clust_id)
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
        # determine the save path
        save_path = Path(save_path)
        if save_path.suffix == '.npy':
            if not str(save_path).endswith('spike_train.npy'):
                raise AttributeError("'save_path' is either a folder or an npy file "
                                     "ending with 'spike_train.npy")
        else:
            if not save_path.is_dir():
                save_path.mkdir()
            save_path = save_path / 'spike_train.npy'

        # write sorting (copyed from kilosortsortingextractor.py)
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
        np_results = np.stack((spike_times, spike_clusters),axis=1)
        np.save(save_path, np_results.astype(int))
