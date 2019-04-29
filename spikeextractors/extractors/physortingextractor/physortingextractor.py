from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path
import csv

class PhySortingExtractor(SortingExtractor):

    extractor_name = 'PhySortingExtractor'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'phy_folder', 'type': 'str', 'title': "Path to folder"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, phy_folder):
        SortingExtractor.__init__(self)
        phy_folder = Path(phy_folder)

        spike_times = np.load(phy_folder / 'spike_times.npy')
        spike_templates = np.load(phy_folder / 'spike_templates.npy')
        amplitudes = np.load(phy_folder / 'amplitudes.npy')
        pc_features = np.load(phy_folder / 'pc_features.npy')

        if (phy_folder /'spike_clusters.npy').is_file():
            spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            spike_clusters = spike_templates

        self._spiketrains = []
        clust_id = np.unique(spike_clusters)
        self._unit_ids = list(clust_id)
        spike_times.astype(int)
        # set features
        for clust in self._unit_ids:
            idx = np.where(spike_clusters == clust)[0]
            self._spiketrains.append(spike_times[idx])
            self.set_unit_spike_features(clust, 'amplitudes', amplitudes[idx])
            self.set_unit_spike_features(clust, 'pc_features', pc_features[idx])

        # set unit quality properties
        if (phy_folder / 'cluster_groups.csv').is_file():
            with open(phy_folder / 'cluster_groups.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        tokens = row[0].split("\t")
                        self.set_unit_property(int(tokens[0]), 'quality', tokens[1])
                        line_count += 1

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
        spike_clusters = np.array([])
        amplitudes = np.array([])
        pc_features = np.array([])

        for id in sorting.get_unit_ids():
            st = sorting.get_unit_spike_train(id)
            cl = [id] * len(sorting.get_unit_spike_train(id))
            spike_times = np.concatenate((spike_times, np.array(st)))
            spike_clusters = np.concatenate((spike_clusters, np.array(cl)))
            if 'amplitudes' in sorting.get_unit_spike_feature_names():
                amp = sorting.get_unit_spike_features(id, 'amplitudes')
                amplitudes = np.concatenate((amplitudes, np.array(amp)))
            if 'pc_features' in sorting.get_unit_spike_feature_names():
                pc_feat = sorting.get_unit_spike_features(id, 'pc_features')
                if len(pc_features) == 0:
                    pc_features = pc_feat
                else:
                    pc_features = np.vstack((pc_features, np.array(pc_feat)))

        sorting_idxs = np.argsort(spike_times)
        spike_times = spike_times[sorting_idxs]
        spike_clusters = spike_clusters[sorting_idxs]

        if not save_path.is_dir():
            save_path.mkdirs()
        np.save(save_path /'spike_times.npy', spike_times[:, np.newaxis].astype(int))
        np.save(save_path / 'spike_clusters.npy', spike_clusters[:, np.newaxis].astype(int))
        if len(amplitudes) > 0:
            amplitudes = amplitudes[sorting_idxs]
            np.save(save_path / 'amplitudes.npy', amplitudes[:, np.newaxis].astype(int))
        if len(pc_features) > 0:
            pc_features = pc_features[sorting_idxs]
            np.save(save_path / 'pc_features.npy', pc_features)
            pc_feature_ind = np.tile(np.arange(pc_features.shape[-1]), (len(sorting.get_unit_ids()), 1))
            np.save(save_path / 'pc_feature_ind.npy', pc_feature_ind)
