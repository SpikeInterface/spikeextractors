from spikeextractors import SortingExtractor

import numpy as np
import os
from os.path import join


class PhySortingExtractor(SortingExtractor):
    def __init__(self, phy_folder):
        SortingExtractor.__init__(self)
        spike_times = np.load(join(phy_folder, 'spike_times.npy'))
        spike_templates = np.load(join(phy_folder, 'spike_templates.npy'))
        amplitudes = np.load(join(phy_folder, 'amplitudes.npy'))
        pc_features = np.load(join(phy_folder, 'pc_features.npy'))
        # pc_feature_ind = np.load(join(phy_folder, 'pc_feature_ind.npy'))
        # self._templates = np.load(join(phy_folder, 'templates.npy'))
        # template_ind = np.load(join(phy_folder, 'template_ind.npy'))

        if os.path.isfile(join(phy_folder, 'spike_clusters.npy')):
            spike_clusters = np.load(join(phy_folder, 'spike_clusters.npy'))
        else:
            spike_clusters = spike_templates

        self._spiketrains = []
        self._amps = []
        self._pc_features = []
        clust_id = np.unique(spike_clusters)
        self._unit_ids = list(clust_id)
        spike_times.astype(int)

        self._spiketrais = []
        for clust in self._unit_ids:
            idx = np.where(spike_clusters == clust)[0]
            self._spiketrains.append(spike_times[idx])
            self._amps.append(amplitudes[idx])
            self._pc_features.append(pc_features[idx])

        # set features
        for u_i, unit in enumerate(self.getUnitIds()):
            self.setUnitSpikeFeatures(unit, 'amplitudes', self._amps[u_i])
            self.setUnitSpikeFeatures(unit, 'pc_features', self._pc_features[u_i])

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
        spike_times = np.array([])
        spike_clusters = np.array([])
        amplitudes = np.array([])
        pc_features = np.array([])

        for id in sorting.getUnitIds():
            st = sorting.getUnitSpikeTrain(id)
            cl = [id] * len(sorting.getUnitSpikeTrain(id))
            spike_times = np.concatenate((spike_times, np.array(st)))
            spike_clusters = np.concatenate((spike_clusters, np.array(cl)))
            if 'amplitudes' in sorting.getUnitSpikeFeatureNames():
                amp = sorting.getUnitSpikeFeatures(id, 'amplitudes')
                amplitudes = np.concatenate((amplitudes, np.array(amp)))
            if 'pc_features' in sorting.getUnitSpikeFeatureNames():
                pc_feat = sorting.getUnitSpikeFeatures(id, 'pc_features')
                if len(pc_features) == 0:
                    pc_features = pc_feat
                else:
                    pc_features = np.vstack((pc_features, np.array(pc_feat)))

        sorting_idxs = np.argsort(spike_times)
        spike_times = spike_times[sorting_idxs]
        spike_clusters = spike_clusters[sorting_idxs]

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        np.save(join(save_path, 'spike_times.npy'), spike_times[:, np.newaxis].astype(int))
        np.save(join(save_path, 'spike_clusters.npy'), spike_clusters[:, np.newaxis].astype(int))
        if len(amplitudes) > 0:
            amplitudes = amplitudes[sorting_idxs]
            np.save(join(save_path, 'amplitudes.npy'), amplitudes[:, np.newaxis].astype(int))
        if len(pc_features) > 0:
            pc_features = pc_features[sorting_idxs]
            np.save(join(save_path, 'pc_features.npy'), pc_features)
            pc_feature_ind = np.tile(np.arange(pc_features.shape[-1]), (len(sorting.getUnitIds()), 1))
            np.save(join(save_path, 'pc_feature_ind.npy'), pc_feature_ind)
