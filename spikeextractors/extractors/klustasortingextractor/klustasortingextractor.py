from spikeextractors import SortingExtractor
from spikeextractors.tools import read_python

import numpy as np
import os
from os.path import join
import h5py


class KlustaSortingExtractor(SortingExtractor):
    def __init__(self, kwikfile):
        SortingExtractor.__init__(self)
        kwikfile = os.path.abspath(kwikfile)
        F = h5py.File(kwikfile)
        channel_groups = F.get('channel_groups')
        self._spiketrains = []
        self._unit_ids = []
        for cgroup in channel_groups:
            group_id = int(cgroup)
            for cluster_id in channel_groups[cgroup]['clusters']['main']:
                clusters = np.array(channel_groups[cgroup]['spikes']['clusters']['main'])
                idx = np.nonzero(clusters == int(cluster_id))
                st = np.array(channel_groups[cgroup]['spikes']['time_samples'])[idx]
                self._spiketrains.append(st)
                self._unit_ids.append(int(cluster_id))
                self.setUnitProperty(int(cluster_id), 'channel_group', group_id)

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
        if os.path.isdir(save_path):
            save_path = join(save_path, 'klusta.kwik')
        elif save_path.endswith('kwik'):
            pass
        else:
            os.makedirs(save_path)
            save_path = join(save_path, 'klusta.kwik')
        F = h5py.File(save_path, 'w')
        F.attrs.create('kwik_version', data=2)
        if 'channel_group' in sorting.getUnitPropertyNames():
            cgroups = np.unique([sorting.getUnitProperty(unit, 'channel_group') for unit in sorting.getUnitIds()])
        else:
            cgroups = [0]

        channel_groups = F.create_group('channel_groups')

        for cgroup in cgroups:
            channel_group = channel_groups.create_group(str(cgroup))
            time_samples = np.array([])
            cluster_main = np.array([])
            if 'channel_group' in sorting.getUnitPropertyNames():
                idxs = [unit for unit in sorting.getUnitIds() if
                        sorting.getUnitProperty(unit, 'channel_group') == cgroup]
            else:
                idxs = sorting.getUnitIds()
            clust = channel_group.create_group('clusters')
            clust.create_dataset('main', data=idxs)
            clust.create_dataset('original', data=idxs)
            for id in idxs:
                st = sorting.getUnitSpikeTrain(id)
                cl = [id] * len(sorting.getUnitSpikeTrain(id))
                time_samples = np.concatenate((time_samples, np.array(st)))
                cluster_main = np.concatenate((cluster_main, np.array(cl)))
            sorting_idxs = np.argsort(time_samples)
            time_samples = time_samples[sorting_idxs].astype(int)
            cluster_main = cluster_main[sorting_idxs].astype(int)
            spikes = channel_group.create_group('spikes')
            spikes.create_dataset('time_samples', data=time_samples)
            clusters = spikes.create_group('clusters')
            clusters.create_dataset('main', data=cluster_main)
            clusters.create_dataset('original', data=cluster_main)
