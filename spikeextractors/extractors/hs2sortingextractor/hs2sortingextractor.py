from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path

try:
    import h5py
    HAVE_HS2SX = True
except ImportError:
    HAVE_HS2SX = False

class HS2SortingExtractor(SortingExtractor):
    extractor_name = 'HS2SortingExtractor'
    installed = HAVE_HS2SX  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path, load_unit_info=True):
        assert HAVE_HS2SX, "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"
        SortingExtractor.__init__(self)
        self._recording_file = file_path
        self._rf = h5py.File(self._recording_file, mode='r')
        if 'Sampling' in self._rf:
            if self._rf['Sampling'][()] == 0:
                self._sampling_frequency = None
            else:
                self._sampling_frequency = self._rf['Sampling'][()]

        self._cluster_id = self._rf['cluster_id'][()]
        self._unit_ids = set(self._cluster_id)
        self._times = self._rf['times'][()]

        if load_unit_info:
            self.load_unit_info()

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'load_unit_info': load_unit_info}

    def load_unit_info(self):
        if ('centres' in self._rf.keys()) and (len(self._times)>0):
            self._unit_locs = self._rf['centres'][()]  # cache for faster access
            for u_i, unit_id in enumerate(self._unit_ids):
                self.set_unit_property(unit_id, property_name='unit_location', value=self._unit_locs[u_i])
        inds = []  # get these only once
        for unit_id in self._unit_ids:
            inds.append(np.where(self._cluster_id==unit_id)[0])
        if ('data' in self._rf.keys()) and (len(self._times)>0):
            d = self._rf['data'][()]
            for i, unit_id in enumerate(self._unit_ids):
                self._unit_features[unit_id] = {}
                self._unit_features[unit_id]['spike_location'] = d[:, inds[i]].T
        else:
            for i, unit_id in enumerate(self._unit_ids):
                self._unit_features[unit_id] = {}
        if ('ch' in self._rf.keys()) and (len(self._times)>0):
            d = self._rf['ch'][()]
            for i, unit_id in enumerate(self._unit_ids):
                self._unit_features[unit_id]['max_channel'] = d[inds[i]]

    def get_unit_indices(self, x):
        return np.where(self._cluster_id == x)[0]

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._times[self.get_unit_indices(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def write_sorting(sorting, save_path):
        assert HAVE_HS2SX, "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"
        unit_ids = sorting.get_unit_ids()
        times_list = []
        labels_list = []
        for i in range(len(unit_ids)):
            unit = unit_ids[i]
            times = sorting.get_unit_spike_train(unit_id=unit)
            times_list.append(times)
            labels_list.append(np.ones(times.shape, dtype=int) * unit)
        all_times = np.concatenate(times_list)
        all_labels = np.concatenate(labels_list)
        
        rf = h5py.File(save_path, mode='w')
        if sorting.get_sampling_frequency() is not None:
            rf.create_dataset("Sampling", data=sorting.get_sampling_frequency())
        else:
            rf.create_dataset("Sampling", data=0)
        if 'unit_location' in sorting.get_shared_unit_property_names():
            spike_centres = [sorting.get_unit_property(u,'unit_location') for u in sorting.get_unit_ids()]
            spike_centres = np.array(spike_centres)
            rf.create_dataset("centres", data=spike_centres)
        if 'spike_location' in sorting.get_shared_unit_spike_feature_names():
            spike_loc_x = []
            spike_loc_y = []
            for u in sorting.get_unit_ids():
                l = sorting.get_unit_spike_features(u,'spike_location')
                spike_loc_x.append(l[:,0])
                spike_loc_y.append(l[:,1])        
            spike_loc = np.vstack((np.concatenate(spike_loc_x),np.concatenate(spike_loc_y)))
            rf.create_dataset("data", data=spike_loc)
        if 'max_channel' in sorting.get_shared_unit_spike_feature_names():
            spike_max_channel = np.concatenate([sorting.get_unit_spike_features(u,'max_channel') for u in sorting.get_unit_ids()])
            rf.create_dataset("ch", data=spike_max_channel)
            
        rf.create_dataset("times", data=all_times)
        rf.create_dataset("cluster_id", data=all_labels)
        rf.close()
