from spikeextractors import SortingExtractor
import numpy as np

try:
    import h5py
    HAVE_HS2SX = True
except ImportError:
    HAVE_HS2SX = False

class HS2SortingExtractor(SortingExtractor):

    extractor_name = 'HS2SortingExtractor'
    installed = HAVE_HS2SX  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording_file', 'type': 'path', 'title': "Path to file"},
    ]
    installation_mesg = "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, recording_file, load_unit_info=False):
        assert HAVE_HS2SX, "To use the HS2SortingExtractor install h5py: \n\n pip install h5py\n\n"
        SortingExtractor.__init__(self)
        self._recording_file = recording_file
        self._rf = h5py.File(self._recording_file, mode='r')
        if 'Sampling' in self._rf:
            if(self._rf['Sampling'][()] == 0):
                self._sampling_frequency = None
            else:
                self._sampling_frequency = self._rf['Sampling'][()]

        self._cluster_id = self._rf['cluster_id'][()]
        self._unit_ids = set(self._cluster_id)
        self._times = self._rf['times'][()]

        if(load_unit_info):
            self.load_unit_info()

    def load_unit_info(self):
        if 'centres' in self._rf.keys():
            self._unit_locs = self._rf['centres'][()]  # cache for faster access
            if self._unit_locs.shape[0] < 5:  # check if old, transposed format
                self._unit_locs = self._unit_locs.T
            for unit_id in self._unit_ids:
                self._unit_properties[unit_id] = {}
                self._unit_properties[unit_id]['unit_location'] = self._unit_locs[unit_id]
        inds = []  # get these only once
        for unit_id in self._unit_ids:
            inds.append(np.where(self._cluster_id==unit_id)[0])
        if 'data' in self._rf.keys():
            d = self._rf['data'][()]
            for i, unit_id in enumerate(self._unit_ids):
                self._unit_features[unit_id] = {}
                self._unit_features[unit_id]['spike_location'] = d[:, inds[i]].T
        else:
            for i, unit_id in enumerate(self._unit_ids):
                self._unit_features[unit_id] = {}
        if 'ch' in self._rf.keys():
            d = self._rf['ch'][()]
            for i, unit_id in enumerate(self._unit_ids):
                self._unit_features[unit_id]['max_channel'] = d[inds[i]]

    def get_unit_indices(self, x):
        return np.where(self._cluster_id == x)[0]

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
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
        # for now only create the entries required by any RecordingExtractor
        if sorting.get_sampling_frequency() is not None:
            rf.create_dataset("Sampling", data=sorting.get_sampling_frequency())
        else:
            rf.create_dataset("Sampling", data=0)
        rf.create_dataset("times", data=all_times)
        rf.create_dataset("cluster_id", data=all_labels)
        rf.close()
