from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path

try:
    import h5py
    HAVE_SCSX = True
except ImportError:
    HAVE_SCSX = False

class SpykingCircusSortingExtractor(SortingExtractor):

    extractor_name = 'SpykingCircusSortingExtractor'
    installed = HAVE_SCSX  # check at class level if installed or not
    _gui_params = [
        {'name': 'spykingcircus_folder', 'type': 'path', 'title': "Path to folder"},
    ]
    installation_mesg = "To use the SpykingCircusSortingExtractor install h5py: \n\n pip install h5py\n\n"
                               # error message when not installed
    def __init__(self, spykingcircus_folder):
        assert HAVE_SCSX, "To use the SpykingCircusSortingExtractor install h5py: \n\n pip install h5py\n\n"
        SortingExtractor.__init__(self)
        spykingcircus_folder = Path(spykingcircus_folder)
        files = spykingcircus_folder.iterdir()
        results = None
        for f in files:
            if 'result.hdf5' in str(f):
                results = f
            if 'result-merged.hdf5' in str(f):
                results = f
                break
        if results is None:
            raise Exception(spykingcircus_folder, " is not a spyking circus folder")
        f_results = h5py.File(results)
        self._spiketrains = []
        self._unit_ids = []
        for temp in f_results['spiketimes'].keys():
            self._spiketrains.append(f_results['spiketimes'][temp].value)
            self._unit_ids.append(int(temp.split('_')[-1]))

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
        assert HAVE_SCSX, "To use the SpykingCircusSortingExtractor install h5py: \n\n pip install h5py\n\n"
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / 'data.result.hdf5'
        elif save_path.suffix == '.hdf5':
            if not str(save_path).endswith('result.hdf5') or not str(save_path).endswith('result-merged.hdf5'):
                raise AttributeError("'save_path' is either a folder or an hdf5 file "
                                     "ending with 'result.hdf5' or 'result-merged.hdf5")
        else:
            save_path.mkdir()
            save_path = save_path / 'data.result.hdf5'
        F = h5py.File(save_path, 'w')
        spiketimes = F.create_group('spiketimes')

        for id in sorting.get_unit_ids():
            spiketimes.create_dataset('tmp_' + str(id), data=sorting.get_unit_spike_train(id))
