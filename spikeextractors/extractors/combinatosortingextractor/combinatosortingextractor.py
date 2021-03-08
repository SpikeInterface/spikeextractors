from pathlib import Path
import numpy as np
from spikeextractors import SortingExtractor
from spikeextractors.extraction_tools import check_get_unit_spike_train
from typing import Union


try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

PathType = Union[str, Path]


class CombinatoSortingExtractor(SortingExtractor):
    extractor_name = 'CombinatoSorting'
    installation_mesg = ""  # error message when not installed
    installed = HAVE_H5PY
    is_writable = False
    installation_mesg = "To use the CombinatoSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, datapath: PathType, sampling_frequency=None, user='simple',det_sign = 'both'):
        super().__init__()
        datapath = Path(datapath)
        assert datapath.is_dir(), 'Folder {} doesn\'t exist'.format(datapath)
        if sampling_frequency is None:
            h5_path = str(datapath) + '.h5'
            if Path(h5_path).exists():
                with h5py.File(h5_path, mode='r') as f:
                    sampling_frequency = f['sr'][0]
        self.set_sampling_frequency(sampling_frequency)
        det_file = str(datapath / Path('data_' + datapath.stem + '.h5'))
        sort_cat_files = []
        for sign in ['neg', 'pos']:
            if det_sign in ['both', sign]:
                sort_cat_file = datapath / Path('sort_{}_{}/sort_cat.h5'.format(sign,user))
                if sort_cat_file.exists():
                    sort_cat_files.append((sign, str(sort_cat_file)))
        unit_counter = 0
        self._spike_trains = {}
        metadata = {}
        unsorted = []
        fdet = h5py.File(det_file, mode='r')
        for sign, sfile in sort_cat_files:
            with h5py.File(sfile, mode='r') as f:
                sp_class = f['classes'][()]
                gaux = f['groups'][()]
                groups = {g:gaux[gaux[:, 1] == g, 0] for g in np.unique(gaux[:, 1])} #array of classes per group
                group_type = {group: g_type for group,g_type in f['types'][()]}
                sp_index = f['index'][()]

            times_css = fdet[sign]['times'][()]
            for gr, cls in groups.items():
                if group_type[gr] == -1: #artifacts
                    continue
                elif group_type[gr] == 0: #unsorted
                    unsorted.append(np.rint(times_css[sp_index[np.isin(sp_class,cls)]] * (sampling_frequency/1000)))
                    continue

                unit_counter = unit_counter + 1
                self._spike_trains[unit_counter] = np.rint(times_css[sp_index[np.isin(sp_class, cls)]] * (sampling_frequency / 1000))
                metadata[unit_counter] = {'det_sign': sign,
                                          'group_type': 'single-unit' if group_type[gr] else 'multi-unit'}

        fdet.close()

        self._unsorted_train = np.array([])
        if len(unsorted) == 1:
            self._unsorted_train = unsorted[0]
        elif len(unsorted) == 2: #unsorted in both signs
            self._unsorted_train = np.sort(np.concatenate(unsorted), kind='mergesort')

        self._unit_ids = list(range(1, unit_counter+1))
        for u in self._unit_ids:
            for prop,value in metadata[u].items():
                self.set_unit_property(u, prop, value)

    def get_unit_ids(self):
        return self._unit_ids


    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        start_frame = start_frame or 0
        end_frame = end_frame or np.infty
        st = self._spike_trains[unit_id]
        return st[(st >= start_frame) & (st < end_frame)]

    def get_unsorted_spike_train(self, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        start_frame = start_frame or 0
        end_frame = end_frame or np.infty
        u = self._unsorted_train
        return u[(u >= start_frame) & (u < end_frame)]



