"""
kwik structure based on:
https://github.com/kwikteam/phy-doc/blob/master/docs/kwik-format.md

cluster_group defaults based on:
https://github.com/kwikteam/phy-doc/blob/master/docs/kwik-model.md

04/08/20
"""


from spikeextractors import SortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors.extraction_tools import read_python, check_valid_unit_id
import numpy as np
from pathlib import Path


try:
    import h5py
    HAVE_KLSX = True
except ImportError:
    HAVE_KLSX = False


# noinspection SpellCheckingInspection
class KlustaRecordingExtractor(BinDatRecordingExtractor):
    extractor_name = 'KlustaRecordingExtractor'
    has_default_locations = False
    installed = HAVE_KLSX  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, folder_path):
        assert HAVE_KLSX, self.installation_mesg
        klustafolder = Path(folder_path).absolute()
        config_file = [f for f in klustafolder.iterdir() if f.suffix == '.prm'][0]
        dat_file = [f for f in klustafolder.iterdir() if f.suffix == '.dat'][0]
        assert config_file.is_file() and dat_file.is_file(), "Not a valid klusta folder"
        config = read_python(str(config_file))
        sampling_frequency = config['traces']['sample_rate']
        n_channels = config['traces']['n_channels']
        dtype = config['traces']['dtype']

        BinDatRecordingExtractor.__init__(self, file_path=dat_file, sampling_frequency=sampling_frequency, numchan=n_channels,
                                          dtype=dtype)

        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}


# noinspection SpellCheckingInspection
class KlustaSortingExtractor(SortingExtractor):
    extractor_name = 'KlustaSortingExtractor'
    installed = HAVE_KLSX  # check at class level if installed or not
    installation_mesg = "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed
    is_writable = True
    mode = 'file_or_folder'

    default_cluster_groups = {0: 'Noise', 1: 'MUA', 2: 'Good', 3: 'Unsorted'}

    def __init__(self, file_or_folder_path, exclude_cluster_groups=None):
        assert HAVE_KLSX, self.installation_mesg
        SortingExtractor.__init__(self)
        kwik_file_or_folder = Path(file_or_folder_path)
        kwikfile = None
        klustafolder = None
        if kwik_file_or_folder.is_file():
            assert kwik_file_or_folder.suffix == '.kwik', "Not a '.kwik' file"
            kwikfile = Path(kwik_file_or_folder).absolute()
            klustafolder = kwikfile.parent
        elif kwik_file_or_folder.is_dir():
            klustafolder = kwik_file_or_folder
            kwikfiles = [f for f in kwik_file_or_folder.iterdir() if f.suffix == '.kwik']
            if len(kwikfiles) == 1:
                kwikfile = kwikfiles[0]
        assert kwikfile is not None, "Could not load '.kwik' file"

        try:
            config_file = [f for f in klustafolder.iterdir() if f.suffix == '.prm'][0]
            config = read_python(str(config_file))
            sampling_frequency = config['traces']['sample_rate']
            self._sampling_frequency = sampling_frequency
        except Exception as e:
            print("Could not load sampling frequency info")

        kf_reader = h5py.File(kwikfile, 'r')
        self._spiketrains = []
        self._unit_ids = []
        unique_units = []
        klusta_units = []
        cluster_groups_name = []
        groups = []
        unit = 0

        cs_to_exclude = []
        valid_group_names = [i[1].lower() for i in self.default_cluster_groups.items()]
        if exclude_cluster_groups is not None:
            assert isinstance(exclude_cluster_groups, list), 'exclude_cluster_groups should be a list'
            for ec in exclude_cluster_groups:
                assert ec in valid_group_names, f'select exclude names out of: {valid_group_names}'
                cs_to_exclude.append(ec.lower())

        for channel_group in kf_reader.get('/channel_groups'):
            chan_cluster_id_arr = kf_reader.get(f'/channel_groups/{channel_group}/spikes/clusters/main')[()]
            chan_cluster_times_arr = kf_reader.get(f'/channel_groups/{channel_group}/spikes/time_samples')[()]
            chan_cluster_ids = np.unique(chan_cluster_id_arr)  # if clusters were merged in gui,
                                                                # the original id's are still in the kwiktree, but
                                                                # in this array

            for cluster_id in chan_cluster_ids:
                cluster_frame_idx = np.nonzero(chan_cluster_id_arr == cluster_id)  # the [()] is a h5py thing
                st = chan_cluster_times_arr[cluster_frame_idx]
                assert st.shape[0] > 0, 'no spikes in cluster'
                cluster_group = kf_reader.get(f'/channel_groups/{channel_group}/clusters/main/{cluster_id}').attrs['cluster_group']

                assert cluster_group in self.default_cluster_groups.keys(), f'cluster_group not in "default_dict: {cluster_group}'
                cluster_group_name = self.default_cluster_groups[cluster_group]

                if cluster_group_name.lower() in cs_to_exclude:
                    continue

                self._spiketrains.append(st)
                klusta_units.append(int(cluster_id))
                unique_units.append(unit)
                unit += 1
                groups.append(int(channel_group))
                cluster_groups_name.append(cluster_group_name)

        if len(np.unique(klusta_units)) == len(np.unique(unique_units)):
            self._unit_ids = klusta_units
        else:
            print('Klusta units are not unique! Using unique unit ids')
            self._unit_ids = unique_units
        for i, u in enumerate(self._unit_ids):
            self.set_unit_property(u, 'group', groups[i])
            self.set_unit_property(u, 'quality', cluster_groups_name[i].lower())

        self._kwargs = {'file_or_folder_path': str(Path(file_or_folder_path).absolute())}

    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]
