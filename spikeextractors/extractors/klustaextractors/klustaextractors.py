from spikeextractors import SortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors.extraction_tools import read_python
import numpy as np
from pathlib import Path


try:
    import h5py
    HAVE_KLSX = True
except ImportError:
    HAVE_KLSX = False


class KlustaRecordingExtractor(BinDatRecordingExtractor):
    extractor_name = 'KlustaRecordingExtractor'
    installed = HAVE_KLSX  # check at class level if installed or not
    _gui_params = [
        {'name': 'kwikfile', 'type': 'path', 'title': "Path to file"},
        {'name': 'probe_path', 'type': 'path', 'title': "Path to probe file (.csv or .prb)"}
    ]
    installation_mesg = "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, klustafolder):
        assert HAVE_KLSX, "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"
        klustafolder = Path(klustafolder).absolute()
        config_file = [f for f in klustafolder.iterdir() if f.suffix == '.prm'][0]
        dat_file = [f for f in klustafolder.iterdir() if f.suffix == '.dat'][0]
        assert config_file.is_file() and dat_file.is_file(), "Not a valid klusta folder"
        config = read_python(str(config_file))
        sample_rate = config['traces']['sample_rate']
        n_channels = config['traces']['n_channels']
        dtype = config['traces']['dtype']

        BinDatRecordingExtractor.__init__(self, datfile=dat_file, samplerate=sample_rate, numchan=n_channels,
                                          dtype=dtype)


class KlustaSortingExtractor(SortingExtractor):
    extractor_name = 'KlustaSortingExtractor'
    installed = HAVE_KLSX  # check at class level if installed or not
    installation_mesg = "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, kwik_file_or_folder):
        assert HAVE_KLSX, "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"
        SortingExtractor.__init__(self)
        kwik_file_or_folder = Path(kwik_file_or_folder)
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
            sample_rate = config['traces']['sample_rate']
            self._sampling_frequency = sample_rate
        except Exception as e:
            print("Could not load sampling frequency info")

        F = h5py.File(kwikfile)
        channel_groups = F.get('channel_groups')
        self._spiketrains = []
        self._unit_ids = []
        unique_units = []
        klusta_units = []
        groups = []
        unit = 0
        for cgroup in channel_groups:
            group_id = int(cgroup)
            try:
                cluster_ids = channel_groups[cgroup]['clusters']['main']
            except Exception as e:
                print('Unable to extract clusters from', kwikfile)
                continue
            for cluster_id in channel_groups[cgroup]['clusters']['main']:
                clusters = np.array(channel_groups[cgroup]['spikes']['clusters']['main'])
                idx = np.nonzero(clusters == int(cluster_id))
                st = np.array(channel_groups[cgroup]['spikes']['time_samples'])[idx]
                self._spiketrains.append(st)
                klusta_units.append(int(cluster_id))
                unique_units.append(unit)
                unit += 1
                groups.append(group_id)
        if len(np.unique(klusta_units)) == len(np.unique(unique_units)):
            self._unit_ids = klusta_units
        else:
            print('Klusta units are not unique! Using unique unit ids')
            self._unit_ids = unique_units
        for i, u in enumerate(self._unit_ids):
            self.set_unit_property(u, 'group', groups[i])

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
        assert HAVE_KLSX, "To use the KlustaSortingExtractor install h5py: \n\n pip install h5py\n\n"
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / 'klusta.kwik'
        elif save_path.suffix == '.kwik':
            pass
        else:
            save_path.mkdir()
            save_path = save_path / 'klusta.kwik'
        F = h5py.File(save_path, 'w')
        F.attrs.create('kwik_version', data=2)
        if 'group' in sorting.get_unit_property_names():
            cgroups = np.unique([sorting.get_unit_property(unit, 'group') for unit in sorting.get_unit_ids()])
        else:
            cgroups = [0]

        channel_groups = F.create_group('channel_groups')

        for cgroup in cgroups:
            channel_group = channel_groups.create_group(str(cgroup))
            time_samples = np.array([])
            cluster_main = np.array([])
            if 'group' in sorting.get_unit_property_names():
                idxs = [unit for unit in sorting.get_unit_ids() if
                        sorting.get_unit_property(unit, 'group') == cgroup]
            else:
                idxs = sorting.get_unit_ids()
            clust = channel_group.create_group('clusters')
            clust.create_dataset('main', data=idxs)
            clust.create_dataset('original', data=idxs)
            for id in idxs:
                st = sorting.get_unit_spike_train(id)
                cl = [id] * len(sorting.get_unit_spike_train(id))
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
