import numpy as np
from pathlib import Path
import csv
from typing import Union, Optional

from spikeextractors import SortingExtractor, RecordingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors.extraction_tools import read_python, check_get_unit_spike_train

PathType = Union[str, Path]


class PhyRecordingExtractor(BinDatRecordingExtractor):
    """
    RecordingExtractor for a Phy output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    """
    extractor_name = 'PhyRecording'
    has_default_locations = True
    has_unscaled = False
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path: PathType):
        RecordingExtractor.__init__(self)
        phy_folder = Path(folder_path)

        self.params = read_python(str(phy_folder / 'params.py'))
        datfile = [x for x in phy_folder.iterdir() if x.suffix == '.dat' or x.suffix == '.bin']

        if (phy_folder / 'channel_map_si.npy').is_file():
            channel_map = list(np.squeeze(np.load(phy_folder / 'channel_map_si.npy')))
            assert max(channel_map) < self.params['n_channels_dat'], "Channel map inconsistent with dat file."
        elif (phy_folder / 'channel_map.npy').is_file():
            channel_map = list(np.squeeze(np.load(phy_folder / 'channel_map.npy')))
            assert max(channel_map) < self.params['n_channels_dat'], "Channel map inconsistent with dat file."
        else:
            channel_map = list(range(self.params['n_channels_dat']))

        BinDatRecordingExtractor.__init__(self, datfile[0], sampling_frequency=float(self.params['sample_rate']),
                                          dtype=self.params['dtype'], numchan=self.params['n_channels_dat'],
                                          recording_channels=list(channel_map))

        if (phy_folder / 'channel_groups.npy').is_file():
            channel_groups = np.load(phy_folder / 'channel_groups.npy')
            assert len(channel_groups) == self.get_num_channels()
            self.set_channel_groups(channel_groups)

        if (phy_folder / 'channel_positions.npy').is_file():
            channel_locations = np.load(phy_folder / 'channel_positions.npy')
            assert len(channel_locations) == self.get_num_channels()
            self.set_channel_locations(channel_locations)

        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}


class PhySortingExtractor(SortingExtractor):
    """
    SortingExtractor for a Phy output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Phy folder (containing the params.py)
    exclude_cluster_groups: list (optional)
        List of cluster groups to exclude (e.g. ["noise", "mua"]
    """
    extractor_name = 'PhySorting'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path: PathType, exclude_cluster_groups: Optional[list] = None):
        SortingExtractor.__init__(self)
        phy_folder = Path(folder_path)

        spike_times = np.load(phy_folder / 'spike_times.npy')
        spike_templates = np.load(phy_folder / 'spike_templates.npy')

        if (phy_folder / 'spike_clusters.npy').is_file():
            spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            spike_clusters = spike_templates

        if (phy_folder / 'amplitudes.npy').is_file():
            amplitudes = np.squeeze(np.load(phy_folder / 'amplitudes.npy'))
        else:
            amplitudes = np.ones(len(spike_times))

        if (phy_folder / 'pc_features.npy').is_file():
            pc_features = np.squeeze(np.load(phy_folder / 'pc_features.npy'))
        else:
            pc_features = None

        clust_id = np.unique(spike_clusters)
        self._unit_ids = list(clust_id)
        spike_times.astype(int)
        self.params = read_python(str(phy_folder / 'params.py'))
        self._sampling_frequency = self.params['sample_rate']

        # set unit quality properties
        csv_tsv_files = [x for x in phy_folder.iterdir() if x.suffix == '.csv' or x.suffix == '.tsv']
        for f in csv_tsv_files:
            if f.suffix == '.csv':
                with f.open() as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            tokens = row[0].split("\t")
                            property_name = tokens[1]
                        else:
                            tokens = row[0].split("\t")
                            if int(tokens[0]) in self.get_unit_ids():
                                if 'cluster_group' in str(f):
                                    self.set_unit_property(int(tokens[0]), 'quality', tokens[1])
                                elif property_name == 'chan_grp' or property_name == 'ch_group':
                                    self.set_unit_property(int(tokens[0]), 'group', int(tokens[1]))
                                else:
                                    if isinstance(tokens[1], (int, np.int, float, str)):
                                        self.set_unit_property(int(tokens[0]), property_name, tokens[1])
                            line_count += 1
            elif f.suffix == '.tsv':
                with f.open() as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter='\t')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            property_name = row[1]
                        else:
                            if len(row) == 2:
                                if int(row[0]) in self.get_unit_ids():
                                    if 'cluster_group' in str(f):
                                        self.set_unit_property(int(row[0]), 'quality', row[1])
                                    elif property_name == 'chan_grp' or property_name == 'ch_group':
                                        self.set_unit_property(int(row[0]), 'group', int(row[1]))
                                    else:
                                        if isinstance(row[1], (int, float, str)) and len(row) == 2:
                                            self.set_unit_property(int(row[0]), property_name, row[1])
                        line_count += 1

        for unit in self.get_unit_ids():
            if 'quality' not in self.get_unit_property_names(unit):
                self.set_unit_property(unit, 'quality', 'unsorted')

        if exclude_cluster_groups is not None:
            if len(exclude_cluster_groups) > 0:
                included_units = []
                for u in self.get_unit_ids():
                    if self.get_unit_property(u, 'quality') not in exclude_cluster_groups:
                        included_units.append(u)
            else:
                included_units = self._unit_ids
        else:
            included_units = self._unit_ids

        original_units = self._unit_ids
        self._unit_ids = included_units
        # set features
        self._spiketrains = []
        for clust in self._unit_ids:
            idx = np.where(spike_clusters == clust)[0]
            self._spiketrains.append(spike_times[idx])
            self.set_unit_spike_features(clust, 'amplitudes', amplitudes[idx])
            if pc_features is not None:
                self.set_unit_spike_features(clust, 'pc_features', pc_features[idx])

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'exclude_cluster_groups': exclude_cluster_groups}

    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]
