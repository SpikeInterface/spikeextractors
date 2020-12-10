from spikeextractors import SortingExtractor, RecordingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors.extraction_tools import read_python, write_python, check_valid_unit_id
import numpy as np
from pathlib import Path
import csv


class PhyRecordingExtractor(BinDatRecordingExtractor):
    extractor_name = 'PhyRecording'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path):
        RecordingExtractor.__init__(self)
        phy_folder = Path(folder_path)

        self.params = read_python(str(phy_folder / 'params.py'))
        datfile = [x for x in phy_folder.iterdir() if x.suffix == '.dat' or x.suffix == '.bin']

        if (phy_folder / 'channel_map_si.npy').is_file():
            channel_map = list(np.squeeze(np.load(phy_folder / 'channel_map_si.npy')))
            assert len(channel_map) == self.params['n_channels_dat']
        elif (phy_folder / 'channel_map.npy').is_file():
            channel_map = list(np.squeeze(np.load(phy_folder / 'channel_map.npy')))
            assert len(channel_map) == self.params['n_channels_dat']
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
    extractor_name = 'PhySortingExtractor'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path, exclude_cluster_groups=None, load_waveforms=False, verbose=False):
        SortingExtractor.__init__(self)
        phy_folder = Path(folder_path)

        spike_times = np.load(phy_folder / 'spike_times.npy')
        spike_templates = np.load(phy_folder / 'spike_templates.npy')

        if (phy_folder /'spike_clusters.npy').is_file():
            spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            spike_clusters = spike_templates

        if (phy_folder / 'amplitudes.npy').is_file():
            amplitudes = np.load(phy_folder / 'amplitudes.npy')
        else:
            amplitudes = np.ones(len(spike_times))

        if (phy_folder /'pc_features.npy').is_file():
            pc_features = np.load(phy_folder / 'pc_features.npy')
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
                                elif property_name == 'chan_grp':
                                    self.set_unit_property(int(tokens[0]), 'group', tokens[1])
                                else:
                                    if isinstance(tokens[1], (int, np.int, float, np.float, str)):
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
                                    elif property_name == 'chan_grp':
                                        self.set_unit_property(int(row[0]), 'group', row[1])
                                    else:
                                        if isinstance(row[1], (int, np.int, float, np.float, str)) and len(row) == 2:
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

        if load_waveforms:
            datfile = [x for x in phy_folder.iterdir() if x.suffix == '.dat' or x.suffix == '.bin']

            recording = BinDatRecordingExtractor(datfile[0], sampling_frequency=float(self.params['sample_rate']),
                                                 dtype=self.params['dtype'], numchan=self.params['n_channels_dat'])
            # if channel groups are present, compute waveforms by group
            if (phy_folder / 'channel_groups.npy').is_file():
                channel_groups = np.load(phy_folder / 'channel_groups.npy')
                assert len(channel_groups) == recording.get_num_channels()
                recording.set_channel_groups(channel_groups)
                for u_i, u in enumerate(self.get_unit_ids()):
                    if verbose:
                        print('Computing waveform by group for unit', u)
                    frames_before = int(0.5 / 1000. * recording.get_sampling_frequency())
                    frames_after = int(2 / 1000. * recording.get_sampling_frequency())
                    spiketrain = self.get_unit_spike_train(u)
                    if 'group' in self.get_unit_property_names(u):
                        group_idx = np.where(channel_groups == int(self.get_unit_property(u, 'group')))[0]
                        wf = recording.get_snippets(reference_frames=spiketrain,
                                                    snippet_len=[frames_before, frames_after],
                                                    channel_ids=group_idx)
                    else:
                        wf = recording.get_snippets(reference_frames=spiketrain,
                                                    snippet_len=[frames_before, frames_after])
                        max_chan = np.unravel_index(np.argmin(np.mean(wf, axis=0)), np.mean(wf, axis=0).shape)[0]
                        group = recording.get_channel_groups(int(max_chan))
                        self.set_unit_property(u, 'group', group)
                        group_idx = np.where(channel_groups == group)[0]
                        wf = wf[:, group_idx]
                    self.set_unit_spike_features(u, 'waveforms', wf)
            else:
                for u_i, u in enumerate(self.get_unit_ids()):
                    if verbose:
                        print('Computing full waveform for unit', u)
                    frames_before = 0.5 * recording.get_sampling_frequency()
                    frames_after = 2 * recording.get_sampling_frequency()
                    spiketrain = self.get_unit_spike_train(u)
                    wf = recording.get_snippets(reference_frames=spiketrain,
                                                snippet_len=[int(frames_before), int(frames_after)])
                    self.set_unit_spike_features(u, 'waveforms', wf)
        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'exclude_cluster_groups': exclude_cluster_groups,
                        'load_waveforms': load_waveforms, 'verbose': verbose}

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
