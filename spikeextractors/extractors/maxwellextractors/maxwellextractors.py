from spikeextractors import RecordingExtractor, SortingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args, check_get_unit_spike_train

try:
    import h5py
    HAVE_MAX = True
except ImportError:
    HAVE_MAX = False

installation_mesg = "To use the MaxOneRecordingExtractor install h5py: \n\n pip install h5py\n\n"


class MaxOneRecordingExtractor(RecordingExtractor):
    extractor_name = 'MaxOneRecording'
    has_default_locations = True
    has_unscaled = True
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = installation_mesg

    def __init__(self, file_path, load_spikes=True, rec_name='rec0000'):
        assert self.installed, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._file_path = file_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._load_spikes = load_spikes
        self._mapping = None
        self._rec_name = rec_name
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'load_spikes': load_spikes}

    def __del__(self):
        self._filehandle.close()

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, 'r')
        self._version = self._filehandle['version'][0].decode()

        if int(self._version) == 20160704:
            # old format
            self._mapping = self._filehandle['mapping']
            if 'lsb' in self._filehandle['settings'].keys():
                self._lsb = self._filehandle['settings']['lsb'][0] * 1e6
            else:
                print("Couldn't read lsb. Setting lsb to 1")
                self._lsb = 1.
            channels = np.array(self._mapping['channel'])
            electrodes = np.array(self._mapping['electrode'])
            # remove unused channels
            routed_idxs = np.where(electrodes > -1)[0]
            self._channel_ids = list(channels[routed_idxs])
            self._electrode_ids = list(electrodes[routed_idxs])
            self._num_channels = len(self._channel_ids)
            self._fs = float(20000)
            self._signals = self._filehandle['sig']
            self._num_frames = self._signals.shape[1]
        elif int(self._version) > 20160704:
            # new format
            well_name = 'well000'
            rec_name = self._rec_name
            settings = self._filehandle['wells'][well_name][rec_name]['settings']
            self._mapping = settings['mapping']
            if 'lsb' in settings.keys():
                self._lsb = settings['lsb'][()][0] * 1e6
            else:
                self._lsb = 1.
            channels = np.array(self._mapping['channel'])
            electrodes = np.array(self._mapping['electrode'])
            # remove unused channels
            routed_idxs = np.where(electrodes > -1)[0]
            self._channel_ids = list(channels[routed_idxs])
            self._electrode_ids = list(electrodes[routed_idxs])
            self._num_channels = len(self._channel_ids)
            self._fs = settings['sampling'][()][0]
            self._signals = self._filehandle['wells'][well_name][rec_name]['groups']['routed']['raw']
            self._num_frames = self._signals.shape[1]
        else:
            raise Exception("Could not parse the MaxOne file")

        # This happens when only spikes are recorded
        if self._num_frames == 0:
            find_max_frame = True
        else:
            find_max_frame = False

        for i_ch, ch, el in zip(routed_idxs, self._channel_ids, self._electrode_ids):
            self.set_channel_locations([self._mapping['x'][i_ch], self._mapping['y'][i_ch]], ch)
            self.set_channel_property(ch, 'electrode', el)

        # set gains
        self.set_channel_gains(self._lsb)

        if self._load_spikes:
            if 'proc0' in self._filehandle:
                if 'spikeTimes' in self._filehandle['proc0']:
                    spikes = self._filehandle['proc0']['spikeTimes']

                    spike_mask = [True] * len(spikes)
                    for i, ch in enumerate(spikes['channel']):
                        if ch not in self._channel_ids:
                            spike_mask[i] = False
                    spikes_channels = np.array(spikes['channel'])[spike_mask]

                    if find_max_frame:
                        self._num_frames = np.ptp(spikes['frameno'])

                    # load activity as property
                    activity_channels, counts = np.unique(spikes_channels, return_counts=True)
                    # transform to spike rate
                    duration = float(self._num_frames) / self._fs
                    counts = counts.astype(float) / duration
                    activity_channels = list(activity_channels)
                    for ch in self.get_channel_ids():
                        if ch in activity_channels:
                            self.set_channel_property(ch, 'spike_rate', counts[activity_channels.index(ch)])
                            spike_amplitudes = spikes[np.where(spikes['channel'] == ch)]['amplitude']
                            self.set_channel_property(ch, 'spike_amplitude', np.median(spike_amplitudes))
                        else:
                            self.set_channel_property(ch, 'spike_rate', 0)
                            self.set_channel_property(ch, 'spike_amplitude', 0)

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_electrode_ids(self):
        return list(self._electrode_ids)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    def correct_for_missing_frames(self, verbose=False):
        """
        Corrects for missing frames. The correct times can be retrieved with the frame_to_time and time_to_frame
        functions.
        Parameters
        ----------
        verbose: bool
            If True, output is verbose
        """
        frame_idxs_span = self._get_frame_number(self.get_num_frames() - 1) - self._get_frame_number(0)
        if frame_idxs_span > self.get_num_frames():
            if verbose:
                print(f"Found missing frames! Correcting for it (this might take a while)")

            framenos = self._get_frame_numbers()
            # find missing frames
            diff_frames = np.diff(framenos)
            missing_frames_idxs = np.where(diff_frames > 1)[0]

            delays_in_frames = []
            for mf_idx in missing_frames_idxs:
                delays_in_frames.append(diff_frames[mf_idx])

            if verbose:
                print(f"Found {len(delays_in_frames)} missing intervals")

            times = np.round(np.arange(self.get_num_frames()) / self.get_sampling_frequency(), 6)

            for mf_idx, duration in zip(missing_frames_idxs, delays_in_frames):
                times[mf_idx:] += np.round(duration / self.get_sampling_frequency(), 6)
            self.set_times(times)
        else:
            if verbose:
                print("No missing frames found")

    def _get_frame_numbers(self):
        bitvals = self._signals[-2:, :]
        frame_nos = np.bitwise_or(np.left_shift(bitvals[-1].astype('int64'), 16), bitvals[0])
        return frame_nos

    def _get_frame_number(self, index):
        bitvals = self._signals[-2:, index]
        frameno = bitvals[1] << 16 | bitvals[0]
        return frameno

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        if np.array(channel_ids).size > 1:
            if np.any(np.diff(channel_ids) < 0):
                sorted_channel_ids = np.sort(channel_ids)
                sorted_idx = np.array([list(sorted_channel_ids).index(ch) for ch in channel_ids])
                traces = self._signals[sorted_channel_ids, start_frame:end_frame][sorted_idx]
            else:
                traces = self._signals[np.array(channel_ids), start_frame:end_frame]
        else:
            traces = self._signals[np.array(channel_ids), start_frame:end_frame]
        return traces

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        bitvals = self._signals[-2:, 0]
        first_frame = bitvals[1] << 16 | bitvals[0]
        bits = self._filehandle['bits']
        bit_frames = bits['frameno'] - first_frame
        bit_states = bits['bits']
        bit_idxs = np.where((bit_frames >= start_frame) & (bit_frames < end_frame))[0]
        ttl_frames = bit_frames[bit_idxs]
        ttl_states = bit_states[bit_idxs]
        ttl_states[ttl_states == 0] = -1
        return ttl_frames, ttl_states


class MaxOneSortingExtractor(SortingExtractor):
    extractor_name = 'MaxOneSorting'
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = installation_mesg

    def __init__(self, file_path):
        assert self.installed, self.installation_mesg
        SortingExtractor.__init__(self)
        self._file_path = file_path
        self._filehandle = None
        self._mapping = None
        self._version = None
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, 'r')
        self._mapping = self._filehandle['mapping']
        self._signals = self._filehandle['sig']

        bitvals = self._signals[-2:, 0]
        self._first_frame = bitvals[1] << 16 | bitvals[0]

        channels = np.array(self._mapping['channel'])
        electrodes = np.array(self._mapping['electrode'])
        # remove unused channels
        routed_idxs = np.where(electrodes > -1)[0]
        self._channel_ids = list(channels[routed_idxs])
        self._unit_ids = list(electrodes[routed_idxs])
        self._sampling_frequency = float(20000)

        self._spiketrains = []
        self._unit_ids = []

        try:
            spikes = self._filehandle['proc0']['spikeTimes']
            for u in self._channel_ids:
                spiketrain_idx = np.where(spikes['channel'] == u)[0]
                if len(spiketrain_idx) > 0:
                    self._unit_ids.append(u)
                    spiketrain = spikes['frameno'][spiketrain_idx] - self._first_frame
                    idxs_greater_0 = np.where(spiketrain >= 0)[0]
                    self._spiketrains.append(spiketrain[idxs_greater_0])
                    self.set_unit_spike_features(u, 'amplitude', spikes['amplitude'][spiketrain_idx][idxs_greater_0])
        except:
            raise AttributeError("Spike times information are missing from the .h5 file")

    def get_unit_ids(self):
        return self._unit_ids

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        unit_idx = self._unit_ids.index(unit_id)
        spiketrain = self._spiketrains[unit_idx]
        inds = np.where((start_frame <= spiketrain) & (spiketrain < end_frame))
        return spiketrain[inds]


class MaxTwoRecordingExtractor(RecordingExtractor):
    extractor_name = 'MaxTwoRecording'
    has_default_locations = True
    has_unscaled = True
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = installation_mesg

    def __init__(self, file_path, well_name='well000', rec_name='rec0000', load_spikes=True):
        assert self.installed, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._file_path = file_path
        self._well_name = well_name
        self._rec_name = rec_name
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._mapping = None
        self._load_spikes = load_spikes
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'well_name': well_name, 'rec_name': rec_name,
                        'load_spikes': load_spikes}

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, 'r')
        settings = self._filehandle['wells'][self._well_name][self._rec_name]['settings']
        self._mapping = settings['mapping']
        if 'lsb' in settings.keys():
            self._lsb = settings['lsb'][()][0] * 1e6
        else:
            self._lsb = 1.
        channels = np.array(self._mapping['channel'])
        electrodes = np.array(self._mapping['electrode'])
        # remove unused channels
        routed_idxs = np.where(electrodes > -1)[0]
        self._channel_ids = list(channels[routed_idxs])
        self._electrode_ids = list(electrodes[routed_idxs])
        self._num_channels = len(self._channel_ids)
        self._fs = settings['sampling'][()][0]
        self._signals = self._filehandle['wells'][self._well_name][self._rec_name]['groups']['routed']['raw']
        self._num_frames = self._signals.shape[1]

        # This happens when only spikes are recorded
        if self._num_frames == 0:
            find_max_frame = True
        else:
            find_max_frame = False

        for i_ch, ch, el in zip(routed_idxs, self._channel_ids, self._electrode_ids):
            self.set_channel_locations([self._mapping['x'][i_ch], self._mapping['y'][i_ch]], ch)
            self.set_channel_property(ch, 'electrode', el)
        # set gains
        self.set_channel_gains(self._lsb)

        if self._load_spikes:
            if "spikes" in self._filehandle["wells"][self._well_name][self._rec_name].keys():
                spikes = self._filehandle["wells"][self._well_name][self._rec_name]["spikes"]

                spike_mask = [True] * len(spikes)
                for i, ch in enumerate(spikes['channel']):
                    if ch not in self._channel_ids:
                        spike_mask[i] = False
                spikes_channels = np.array(spikes['channel'])[spike_mask]

                if find_max_frame:
                    self._num_frames = np.ptp(spikes['frameno'])

                # load activity as property
                activity_channels, counts = np.unique(spikes_channels, return_counts=True)
                # transform to spike rate
                duration = float(self._num_frames) / self._fs
                counts = counts.astype(float) / duration
                activity_channels = list(activity_channels)
                for ch in self.get_channel_ids():
                    if ch in activity_channels:
                        self.set_channel_property(ch, 'spike_rate', counts[activity_channels.index(ch)])
                        spike_amplitudes = spikes[np.where(spikes['channel'] == ch)]['amplitude']
                        self.set_channel_property(ch, 'spike_amplitude', np.median(spike_amplitudes))
                    else:
                        self.set_channel_property(ch, 'spike_rate', 0)
                        self.set_channel_property(ch, 'spike_amplitude', 0)

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    @staticmethod
    def get_well_names(file_path):
        with h5py.File(file_path, 'r') as f:
            wells = list(f["wells"])
        return wells

    @staticmethod
    def get_recording_names(file_path, well_name):
        with h5py.File(file_path, 'r') as f:
            assert well_name in f["wells"], f"Well name should be among: " \
                                            f"{MaxTwoRecordingExtractor.get_well_names(file_path)}"
            rec_names = list(f["wells"][well_name].keys())
        return rec_names

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
        if np.array(channel_idxs).size > 1:
            if np.any(np.diff(channel_idxs) < 0):
                sorted_channel_ids = np.sort(channel_idxs)
                sorted_idx = np.array([list(sorted_channel_ids).index(ch) for ch in channel_idxs])
                traces = self._signals[sorted_channel_ids, start_frame:end_frame][sorted_idx]
            else:
                traces = self._signals[np.array(channel_idxs), start_frame:end_frame]
        else:
            traces = self._signals[np.array(channel_idxs), start_frame:end_frame]
        return traces


class MaxTwoSortingExtractor(SortingExtractor):
    extractor_name = 'MaxTwoSorting'
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = installation_mesg

    def __init__(self, file_path, well_name='well000', rec_name='rec0000'):
        assert self.installed, self.installation_mesg
        SortingExtractor.__init__(self)
        self._file_path = file_path
        self._well_name = well_name
        self._rec_name = rec_name
        self._filehandle = None
        self._mapping = None
        self._version = None
        self._initialize()
        self._sampling_frequency = self._fs
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'well_name': well_name, 'rec_name': rec_name}

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, 'r')
        settings = self._filehandle['wells'][self._well_name][self._rec_name]['settings']
        self._mapping = settings['mapping']
        if 'lsb' in settings.keys():
            self._lsb = settings['lsb'][()] * 1e6
        else:
            self._lsb = 1.
        channels = np.array(self._mapping['channel'])
        electrodes = np.array(self._mapping['electrode'])
        # remove unused channels
        routed_idxs = np.where(electrodes > -1)[0]
        self._channel_ids = list(channels[routed_idxs])
        self._unit_ids = list(electrodes[routed_idxs])
        self._fs = settings['sampling'][()][0]
        self._first_frame = self._filehandle['wells'][self._well_name][self._rec_name] \
            ['groups']['routed']['frame_nos'][0]

        self._spiketrains = []
        self._unit_ids = []
        try:
            spikes = self._filehandle["wells"][self._well_name][self._rec_name]["spikes"]
            for u in self._channel_ids:
                spiketrain_idx = np.where(spikes['channel'] == u)[0]
                if len(spiketrain_idx) > 0:
                    self._unit_ids.append(u)
                    spiketrain = spikes['frameno'][spiketrain_idx] - self._first_frame
                    idxs_greater_0 = np.where(spiketrain >= 0)[0]
                    self._spiketrains.append(spiketrain[idxs_greater_0])
                    self.set_unit_spike_features(u, 'amplitude', spikes['amplitude'][spiketrain_idx][idxs_greater_0])
        except:
            raise AttributeError("Spike times information are missing from the .h5 file")

    def get_unit_ids(self):
        return self._unit_ids

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        unit_idx = self._unit_ids.index(unit_id)
        spiketrain = self._spiketrains[unit_idx]
        inds = np.where((start_frame <= spiketrain) & (spiketrain < end_frame))
        return spiketrain[inds]

