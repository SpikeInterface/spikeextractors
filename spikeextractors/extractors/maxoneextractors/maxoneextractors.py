from spikeextractors import RecordingExtractor, SortingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args, check_valid_unit_id

try:
    import h5py
    HAVE_MAX = True
except ImportError:
    HAVE_MAX = False


class MaxOneRecordingExtractor(RecordingExtractor):
    extractor_name = 'MaxOneRecording'
    has_default_locations = True
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the MaxOneRecordingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path):
        assert HAVE_MAX, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._file_path = file_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._mapping = None
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    def __del__(self):
        self._filehandle.close()

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, 'r')
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

        # This happens when only spikes are recorded
        if self._num_frames == 0:
            find_max_frame = True
        else:
            find_max_frame = False

        for i_ch, ch, el in zip(routed_idxs, self._channel_ids, self._electrode_ids):
            self.set_channel_locations([self._mapping['x'][i_ch], self._mapping['y'][i_ch]], ch)
            self.set_channel_property(ch, 'electrode', el)

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
                        self.set_channel_property(ch, 'activity', counts[activity_channels.index(ch)])
                    else:
                        self.set_channel_property(ch, 'activity', 0)

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_electrode_ids(self):
        return list(self._electrode_ids)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if np.array(channel_ids).size > 1:
            if np.any(np.diff(channel_ids) < 0):
                sorted_idx = np.argsort(channel_ids)
                recordings = self._signals[np.sort(channel_ids), start_frame:end_frame]
                return (recordings[sorted_idx] * self._lsb).astype('float')
            else:
                return (self._signals[np.array(channel_ids), start_frame:end_frame] * self._lsb).astype('float32')
        else:
            return (self._signals[np.array(channel_ids), start_frame:end_frame] * self._lsb).astype('float32')

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        bitvals = self._signals[-2:, 0]
        first_frame = bitvals[1] << 16 | bitvals[0]
        bits = self._filehandle['bits']
        bit_frames = bits['frameno'] - first_frame
        bit_states = bits['bits']
        bit_idxs = np.where((bit_frames >= start_frame) & (bit_frames < end_frame))[0]
        ttl_frames = bit_frames[bit_idxs]
        ttl_states = bit_states['bit_idxs']
        ttl_states[ttl_states == 0] = -1
        return ttl_frames, ttl_states


class MaxOneSortingExtractor(SortingExtractor):
    extractor_name = 'MaxOneSorting'
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the MaxOneSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path):
        assert HAVE_MAX, self.installation_mesg
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

    @check_valid_unit_id
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
