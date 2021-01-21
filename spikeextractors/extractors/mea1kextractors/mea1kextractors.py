from spikeextractors import RecordingExtractor, SortingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args, check_get_unit_spike_train

try:
    import h5py
    HAVE_MEA1k = True
except ImportError:
    HAVE_MEA1k = False


class Mea1kRecordingExtractor(RecordingExtractor):
    extractor_name = 'Mea1kRecording'
    has_default_locations = True
    installed = HAVE_MEA1k  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Mea1kRecordingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path, load_spikes=True):
        assert HAVE_MEA1k, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._file_path = file_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._mapping = None
        self._signals = None
        self._version = None
        self._timestamps = None
        self._load_spikes = load_spikes
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'load_spikes': load_spikes}

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, mode='r')
        try:
            self._version = self._filehandle['version'][0].decode()
        except:
            try:
                self._version = self._filehandle['chipinformation']['software_version'][0].decode()
            except:
                self._version = '20161003'

        self._lsb = 1
        if int(self._version) == 20160704:
            self._signals = self._filehandle.get('sig')
            settings = self._filehandle.get('settings')

            if 'gain' in settings:
                self._gain = self._filehandle.get('settings/gain')[0]
            else:
                print("Couldn't read gain. Setting gain to 512")
                self._gain = 512

            self._bits = self._filehandle.get('bits', [])

            if 'lsb' in settings:
                self._lsb = self._filehandle['settings']['lsb'][0] * 1e6
            else:
                self._lsb = 3.3 / (1024 * self._gain) * 1e6

            assert 'mapping' in self._filehandle.keys(), "Could not load 'mapping' field"
            self._mapping = self._filehandle['mapping']
            self._fs = 20000
        elif int(self._version) >= 20161003:
            self._mapping = self._filehandle['ephys']['mapping']
            self._fs = float(self._filehandle['ephys']['frame_rate'][()])
            self._signals = self._filehandle['ephys']['signal']
        else:
            raise NotImplementedError(f"Version {self._version} of the Mea1k chip is not supported")

        channels = np.array(self._mapping['channel'])
        electrodes = np.array(self._mapping['electrode'])
        # remove unused channels
        routed_idxs = np.where(electrodes > -1)[0]
        self._channel_ids = list(channels[routed_idxs])
        self._electrode_ids = list(electrodes[routed_idxs])
        self._num_channels = len(self._channel_ids)
        self._num_frames = self._signals.shape[1]

        # This happens when only spikes are recorded
        if self._num_frames == 0:
            find_max_frame = True
        else:
            find_max_frame = False

        for i_ch, ch, el in zip(routed_idxs, self._channel_ids, self._electrode_ids):
            self.set_channel_locations([self._mapping['x'][i_ch], self._mapping['y'][i_ch]], ch)
            self.set_channel_property(ch, 'electrode', el)

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

            self._timestamps = np.round(np.arange(self.get_num_frames()) / self.get_sampling_frequency(), 6)

            for mf_idx, duration in zip(missing_frames_idxs, delays_in_frames):
                self._timestamps[mf_idx:] += np.round(duration / self.get_sampling_frequency(), 6)
        else:
            if verbose:
                print("No missing frames found")

    def frame_to_time(self, frames):
        if self._timestamps is None:
            return super().frame_to_time(frames)
        else:
            return self._timestamps[frames]

    def time_to_frame(self, times):
        if self._timestamps is None:
            return super().time_to_frame(times)
        else:
            return np.searchsorted(self._timestamps, times).astype('int64')

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_electrode_ids(self):
        return list(self._electrode_ids)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        if np.array(channel_ids).size > 1:
            if np.any(np.diff(channel_ids) < 0):
                sorted_channel_ids = np.sort(channel_ids)
                sorted_idx = np.array([list(sorted_channel_ids).index(ch) for ch in channel_ids])
                signals = self._signals[sorted_channel_ids, start_frame:end_frame][sorted_idx]
            else:
                signals = self._signals[np.array(channel_ids), start_frame:end_frame]
        else:
            signals = self._signals[np.array(channel_ids), start_frame:end_frame]
        if return_scaled:
            signals = signals.astype('float32')
            signals *= self._lsb
        return signals

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

    def _get_frame_numbers(self):
        bitvals = self._signals[-2:, :]
        frame_nos = np.bitwise_or(np.left_shift(bitvals[-1].astype('int64'), 16), bitvals[0])
        return frame_nos

    def _get_frame_number(self, index):
        bitvals = self._signals[-2:, index]
        frameno = bitvals[1] << 16 | bitvals[0]
        return frameno

    @staticmethod
    def write_recording(recording, save_path, chunk_size=None, chunk_mb=500):
        assert HAVE_MEA1k, Mea1kRecordingExtractor.installation_mesg
        save_path = Path(save_path)
        if save_path.suffix == '':
            save_path = Path(str(save_path) + '.h5')
        mapping_dtype = np.dtype([('electrode', np.int32), ('x', np.float64), ('y', np.float64),
                                  ('channel', np.int32)])

        assert 'location' in recording.get_shared_channel_property_names(), "'location' property is needed to write " \
                                                                            "max1k format"

        with h5py.File(save_path, 'w') as f:
            f.create_group('ephys')
            f.create_dataset('version', data=str(20161003))
            ephys = f['ephys']
            ephys.create_dataset('frame_rate', data=recording.get_sampling_frequency())
            ephys.create_dataset('frame_numbers', data=np.arange(recording.get_num_frames()))
            # save mapping
            mapping = np.empty(recording.get_num_channels(), dtype=mapping_dtype)
            x = recording.get_channel_locations()[:, 0]
            y = recording.get_channel_locations()[:, 1]
            for i, ch in enumerate(recording.get_channel_ids()):
                mapping[i] = (ch, x[i], y[i], ch)
            ephys.create_dataset('mapping', data=mapping)
            # save traces
            recording.write_to_h5_dataset_format('/ephys/signal', file_handle=f, time_axis=1,
                                                 chunk_size=chunk_size, chunk_mb=chunk_mb)


class Mea1kSortingExtractor(SortingExtractor):
    extractor_name = 'Mea1kSorting'
    installed = HAVE_MEA1k  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the Mea1kSortingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path):
        assert HAVE_MEA1k, self.installation_mesg
        SortingExtractor.__init__(self)
        self._file_path = file_path
        self._filehandle = None
        self._mapping = None
        self._version = None
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, mode='r')
        try:
            self._version = self._filehandle['version'][0].decode()
        except:
            try:
                self._version = self._filehandle['chipinformation']['software_version'][0].decode()
            except:
                self._version = '20161003'

        print(f"Chip version: {self._version}")
        self._lsb = 1
        if int(self._version) == 20160704:
            assert 'mapping' in self._filehandle.keys(), "Could not load 'mapping' field"
            self._mapping = self._filehandle['mapping']
            self._signals = self._filehandle.get('sig')
            self._sampling_frequency = 20000
        elif int(self._version) >= 20161003:
            self._mapping = self._filehandle['ephys']['mapping']
            self._sampling_frequency = float(self._filehandle['ephys']['frame_rate'][()])
            self._signals = self._filehandle['ephys']['signal']
        else:
            raise NotImplementedError(f"Version {self._version} of the Mea1k chip is not supported")

        self._first_frame = 0
        try:
            bitvals = self._signals[-2:, 0]
            self._first_frame = bitvals[1] << 16 | bitvals[0]
        except:
            print("Couldn't find first frame information. Setting to 0.")
        channels = np.array(self._mapping['channel'])
        electrodes = np.array(self._mapping['electrode'])
        # remove unused channels
        routed_idxs = np.where(electrodes > -1)[0]
        self._channel_ids = list(channels[routed_idxs])
        self._electrode_ids = list(electrodes[routed_idxs])

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

        unit_idx = self._unit_ids.index(unit_id)
        spiketrain = self._spiketrains[unit_idx]
        inds = np.where((start_frame <= spiketrain) & (spiketrain < end_frame))
        return spiketrain[inds]
