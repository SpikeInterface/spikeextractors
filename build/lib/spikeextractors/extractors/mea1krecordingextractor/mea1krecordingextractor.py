from spikeextractors import RecordingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args

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

    def __init__(self, file_path):
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
                return recordings[sorted_idx].astype('float')
            else:
                return (self._signals[np.array(channel_ids), start_frame:end_frame] * self._lsb).astype('float32')
        else:
            return (self._signals[np.array(channel_ids), start_frame:end_frame] * self._lsb).astype('float32')

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
