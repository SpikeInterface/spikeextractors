from spikeextractors import RecordingExtractor
from pathlib import Path
import numpy as np

try:
    import h5py
    HAVE_MEA1k = True
except ImportError:
    HAVE_MEA1k = False


class Mea1kRecordingExtractor(RecordingExtractor):

    extractor_name = 'MaxOneRecording'
    has_default_locations = True
    installed = HAVE_MEA1k  # check at class level if installed or not
    is_writable = True
    is_dumpable = True
    mode = 'file'
    extractor_gui_params = [
        {'name': 'file_path', 'type': 'file', 'title': "Path to file"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        RecordingExtractor.__init__(self)
        self._file_path = file_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._mapping = None
        self._signals = None
        self._initialize()
        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path, mode='r')
        self._mapping = self._filehandle['ephys']['mapping']
        self._channel_ids = list(self._mapping['channel'])
        self._num_channels = len(self._channel_ids)
        self._fs = float(self._filehandle['ephys']['frame_rate'][()])
        self._signals = self._filehandle['ephys']['signal']
        self._num_frames = self._signals.shape[1]

        for i_ch, ch in enumerate(self.get_channel_ids()):
            self.set_channel_property(ch, 'location', [self._mapping['x'][i_ch], self._mapping['y'][i_ch]])

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        if np.array(channel_ids).size > 1:
            assert np.all([ch in self.get_channel_ids() for ch in channel_ids])
            channel_idxs = [self.get_channel_ids().index(ch) for ch in channel_ids]
            if np.any(np.diff(channel_idxs) < 0):
                sorted_idx = np.argsort(channel_idxs)
                recordings = self._signals[np.sort(channel_idxs), start_frame:end_frame]
                return recordings[sorted_idx]
            else:
                return self._signals[np.array(channel_idxs), start_frame:end_frame]
        else:
            assert channel_ids in self.get_channel_ids()
            channel_idx = self.get_channel_ids().index(channel_ids)
            return self._signals[np.array(channel_idx), start_frame:end_frame]

    @staticmethod
    def write_recording(recording, save_path):
        save_path = Path(save_path)
        if save_path.suffix == '':
            save_path = Path(str(save_path) + '.h5')
        mapping_dtype = np.dtype([('electrode', np.int32), ('x', np.float64), ('y', np.float64),
                                  ('channel', np.int32)])

        assert 'location' in recording.get_shared_channel_property_names(), "'location' property is needed to write " \
                                                                            "max1k format"

        with h5py.File(save_path, 'w') as f:
            f.create_group('ephys')
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
            ephys.create_dataset('signal', data=recording.get_traces())
            


