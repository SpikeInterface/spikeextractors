from spikeextractors import RecordingExtractor

try:
    import h5py
    HAVE_MAX = True
except ImportError:
    HAVE_MAX = False


class MaxOneRecordingExtractor(RecordingExtractor):

    extractor_name = 'MaxOneRecordingExtractor'
    has_default_locations = True
    installed = HAVE_MAX  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    extractor_gui_params = [
        {'name': 'file_path', 'type': 'file', 'title': "Path to file"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        RecordingExtractor.__init__(self)
        self._file_path= file_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._mapping = None
        self._initialize()

    def _initialize(self):
        self._filehandle = h5py.File(self._file_path)
        self._mapping = self._filehandle['mapping']
        self._channel_ids = self._mapping['channel']
        self._num_channels = len(self._channel_ids)
        self._fs = 20000
        self._num_frames = self._filehandle.get('sig').shape[1]

        for i_ch, ch in enumerate(self.get_channel_ids()):
            self.set_channel_property(ch, 'location', [self._mapping['x'][i_ch], self._mapping['y'][i_ch]])

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        return self._filehandle['sig'][channel_ids, start_frame:end_frame] * self._filehandle['settings']['lsb'] * 1e6
