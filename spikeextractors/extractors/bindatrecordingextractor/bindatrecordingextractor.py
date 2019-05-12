from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import read_binary
import os
import numpy as np
from pathlib import Path


class BinDatRecordingExtractor(RecordingExtractor):

    extractor_name = 'BinDatRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'datfile', 'type': 'path', 'title': "Path to file"},
        {'name': 'samplerate', 'type': 'float', 'title': "Sampling rate in HZ"},
        {'name': 'numchan', 'type': 'int', 'title': "Number of channels"},
        {'name': 'dtype', 'type': 'np.dtype', 'title': "The dtype of underlying data"},
        {'name': 'recording_channels', 'type': 'list', 'value':None, 'default':None, 'title': "List of recording channels"},
        {'name': 'frames_first', 'type': 'bool', 'value':True, 'default':True, 'title': "Frames first"},
        {'name': 'offset', 'type': 'int', 'value':0, 'default':0, 'title': "Offset in binary file"},
        {'name': 'probe_path', 'type': 'str', 'title': "Path to probe file (csv or prb)"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, datfile, samplerate, numchan, dtype, recording_channels=None, frames_first=True, geom=None, offset=0):
        RecordingExtractor.__init__(self)
        self._datfile = Path(datfile)
        self._frame_first = frames_first
        self._timeseries = read_binary(self._datfile, numchan, dtype, frames_first, offset)
        self._samplerate = float(samplerate)
        self._geom = geom

        if recording_channels is not None:
            assert len(recording_channels) == self._timeseries.shape[0], \
                'Provided recording channels have the wrong length'
            self._channels = recording_channels
        else:
            self._channels = list(range(self._timeseries.shape[0]))

        if geom is not None:
            for m in range(self._timeseries.shape[0]):
                self.set_channel_property(m, 'location', self._geom[m, :])

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._samplerate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = list(range(self._timeseries.shape[0]))
        else:
            channel_ids = [self._channels.index(ch) for ch in channel_ids]
        recordings = self._timeseries[:, start_frame:end_frame][channel_ids, :]
        return recordings

    @staticmethod
    def write_recording(recording, save_path, dtype=None, transpose=False):
        save_path = Path(save_path)
        if dtype == None:
            dtype = np.float32
        if not transpose:
            with save_path.open('wb') as f:
                np.transpose(np.array(recording.get_traces(), dtype=dtype)).tofile(f)
        elif transpose:
            with save_path.open('wb') as f:
                np.array(recording.get_traces(), dtype=dtype).tofile(f)
