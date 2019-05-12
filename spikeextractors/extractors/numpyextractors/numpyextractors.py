from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from pathlib import Path
import numpy as np


class NumpyRecordingExtractor(RecordingExtractor):
    has_default_locations = False
    def __init__(self, timeseries, samplerate, geom=None):
        RecordingExtractor.__init__(self)
        if isinstance(timeseries, str):
            if Path(timeseries).is_file():
                self._timeseries = np.load(timeseries)
        elif isinstance(timeseries, np.ndarray):
            self._timeseries = timeseries
        else:
            raise ValueError("'timeseries must be a .npy file name or a numpy array")
        self._samplerate = float(samplerate)
        self._geom = geom
        if geom is not None:
            for m in range(self._timeseries.shape[0]):
                self.set_channel_property(m, 'location', self._geom[m, :])

    def get_channel_ids(self):
        return list(range(self._timeseries.shape[0]))

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
            channel_ids = self.get_channel_ids()
        recordings = self._timeseries[:, start_frame:end_frame][channel_ids, :]
        return recordings

    @staticmethod
    def write_recording(recording, save_path):
        save_path = Path(save_path)
        np.save(save_path, recording.get_traces())


class NumpySortingExtractor(SortingExtractor):
    def __init__(self):
        SortingExtractor.__init__(self)
        self._units = {}
        # self._properties = {}

    def load_from_extractor(self, sorting):
        ids = sorting.get_unit_ids()
        for id in ids:
            self.add_unit(id, sorting.get_unit_spike_train(id))

    def set_times_labels(self, times, labels):
        units = np.sort(np.unique(labels))
        for unit in units:
            times0 = times[np.where(labels == unit)[0]]
            self.add_unit(unit_id=int(unit), times=times0)

    def add_unit(self, unit_id, times):
        self._units[unit_id] = dict(times=times)

    def get_unit_ids(self):
        return list(self._units.keys())

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._units[unit_id]['times']
        inds = np.where((start_frame <= times) & (times < end_frame))[0]
        return np.rint(times[inds]).astype(int)
