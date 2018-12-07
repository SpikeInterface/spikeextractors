from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from pathlib import Path
import numpy as np


class NumpyRecordingExtractor(RecordingExtractor):
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
                self.setChannelProperty(m, 'location', self._geom[m, :])

    def getChannelIds(self):
        return list(range(self._timeseries.shape[0]))

    def getNumFrames(self):
        return self._timeseries.shape[1]

    def getSamplingFrequency(self):
        return self._samplerate

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        recordings = self._timeseries[:, start_frame:end_frame][channel_ids, :]
        return recordings

    @staticmethod
    def writeRecording(recording, save_path):
        save_path = Path(save_path)
        np.save(save_path, recording.getTraces())


class NumpySortingExtractor(SortingExtractor):
    def __init__(self):
        SortingExtractor.__init__(self)
        self._unit_ids = []
        self._units = {}
        # self._properties = {}

    def loadFromExtractor(self, sorting):
        ids = sorting.getUnitIds()
        for id in ids:
            self.addUnit(id, sorting.getUnitSpikeTrain(id))

    def setTimesLabels(self, times, labels):
        units = np.sort(np.unique(labels))
        for unit in units:
            times0 = times[np.where(labels == unit)[0]]
            self.addUnit(unit_id=int(unit), times=times0)

    def addUnit(self, unit_id, times):
        self._unit_ids.append(unit_id)
        self._units[unit_id] = dict(times=times)

    def getUnitIds(self):
        return self._unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._units[unit_id]['times']
        inds = np.where((start_frame <= times) & (times < end_frame))[0]
        return np.rint(times[inds]).astype(int)
