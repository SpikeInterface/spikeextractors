from spikeextractors import RecordingExtractor, SortingExtractor
import numpy as np


class OpenEphysRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_file, *, probefile=None, experiment_id=0, recording_id=0):
        try:
            import pyopenephys
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the OpenEphys extractor, install pyopenephys: \n\n"
                                      "pip install pyopenephys\n\n")
        RecordingExtractor.__init__(self)
        self._recording_file = recording_file
        self._recording = pyopenephys.File(recording_file,
                                           probefile).experiments[experiment_id].recordings[recording_id]

    def getChannelIds(self):
        return list(range(self._recording.analog_signals[0].signal.shape[0]))

    def getNumFrames(self):
        return self._recording.analog_signals[0].signal.shape[1]

    def getSamplingFrequency(self):
        return float(self._recording.sample_rate.rescale('Hz').magnitude)

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = range(self.getNumChannels())

        return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame]


class OpenEphysSortingExtractor(SortingExtractor):
    def __init__(self, recording_file, *, probefile=None, experiment_id=0, recording_id=0):
        try:
            import pyopenephys
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the OpenEphys extractor, install pyopenephys: \n\n"
                                      "pip install pyopenephys\n\n")
        SortingExtractor.__init__(self)
        self._recording_file = recording_file
        self._recording = pyopenephys.File(recording_file,
                                           probefile).experiments[experiment_id].recordings[recording_id]
        self._spiketrains = self._recording.spiketrains
        self._unit_ids = list([np.unique(st.clusters)[0] for st in self._spiketrains])

    def getUnitIds(self):
        return self._unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        st = self._spiketrains[unit_id]
        inds = np.where((start_frame <= (st.times * self._recording.sample_rate)) &
                        ((st.times * self._recording.sample_rate) < end_frame))
        return (st.times[inds] * self._recording.sample_rate).magnitude
