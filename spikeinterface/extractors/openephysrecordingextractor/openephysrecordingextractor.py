from spikeinterface import RecordingExtractor
import pyopenephys

class OpenEphysRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_file, *, probefile=None, experiment_id=0, recording_id=0):
        RecordingExtractor.__init__(self)
        self._recording_file = recording_file
        self._recording = pyopenephys.File(recording_file,
                                           probefile).experiments[experiment_id].recordings[recording_id]

    def getNumChannels(self):
        return self._recording.analog_signals[0].signal.shape[0]

    def getNumFrames(self):
        return self._recording.analog_signals[0].signal.shape[1]

    def getSamplingFrequency(self):
        return self._recording.sample_rate

    def getTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = range(self.getNumChannels())

        return self._recording.analog_signals[0].signal[channel_ids, start_frame:end_frame]

    def getChannelInfo(self, channel_id):
        return dict()