from .RecordingExtractor import RecordingExtractor
import numpy as np


# Encapsulates a sub-dataset

class SubRecordingExtractor(RecordingExtractor):
    def __init__(self, parent_recording, *, channel_ids=None, renamed_channel_ids=None, start_frame=None,
                 end_frame=None):
        RecordingExtractor.__init__(self)
        self._parent_recording = parent_recording
        self._channel_ids = channel_ids
        self._renamed_channel_ids = renamed_channel_ids
        self._start_frame = start_frame
        self._end_frame = end_frame
        if self._channel_ids is None:
            self._channel_ids = self._parent_recording.getChannelIds()
        if self._renamed_channel_ids is None:
            self._renamed_channel_ids = self._channel_ids
        if self._start_frame is None:
            self._start_frame = 0
        if self._end_frame is None:
            self._end_frame = self._parent_recording.getNumFrames()
        self._original_channel_id_lookup = {}
        for i in range(len(self._channel_ids)):
            self._original_channel_id_lookup[self._renamed_channel_ids[i]] = self._channel_ids[i]
        self.copyChannelProperties(parent_recording, channel_ids=self._renamed_channel_ids)

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        sf = self._start_frame + start_frame
        ef = self._start_frame + end_frame
        original_ch_ids = self.getOriginalChannelIds(channel_ids)
        return self._parent_recording.getTraces(channel_ids=original_ch_ids, start_frame=sf, end_frame=ef)

    def getChannelIds(self):
        return self._renamed_channel_ids

    def getNumFrames(self):
        return self._end_frame - self._start_frame

    def getSamplingFrequency(self):
        return self._parent_recording.getSamplingFrequency()

    def frameToTime(self, frame):
        frame2 = frame + self._start_frame
        time1 = self._parent_recording.frameToTime(frame2)
        time2 = time1 - self._parent_recording.frameToTime(self._start_frame)
        return time2

    def timeToFrame(self, time):
        time2 = time + self._parent_recording.frameToTime(self._start_frame)
        frame1 = self._parent_recording.timeToFrame(time2)
        frame2 = frame1 - self._start_frame
        return frame2

    def getSnippets(self, *, reference_frames, snippet_len, channel_ids=None):
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        cf = self._start_frame + np.array(reference_frames)
        original_ch_ids = []
        original_ch_ids = self.getOriginalChannelIds(channel_ids)
        return self._parent_recording.getSnippets(reference_frames=reference_frames, snippet_len=snippet_len,
                                                  channel_ids=original_ch_ids)

    def copyChannelProperties(self, recording, channel_ids=None):
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        if isinstance(channel_ids, int):
            recording_ch_id = channel_ids
            if recording is self._parent_recording:
                recording_ch_id = self.getOriginalChannelIds(channel_ids)
            curr_property_names = recording.getChannelPropertyNames(channel_id=recording_ch_id)
            for curr_property_name in curr_property_names:
                value = recording.getChannelProperty(channel_id=recording_ch_id, property_name=curr_property_name)
                self.setChannelProperty(channel_id=channel_ids, property_name=curr_property_name, value=value)
        else:
            for channel_id in channel_ids:
                recording_ch_id = channel_id
                if recording is self._parent_recording:
                    recording_ch_id = self.getOriginalChannelIds(channel_id)
                curr_property_names = recording.getChannelPropertyNames(channel_id=recording_ch_id)
                for curr_property_name in curr_property_names:
                    value = recording.getChannelProperty(channel_id=recording_ch_id, property_name=curr_property_name)
                    self.setChannelProperty(channel_id=channel_id, property_name=curr_property_name, value=value)

    def getOriginalChannelIds(self, channel_ids):
        if isinstance(channel_ids, (int, np.integer)):
            if channel_ids in self.getChannelIds():
                original_ch_ids = self._original_channel_id_lookup[channel_ids]
            else:
                raise ValueError("Non-valid channel_id")
        else:
            original_ch_ids = []
            for channel_id in channel_ids:
                if isinstance(channel_id, (int, np.integer)):
                    if channel_id in self.getChannelIds():
                        original_ch_id = self._original_channel_id_lookup[channel_id]
                        original_ch_ids.append(original_ch_id)
                    else:
                        raise ValueError("Non-valid channel_id")
                else:
                    raise ValueError("channel_id must be an int")
        return original_ch_ids
