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
            self._channel_ids = self._parent_recording.get_channel_ids()
        if self._renamed_channel_ids is None:
            self._renamed_channel_ids = self._channel_ids
        if self._start_frame is None:
            self._start_frame = 0
        if self._end_frame is None:
            self._end_frame = self._parent_recording.get_num_frames()
        self._original_channel_id_lookup = {}
        for i in range(len(self._channel_ids)):
            self._original_channel_id_lookup[self._renamed_channel_ids[i]] = self._channel_ids[i]
        self.copy_channel_properties(parent_recording, channel_ids=self._renamed_channel_ids)

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        sf = self._start_frame + start_frame
        ef = self._start_frame + end_frame
        original_ch_ids = self.get_original_channel_ids(channel_ids)
        return self._parent_recording.get_traces(channel_ids=original_ch_ids, start_frame=sf, end_frame=ef)

    def get_channel_ids(self):
        return self._renamed_channel_ids

    def get_num_frames(self):
        return self._end_frame - self._start_frame

    def get_sampling_frequency(self):
        return self._parent_recording.get_sampling_frequency()

    def frame_to_time(self, frame):
        frame2 = frame + self._start_frame
        time1 = self._parent_recording.frame_to_time(frame2)
        time2 = time1 - self._parent_recording.frame_to_time(self._start_frame)
        return time2

    def time_to_frame(self, time):
        time2 = time + self._parent_recording.frame_to_time(self._start_frame)
        frame1 = self._parent_recording.time_to_frame(time2)
        frame2 = frame1 - self._start_frame
        return frame2

    def get_snippets(self, *, reference_frames, snippet_len, channel_ids=None):
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        cf = self._start_frame + np.array(reference_frames)
        original_ch_ids = []
        original_ch_ids = self.get_original_channel_ids(channel_ids)
        return self._parent_recording.get_snippets(reference_frames=reference_frames, snippet_len=snippet_len,
                                                  channel_ids=original_ch_ids)

    def copy_channel_properties(self, recording, channel_ids=None):
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        if isinstance(channel_ids, int):
            recording_ch_id = channel_ids
            if recording is self._parent_recording:
                recording_ch_id = self.get_original_channel_ids(channel_ids)
            curr_property_names = recording.get_channel_property_names(channel_id=recording_ch_id)
            for curr_property_name in curr_property_names:
                value = recording.get_channel_property(channel_id=recording_ch_id, property_name=curr_property_name)
                self.set_channel_property(channel_id=channel_ids, property_name=curr_property_name, value=value)
        else:
            for channel_id in channel_ids:
                recording_ch_id = channel_id
                if recording is self._parent_recording:
                    recording_ch_id = self.get_original_channel_ids(channel_id)
                curr_property_names = recording.get_channel_property_names(channel_id=recording_ch_id)
                for curr_property_name in curr_property_names:
                    value = recording.get_channel_property(channel_id=recording_ch_id, property_name=curr_property_name)
                    self.set_channel_property(channel_id=channel_id, property_name=curr_property_name, value=value)

    def get_original_channel_ids(self, channel_ids):
        if isinstance(channel_ids, (int, np.integer)):
            if channel_ids in self.get_channel_ids():
                original_ch_ids = self._original_channel_id_lookup[channel_ids]
            else:
                raise ValueError("Non-valid channel_id")
        else:
            original_ch_ids = []
            for channel_id in channel_ids:
                if isinstance(channel_id, (int, np.integer)):
                    if channel_id in self.get_channel_ids():
                        original_ch_id = self._original_channel_id_lookup[channel_id]
                        original_ch_ids.append(original_ch_id)
                    else:
                        raise ValueError("Non-valid channel_id")
                else:
                    raise ValueError("channel_id must be an int")
        return original_ch_ids
