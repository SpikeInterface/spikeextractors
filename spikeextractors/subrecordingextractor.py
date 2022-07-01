from .recordingextractor import RecordingExtractor
from .extraction_tools import check_get_traces_args, cast_start_end_frame, check_get_ttl_args
import numpy as np


# Encapsulates a sub-dataset
class SubRecordingExtractor(RecordingExtractor):
    def __init__(self, parent_recording, *, channel_ids=None, renamed_channel_ids=None, start_frame=None,
                 end_frame=None):
        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
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
        if self._end_frame > self._parent_recording.get_num_frames():
            self._end_frame = self._parent_recording.get_num_frames()
        self._original_channel_id_lookup = {}

        for i in range(len(self._channel_ids)):
            self._original_channel_id_lookup[self._renamed_channel_ids[i]] = self._channel_ids[i]
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(parent_recording, channel_ids=self._renamed_channel_ids)

        # avoid rescaling twice
        self.clear_channel_gains()
        self.clear_channel_offsets()

        self.is_filtered = self._parent_recording.is_filtered
        self.has_unscaled = self._parent_recording.has_unscaled

        # update dump dict
        self._kwargs = {'parent_recording': parent_recording.make_serialized_dict(), 'channel_ids': channel_ids,
                        'renamed_channel_ids': renamed_channel_ids, 'start_frame': start_frame, 'end_frame': end_frame}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        sf = self._start_frame + start_frame
        ef = self._start_frame + end_frame
        original_ch_ids = self.get_original_channel_ids(channel_ids)
        return self._parent_recording.get_traces(channel_ids=original_ch_ids, start_frame=sf, end_frame=ef,
                                                 return_scaled=return_scaled)

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        sf = self._start_frame + start_frame
        ef = self._start_frame + end_frame
        sf, ef = cast_start_end_frame(sf, ef)
        try:
            ttl_frames, ttl_states = self._parent_recording.get_ttl_events(start_frame=sf, end_frame=ef,
                                                                           channel_id=channel_id)
            ttl_frames -= self._start_frame
            return ttl_frames, ttl_states
        except NotImplementedError:
            raise NotImplementedError("The parent recording does implement the 'get_ttl_events method'")

    def get_channel_ids(self):
        return list(self._renamed_channel_ids)

    def get_num_frames(self):
        return self._end_frame - self._start_frame

    def get_sampling_frequency(self):
        return self._parent_recording.get_sampling_frequency()

    def frame_to_time(self, frame):
        frame2 = frame + self._start_frame
        time1 = self._parent_recording.frame_to_time(frame2)
        start_time = self._parent_recording.frame_to_time(self._start_frame)
        return np.round(time1 - start_time, 6)

    def time_to_frame(self, time):
        time2 = time + self._parent_recording.frame_to_time(self._start_frame)
        frame1 = self._parent_recording.time_to_frame(time2)
        frame2 = frame1 - self._start_frame
        return frame2.astype('int64')

    def get_snippets(self, reference_frames, snippet_len, channel_ids=None, return_scaled=True):
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        reference_frames_shift = self._start_frame + np.array(reference_frames)
        original_ch_ids = self.get_original_channel_ids(channel_ids)
        return self._parent_recording.get_snippets(reference_frames=reference_frames_shift, snippet_len=snippet_len,
                                                   channel_ids=original_ch_ids, return_scaled=return_scaled)

    def copy_channel_properties(self, recording, channel_ids=None):
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        if isinstance(channel_ids, (int, np.integer)):
            recording_ch_id = channel_ids
            if recording is self._parent_recording:
                recording_ch_id = self.get_original_channel_ids(channel_ids)
            curr_property_names = recording.get_channel_property_names(channel_id=recording_ch_id)
            for curr_property_name in curr_property_names:
                if curr_property_name not in self._key_properties.keys():  # key property
                    value = recording.get_channel_property(channel_id=recording_ch_id, property_name=curr_property_name)
                    self.set_channel_property(channel_id=channel_ids, property_name=curr_property_name, value=value)
                else:
                    if curr_property_name == 'group':
                        group = recording.get_channel_groups(channel_ids=recording_ch_id)
                        self.set_channel_groups(groups=group, channel_ids=channel_ids)
                    elif curr_property_name == 'location':
                        location = recording.get_channel_locations(channel_ids=recording_ch_id)
                        self.set_channel_locations(locations=location, channel_ids=channel_ids)
        else:
            # copy key properties
            original_channel_ids = self.get_original_channel_ids(channel_ids)
            groups = recording.get_channel_groups(channel_ids=original_channel_ids)
            locations = recording.get_channel_locations(channel_ids=original_channel_ids)
            gains = recording.get_channel_gains(channel_ids=original_channel_ids)
            offsets = recording.get_channel_offsets(channel_ids=original_channel_ids)
            self.set_channel_groups(groups=groups, channel_ids=channel_ids)
            self.set_channel_locations(locations=locations, channel_ids=channel_ids)
            self.set_channel_gains(gains=gains, channel_ids=channel_ids)
            self.set_channel_offsets(offsets=offsets, channel_ids=channel_ids)

            # copy normal properties
            for channel_id in channel_ids:
                recording_ch_id = channel_id
                if recording is self._parent_recording:
                    recording_ch_id = self.get_original_channel_ids(channel_id)
                curr_property_names = recording.get_channel_property_names(channel_id=recording_ch_id)
                for curr_property_name in curr_property_names:
                    if curr_property_name not in self._key_properties.keys():  # key property
                        value = recording.get_channel_property(channel_id=recording_ch_id,
                                                               property_name=curr_property_name)
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
