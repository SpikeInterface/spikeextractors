from .RecordingExtractor import RecordingExtractor
import numpy as np

# Concatenates the given recordings by channel

class MultiRecordingChannelExtractor(RecordingExtractor):
    def __init__(self, recordings, groups=None):
        RecordingExtractor.__init__(self)
        self._recordings = recordings
        self._all_channel_ids = []
        self._channel_map = {}

        # Sampling frequency based off the initial extractor
        self._first_recording = recordings[0]
        self._sampling_frequency = self._first_recording.get_sampling_frequency()
        self._num_frames =  self._first_recording.get_num_frames()

        # Test if all recording extractors have same sampling frequency
        for i, recording in enumerate(recordings[1:]):
            sampling_frequency = recording.get_sampling_frequency()
            if (self._sampling_frequency != sampling_frequency):
                raise ValueError("Inconsistent sampling frequency between extractor 0 and extractor " + str(i + 1))

        #set channel map for new channel ids to old channel ids
        new_channel_id  = 0
        for r_i, recording in enumerate(self._recordings):
            channel_ids = recording.get_channel_ids()
            for channel_id in channel_ids:
                self._all_channel_ids.append(new_channel_id)
                self._channel_map[new_channel_id] = {'recording': r_i, 'channel_id': channel_id}
                new_channel_id += 1
        
        #set group information for channels if available
        if groups is not None:
            if len(groups) == len(recordings):
                for i, group in enumerate(groups):
                    recording = recordings[i]                    
                    channel_ids = recording.get_channel_ids()
                    recording.set_channel_groups(channel_ids, groups=np.repeat(group,len(channel_ids)))
            else:
                raise ValueError("recordings and groups must have same length")

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        traces = []
        if channel_ids is not None:
            for channel_id in channel_ids:
                recording = self._recordings[self._channel_map[channel_id]['recording']]
                channel_id_recording = self._channel_map[channel_id]['channel_id']
                traces_recording = recording.get_traces(channel_ids=[channel_id_recording], start_frame=start_frame, end_frame=end_frame)
                traces.append(traces_recording)
        else:
            for recording in self._recordings:
                traces_all_recording = recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
                traces.append(traces_all_recording)
        return np.concatenate(traces, axis=0)

    def get_channel_ids(self):
        return self._all_channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def set_channel_property(self, channel_id, property_name, value):
        recording = self._recordings[self._channel_map[channel_id]['recording']]
        channel_id_recording = self._channel_map[channel_id]['channel_id']
        recording.set_channel_property(channel_id_recording, property_name=property_name, value=value)

    def get_channel_property(self, channel_id, property_name):
        recording = self._recordings[self._channel_map[channel_id]['recording']]
        channel_id_recording = self._channel_map[channel_id]['channel_id']
        return recording.get_channel_property(channel_id_recording, property_name=property_name)

    def get_channel_property_names(self, channel_id=None):
        if channel_id is None:
            property_names = []
            for recording in self._recordings:
                property_names_recording = recording.get_channel_property_names()
                for property_name_recording in property_names_recording:
                    property_names.append(property_name_recording)
            property_names = sorted(list(set(property_names)))
        else:
            recording = self._recordings[self._channel_map[channel_id]['recording']]
            channel_id_recording = self._channel_map[channel_id]['channel_id']
            property_names = recording.get_channel_property_names(channel_id_recording)
        return property_names

def concatenate_recordings_by_channel(recordings, groups=None):
    '''
    Concatenates recordings together by channel. The order of the recordings
    determines the order of the channels in the concatenated recording.

    Parameters
    ----------
    recordings: list
        The list of RecordingExtractors to be concatenated by channel.
    groups: list
        A list of ints corresponding to the group identity of each recording's
        channel ids.
    Returns
    -------
    recording: MultiRecordingChannelExtractor
        The concatenated recording extractors enscapsulated in the
        MultiRecordingChannelExtractor object (which is also a recording extractor)
    '''
    return MultiRecordingChannelExtractor(
        recordings=recordings,
        groups=groups,
    )
