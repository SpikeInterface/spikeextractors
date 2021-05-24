from .recordingextractor import RecordingExtractor
from .extraction_tools import check_get_traces_args
import numpy as np
import warnings


# Concatenates the given recordings by channel
class MultiRecordingChannelExtractor(RecordingExtractor):
    def __init__(self, recordings, groups=None):
        self._recordings = recordings
        self._all_channel_ids = []
        self._channel_map = {}

        # Sampling frequency based off the initial extractor
        self._first_recording = recordings[0]
        self._sampling_frequency = self._first_recording.get_sampling_frequency()
        self._num_frames = self._first_recording.get_num_frames()

        use_times = True
        if np.all([rec._times is not None for rec in self._recordings]):
            times_0 = self._recordings[0]._times
            for rec in self._recordings[1:]:
                times_i = rec._times
                if not np.allclose(times_0, times_i):
                    use_times = False
                    warnings.warn("The recordings have different times! Reset times with the "
                                  "'set_times() function")
        elif np.all([rec._times is not None for rec in self._recordings]):
            warnings.warn("Not all the recordings have times! Reset times with the "
                          "'set_times() function")
        else:
            use_times = False

        # Test if all recording extractors have same sampling frequency
        for i, recording in enumerate(recordings[1:]):
            sampling_frequency = recording.get_sampling_frequency()
            if self._sampling_frequency != sampling_frequency:
                raise ValueError("Inconsistent sampling frequency between extractor 0 and extractor " + str(i + 1))

        # set channel map for new channel ids to old channel ids
        new_channel_id = 0
        for r_i, recording in enumerate(self._recordings):
            channel_ids = recording.get_channel_ids()
            for channel_id in channel_ids:
                self._all_channel_ids.append(new_channel_id)
                self._channel_map[new_channel_id] = {'recording': r_i, 'channel_id': channel_id}
                new_channel_id += 1

        RecordingExtractor.__init__(self)

        if use_times:
            self.copy_times(self._recordings[0])

        # set group information for channels if available
        if groups is not None:
            if len(groups) == len(recordings):
                group_values = []
                for i, group in enumerate(groups):
                    recording = recordings[i]
                    channel_ids = recording.get_channel_ids()
                    recording_groups = [group] * len(channel_ids)
                    group_values += recording_groups

                self.set_channel_groups(groups=group_values)
            else:
                raise ValueError("recordings and groups must have same length")

        # set channel locations
        locations = np.empty([0, 2])
        for i, recording in enumerate(recordings):
            locations = np.vstack((locations, recording.get_channel_locations()))
        self.set_channel_locations(locations)

        #set all normal properties
        for channel_id in self.get_channel_ids():
            recording = self._recordings[self._channel_map[channel_id]['recording']]
            channel_id_recording = self._channel_map[channel_id]['channel_id']
            for property_name in recording.get_channel_property_names(channel_id_recording):
                if property_name not in ("group", "location"):
                    value = recording.get_channel_property(channel_id_recording, property_name)
                    self.set_channel_property(channel_id=channel_id, property_name=property_name, value=value)

        # avoid rescaling twice
        self.clear_channel_gains()
        self.clear_channel_offsets()

        self.is_filtered = self._first_recording.is_filtered
        self.has_unscaled = self._first_recording.has_unscaled

        self._kwargs = {'recordings': [rec.make_serialized_dict() for rec in recordings], 'groups': groups}

    @property
    def recordings(self):
        return self._recordings

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        traces = []
        if channel_ids is not None:
            for channel_id in channel_ids:
                recording = self._recordings[self._channel_map[channel_id]['recording']]
                channel_id_recording = self._channel_map[channel_id]['channel_id']
                traces_recording = recording.get_traces(channel_ids=[channel_id_recording], start_frame=start_frame,
                                                        end_frame=end_frame, return_scaled=return_scaled)
                traces.append(traces_recording)
        else:
            for recording in self._recordings:
                traces_all_recording = recording.get_traces(channel_ids=channel_ids, start_frame=start_frame,
                                                            end_frame=end_frame, return_scaled=return_scaled)
                traces.append(traces_all_recording)
        return np.concatenate(traces, axis=0)

    def get_channel_ids(self):
        return self._all_channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency


def concatenate_recordings_by_channel(recordings, groups=None):
    """
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
    """
    return MultiRecordingChannelExtractor(
        recordings=recordings,
        groups=groups,
    )
