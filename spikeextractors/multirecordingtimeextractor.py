from .recordingextractor import RecordingExtractor
from .extraction_tools import check_get_traces_args, check_get_ttl_args
import numpy as np


# Concatenates the given recordings by time
class MultiRecordingTimeExtractor(RecordingExtractor):
    def __init__(self, recordings, epoch_names=None):
        self._recordings = recordings

        # Num channels and sampling frequency based off the initial extractor
        self._first_recording = recordings[0]
        self._num_channels = self._first_recording.get_num_channels()
        self._channel_ids = self._first_recording.get_channel_ids()
        self._sampling_frequency = self._first_recording.get_sampling_frequency()

        if epoch_names is None:
            epoch_names = [str(i) for i in range(len(recordings))]

        RecordingExtractor.__init__(self)

        # Add all epochs to the epochs data structure
        start_frames = 0
        for i, epoch_name in enumerate(epoch_names):
            num_frames = recordings[i].get_num_frames()
            self.add_epoch(epoch_name, start_frames, start_frames + num_frames)
            start_frames += num_frames

        # Test if all recording extractors have same num channels and sampling frequency
        for i, recording in enumerate(recordings[1:]):
            channel_ids = recording.get_channel_ids()
            sampling_frequency = recording.get_sampling_frequency()

            if self._channel_ids != channel_ids:
                raise ValueError("Inconsistent channel ids between extractor 0 and extractor " + str(i + 1))
            if self._sampling_frequency != sampling_frequency:
                raise ValueError("Inconsistent sampling frequency between extractor 0 and extractor " + str(i + 1))

        self._start_frames = []
        self._end_frames = []
        self._start_times = []
        self._end_times = []
        ff = 0
        tt = 0.
        for recording in self._recordings:
            self._start_frames.append(ff)
            self._start_times.append(tt)
            ff = ff + recording.get_num_frames()
            tt = tt + recording.frame_to_time(recording.get_num_frames() - 1) - recording.frame_to_time(0)
            self._end_frames.append(ff)
            self._end_times.append(tt)
        self._num_frames = ff

        # Set the channel properties based on the first recording extractor
        self.copy_channel_properties(self._first_recording)

        # avoid rescaling twice
        self.clear_channel_gains()
        self.clear_channel_offsets()

        self.is_filtered = self._first_recording.is_filtered
        self.has_unscaled = self._first_recording.has_unscaled

        self._kwargs = {'recordings': [rec.make_serialized_dict() for rec in recordings], 'epoch_names': epoch_names}

    @property
    def recordings(self):
        return self._recordings

    def _find_section_for_frame(self, frame):
        start_frames = np.array(self._start_frames)
        end_frames = np.array(self._end_frames)
        inds = np.where((frame >= start_frames) & (frame < end_frames))[0]
        if len(inds) == 0:
            # can only happen if frame == end_frame
            ind = len(self._start_frames) - 1
        else:
            ind = inds[0]
        return self._recordings[ind], ind, frame - self._start_frames[ind]

    def _find_section_for_time(self, time):
        start_times = np.array(self._start_times)
        end_times = np.array(self._end_times)
        inds = np.where((time >= start_times) & (time < end_times))[0]
        if len(inds) == 0:
            # can only happen if frame == end_frame
            ind = len(self._start_times) - 1
        else:
            ind = inds[0]
        return self._recordings[ind], ind, time - self._start_times[ind]

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        recording1, i_sec1, i_start_frame = self._find_section_for_frame(start_frame)
        _, i_sec2, i_end_frame = self._find_section_for_frame(end_frame)
        if i_sec1 == i_sec2:
            return recording1.get_traces(channel_ids=channel_ids, start_frame=i_start_frame, end_frame=i_end_frame,
                                         return_scaled=return_scaled)
        traces = []
        traces.append(
            self._recordings[i_sec1].get_traces(channel_ids=channel_ids, start_frame=i_start_frame,
                                                end_frame=self._recordings[i_sec1].get_num_frames(),
                                                return_scaled=return_scaled)
        )
        for i_sec in range(i_sec1 + 1, i_sec2):
            traces.append(
                self._recordings[i_sec].get_traces(channel_ids=channel_ids, return_scaled=return_scaled)
            )
        if i_end_frame != 0:
            traces.append(
                self._recordings[i_sec2].get_traces(channel_ids=channel_ids, start_frame=0, end_frame=i_end_frame,
                                                    return_scaled=return_scaled)
            )
        return np.concatenate(traces, axis=1)

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        recording1, i_sec1, i_start_frame = self._find_section_for_frame(start_frame)
        _, i_sec2, i_end_frame = self._find_section_for_frame(end_frame)

        if i_sec1 == i_sec2:
            ttl_frames, ttl_states = recording1.get_ttl_events(start_frame=i_start_frame,
                                                               end_frame=i_end_frame,
                                                               channel_id=channel_id)
            ttl_frames += self._start_frames[i_sec1]
        else:
            ttl_frames, ttl_states = [], []

            ttl_frames_1, ttl_states_1 = self._recordings[i_sec1].get_ttl_events(
                start_frame=i_start_frame,
                end_frame=self._recordings[i_sec1].get_num_frames(),
                channel_id=channel_id)
            ttl_frames_1 = (ttl_frames_1 + self._start_frames[i_sec1]).astype('int64')
            ttl_frames.append(ttl_frames_1)
            ttl_states.append(ttl_states_1)

            for i_sec in range(i_sec1 + 1, i_sec2):
                ttl_frames_i, ttl_states_i = self._recordings[i_sec].get_ttl_events(channel_id=channel_id)
                ttl_frames_i = (ttl_frames_i + self._start_frames[i_sec]).astype('int64')
                ttl_frames.append(ttl_frames_i)
                ttl_states.append(ttl_states_i)

            ttl_frames_2, ttl_states_2 = self._recordings[i_sec2].get_ttl_events(start_frame=0,
                                                                                 end_frame=i_end_frame,
                                                                                 channel_id=channel_id)
            ttl_frames_2 = (ttl_frames_2 + self._start_frames[i_sec2]).astype('int64')
            ttl_frames.append(ttl_frames_2)
            ttl_states.append(ttl_states_2)

            ttl_frames = np.concatenate(np.array(ttl_frames))
            ttl_states = np.concatenate(np.array(ttl_states))

        return ttl_frames, ttl_states

    def get_channel_ids(self):
        return self._channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def frame_to_time(self, frame):
        recording, i_epoch, rel_frame = self._find_section_for_frame(frame)
        return np.round(recording.frame_to_time(rel_frame) + self._start_times[i_epoch], 6)

    def time_to_frame(self, time):
        recording, i_epoch, rel_time = self._find_section_for_time(time)
        return (recording.time_to_frame(rel_time) + self._start_frames[i_epoch]).astype('int64')


def concatenate_recordings_by_time(recordings, epoch_names=None):
    """
    Concatenates recordings together by time. The order of the recordings
    determines the order of the time series in the concatenated recording.

    Parameters
    ----------
    recordings: list
        The list of RecordingExtractors to be concatenated by time
    epoch_names: list
        The list of strings corresponding to the names of recording time period.

    Returns
    -------
    recording: MultiRecordingTimeExtractor
        The concatenated recording extractors enscapsulated in the
        MultiRecordingTimeExtractor object (which is also a recording extractor)
    """
    return MultiRecordingTimeExtractor(
        recordings=recordings,
        epoch_names=epoch_names,
    )
