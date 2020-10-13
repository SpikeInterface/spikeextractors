from .recordingextractor import RecordingExtractor
from .extraction_tools import check_get_traces_args
import numpy as np


# Concatenates the given recordings by time
class MultiRecordingTimeExtractor(RecordingExtractor):
    def __init__(self, recordings, epoch_names=None):
        self._recordings = recordings

        # Num channels and sampling frequency based off the initial extractor
        self._first_recording = recordings[0]
        self._num_channels = self._first_recording.get_num_channels()
        self._channel_ids = self._first_recording.get_channel_ids()
        self._channel_ids_dict = {}

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
        reset_channels = False
        for i, recording in enumerate(recordings[1:]):
            channel_ids = recording.get_channel_ids()
            sampling_frequency = recording.get_sampling_frequency()

            assert len(self._channel_ids) == len(channel_ids), f"Inconsistent number of channels between " \
                                                               f"extractor 0 and extractor {i + 1}"
            assert self._sampling_frequency == sampling_frequency, f"Inconsistent sampling frequency between " \
                                                                   f"extractor 0 and extractor {i + 1}"
            assert np.array_equal(self._first_recording.get_channel_locations(), recording.get_channel_locations()), \
                f"Inconsistent locations between extractor 0 and extractor {i + 1}"
            if self._channel_ids != channel_ids:
                reset_channels = True

        if reset_channels:
            print("The recordings have different channel ids. Resetting channel ids.")
            self._channel_ids = list(np.arange(len(self._channel_ids)))

            for i, rec in enumerate(recordings):
                self._channel_ids_dict[i] = rec.get_channel_ids()

        self._start_frames = []
        self._start_times = []
        ff = 0
        tt = 0
        for recording in self._recordings:
            self._start_frames.append(ff)
            ff = ff + recording.get_num_frames()
            tt = tt + recording.frame_to_time(0)
            self._start_times.append(tt)
            tt = tt + recording.frame_to_time(recording.get_num_frames()) - recording.frame_to_time(0)
        self._start_frames.append(ff)
        self._start_times.append(tt)
        self._num_frames = ff

        # Set the channel properties based on the first recording extractor
        if len(self._channel_ids_dict) == 0:
            self.copy_channel_properties(self._first_recording)
        else:
            # only copy key properties
            self.set_channel_locations(self._first_recording.get_channel_locations())
            self.set_channel_groups(self._first_recording.get_channel_groups())
        self._kwargs = {'recordings': [rec.make_serialized_dict() for rec in recordings], 'epoch_names': epoch_names}

    @property
    def recordings(self):
        return self._recordings

    def _find_section_for_frame(self, frame):
        inds = np.where(np.array(self._start_frames[:-1]) <= frame)[0]
        ind = inds[-1]
        return self._recordings[ind], ind, frame - self._start_frames[ind]

    def _find_section_for_time(self, time):
        inds = np.where(np.array(self._start_times[:-1]) <= time)[0]
        ind = inds[-1]
        return self._recordings[ind], ind, time - self._start_times[ind]

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        recording1, i_sec1, i_start_frame = self._find_section_for_frame(start_frame)
        _, i_sec2, i_end_frame = self._find_section_for_frame(end_frame)
        if i_sec1 == i_sec2:
            if len(self._channel_ids_dict) == 0:
                return recording1.get_traces(channel_ids=channel_ids, start_frame=i_start_frame, end_frame=i_end_frame)
            else:
                channel_ids_idxs = [self._channel_ids.index(ch) for ch in channel_ids]
                channel_ids_1 = list(np.array(self._channel_ids_dict[i_sec1])[channel_ids_idxs])
                return recording1.get_traces(channel_ids=channel_ids_1, start_frame=i_start_frame,
                                             end_frame=i_end_frame)

        traces = []
        if len(self._channel_ids_dict) == 0:
            traces.append(self._recordings[i_sec1].get_traces(channel_ids=channel_ids, start_frame=i_start_frame,
                                                              end_frame=self._recordings[i_sec1].get_num_frames()))
            for i_sec in range(i_sec1 + 1, i_sec2):
                traces.append(self._recordings[i_sec].get_traces(channel_ids=channel_ids))
            traces.append(self._recordings[i_sec2].get_traces(channel_ids=channel_ids, start_frame=0,
                                                              end_frame=i_end_frame))
        else:
            channel_ids_idxs = [self._channel_ids.index(ch) for ch in channel_ids]
            channel_ids_1 = list(np.array(self._channel_ids_dict[i_sec1])[channel_ids_idxs])
            traces.append(self._recordings[i_sec1].get_traces(channel_ids=channel_ids_1, start_frame=i_start_frame,
                                                              end_frame=self._recordings[i_sec1].get_num_frames()))
            for i_sec in range(i_sec1 + 1, i_sec2):
                channel_ids_i = list(np.array(self._channel_ids_dict[i_sec])[channel_ids_idxs])
                traces.append(self._recordings[i_sec].get_traces(channel_ids=channel_ids_i))
            channel_ids_2 = list(np.array(self._channel_ids_dict[i_sec2])[channel_ids_idxs])
            traces.append(self._recordings[i_sec2].get_traces(channel_ids=channel_ids_2, start_frame=0,
                                                              end_frame=i_end_frame))

        return np.concatenate(traces, axis=1)

    def get_channel_ids(self):
        return self._channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def frame_to_time(self, frame):
        recording, i_epoch, rel_frame = self._find_section_for_frame(frame)
        return recording.frame_to_time(rel_frame) + self._start_times[i_epoch]

    def time_to_frame(self, time):
        recording, i_epoch, rel_time = self._find_section_for_time(time)
        return recording.time_to_frame(rel_time) + self._start_frames[i_epoch]


def concatenate_recordings_by_time(recordings, epoch_names=None):
    '''
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
    '''
    return MultiRecordingTimeExtractor(
        recordings=recordings,
        epoch_names=epoch_names,
    )
