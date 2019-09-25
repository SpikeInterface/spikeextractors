from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor


class SHYBRIDRecordingExtractor(RecordingExtractor):
    def __init__(self, ex_parameter_1, ex_parameter_2):
        RecordingExtractor.__init__(self)

        ## All file specific initialization code can go here.

    def get_channel_ids(self):

        # Fill code to get a list of channel_ids. If channel ids are not specified, you can use:
        # channel_ids = range(num_channels)

        return channel_ids

    def get_num_frames(self):

        # Fill code to get the number of frames (samples) in the recordings.

        return num_frames

    def get_sampling_frequency(self, unit_id, start_frame=None, end_frame=None):

        # Fill code to get the sampling frequency of the recordings.

        return sampling_frequency

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        '''This function extracts and returns a trace from the recorded data from the
        given channels ids and the given start and end frame. It will return
        traces from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_recording_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_recording_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Traces are returned in a 2D array that
        contains all of the traces from each channel with dimensions
        (num_channels x num_frames). In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        start_frame: int
            The starting frame of the trace to be returned (inclusive).
        end_frame: int
            The ending frame of the trace to be returned (exclusive).
        channel_ids: array_like
            A list or 1D array of channel ids (ints) from which each trace will be
            extracted.

        Returns
        ----------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (num_channels x num_frames)
        '''

        # Fill code to get the the traces of the specified channel_ids, from start_frame to end_frame

        return traces

    .
    .
    .
    .
    . #Optional functions and pre-implemented functions that a new RecordingExtractor doesn't need to implement
    .
    .
    .
    .

    @staticmethod
    def write_recording(recording, save_path, other_params):
        '''
        This is an example of a function that is not abstract so it is optional if you want to override it.
        It allows other RecordingExtractor to use your new RecordingExtractor to convert their recorded data into
        your recording file format.
        '''


class SHYBRIDSortingExtractor(SortingExtractor):
    def __init__(self, ex_parameter_1, ex_parameter_2):
        SortingExtractor.__init__(self)

        ## All file specific initialization code can go here.
        # If your format stores the sampling frequency, you can overweite the self._sampling_frequency. This way,
        # the base method self.get_sampling_frequency() will return the correct sampling frequency

        self._sampling_frequency = my_sampling_frequency

    def get_unit_ids(self):

        #Fill code to get a unit_ids list containing all the ids (ints) of detected units in the recording

        return unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

        '''Code to extract spike frames from the specified unit.
        It will return spike frames from within three ranges:
            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]
        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike frames are returned in the form of an
        array_like of spike frames. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        '''

        return spike_train

    .
    .
    .
    .
    . #Optional functions and pre-implemented functions that a new SortingExtractor doesn't need to implement
    .
    .
    .
    .

    @staticmethod
    def write_sorting(sorting, save_path):
        '''
        This is an example of a function that is not abstract so it is optional if you want to override it. It allows other
        SortingExtractors to use your new SortingExtractor to convert their sorted data into your
        sorting file format.
        '''

