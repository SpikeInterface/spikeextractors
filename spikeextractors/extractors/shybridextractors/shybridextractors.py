from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor

try:
    import hybridizer.io as sbio
    import hybridizer.probes as sbprb
    HAVE_SBEX = True
except ImportError:
    HAVE_SBEX = False

class SHYBRIDRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_fn):
        RecordingExtractor.__init__(self)

        # load params file related to the given shybrid recording
        self._params = sbio.get_params(recording_fn)['data']

        self._nb_channels = None
        self._channels = None
        self._geom = None

        self._convert_shybrid_probe()

        # translate the byte ordering
        # TODO still ambiguous, shybrid assumes time_axis=1
        byte_order = self._params['order']
        if byte_order == 'C':
            time_axis = 1
        elif byte_order == 'F':
            time_axis = 0

        # piggyback on binary data recording extractor
        self._bindatext = BinDatRecordingExtractor(recording_fn,
                                                   self._params['fs'],
                                                   self._nb_channels,
                                                   self._params['dtype'],
                                                   recording_channels=self._channels,
                                                   time_axis=time_axis,
                                                   geom=self._geom)

    def get_channel_ids(self):
        return self._bindatext.get_channel_ids()

    def get_num_frames(self):
        return self._bindatext.get_num_frames(self)

    def get_sampling_frequency(self):
        return self._bindatext.get_sampling_frequency()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        return self._bindatext.get_traces(channel_ids=channel_ids,
                                          start_frame=start_frame,
                                          end_frame=end_frame)

    def _convert_shybrid_probe(self):
        """ Convert shybrid probe data
        """
        # create a shybrid probe object
        probe = sbprb.Probe(self._params['probe'])

        # extract info from the probe
        self._nb_channels = probe.total_nb_channels
        self._channels = probe.channels.tolist()

        # convert geometry dictionary into simple 2d list
        geom = []
        for channel in self._channels:
            geom.append(probe.geometry[channel])

        self._geom = geom


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

    @staticmethod
    def write_sorting(sorting, save_path):
        '''
        This is an example of a function that is not abstract so it is optional if you want to override it. It allows other
        SortingExtractors to use your new SortingExtractor to convert their sorted data into your
        sorting file format.
        '''

