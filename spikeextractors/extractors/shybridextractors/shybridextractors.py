import os

import numpy as np

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
    extractor_name = 'SHYBRIDRecordingExtractor'
    installed = HAVE_SBEX
    extractor_gui_params = [
        {'name': 'recording_fn', 'type': 'file', 'title': "Full path to hybrid recording (.bin)"},
    ]

    def __init__(self, recording_fn):
        RecordingExtractor.__init__(self)

        # load params file related to the given shybrid recording
        self._params = sbio.get_params(recording_fn)['data']

        self._nb_channels = None
        self._channels = None
        self._geom = None

        self._convert_shybrid_probe()

        # translate the byte ordering
        # TODO still ambiguous, shybrid should assume time_axis=1, since spike interface makes an assumption on the byte ordering
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

        self._geom = np.array(geom)


class SHYBRIDSortingExtractor(SortingExtractor):
    extractor_name = 'SHYBRIDSortingExtractor'
    installed = HAVE_SBEX
    extractor_gui_params = [
        {'name': 'gt_fn', 'type': 'file', 'title': "Full path to hybrid ground truth labels (.csv)"},
    ]
    is_writable = False

    def __init__(self, gt_fn, delimiter=','):
        SortingExtractor.__init__(self)

        if os.path.isfile(gt_fn):
            self._spike_clusters = sbio.SpikeClusters()
            self._spike_clusters.fromCSV(gt_fn, None, delimiter=delimiter)
        else:
            raise FileNotFoundError('the ground truth file "{}" could not be found'.format(gt_fn))

    def get_unit_ids(self):
        return self._spike_clusters.keys()

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        train = self._spike_clusters[unit_id].get_actual_spike_train().spikes

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf

        idxs = np.where((start_frame <= train) & (train < end_frame))
        return train[idxs]
