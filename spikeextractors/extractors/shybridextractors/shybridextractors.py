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
        params = sbio.get_params(recording_fn)['data']

        # create a shybrid probe object
        probe = sbprb.Probe(params['probe'])
        nb_channels = probe.total_nb_channels

        # translate the byte ordering
        # TODO still ambiguous, shybrid should assume time_axis=1, since spike interface makes an assumption on the byte ordering
        byte_order = params['order']
        if byte_order == 'C':
            time_axis = 1
        elif byte_order == 'F':
            time_axis = 0

        # piggyback on binary data recording extractor
        bindatext = BinDatRecordingExtractor(recording_fn,
                                             params['fs'],
                                             nb_channels,
                                             params['dtype'],
                                             time_axis=time_axis)

        # attach probe file to binary extractor
        # TODO why doesn't the load probe file writes the properties of its object?
        self._extractor_w_prb = bindatext.load_probe_file(params['probe'])
        # copy properties from sub extractor to current object
        self.copy_channel_properties(self._extractor_w_prb)

    def get_channel_ids(self):
        return self._extractor_w_prb.get_channel_ids()

    def get_num_frames(self):
        return self._extractor_w_prb.get_num_frames()

    def get_sampling_frequency(self):
        return self._extractor_w_prb.get_sampling_frequency()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        return self._extractor_w_prb.get_traces(channel_ids=channel_ids,
                                          start_frame=start_frame,
                                          end_frame=end_frame)


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
