import os
from pathlib import Path
import numpy as np

from spikeextractors import SortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors.extraction_tools import save_to_probe_file, load_probe_file

try:
    import hybridizer.io as sbio
    import hybridizer.probes as sbprb
    HAVE_SBEX = True
except ImportError:
    HAVE_SBEX = False


class SHYBRIDRecordingExtractor(BinDatRecordingExtractor):
    extractor_name = 'SHYBRIDRecording'
    installed = HAVE_SBEX
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the SHYBRID extractors, install SHYBRID: \n\n pip install shybrid\n\n"

    def __init__(self, file_path):
        # load params file related to the given shybrid recording
        assert HAVE_SBEX, "To use the SHYBRID extractors, install shybrid: \n\n pip install shybrid\n\n"
        params = sbio.get_params(file_path)['data']

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
        BinDatRecordingExtractor.__init__(self,
                                          file_path,
                                          params['fs'],
                                          nb_channels,
                                          params['dtype'],
                                          time_axis=time_axis)
        self._kwargs = {'file_path': str(Path(file_path).absolute())}
        self = load_probe_file(self, params['probe'])


    @staticmethod
    def write_recording(recording, save_path, initial_sorting_fn, dtype='float32'):
        """ Convert and save the recording extractor to SHYBRID format

        parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved
        save_path: str
            Full path to desired target folder
        initial_sorting_fn: str
            Full path to the initial sorting csv file (can also be generated
            using write_sorting static method from the SHYBRIDSortingExtractor)
        dtype: dtype
            Type of the saved data. Default float32.
        """
        assert HAVE_SBEX, "To use the SHYBRID extractors, install shybrid: \n\n pip install shybrid\n\n"
        RECORDING_NAME = 'recording.bin'
        PROBE_NAME = 'probe.prb'
        PARAMETERS_NAME = 'recording.yml'

        # location information has to be present in order for shybrid to
        # be able to operate on the recording
        if 'location' not in recording.get_shared_channel_property_names():
            raise GeometryNotLoadedError("Channel locations were not found")

        # write recording
        recording_fn = os.path.join(save_path, RECORDING_NAME)
        BinDatRecordingExtractor.write_recording(recording, recording_fn,
                                                 time_axis=0, dtype=dtype)

        # write probe file
        probe_fn = os.path.join(save_path, PROBE_NAME)
        save_to_probe_file(recording, probe_fn)

        # create parameters file
        parameters = params_template.format(initial_sorting_fn=initial_sorting_fn,
                                            data_type=dtype,
                                            sampling_frequency=str(recording.get_sampling_frequency()),
                                            byte_ordering='F',
                                            probe_fn=probe_fn)

        # write parameters file
        parameters_fn = os.path.join(save_path, PARAMETERS_NAME)
        with open(parameters_fn, 'w') as fp:
            fp.write(parameters)


class SHYBRIDSortingExtractor(SortingExtractor):
    extractor_name = 'SHYBRIDSortingExtractor'
    installed = HAVE_SBEX
    is_writable = True
    installation_mesg = "To use the SHYBRID extractors, install SHYBRID: \n\n pip install shybrid\n\n"

    def __init__(self, file_path, delimiter=','):
        assert HAVE_SBEX, "To use the SHYBRID extractors, install shybrid: \n\n pip install shybrid\n\n"
        SortingExtractor.__init__(self)

        if os.path.isfile(file_path):
            self._spike_clusters = sbio.SpikeClusters()
            self._spike_clusters.fromCSV(file_path, None, delimiter=delimiter)
        else:
            raise FileNotFoundError('the ground truth file "{}" could not be found'.format(file_path))
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'delimiter': delimiter}

    def get_unit_ids(self):
        return self._spike_clusters.keys()

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        train = self._spike_clusters[unit_id].get_actual_spike_train().spikes

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf

        idxs = np.where((start_frame <= train) & (train < end_frame))
        return train[idxs]

    @staticmethod
    def write_sorting(sorting, save_path):
        """ Convert and save the sorting extractor to SHYBRID CSV format

        parameters
        ----------
        sorting : SortingExtractor
            The sorting extractor to be converted and saved
        save_path : str
            Full path to the desired target folder
        """
        assert HAVE_SBEX, "To use the SHYBRID extractors, install shybrid: \n\n pip install shybrid\n\n"
        dump = np.empty((0,2))

        for unit_id in sorting.get_unit_ids():
            spikes = sorting.get_unit_spike_train(unit_id)[:,np.newaxis]
            expanded_id = (np.ones(spikes.size) * unit_id)[:,np.newaxis]
            tmp_concat = np.concatenate((expanded_id, spikes), axis=1)

            dump = np.concatenate((dump, tmp_concat), axis=0)

        sorting_fn = os.path.join(save_path, 'initial_sorting.csv')
        np.savetxt(sorting_fn, dump, delimiter=',', fmt='%i')


class GeometryNotLoadedError(Exception):
    """ Raised when the recording extractor has no associated channel locations
    """
    pass

params_template = \
"""clusters:
  csv: {initial_sorting_fn}
data:
  dtype: {data_type}
  fs: {sampling_frequency}
  order: {byte_ordering}
  probe: {probe_fn}
"""
