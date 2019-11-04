from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo


class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'plexon'
    mode = 'file'
    NeoRawIOClass = neo.rawio.PleonRawIO

class PlexonSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'plexon'
    mode = 'file'
    NeoRawIOClass = neo.rawio.PleonRawIO
    
    def _handle_sampling_frequency(self):
        # in plexon the is only one sampling rate
        # so easy
        self._sampling_frequency = self.neo_reader._global_ssampling_rate
        self._time_start = 0.