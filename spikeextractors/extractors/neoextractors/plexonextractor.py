from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo


class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'plexon'
    mode = 'file'
    NeoRawIOClass = neo.rawio.PlexonRawIO

class PlexonSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'plexon'
    mode = 'file'
    NeoRawIOClass = neo.rawio.PlexonRawIO
    
