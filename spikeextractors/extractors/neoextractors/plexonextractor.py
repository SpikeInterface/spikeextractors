from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo


class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    NeoRawIOClass = neo.rawio.PleonRawIO

class PlexonSortingExtractor(NeoBaseSortingExtractor):
    NeoRawIOClass = neo.rawio.PleonRawIO