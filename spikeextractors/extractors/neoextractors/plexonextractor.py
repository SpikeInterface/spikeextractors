from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False

class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'PlexonRecording'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'PlexonRawIO'

class PlexonSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'PlexonSorting'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'PlexonRawIO'
