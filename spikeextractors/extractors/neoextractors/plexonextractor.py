from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False

class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'PlexonRecording'
    mode = 'file'
    assert HAVE_NEO, "To use the Neo extractors, install Neo: \n\n pip install neo\n\n"
    NeoRawIOClass = neo.rawio.PlexonRawIO

class PlexonSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'PlexonSorting'
    mode = 'file'
    assert HAVE_NEO, "To use the Neo extractors, install Neo: \n\n pip install neo\n\n"
    NeoRawIOClass = neo.rawio.PlexonRawIO
