from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'NeuralynxRecording'
    mode = 'folder'
    assert HAVE_NEO, "To use the Neo extractors, install Neo: \n\n pip install neo\n\n"
    NeoRawIOClass = neo.rawio.NeuralynxRawIO


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'NeuralynxSorting'
    mode = 'folder'
    assert HAVE_NEO, "To use the Neo extractors, install Neo: \n\n pip install neo\n\n"
    NeoRawIOClass = neo.rawio.NeuralynxRawIO
