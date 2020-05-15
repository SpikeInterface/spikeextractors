from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'NeuralynxRecording'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'NeuralynxSorting'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'
