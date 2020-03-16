from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'NeuralynxRecording'
    mode = 'folder'
    NeoRawIOClass = neo.rawio.NeuralynxRawIO

class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'NeuralynxSorting'
    mode = 'folder'
    NeoRawIOClass = neo.rawio.NeuralynxRawIO
