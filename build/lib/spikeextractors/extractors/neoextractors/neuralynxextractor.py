from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    """
    The neruralynx extractor is wrapped from neo.rawio.NeuralynxRawIO.
    
    Parameters
    ----------
    dirname: str
        The neuralynx folder that contain all neuralynx files ('nse', 'ncs', 'nev', 'ntt')
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.
    
    """

    extractor_name = 'NeuralynxRecording'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'NeuralynxSorting'
    mode = 'folder'
    installed = HAVE_NEO
    NeoRawIOClass = 'NeuralynxRawIO'
