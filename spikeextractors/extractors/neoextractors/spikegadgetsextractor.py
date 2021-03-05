from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False

class SpikeGadgetsRecordingExtractor(NeoBaseRecordingExtractor):
    """
    The spikegadgets extractor is wrapped from neo.rawio.SpikegadgetsRawIO.
    
    Parameters
    ----------
    filename: str
        The spike gadgets file ('rec')
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.
    """    
    extractor_name = 'SpikeGadgetsRecording'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'SpikegadgetsRawIO'
