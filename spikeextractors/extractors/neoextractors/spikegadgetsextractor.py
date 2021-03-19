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
    selected_streams: str
        The id of the stream to load 'trodes' is ephy channels.
        Can also be ECU, ...
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.
    """    
    extractor_name = 'SpikeGadgetsRecording'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'SpikeGadgetsRawIO'
    def __init__(self, filename, selected_streams='trodes',**kwargs):
        super().__init__(filename=filename, selected_streams=selected_streams, **kwargs)
