from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False

class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    """
    The plxon extractor is wrapped from neo.rawio.PlexonRawIO.
    
    Parameters
    ----------
    filename: str
        The plexon file ('plx')
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.
    
    """    
    extractor_name = 'PlexonRecording'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'PlexonRawIO'

class PlexonSortingExtractor(NeoBaseSortingExtractor):
    extractor_name = 'PlexonSorting'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'PlexonRawIO'
