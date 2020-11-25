from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class BlackrockRecordingExtractor(NeoBaseRecordingExtractor):
    """
    The Blackrock extractor is wrapped from neo.rawio.BlackrockRawIO.
    
    Parameters
    ----------
    filename: str
        The Blackrock file (.ns1, .ns2, .ns3, .ns4m .ns4, or .ns6)
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.
    
    """
    extractor_name = 'BlackrockRecording'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'BlackrockRawIO'


class BlackrockSortingExtractor(NeoBaseSortingExtractor):
    """
    The Blackrock extractor is wrapped from neo.rawio.BlackrockRawIO.

    Parameters
    ----------
    filename: str
        The Blackrock file (.nev)
    block_index: None or int
        If the underlying dataset have several blocks the index must be specified.
    seg_index_index: None or int
        If the underlying dataset have several segments the index must be specified.

    """
    extractor_name = 'BlackrockSorting'
    mode = 'file'
    installed = HAVE_NEO
    NeoRawIOClass = 'BlackrockRawIO'
