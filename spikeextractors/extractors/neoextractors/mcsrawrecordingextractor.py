from .neobaseextractor import NeoBaseRecordingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False
 

class MCSRawRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name='mcsrawRecoding'
    mode='file'
    NeoRawIOClass='RawMCSRawIO'
