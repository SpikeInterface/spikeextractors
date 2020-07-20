from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False
 

class mcsrawRecordingExtractor(NeoBaseRecordingExtractor):
  extractor_name='mcsrawRecoding'
  mode='file'
  NeoRawIOClass='RawMCSRawIO'


