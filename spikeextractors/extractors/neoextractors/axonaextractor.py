from .neobaseextractor import NeoBaseRecordingExtractor

try:
    import neo

    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    extractor_name = 'AxonaRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'
