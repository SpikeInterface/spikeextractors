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

    def __init__(self, **kargs):
        NeoBaseRecordingExtractor.__init__(self, **kargs)
        self._key_properties['group'] = self.neo_reader.\
            raw_annotations['blocks'][0]['segments'][0]['signals']\
                           [0]['__array_annotations__']['tetrode_id']
