from spikeextractors.extractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor


class KiloSortRecordingExtractor(PhyRecordingExtractor):
    extractor_name = 'KiloSortRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'kilosort_folder', 'type': 'path', 'title': "str, Path to folder"},        
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, kilosort_folder):
        PhyRecordingExtractor.__init__(self, kilosort_folder)


class KiloSortSortingExtractor(PhySortingExtractor):

    extractor_name = 'KiloSortSortingExtractor'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # error message when not installed

    def __init__(self, kilosort_folder):
        PhySortingExtractor.__init__(self, kilosort_folder)
