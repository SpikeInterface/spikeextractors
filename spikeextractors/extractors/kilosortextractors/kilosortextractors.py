from spikeextractors.extractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor


class KiloSortRecordingExtractor(PhyRecordingExtractor):
    extractor_name = 'KiloSortRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'dir'
    _gui_params = [
        {'name': 'dir_path', 'type': 'dir', 'title': "str, Path to directory"},        
    ]
    installation_mesg = ""  # error message when not installed
    
    def __init__(self, dir_path):
        PhyRecordingExtractor.__init__(self, dir_path)


class KiloSortSortingExtractor(PhySortingExtractor):

    extractor_name = 'KiloSortSortingExtractor'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # error message when not installed
    is_writable = True
    mode = 'dir'

    def __init__(self, dir_path, exclude_cluster_groups=None, load_waveforms=False, verbose=False):
        PhySortingExtractor.__init__(self, dir_path, exclude_cluster_groups, load_waveforms, verbose)
