from spikeextractors.extractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor


class KiloSortRecordingExtractor(PhyRecordingExtractor):
    extractor_name = 'KiloSortRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    extractor_gui_params = [
        {'name': 'folder_path', 'type': 'folder', 'title': "str, Path to folder"},        
    ]
    installation_mesg = ""  # error message when not installed
    
    def __init__(self, folder_path):
        PhyRecordingExtractor.__init__(self, folder_path)


class KiloSortSortingExtractor(PhySortingExtractor):

    extractor_name = 'KiloSortSortingExtractor'
    exporter_name = 'KiloSortSortingExporter'
    exporter_gui_params = [
        {'name': 'save_path', 'type': 'folder', 'title': "Save path"},
    ]
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # error message when not installed
    is_writable = True
    mode = 'folder'

    def __init__(self, folder_path, exclude_cluster_groups=None, load_waveforms=False, verbose=False):
        PhySortingExtractor.__init__(self, folder_path, exclude_cluster_groups, load_waveforms, verbose)
