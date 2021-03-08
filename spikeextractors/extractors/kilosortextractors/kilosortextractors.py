from spikeextractors.extractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor
from pathlib import Path


class KiloSortRecordingExtractor(PhyRecordingExtractor):
    extractor_name = 'KiloSortRecording'
    has_default_locations = True
    has_unscaled = False
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path):
        PhyRecordingExtractor.__init__(self, folder_path)


class KiloSortSortingExtractor(PhySortingExtractor):
    extractor_name = 'KiloSortSorting'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # error message when not installed
    is_writable = False
    mode = 'folder'

    def __init__(self, folder_path, exclude_cluster_groups=None, keep_good_only=False):
        PhySortingExtractor.__init__(self, folder_path, exclude_cluster_groups)
        self._keep_good_only = keep_good_only
        self._good_units = []

        if keep_good_only:
            for u in self.get_unit_ids():
                if 'KSLabel' in self.get_unit_property_names(u):
                    if self.get_unit_property(u, 'KSLabel') == 'good':
                        self._good_units.append(u)
            self._unit_ids = self._good_units

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'exclude_cluster_groups': exclude_cluster_groups, 'keep_good_only': keep_good_only}
