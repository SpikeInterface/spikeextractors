from spikeextractors.extractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor
from pathlib import Path


class KiloSortRecordingExtractor(PhyRecordingExtractor):
    extractor_name = 'KiloSortRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path):
        PhyRecordingExtractor.__init__(self, folder_path)


class KiloSortSortingExtractor(PhySortingExtractor):
    extractor_name = 'KiloSortSortingExtractor'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # error message when not installed
    is_writable = True
    mode = 'folder'

    def __init__(self, folder_path, exclude_cluster_groups=None, load_waveforms=False, keep_good_only=False,
                 verbose=False):
        PhySortingExtractor.__init__(self, folder_path, exclude_cluster_groups, load_waveforms, verbose)
        self._keep_good_only = keep_good_only
        self._good_units = []

        if keep_good_only:
            for u in self.get_unit_ids():
                if 'KSLabel' in self.get_unit_property_names(u):
                    if self.get_unit_property(u, 'KSLabel') == 'good':
                        self._good_units.append(u)
            self._unit_ids = self._good_units

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'exclude_cluster_groups': exclude_cluster_groups, 'keep_good_only': keep_good_only,
                        'verbose': verbose}
