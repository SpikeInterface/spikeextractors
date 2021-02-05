import tempfile
import unittest
from pathlib import Path

from datalad.api import install, Dataset
from parameterized import parameterized

import spikeextractors as se
from spikeextractors.testing import check_recordings_equal, check_sortings_equal


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        pt = Path.cwd() / 'ephy_testing_data'
        if pt.exists():
            self.dataset = Dataset(pt)
        else:
            self.dataset = install('https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')
        self.savedir = Path(tempfile.mkdtemp())

    @parameterized.expand([
        # Intan - strptime issue in pyintan
        # (
        #     se.IntanRecordingExtractor,
        #     "intan",
        #     "intan/intan_rhd_test_1.rhd"
        # ),
        # (
        #     se.IntanRecordingExtractor,
        #     "intan",
        #     "intan/intan_rhd_test_1.rhs"
        # )
        # Neuralynx - fix input arguments
        # (
        #     se.NeuralynxRecordingExtractor,
        #     'neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        #     'neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        #     'neuralynx_test.nwb',
        #     'neuralynx_test.Ncs'
        # )
        (
            se.NeuroscopeRecordingExtractor,
            "neuroscope/test1",
            "neuroscope/test1/test1.dat"
        ),
    ])
    def test_convert_recording_extractor_to_nwb(self, se_class, dataset_path, se_path_arg):
        nwb_save_path = self.savedir / f"{se_class.__name__}_test.nwb"
        self.dataset.get(dataset_path)

        recording = se_class(Path.cwd() / "ephy_testing_data" / se_path_arg)
        se.NwbRecordingExtractor.write_recording(recording, nwb_save_path)
        nwb_recording = se.NwbRecordingExtractor(nwb_save_path)
        check_recordings_equal(recording, nwb_recording)

    @parameterized.expand([
        # Klusta sorting - needs sampling frequency set before write
        (
            se.KlustaSortingExtractor,
            "kwik",
            "kwik/neo.kwik"
        ),
        # NIXIO - PosixPath has no attribute encode in nixio/pycore/file
#        (
#            se.NIXIOSortingExtractor,
#            "nix/nixio_fr.nix",
#            "nix/nixio_fr.nix"
#        )
        # Phy - Running into unit_id non-int typing issues in spikeextractors or between spikeextractors and pynwb
#        (
#            se.PhySortingExtractor,
#            "phy/phy_example_0",
#            "phy/phy_example_0"
#        )
        # Plexon - no documentation on use of Neo inheritor, doesn't work out of box like other extractors
#        (
#            se.PlexonSortingExtractor,
#            "plexon",
#            "plexon/File_plexon_1.plx"
#        ),
        # SpykingCircus - read/write is passing but re-loaded sortings are not equal
#        (
#            se.SpykingCircusSortingExtractor,
#            "spykingcircus/spykingcircus_example0",
#            "spykingcircus/spykingcircus_example0/recording"
#        )
        # Tridesclous - version issues with PyQt5 backend
#        (
#            se.TridesclousSortingExtractor,
#            "tridesclous/tdc_example0",
#            "tridesclous/tdc_example0"
#        )
    ])
    def test_convert_sorting_extractor_to_nwb(self, se_class, dataset_path, se_path_arg):
        nwb_save_path = self.savedir / f"{se_class.__name__}_test.nwb"
        self.dataset.get(dataset_path)

        sorting = se_class(Path.cwd() / "ephy_testing_data" / se_path_arg)
        sf = sorting.get_sampling_frequency()
        if sf is None:
            sorting.set_sampling_frequency(1)
        se.NwbSortingExtractor.write_sorting(sorting, nwb_save_path)
        nwb_sorting = se.NwbSortingExtractor(nwb_save_path, sampling_frequency=1)  # dummy sampling frequency b/c no associated acquisition
        check_sortings_equal(sorting, nwb_sorting)


if __name__ == '__main__':
    unittest.main()
