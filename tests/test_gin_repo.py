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
        # Klusta sorting - no known sampling frequency
        # (
        #     se.KlustaSortingExtractor,
        #     "kwik",
        #     "kwik/neo.kwik"
        # )
    ])
    def test_convert_sorting_extractor_to_nwb(self, se_class, dataset_path, se_path_arg):
        nwb_save_path = self.savedir / f"{se_class.__name__}_test.nwb"
        self.dataset.get(dataset_path)

        sorting = se_class(Path.cwd() / "ephy_testing_data" / se_path_arg)
        se.NwbRecordingExtractor.write_sorting(sorting, nwb_save_path)
        nwb_sorting = se.NwbRecordingExtractor(nwb_save_path)
        check_sortings_equal(sorting, nwb_sorting)


if __name__ == '__main__':
    unittest.main()
