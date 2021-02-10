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
        # Blackrock - PosixPath has no attribute endswith stemming from base neo io
        (
            se.BlackrockRecordingExtractor,
            "blackrock/blackrock_2_1",
            dict(filename=Path.cwd() / "ephy_testing_data" / "I101210-001.ns2")
        ),
        # Intan - strptime issue in pyintan
        (
            se.IntanRecordingExtractor,
            "intan",
            "intan/intan_rhd_test_1.rhd"
        ),
        (
            se.IntanRecordingExtractor,
            "intan",
            "intan/intan_rhd_test_1.rhs"
        )
        (
            se.MEArecRecordingExtractor,
            "mearec/mearec_test_10s.h5",
            dict(file_path=Path.cwd() / "ephy_testing_data" / "mearec/mearec_test_10s.h5")
        ),
        # Neuralynx - a lot of versions on the testing_data, not sure which we want to support;
        # main issue is setting seg_index for many of the blocked files
        (
            se.NeuralynxRecordingExtractor,
            "neuralynx/Cheetah_v5.7.4/original_data",
            dict(dirname=Path.cwd() / "ephy_testing_data" / "neuralynx/Cheetah_v5.7.4/original_data"),
        ),
        (
            se.NeuroscopeRecordingExtractor,
            "neuroscope/test1",
            dict(file_path=Path.cwd() / "ephy_testing_data" / "neuroscope/test1/test1.dat")
        ),
        # Nixio - PosixPath has no attribute encode
        (
            se.NIXIORecordingExtractor,
            "nix",
            dict(file_path=Path.cwd() / "ephy_testing_data" / "neoraw.nix")
        )
        (
            se.OpenEphysRecordingExtractor,
            "openephys/OpenEphys_SampleData_1",
            dict(folder_path=Path.cwd() / "ephy_testing_data" / "openephys/OpenEphys_SampleData_1")
        ),
        (
            se.SpikeGLXRecordingExtractor,
            "spikeglx/Noise4Sam_g0",
            dict(
                file_path=Path.cwd() / "ephy_testing_data" / "spikeglx" / "Noise4Sam_g0" / "Noise4Sam_g0_imec0" / "Noise4Sam_g0_t0.imec0.ap.bin"
            )
        )
    ])
    def test_convert_recording_extractor_to_nwb(self, se_class, dataset_path, se_kwargs):
        nwb_save_path = self.savedir / f"{se_class.__name__}_test.nwb"
        self.dataset.get(dataset_path)

        recording = se_class(**se_kwargs)
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
        (
            se.NIXIOSortingExtractor,
            "nix/nixio_fr.nix",
            "nix/nixio_fr.nix"
        )
        # Phy - Running into unit_id non-int typing issues in spikeextractors or between spikeextractors and pynwb
        (
            se.PhySortingExtractor,
            "phy/phy_example_0",
            "phy/phy_example_0"
        )
        # Plexon - no documentation on use of Neo inheritor, doesn't work out of box like other extractors
        (
            se.PlexonSortingExtractor,
            "plexon",
            "plexon/File_plexon_1.plx"
        ),
        # SpykingCircus - read/write is passing but re-loaded sortings are not equal
        (
            se.SpykingCircusSortingExtractor,
            "spykingcircus/spykingcircus_example0",
            "spykingcircus/spykingcircus_example0/recording"
        )
        # Tridesclous - version issues with PyQt5 backend
        (
            se.TridesclousSortingExtractor,
            "tridesclous/tdc_example0",
            "tridesclous/tdc_example0"
        )
    ])
    def test_convert_sorting_extractor_to_nwb(self, se_class, dataset_path, se_path_arg):
        nwb_save_path = self.savedir / f"{se_class.__name__}_test.nwb"
        self.dataset.get(dataset_path)

        sorting = se_class(Path.cwd() / "ephy_testing_data" / se_path_arg)
        sf = sorting.get_sampling_frequency()
        if sf is None:
            sorting.set_sampling_frequency(1)
        se.NwbSortingExtractor.write_sorting(sorting, nwb_save_path)
        # dummy sampling frequency b/c no associated acquisition
        nwb_sorting = se.NwbSortingExtractor(nwb_save_path, sampling_frequency=1)
        check_sortings_equal(sorting, nwb_sorting)


if __name__ == '__main__':
    unittest.main()
