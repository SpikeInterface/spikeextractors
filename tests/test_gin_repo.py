import tempfile
import unittest
from pathlib import Path

from datalad.api import install, Dataset
from parameterized import parameterized

from spikeextractors import NwbRecordingExtractor, NeuralynxRecordingExtractor, NeuroscopeRecordingExtractor


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        pt = Path.cwd()/'ephy_testing_data'
        if pt.exists():
            self.dataset = Dataset(pt)
        else:
            self.dataset = install('https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')
        self.savedir = Path(tempfile.mkdtemp())

    def get_data(self, rt_write_fname, rt_read_fname, save_fname, dataset_path):
        if rt_read_fname is None:
            rt_read_fname = rt_write_fname
        save_path = self.savedir/save_fname
        rt_write_path = self.savedir/rt_write_fname
        rt_read_path = self.savedir/rt_read_fname
        resp = self.dataset.get(dataset_path)

        return rt_write_path, rt_read_path, save_path

    @parameterized.expand([
        #(
        #    NeuralynxRecordingExtractor,
        #    'neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        #    'neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        #    'neuralynx_test.nwb',
        #    'neuralynx_test.Ncs'
        #)
        (
            NeuroscopeRecordingExtractor,
            "neuroscope/test1",
            "neuroscope/test1/test1.dat",
            "neuroscope_test.nwb",
            "neuroscope_test.dat"
        )
    ])
    def test_convert_recording_extractor_to_nwb(
        self, se_class, dataset_path, dataset_path_arg, save_fname, rt_write_fname, rt_read_fname=None
    ):

        rt_write_path, rt_read_path, save_path = self.get_data(rt_write_fname, rt_read_fname, save_fname, dataset_path)

        path = Path.cwd()/'ephy_testing_data'/dataset_path_arg
        re = se_class(path)
        NwbRecordingExtractor.write_recording(re, save_path)
        # nwb_seg_ex = NwbSegmentationExtractor(save_path)
        # check_segmentations_equal(roi_ex, nwb_seg_ex)
        # try:
        #     roi_ex_class.write_segmentation(nwb_seg_ex, rt_write_path)
        # except NotImplementedError:
        #     return
        # seg_ex_rt = roi_ex_class(rt_read_path)

if __name__ == '__main__':
    unittest.main()
