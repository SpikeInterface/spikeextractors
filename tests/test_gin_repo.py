import unittest

from datalad.api import install

from spikeextractors import NwbRecordingExtractor, NeuralynxRecordingExtractor


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        self.dataset = install('https://gin.g-node.org/NeuralEnsemble/ephy_testing_data')

    def test_convert_neuralynx(self):
        resp = self.dataset.get('neuralynx/Cheetah_v1.1.0/orginial_data/CSC67_trunc.Ncs')
        path = resp[0]['path']
        re = NeuralynxRecordingExtractor(path)

        NwbRecordingExtractor.write_recording(re, 'nlx_test.nwb')


if __name__ == '__main__':
    unittest.main()
