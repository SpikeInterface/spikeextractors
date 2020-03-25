import numpy as np
import unittest
import spikeextractors as se


class TestNumpyExtractors(unittest.TestCase):
    def setUp(self):
        M = 4
        N = 10000
        seed= 0
        sampling_frequency = 30000
        X = np.random.RandomState(seed=seed).normal(0, 1, (M, N))
        geom = np.random.RandomState(seed=seed).normal(0, 1, (M, 2))
        self._X = X
        self._geom = geom
        self._sampling_frequency = sampling_frequency
        self.RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        self.SX = se.NumpySortingExtractor()
        L = 200
        self._train1 = np.rint(np.random.RandomState(seed=seed).uniform(0, N, L)).astype(int)
        self.SX.add_unit(unit_id=1, times=self._train1)
        self.SX.add_unit(unit_id=2, times=np.random.RandomState(seed=seed).uniform(0, N, L))
        self.SX.add_unit(unit_id=3, times=np.random.RandomState(seed=seed).uniform(0, N, L))

    def tearDown(self):
        pass

    def test_recording_extractor(self):
        # get_channel_ids
        self.assertEqual(self.RX.get_channel_ids(), [i for i in range(self._X.shape[0])])
        # get_num_channels
        self.assertEqual(self.RX.get_num_channels(), self._X.shape[0])
        # get_num_frames
        self.assertEqual(self.RX.get_num_frames(), self._X.shape[1])
        # get_sampling_frequency
        self.assertEqual(self.RX.get_sampling_frequency(), self._sampling_frequency)
        # get_traces
        self.assertTrue(np.allclose(self.RX.get_traces(), self._X))
        self.assertTrue(
            np.allclose(self.RX.get_traces(channel_ids=[0, 3], start_frame=0, end_frame=12), self._X[[0, 3], 0:12]))
        # get_channel_property - location
        self.assertTrue(np.allclose(np.array(self.RX.get_channel_property(1, 'location')), self._geom[1, :]))
        # time_to_frame / frame_to_time
        self.assertEqual(self.RX.time_to_frame(12), 12 * self.RX.get_sampling_frequency())
        self.assertEqual(self.RX.frame_to_time(12), 12 / self.RX.get_sampling_frequency())
        # get_snippets
        snippets = self.RX.get_snippets(reference_frames=[0, 30, 50], snippet_len=20)
        self.assertTrue(np.allclose(snippets[1], self._X[:, 20:40]))

    def test_sorting_extractor(self):
        unit_ids = [1, 2, 3]
        # get_unit_ids
        self.assertEqual(self.SX.get_unit_ids(), unit_ids)
        # get_unit_spike_train
        st = self.SX.get_unit_spike_train(unit_id=1)
        self.assertTrue(np.allclose(st, self._train1))


if __name__ == '__main__':
    unittest.main()
