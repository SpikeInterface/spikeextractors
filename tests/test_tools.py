import numpy as np
import unittest
import tempfile, shutil
import spikeextractors as se
from pathlib import Path


class TestTools(unittest.TestCase):
    def setUp(self):
        M = 32
        N = 10000
        sampling_frequency = 30000
        X = np.random.normal(0, 1, (M, N))
        self._X = X
        self._sampling_frequency = sampling_frequency
        self.RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_save_probes(self):
        sub_RX = se.load_probe_file(self.RX, 'tests/probe_test.prb')
        # print(SX.get_channel_property_names())
        assert 'location' in sub_RX.get_shared_channel_property_names()
        assert 'group' in sub_RX.get_shared_channel_property_names()
        positions = [sub_RX.get_channel_property(chan, 'location') for chan in range(self.RX.get_num_channels())]
        # save in csv
        sub_RX.save_to_probe_file(Path(self.test_dir) / 'geom.csv')
        # load csv locations
        sub_RX_load = sub_RX.load_probe_file(Path(self.test_dir) / 'geom.csv')
        position_loaded = [sub_RX_load.get_channel_property(chan, 'location') for chan in range(sub_RX_load.get_num_channels())]
        self.assertTrue(np.allclose(positions[10], position_loaded[10]))

    def test_write_dat_file(self):
        nb_sample = self.RX.get_num_frames()
        nb_chan = self.RX.get_num_channels()

        # time_axis=0 chunksize=None
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunksize=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=1 chunksize=None
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunksize=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_chan, nb_sample))
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=0 chunksize=99
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunksize=99)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=1 chunksize=99 do not work
        with self.assertRaises(Exception) as context:
            self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunksize=99)


if __name__ == '__main__':
    unittest.main()
