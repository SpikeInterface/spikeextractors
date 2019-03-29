import numpy as np
import unittest
import tempfile, shutil
import spikeextractors as se
from pathlib import Path


class TestTools(unittest.TestCase):
    def setUp(self):
        M = 32
        N = 10000
        samplerate = 30000
        X = np.random.normal(0, 1, (M, N))
        self._X = X
        self._samplerate = samplerate
        self.RX = se.NumpyRecordingExtractor(timeseries=X, samplerate=samplerate)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_save_probes(self):
        SX = se.load_probe_file(self.RX, 'tests/probe_test.prb')
        # print(SX.get_channel_property_names())
        assert 'location' in SX.get_channel_property_names()
        assert 'group' in SX.get_channel_property_names()
        positions = [SX.get_channel_property(chan, 'location') for chan in range(self.RX.get_num_channels())]
        # save in csv
        se.save_probe_file(SX, Path(self.test_dir) / 'geom.csv')
        # load csv locations
        SX_load = se.load_probe_file(SX, Path(self.test_dir) / 'geom.csv')
        position_loaded = [SX_load.get_channel_property(chan, 'location') for chan in range(SX_load.get_num_channels())]
        self.assertTrue(np.allclose(positions[10], position_loaded[10]))

    def test_write_dat_file(self):
        nb_sample = self.RX.get_num_frames()
        nb_chan = self.RX.get_num_channels()

        # time_axis=0 chunksize=None
        se.write_binary_dat_format(self.RX, self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunksize=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=1 chunksize=None
        se.write_binary_dat_format(self.RX, self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunksize=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_chan, nb_sample))
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=0 chunksize=99
        se.write_binary_dat_format(self.RX, self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunksize=99)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=1 chunksize=99 do not work
        with self.assertRaises(Exception) as context:
            se.write_binary_dat_format(self.RX, self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunksize=99)



if __name__ == '__main__':
    unittest.main()
