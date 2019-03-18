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
        SX = se.loadProbeFile(self.RX, 'tests/probe_test.prb')
        # print(SX.getChannelPropertyNames())
        assert 'location' in SX.getChannelPropertyNames()
        assert 'group' in SX.getChannelPropertyNames()
        positions = [SX.getChannelProperty(chan, 'location') for chan in range(self.RX.getNumChannels())]
        # save in csv
        se.saveProbeFile(SX, Path(self.test_dir) / 'geom.csv')
        # load csv locations
        SX_load = se.loadProbeFile(SX, Path(self.test_dir) / 'geom.csv')
        position_loaded = [SX_load.getChannelProperty(chan, 'location') for chan in range(SX_load.getNumChannels())]
        self.assertTrue(np.allclose(positions[10], position_loaded[10]))

    def test_write_dat_file(self):
        nb_sample = self.RX.getNumFrames()
        nb_chan = self.RX.getNumChannels()
        
        # time_axis=0 chunksize=None
        se.writeBinaryDatFormat(self.RX, self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunksize=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.getTraces())
        del(data) # this close the file

        # time_axis=1 chunksize=None
        se.writeBinaryDatFormat(self.RX, self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunksize=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_chan, nb_sample))
        assert np.allclose(data, self.RX.getTraces())
        del(data) # this close the file

        # time_axis=0 chunksize=99
        se.writeBinaryDatFormat(self.RX, self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunksize=99)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.getTraces())
        del(data) # this close the file

        # time_axis=1 chunksize=99 do not work
        with self.assertRaises(Exception) as context:
            se.writeBinaryDatFormat(self.RX, self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunksize=99)



if __name__ == '__main__':
    unittest.main()
