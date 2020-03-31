import numpy as np
import unittest
import tempfile, shutil
import spikeextractors as se
from copy import copy
from pathlib import Path


class TestTools(unittest.TestCase):
    def setUp(self):
        M = 32
        N = 10000
        seed = 0
        sampling_frequency = 30000
        X = np.random.RandomState(seed=seed).normal(0, 1, (M, N))
        self._X = X
        self._sampling_frequency = sampling_frequency
        self.RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_save_probes(self):
        sub_RX = se.load_probe_file(self.RX, 'spikeextractors/tests/probe_test.prb')
        # print(SX.get_channel_property_names())
        assert 'location' in sub_RX.get_shared_channel_property_names()
        assert 'group' in sub_RX.get_shared_channel_property_names()
        positions = [sub_RX.get_channel_property(chan, 'location') for chan in range(self.RX.get_num_channels())]
        # save in csv
        sub_RX.save_to_probe_file(Path(self.test_dir) / 'geom.csv')
        # load csv locations
        sub_RX_load = sub_RX.load_probe_file(Path(self.test_dir) / 'geom.csv')
        position_loaded = [sub_RX_load.get_channel_property(chan, 'location') for
                           chan in range(sub_RX_load.get_num_channels())]
        self.assertTrue(np.allclose(positions[10], position_loaded[10]))

        # prb file
        RX = copy(self.RX)
        channel_groups = []
        n_group = 4
        for i in RX.get_channel_ids():
            channel_groups.append(i // n_group)
        RX.set_channel_groups(channel_groups)
        RX.save_to_probe_file('spikeextractors/tests/probe_test_no_groups.prb')
        RX.save_to_probe_file('spikeextractors/tests/probe_test_groups.prb', grouping_property='group')

        # load
        RX_loaded_no_groups = se.load_probe_file(RX, 'spikeextractors/tests/probe_test_no_groups.prb')
        RX_loaded_groups = se.load_probe_file(RX, 'spikeextractors/tests/probe_test_groups.prb')

        assert len(np.unique(RX_loaded_no_groups.get_channel_groups())) == 1
        assert len(np.unique(RX_loaded_groups.get_channel_groups())) == RX.get_num_channels() // n_group

    def test_write_dat_file(self):
        nb_sample = self.RX.get_num_frames()
        nb_chan = self.RX.get_num_channels()

        # time_axis=0 chunk_size=None
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunk_size=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=1 chunk_size=None
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunk_size=None)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_chan, nb_sample))
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=0 chunk_size=99
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunk_size=99)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del(data) # this close the file

        # time_axis=0 chunk_mb=2
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=0, dtype='float32', chunk_mb=2)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_sample, nb_chan)).T
        assert np.allclose(data, self.RX.get_traces())
        del (data)  # this close the file

        # time_axis=1 chunk_mb=2
        self.RX.write_to_binary_dat_format(self.test_dir + 'rec.dat', time_axis=1, dtype='float32', chunk_mb=2)
        data = np.memmap(open(self.test_dir + 'rec.dat'), dtype='float32', mode='r', shape=(nb_chan, nb_sample))
        assert np.allclose(data, self.RX.get_traces())
        del (data)  # this close the file

if __name__ == '__main__':
    unittest.main()
