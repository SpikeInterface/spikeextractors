import numpy as np
import os, sys
from pathlib import Path
import unittest
import tempfile
import shutil
import spikeextractors as se


class TestExtractors(unittest.TestCase):
    def setUp(self):
        self.RX, self.RX2, self.RX3, self.SX, self.SX2, self.SX3, self.example_info = self._create_example()
        self.test_dir = tempfile.mkdtemp()
        # self.test_dir = '.'

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        # pass

    def _create_example(self):
        channel_ids = [0, 1, 2, 3]
        num_channels = 4
        num_frames = 10000
        sampling_frequency = 30000
        X = np.random.normal(0, 1, (num_channels, num_frames))
        geom = np.random.normal(0, 1, (num_channels, 2))
        X = (X * 100).astype(int)
        RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        RX2 = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        RX3 = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        SX = se.NumpySortingExtractor()
        spike_times = [200, 300, 400]
        train1 = np.sort(np.rint(np.random.uniform(0, num_frames, spike_times[0])).astype(int))
        SX.add_unit(unit_id=1, times=train1)
        SX.add_unit(unit_id=2, times=np.sort(np.random.uniform(0, num_frames, spike_times[1])))
        SX.add_unit(unit_id=3, times=np.sort(np.random.uniform(0, num_frames, spike_times[2])))
        SX.set_unit_property(unit_id=1, property_name='stability', value=80)
        SX.set_sampling_frequency(sampling_frequency)
        SX2 = se.NumpySortingExtractor()
        spike_times2 = [100, 150, 450]
        train2 = np.rint(np.random.uniform(0, num_frames, spike_times2[0])).astype(int)
        SX2.add_unit(unit_id=3, times=train2)
        SX2.add_unit(unit_id=4, times=np.random.uniform(0, num_frames, spike_times2[1]))
        SX2.add_unit(unit_id=5, times=np.random.uniform(0, num_frames, spike_times2[2]))
        SX2.set_unit_property(unit_id=4, property_name='stability', value=80)
        SX2.set_unit_spike_features(unit_id=3, feature_name='widths', value=np.asarray([3] * spike_times2[0]))
        RX.set_channel_property(channel_id=0, property_name='location', value=(0, 0))
        for i, unit_id in enumerate(SX2.get_unit_ids()):
            SX2.set_unit_property(unit_id=unit_id, property_name='shared_unit_prop', value=i)
            SX2.set_unit_spike_features(unit_id=unit_id, feature_name='shared_unit_feature',
                                        value=np.asarray([i] * spike_times2[i]))
        for i, channel_id in enumerate(RX.get_channel_ids()):
            RX.set_channel_property(channel_id=channel_id, property_name='shared_channel_prop', value=i)

        SX3 = se.NumpySortingExtractor()
        train3= np.asarray([1,20,21,35,38,45,46,47])
        SX3.add_unit(unit_id=0, times=train3)
        features3 = np.asarray([0,5,10,15,20,25,30,35])
        SX3.set_unit_spike_features(unit_id=0, feature_name='dummy', value=features3)

        example_info = dict(
            channel_ids=channel_ids,
            num_channels=num_channels,
            num_frames=num_frames,
            sampling_frequency=sampling_frequency,
            unit_ids=[1, 2, 3],
            train1=train1,
            train2=train2,
            train3=train3,
            features3=features3,
            unit_prop=80,
            channel_prop=(0, 0)
        )

        return (RX, RX2, RX3, SX, SX2, SX3, example_info)

    def test_example(self):
        self.assertEqual(self.RX.get_channel_ids(), self.example_info['channel_ids'])
        self.assertEqual(self.RX.get_num_channels(), self.example_info['num_channels'])
        self.assertEqual(self.RX.get_num_frames(), self.example_info['num_frames'])
        self.assertEqual(self.RX.get_sampling_frequency(), self.example_info['sampling_frequency'])
        self.assertEqual(self.SX.get_unit_ids(), self.example_info['unit_ids'])
        self.assertEqual(self.RX.get_channel_property(channel_id=0, property_name='location'),
                         self.example_info['channel_prop'])
        self.assertEqual(self.SX.get_unit_property(unit_id=1, property_name='stability'),
                         self.example_info['unit_prop'])
        self.assertTrue(np.array_equal(self.SX.get_unit_spike_train(1), self.example_info['train1']))
        self.assertTrue(issubclass(self.SX.get_unit_spike_train(1).dtype.type, np.integer))
        self.assertTrue(self.RX.get_shared_channel_property_names(), ['shared_channel_prop'])
        self.assertTrue(self.RX.get_channel_property_names(0), ['location', 'shared_channel_prop'])
        self.assertTrue(self.SX2.get_shared_unit_property_names(), ['shared_unit_prop'])
        self.assertTrue(self.SX2.get_unit_property_names(4), ['shared_unit_prop', 'stability'])
        self.assertTrue(self.SX2.get_shared_unit_spike_feature_names(), ['shared_unit_feature'])
        self.assertTrue(self.SX2.get_unit_spike_feature_names(3), ['shared_channel_prop', 'widths'])

        print(self.SX3.get_unit_spike_features(0, 'dummy'))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy'), self.example_info['features3']))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy', start_frame=4), self.example_info['features3'][1:]))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy', end_frame=4), self.example_info['features3'][:1]))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy', start_frame=20, end_frame=46), self.example_info['features3'][1:6]))

        sub_extractor_full = se.SubSortingExtractor(self.SX3)
        sub_extractor_partial = se.SubSortingExtractor(self.SX3, start_frame=20, end_frame=46)

        self.assertTrue(np.array_equal(sub_extractor_full.get_unit_spike_features(0, 'dummy'), self.SX3.get_unit_spike_features(0, 'dummy')))
        self.assertTrue(np.array_equal(sub_extractor_partial.get_unit_spike_features(0, 'dummy'), self.SX3.get_unit_spike_features(0, 'dummy', start_frame=20, end_frame=46)))

        self._check_recording_return_types(self.RX)

    def test_cache_extractor(self):
        cache_extractor = se.CacheRecordingExtractor(self.RX)
        self._check_recording_return_types(cache_extractor)
        self._check_recordings_equal(self.RX, cache_extractor)
        cache_extractor.save_to_file('cache')

        assert cache_extractor.get_filename() == 'cache.dat'
        del cache_extractor
        assert not Path('cache.dat').is_file()

    def test_mda_extractor(self):
        path1 = self.test_dir + '/mda'
        path2 = path1 + '/firings_true.mda'
        se.MdaRecordingExtractor.write_recording(self.RX, path1)
        se.MdaSortingExtractor.write_sorting(self.SX, path2)
        RX_mda = se.MdaRecordingExtractor(path1)
        SX_mda = se.MdaSortingExtractor(path2)
        self._check_recording_return_types(RX_mda)
        self._check_recordings_equal(self.RX, RX_mda)
        self._check_sorting_return_types(SX_mda)
        self._check_sortings_equal(self.SX, SX_mda)

    def _check_recording_return_types(self, RX):
        channel_ids = RX.get_channel_ids()
        M = RX.get_num_channels()
        N = RX.get_num_frames()
        self.assertTrue((type(RX.get_num_channels()) == int) or (type(RX.get_num_channels()) == np.int64))
        self.assertTrue((type(RX.get_num_frames()) == int) or (type(RX.get_num_frames()) == np.int64))
        self.assertTrue(
            (type(RX.get_sampling_frequency()) == float) or (type(RX.get_sampling_frequency()) == np.float64))
        self.assertTrue(type(RX.get_traces(start_frame=0, end_frame=10)) == np.ndarray)
        for channel_id in channel_ids:
            self.assertTrue((type(channel_id) == int) or (type(channel_id) == np.int64))

    def test_biocam_extractor(self):
        path1 = self.test_dir + '/raw.brw'
        se.BiocamRecordingExtractor.write_recording(self.RX, path1)
        RX_biocam = se.BiocamRecordingExtractor(path1)
        self._check_recording_return_types(RX_biocam)
        self._check_recordings_equal(self.RX, RX_biocam)

    def test_mearec_extractors(self):
        path1 = self.test_dir + '/raw.h5'
        se.MEArecRecordingExtractor.write_recording(self.RX, path1)
        RX_mearec = se.MEArecRecordingExtractor(path1)
        tr = RX_mearec.get_traces(channel_ids=[0, 1], end_frame=1000)
        self._check_recording_return_types(RX_mearec)
        self._check_recordings_equal(self.RX, RX_mearec)

        path2 = self.test_dir + '/firings_true.h5'
        se.MEArecSortingExtractor.write_sorting(self.SX, path2, self.RX.get_sampling_frequency())
        SX_mearec = se.MEArecSortingExtractor(path2)
        self._check_sorting_return_types(SX_mearec)
        self._check_sortings_equal(self.SX, SX_mearec)

    def test_hs2_extractor(self):
        path1 = self.test_dir + '/firings_true.hdf5'
        se.HS2SortingExtractor.write_sorting(self.SX, path1)
        SX_hs2 = se.HS2SortingExtractor(path1)
        self._check_sorting_return_types(SX_hs2)
        self._check_sortings_equal(self.SX, SX_hs2)
        self.assertEqual(SX_hs2.get_sampling_frequency(), self.SX.get_sampling_frequency())

    def test_exdir_extractors(self):
        path1 = self.test_dir + '/raw.exdir'
        se.ExdirRecordingExtractor.write_recording(self.RX, path1)
        RX_exdir = se.ExdirRecordingExtractor(path1)
        self._check_recording_return_types(RX_exdir)
        self._check_recordings_equal(self.RX, RX_exdir)

        path2 = self.test_dir + '/firings.exdir'
        se.ExdirSortingExtractor.write_sorting(self.SX, path2, self.RX)
        SX_exdir = se.ExdirSortingExtractor(path2)
        self._check_sorting_return_types(SX_exdir)
        self._check_sortings_equal(self.SX, SX_exdir)

    def test_kilosort_extractor(self):
        path1 = self.test_dir + '/ks'
        se.KiloSortSortingExtractor.write_sorting(self.SX, path1)
        SX_ks = se.KiloSortSortingExtractor(path1)
        self._check_sorting_return_types(SX_ks)
        self._check_sortings_equal(self.SX, SX_ks)

    def test_klusta_extractor(self):
        path1 = self.test_dir + '/firings_true.kwik'
        se.KlustaSortingExtractor.write_sorting(self.SX, path1)
        SX_kl = se.KlustaSortingExtractor(path1)
        self._check_sorting_return_types(SX_kl)
        self._check_sortings_equal(self.SX, SX_kl)

    def test_spykingcircus_extractor(self):
        path1 = self.test_dir + '/sc'
        se.SpykingCircusSortingExtractor.write_sorting(self.SX, path1)
        SX_spy = se.SpykingCircusSortingExtractor(path1)
        self._check_sorting_return_types(SX_spy)
        self._check_sortings_equal(self.SX, SX_spy)

    def test_multi_sub_recording_extractor(self):
        RX_multi = se.MultiRecordingTimeExtractor(
            recordings=[self.RX, self.RX, self.RX],
            epoch_names=['A', 'B', 'C']
        )
        RX_sub = RX_multi.get_epoch('C')
        self._check_recordings_equal(self.RX, RX_sub)
        self.assertEqual(4, len(RX_sub.get_channel_ids()))

        RX_multi = se.MultiRecordingChannelExtractor(
            recordings=[self.RX, self.RX2, self.RX3],
            groups=[1, 2, 3]
        )
        print(RX_multi.get_channel_groups())
        RX_sub = se.SubRecordingExtractor(RX_multi, channel_ids=[4, 5, 6, 7], renamed_channel_ids=[0, 1, 2, 3])
        self._check_recordings_equal(self.RX2, RX_sub)
        self.assertEqual([2, 2, 2, 2], RX_sub.get_channel_groups())
        self.assertEqual(12, len(RX_multi.get_channel_ids()))

    def test_multi_sub_sorting_extractor(self):
        N = self.RX.get_num_frames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX, self.SX],
        )
        SX_multi.set_unit_property(unit_id=1, property_name='dummy', value=5)
        SX_sub = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0)
        self._check_sortings_equal(SX_multi, SX_sub)
        self.assertEqual(SX_multi.get_unit_property(1, 'dummy'), SX_sub.get_unit_property(1, 'dummy'))

        N = self.RX.get_num_frames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX2],
        )
        SX_sub1 = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0, end_frame=N)
        self._check_sortings_equal(SX_multi, SX_sub1)

    def test_nwb_extractor(self):
        path1 = self.test_dir + '/test.nwb'
        se.NwbRecordingExtractor.write_recording(self.RX, path1)
        RX_nwb = se.NwbRecordingExtractor(path1)
        self._check_recording_return_types(RX_nwb)
        self._check_recordings_equal(self.RX, RX_nwb)
        del RX_nwb
        # overwrite
        se.NwbRecordingExtractor.write_recording(self.RX, path1, session_description='second',
                                                 identifier='19475')
        RX_nwb = se.NwbRecordingExtractor(path1)
        self._check_recording_return_types(RX_nwb)
        self._check_recordings_equal(self.RX, RX_nwb)
        # add sorting to existing
        se.NwbSortingExtractor.write_sorting(self.SX, path1)
        # create new
        path2 = self.test_dir + '/firings_true.nwb'
        se.NwbSortingExtractor.write_sorting(self.SX, path2, session_description='second',
                                             identifier='19475')
        SX_nwb = se.NwbSortingExtractor(path1)
        self._check_sortings_equal(self.SX, SX_nwb)

    def test_nixio_extractor(self):
        path1 = os.path.join(self.test_dir, 'raw.nix')
        se.NIXIORecordingExtractor.write_recording(self.RX, path1)
        RX_nixio = se.NIXIORecordingExtractor(path1)
        self._check_recording_return_types(RX_nixio)
        self._check_recordings_equal(self.RX, RX_nixio)
        del RX_nixio
        # test force overwrite
        se.NIXIORecordingExtractor.write_recording(self.RX, path1,
                                                   overwrite=True)

        path2 = self.test_dir + '/firings_true.nix'
        se.NIXIOSortingExtractor.write_sorting(self.SX, path2)
        SX_nixio = se.NIXIOSortingExtractor(path2)
        self._check_sorting_return_types(SX_nixio)
        self._check_sortings_equal(self.SX, SX_nixio)

    # def test_shybrid_extractors(self):
    #     # test sorting extractor
    #     se.SHYBRIDSortingExtractor.write_sorting(self.SX, self.test_dir)
    #     initial_sorting_file = os.path.join(self.test_dir, 'initial_sorting.csv')
    #     SX_shybrid = se.SHYBRIDSortingExtractor(initial_sorting_file)
    #     self._check_sorting_return_types(SX_shybrid)
    #     self._check_sortings_equal(self.SX, SX_shybrid)
    # 
    #     # test recording extractor
    #     se.SHYBRIDRecordingExtractor.write_recording(self.RX,
    #                                                  self.test_dir,
    #                                                  initial_sorting_file)
    #     RX_shybrid = se.SHYBRIDRecordingExtractor(os.path.join(self.test_dir,
    #                                                            'recording.bin'))
    #     self._check_recording_return_types(RX_shybrid)
    #     self._check_recordings_equal(self.RX, RX_shybrid)

    def _check_recordings_equal(self, RX1, RX2):
        M = RX1.get_num_channels()
        N = RX1.get_num_frames()
        # get_channel_ids
        self.assertEqual(RX1.get_channel_ids(), RX2.get_channel_ids())
        # get_num_channels
        self.assertEqual(RX1.get_num_channels(), RX2.get_num_channels())
        # get_num_frames
        self.assertEqual(RX1.get_num_frames(), RX2.get_num_frames())
        # get_sampling_frequency
        self.assertEqual(RX1.get_sampling_frequency(), RX2.get_sampling_frequency())
        # get_traces
        self.assertTrue(np.allclose(
            RX1.get_traces(),
            RX2.get_traces()
        ))
        sf = 0
        ef = N
        ch = [0, M - 1]
        self.assertTrue(np.allclose(
            RX1.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef),
            RX2.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef)
        ))
        for f in range(0, RX1.get_num_frames(), 10):
            self.assertTrue(np.isclose(RX1.frame_to_time(f), RX2.frame_to_time(f)))
            self.assertTrue(
                np.isclose(RX1.time_to_frame(RX1.frame_to_time(f)), RX2.time_to_frame(RX2.frame_to_time(f))))
        # get_snippets
        frames = [30, 50, 80]
        snippets1 = RX1.get_snippets(reference_frames=frames, snippet_len=20)
        snippets2 = RX2.get_snippets(reference_frames=frames, snippet_len=(10, 10))
        for ii in range(len(frames)):
            self.assertTrue(np.allclose(snippets1[ii], snippets2[ii]))

    def _check_sorting_return_types(self, SX):
        unit_ids = SX.get_unit_ids()
        self.assertTrue(all(isinstance(id, int) or isinstance(id, np.integer) for id in unit_ids))
        for id in unit_ids:
            train = SX.get_unit_spike_train(id)
            # print(train)
            self.assertTrue(all(isinstance(x, int) or isinstance(x, np.integer) for x in train))

    def _check_sortings_equal(self, SX1, SX2):
        K = len(SX1.get_unit_ids())
        # get_unit_ids
        ids1 = np.sort(np.array(SX1.get_unit_ids()))
        ids2 = np.sort(np.array(SX2.get_unit_ids()))
        self.assertTrue(np.allclose(ids1, ids2))
        for id in ids1:
            train1 = np.sort(SX1.get_unit_spike_train(id))
            train2 = np.sort(SX2.get_unit_spike_train(id))
            # print(train1)
            # print(train2)
            self.assertTrue(np.array_equal(train1, train2))


if __name__ == '__main__':
    unittest.main()
