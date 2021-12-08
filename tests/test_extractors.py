import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import spikeextractors as se
from spikeextractors.exceptions import NotDumpableExtractorError
from spikeextractors.testing import (check_sortings_equal, check_recordings_equal, check_dumping,
    check_recording_return_types, check_sorting_return_types, get_default_nwbfile_metadata)


class TestExtractors(unittest.TestCase):
    def setUp(self):
        self.RX, self.RX2, self.RX3, self.SX, self.SX2, self.SX3, self.example_info = self._create_example(seed=0)
        self.test_dir = tempfile.mkdtemp()
        # self.test_dir = '.'

    def tearDown(self):
        # Remove the directory after the test
        del self.RX, self.RX2, self.RX3, self.SX, self.SX2, self.SX3
        shutil.rmtree(self.test_dir)
        # pass

    def _create_example(self, seed):
        channel_ids = [0, 1, 2, 3]
        num_channels = 4
        num_frames = 10000
        num_ttls = 30
        sampling_frequency = 30000
        X = np.random.RandomState(seed=seed).normal(0, 1, (num_channels, num_frames))
        geom = np.random.RandomState(seed=seed).normal(0, 1, (num_channels, 2))
        X = (X * 100).astype(int)
        ttls = np.sort(np.random.permutation(num_frames)[:num_ttls])

        RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        RX.set_ttls(ttls)
        RX.set_channel_locations([0, 0], channel_ids=0)
        RX.add_epoch("epoch1", 0, 10)
        RX.add_epoch("epoch2", 10, 20)
        for i, channel_id in enumerate(RX.get_channel_ids()):
            RX.set_channel_property(channel_id=channel_id, property_name='shared_channel_prop', value=i)

        RX2 = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        RX2.copy_epochs(RX)
        times = np.arange(RX.get_num_frames()) / RX.get_sampling_frequency() + 5
        RX2.set_times(times)

        RX3 = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)

        SX = se.NumpySortingExtractor()
        SX.set_sampling_frequency(sampling_frequency)
        spike_times = [200, 300, 400]
        train1 = np.sort(np.rint(np.random.RandomState(seed=seed).uniform(0, num_frames, spike_times[0])).astype(int))
        SX.add_unit(unit_id=1, times=train1)
        SX.add_unit(unit_id=2, times=np.sort(np.random.RandomState(seed=seed).uniform(0, num_frames, spike_times[1])))
        SX.add_unit(unit_id=3, times=np.sort(np.random.RandomState(seed=seed).uniform(0, num_frames, spike_times[2])))
        SX.set_unit_property(unit_id=1, property_name='stability', value=80)
        SX.add_epoch("epoch1", 0, 10)
        SX.add_epoch("epoch2", 10, 20)

        SX2 = se.NumpySortingExtractor()
        SX2.set_sampling_frequency(sampling_frequency)
        spike_times2 = [100, 150, 450]
        train2 = np.rint(np.random.RandomState(seed=seed).uniform(0, num_frames, spike_times2[0])).astype(int)
        SX2.add_unit(unit_id=3, times=train2)
        SX2.add_unit(unit_id=4, times=np.random.RandomState(seed=seed).uniform(0, num_frames, spike_times2[1]))
        SX2.add_unit(unit_id=5, times=np.random.RandomState(seed=seed).uniform(0, num_frames, spike_times2[2]))
        SX2.set_unit_property(unit_id=4, property_name='stability', value=80)
        SX2.set_unit_spike_features(unit_id=3, feature_name='widths', value=np.asarray([3] * spike_times2[0]))
        SX2.copy_epochs(SX)
        SX2.copy_times(RX2)
        for i, unit_id in enumerate(SX2.get_unit_ids()):
            SX2.set_unit_property(unit_id=unit_id, property_name='shared_unit_prop', value=i)
            SX2.set_unit_spike_features(
                unit_id=unit_id,
                feature_name='shared_unit_feature',
                value=np.asarray([i] * spike_times2[i])
            )

        SX3 = se.NumpySortingExtractor()
        train3 = np.asarray([1, 20, 21, 35, 38, 45, 46, 47])
        SX3.add_unit(unit_id=0, times=train3)
        features3 = np.asarray([0, 5, 10, 15, 20, 25, 30, 35])
        features4 = np.asarray([0, 10, 20, 30])
        feature4_idx = np.asarray([0, 2, 4, 6])
        SX3.set_unit_spike_features(unit_id=0, feature_name='dummy', value=features3)
        SX3.set_unit_spike_features(unit_id=0, feature_name='dummy2', value=features4, indexes=feature4_idx)

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
            channel_prop=(0, 0),
            ttls=ttls,
            epochs_info=((0, 10), (10, 20)),
            geom=geom,
            times=times
        )

        return (RX, RX2, RX3, SX, SX2, SX3, example_info)

    def test_example(self):
        self.assertEqual(self.RX.get_channel_ids(), self.example_info['channel_ids'])
        self.assertEqual(self.RX.get_num_channels(), self.example_info['num_channels'])
        self.assertEqual(self.RX.get_num_frames(), self.example_info['num_frames'])
        self.assertEqual(self.RX.get_sampling_frequency(), self.example_info['sampling_frequency'])
        self.assertEqual(self.SX.get_unit_ids(), self.example_info['unit_ids'])
        self.assertEqual(self.RX.get_channel_locations(0)[0][0], self.example_info['channel_prop'][0])
        self.assertEqual(self.RX.get_channel_locations(0)[0][1], self.example_info['channel_prop'][1])
        self.assertTrue(np.array_equal(self.RX.get_ttl_events()[0], self.example_info['ttls']))
        self.assertEqual(self.SX.get_unit_property(unit_id=1, property_name='stability'),
                         self.example_info['unit_prop'])
        self.assertTrue(np.array_equal(self.SX.get_unit_spike_train(1), self.example_info['train1']))

        self.assertTrue(issubclass(self.SX.get_unit_spike_train(1).dtype.type, np.integer))
        self.assertTrue(self.RX.get_shared_channel_property_names(), ['group', 'location', 'shared_channel_prop'])
        self.assertTrue(self.RX.get_channel_property_names(0), ['group', 'location', 'shared_channel_prop'])
        self.assertTrue(self.SX2.get_shared_unit_property_names(), ['shared_unit_prop'])
        self.assertTrue(self.SX2.get_unit_property_names(4), ['shared_unit_prop', 'stability'])
        self.assertTrue(self.SX2.get_shared_unit_spike_feature_names(), ['shared_unit_feature'])
        self.assertTrue(self.SX2.get_unit_spike_feature_names(3), ['shared_channel_prop', 'widths'])

        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy'), self.example_info['features3']))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy', start_frame=4),
                                       self.example_info['features3'][1:]))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy', end_frame=4),
                                       self.example_info['features3'][:1]))
        self.assertTrue(np.array_equal(self.SX3.get_unit_spike_features(0, 'dummy', start_frame=20, end_frame=46),
                                       self.example_info['features3'][1:6]))
        self.assertTrue('dummy2' in self.SX3.get_unit_spike_feature_names(0))
        self.assertTrue('dummy2_idxs' in self.SX3.get_unit_spike_feature_names(0))

        sub_extractor_full = se.SubSortingExtractor(self.SX3)
        sub_extractor_partial = se.SubSortingExtractor(self.SX3, start_frame=20, end_frame=46)

        self.assertTrue(np.array_equal(sub_extractor_full.get_unit_spike_features(0, 'dummy'),
                                       self.SX3.get_unit_spike_features(0, 'dummy')))
        self.assertTrue(np.array_equal(sub_extractor_partial.get_unit_spike_features(0, 'dummy'),
                                       self.SX3.get_unit_spike_features(0, 'dummy', start_frame=20, end_frame=46)))

        self.assertEqual(tuple(self.RX.get_epoch_info("epoch1").values()), self.example_info['epochs_info'][0])
        self.assertEqual(tuple(self.RX.get_epoch_info("epoch2").values()), self.example_info['epochs_info'][1])
        self.assertEqual(tuple(self.SX.get_epoch_info("epoch1").values()), self.example_info['epochs_info'][0])
        self.assertEqual(tuple(self.SX.get_epoch_info("epoch2").values()), self.example_info['epochs_info'][1])

        self.assertEqual(tuple(self.RX.get_epoch_info("epoch1").values()),
                         tuple(self.RX2.get_epoch_info("epoch1").values()))
        self.assertEqual(tuple(self.RX.get_epoch_info("epoch2").values()),
                         tuple(self.RX2.get_epoch_info("epoch2").values()))
        self.assertEqual(tuple(self.SX.get_epoch_info("epoch1").values()),
                         tuple(self.SX2.get_epoch_info("epoch1").values()))
        self.assertEqual(tuple(self.SX.get_epoch_info("epoch2").values()),
                         tuple(self.SX2.get_epoch_info("epoch2").values()))

        self.assertTrue(np.array_equal(self.RX2.frame_to_time(np.arange(self.RX2.get_num_frames())),
                                       self.example_info['times']))
        self.assertTrue(np.array_equal(self.SX2.get_unit_spike_train(3) / self.SX2.get_sampling_frequency() + 5,
                                       self.SX2.frame_to_time(self.SX2.get_unit_spike_train(3))))

        self.RX3.clear_channel_locations()
        self.assertTrue('location' not in self.RX3.get_shared_channel_property_names())
        self.RX3.set_channel_locations(self.example_info['geom'])
        self.assertTrue(np.array_equal(self.RX3.get_channel_locations(),
                                       self.RX2.get_channel_locations()))
        self.RX3.set_channel_groups(groups=[1], channel_ids=[1])
        self.assertEqual(self.RX3.get_channel_groups(channel_ids=[1]), 1)
        self.RX3.clear_channel_groups()
        self.assertEqual(self.RX3.get_channel_groups(channel_ids=[1]), 0)
        self.RX3.set_channel_locations(locations=[[np.nan, np.nan, np.nan]], channel_ids=[1])
        self.assertTrue('location' not in self.RX3.get_shared_channel_property_names())
        self.RX3.set_channel_locations(locations=[[0, 0, 0]], channel_ids=[1])
        self.assertTrue('location' in self.RX3.get_shared_channel_property_names())
        check_recording_return_types(self.RX)

    def test_allocate_arrays(self):
        shape = (30, 1000)
        dtype = 'int16'

        arr_in_memory = self.RX.allocate_array(shape=shape, dtype=dtype, memmap=False)
        arr_memmap = self.RX.allocate_array(shape=shape, dtype=dtype, memmap=True)

        assert isinstance(arr_in_memory, np.ndarray)
        assert isinstance(arr_memmap, np.memmap)
        assert arr_in_memory.shape == shape
        assert arr_memmap.shape == shape
        assert arr_in_memory.dtype == dtype
        assert arr_memmap.dtype == dtype

        arr_in_memory = self.SX.allocate_array(shape=shape, dtype=dtype, memmap=False)
        arr_memmap = self.SX.allocate_array(shape=shape, dtype=dtype, memmap=True)

        assert isinstance(arr_in_memory, np.ndarray)
        assert isinstance(arr_memmap, np.memmap)
        assert arr_in_memory.shape == shape
        assert arr_memmap.shape == shape
        assert arr_in_memory.dtype == dtype
        assert arr_memmap.dtype == dtype

    def test_cache_extractor(self):
        cache_rec = se.CacheRecordingExtractor(self.RX)
        check_recording_return_types(cache_rec)
        check_recordings_equal(self.RX, cache_rec)
        cache_rec.move_to('cache_rec')

        assert cache_rec.filename == 'cache_rec.dat'
        check_dumping(cache_rec, test_relative=True)

        cache_rec = se.CacheRecordingExtractor(self.RX, save_path='cache_rec2')
        check_recording_return_types(cache_rec)
        check_recordings_equal(self.RX, cache_rec)

        assert cache_rec.filename == 'cache_rec2.dat'
        check_dumping(cache_rec, test_relative=True)

        # test saving to file
        del cache_rec
        assert Path('cache_rec2.dat').is_file()

        # test tmp
        cache_rec = se.CacheRecordingExtractor(self.RX)
        tmp_file = cache_rec.filename
        del cache_rec
        assert not Path(tmp_file).is_file()

        cache_sort = se.CacheSortingExtractor(self.SX)
        check_sorting_return_types(cache_sort)
        check_sortings_equal(self.SX, cache_sort)
        cache_sort.move_to('cache_sort')

        assert cache_sort.filename == 'cache_sort.npz'
        check_dumping(cache_sort, test_relative=True)

        # test saving to file
        del cache_sort
        assert Path('cache_sort.npz').is_file()

        cache_sort = se.CacheSortingExtractor(self.SX, save_path='cache_sort2')
        check_sorting_return_types(cache_sort)
        check_sortings_equal(self.SX, cache_sort)

        assert cache_sort.filename == 'cache_sort2.npz'
        check_dumping(cache_sort, test_relative=True)

        # test saving to file
        del cache_sort
        assert Path('cache_sort2.npz').is_file()

        # test tmp
        cache_sort = se.CacheSortingExtractor(self.SX)
        tmp_file = cache_sort.filename
        del cache_sort
        assert not Path(tmp_file).is_file()

        # cleanup
        os.remove('cache_rec.dat')
        os.remove('cache_rec2.dat')
        os.remove('cache_sort.npz')
        os.remove('cache_sort2.npz')

    def test_not_dumpable_exception(self):
        try:
            self.RX.dump_to_json()
        except Exception as e:
            assert isinstance(e, NotDumpableExtractorError)

        try:
            self.RX.dump_to_pickle()
        except Exception as e:
            assert isinstance(e, NotDumpableExtractorError)

    def test_mda_extractor(self):
        path1 = self.test_dir + '/mda'
        path2 = path1 + '/firings_true.mda'
        se.MdaRecordingExtractor.write_recording(self.RX, path1)
        se.MdaSortingExtractor.write_sorting(self.SX, path2)
        RX_mda = se.MdaRecordingExtractor(path1)
        SX_mda = se.MdaSortingExtractor(path2)
        check_recording_return_types(RX_mda)
        check_recordings_equal(self.RX, RX_mda)
        check_sorting_return_types(SX_mda)
        check_sortings_equal(self.SX, SX_mda)
        check_dumping(RX_mda)
        check_dumping(SX_mda)

    def test_hdsort_extractor(self):
        path = self.test_dir + '/results_test_hdsort_extractor.mat'
        locations = np.ones((10, 2))
        se.HDSortSortingExtractor.write_sorting(self.SX, path, locations=locations, noise_std_by_channel=None)
        SX_hd = se.HDSortSortingExtractor(path)
        check_sorting_return_types(SX_hd)
        check_sortings_equal(self.SX, SX_hd)
        check_dumping(SX_hd)

    def test_npz_extractor(self):
        path = self.test_dir + '/sorting.npz'
        se.NpzSortingExtractor.write_sorting(self.SX, path)
        SX_npz = se.NpzSortingExtractor(path)

        # empty write
        sorting_empty = se.NumpySortingExtractor()
        path_empty = self.test_dir + '/sorting_empty.npz'
        se.NpzSortingExtractor.write_sorting(sorting_empty, path_empty)

        check_sorting_return_types(SX_npz)
        check_sortings_equal(self.SX, SX_npz)
        check_dumping(SX_npz)

    def test_biocam_extractor(self):
        path1 = self.test_dir + '/raw.brw'
        se.BiocamRecordingExtractor.write_recording(self.RX, path1)
        RX_biocam = se.BiocamRecordingExtractor(path1)
        check_recording_return_types(RX_biocam)
        check_recordings_equal(self.RX, RX_biocam)
        check_dumping(RX_biocam)

    def test_mearec_extractors(self):
        path1 = self.test_dir + '/raw.h5'
        se.MEArecRecordingExtractor.write_recording(self.RX, path1)
        RX_mearec = se.MEArecRecordingExtractor(path1)
        tr = RX_mearec.get_traces(channel_ids=[0, 1], end_frame=1000)
        check_recording_return_types(RX_mearec)
        check_recordings_equal(self.RX, RX_mearec)
        check_dumping(RX_mearec)

        path2 = self.test_dir + '/firings_true.h5'
        se.MEArecSortingExtractor.write_sorting(self.SX, path2, self.RX.get_sampling_frequency())
        SX_mearec = se.MEArecSortingExtractor(path2)
        check_sorting_return_types(SX_mearec)
        check_sortings_equal(self.SX, SX_mearec)
        check_dumping(SX_mearec)

    def test_hs2_extractor(self):
        path1 = self.test_dir + '/firings_true.hdf5'
        se.HS2SortingExtractor.write_sorting(self.SX, path1)
        SX_hs2 = se.HS2SortingExtractor(path1)
        check_sorting_return_types(SX_hs2)
        check_sortings_equal(self.SX, SX_hs2)
        self.assertEqual(SX_hs2.get_sampling_frequency(), self.SX.get_sampling_frequency())
        check_dumping(SX_hs2)

    def test_exdir_extractors(self):
        path1 = self.test_dir + '/raw.exdir'
        se.ExdirRecordingExtractor.write_recording(self.RX, path1)
        RX_exdir = se.ExdirRecordingExtractor(path1)
        check_recording_return_types(RX_exdir)
        check_recordings_equal(self.RX, RX_exdir)
        check_dumping(RX_exdir)

        path2 = self.test_dir + '/firings.exdir'
        se.ExdirSortingExtractor.write_sorting(self.SX, path2, self.RX)
        SX_exdir = se.ExdirSortingExtractor(path2)
        check_sorting_return_types(SX_exdir)
        check_sortings_equal(self.SX, SX_exdir)
        check_dumping(SX_exdir)

    def test_spykingcircus_extractor(self):
        path1 = self.test_dir + '/sc'
        se.SpykingCircusSortingExtractor.write_sorting(self.SX, path1)
        SX_spy = se.SpykingCircusSortingExtractor(path1)
        check_sorting_return_types(SX_spy)
        check_sortings_equal(self.SX, SX_spy)
        check_dumping(SX_spy)

    def test_multi_sub_recording_extractor(self):
        RX_multi = se.MultiRecordingTimeExtractor(
            recordings=[self.RX, self.RX, self.RX],
            epoch_names=['A', 'B', 'C']
        )
        RX_sub = RX_multi.get_epoch('C')
        check_recordings_equal(self.RX, RX_sub)
        check_recordings_equal(self.RX, RX_multi.recordings[0])
        check_recordings_equal(self.RX, RX_multi.recordings[1])
        check_recordings_equal(self.RX, RX_multi.recordings[2])
        self.assertEqual(4, len(RX_sub.get_channel_ids()))

        RX_multi = se.MultiRecordingChannelExtractor(
            recordings=[self.RX, self.RX2, self.RX3],
            groups=[1, 2, 3]
        )
        RX_sub = se.SubRecordingExtractor(RX_multi, channel_ids=[4, 5, 6, 7], renamed_channel_ids=[0, 1, 2, 3])
        # RX2 has times
        check_recordings_equal(self.RX2, RX_sub, check_times=False)
        check_recordings_equal(self.RX, RX_multi.recordings[0])
        check_recordings_equal(self.RX2, RX_multi.recordings[1], check_times=False)
        check_recordings_equal(self.RX3, RX_multi.recordings[2])
        self.assertEqual([2, 2, 2, 2], list(RX_sub.get_channel_groups()))
        self.assertEqual(12, len(RX_multi.get_channel_ids()))

        RX_multi = se.MultiRecordingChannelExtractor(
            recordings=[self.RX2, self.RX2, self.RX2],
            groups=[1, 2, 3]
        )
        RX_sub = se.SubRecordingExtractor(RX_multi, channel_ids=[4, 5, 6, 7], renamed_channel_ids=[0, 1, 2, 3])
        check_recordings_equal(self.RX2, RX_sub, check_times=False)
        check_recordings_equal(self.RX2, RX_multi.recordings[0])
        check_recordings_equal(self.RX2, RX_multi.recordings[1], check_times=False)
        check_recordings_equal(self.RX2, RX_multi.recordings[2])
        self.assertTrue(np.array_equal([2, 2, 2, 2], list(RX_sub.get_channel_groups())))
        self.assertTrue(12 == len(RX_multi.get_channel_ids()))
        self.assertTrue(np.array_equal(RX_multi.frame_to_time(np.arange(RX_multi.get_num_frames())),
                        np.arange(RX_multi.get_num_frames()) / RX_multi.get_sampling_frequency() + 5))

        rx1 = self.RX
        rx2 = self.RX2
        rx3 = self.RX3
        rx2.set_channel_property(0, "foo", 100)
        rx3.set_channel_locations([11, 11], channel_ids=0)
        RX_multi_c = se.MultiRecordingChannelExtractor(
            recordings=[rx1, rx2, rx3],
            groups=[0, 0, 1]
        )
        self.assertTrue(np.array_equal([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], RX_multi_c.get_channel_ids()))
        self.assertTrue(np.array_equal([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], RX_multi_c.get_channel_groups()))
        self.assertEqual(rx2.get_channel_property(0, "foo"), RX_multi_c.get_channel_property(4, "foo"))
        self.assertTrue(np.array_equal(rx3.get_channel_locations([0])[0], RX_multi_c.get_channel_locations([8])[0]))

    def test_ttl_frames_in_sub_multi(self):
        # sub recording
        start_frame = self.example_info['num_frames'] // 3
        end_frame = 2 * self.example_info['num_frames'] // 3
        RX_sub = se.SubRecordingExtractor(self.RX, start_frame=start_frame, end_frame=end_frame)
        original_ttls = self.RX.get_ttl_events()[0]
        ttls_in_sub = original_ttls[np.where((original_ttls >= start_frame) & (original_ttls < end_frame))[0]]
        self.assertTrue(np.array_equal(RX_sub.get_ttl_events()[0], ttls_in_sub - start_frame))

        # multirecording
        RX_multi = se.MultiRecordingTimeExtractor(recordings=[self.RX, self.RX, self.RX])
        ttls_originals = self.RX.get_ttl_events()[0]
        num_ttls = len(ttls_originals)
        self.assertEqual(len(RX_multi.get_ttl_events()[0]), 3 * num_ttls)
        self.assertTrue(np.array_equal(RX_multi.get_ttl_events()[0][:num_ttls], ttls_originals))
        self.assertTrue(np.array_equal(RX_multi.get_ttl_events()[0][num_ttls:2 * num_ttls],
                                       ttls_originals + self.RX.get_num_frames()))
        self.assertTrue(np.array_equal(RX_multi.get_ttl_events()[0][2 * num_ttls:],
                                       ttls_originals + 2 * self.RX.get_num_frames()))

    def test_multi_sub_sorting_extractor(self):
        N = self.RX.get_num_frames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX, self.SX],
        )
        SX_multi.set_unit_property(unit_id=1, property_name='dummy', value=5)
        SX_sub = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0)
        check_sortings_equal(SX_multi, SX_sub)
        self.assertEqual(SX_multi.get_unit_property(1, 'dummy'), SX_sub.get_unit_property(1, 'dummy'))

        N = self.RX.get_num_frames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX2],
        )
        SX_sub1 = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0, end_frame=N)
        check_sortings_equal(SX_multi, SX_sub1)
        check_sortings_equal(self.SX, SX_multi.sortings[0])
        check_sortings_equal(self.SX2, SX_multi.sortings[1])

    def test_dump_load_multi_sub_extractor(self):
        # generate dumpable formats
        path1 = self.test_dir + '/mda'
        path2 = path1 + '/firings_true.mda'
        se.MdaRecordingExtractor.write_recording(self.RX, path1)
        se.MdaSortingExtractor.write_sorting(self.SX, path2)
        RX_mda = se.MdaRecordingExtractor(path1)
        SX_mda = se.MdaSortingExtractor(path2)

        RX_multi_chan = se.MultiRecordingChannelExtractor(recordings=[RX_mda, RX_mda, RX_mda])
        check_dumping(RX_multi_chan)
        RX_multi_time = se.MultiRecordingTimeExtractor(recordings=[RX_mda, RX_mda, RX_mda], )
        check_dumping(RX_multi_time)
        RX_multi_chan = se.SubRecordingExtractor(RX_mda, channel_ids=[0, 1])
        check_dumping(RX_multi_chan)

        SX_sub = se.SubSortingExtractor(SX_mda, unit_ids=[1, 2])
        check_dumping(SX_sub)
        SX_multi = se.MultiSortingExtractor(sortings=[SX_mda, SX_mda, SX_mda])
        check_dumping(SX_multi)

    def test_nwb_extractor(self):
        path1 = self.test_dir + '/test.nwb'
        se.NwbRecordingExtractor.write_recording(self.RX, path1)
        RX_nwb = se.NwbRecordingExtractor(path1)
        check_recording_return_types(RX_nwb)
        check_recordings_equal(self.RX, RX_nwb)
        check_dumping(RX_nwb)

        del RX_nwb
        se.NwbRecordingExtractor.write_recording(recording=self.RX, save_path=path1, overwrite=True)
        RX_nwb = se.NwbRecordingExtractor(path1)
        check_recording_return_types(RX_nwb)
        check_recordings_equal(self.RX, RX_nwb)
        check_dumping(RX_nwb)

        # append sorting to existing file
        se.NwbSortingExtractor.write_sorting(sorting=self.SX, save_path=path1, overwrite=False)

        path2 = self.test_dir + "/firings_true.nwb"
        se.NwbRecordingExtractor.write_recording(recording=self.RX, save_path=path2)
        se.NwbSortingExtractor.write_sorting(sorting=self.SX, save_path=path2)
        SX_nwb = se.NwbSortingExtractor(path2)
        check_sortings_equal(self.SX, SX_nwb)
        check_dumping(SX_nwb)

        # Test for handling unit property descriptions argument
        property_descriptions = dict(stability="This is a description of stability.")
        se.NwbRecordingExtractor.write_recording(recording=self.RX, save_path=path1, overwrite=True)
        se.NwbSortingExtractor.write_sorting(
            sorting=self.SX,
            save_path=path1,
            property_descriptions=property_descriptions
        )
        SX_nwb = se.NwbSortingExtractor(path1)
        check_sortings_equal(self.SX, SX_nwb)
        check_dumping(SX_nwb)

        # Test for handling skip_properties argument
        se.NwbRecordingExtractor.write_recording(recording=self.RX, save_path=path1, overwrite=True)
        se.NwbSortingExtractor.write_sorting(
            sorting=self.SX,
            save_path=path1,
            skip_properties=['stability']
        )
        SX_nwb = se.NwbSortingExtractor(path1)
        assert 'stability' not in SX_nwb.get_shared_unit_property_names()
        check_sortings_equal(self.SX, SX_nwb)
        check_dumping(SX_nwb)

        # Test for handling skip_features argument
        se.NwbRecordingExtractor.write_recording(recording=self.RX, save_path=path1, overwrite=True)
        # SX2 has timestamps, so loading it back from Nwb will not recover the same spike frames. USe use_times=False
        se.NwbSortingExtractor.write_sorting(
            sorting=self.SX2,
            save_path=path1,
            skip_features=['widths'],
            use_times=False
        )
        SX_nwb = se.NwbSortingExtractor(path1)
        assert 'widths' not in SX_nwb.get_shared_unit_spike_feature_names()
        check_sortings_equal(self.SX2, SX_nwb)
        check_dumping(SX_nwb)

        # Test writting multiple recordings using metadata
        metadata = get_default_nwbfile_metadata()
        path_nwb = self.test_dir + '/test_multiple.nwb'
        se.NwbRecordingExtractor.write_recording(
            recording=self.RX, 
            save_path=path_nwb,
            metadata=metadata,
            write_as='raw',
            es_key='ElectricalSeries_raw',
        )
        se.NwbRecordingExtractor.write_recording(
            recording=self.RX2, 
            save_path=path_nwb,
            metadata=metadata,
            write_as='processed',
            es_key='ElectricalSeries_processed',
        )
        se.NwbRecordingExtractor.write_recording(
            recording=self.RX3, 
            save_path=path_nwb,
            metadata=metadata,
            write_as='lfp',
            es_key='ElectricalSeries_lfp',
        )

        RX_nwb = se.NwbRecordingExtractor(
            file_path=path_nwb,
            electrical_series_name='raw_traces'
        )
        check_recording_return_types(RX_nwb)
        check_recordings_equal(self.RX, RX_nwb)
        check_dumping(RX_nwb)
        del RX_nwb

    def test_nixio_extractor(self):
        path1 = os.path.join(self.test_dir, 'raw.nix')
        se.NIXIORecordingExtractor.write_recording(self.RX, path1)
        RX_nixio = se.NIXIORecordingExtractor(path1)
        check_recording_return_types(RX_nixio)
        check_recordings_equal(self.RX, RX_nixio)
        check_dumping(RX_nixio)
        del RX_nixio
        # test force overwrite
        se.NIXIORecordingExtractor.write_recording(self.RX, path1,
                                                   overwrite=True)

        path2 = self.test_dir + '/firings_true.nix'
        se.NIXIOSortingExtractor.write_sorting(self.SX, path2)
        SX_nixio = se.NIXIOSortingExtractor(path2)
        check_sorting_return_types(SX_nixio)
        check_sortings_equal(self.SX, SX_nixio)
        check_dumping(SX_nixio)

    def test_shybrid_extractors(self):
        # test sorting extractor
        se.SHYBRIDSortingExtractor.write_sorting(self.SX, self.test_dir)
        initial_sorting_file = os.path.join(self.test_dir, 'initial_sorting.csv')
        SX_shybrid = se.SHYBRIDSortingExtractor(initial_sorting_file)
        check_sorting_return_types(SX_shybrid)
        check_sortings_equal(self.SX, SX_shybrid)
        check_dumping(SX_shybrid)

        # test recording extractor
        se.SHYBRIDRecordingExtractor.write_recording(self.RX,
                                                     self.test_dir,
                                                     initial_sorting_file)
        RX_shybrid = se.SHYBRIDRecordingExtractor(os.path.join(self.test_dir,
                                                               'recording.bin'))
        check_recording_return_types(RX_shybrid)
        check_recordings_equal(self.RX, RX_shybrid)
        check_dumping(RX_shybrid)

    def test_neuroscope_extractors(self):
        # NeuroscopeRecordingExtractor tests
        nscope_dir = Path(self.test_dir) / 'neuroscope_rec0'
        dat_file = nscope_dir / 'neuroscope_rec0.dat'
        se.NeuroscopeRecordingExtractor.write_recording(self.RX, nscope_dir)
        RX_ns = se.NeuroscopeRecordingExtractor(dat_file)

        check_recording_return_types(RX_ns)
        check_recordings_equal(self.RX, RX_ns, force_dtype='int32')
        check_dumping(RX_ns)

        check_recording_return_types(RX_ns)
        check_recordings_equal(self.RX, RX_ns, force_dtype='int32')
        check_dumping(RX_ns)

        del RX_ns
        # overwrite
        nscope_dir = Path(self.test_dir) / 'neuroscope_rec1'
        dat_file = nscope_dir / 'neuroscope_rec1.dat'
        se.NeuroscopeRecordingExtractor.write_recording(recording=self.RX, save_path=nscope_dir)
        RX_ns = se.NeuroscopeRecordingExtractor(dat_file)
        check_recording_return_types(RX_ns)
        check_recordings_equal(self.RX, RX_ns)
        check_dumping(RX_ns)

        # NeuroscopeMultiRecordingTimeExtractor tests
        nscope_dir = Path(self.test_dir) / "neuroscope_rec2"
        dat_file = nscope_dir / "neuroscope_rec2.dat"
        RX_multirecording = se.MultiRecordingTimeExtractor(recordings=[self.RX, self.RX])
        se.NeuroscopeMultiRecordingTimeExtractor.write_recording(recording=RX_multirecording, save_path=nscope_dir)
        RX_mre = se.NeuroscopeMultiRecordingTimeExtractor(folder_path=nscope_dir)
        check_recording_return_types(RX_mre)
        check_recordings_equal(RX_multirecording, RX_mre)
        check_dumping(RX_mre)

        # NeuroscopeSortingExtractor tests
        nscope_dir = Path(self.test_dir) / 'neuroscope_sort0'
        sort_name = 'neuroscope_sort0'
        initial_sorting_resfile = Path(self.test_dir) / sort_name / f'{sort_name}.res'
        initial_sorting_clufile = Path(self.test_dir) / sort_name / f'{sort_name}.clu'
        se.NeuroscopeSortingExtractor.write_sorting(self.SX, nscope_dir)
        SX_neuroscope = se.NeuroscopeSortingExtractor(resfile_path=initial_sorting_resfile,
                                                      clufile_path=initial_sorting_clufile)
        check_sorting_return_types(SX_neuroscope)
        check_sortings_equal(self.SX, SX_neuroscope)
        check_dumping(SX_neuroscope)
        SX_neuroscope_no_mua = se.NeuroscopeSortingExtractor(resfile_path=initial_sorting_resfile,
                                                             clufile_path=initial_sorting_clufile,
                                                             keep_mua_units=False)
        check_sorting_return_types(SX_neuroscope_no_mua)
        check_dumping(SX_neuroscope_no_mua)

        # Test for extra argument 'keep_mua_units' resulted in the right output
        SX_neuroscope_no_mua = se.NeuroscopeSortingExtractor(resfile_path=initial_sorting_resfile,
                                                             clufile_path=initial_sorting_clufile,
                                                             keep_mua_units=False)
        check_sorting_return_types(SX_neuroscope_no_mua)
        check_dumping(SX_neuroscope_no_mua)

        num_original_units = len(SX_neuroscope.get_unit_ids())
        self.assertEqual(list(SX_neuroscope.get_unit_ids()), list(range(1, num_original_units + 1)))
        self.assertEqual(list(SX_neuroscope_no_mua.get_unit_ids()), list(range(2, num_original_units + 1)))

        # Tests for the auto-detection of format for NeuroscopeSortingExtractor
        SX_neuroscope_from_fp = se.NeuroscopeSortingExtractor(folder_path=nscope_dir)
        check_sorting_return_types(SX_neuroscope_from_fp)
        check_sortings_equal(self.SX, SX_neuroscope_from_fp)
        check_dumping(SX_neuroscope_from_fp)

        # Tests for the NeuroscopeMultiSortingExtractor
        nscope_dir = Path(self.test_dir) / 'neuroscope_sort1'
        SX_multisorting = se.MultiSortingExtractor(sortings=[self.SX, self.SX])
        se.NeuroscopeMultiSortingExtractor.write_sorting(SX_multisorting, nscope_dir)
        SX_neuroscope_mse = se.NeuroscopeMultiSortingExtractor(nscope_dir)
        check_sorting_return_types(SX_neuroscope_mse)
        check_sortings_equal(SX_multisorting, SX_neuroscope_mse)
        check_dumping(SX_neuroscope_mse)

    def test_cell_explorer_extractor(self):
        sorter_id = "cell_explorer_sorter"
        cell_explorer_dir = Path(self.test_dir) / sorter_id
        spikes_matfile_path = cell_explorer_dir / f"{sorter_id}.spikes.cellinfo.mat"
        se.CellExplorerSortingExtractor.write_sorting(sorting=self.SX, save_path=spikes_matfile_path)
        SX_cell_explorer = se.CellExplorerSortingExtractor(spikes_matfile_path=spikes_matfile_path)
        check_sorting_return_types(SX_cell_explorer)
        check_sortings_equal(self.SX, SX_cell_explorer)
        check_dumping(SX_cell_explorer)


if __name__ == '__main__':
    unittest.main()
