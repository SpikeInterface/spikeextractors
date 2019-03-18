import numpy as np
import os, sys
import unittest
import tempfile
import shutil


def append_to_path(dir0):  # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)


append_to_path(os.getcwd() + '/..')
import spikeextractors as se


class TestExtractors(unittest.TestCase):
    def setUp(self):
        self.RX, self.SX, self.SX2, self.example_info = self._create_example()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _create_example(self):
        channel_ids = [0, 1, 2, 3]
        num_channels = 4
        num_frames = 10000
        samplerate = 30000
        X = np.random.normal(0, 1, (num_channels, num_frames))
        geom = np.random.normal(0, 1, (num_channels, 2))
        X = (X * 100).astype(int)
        RX = se.NumpyRecordingExtractor(timeseries=X, samplerate=samplerate, geom=geom)
        SX = se.NumpySortingExtractor()
        spike_times = [200, 300, 400]
        train1 = np.sort(np.rint(np.random.uniform(0, num_frames, spike_times[0])).astype(int))
        SX.addUnit(unit_id=1, times=train1)
        SX.addUnit(unit_id=2, times=np.sort(np.random.uniform(0, num_frames, spike_times[1])))
        SX.addUnit(unit_id=3, times=np.sort(np.random.uniform(0, num_frames, spike_times[2])))
        SX.setUnitProperty(unit_id=1, property_name='stablility', value=80)
        SX2 = se.NumpySortingExtractor()
        spike_times2 = [100, 150, 450]
        train2 = np.rint(np.random.uniform(0, num_frames, spike_times[0])).astype(int)
        SX2.addUnit(unit_id=3, times=train2)
        SX2.addUnit(unit_id=4, times=np.random.uniform(0, num_frames, spike_times2[1]))
        SX2.addUnit(unit_id=5, times=np.random.uniform(0, num_frames, spike_times2[2]))
        SX2.setUnitProperty(unit_id=4, property_name='stablility', value=80)
        RX.setChannelProperty(channel_id=0, property_name='location', value=(0, 0))
        example_info = dict(
            channel_ids=channel_ids,
            num_channels=num_channels,
            num_frames=num_frames,
            samplerate=samplerate,
            unit_ids=[1, 2, 3],
            train1=train1,
            unit_prop=80,
            channel_prop=(0, 0)
        )

        return (RX, SX, SX2, example_info)

    def test_example(self):
        self.assertEqual(self.RX.getChannelIds(), self.example_info['channel_ids'])
        self.assertEqual(self.RX.getNumChannels(), self.example_info['num_channels'])
        self.assertEqual(self.RX.getNumFrames(), self.example_info['num_frames'])
        self.assertEqual(self.RX.getSamplingFrequency(), self.example_info['samplerate'])
        self.assertEqual(self.SX.getUnitIds(), self.example_info['unit_ids'])
        self.assertEqual(self.RX.getChannelProperty(channel_id=0, property_name='location'),
                         self.example_info['channel_prop'])
        self.assertEqual(self.SX.getUnitProperty(unit_id=1, property_name='stablility'), self.example_info['unit_prop'])
        self.assertTrue(np.array_equal(self.SX.getUnitSpikeTrain(1), self.example_info['train1']))
        self.assertTrue(issubclass(self.SX.getUnitSpikeTrain(1).dtype.type, np.integer))
        self._check_recording_return_types(self.RX)

    def test_mda_extractor(self):
        path1 = self.test_dir + '/mda'
        path2 = path1 + '/firings_true.mda'
        se.MdaRecordingExtractor.writeRecording(self.RX, path1)
        se.MdaSortingExtractor.writeSorting(self.SX, path2)
        RX_mda = se.MdaRecordingExtractor(path1)
        SX_mda = se.MdaSortingExtractor(path2)
        self._check_recording_return_types(RX_mda)
        self._check_recordings_equal(self.RX, RX_mda)
        self._check_sorting_return_types(SX_mda)
        self._check_sortings_equal(self.SX, SX_mda)

    # old: don't do this test because pynwb causes a seg fault!
    # don't do this test because pynwb interface has changed
    # def test_nwb_extractor(self):
    #    path1=self.test_dir+'/test.nwb'
    #    se.NwbRecordingExtractor.writeRecording(self.RX,path1,acquisition_name='test')
    #    RX_nwb=se.NwbRecordingExtractor(path1,acquisition_name='test')
    #    self._check_recording_return_types(RX_nwb)
    #    self._check_recordings_equal(self.RX,RX_nwb)

    def _check_recording_return_types(self, RX):
        channel_ids = RX.getChannelIds()
        M = RX.getNumChannels()
        N = RX.getNumFrames()
        self.assertTrue((type(RX.getNumChannels()) == int) or (type(RX.getNumChannels()) == np.int64))
        self.assertTrue((type(RX.getNumFrames()) == int) or (type(RX.getNumFrames()) == np.int64))
        self.assertTrue((type(RX.getSamplingFrequency()) == float) or (type(RX.getSamplingFrequency()) == np.float64))
        self.assertTrue(type(RX.getTraces(start_frame=0, end_frame=10)) == np.ndarray)
        for channel_id in channel_ids:
            self.assertTrue((type(channel_id) == int) or (type(channel_id) == np.int64))

    def test_biocam_extractor(self):
        path1 = self.test_dir + '/raw.brw'
        se.BiocamRecordingExtractor.writeRecording(self.RX, path1)
        RX_biocam = se.BiocamRecordingExtractor(path1)
        self._check_recording_return_types(RX_biocam)
        self._check_recordings_equal(self.RX, RX_biocam)

    def test_mearec_extractors(self):
        path1 = self.test_dir + '/raw.h5'
        se.MEArecRecordingExtractor.writeRecording(self.RX, path1)
        RX_mearec = se.MEArecRecordingExtractor(path1)
        tr = RX_mearec.getTraces(channel_ids=[0,1], end_frame=1000)
        self._check_recording_return_types(RX_mearec)
        self._check_recordings_equal(self.RX, RX_mearec)

        path2 = self.test_dir + '/firings_true.h5'
        se.MEArecSortingExtractor.writeSorting(self.SX, path2, self.RX.getSamplingFrequency())
        SX_mearec = se.MEArecSortingExtractor(path2)
        self._check_sorting_return_types(SX_mearec)
        self._check_sortings_equal(self.SX, SX_mearec)

    def test_hs2_extractor(self):
        path1 = self.test_dir + '/firings_true.hdf5'
        se.HS2SortingExtractor.writeSorting(self.SX, path1)
        SX_hs2 = se.HS2SortingExtractor(path1)
        self._check_sorting_return_types(SX_hs2)
        self._check_sortings_equal(self.SX, SX_hs2)


    # def test_exdir_extractors(self):
    #     path1 = self.test_dir + '/raw'
    #     se.ExdirRecordingExtractor.writeRecording(self.RX, path1)
    #     RX_exdir = se.ExdirRecordingExtractor(path1)
    #     self._check_recording_return_types(RX_exdir)
    #     self._check_recordings_equal(self.RX, RX_exdir)
    #
    #     path2 = self.test_dir + '/firings_true'
    #     se.ExdirSortingExtractor.writeSorting(self.SX, path2, self.RX.getSamplingFrequency())
    #     SX_exdir = se.ExdirSortingExtractor(path2)
    #     self._check_sorting_return_types(SX_exdir)
    #     self._check_sortings_equal(self.SX, SX_exdir)


    def test_kilosort_extractor(self):
        path1 = self.test_dir + '/firings_true'
        se.KiloSortSortingExtractor.writeSorting(self.SX, path1)
        SX_ks = se.KiloSortSortingExtractor(path1)
        self._check_sorting_return_types(SX_ks)
        self._check_sortings_equal(self.SX, SX_ks)

    def test_klusta_extractor(self):
        path1 = self.test_dir + '/firings_true.kwik'
        se.KlustaSortingExtractor.writeSorting(self.SX, path1)
        SX_kl = se.KlustaSortingExtractor(path1)
        self._check_sorting_return_types(SX_kl)
        self._check_sortings_equal(self.SX, SX_kl)

    def test_spykingcircus_extractor(self):
        path1 = self.test_dir + '/firings_true'
        se.SpykingCircusSortingExtractor.writeSorting(self.SX, path1)
        SX_spy = se.SpykingCircusSortingExtractor(path1)
        self._check_sorting_return_types(SX_spy)
        self._check_sortings_equal(self.SX, SX_spy)

    def test_multi_sub_recording_extractor(self):
        RX_multi = se.MultiRecordingExtractor(
            recordings=[self.RX, self.RX, self.RX],
            epoch_names=['A', 'B', 'C']
        )
        RX_sub = RX_multi.getEpoch('C')
        self._check_recordings_equal(self.RX, RX_sub)

    def test_curated_sorting_extractor(self):
        CSX = se.CuratedSortingExtractor(
            parent_sorting=self.SX
        )
        CSX.mergeUnits(unit_ids=[1, 2])
        original_spike_train = np.sort(np.concatenate((self.SX.getUnitSpikeTrain(1), self.SX.getUnitSpikeTrain(2))))
        self.assertTrue(np.array_equal(CSX.getUnitSpikeTrain(4), original_spike_train))

        CSX.splitUnit(unit_id=3, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_spike_train = self.SX.getUnitSpikeTrain(3)
        split_spike_train_1 = CSX.getUnitSpikeTrain(5)
        split_spike_train_2 = CSX.getUnitSpikeTrain(6)
        self.assertTrue(np.array_equal(original_spike_train[:10], split_spike_train_1))
        self.assertTrue(np.array_equal(original_spike_train[10:], split_spike_train_2))


    def test_multi_sub_sorting_extractor(self):
        N = self.RX.getNumFrames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX, self.SX],
            start_frames=[0, N, 2 * N]
        )
        SX_sub = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=N, end_frame=2 * N)
        self._check_sortings_equal(self.SX, SX_sub)

        N = self.RX.getNumFrames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX, self.SX],
            start_frames=[0, N, 2 * N]
        )
        SX_sub = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0)
        self._check_sortings_equal(SX_multi, SX_sub)

        N = self.RX.getNumFrames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX, self.SX],
            start_frames=[2 * N, 0, N]
        )
        SX_sub = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=N, end_frame=2 * N)
        self._check_sortings_equal(self.SX, SX_sub)

        N = self.RX.getNumFrames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX, self.SX],
            start_frames=[0, 0, 0]
        )
        SX_sub = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0)
        self._check_sortings_equal(SX_multi, SX_sub)

        N = self.RX.getNumFrames()
        SX_multi = se.MultiSortingExtractor(
            sortings=[self.SX, self.SX2],
            start_frames=[0, 0]
        )
        SX_sub1 = se.SubSortingExtractor(parent_sorting=SX_multi, start_frame=0, end_frame=N)
        self._check_sortings_equal(SX_multi, SX_sub1)

    def _check_recordings_equal(self, RX1, RX2):
        M = RX1.getNumChannels()
        N = RX1.getNumFrames()
        # getChannelIds
        self.assertEqual(RX1.getChannelIds(), RX2.getChannelIds())
        # getNumChannels
        self.assertEqual(RX1.getNumChannels(), RX2.getNumChannels())
        # getNumFrames
        self.assertEqual(RX1.getNumFrames(), RX2.getNumFrames())
        # getSamplingFrequency
        self.assertEqual(RX1.getSamplingFrequency(), RX2.getSamplingFrequency())
        # getTraces
        tmp1 = RX1.getTraces()
        tmp2 = RX2.getTraces()
        self.assertTrue(np.allclose(
            RX1.getTraces(),
            RX2.getTraces()
        ))
        sf = 0
        ef = N
        ch = [0, M - 1]
        self.assertTrue(np.allclose(
            RX1.getTraces(channel_ids=ch, start_frame=sf, end_frame=ef),
            RX2.getTraces(channel_ids=ch, start_frame=sf, end_frame=ef)
        ))
        for f in range(0, RX1.getNumFrames(), 10):
            self.assertTrue(np.isclose(RX1.frameToTime(f), RX2.frameToTime(f)))
            self.assertTrue(np.isclose(RX1.timeToFrame(RX1.frameToTime(f)), RX2.timeToFrame(RX2.frameToTime(f))))
        # getSnippets
        frames = [30, 50, 80]
        snippets1 = RX1.getSnippets(reference_frames=frames, snippet_len=20)
        snippets2 = RX2.getSnippets(reference_frames=frames, snippet_len=(10, 10))
        for ii in range(len(frames)):
            self.assertTrue(np.allclose(snippets1[ii], snippets2[ii]))

    def _check_sorting_return_types(self, SX):
        unit_ids = SX.getUnitIds()
        self.assertTrue(all(isinstance(id, int) or isinstance(id, np.integer) for id in unit_ids))
        for id in unit_ids:
            train = SX.getUnitSpikeTrain(id)
            # print(train)
            self.assertTrue(all(isinstance(x, int) or isinstance(x, np.integer) for x in train))

    def _check_sortings_equal(self, SX1, SX2):
        K = len(SX1.getUnitIds())
        # getUnitIds
        ids1 = np.sort(np.array(SX1.getUnitIds()))
        ids2 = np.sort(np.array(SX2.getUnitIds()))
        self.assertTrue(np.allclose(ids1, ids2))
        for id in ids1:
            train1 = np.sort(SX1.getUnitSpikeTrain(id))
            train2 = np.sort(SX2.getUnitSpikeTrain(id))
            # print(train1)
            # print(train2)
            self.assertTrue(np.array_equal(train1, train2))


if __name__ == '__main__':
    unittest.main()
