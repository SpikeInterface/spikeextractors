import numpy as np
import os, sys
import unittest
import tempfile
import shutil
def append_to_path(dir0): # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)
append_to_path(os.getcwd()+'/..')
import spikeinterface as si
 
class TestExtractors(unittest.TestCase):
    def setUp(self):
        self.RX, self.SX, self.example_info = self._create_example()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _create_example(self):
        M=4
        N=10000
        samplerate=30000
        X=np.random.normal(0,1,(M,N))
        geom=np.random.normal(0,1,(M,2))
        X=(X*100).astype(int)
        RX=si.NumpyRecordingExtractor(timeseries=X,samplerate=samplerate,geom=geom)
        SX=si.NumpySortingExtractor()
        L=200
        train1=np.random.uniform(0,N,L)
        SX.addUnit(unit_id=1,times=train1)
        SX.addUnit(unit_id=2,times=np.random.uniform(0,N,L))
        SX.addUnit(unit_id=3,times=np.random.uniform(0,N,L))
        example_info=dict(
            M=M,
            N=N,
            samplerate=samplerate,
            unit_ids=[1,2,3],
            train1=train1
        )
        return (RX,SX,example_info)

    def test_example(self):
        self.assertEqual(self.RX.getNumChannels(),self.example_info['M'])
        self.assertEqual(self.RX.getNumFrames(),self.example_info['N'])
        self.assertEqual(self.RX.getSamplingFrequency(),self.example_info['samplerate'])
        self.assertEqual(self.SX.getUnitIds(),self.example_info['unit_ids'])
        self.assertTrue(np.allclose(self.SX.getUnitSpikeTrain(1),self.example_info['train1']))
        self._check_recording_return_types(self.RX)
     
    def test_mda_extractor(self):
        path1=self.test_dir+'/mda'
        path2=path1+'/firings_true.mda'
        si.MdaRecordingExtractor.writeRecording(self.RX,path1)
        si.MdaSortingExtractor.writeSorting(self.SX,path2)
        RX_mda=si.MdaRecordingExtractor(path1)
        SX_mda=si.MdaSortingExtractor(path2)
        self._check_recording_return_types(RX_mda)
        self._check_recordings_equal(self.RX,RX_mda)
        self._check_sorting_return_types(SX_mda)
        self._check_sortings_equal(self.SX,SX_mda)

    # don't do this test because pynwb causes a seg fault!
    def test_nwb_extractor(self):
        path1=self.test_dir+'/test.nwb'
        si.NwbRecordingExtractor.writeRecording(self.RX,path1,acquisition_name='test')
        RX_nwb=si.NwbRecordingExtractor(path1,acquisition_name='test')
        self._check_recording_return_types(RX_nwb)
        self._check_recordings_equal(self.RX,RX_nwb)

    def test_biocam_extractor(self):
        path1=self.test_dir+'/raw.brw'
        si.BiocamRecordingExtractor.writeRecording(self.RX,path1)
        RX_biocam=si.BiocamRecordingExtractor(path1)
        self._check_recording_return_types(RX_biocam)
        self._check_recordings_equal(self.RX,RX_biocam)

    def test_hs2_extractor(self):
        path1=self.test_dir+'/firings_true.hdf5'
        si.HS2SortingExtractor.writeSorting(self.SX,path1)
        SX_hs2=si.HS2SortingExtractor(path1)
        self._check_sorting_return_types(SX_hs2)
        self._check_sortings_equal(self.SX,SX_hs2)

    def test_multi_sub_extractor(self):
        RX_multi=si.MultiRecordingExtractor(
            recording_extractors=[self.RX,self.RX,self.RX],
            epoch_names=['A','B','C']
        )
        RX_sub = RX_multi.getEpoch('C')
        self._check_recordings_equal(self.RX,RX_sub)

    def _check_recording_return_types(self,RX):
        M=RX.getNumChannels()
        N=RX.getNumFrames()
        self.assertTrue((type(RX.getNumChannels())==int) or (type(RX.getNumChannels())==np.int64))
        self.assertTrue((type(RX.getSamplingFrequency())==float) or (type(RX.getSamplingFrequency())==np.float64))
        self.assertTrue(type(RX.getTraces(start_frame=0,end_frame=10))==np.ndarray)
        self.assertTrue(type(RX.getChannelInfo(channel_id=0))==dict)

    def _check_recordings_equal(self,RX1,RX2):
        M=RX1.getNumChannels()
        N=RX1.getNumFrames()
        # getNumChannels
        self.assertEqual(RX1.getNumChannels(),RX2.getNumChannels())
        # getNumFrames
        self.assertEqual(RX1.getNumFrames(),RX2.getNumFrames())
        # getSamplingFrequency
        self.assertEqual(RX1.getSamplingFrequency(),RX2.getSamplingFrequency())
        # getTraces
        tmp1=RX1.getTraces()
        tmp2=RX2.getTraces()
        self.assertTrue(np.allclose(
            RX1.getTraces(),
            RX2.getTraces()
        ))
        sf=0; ef=0; ch=[0,M-1]
        self.assertTrue(np.allclose(
            RX1.getTraces(start_frame=sf,end_frame=ef,channel_ids=ch),
            RX2.getTraces(start_frame=sf,end_frame=ef,channel_ids=ch)
        ))
        # getChannelInfo
        # commented out as not part of common API
        # for m in range(M):
        #     self.assertTrue(np.allclose(
        #         np.array(RX1.getChannelInfo(channel_id=m)['location']),
        #         np.array(RX2.getChannelInfo(channel_id=m)['location'])
        #     ))
        # timeToFrame / frameToTime
        for f in range(0,RX1.getNumFrames(),10):
            self.assertTrue(np.isclose(RX1.frameToTime(f),RX2.frameToTime(f)))
            self.assertTrue(np.isclose(RX1.timeToFrame(RX1.frameToTime(f)),RX2.timeToFrame(RX2.frameToTime(f))))
        # getSnippets
        frames=[30,50,80]
        snippets1=RX1.getSnippets(snippet_len_before=10,snippet_len_after=10,start_frames=frames)
        snippets2=RX2.getSnippets(snippet_len_before=10,snippet_len_after=10,start_frames=frames)
        for ii in range(len(frames)):
            self.assertTrue(np.allclose(snippets1[ii],snippets2[ii]))

    def _check_sorting_return_types(self,SX):
        unit_ids=SX.getUnitIds()
        self.assertTrue(all(isinstance(id, int) or isinstance(id, np.int64) for id in unit_ids))
        for id in unit_ids:
            train=SX.getUnitSpikeTrain(id)
            self.assertTrue(all(isinstance(x, float) or isinstance(x, np.float64) for x in train))

    def _check_sortings_equal(self,SX1,SX2):
        K=len(SX1.getUnitIds())
        # getUnitIds
        ids1=np.sort(np.array(SX1.getUnitIds()))
        ids2=np.sort(np.array(SX2.getUnitIds()))
        self.assertTrue(np.allclose(ids1,ids2))
        for id in ids1:
            train1=np.sort(SX1.getUnitSpikeTrain(id))
            train2=np.sort(SX2.getUnitSpikeTrain(id))
            self.assertTrue(np.allclose(train1,train2))
 
if __name__ == '__main__':
    unittest.main()
