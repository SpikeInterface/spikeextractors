import numpy as np
import os, sys
import unittest
def append_to_path(dir0): # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)
append_to_path(os.getcwd()+'/..')
import spikeinterface as si
 
class TestNumpyExtractors(unittest.TestCase):
    def setUp(self):
        M=4
        N=10000
        samplerate=30000
        X=np.random.normal(0,1,(M,N))
        geom=np.random.normal(0,1,(M,2))
        self._X=X
        self._geom=geom
        self._samplerate=samplerate
        self.RX=si.NumpyRecordingExtractor(timeseries=X,samplerate=samplerate,geom=geom)
        self.SX=si.NumpySortingExtractor()
        L=200
        self._train1=np.random.uniform(0,N,L)
        self.SX.addUnit(unit_id=1,times=self._train1)
        self.SX.addUnit(unit_id=2,times=np.random.uniform(0,N,L))
        self.SX.addUnit(unit_id=3,times=np.random.uniform(0,N,L))
        
    def tearDown(self):
        pass
     
    def test_recording_extractor(self):
        # getNumChannels
        self.assertEqual(self.RX.getNumChannels(),self._X.shape[0])
        # getNumFrames
        self.assertEqual(self.RX.getNumFrames(),self._X.shape[1])
        # getSamplingFrequency
        self.assertEqual(self.RX.getSamplingFrequency(),self._samplerate)
        # getTraces
        self.assertTrue(np.allclose(self.RX.getTraces(),self._X))
        self.assertTrue(np.allclose(self.RX.getTraces(start_frame=0,end_frame=12,channel_ids=[0,3]),self._X[[0,3],0:12]))
        # getChannelInfo
        self.assertTrue(np.allclose(np.array(self.RX.getChannelInfo(channel_id=1)['location']),self._geom[1,:]))
        # timeToFrame / frameToTime
        self.assertEqual(self.RX.timeToFrame(12),12*self.RX.getSamplingFrequency())
        self.assertEqual(self.RX.frameToTime(12),12/self.RX.getSamplingFrequency())
        # getSnippets
        snippets=self.RX.getSnippets(snippet_len=20,center_frames=[0,30,50])
        self.assertTrue(np.allclose(snippets[1],self._X[:,20:40]))
    
    def test_sorting_extractor(self):
        unit_ids=[1,2,3]
        # getUnitIds
        self.assertEqual(self.SX.getUnitIds(),unit_ids)
        # getUnitSpikeTrain
        st=self.SX.getUnitSpikeTrain(unit_id=1)
        self.assertTrue(np.allclose(st,self._train1))
 
if __name__ == '__main__':
    unittest.main()
