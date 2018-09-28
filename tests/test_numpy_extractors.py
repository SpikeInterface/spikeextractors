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
        self.IX=si.NumpyInputExtractor(timeseries=X,samplerate=samplerate,geom=geom)
        self.OX=si.NumpyOutputExtractor()
        L=200
        self._train1=np.random.uniform(0,N,L)
        self.OX.addUnit(unit_id=1,times=self._train1)
        self.OX.addUnit(unit_id=2,times=np.random.uniform(0,N,L))
        self.OX.addUnit(unit_id=3,times=np.random.uniform(0,N,L))
        
    def tearDown(self):
        pass
     
    def test_input_extractor(self):
        # getNumChannels
        self.assertEqual(self.IX.getNumChannels(),self._X.shape[0])
        # getNumFrames
        self.assertEqual(self.IX.getNumFrames(),self._X.shape[1])
        # getSamplingFrequency
        self.assertEqual(self.IX.getSamplingFrequency(),self._samplerate)
        # getRawTraces
        self.assertTrue(np.allclose(self.IX.getRawTraces(),self._X))
        self.assertTrue(np.allclose(self.IX.getRawTraces(start_frame=0,end_frame=12,channel_ids=[0,3]),self._X[[0,3],0:12]))
        # getChannelInfo
        self.assertTrue(np.allclose(np.array(self.IX.getChannelInfo(channel_id=1)['location']),self._geom[1,:]))
        # timeToFrame / frameToTime
        self.assertEqual(self.IX.timeToFrame(12),12*self.IX.getSamplingFrequency())
        self.assertEqual(self.IX.frameToTime(12),12/self.IX.getSamplingFrequency())
        # getRawSnippets
        snippets=self.IX.getRawSnippets(snippet_len=20,center_frames=[0,30,50])
        self.assertTrue(np.allclose(snippets[1],self._X[:,20:40]))
    
    def test_output_extractor(self):
        unit_ids=[1,2,3]
        # getUnitIds
        self.assertEqual(self.OX.getUnitIds(),unit_ids)
        # getUnitSpikeTrain
        st=self.OX.getUnitSpikeTrain(unit_id=1)
        self.assertTrue(np.allclose(st,self._train1))
 
if __name__ == '__main__':
    unittest.main()