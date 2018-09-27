import unittest
from mountainlab_pytools import mdaio
import numpy as np
import tempfile
import shutil
import json
import os, sys
def append_to_path(dir0): # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)
append_to_path(os.getcwd()+'/..')
import spikeinterface as si
 
class TestMdaExtractors(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self._create_dataset()
        self.IX=si.MdaInputExtractor(dataset_directory=self.test_dir+'/dataset')
        self.OX=si.MdaOutputExtractor(firings_file=self.test_dir+'/dataset/firings_true.mda')
        
    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        
    def _create_dataset(self):
        M=7
        N=10000
        K=12
        L=150
        self.dataset=dict(
            num_channels=M,
            num_timepoints=N,
            num_events=L,
            num_units=K,
            raw=np.random.normal(0,1,(M,N)),
            times=np.sort(np.random.uniform(0,N,L)),
            labels=np.random.randint(1,K,L),
            geom=np.random.normal(0,1,(M,2))
        )
        self.dataset['labels'][0:K]=range(1,K+1) # ensure that each unit is realized at least once
        firings=np.zeros((3,L))
        firings[1,:]=self.dataset['times']
        firings[2,:]=self.dataset['labels']
        os.mkdir(self.test_dir+'/dataset')
        mdaio.writemda32(self.dataset['raw'],self.test_dir+'/dataset/raw.mda')
        mdaio.writemda64(firings,self.test_dir+'/dataset/firings_true.mda')
        params=dict(
            samplerate=30000,
            spike_sign=1
        )
        with open(self.test_dir+'/dataset/params.json','w') as f:
            json.dump(params,f)
        
        np.savetxt(self.test_dir+'/dataset/geom.csv', self.dataset['geom'], delimiter=',')
     
    def test_input_extractor(self):
        X=self.dataset['raw']
        # getNumChannels
        self.assertEqual(self.IX.getNumChannels(),self.dataset['num_channels'])
        # getNumFrames
        self.assertEqual(self.IX.getNumFrames(),self.dataset['num_timepoints'])
        # getSamplingFrequency
        self.assertEqual(self.IX.getSamplingFrequency(),30000)
        # getRawTraces
        self.assertTrue(np.allclose(self.IX.getRawTraces(),X))
        self.assertTrue(np.allclose(self.IX.getRawTraces(start_frame=0,end_frame=12,channel_ids=[0,3]),X[[0,3],0:12]))
        # getChannelInfo
        self.assertTrue(np.allclose(np.array(self.IX.getChannelInfo(channel_id=1)['location']),self.dataset['geom'][1,:]))
    
    def test_output_extractor(self):
        K=self.dataset['num_units']
        # getUnitIds
        self.assertEqual(self.OX.getUnitIds(),range(1,K+1))
        # getUnitSpikeTrain
        st=self.OX.getUnitSpikeTrain(unit_id=1)
        inds=np.where(self.dataset['labels']==1)[0]
        st2=self.dataset['times'][inds]
        self.assertTrue(np.allclose(st,st2))
 
if __name__ == '__main__':
    unittest.main()