import unittest
import numpy as np
import spikeextractors as se
import os
import tempfile
import shutil
from pathlib import Path

class TestNeuroscopeSortingExtractors(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_keep_mua_option(self):


        recording, sorting_gt = se.example_datasets.toy_example(num_channels=4, duration=10, seed=0)
        
        # Now, re-generate the data but force spikeInds to go from 0 to 7 instead of 1 to 8,
        # which is the actual Neuroscope format.
        #
        # Additionally, the 0 index group consists of unsorted spikes
        # while the 1 index group consists of multi-unit recordings as well...

        # Generate test data
        np.random.seed(1234)

        nUnits = 8;
        avgFiringRate = 10; # Hz
        recordingLength = 60; # in seconds

        # Generate spike times
        numSpikes = np.random.poisson(avgFiringRate*nUnits);
        spikeTimes = np.random.uniform(0,recordingLength,numSpikes);
        spikeTimes.sort(); # .res format expects total spikes to be in order

        # Generate indices of spikes
        spikeInds = np.random.randint(0,nUnits,numSpikes);
        
        # Save test data
        f1 = open(Path(self.test_dir) / "TestData.res","w+")
        f2 = open(Path(self.test_dir) / "TestData.clu","w+")
        f2.write("%d\n" % nUnits)
        for j in range(numSpikes):
            f1.write("%d\n" % spikeTimes[j])
            f2.write("%d\n" % spikeInds[j])

        f1.close() 
        f2.close()
        
        testNSE = se.NeuroscopeSortingExtractor(Path(self.test_dir) / 'TestData.res', Path(self.test_dir) / 'TestData.clu')
        testNSE_no_mua = se.NeuroscopeSortingExtractor(Path(self.test_dir) / 'TestData.res', Path(self.test_dir) /'TestData.clu', keep_mua_units=False)

        #os.remove('./TestData.res')
        #os.remove('./TestData.clu')
        
        units_ids = testNSE.get_unit_ids()
        no_mua_units_ids = testNSE_no_mua.get_unit_ids()
        
        self.assertEqual(list(units_ids), list(range(1,nUnits)))
        self.assertEqual(list(no_mua_units_ids), list(range(1,nUnits-1)))


if __name__ == '__main__':
    unittest.main()
