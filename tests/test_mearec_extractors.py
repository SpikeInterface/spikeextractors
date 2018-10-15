import unittest
import numpy as np
import tempfile
import shutil
import yaml
import os, sys
import neo
import quantities as pq

def append_to_path(dir0): # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)
append_to_path(os.getcwd()+'/..')
import spikeinterface as si

class TestMearecExtractors(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self._create_dataset()
        self.RX=si.MEArecRecordingExtractor(recording_path=self.test_dir+'/recordings')
        self.SX=si.MEArecSortingExtractor(recording_path=self.test_dir+'/recordings')

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def _create_dataset(self):
        M=32
        N=10000
        K=10
        L=150
        T=224
        fs = 30000 * pq.Hz
        duration = N/fs

        self.dataset=dict(
            num_channels=M,
            num_timepoints=N,
            num_events=L,
            num_units=K,
        )
        # create neo spike trains
        times = np.arange(N)
        recordings = np.random.randn(M, N)
        positions = np.random.randn(M, 3)
        templates = np.random.randn(K, T)
        peaks = np.random.randn(K, M)
        sources = np.random.randn(K, N)
        spiketrains = []
        labels = []
        for i in range(K):
            times = np.sort(np.random.uniform(0, N, np.random.randint(L))) / fs
            st = neo.SpikeTrain(times, t_start = 0 * pq.s, t_stop = N/fs)
            st.annotate(unit_id=i)
            st.waveforms = np.zeros((len(st), T))
            spiketrains.append(st)
            labels.append([i] * len(st))

        self.dataset.update({'recordings': recordings, 'spiketrains': spiketrains, 'positions': positions,
                             'templates': templates, 'peaks': peaks, 'sources': sources, 'times': times, 'fs': fs})

        rec_folder = self.test_dir+'/recordings'
        if not os.path.isdir(rec_folder):
            os.makedirs(rec_folder)
        np.save(os.path.join(rec_folder, 'recordings'), recordings)
        np.save(os.path.join(rec_folder, 'spiketrains'), spiketrains)
        np.save(os.path.join(rec_folder, 'times'), times)
        np.save(os.path.join(rec_folder, 'positions'), positions)
        np.save(os.path.join(rec_folder, 'templates'), templates)
        np.save(os.path.join(rec_folder, 'peaks'), peaks)
        np.save(os.path.join(rec_folder, 'sources'), sources)

        info = {'recordings':{'fs': float(fs.rescale('Hz').magnitude)}}

        with open(os.path.join(rec_folder, 'info.yaml'), 'w') as f:
            yaml.dump(info, f)


    def test_recording_extractor(self):
        X=self.dataset['recordings']
        # getNumChannels
        self.assertEqual(self.RX.getNumChannels(),self.dataset['num_channels'])
        # getNumFrames
        self.assertEqual(self.RX.getNumFrames(),self.dataset['num_timepoints'])
        # getSamplingFrequency
        self.assertEqual(self.RX.getSamplingFrequency(),30000)
        # getTraces
        self.assertTrue(np.allclose(self.RX.getTraces(),X))
        self.assertTrue(np.allclose(self.RX.getTraces(start_frame=0,end_frame=12,channel_ids=[0,3]),X[[0,3],0:12]))
        # getChannelInfo
        self.assertTrue(np.allclose(np.array(self.RX.getChannelProperty(1, 'location')),
                                    self.dataset['positions'][1,:]))

    def test_sorting_extractor(self):
        K=self.dataset['num_units']
        # getUnitIds
        self.assertEqual(self.SX.getUnitIds(), range(0,K))
        # getUnitSpikeTrain
        st = self.SX.getUnitSpikeTrain(unit_id=1)
        st2 = (self.dataset['spiketrains'][1].times.rescale('s') * self.dataset['fs'].rescale('Hz')).magnitude
        self.assertTrue(np.allclose(st,st2))

if __name__ == '__main__':
    unittest.main()
