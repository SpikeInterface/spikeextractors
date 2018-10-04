from spikeinterface import RecordingExtractor
from spikeinterface import SortingExtractor

import quantities as pq
import numpy as np
from os.path import join
import h5py

class MEArecRecordingExtractor(RecordingExtractor):
    def __init__(self, *, recording_folder=None, recording_file=None):
        RecordingExtractor.__init__(self)
        self._recording_folder = recording_folder
        self._recording_file = recording_file
        self._fs = None
        self._positions = None
        self._recordings = None
        
    def _initialize(self):
        recordings, times, positions, templates, spiketrains, sources, peaks, info = \
            load_recordings(recording_folder=self._recording_folder,recording_file=self._recording_file)
        self._fs  = info['General']['fs']*1000 #fs is in kHz
        self._positions = positions
        self._recordings = recordings
        
    def getNumChannels(self):
        if self._recordings is None:
            self._initialize()
        return self._recordings.shape[0]
    
    def getNumFrames(self):
        if self._recordings is None:
            self._initialize()
        return self._recordings.shape[1]
    
    def getSamplingFrequency(self):
        if self._fs is None:
            self._initialize()
        return self._fs
        
    def getTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if self._recordings is None:
            self._initialize()
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self.getNumFrames()
        if channel_ids is None:
            channel_ids=range(self.getNumChannels())
        return self._recordings[channel_ids,:][:,start_frame:end_frame]
    
    def getChannelInfo(self, channel_id):
        if self._positions is None:
            self._initialize()
        return dict(
            location=self._positions[channel_id,:]
        )

class MEArecSortingExtractor(SortingExtractor):
    def __init__(self, *, recording_folder=None, recording_file=None):
        SortingExtractor.__init__(self)
        self._recording_folder = recording_folder
        self._recording_file = recording_file
        self._num_units = None
        self._spike_trains = None
        self._unit_ids = None
        self._fs = None
        
    def _initialize(self):
        print(self._recording_file)
        recordings, times, positions, templates, spiketrains, sources, peaks, info = \
            load_recordings(recording_folder=self._recording_folder,recording_file=self._recording_file)
        self._num_units = len(spiketrains)
        self._unit_ids = range(self._num_units)
        self._spike_trains = spiketrains
        self._fs  = info['General']['fs'] * pq.kHz #fs is in kHz

    def getUnitIds(self):
        if self._unit_ids is None:
            self._initialize()
        return self._unit_ids
        
    def getNumUnits(self):
        if self._num_units is None:
            self._initialize()
        return self._num_units

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=np.Inf
        if self._spike_trains is None:
            self._initialize()
        times = (self._spike_trains[unit_id].times.rescale('s') * self._fs.rescale('Hz')).magnitude
        inds = np.where((start_frame<=times)&(times<end_frame))
        return times[inds]        
        
def load_recordings(*, recording_folder=None, recording_file=None):
    '''
    Load generated recordings (from template_gen.py)

    Parameters
    ----------
    recording_folder: recordings folder

    Returns
    -------
    recordings, times, positions, templates, spiketrains, sources, peaks - np.arrays
    info - dict

    '''
    import yaml
    if recording_folder:
        print("Loading recordings from folder...")
        recordings = np.load(join(recording_folder, 'recordings.npy'))
        positions = np.load(join(recording_folder, 'positions.npy'))
        times = np.load(join(recording_folder, 'times.npy'))
        templates = np.load(join(recording_folder, 'templates.npy'))
        spiketrains = np.load(join(recording_folder, 'spiketrains.npy'))
        sources = np.load(join(recording_folder, 'sources.npy'))
        peaks = np.load(join(recording_folder, 'peaks.npy'))

        with open(join(recording_folder, 'info.yaml'), 'r') as f:
            info = yaml.load(f)
    elif recording_file:
        print("Loading recordings from file...")
        with h5py.File(recording_file,'r') as F:
            recordings=np.array(F.get('recordings'))
            positions=np.array(F.get('positions'))
            times=np.array(F.get('times'))
            templates=np.array(F.get('templates'))
            n_neurons=F.attrs['n_neurons']
            spiketrains=[]
            for i in range(n_neurons):
                spiketrains.append(dict(
                    times=np.array(F.get('spiketrains/{}/times'.format(i)))
                ))
            sources=np.array(F.get('sources'))
            peaks=np.array(F.get('peaks.npy'))
            info=dict(
                General=dict(F.attrs)
            )
            
    if not isinstance(times, pq.Quantity):
        times = times * pq.ms

    print("Done loading recordings...")
    return recordings, times, positions, templates, spiketrains, sources, peaks, info
