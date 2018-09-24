from spikeinterface import InputExtractor
from spikeinterface import OutputExtractor

import quantities as pq
import numpy as np
from os.path import join

class MEArecInputExtractor(InputExtractor):
    def __init__(self, *, recording_folder):
        InputExtractor.__init__(self)
        self.recording_folder = recording_folder
        self.fs = None
        self.positions = None
        self.recordings = None
        
    def _initialize(self):
        recordings, times, positions, templates, spiketrains, sources, peaks, info = \
            load_recordings(self.recording_folder)
        self.fs  =info['General']['fs']
        self.positions = positions
        self.recordings = recordings
        
    def getNumChannels(self):
        if self.recordings is None:
            self._initialize()
        return self.recordings.shape[0]
    
    def getNumFrames(self):
        if self.recordings is None:
            self._initialize()
        return self.recordings.shape[1]
    
    def getSamplingFrequency(self):
        if self.fs is None:
            self._initialize()
        return self.fs * pq.kHz
        
    def getRawTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if self.recordings is None:
            self._initialize()
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self.getNumFrames()
        if channel_ids is None:
            channel_ids=range(self.getNumChannels())
        return recordings[channel_ids,:]
    
    def getChannelInfo(self, channel_id):
        if self.positions is None:
            self._initialize()
        return dict(
            location=self.positions[channel_id,:]
        )

class MEArecOutputExtractor(OutputExtractor):
    def __init__(self, *, recording_folder):
        OutputExtractor.__init__(self)
        self.recording_folder = recording_folder
        self.numUnits = None
        self.spike_trains = None
        
    def _initialize(self):
        recordings, times, positions, templates, spiketrains, sources, peaks, info = \
            load_recordings(self.recording_folder)
        self.numUnits = len(spiketrains)
        
    def getNumUnits(self):
        if self.numUnits is None:
            self._initialize()
        return self.numUnits

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=np.Inf
        if self.spike_trains is None:
            self._initialize()
        times=self.spike_trains[unit_id].times
        inds=np.where((start_frame<=times)&(times<end_frame))
        return times[inds]        
        
def load_recordings(recording_folder):
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
    print("Loading recordings...")

    recordings = np.load(join(recording_folder, 'recordings.npy'))
    positions = np.load(join(recording_folder, 'positions.npy'))
    times = np.load(join(recording_folder, 'times.npy'))
    templates = np.load(join(recording_folder, 'templates.npy'))
    spiketrains = np.load(join(recording_folder, 'spiketrains.npy'))
    sources = np.load(join(recording_folder, 'sources.npy'))
    peaks = np.load(join(recording_folder, 'peaks.npy'))

    with open(join(recording_folder, 'info.yaml'), 'r') as f:
        info = yaml.load(f)

    if not isinstance(times, pq.Quantity):
        times = times * pq.ms

    print("Done loading recordings...")
    return recordings, times, positions, templates, spiketrains, sources, peaks, info