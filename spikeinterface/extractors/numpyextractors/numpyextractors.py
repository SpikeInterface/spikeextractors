from spikeinterface import InputExtractor
from spikeinterface import OutputExtractor

import numpy as np

class NumpyInputExtractor(InputExtractor):
    def __init__(self, *, timeseries, samplerate, geom=None):
        InputExtractor.__init__(self)
        self._timeseries=timeseries
        self._samplerate=samplerate
        self._geom=geom
        
    def getNumChannels(self):
        return self._timeseries.shape[0]+1
    
    def getNumFrames(self):
        return self._timeseries.shape[1]
    
    def getSamplingFrequency(self):
        return self._samplerate
        
    def getRawTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self.getNumFrames()
        if channel_ids is None:
            channel_ids=range(self.getNumChannels())
        recordings=self._timeseries[:,start_frame:end_frame][channel_ids,:]
        return recordings
    
    def getChannelInfo(self, channel_id):
        return dict(
            location=self._geom[channel_id,:]
        )

class NumpyOutputExtractor(OutputExtractor):
    def __init__(self):
        OutputExtractor.__init__(self)
        self._unit_ids=[]
        self._units={}

    def loadFromExtractor(output_extractor):
        ids=output_extractor.getUnitIds()
        for id in ids:
            self.addUnit(id,output_extractor.getUnitSpikeTrain(id))

    def addUnit(self,unit_id,times):
        self._unit_ids.append(unit_id)
        self._units[unit_id]=dict(times=times)
        
    def getUnitIds(self):
        return self._unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=np.Inf
        times=self._units[unit_id]['times']
        inds=np.where((start_frame<=times)&(times<end_frame))[0]
        return times[inds]
