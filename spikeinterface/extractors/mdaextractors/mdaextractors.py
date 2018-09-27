from spikeinterface import InputExtractor
from spikeinterface import OutputExtractor

from mountainlab_pytools import mlproc as mlp
from mountainlab_pytools import mdaio
import os, json
import numpy as np

class MdaInputExtractor(InputExtractor):
    def __init__(self, *, dataset_directory, download=True):
        InputExtractor.__init__(self)
        self._dataset_directory=dataset_directory
        timeseries0=dataset_directory+'/raw.mda'
        self._dataset_params=read_dataset_params(dataset_directory)
        self._samplerate=self._dataset_params['samplerate']
        verbose=is_url(timeseries0)
        if download:
            if verbose:
                print('Downloading file if needed: '+timeseries0)
            self._timeseries_path=mlp.realizeFile(timeseries0)
            if verbose:
                print('Done.')
        else:
            self._timeseries_path=mlp.locateFile(timeseries0)
        geom0=dataset_directory+'/geom.csv'
        self._geom_fname=mlp.realizeFile(geom0)
        self._geom=np.genfromtxt(self._geom_fname, delimiter=',')
        X=mdaio.DiskReadMda(self._timeseries_path)
        if self._geom.shape[0] != X.N1():
            raise Exception('Incompatible dimensions between geom.csv and timeseries file {} <> {}'.format(self._geom.shape[0],X.N1()))
        self._num_channels=X.N1()
        self._num_timepoints=X.N2()
        
    def getNumChannels(self):
        return self._num_channels
    
    def getNumFrames(self):
        return self._num_timepoints
    
    def getSamplingFrequency(self):
        return self._samplerate
        
    def getRawTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self.getNumFrames()
        if channel_ids is None:
            channel_ids=range(self.getNumChannels())
        X=mdaio.DiskReadMda(self._timeseries_path)
        recordings=X.readChunk(i1=0,i2=start_frame,N1=X.N1(),N2=end_frame-start_frame)
        recordings=recordings[channel_ids,:]
        return recordings
    
    def getChannelInfo(self, channel_id):
        return dict(
            location=self._geom[channel_id,:]
        )

class MdaOutputExtractor(OutputExtractor):
    def __init__(self, *, firings_file):
        OutputExtractor.__init__(self)
        verbose=is_url(firings_file)
        if verbose:
            print('Downloading file if needed: '+firings_file)
        self._firings_path=mlp.realizeFile(firings_file)
        if verbose:
            print('Done.')
        self._firings=mdaio.readmda(self._firings_path)
        self._times=self._firings[1,:]
        self._labels=self._firings[2,:]
        self._num_units=int(np.max(self._labels))
        
    def getUnitIds(self):
        return range(1,self._num_units+1)

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=np.Inf
        inds=np.where((self._labels==unit_id)&(start_frame<=self._times)&(self._times<end_frame))
        return self._times[inds]
    
def is_url(path):
    return path.startswith('http://') or path.startswith('https://') or path.startswith('kbucket://') or path.startswith('sha1://')
    
def read_dataset_params(dsdir):
    params_fname=mlp.realizeFile(dsdir+'/params.json')
    if not os.path.exists(params_fname):
        raise Exception('Dataset parameter file does not exist: '+params_fname)
    with open(params_fname) as f:
        return json.load(f)
