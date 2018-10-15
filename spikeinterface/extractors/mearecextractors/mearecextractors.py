from spikeinterface import RecordingExtractor
from spikeinterface import SortingExtractor

import quantities as pq
import numpy as np
import os
from os.path import join
import h5py
import yaml, json

class MEArecRecordingExtractor(RecordingExtractor):
    def __init__(self, *, recording_folder=None, recording_file=None):
        RecordingExtractor.__init__(self)
        self._recording_folder = recording_folder
        self._recording_file = recording_file
        self._fs = None
        self._positions = None
        self._recordings = None
        
    def _initialize(self):
        rec_dict, info = load_recordings(recording_folder=self._recording_folder)
        self._fs  = info['recordings']['fs']*1000 #fs is in kHz
        self._recordings = rec_dict['recordings']
        for chan, pos in enumerate(rec_dict['positions']):
            self.setChannelProperty(chan, 'location', pos)

        
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
        rec_dict,  info = load_recordings(recording_folder=self._recording_folder)
        self._num_units = len(rec_dict['spiketrains'])
        self._unit_ids = range(self._num_units)
        self._spike_trains = rec_dict['spiketrains']
        self._fs  = info['recordings']['fs'] * pq.kHz #fs is in kHz

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


def load_recordings(recording_folder, verbose=False):
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
    if verbose:
        print("Loading recordings...")

    rec_dict = {}

    if os.path.isfile(join(recording_folder, 'recordings.npy')):
        recordings = np.load(join(recording_folder, 'recordings.npy'))
        rec_dict.update({'recordings': recordings})
    if os.path.isfile(join(recording_folder, 'positions.npy')):
        positions = np.load(join(recording_folder, 'positions.npy'))
        rec_dict.update({'positions': positions})
    if os.path.isfile(join(recording_folder, 'times.npy')):
        times = np.load(join(recording_folder, 'times.npy'))
        rec_dict.update({'times': times})
    if os.path.isfile(join(recording_folder, 'templates.npy')):
        templates = np.load(join(recording_folder, 'templates.npy'))
        rec_dict.update({'templates': templates})
    if os.path.isfile(join(recording_folder, 'spiketrains.npy')):
        spiketrains = np.load(join(recording_folder, 'spiketrains.npy'))
        rec_dict.update({'spiketrains': spiketrains})
    if os.path.isfile(join(recording_folder, 'sources.npy')):
        sources = np.load(join(recording_folder, 'sources.npy'))
        rec_dict.update({'sources': sources})
    if os.path.isfile(join(recording_folder, 'peaks.npy')):
        peaks = np.load(join(recording_folder, 'peaks.npy'))
        rec_dict.update({'peaks': peaks})

    with open(join(recording_folder, 'info.yaml'), 'r') as f:
        info = yaml.load(f)

    if verbose:
        print("Done loading recordings...")

    return rec_dict, info

def hdf5_to_recording(input_file,output_folder):
  if os.path.exists(output_folder):
    raise Exception('Output folder already exists: '+output_folder)

  os.mkdir(output_folder)

  with h5py.File(input_file,'r') as F:
    info=json.loads(str(F['info'][()]))
    with open(output_folder+'/info.yaml','w') as f:
      yaml.dump(info,f,default_flow_style=False)

    peaks=np.array(F.get('peaks'))
    np.save(output_folder+'/peaks.npy',peaks)
    positions=np.array(F.get('positions'))
    np.save(output_folder+'/positions.npy',positions)
    recordings=np.array(F.get('recordings'))
    np.save(output_folder+'/recordings.npy',recordings)
    sources=np.array(F.get('sources'))
    np.save(output_folder+'/sources.npy',sources)
    templates=np.array(F.get('templates'))
    np.save(output_folder+'/templates.npy',templates)
    times=np.array(F.get('times'))
    np.save(output_folder+'/times.npy',times)
    spiketrains=[]
    for ii in range(F.attrs['n_neurons']):
      times=np.array(F.get('spiketrains/{}/times'.format(ii)))
      t_stop=np.array(F.get('spiketrains/{}/t_stop'.format(ii)))
      annotations_str=str(F.get('spiketrains/{}/annotations'.format(ii))[()])
      annotations=json.loads(annotations_str)
      st=neo.core.SpikeTrain(
        times,
        t_stop=t_stop,
        units=quantities.s
      )
      st.annotations=annotations
      spiketrains.append(st)
    np.save(output_folder+'/spiketrains.npy',spiketrains)