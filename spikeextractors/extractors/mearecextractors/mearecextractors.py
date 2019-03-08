from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import numpy as np
from pathlib import Path
import json #pyyaml

def _load_required_modules():
    try:
        import neo
        import quantities as pq
        import h5py
        import yaml
    except ModuleNotFoundError:
        raise ModuleNotFoundError("To use the MEArec extractors, install neo, quantities, pyyaml, and h5py: \n\n"
                                  "pip install neo quantities pyyaml h5py\n\n")
    return neo, pq, h5py, yaml


class MEArecRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_path=None):
        RecordingExtractor.__init__(self)
        self._recording_path = recording_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._filehandle = None
        self._initialize()

    def __del__(self):
        if self._filehandle is not None:
            self._filehandle.close()

    def _initialize(self):
        rec_dict, info, fh = load_recordings(recordings=self._recording_path)

        self._filehandle = fh
        self._fs = info['recordings']['fs']
        self._recordings = rec_dict['recordings']
        for chan, pos in enumerate(rec_dict['channel_positions']):
            self.setChannelProperty(chan, 'location', pos)

    def getChannelIds(self):
        if self._recordings is None:
            self._initialize()
        return list(range(self._recordings.shape[0]))

    def getNumFrames(self):
        if self._recordings is None:
            self._initialize()
        return self._recordings.shape[1]

    def getSamplingFrequency(self):
        if self._fs is None:
            self._initialize()
        return self._fs

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if self._recordings is None:
            self._initialize()
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = range(self.getNumChannels())
        return self._recordings[channel_ids, start_frame:end_frame]

    @staticmethod
    def writeRecording(recording, save_path):
        neo, pq, h5py, yaml = _load_required_modules()
        save_path = Path(save_path)
        if save_path.is_dir():
            raise print("The file will be saved as recording.h5 in the provided folder")
            save_path = save_path / 'recording.h5'
        if save_path.suffix == '.h5' or save_path.suffix == '.hdf5':
            F = h5py.File(save_path, 'w')
            info = {'recordings': {'fs': recording.getSamplingFrequency()}}
            F.create_dataset('info', data=json.dumps(info))
            if 'location' in recording.getChannelPropertyNames():
                positions = np.array([recording.getChannelProperty(chan, 'location')
                                      for chan in range(recording.getNumChannels())])
                F.create_dataset('channel_positions', data=positions)
            F.create_dataset('recordings', data=recording.getTraces())
            F.create_dataset('times', data=np.arange(recording.getNumFrames() / recording.getSamplingFrequency()))
            F.close()
            print('\nSaved recordings in', save_path, '\n')
        else:
            raise Exception("Provide a folder or an .h5/.hdf5 as 'save_path'")


class MEArecSortingExtractor(SortingExtractor):
    def __init__(self, recording_path=None):
        neo, pq, h5py, yaml = _load_required_modules()
        SortingExtractor.__init__(self)
        self._recording_path = recording_path
        self._num_units = None
        self._spike_trains = None
        self._unit_ids = None
        self._fs = None
        self._initialize()

    def _initialize(self):
        neo, pq, h5py, yaml = _load_required_modules()
        rec_dict, info, fh = load_recordings(recordings=self._recording_path)
        fh.close() # not required as no raw data is read
        self._num_units = len(rec_dict['spiketrains'])
        if 'unit_id' in rec_dict['spiketrains'][0].annotations:
            self._unit_ids = [int(st.annotations['unit_id']) for st in rec_dict['spiketrains']]
        else:
            self._unit_ids = list(range(self._num_units))
        self._spike_trains = rec_dict['spiketrains']
        self._fs = info['recordings']['fs'] * pq.Hz  # fs is in kHz

        if 'soma_position' in self._spike_trains[0].annotations:
            for u, st in zip(self._unit_ids, self._spike_trains):
                self.setUnitProperty(u, 'soma_location', st.annotations['soma_position'])

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
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        if self._spike_trains is None:
            self._initialize()
        times = (self._spike_trains[self.getUnitIds().index(unit_id)].times.rescale('s') *
                 self._fs.rescale('Hz')).magnitude
        inds = np.where((start_frame <= times) & (times < end_frame))
        return np.rint(times[inds]).astype(int)

    @staticmethod
    def writeSorting(sorting, save_path, sampling_frequency):
        neo, pq, h5py, yaml = _load_required_modules()
        save_path = Path(save_path)
        if save_path.is_dir():
            raise print("The file will be saved as sorting.h5 in the provided folder")
            save_path = save_path / 'sorting.h5'
        if save_path.suffix == '.h5' or save_path.suffix == '.hdf5':
            F = h5py.File(save_path, 'w')
            info = {'recordings': {'fs': sampling_frequency, 'n_neurons': len(sorting.getUnitIds())}}
            F.create_dataset('info', data=json.dumps(info))
            for ii, unit in enumerate(sorting.getUnitIds()):
                st = sorting.getUnitSpikeTrain(unit) / float(sampling_frequency) * pq.s
                t_stop = np.max(sorting.getUnitSpikeTrain(unit)) / float(sampling_frequency) * pq.s
                F.create_dataset('spiketrains/{}/times'.format(unit), data=st)
                F.create_dataset('spiketrains/{}/t_stop'.format(unit), data=t_stop)
                annotations = {'unit_id': str(unit)}
                annotations_str = json.dumps(annotations)
                F.create_dataset('spiketrains/{}/annotations'.format(unit), data=annotations_str)
            F.close()
            print('\nSaved sorting in', save_path, '\n')
        else:
            raise Exception("Provide a folder or an .h5/.hdf5 as 'save_path'")


def load_recordings(recordings, verbose=False):
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
    neo, pq, h5py, yaml = _load_required_modules()
    if verbose:
        print("Loading recordings...")

    rec_dict = {}
    info = {}
    recordings = Path(recordings)
    if recordings.is_dir():
        raise Exception("Folders not supported for MEArec recordings")
    elif recordings.suffix == '.h5' or recordings.suffix == '.hdf5':
        F = h5py.File(recordings, 'r')
        info = json.loads(str(F['info'][()]))
        rec_dict['channel_positions'] = np.array(F.get('channel_positions'))
        rec_dict['recordings'] = F.get('recordings')
        spiketrains = []
        if 'n_neurons' in info['recordings']:
            if 'spiketrains' in F:
                ordered_keys = sorted([int(k) for k in F.get('spiketrains').keys()])
                ordered_keys = [str(k) for k in ordered_keys]
                for k in ordered_keys:
                    times = np.array(F.get('spiketrains/{}/times'.format(k))) * pq.s
                    t_stop = np.array(F.get('spiketrains/{}/t_stop'.format(k))) * pq.s
                    annotations_str = str(F.get('spiketrains/{}/annotations'.format(k))[()])
                    annotations = json.loads(annotations_str)
                    st = neo.core.SpikeTrain(
                        times,
                        t_stop=t_stop,
                        units=pq.s
                    )
                    st.annotations = annotations
                    spiketrains.append(st)
                rec_dict['spiketrains'] = spiketrains
    else:
        raise Exception("The provided file is not a MEArec recording")

    if verbose:
        print("Done loading recordings...")

    return rec_dict, info, F
