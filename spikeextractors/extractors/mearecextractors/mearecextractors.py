from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import quantities as pq
import numpy as np
import os
from os.path import join
import h5py
import yaml, json
import neo


class MEArecRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_path=None):
        RecordingExtractor.__init__(self)
        self._recording_path = recording_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._initialize()

    def _initialize(self):
        rec_dict, info = load_recordings(recordings=self._recording_path)

        self._fs = info['recordings']['fs']
        self._recordings = rec_dict['recordings']
        for chan, pos in enumerate(rec_dict['positions']):
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
        return self._recordings[channel_ids, :][:, start_frame:end_frame]

    @staticmethod
    def writeRecording(recording, save_path):
        if save_path.endswith('h5') or save_path.endswith('hdf5'):
            F = h5py.File(save_path, 'w')
            info = {'recordings': {'fs': recording.getSamplingFrequency()}}
            F.create_dataset('info', data=json.dumps(info))
            if 'location' in recording.getChannelPropertyNames():
                positions = np.array([recording.getChannelProperty(chan, 'location')
                                      for chan in range(recording.getNumChannels())])
                F.create_dataset('positions', data=positions)
            F.create_dataset('recordings', data=recording.getTraces())
            F.create_dataset('times', data=np.arange(recording.getNumFrames() / recording.getSamplingFrequency()))
            F.close()
            print('\nSaved recordings in', save_path, '\n')
        elif save_path is not None:
            save_folder = save_path
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            np.save(join(save_folder, 'recordings'), recording.getTraces())
            np.save(join(save_folder, 'times'), np.arange(recording.getNumFrames() / recording.getSamplingFrequency()))
            if 'location' in recording.getChannelPropertyNames():
                positions = np.array([recording.getChannelProperty(chan, 'location')
                                      for chan in range(recording.getNumChannels())])
                np.save(join(save_folder, 'positions'), positions)
            info = {'recordings': {'fs': recording.getSamplingFrequency()}}
            with open(join(save_folder, 'info.yaml'), 'w') as f:
                yaml.dump(info, f, default_flow_style=False)
            print('\nSaved recordings in', save_folder, ' folder\n')


class MEArecSortingExtractor(SortingExtractor):
    def __init__(self, recording_path=None):
        SortingExtractor.__init__(self)
        self._recording_path = recording_path
        self._num_units = None
        self._spike_trains = None
        self._unit_ids = None
        self._fs = None

    def _initialize(self):
        rec_dict, info = load_recordings(recordings=self._recording_path)

        self._num_units = len(rec_dict['spiketrains'])
        if 'unit_id' in rec_dict['spiketrains'][0].annotations:
            self._unit_ids = [st.annotations['unit_id'] for st in rec_dict['spiketrains']]
        else:
            self._unit_ids = list(range(self._num_units))
        self._spike_trains = rec_dict['spiketrains']
        self._fs = info['recordings']['fs'] * pq.Hz  # fs is in kHz

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
        if save_path.endswith('h5') or save_path.endswith('hdf5'):
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
                F.create_dataset('spiketrains/{}/annotations'.format(ii), data=annotations_str)
            F.close()
            print('\nSaved sorting in', save_path, '\n')
        elif save_path is not None:
            save_folder = save_path
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            spiketrains = []
            for ii, unit in enumerate(sorting.getUnitIds()):
                st = sorting.getUnitSpikeTrain(unit) / sampling_frequency * pq.s
                t_stop = np.max(sorting.getUnitSpikeTrain(unit)) / sampling_frequency * pq.s
                spiketrain = neo.core.SpikeTrain(times=st, t_start=0 * pq.s, t_stop=t_stop)
                spiketrain.annotate(unit_id=unit)
                spiketrains.append(spiketrain)
            info = {'recordings': {'fs': sampling_frequency, 'n_neurons': len(spiketrains)}}
            np.save(join(save_folder, 'spiketrains'), spiketrains)
            with open(join(save_folder, 'info.yaml'), 'w') as f:
                yaml.dump(info, f, default_flow_style=False)
            print('\nSaved sorting in', save_folder, ' folder\n')


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
    if verbose:
        print("Loading recordings...")

    rec_dict = {}
    info = {}

    if os.path.isdir(recordings):
        recording_folder = recordings
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
    elif recordings.endswith('h5') or recordings.endswith('hdf5'):
        with h5py.File(recordings, 'r') as F:
            info = json.loads(str(F['info'][()]))
            rec_dict['peaks'] = np.array(F.get('peaks'))
            rec_dict['positions'] = np.array(F.get('positions'))
            rec_dict['recordings'] = np.array(F.get('recordings'))
            rec_dict['sources'] = np.array(F.get('sources'))
            rec_dict['templates'] = np.array(F.get('templates'))
            rec_dict['times'] = np.array(F.get('times'))
            spiketrains = []
            if 'n_neurons' in info['recordings']:
                for ii in range(info['recordings']['n_neurons']):
                    times = np.array(F.get('spiketrains/{}/times'.format(ii))) * pq.s
                    t_stop = np.array(F.get('spiketrains/{}/t_stop'.format(ii))) * pq.s
                    annotations_str = str(F.get('spiketrains/{}/annotations'.format(ii))[()])
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
        raise Exception("The provided file-folder is not a MEArec recording")

    if verbose:
        print("Done loading recordings...")

    return rec_dict, info
