from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import quantities as pq
import numpy as np
from pathlib import Path
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
        return self._recordings[channel_ids, :][:, start_frame:end_frame]

    @staticmethod
    def writeRecording(recording, save_path):
        save_path = Path(save_path)
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
        elif save_path is not None:
            save_folder = save_path
            if not save_folder.is_dir():
                save_folder.mkdir()
            np.save(save_folder / 'recordings', recording.getTraces())
            np.save(save_folder / 'timestamps', np.arange(recording.getNumFrames() / recording.getSamplingFrequency()))
            if 'location' in recording.getChannelPropertyNames():
                positions = np.array([recording.getChannelProperty(chan, 'location')
                                      for chan in range(recording.getNumChannels())])
                np.save(save_folder / 'channel_positions', positions)
            info = {'recordings': {'fs': recording.getSamplingFrequency()}}
            with (save_folder /'info.yaml').open('w') as f:
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
        self._initialize()

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
        save_path = Path(save_path)
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
                F.create_dataset('spiketrains/{}/annotations'.format(ii), data=annotations_str)
            F.close()
            print('\nSaved sorting in', save_path, '\n')
        elif save_path is not None:
            save_folder = save_path
            if not save_folder.is_dir():
                save_folder.mkdir()
            spiketrains = []
            for ii, unit in enumerate(sorting.getUnitIds()):
                st = sorting.getUnitSpikeTrain(unit) / sampling_frequency * pq.s
                t_stop = np.max(sorting.getUnitSpikeTrain(unit)) / sampling_frequency * pq.s
                spiketrain = neo.core.SpikeTrain(times=st, t_start=0 * pq.s, t_stop=t_stop)
                spiketrain.annotate(unit_id=unit)
                spiketrains.append(spiketrain)
            info = {'recordings': {'fs': sampling_frequency, 'n_neurons': len(spiketrains)}}
            np.save(save_folder / 'spiketrains', spiketrains)
            with (save_folder / 'info.yaml').open('w') as f:
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
    recordings = Path(recordings)
    if recordings.is_dir():
        recording_folder = recordings
        if (recording_folder /'recordings.npy').is_file():
            recordings = np.load(recording_folder /'recordings.npy')
            rec_dict.update({'recordings': recordings})
        if (recording_folder / 'channel_positions.npy').is_file():
            channel_positions = np.load(recording_folder / 'channel_positions.npy')
            rec_dict.update({'channel_positions': channel_positions})
        if (recording_folder / 'timestamps.npy').is_file():
            timestamps = np.load(recording_folder / 'timestamps.npy')
            rec_dict.update({'timestamps': timestamps})
        if (recording_folder / 'templates.npy').is_file():
            templates = np.load(recording_folder / 'templates.npy')
            rec_dict.update({'templates': templates})
        if (recording_folder / 'spiketrains.npy').is_file():
            spiketrains = np.load(recording_folder / 'spiketrains.npy')
            rec_dict.update({'spiketrains': spiketrains})
        if (recording_folder / 'spike_traces.npy').is_file():
            spike_traces = np.load(recording_folder / 'spike_traces.npy')
            rec_dict.update({'spike_traces': spike_traces})
        if (recording_folder / 'voltage_peaks.npy').is_file():
            voltage_peaks = np.load(recording_folder / 'voltage_peaks.npy')
            rec_dict.update({'voltage_peaks': voltage_peaks})
        with (recording_folder / 'info.yaml').open('r') as f:
            info = yaml.load(f)
    elif recordings.suffix == '.h5' or recordings.suffix == '.hdf5':
        with h5py.File(recordings, 'r') as F:
            info = json.loads(str(F['info'][()]))
            rec_dict['voltage_peaks'] = np.array(F.get('voltage_peaks'))
            rec_dict['channel_positions'] = np.array(F.get('channel_positions'))
            rec_dict['recordings'] = np.array(F.get('recordings'))
            rec_dict['spike_traces'] = np.array(F.get('spike_traces'))
            rec_dict['templates'] = np.array(F.get('templates'))
            rec_dict['timestamps'] = np.array(F.get('timestamps'))
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
