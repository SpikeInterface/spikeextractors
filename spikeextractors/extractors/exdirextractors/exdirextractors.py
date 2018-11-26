from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import os, json
import numpy as np


def _load_required_modules():
    try:
        import exdir, quantities
    except ModuleNotFoundError:
        raise ModuleNotFoundError("To use the ExdirExtractors run:\n\n"
                                  "pip install exdir\n\n")
    return exdir, quantities


class ExdirRecordingExtractor(RecordingExtractor):
    def __init__(self, exdir_file):
        exdir, pq = _load_required_modules()

        RecordingExtractor.__init__(self)
        self._exdir_file = exdir_file
        exdir_group = exdir.File(exdir_file)

        self._recordings = exdir_group['acquisition']['timeseries']
        self._samplerate = self._recordings.attrs['sample_rate']

        self._num_channels = self._recordings.shape[0]
        self._num_timepoints = self._recordings.shape[1]

    def getChannelIds(self):
        return list(range(self._num_channels))

    def getNumFrames(self):
        return self._num_timepoints

    def getSamplingFrequency(self):
        return self._samplerate

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        return self._recordings[channel_ids, start_frame:end_frame]

    @staticmethod
    def writeRecording(recording, exdir_file):
        exdir, pq = _load_required_modules()

        channel_ids = recording.getChannelIds()
        M = len(channel_ids)
        N = recording.getNumFrames()
        raw = recording.getTraces()
        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)
        timeseries = exdir_group.require_group('acquisition').require_dataset('timeseries', data=raw)
        timeseries.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz


class ExdirSortingExtractor(SortingExtractor):
    def __init__(self, exdir_file, sample_rate=None):
        exdir, pq = _load_required_modules()

        SortingExtractor.__init__(self)
        self._exdir_file = exdir_file
        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)

        if 'acquisition' in exdir_group.keys():
            if 'timeseries' in exdir_group['acquisition'].keys():
                sample_rate = exdir_group['acquisition']['timeseries'].attrs['sample_rate']
        else:
            if sample_rate is None:
                raise Exception("Provide 'sample_rate' argument (Hz)")
            else:
                sample_rate = sample_rate * pq.Hz

        electrophysiology = exdir_group['processing']['electrophysiology']
        self._unit_ids = []
        current_unit = 1
        self._spike_trains = []
        for chan_name, channel in electrophysiology.items():
            group = int(chan_name.split('_')[-1])
            for units, unit_times in channel['UnitTimes'].items():
                self._unit_ids.append(current_unit)
                self._spike_trains.append((unit_times['times'].data.rescale('s')*sample_rate).magnitude)
                self.setUnitProperty(current_unit, 'group', group)
                current_unit += 1

    def getUnitIds(self):
        return self._unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spike_trains[unit_id]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return np.rint(times[inds]).astype(int)

    @staticmethod
    def writeSorting(sorting, exdir_file, sample_rate=None):
        exdir, pq = _load_required_modules()

        if sample_rate is None:
            raise Exception("Provide 'sample_rate' argument (Hz)")
        else:
            sample_rate = sample_rate * pq.Hz

        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)
        ephys = exdir_group.require_group('processing').require_group('electrophysiology')
        if 'group' not in sorting.getUnitPropertyNames():
            unittimes = ephys.require_group('Channel_group_0').require_group('UnitTimes')
            for unit in sorting.getUnitIds():
                unit_group = unittimes.require_group(str(unit))
                unit_group.require_dataset('times',
                                           data=(sorting.getUnitSpikeTrain(unit).astype(float)
                                                 /sample_rate).rescale('s'))
        else:
            channel_groups = np.unique([sorting.getUnitProperty(unit, 'group') for unit in sorting.getUnitIds()])
            for chan in channel_groups:
                unittimes = ephys.require_group('Channel_group_'+str(chan)).require_group('UnitTimes')
                for unit in sorting.getUnitIds():
                    if sorting.getUnitProperty(unit, 'group') == chan:
                        unit_group = unittimes.require_group(str(unit))
                        unit_group.require_dataset('times',
                                                   data=(sorting.getUnitSpikeTrain(unit).astype(float)
                                                         / sample_rate).rescale('s'))
