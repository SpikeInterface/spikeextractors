from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
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
    def writeRecording(recording, exdir_file, lfp=False, mua=False):
        exdir, pq = _load_required_modules()

        channel_ids = recording.getChannelIds()
        M = len(channel_ids)
        N = recording.getNumFrames()
        raw = recording.getTraces()
        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)

        if not lfp and not mua:
            timeseries = exdir_group.require_group('acquisition').require_dataset('timeseries', data=raw)
            timeseries.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
            return
        elif lfp:
            ephys = exdir_group.require_group('processing').require_group('electrophysiology')
            if 'group' in recording.getChannelPropertyNames():
                channel_groups = np.unique([recording.getChannelProperty(ch, 'group')
                                            for ch in recording.getChannelIds()])
            else:
                channel_groups  =[0]

            if len(channel_groups) == 1:
                chan = 0
                ch_group = ephys.require_group('Channel_group_' + str(chan))
                lfp_group = ch_group.require_group('LFP')
                ch_group.attrs['electrode_group_id'] = chan
                ch_group.attrs['electrode_identities'] = np.array(recording.getChannelIds())
                ch_group.attrs['electrode_idx'] = np.array(recording.getChannelIds())
                ch_group.attrs['start_time'] = 0 * pq.s
                ch_group.attrs['stop_time'] = recording.getNumFrames() / \
                                              float(recording.getSamplingFrequency()) * pq.s
                for i_c, ch in enumerate(recording.getChannelIds()):
                    ts_group = lfp_group.require_group('LFP_timeseries_' + str(ch))
                    ts_group.attrs['electrode_group_id'] = chan
                    ts_group.attrs['electrode_identity'] = ch
                    ts_group.attrs['num_samples'] = recording.getNumFrames()
                    ts_group.attrs['electrode_idx'] = i_c
                    ts_group.attrs['start_time'] = 0 * pq.s
                    ts_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                  float(recording.getSamplingFrequency()) * pq.s
                    ts_group.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                    data = ts_group.require_dataset('data', data=recording.getTraces(channel_ids=[ch]))
                    data.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                    data.attrs['unit'] = pq.uV
            else:
                channel_groups = np.unique([recording.getChannelProperty(ch, 'group')
                                            for ch in recording.getChannelIds()])
                for chan in channel_groups:
                    ch_group = ephys.require_group('Channel_group_' + str(chan))
                    lfp_group = ch_group.require_group('LFP')
                    ch_group.attrs['electrode_group_id'] = chan
                    ch_group.attrs['electrode_identities'] = np.array([i_c for i_c, ch in enumerate(recording.getChannelIds())
                                                               if recording.getChannelProperty(ch, 'group') == chan])
                    ch_group.attrs['electrode_idx'] = np.array([i_c for i_c, ch in enumerate(recording.getChannelIds())
                                                        if recording.getChannelProperty(ch, 'group') == chan])
                    ch_group.attrs['start_time'] = 0 * pq.s
                    ch_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                   float(recording.getSamplingFrequency()) * pq.s
                    for i_c, ch in enumerate(recording.getChannelIds()):
                        if recording.getChannelProperty(ch, 'group') == chan:
                            ts_group = lfp_group.require_group('LFP_timeseries_'+str(ch))
                            ts_group.attrs['electrode_group_id'] = chan
                            ts_group.attrs['electrode_identity'] = ch
                            ts_group.attrs['num_samples'] = recording.getNumFrames()
                            ts_group.attrs['electrode_idx'] = i_c
                            ts_group.attrs['start_time'] = 0 * pq.s
                            ts_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                          float(recording.getSamplingFrequency()) * pq.s
                            ts_group.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                            data = ts_group.require_dataset('data', data=recording.getTraces(channel_ids=[ch]))
                            data.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                            data.attrs['unit'] = pq.uV
            return
        elif mua:
            ephys = exdir_group.require_group('processing').require_group('electrophysiology')
            if 'group' in recording.getChannelPropertyNames():
                channel_groups = np.unique([recording.getChannelProperty(ch, 'group')
                                            for ch in recording.getChannelIds()])
            else:
                channel_groups  =[0]

            if len(channel_groups) == 1:
                chan = 0
                ch_group = ephys.require_group('Channel_group_' + str(chan))
                mua_group = ch_group.require_group('MUA')
                ch_group.attrs['electrode_group_id'] = chan
                ch_group.attrs['electrode_identities'] = np.array(recording.getChannelIds())
                ch_group.attrs['electrode_idx'] = np.array(recording.getChannelIds())
                ch_group.attrs['start_time'] = 0 * pq.s
                ch_group.attrs['stop_time'] = recording.getNumFrames() / \
                                              float(recording.getSamplingFrequency()) * pq.s
                for i_c, ch in enumerate(recording.getChannelIds()):
                    ts_group = mua_group.require_group('MUA_timeseries_' + str(ch))
                    ts_group.attrs['electrode_group_id'] = chan
                    ts_group.attrs['electrode_identity'] = ch
                    ts_group.attrs['num_samples'] = recording.getNumFrames()
                    ts_group.attrs['electrode_idx'] = i_c
                    ts_group.attrs['start_time'] = 0 * pq.s
                    ts_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                  float(recording.getSamplingFrequency()) * pq.s
                    ts_group.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                    data = ts_group.require_dataset('data', data=recording.getTraces(channel_ids=[ch]))
                    data.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                    data.attrs['unit'] = pq.uV
            else:
                channel_groups = np.unique([recording.getChannelProperty(ch, 'group')
                                            for ch in recording.getChannelIds()])
                for chan in channel_groups:
                    ch_group = ephys.require_group('Channel_group_' + str(chan))
                    mua_group = ch_group.require_group('MUA')
                    ch_group.attrs['electrode_group_id'] = chan
                    ch_group.attrs['electrode_identities'] = np.array([i_c for i_c, ch in enumerate(recording.getChannelIds())
                                                               if recording.getChannelProperty(ch, 'group') == chan])
                    ch_group.attrs['electrode_idx'] = np.array([i_c for i_c, ch in enumerate(recording.getChannelIds())
                                                        if recording.getChannelProperty(ch, 'group') == chan])
                    ch_group.attrs['start_time'] = 0 * pq.s
                    ch_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                   float(recording.getSamplingFrequency()) * pq.s
                    for i_c, ch in enumerate(recording.getChannelIds()):
                        if recording.getChannelProperty(ch, 'group') == chan:
                            ts_group = mua_group.require_group('MUA_timeseries_'+str(ch))
                            ts_group.attrs['electrode_group_id'] = chan
                            ts_group.attrs['electrode_identity'] = ch
                            ts_group.attrs['num_samples'] = recording.getNumFrames()
                            ts_group.attrs['electrode_idx'] = i_c
                            ts_group.attrs['start_time'] = 0 * pq.s
                            ts_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                          float(recording.getSamplingFrequency()) * pq.s
                            ts_group.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                            data = ts_group.require_dataset('data', data=recording.getTraces(channel_ids=[ch]))
                            data.attrs['sample_rate'] = recording.getSamplingFrequency() * pq.Hz
                            data.attrs['unit'] = pq.uV



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
    def writeSorting(sorting, exdir_file, recording=None, sample_rate=None):
        exdir, pq = _load_required_modules()

        if sample_rate is None and recording is None:
            raise Exception("Provide 'sample_rate' argument (Hz)")
        else:
            if recording is None:
                sample_rate = sample_rate * pq.Hz
            else:
                sample_rate = recording.getSamplingFrequency() * pq.Hz

        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)
        ephys = exdir_group.require_group('processing').require_group('electrophysiology')

        if 'group' in recording.getChannelPropertyNames():
            channel_groups = np.unique([sorting.getUnitProperty(unit, 'group') for unit in sorting.getUnitIds()])
        else:
            channel_groups = [0]

        if len(channel_groups) == 1:
            chan = 0
            print("Single group: ", chan)
            ch_group = ephys.require_group('Channel_group_' + str(chan))
            unittimes = ch_group.require_group('UnitTimes')
            eventwaveform = ch_group.require_group('EventWaveform')
            if recording is not None:
                ch_group.attrs['electrode_group_id'] = chan
                ch_group.attrs['electrode_identities'] = np.array([])
                ch_group.attrs['electrode_idx'] = np.array(recording.getChannelIds())
                ch_group.attrs['start_time'] = 0 * pq.s
                ch_group.attrs['stop_time'] = recording.getNumFrames() / \
                                              float(recording.getSamplingFrequency()) * pq.s
                unittimes.attrs['electrode_group_id'] = chan
                unittimes.attrs['electrode_identities'] = np.array([])
                unittimes.attrs['electrode_idx'] = np.array(recording.getChannelIds())
                unittimes.attrs['start_time'] = 0 * pq.s
                unittimes.attrs['stop_time'] = recording.getNumFrames() / \
                                               float(recording.getSamplingFrequency()) * pq.s
            nums = np.array([])
            timestamps = np.array([])
            waveforms = np.array([])
            for unit in sorting.getUnitIds():
                unit_group = unittimes.require_group(str(unit))
                unit_group.require_dataset('times',
                                           data=(sorting.getUnitSpikeTrain(unit).astype(float)
                                                 / sample_rate).rescale('s'))
                unit_group.attrs['cluster_group'] = 'unsorted'
                unit_group.attrs['group_id'] = chan
                unit_group.attrs['name'] = 'unit #' + str(unit)

                timestamps = np.concatenate((timestamps, (sorting.getUnitSpikeTrain(unit).astype(float)
                                                          / sample_rate).rescale('s')))
                nums = np.concatenate((nums, [unit] * len(sorting.getUnitSpikeTrain(unit))))

                if 'waveforms' in sorting.getUnitSpikeFeatureNames(unit):
                    if len(waveforms) == 0:
                        waveforms = sorting.getUnitSpikeFeatures(unit, 'waveforms')
                    else:
                        waveforms = np.vstack((waveforms, sorting.getUnitSpikeFeatures(unit, 'waveforms')))

            print("Saving eventwaveforms and clustering")
            if 'waveforms' in sorting.getUnitSpikeFeatureNames():
                waveform_ts = eventwaveform.require_group('waveform_timeseries')
                data = waveform_ts.require_dataset('data', data=waveforms)
                data.attrs['num_samples'] = len(waveforms)
                data.attrs['sample_rate'] = sample_rate
                data.attrs['unit'] = pq.dimensionless
                times = waveform_ts.require_dataset('timestamps', data=timestamps)
                times.attrs['num_samples'] = len(timestamps)
                times.attrs['unit'] = pq.s
                waveform_ts.attrs['electrode_group_id'] = chan
                if recording is not None:
                    waveform_ts.attrs['electrode_identities'] = np.array([])
                    waveform_ts.attrs['electrode_idx'] = np.array(recording.getChannelIds())
                    waveform_ts.attrs['start_time'] = 0 * pq.s
                    waveform_ts.attrs['stop_time'] = recording.getNumFrames() / \
                                                     float(recording.getSamplingFrequency()) * pq.s
                    waveform_ts.attrs['sample_rate'] = sample_rate
                    waveform_ts.attrs['sample_length'] = waveforms.shape[1]
                    waveform_ts.attrs['num_samples'] = len(waveforms)
            clustering = ephys.require_group('Channel_group_' + str(chan)).require_group('Clustering')
            ts = clustering.require_dataset('timestamps', data=timestamps * pq.s)
            ts.attrs['num_samples'] = len(timestamps)
            ts.attrs['unit'] = pq.s
            ns = clustering.require_dataset('nums', data=nums)
            ns.attrs['num_samples'] = len(nums)
            cn = clustering.require_dataset('cluster_nums', data=np.array(sorting.getUnitIds()))
            cn.attrs['num_samples'] = len(sorting.getUnitIds())
        else:
            channel_groups = np.unique([sorting.getUnitProperty(unit, 'group') for unit in sorting.getUnitIds()])
            for chan in channel_groups:
                print("Group: ", chan)
                ch_group = ephys.require_group('Channel_group_' + str(chan))
                unittimes = ch_group.require_group('UnitTimes')
                eventwaveform = ch_group.require_group('EventWaveform')
                if recording is not None:
                    unittimes.attrs['electrode_group_id'] = chan
                    unittimes.attrs['electrode_identities'] = np.array([])
                    unittimes.attrs['electrode_idx'] = np.array([ch for i_c, ch in enumerate(recording.getChannelIds())
                                                                 if recording.getChannelProperty(ch, 'group') == chan])
                    unittimes.attrs['start_time'] = 0 * pq.s
                    unittimes.attrs['stop_time'] = recording.getNumFrames() / \
                                                  float(recording.getSamplingFrequency()) * pq.s
                    ch_group.attrs['electrode_group_id'] = chan
                    ch_group.attrs['electrode_identities'] = np.array([i_c for i_c, ch in enumerate(recording.getChannelIds())
                                                                       if recording.getChannelProperty(ch, 'group') == chan])
                    ch_group.attrs['electrode_idx'] = np.array([i_c for i_c, ch in enumerate(recording.getChannelIds())
                                                       if recording.getChannelProperty(ch, 'group') == chan])
                    ch_group.attrs['start_time'] = 0 * pq.s
                    ch_group.attrs['stop_time'] = recording.getNumFrames() / \
                                                  float(recording.getSamplingFrequency()) * pq.s
                nums = np.array([])
                timestamps = np.array([])
                waveforms = np.array([])
                for unit in sorting.getUnitIds():
                    if sorting.getUnitProperty(unit, 'group') == chan:
                        print("Unit: ", unit)
                        unit_group = unittimes.require_group(str(unit))
                        unit_group.require_dataset('times',
                                                   data=(sorting.getUnitSpikeTrain(unit).astype(float)
                                                         / sample_rate).rescale('s'))
                        unit_group.attrs['cluster_group'] = 'unsorted'
                        unit_group.attrs['group_id'] = chan
                        unit_group.attrs['name'] = 'unit #' + str(unit)

                        timestamps = np.concatenate((timestamps, (sorting.getUnitSpikeTrain(unit).astype(float)
                                                         / sample_rate).rescale('s')))
                        nums = np.concatenate((nums, [unit]*len(sorting.getUnitSpikeTrain(unit))))

                    if 'waveforms' in sorting.getUnitSpikeFeatureNames(unit):
                        if len(waveforms) == 0:
                            waveforms = sorting.getUnitSpikeFeatures(unit, 'waveforms')
                        else:
                            waveforms = np.vstack((waveforms, sorting.getUnitSpikeFeatures(unit, 'waveforms')))
                print("Saving eventwaveforms and clustering")
                if 'waveforms' in sorting.getUnitSpikeFeatureNames():
                    waveform_ts = eventwaveform.require_group('waveform_timeseries')
                    data = waveform_ts.require_dataset('data', data=waveforms)
                    data.attrs['num_samples'] = len(waveforms)
                    data.attrs['sample_rate'] = sample_rate
                    data.attrs['unit'] = pq.dimensionless
                    times = waveform_ts.require_dataset('timestamps', data=timestamps)
                    times.attrs['num_samples'] = len(timestamps)
                    times.attrs['unit'] = pq.s
                    waveform_ts.attrs['electrode_group_id'] = chan
                    if recording is not None:
                        waveform_ts.attrs['electrode_identities'] = np.array([])
                        waveform_ts.attrs['electrode_idx'] = np.array([ch for i_c, ch in
                                                                       enumerate(recording.getChannelIds())
                                                                       if recording.getChannelProperty(ch, 'group') == chan])
                        waveform_ts.attrs['start_time'] = 0 * pq.s
                        waveform_ts.attrs['stop_time'] = recording.getNumFrames() / \
                                                       float(recording.getSamplingFrequency()) * pq.s
                        waveform_ts.attrs['sample_rate'] = sample_rate
                        waveform_ts.attrs['sample_length'] = waveforms.shape[1]
                        waveform_ts.attrs['num_samples'] = len(waveforms)
                clustering = ephys.require_group('Channel_group_' + str(chan)).require_group('Clustering')
                ts = clustering.require_dataset('timestamps', data=timestamps*pq.s)
                ts.attrs['num_samples'] = len(timestamps)
                ts.attrs['unit'] = pq.s
                ns = clustering.require_dataset('nums', data=nums)
                ns.attrs['num_samples'] = len(nums)
                cn = clustering.require_dataset('cluster_nums', data=np.array(sorting.getUnitIds()))
                cn.attrs['num_samples'] = len(sorting.getUnitIds())