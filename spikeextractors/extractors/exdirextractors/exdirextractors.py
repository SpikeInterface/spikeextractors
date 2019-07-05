from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
import numpy as np

try:
    import exdir
    import exdir.plugins.quantities
    import quantities as pq
    HAVE_EXDIR = True
except ImportError:
    HAVE_EXDIR = False


class ExdirRecordingExtractor(RecordingExtractor):
    extractor_name = 'ExdirRecordingExtractor'
    installed = HAVE_EXDIR  # check at class level if installed or not
    _gui_params = [
        {'name': 'exdir_file', 'type': 'path', 'title': "Path to file"},
        {'name': 'probe_path', 'type': 'path', 'title': "Path to probe file (.csv or .prb)"},
    ]
    installation_mesg = "To use the ExdirExtractors run:\n\n pip install exdir\n\n"  # error message when not installed

    def __init__(self, exdir_file):
        assert HAVE_EXDIR, "To use the ExdirExtractors run:\n\n pip install exdir\n\n"
        RecordingExtractor.__init__(self)
        self._exdir_file = exdir_file
        exdir_group = exdir.File(exdir_file, plugins=[exdir.plugins.quantities])

        self._recordings = exdir_group['acquisition']['timeseries']
        self._samplerate = float(self._recordings.attrs['sample_rate'].rescale('Hz').magnitude)

        self._num_channels = self._recordings.shape[0]
        self._num_timepoints = self._recordings.shape[1]

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_timepoints

    def get_sampling_frequency(self):
        return self._samplerate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        return self._recordings.data[np.array(channel_ids), start_frame:end_frame]

    @staticmethod
    def write_recording(recording, exdir_file, lfp=False, mua=False):
        assert HAVE_EXDIR, "To use the ExdirExtractors run:\n\n pip install exdir\n\n"
        channel_ids = recording.get_channel_ids()
        raw = recording.get_traces()
        exdir_group = exdir.File(exdir_file, plugins=[exdir.plugins.quantities])

        if not lfp and not mua:
            acq = exdir_group.require_group('acquisition')
            timeseries = acq.require_dataset('timeseries', data=raw)
            timeseries.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
            timeseries.attrs['electrode_identities'] = np.array(channel_ids)
            return
        elif lfp:
            ephys = exdir_group.require_group('processing').require_group('electrophysiology')
            ephys.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
            if 'group' in recording.get_channel_property_names():
                channel_groups = np.unique([recording.get_channel_property(ch, 'group')
                                            for ch in recording.get_channel_ids()])
            else:
                channel_groups = [0]

            if len(channel_groups) == 1:
                chan = 0
                ch_group = ephys.require_group('channel_group_' + str(chan))
                lfp_group = ch_group.require_group('LFP')
                ch_group.attrs['electrode_group_id'] = chan
                ch_group.attrs['electrode_identities'] = np.array(recording.get_channel_ids())
                ch_group.attrs['electrode_idx'] = np.arange(len(recording.get_channel_ids()))
                ch_group.attrs['start_time'] = 0 * pq.s
                ch_group.attrs['stop_time'] = recording.get_num_frames() / \
                                              float(recording.get_sampling_frequency()) * pq.s
                for i_c, ch in enumerate(recording.get_channel_ids()):
                    ts_group = lfp_group.require_group('LFP_timeseries_' + str(ch))
                    ts_group.attrs['electrode_group_id'] = chan
                    ts_group.attrs['electrode_identity'] = ch
                    ts_group.attrs['num_samples'] = recording.get_num_frames()
                    ts_group.attrs['electrode_idx'] = i_c
                    ts_group.attrs['start_time'] = 0 * pq.s
                    ts_group.attrs['stop_time'] = recording.get_num_frames() / \
                                                  float(recording.get_sampling_frequency()) * pq.s
                    ts_group.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                    data = ts_group.require_dataset('data', data=recording.get_traces(channel_ids=[ch]))
                    data.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                    data.attrs['unit'] = pq.uV
            else:
                channel_groups = np.unique([recording.get_channel_property(ch, 'group')
                                            for ch in recording.get_channel_ids()])
                for chan in channel_groups:
                    ch_group = ephys.require_group('channel_group_' + str(chan))
                    lfp_group = ch_group.require_group('LFP')
                    ch_group.attrs['electrode_group_id'] = chan
                    ch_group.attrs['electrode_identities'] = np.array([ch for ch in recording.get_channel_ids()
                                                                       if recording.get_channel_property(ch, 'group')
                                                                       == chan])
                    ch_group.attrs['electrode_idx'] = np.array([i_c for i_c, ch in
                                                                enumerate(recording.get_channel_ids())
                                                                if recording.get_channel_property(ch, 'group') == chan])
                    ch_group.attrs['start_time'] = 0 * pq.s
                    ch_group.attrs['stop_time'] = recording.get_num_frames() / \
                                                   float(recording.get_sampling_frequency()) * pq.s
                    for i_c, ch in enumerate(recording.get_channel_ids()):
                        if recording.get_channel_property(ch, 'group') == chan:
                            ts_group = lfp_group.require_group('LFP_timeseries_'+str(ch))
                            ts_group.attrs['electrode_group_id'] = chan
                            ts_group.attrs['electrode_identity'] = ch
                            ts_group.attrs['num_samples'] = recording.get_num_frames()
                            ts_group.attrs['electrode_idx'] = i_c
                            ts_group.attrs['start_time'] = 0 * pq.s
                            ts_group.attrs['stop_time'] = recording.get_num_frames() / \
                                                          float(recording.get_sampling_frequency()) * pq.s
                            ts_group.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                            data = ts_group.require_dataset('data', data=recording.get_traces(channel_ids=[ch]))
                            data.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                            data.attrs['unit'] = pq.uV
            return
        elif mua:
            ephys = exdir_group.require_group('processing').require_group('electrophysiology')
            ephys.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
            if 'group' in recording.get_channel_property_names():
                channel_groups = np.unique([recording.get_channel_property(ch, 'group')
                                            for ch in recording.get_channel_ids()])
            else:
                channel_groups  =[0]

            if len(channel_groups) == 1:
                chan = 0
                ch_group = ephys.require_group('channel_group_' + str(chan))
                mua_group = ch_group.require_group('MUA')
                ch_group.attrs['electrode_group_id'] = chan
                ch_group.attrs['electrode_identities'] = np.array(recording.get_channel_ids())
                ch_group.attrs['electrode_idx'] = np.arange(len(recording.get_channel_ids()))
                ch_group.attrs['start_time'] = 0 * pq.s
                ch_group.attrs['stop_time'] = recording.get_num_frames() / \
                                              float(recording.get_sampling_frequency()) * pq.s
                for i_c, ch in enumerate(recording.get_channel_ids()):
                    ts_group = mua_group.require_group('MUA_timeseries_' + str(ch))
                    ts_group.attrs['electrode_group_id'] = chan
                    ts_group.attrs['electrode_identity'] = ch
                    ts_group.attrs['num_samples'] = recording.get_num_frames()
                    ts_group.attrs['electrode_idx'] = i_c
                    ts_group.attrs['start_time'] = 0 * pq.s
                    ts_group.attrs['stop_time'] = recording.get_num_frames() / \
                                                  float(recording.get_sampling_frequency()) * pq.s
                    ts_group.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                    data = ts_group.require_dataset('data', data=recording.get_traces(channel_ids=[ch]))
                    data.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                    data.attrs['unit'] = pq.uV
            else:
                channel_groups = np.unique([recording.get_channel_property(ch, 'group')
                                            for ch in recording.get_channel_ids()])
                for chan in channel_groups:
                    ch_group = ephys.require_group('channel_group_' + str(chan))
                    mua_group = ch_group.require_group('MUA')
                    ch_group.attrs['electrode_group_id'] = chan
                    ch_group.attrs['electrode_identities'] = np.array([ch for ch in recording.get_channel_ids()
                                                                       if recording.get_channel_property(ch, 'group')
                                                                       == chan])
                    ch_group.attrs['electrode_idx'] = np.array([i_c for i_c, ch in
                                                                enumerate(recording.get_channel_ids())
                                                                if recording.get_channel_property(ch, 'group') == chan])
                    ch_group.attrs['start_time'] = 0 * pq.s
                    ch_group.attrs['stop_time'] = recording.get_num_frames() / \
                                                   float(recording.get_sampling_frequency()) * pq.s
                    for i_c, ch in enumerate(recording.get_channel_ids()):
                        if recording.get_channel_property(ch, 'group') == chan:
                            ts_group = mua_group.require_group('MUA_timeseries_'+str(ch))
                            ts_group.attrs['electrode_group_id'] = chan
                            ts_group.attrs['electrode_identity'] = ch
                            ts_group.attrs['num_samples'] = recording.get_num_frames()
                            ts_group.attrs['electrode_idx'] = i_c
                            ts_group.attrs['start_time'] = 0 * pq.s
                            ts_group.attrs['stop_time'] = recording.get_num_frames() / \
                                                          float(recording.get_sampling_frequency()) * pq.s
                            ts_group.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                            data = ts_group.require_dataset('data', data=recording.get_traces(channel_ids=[ch]))
                            data.attrs['sample_rate'] = recording.get_sampling_frequency() * pq.Hz
                            data.attrs['unit'] = pq.uV


class ExdirSortingExtractor(SortingExtractor):

    extractor_name = 'ExdirSortingExtractor'
    installed = HAVE_EXDIR  # check at class level if installed or not
    _gui_params = [
        {'name': 'exdir_file', 'type': 'file_path', 'title': "str, Path to file"},
        {'name': 'sample_rate', 'type': 'float', 'title': "Sampling rate of recording. "
                                                          "It will be overwritten if exdir_file contains this info."},
        {'name': 'channel_group', 'type': 'int', 'title': "Channel group to load spike trains from."},
        {'name': 'load_waveforms', 'type': 'bool', 'title': "if True, waveforms are loaded."},
    ]
    installation_mesg = "To use the ExdirExtractors run:\n\n pip install exdir\n\n"  # error message when not installed

    def __init__(self, exdir_file, sample_rate=None, channel_group=None, load_waveforms=False):
        assert HAVE_EXDIR, "To use the ExdirExtractors run:\n\n pip install exdir\n\n"
        SortingExtractor.__init__(self)
        self._exdir_file = exdir_file
        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)

        electrophysiology = None
        if 'processing' in exdir_group.keys():
            if 'electrophysiology' in exdir_group['processing']:
                electrophysiology = exdir_group['processing']['electrophysiology']
                ephys_attrs = electrophysiology.attrs
                if 'sample_rate' in ephys_attrs:
                    sample_rate = ephys_attrs['sample_rate']
        else:
            if sample_rate is None:
                raise Exception("Sampling rate information not found. Please provide it wiht the 'sample_rate' "
                                "argument")
            else:
                sample_rate = sample_rate * pq.Hz
        self._sampling_frequency = float(sample_rate.rescale('Hz').magnitude)

        if electrophysiology is None:
            raise Exception("'electrophysiology' group not found!")

        self._unit_ids = []
        current_unit = 1
        self._spike_trains = []
        for chan_name, channel in electrophysiology.items():
            if 'channel' in chan_name:
                group = int(chan_name.split('_')[-1])
                if channel_group is not None:
                    if group != channel_group:
                        continue
                if load_waveforms:
                    if 'Clustering' in channel.keys() and 'EventWaveform' in channel.keys():
                        clustering = channel.require_group('Clustering')
                        eventwaveform = channel.require_group('EventWaveform')
                        nums = clustering['nums'].data
                        waveforms = eventwaveform.require_group('waveform_timeseries')['data'].data
                if 'UnitTimes' in channel.keys():
                    for unit, unit_times in channel['UnitTimes'].items():
                        self._unit_ids.append(current_unit)
                        self._spike_trains.append((unit_times['times'].data.rescale('s')*sample_rate).magnitude)
                        attrs = unit_times.attrs
                        for k, v in attrs.items():
                            self.set_unit_property(current_unit, k, v)
                        if load_waveforms:
                            unit_idxs = np.where(nums == int(unit))
                            wf = waveforms[unit_idxs]
                            self.set_unit_spike_features(current_unit, 'waveforms', wf)
                        current_unit += 1

    def get_unit_ids(self):
        return self._unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spike_trains[self._unit_ids.index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return np.rint(times[inds]).astype(int)

    @staticmethod
    def write_sorting(sorting, exdir_file, recording=None, sample_rate=None, save_waveforms=False, verbose=False):
        assert HAVE_EXDIR, "To use the ExdirExtractors run:\n\n pip install exdir\n\n"
        if sample_rate is None and recording is None:
            raise Exception("Provide 'sample_rate' argument (Hz)")
        else:
            if recording is None:
                sample_rate = sample_rate * pq.Hz
            else:
                sample_rate = recording.get_sampling_frequency() * pq.Hz

        exdir_group = exdir.File(exdir_file, plugins=exdir.plugins.quantities)
        ephys = exdir_group.require_group('processing').require_group('electrophysiology')
        ephys.attrs['sample_rate'] = sample_rate

        if 'group' in sorting.get_unit_property_names():
            channel_groups = np.unique([sorting.get_unit_property(unit, 'group') for unit in sorting.get_unit_ids()])
        else:
            channel_groups = [0]

        if len(channel_groups) == 1:
            chan = 0
            if verbose:
                print("Single group: ", chan)
            ch_group = ephys.require_group('channel_group_' + str(chan))
            try:
                del ch_group['UnitTimes']
                del ch_group['EventWaveform']
                del ch_group['Clustering']
            except Exception as e:
                pass
            unittimes = ch_group.require_group('UnitTimes')
            unit_stop_time = np.max([(np.max(sorting.get_unit_spike_train(u).astype(float) / sample_rate).rescale('s'))
                                     for u in sorting.get_unit_ids()]) * pq.s
            recording_stop_time = None
            if recording is not None:
                ch_group.attrs['electrode_group_id'] = chan
                ch_group.attrs['electrode_identities'] = np.array([])
                ch_group.attrs['electrode_idx'] = np.arange(len(recording.get_channel_ids()))
                ch_group.attrs['start_time'] = 0 * pq.s
                recording_stop_time = recording.get_num_frames() / float(recording.get_sampling_frequency()) * pq.s

                unittimes.attrs['electrode_group_id'] = chan
                unittimes.attrs['electrode_identities'] = np.array([])
                unittimes.attrs['electrode_idx'] = np.array(recording.get_channel_ids())
                unittimes.attrs['start_time'] = 0 * pq.s
            ch_group.attrs['sample_rate'] = sample_rate

            if recording_stop_time is not None:
                unittimes.attrs['stop_time'] = recording_stop_time if recording_stop_time > unit_stop_time \
                    else unit_stop_time
                ch_group.attrs['stop_time'] = recording_stop_time if recording_stop_time > unit_stop_time \
                    else unit_stop_time

            nums = np.array([])
            timestamps = np.array([])
            waveforms = np.array([])
            for unit in sorting.get_unit_ids():
                unit_group = unittimes.require_group(str(unit))
                unit_group.require_dataset('times',
                                           data=(sorting.get_unit_spike_train(unit).astype(float)
                                                 / sample_rate).rescale('s'))
                unit_group.attrs['cluster_group'] = 'unsorted'
                unit_group.attrs['group_id'] = chan
                unit_group.attrs['name'] = 'unit #' + str(unit)

                timestamps = np.concatenate((timestamps, (sorting.get_unit_spike_train(unit).astype(float)
                                                          / sample_rate).rescale('s')))
                nums = np.concatenate((nums, [unit] * len(sorting.get_unit_spike_train(unit))))

                if 'waveforms' in sorting.get_unit_spike_feature_names(unit):
                    if len(waveforms) == 0:
                        waveforms = sorting.get_unit_spike_features(unit, 'waveforms')
                    else:
                        waveforms = np.vstack((waveforms, sorting.get_unit_spike_features(unit, 'waveforms')))

            if save_waveforms:
                if verbose:
                    print("Saving EventWaveforms")
                if 'waveforms' in sorting.get_unit_spike_feature_names():
                    eventwaveform = ch_group.require_group('EventWaveform')
                    waveform_ts = eventwaveform.require_group('waveform_timeseries')
                    data = waveform_ts.require_dataset('data', data=waveforms)
                    waveform_ts.attrs['electrode_group_id'] = chan
                    data.attrs['num_samples'] = len(waveforms)
                    data.attrs['sample_rate'] = sample_rate
                    data.attrs['unit'] = pq.dimensionless
                    times = waveform_ts.require_dataset('timestamps', data=timestamps)
                    times.attrs['num_samples'] = len(timestamps)
                    times.attrs['unit'] = pq.s
                    if recording is not None:
                        waveform_ts.attrs['electrode_identities'] = np.array([])
                        waveform_ts.attrs['electrode_idx'] = np.arange(len(recording.get_channel_ids()))
                        waveform_ts.attrs['start_time'] = 0 * pq.s
                        if recording_stop_time is not None:
                            waveform_ts.attrs['stop_time'] = recording_stop_time if recording_stop_time > unit_stop_time \
                                else unit_stop_time
                        waveform_ts.attrs['sample_rate'] = sample_rate
                        waveform_ts.attrs['sample_length'] = waveforms.shape[1]
                        waveform_ts.attrs['num_samples'] = len(waveforms)
                if verbose:
                    print("Saving Clustering")
                clustering = ch_group.require_group('Clustering')
                ts = clustering.require_dataset('timestamps', data=timestamps * pq.s)
                ts.attrs['num_samples'] = len(timestamps)
                ts.attrs['unit'] = pq.s
                ns = clustering.require_dataset('nums', data=nums)
                ns.attrs['num_samples'] = len(nums)
                cn = clustering.require_dataset('cluster_nums', data=np.array(sorting.get_unit_ids()))
                cn.attrs['num_samples'] = len(sorting.get_unit_ids())
        else:
            # remove preexisten spike sorting data
            max_group = 10
            for chan in np.arange(max_group):
                if 'channel_group_' + str(chan) in ephys.keys():
                    if verbose:
                        print('Removing channel', chan, 'info')
                    ch_group = ephys.require_group('channel_group_' + str(chan))
                    try:
                        del ch_group['UnitTimes']
                        del ch_group['EventWaveform']
                        del ch_group['Clustering']
                    except Exception as e:
                        pass
            channel_groups = np.unique([sorting.get_unit_property(unit, 'group') for unit in sorting.get_unit_ids()])
            for chan in channel_groups:
                if verbose:
                    print("Group: ", chan)
                ch_group = ephys.require_group('channel_group_' + str(chan))
                unittimes = ch_group.require_group('UnitTimes')
                unit_stop_time = np.max([(np.max(sorting.get_unit_spike_train(u).astype(float) / sample_rate).rescale('s'))
                                         for u in sorting.get_unit_ids()]) * pq.s
                recording_stop_time = None
                if recording is not None:
                    unittimes.attrs['electrode_group_id'] = chan
                    unittimes.attrs['electrode_identities'] = np.array([])
                    unittimes.attrs['electrode_idx'] = np.array([ch for i_c, ch in enumerate(recording.get_channel_ids())
                                                                 if recording.get_channel_property(ch, 'group') == chan])
                    unittimes.attrs['start_time'] = 0 * pq.s
                    recording_stop_time = recording.get_num_frames() / float(recording.get_sampling_frequency()) * pq.s

                    ch_group.attrs['electrode_group_id'] = chan
                    ch_group.attrs['electrode_identities'] = np.array([i_c for i_c, ch in enumerate(recording.get_channel_ids())
                                                                       if recording.get_channel_property(ch, 'group') == chan])
                    ch_group.attrs['electrode_idx'] = np.array([i_c for i_c, ch in enumerate(recording.get_channel_ids())
                                                                if recording.get_channel_property(ch, 'group') == chan])
                    ch_group.attrs['start_time'] = 0 * pq.s
                ch_group.attrs['sample_rate'] = sample_rate

                if recording_stop_time is not None:
                    unittimes.attrs['stop_time'] = recording_stop_time if recording_stop_time > unit_stop_time \
                        else unit_stop_time
                    ch_group.attrs['stop_time'] = recording_stop_time if recording_stop_time > unit_stop_time \
                        else unit_stop_time
                nums = np.array([])
                timestamps = np.array([])
                waveforms = np.array([])
                for unit in sorting.get_unit_ids():
                    if sorting.get_unit_property(unit, 'group') == chan:
                        if verbose:
                            print("Unit: ", unit)
                        unit_group = unittimes.require_group(str(unit))
                        unit_group.require_dataset('times',
                                                   data=(sorting.get_unit_spike_train(unit).astype(float)
                                                         / sample_rate).rescale('s'))
                        unit_group.attrs['cluster_group'] = 'unsorted'
                        unit_group.attrs['group_id'] = chan
                        unit_group.attrs['name'] = 'unit #' + str(unit)

                        timestamps = np.concatenate((timestamps, (sorting.get_unit_spike_train(unit).astype(float)
                                                         / sample_rate).rescale('s')))
                        nums = np.concatenate((nums, [unit]*len(sorting.get_unit_spike_train(unit))))

                        if 'waveforms' in sorting.get_unit_spike_feature_names(unit):
                            if len(waveforms) == 0:
                                waveforms = sorting.get_unit_spike_features(unit, 'waveforms')
                            else:
                                waveforms = np.vstack((waveforms, sorting.get_unit_spike_features(unit, 'waveforms')))
                if save_waveforms:
                    if verbose:
                        print("Saving EventWaveforms")
                    if 'waveforms' in sorting.get_unit_spike_feature_names():
                        eventwaveform = ch_group.require_group('EventWaveform')
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
                                                                           enumerate(recording.get_channel_ids())
                                                                           if recording.get_channel_property(ch, 'group') == chan])
                            waveform_ts.attrs['start_time'] = 0 * pq.s
                            if recording_stop_time is not None:
                                waveform_ts.attrs['stop_time'] = recording_stop_time if recording_stop_time > unit_stop_time \
                                    else unit_stop_time
                            waveform_ts.attrs['sample_rate'] = sample_rate
                            waveform_ts.attrs['sample_length'] = waveforms.shape[1]
                            waveform_ts.attrs['num_samples'] = len(waveforms)
                if verbose:
                    print("Saving Clustering")
                clustering = ephys.require_group('channel_group_' + str(chan)).require_group('Clustering')
                ts = clustering.require_dataset('timestamps', data=timestamps*pq.s)
                ts.attrs['num_samples'] = len(timestamps)
                ts.attrs['unit'] = pq.s
                ns = clustering.require_dataset('nums', data=nums)
                ns.attrs['num_samples'] = len(nums)
                cn = clustering.require_dataset('cluster_nums', data=np.array(sorting.get_unit_ids()))
                cn.attrs['num_samples'] = len(sorting.get_unit_ids())
