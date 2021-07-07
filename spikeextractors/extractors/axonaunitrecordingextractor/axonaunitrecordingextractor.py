from spikeextractors.extraction_tools import check_get_traces_args
from spikeextractors.extractors.neoextractors.neobaseextractor import (
    _NeoBaseExtractor, NeoBaseRecordingExtractor)
from spikeextractors import RecordingExtractor
from pathlib import Path
import numpy as np
from typing import Union
import warnings

PathType = Union[Path, str]

try:
    import neo
    from neo.rawio.baserawio import _signal_channel_dtype, _signal_stream_dtype
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class AxonaUnitRecordingExtractor(NeoBaseRecordingExtractor, RecordingExtractor, _NeoBaseExtractor):
    """
    Instantiates a RecordingExtractor from an Axona Unit mode file.

    Since the unit mode format only saves waveform cutouts, the get_traces
    function fills in the rest of the recording with Gaussian uncorrelated
    noise

    Parameters
    ----------

    noise_std: float
        Standard deviation of the Gaussian background noise (default 3)
    """
    extractor_name = 'AxonaUnitRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, noise_std: float = 3, block_index=None, seg_index=None, **kargs):
        RecordingExtractor.__init__(self)
        _NeoBaseExtractor.__init__(self, block_index=block_index, seg_index=seg_index, **kargs)

        # Enforce 1 signal stream (there are 0 raw streams), we will create 1 from waveforms
        signal_streams = self.neo_reader._get_signal_streams_header()
        signal_channels = self.neo_reader._get_signal_chan_header()
        self.neo_reader.header['signal_streams'] = np.array(signal_streams,
                                                            dtype=_signal_stream_dtype)
        self.neo_reader.header['signal_channels'] = np.array(signal_channels,
                                                             dtype=_signal_channel_dtype)

        if hasattr(self.neo_reader, 'get_group_signal_channel_indexes'):
            # Neo >= 0.9.0
            channel_indexes_list = self.neo_reader.get_group_signal_channel_indexes()
            num_streams = len(channel_indexes_list)
            assert num_streams <= 1, 'This file have several channel groups spikeextractors support only one groups'
            self.after_v10 = False
        elif hasattr(self.neo_reader, 'get_group_channel_indexes'):
            # Neo < 0.9.0
            channel_indexes_list = self.neo_reader.get_group_channel_indexes()
            num_streams = len(channel_indexes_list)
            self.after_v10 = False
        elif hasattr(self.neo_reader, 'signal_streams_count'):
            # Neo >= 0.10.0 (not release yet in march 2021)
            num_streams = self.neo_reader.signal_streams_count()
            self.after_v10 = True
        else:
            raise ValueError('Strange neo version. Please upgrade your neo package: pip install --upgrade neo')

        assert num_streams <= 1, 'This file have several signal streams spikeextractors support only one streams' \
                                 'Maybe you can use option to select only one stream'

        # spikeextractor for units to be uV implicitly
        # check that units are V, mV or uV
        units = self.neo_reader.header['signal_channels']['units']
        assert np.all(np.isin(units, ['V', 'mV', 'uV'])), 'Signal units no Volt compatible'
        self.additional_gain = np.ones(units.size, dtype='float')
        self.additional_gain[units == 'V'] = 1e6
        self.additional_gain[units == 'mV'] = 1e3
        self.additional_gain[units == 'uV'] = 1.
        self.additional_gain = self.additional_gain.reshape(1, -1)

        # Add channels properties
        header_channels = self.neo_reader.header['signal_channels'][slice(None)]
        self._neo_chan_ids = self.neo_reader.header['signal_channels']['id']

        # In neo there is not guarantee that channel ids are unique.
        # for instance Blacrock can have several times the same chan_id
        # different sampling rate
        # so check it
        assert np.unique(self._neo_chan_ids).size == self._neo_chan_ids.size, 'In this format channel ids are not ' \
                                                                              'unique! Incompatible with SpikeInterface'

        try:
            channel_ids = [int(ch) for ch in self._neo_chan_ids]
        except Exception as e:
            warnings.warn("Could not parse channel ids to int: using linear channel map")
            channel_ids = list(np.arange(len(self._neo_chan_ids)))
        self._channel_ids = channel_ids

        gains = header_channels['gain'] * self.additional_gain[0]
        self.set_channel_gains(gains=gains, channel_ids=self._channel_ids)

        names = header_channels['name']
        for i, ind in enumerate(self._channel_ids):
            self.set_channel_property(channel_id=ind, property_name='name', value=names[i])

        self._noise_std = noise_std

        # Read channel groups by tetrode IDs
        self.set_channel_groups(groups=[
            tetrode_id - 1 for tetrode_id in self.neo_reader.get_active_tetrode() for _ in range(4)])

        header_channels = self.neo_reader.header['signal_channels'][slice(None)]

        names = header_channels['name']
        channel_ids = self.get_channel_ids()
        for i, ind in enumerate(channel_ids):
            self.set_channel_property(channel_id=ind, property_name='name', value=names[i])

        # Set channel gains for int8 .X Unit data
        gains = self.neo_reader._get_channel_gain(bytes_per_sample=1)[0:len(channel_ids)]
        self.set_channel_gains(gains, channel_ids=channel_ids)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):

        timebase_sr = int(self.neo_reader.file_parameters['unit']['timebase'].split(' ')[0])
        samples_pre = int(self.neo_reader.file_parameters['set']['file_header']['pretrigSamps'])
        samples_post = int(self.neo_reader.file_parameters['set']['file_header']['spikeLockout'])
        sampling_rate = self.get_sampling_frequency()

        tcmap = self._get_tetrode_channel_table(channel_ids)

        traces = self._noise_std * np.random.randn(len(channel_ids), end_frame - start_frame)
        if return_scaled:
            traces = traces.astype(np.float32)
        else:
            traces = traces.astype(np.int8)

        # Loop through tetrodes and include requested channels in traces
        itrc = 0
        for tetrode_id in np.unique(tcmap[:, 0]):

            channels_oi = tcmap[tcmap[:, 0] == tetrode_id, 2]

            waveforms = self.neo_reader._get_spike_raw_waveforms(
                block_index=0, seg_index=0,
                unit_index=tetrode_id - 1,  # Tetrodes IDs are 1-indexed
                t_start=start_frame / sampling_rate,
                t_stop=end_frame / sampling_rate
            )
            waveforms = waveforms[:, channels_oi, :]
            nch = len(channels_oi)

            spike_train = self.neo_reader._get_spike_timestamps(
                block_index=0, seg_index=0,
                unit_index=tetrode_id - 1,
                t_start=start_frame / sampling_rate,
                t_stop=end_frame / sampling_rate
            )

            # Fill waveforms into traces timestamp by timestamp
            for t, wf in zip(spike_train, waveforms):

                t = int(t // (timebase_sr / sampling_rate))  # timestamps are sampled at higher frequency
                t = t - start_frame
                if (t - samples_pre < 0) and (t + samples_post > traces.shape[1]):
                    traces[itrc:itrc + nch, :] = wf[:, samples_pre - t:traces.shape[1] - (t - samples_pre)]
                elif t - samples_pre < 0:
                    traces[itrc:itrc + nch, :t + samples_post] = wf[:, samples_pre - t:]
                elif t + samples_post > traces.shape[1]:
                    traces[itrc:itrc + nch, t - samples_pre:] = wf[:, :traces.shape[1] - (t - samples_pre)]
                else:
                    traces[itrc:itrc + nch, t - samples_pre:t + samples_post] = wf

            itrc += nch

        return traces

    def get_num_frames(self):
        n = int(self.neo_reader.segment_t_stop(block_index=0, seg_index=0) * self.get_sampling_frequency())
        if self.get_sampling_frequency() == 24000:
            n = n // 2
        return n

    def get_sampling_frequency(self):
        return int(self.neo_reader.header['spike_channels'][0][-1])

    def get_channel_ids(self):
        return self._channel_ids

    def _get_tetrode_channel_table(self, channel_ids):
        '''Create auxiliary np.array with the following columns:
        Tetrode ID, Channel ID, Channel ID within tetrode
        This is useful in `get_traces()`

        Parameters
        ----------
        channel_ids : list
            List of channel ids to include in table

        Returns
        -------
        np.array
            Rows = channels,
            columns = TetrodeID, ChannelID, ChannelID within Tetrode
        '''
        active_tetrodes = self.neo_reader.get_active_tetrode()

        tcmap = np.zeros((len(active_tetrodes) * 4, 3), dtype=int)
        row_id = 0
        for tetrode_id in [int(s[0].split(' ')[1]) for s in self.neo_reader.header['spike_channels']]:

            all_channel_ids = self.neo_reader._get_channel_from_tetrode(tetrode_id)

            for i in range(4):
                tcmap[row_id, 0] = int(tetrode_id)
                tcmap[row_id, 1] = int(all_channel_ids[i])
                tcmap[row_id, 2] = int(i)
                row_id += 1

        del_idx = [False if i in channel_ids else True for i in tcmap[:, 1]]

        return np.delete(tcmap, del_idx, axis=0)
