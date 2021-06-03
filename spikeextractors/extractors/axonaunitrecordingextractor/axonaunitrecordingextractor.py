from spikeextractors.extraction_tools import check_get_traces_args
from spikeextractors.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor
from pathlib import Path
import numpy as np
from typing import Union

PathType = Union[Path, str]

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class AxonaUnitRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Instantiates a RecordinExtractor from an Axon Unit mode file.

    Since the unit mode format only saves waveform cutouts, the get_traces
    function fills in the rest of the recording with Gaussian uncorrelated
    noise

    Parameters
    ----------

    noise_std: float
        Standard deviation of the Gaussian background noise
    """
    extractor_name = 'AxonaUnitRecording'
    mode = 'file'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, noise_std: float = 3, block_index=None, seg_index=None, **kargs):
        super().__init__(block_index=block_index, seg_index=seg_index, **kargs)
        self._noise_std = noise_std

        # Read channel groups by tetrode IDs
        self.set_channel_groups(groups=[x - 1 for x in self.neo_reader.raw_annotations[
            'blocks'][0]['segments'][0]['signals'][0]['__array_annotations__']['tetrode_id']])

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
                if t - samples_pre < 0:
                    traces[itrc:itrc + nch, :t + samples_post] = wf[:, samples_pre - t:]
                elif t + samples_post > traces.shape[1]:
                    traces[itrc:itrc + nch, t - samples_pre:] = wf[:, :traces.shape[1] - (t - samples_pre)]
                else:
                    traces[itrc:itrc + nch, t - samples_pre:t + samples_post] = wf

            itrc += nch

        return traces

    def get_num_frames(self):
        n = self.neo_reader.get_signal_size(self.block_index, self.seg_index, stream_index=0)
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
