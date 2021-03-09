import numpy as np
import warnings

from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from spikeextractors.extraction_tools import check_get_traces_args, check_get_unit_spike_train

try:
    import neo
    HAVE_NEO = True
except ImportError:
    HAVE_NEO = False


class _NeoBaseExtractor:
    NeoRawIOClass = None
    installed = HAVE_NEO
    is_writable = False
    has_default_locations = False
    has_unscaled = True
    installation_mesg = "To use the Neo extractors, install Neo: \n\n pip install neo\n\n"

    def __init__(self, block_index=None, seg_index=None, **kargs):
        """
        if block_index is None then check if only one block
        if seg_index is None then check if only one segment

        """
        assert self.installed, self.installation_mesg
        neoIOclass = eval('neo.rawio.' + self.NeoRawIOClass)

        self.neo_reader = neoIOclass(**kargs)
        self.neo_reader.parse_header()

        if block_index is None:
            # auto select first block
            num_block = self.neo_reader.block_count()
            assert num_block == 1, 'This file is multi block spikeextractors support only one segment, please provide block_index='
            block_index = 0

        if seg_index is None:
            # auto select first segment
            num_seg = self.neo_reader.segment_count(block_index)
            assert num_seg == 1, 'This file is multi segment spikeextractors support only one segment, please provide seg_index='
            seg_index = 0

        self.block_index = block_index
        self.seg_index = seg_index
        self._kwargs = kargs
        self._kwargs.update({'seg_index': seg_index, 'block_index': block_index})


class NeoBaseRecordingExtractor(RecordingExtractor, _NeoBaseExtractor):

    def __init__(self, block_index=None, seg_index=None, **kargs):
        RecordingExtractor.__init__(self)
        _NeoBaseExtractor.__init__(self, block_index=block_index, seg_index=seg_index, **kargs)

        # TODO propose a meachanisim to select the appropriate channel groups
        # in neo one channel group have the same dtype/sampling_rate/group_id
        try:
            # Neo >= 0.9.0
            channel_indexes_list = self.neo_reader.get_group_signal_channel_indexes()
        except AttributeError:
            # Neo < 0.9.0
            channel_indexes_list = self.neo_reader.get_group_channel_indexes()        
        num_chan_group = len(channel_indexes_list)
        assert num_chan_group == 1, 'This file have several channel groups spikeextractors support only one groups'

        # spikeextractor for units to be uV implicitly
        # check that units are V, mV or uV
        # otherwise raise error
        # @alessio @cole : this can be a problem in extractor evrything is base
        #                     on the fact that the get_traces() give microVolt
        #                     some file don't have units
        #                     do we allow this ?
        units = self.neo_reader.header['signal_channels']['units']
        assert np.all(np.isin(units, ['V', 'mV', 'uV'])), 'Signal units no Volt compatible'
        self.additional_gain = np.ones(units.size, dtype='float')
        self.additional_gain[units == 'V'] = 1e6
        self.additional_gain[units == 'mV'] = 1e3
        self.additional_gain[units == 'uV'] = 1.
        self.additional_gain = self.additional_gain.reshape(1, -1)

        # Add channels properties
        header_channels = self.neo_reader.header['signal_channels'][slice(None)]
        channel_ids = self.get_channel_ids()

        gains = header_channels['gain'] * self.additional_gain[0]
        self.set_channel_gains(gains=gains, channel_ids=channel_ids)
        
        names = header_channels['name']
        for i, ind in enumerate(channel_ids):
            self.set_channel_property(channel_id=ind, property_name='name', value=names[i])

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        # in neo rawio channel can acces by names/ids/indexes
        # there is no garranty that ids/names are unique on some formats
        raw_traces = self.neo_reader.get_analogsignal_chunk(block_index=self.block_index, seg_index=self.seg_index,
                                                            i_start=start_frame, i_stop=end_frame,
                                                            channel_indexes=None, channel_names=None,
                                                            channel_ids=channel_ids)
        # neo works with (samples, channels) strides
        # so transpose to spikeextractors wolrd
        return raw_traces.transpose()

    def get_num_frames(self):
        # channel_indexes=None means all channels
        n = self.neo_reader.get_signal_size(self.block_index, self.seg_index, channel_indexes=None)
        return n

    def get_sampling_frequency(self):
        # channel_indexes=None means all channels
        sf = self.neo_reader.get_signal_sampling_rate(channel_indexes=None)
        return sf

    def get_channel_ids(self):
        chan_ids = self.neo_reader.header['signal_channels']['id']
        # in neo there is not garranty that chann ids are unique
        # for instance Blacrock can have several times the same chan_id
        # different sampling rate
        # so check it
        assert np.unique(chan_ids).size == chan_ids.size, 'In this format channel ids are not unique'
        # to avoid this limitation this could return chan_index which is 0...N-1
        return list(chan_ids)


class NeoBaseSortingExtractor(SortingExtractor, _NeoBaseExtractor):
    def __init__(self, block_index=None, seg_index=None, **kargs):
        SortingExtractor.__init__(self)
        _NeoBaseExtractor.__init__(self, block_index=block_index, seg_index=seg_index, **kargs)

        # the sampling frequency is quite tricky because in neo
        # spike are handle in s or ms
        # internally many format do have have the spike time stamps
        # at the same speed as the signal but at a higher clocks speed.
        # here in spikeinterface we need spike index to be at the same speed
        # that signal it do not make sens to have spikes at 50kHz sample
        # when the sig is 10kHz.
        # neo handle this but not spikeextractors

        self._handle_sampling_frequency()

    def _handle_sampling_frequency(self):
        # bacause neo handle spike in times (s or ms) but spikeextractors in frames related to signals.
        # In neo spikes can have diffrents sampling rate than signals so conversion from
        #  signals frames to times is format dependent

        # here the generic case
        #  all channels are in the same neo group so
        if len(self.neo_reader.header['signal_channels']['sampling_rate']) > 0:
            self._neo_sig_sampling_rate = self.neo_reader.header['signal_channels']['sampling_rate'][0]
            self.set_sampling_frequency(self._neo_sig_sampling_rate)
        else:
            warnings.warn("Sampling frequency not found: setting it to 30 kHz")
            self._sampling_frequency = 30000
            self._neo_sig_sampling_rate = self._sampling_frequency

        if len(self.neo_reader.get_group_signal_channel_indexes()) > 0:
            self._neo_sig_time_start = self.neo_reader.get_signal_t_start(self.block_index, self.seg_index,
                                                                          channel_indexes=[0])
        else:
            warnings.warn("Start time not found: setting it to 0 s")
            self._neo_sig_time_start = 0

        # For some IOs when there is no signals at inside the dataset this could not work
        # in that case the extractor class must overwrite this method

    def get_unit_ids(self):
        # should be this but this is strings in neo
        #  unit_ids = self.neo_reader.header['unit_channels']['id']

        # in neo unit_ids are string so here we take unit_index
        unit_ids = np.arange(self.neo_reader.header['unit_channels'].size, dtype='int64')
        return unit_ids

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        # this is a string
        #  neo_unit_id = self.neo_reader.header['unit_channels']['id'][unit_id]

        # this is an int
        unit_index = unit_id

        # in neo can be a sample, or hiher sample rate or even float
        spike_timestamps = self.neo_reader.get_spike_timestamps(block_index=self.block_index, seg_index=self.seg_index,
                                                                unit_index=unit_index, t_start=None, t_stop=None)

        if start_frame is not None:
            spike_timestamps = spike_timestamps[spike_timestamps >= start_frame]

        if end_frame is not None:
            spike_timestamps = spike_timestamps[spike_timestamps <= end_frame]

        # convert to second second
        spike_times = self.neo_reader.rescale_spike_timestamp(spike_timestamps, dtype='float64')

        # convert to sample related to recording signals
        spike_indexes = ((spike_times - self._neo_sig_time_start) * self._neo_sig_sampling_rate).astype('int64')
        return spike_indexes
