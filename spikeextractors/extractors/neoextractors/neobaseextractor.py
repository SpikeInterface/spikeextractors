import numpy as np
import neo

from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor


class _NeoBaseExtractor:
    NeoRawIOClass = None

    installed = True
    is_writable = False
    extractor_gui_params = []
    
    def __init__(self, **kargs):
        self.neo_reader = self.NeoRawIOClass(**kkargs)
        self.neo_reader.parse_header()
        
        # TODO propose a meachanisim to select the appropriate segment
        # in case there are several
        num_block = self.neo_reader.block_count()
        assert num_block == 1, 'This file is multi block spikeextractors support only one segment'
        num_seg = self.neo_reader.segment_count()
        assert num_seg == 1, 'This file is multi segment spikeextractors support only one segment'
        
        # select first block first segment
        self.block_index = 0
        self.seg_index = 0
    

class NeoBaseRecordingExtractor(RecordingExtractor, _NeoBaseExtractor):
    
    def __init__(self, **kargs):
        _NeoBaseExtractor.__init__(self, **kargs)
        
        
        # TODO propose a meachanisim to select the appropriate channel groups
        # in neo one channel group have the same dtype/sampling_rate/group_id
        num_chan_group = len(self.neo_reader.get_group_channel_indexes())
        assert num_chan_group == 1, 'This file have several channel groups spikeextractors support only one groups'
        
        
        # spikeextractor for units to be uV implicitly
        # check that units are V, mV or uV
        # otherwise raise error
        units = self.neo_reader.header['signal_channels']['units']
        assert np.all(np.isin(units, ['V', 'mV', 'uV'])), 'Signal units no Volt compatible'
        self.additional_gain = np.ones(units.size, dtype='float')
        self.additional_gain[units == 'V'] = 1e6
        self.additional_gain[units == 'mV'] =  1e3
        self.additional_gain[units == 'uV'] = 1.
        self.additional_gain =self.additional_gain.reshape(-1, 1)

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        # in neo rawio channel can acces by names/ids/indexes
        # there is no garranty that ids/names are unique on some formats
        raw_traces = self.neo_reader.get_analogsignal_chunk(block_index=self.clock_index, seg_index=self.seg_index,
                                i_start=start_frame, i_stop=end_frame,
                               channel_indexes=None, channel_names=None, channel_ids=channel_ids)
        
        # rescale gtrace to natural units
        scaled_traces = self.neo_reader.rescale_signal_raw_to_float(raw_traces, dtype='float32',
                                    channel_indexes=None, channel_names=None, channel_ids=channel_ids)
        # and then to uV
        scaled_traces *= self.additional_gain
        
        # fortunatly neo works with (samples, channels) strides
        # so transpose to spieextractors wolrd
        scaled_traces = scaled_traces.transpose()
        
        return scaled_traces
        

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
        return chan_ids


class NeoBaseSortingExtractor(SortingExtractor, _NeoBaseExtractor):
    def __init__(self, **kargs):
        _NeoBaseExtractor.__init__(self, **kargs)
        
        # the sampling frequency is quite tricky because in neo
        # spike are handle in s or ms
        # internally many format do have have the spike time stamps
        # at the same speed as the signal but at a higher clocks speed.
        # here in spikeinterface we need spike index to be at the same speed 
        # that signal it do not make sens to have spikes at 50kHz sample
        # when the sig is 10kHz.
        # neo handle this but not spikeextractors
        
        self._handle_sampling_frequency()
        #self._sampling_frequency = None
    
    def _handle_sampling_frequency(self):
        # must handle :
        #   _sampling_frequency
        #   _time_stamp0
        raise(NotImplementedError)
    
    def get_unit_ids(self):
        # should be this
        unit_ids = self.neo_reader.header['unit_channels']['id']
        # in neo unit_ids are string so here we take index
        unit_ids = np.arange(unit_ids.size, dtype='int64')
        return unit_ids
    
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        neo_unit_id = self.neo_reader.header['unit_channels']['id'][unit_id]
        
        # neo handle limts in times (s, ms) but spikeextractors in frames.
        # but spike can have diffrents sampling rate than signals
        # so conversion from signals frames to times is
        # format dependent
        assert start_frame is None , 'Do not handle slice of spikes'
        assert end_frame is None, 'Do not handle slice of spikes'
        
        spike_timestamps = self.neo_reader.get_spike_timestamps(block_index=self.block_index, seg_index=self.seg_index,
                            unit_index=neo_unit_id, t_start=None, t_stop=None)
        spike_times = self.neo_reader.rescale_spike_timestamp(spike_timestamps, dtype='float64')
        
        spike_indexes = ((spike_times - self._time_start) * self._sampling_rate).astype('int64')
        return spike_indexes
