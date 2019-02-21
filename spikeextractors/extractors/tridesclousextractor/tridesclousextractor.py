from spikeextractors import SortingExtractor

try:
    import tridesclous as tdc
    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSortingExtractor(SortingExtractor):
    def __init__(self, tdc_folder, chan_grp=None):
        assert HAVE_TDC, 'must install tridesclous'
        SortingExtractor.__init__(self)
        self.dataio = tdc.DataIO(tdc_folder)
        if chan_grp is None:
            # if chan_grp is not provide take the first one if unique
            chan_grps = list(self.dataio.channel_groups.keys())
            assert len(chan_grps) == 1, 'There are sevreal in the folder chan_grp, specify it'
            chan_grp = chan_grps[0]
            
        self.chan_grp = chan_grp
        self.catalogue = self.dataio.load_catalogue(name='initial', chan_grp=chan_grp)

    def getUnitIds(self):
        return list(self.catalogue['clusters']['cluster_label'])

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        spikes = self.dataio.get_spikes(seg_num=0, chan_grp=self.chan_grp, i_start=None, i_stop=None)
        spikes = spikes[spikes['cluster_label'] == unit_id]
        spike_times = spikes['index']
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times

