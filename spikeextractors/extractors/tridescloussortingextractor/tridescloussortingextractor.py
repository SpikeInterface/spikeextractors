from spikeextractors import SortingExtractor
from pathlib import Path

try:
    import tridesclous as tdc
    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSortingExtractor(SortingExtractor):

    extractor_name = 'TridesclousSortingExtractor'
    installed = HAVE_TDC  # check at class level if installed or not
    _gui_params = [
        {'name': 'tdc_folder', 'type': 'path', 'title': "Path to folder"},
        {'name': 'chan_grp', 'type': 'list', 'value':None, 'default':None, 'title': "List of channel groups"},
    ]
    installation_mesg = "must install tridesclous" # error message when not installed

    def __init__(self, tdc_folder, chan_grp=None):
        assert HAVE_TDC, "must install tridesclous"
        tdc_folder = Path(tdc_folder)
        SortingExtractor.__init__(self)
        self.dataio = tdc.DataIO(str(tdc_folder))
        if chan_grp is None:
            # if chan_grp is not provided, take the first one if unique
            chan_grps = list(self.dataio.channel_groups.keys())
            assert len(chan_grps) == 1, 'There are several in the folder chan_grp, specify it'
            chan_grp = chan_grps[0]

        self.chan_grp = chan_grp
        self.catalogue = self.dataio.load_catalogue(name='initial', chan_grp=chan_grp)
        
        self._sampling_frequency = self.dataio.sample_rate

    def get_unit_ids(self):
        labels = self.catalogue['clusters']['cluster_label']
        labels = labels[labels>=0]
        return list(labels)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        spikes = self.dataio.get_spikes(seg_num=0, chan_grp=self.chan_grp, i_start=None, i_stop=None)
        spikes = spikes[spikes['cluster_label'] == unit_id]
        spike_times = spikes['index']
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times.copy()  #copy avoid reference to the unerlying memmap
