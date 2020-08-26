from spikeextractors import SortingExtractor
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id

try:
    import tridesclous as tdc

    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSortingExtractor(SortingExtractor):
    extractor_name = 'TridesclousSortingExtractor'
    installed = HAVE_TDC  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = "To use the TridesclousSortingExtractor install tridesclous: \n\n pip install tridesclous\n\n"  # error message when not installed

    def __init__(self, folder_path, chan_grp=None):
        assert HAVE_TDC, self.installation_mesg
        tdc_folder = Path(folder_path)
        SortingExtractor.__init__(self)
        
        dataio = tdc.DataIO(str(tdc_folder))
        if chan_grp is None:
            # if chan_grp is not provided, take the first one if unique
            chan_grps = list(dataio.channel_groups.keys())
            assert len(chan_grps) == 1, 'There are several groups in the folder, specify chan_grp=...'
            chan_grp = chan_grps[0]

        self.chan_grp = chan_grp
        
        catalogue = dataio.load_catalogue(name='initial', chan_grp=chan_grp)
        
        labels = catalogue['clusters']['cluster_label']
        labels = labels[labels >= 0]
        self._unit_ids = list(labels)
        # load all spike in memory (this avoid to lock the folder with memmap throug dataio
        self._all_spikes = dataio.get_spikes(seg_num=0, chan_grp=self.chan_grp, i_start=None, i_stop=None).copy()
    
        self._sampling_frequency = dataio.sample_rate
        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'chan_grp': chan_grp}

    def get_unit_ids(self):
        return self._unit_ids

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        spikes = self._all_spikes
        spikes = spikes[spikes['cluster_label'] == unit_id]
        spike_times = spikes['index']
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times.copy()
