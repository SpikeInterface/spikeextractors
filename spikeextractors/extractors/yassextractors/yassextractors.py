from spikeextractors import SortingExtractor
from spikeextractors.extractors.numpyextractors import NumpyRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id

try:
    import h5py
    HAVE_SCSX = True
except ImportError:
    HAVE_SCSX = False

class YassSortingExtractor(SortingExtractor):

    extractor_name = 'YassExtractor'
    installed = HAVE_SCSX  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = "To use the SpykingCircusSortingExtractor install h5py: \n\n pip install h5py\n\n"
    
    
    def __init__(self, fname):
        SortingExtractor.__init__(self)

        ## All file specific initialization code can go here.
        # If your format stores the sampling frequency, you can overweite the self._sampling_frequency. This way,
        # the base method self.get_sampling_frequency() will return the correct sampling frequency
        
        self.fname_spike_train = fname
        self.spike_train = np.load(self.fname_spike_train)
        #self._sampling_frequency = my_sampling_frequency

    def get_unit_ids(self):

        #Fill code to get a unit_ids list containing all the ids (ints) of detected units in the recording

        unit_ids = np.unique(self.spike_train[:,1])
        
        return unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

        '''Code to extract spike frames from the specified unit.
        '''

        # find unit id spike times
        idx = np.where(self.spike_train[:,1]==unit_id)
        spike_times = self.spike_train[idx,0].squeeze()
        #print (spike_times.shape)

        # find spike times
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = 1E50 # use large time
            
        idx2 = np.where(np.logical_and(spike_times>=start_frame, spike_times<end_frame))[0]
        #print (idx2.shape)
        spike_times = spike_times[idx2]
        
        return spike_times

    
