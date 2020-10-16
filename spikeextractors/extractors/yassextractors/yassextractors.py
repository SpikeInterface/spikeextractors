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

    
    #@staticmethod
    #def write_sorting(sorting, save_path):
    #    '''
    #    This is an example of a function that is not abstract so it is optional if you want to override it. It allows other
    #    SortingExtractors to use your new SortingExtractor to convert their sorted data into your
    #    sorting file format.
    #    '''
    
    
    
# class SpykingCircusRecordingExtractor(NumpyRecordingExtractor):
    # extractor_name = 'SpykingCircusRecordingExtractor'
    # has_default_locations = False
    # installed = True  # check at class level if installed or not
    # is_writable = False
    # mode = 'folder'
    # installation_mesg = ""  # error message when not installed

    # def __init__(self, folder_path):
        # spykingcircus_folder = Path(folder_path)
        # listfiles = spykingcircus_folder.iterdir()
        # sample_rate = None
        # recording_file = None

        # parent_folder = None
        # result_folder = None
        # for f in listfiles:
            # if f.is_dir():
                # if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                    # parent_folder = spykingcircus_folder
                    # result_folder = f

        # if parent_folder is None:
            # parent_folder = spykingcircus_folder.parent
            # for f in parent_folder.iterdir():
                # if f.is_dir():
                    # if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                        # result_folder = spykingcircus_folder

        # assert isinstance(parent_folder, Path) and isinstance(result_folder, Path), "Not a valid spyking circus folder"

        # for f in parent_folder.iterdir():
            # if f.suffix == '.params':
                # sample_rate = _load_sample_rate(f)
            # if f.suffix == '.npy':
                # recording_file = str(f)
        # NumpyRecordingExtractor.__init__(self, recording_file, sample_rate)
        # self._kwargs = {'folder_path': str(Path(folder_path).absolute())}

# def _load_sample_rate(params_file):
    # sample_rate = None
    # with params_file.open('r') as f:
        # for r in f.readlines():
            # if 'sampling_rate' in r:
                # sample_rate = r.split('=')[-1]
                # if '#' in sample_rate:
                    # sample_rate = sample_rate[:sample_rate.find('#')]
                # sample_rate = float(sample_rate)
    # return sample_rate
    
