from spikeextractors import RecordingExtractor,SortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_traces_args,check_valid_unit_id,read_binary
from bs4 import BeautifulSoup
import os

try:
    import bs4
    import lxml
    HAVE_BS4_LXML = True
except ImportError:
    HAVE_BS4_LXML = False

class NeuroscopeRecordingExtractor(BinDatRecordingExtractor):
    
    """
    Extracts raw neural recordings from large binary .dat files in the neuroscope format.
    
    The recording extractor always returns channel IDs starting from 0.
    
    The recording data will always be returned in the shape of (num_channels,num_frames).

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the .dat file.
    """
    extractor_name = 'NeuroscopeRecordingExtractor'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = 'please install bs4 and lxml to use this extractor'  # error message when not installed
            
    def __init__(self, folder_path):
        assert HAVE_BS4_LXML, self.installation_mesg
        RecordingExtractor.__init__(self)
        self._recording_file = folder_path
        
        fpath_base, fname = os.path.split(folder_path)
        xml_filepath = os.path.join(folder_path, fname + '.xml')
        dat_filepath = os.path.join(folder_path, fname + '.dat')
        
        with open(xml_filepath, 'r') as xml_file:
            contents = xml_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            # Normally, this would be a .xml, but there were strange issues
            # in the write_recording method that require it to be a .lxml instead
            # which also requires all capital letters to be removed from the tag names
        
        n_bits = int(soup.nbits.string)
        dtype='int'+str(n_bits)
        num_channels = int(soup.nchannels.string)
        sampling_frequency = float(soup.samplingrate.string)
        
        BinDatRecordingExtractor.__init__(self, dat_filepath, sampling_frequency=sampling_frequency,
                                          dtype=dtype, numchan=num_channels)
        
        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}
        
        
    @staticmethod
    def write_recording(recording, save_path):
        """ Convert and save the recording extractor to Neuroscope format

        parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved
        save_path: str
            Full path to desired target folder
        """
        RECORDING_NAME = save_path
        save_xml = "{}/{}.xml".format(save_path,save_path)

        # write recording
        recording_fn = os.path.join(save_path, RECORDING_NAME)
        BinDatRecordingExtractor.write_recording(recording, recording_fn, dtype=str(recording.get_dtype()))

        # create parameters file if none exists
        if not os.path.isfile(save_xml):
            soup = BeautifulSoup("",'xml')

            new_tag = soup.new_tag('nbits')
            dtype = recording.get_dtype()
            print(dtype)
            print(dtype.type)
            assert any([str(dtype) == x for x in ['int16', 'int32']]),"NeuroscopeRecordingExtractor only permits data of type 'int16' or 'int32'"
            n_bits = str(dtype)[3:5]
            new_tag.string = str(n_bits)
            soup.append(new_tag)

            new_tag = soup.new_tag('nchannels')
            new_tag.string = str(len(recording.get_channel_ids()))
            soup.append(new_tag)

            new_tag = soup.new_tag('samplingrate')
            new_tag.string = str(recording.get_sampling_frequency())
            soup.append(new_tag)

            # write parameters file
            f = open(save_xml, "w")
            f.write(str(soup))
            f.close()
        

class NeuroscopeSortingExtractor(SortingExtractor):

    """
    Extracts spiking information from pair of .res and .clu files. The .res is a text file with
    a sorted list of all spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to
    the total number of unique ids in the file (and may exclude 0 & 1 from this count)
    with the rest of the rows indicating which unit id the corresponding entry in the
    .res file refers to.
    
    In the original Neuroscope format:
        Unit ID 0 is the cluster of unsorted spikes (noise).
        Unit ID 1 is a cluster of multi-unit spikes.
        
    The function defaults to returning multi-unit activity as the first index, and ignoring unsorted noise.
    To return only the fully sorted units, set keep_mua_units=False.
        
    The sorting extractor always returns unit IDs from 1, ..., number of chosen clusters.

    Parameters
    ----------
    resfile_path : str
        Path to the collection of .res text file.
    clufile_path : str
        Path to the collection of .clu text file.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    """
    extractor_name = 'NeuroscopeSortingExtractor'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'custom'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path=None, resfile_path=None, clufile_path=None, keep_mua_units=True, exclude_shanks=None):
        SortingExtractor.__init__(self)
        
        # If either resfile_path and clufile_path were passed
        if resfile_path is not None or clufile_path is not None:
            # If folder_path was also passed with resfile_path and clufile_path
            assert folder_path is not None, 'Either pass a single folder_path location, or a pair of resfile_path and clufile_path. Combination received.'
            # If both were not passed together
            assert resfile_path is not None and clufile_path is not None, 'Either pass a single folder_path location, or a pair of resfile_path and clufile_path. Mixture received.'
            
            folder_path = os.path.split(resfile_path)[0]
        
        # None of the location arguments were passed
        assert folder_path is None and resfile_path is None and clufile_path is None, 'Either pass a single folder_path location, or a pair of resfile_path and clufile_path. None received.'
        
        fpath_base, fname = os.path.split(folder_path)
        xml_filepath = os.path.join(folder_path, fname + '.xml')
        
        with open(xml_filepath, 'r') as xml_file:
            contents = xml_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            # Normally, this would be a .xml, but there were strange issues
            # in the write_recording method that require it to be a .lxml instead
            # which also requires all capital letters to be removed from the tag names
        
        self._sampling_frequency = float(soup.samplingrate.string) # careful not to confuse it with the lfpsamplingsate
        
        # If both resfile_path and clufile_path were passed
        # Classic functionality reading only a single shank
        if resfile_path is not None and clufile_path is not None:
            res = np.loadtxt(resfile_path, dtype=np.int64, usecols=0, ndmin=1)
            clu = np.loadtxt(clufile_path, dtype=np.int64, usecols=0, ndmin=1)
            if len(res) > 0:
                # Extract the number of clusters read as the first line of the clufile
                # then remove it from the clu list
                n_clu = clu[0]
                clu = np.delete(clu, 0)
                unique_ids = np.unique(clu)

                if not unique_ids==np.arange(n_clu+1): # some missing IDs somewhere
                    if 0 not in unique_ids: # missing unsorted IDs
                        n_clu += 1
                    if 1 not in unique_ids: # missing mua IDs
                        n_clu += 1
                    # If it is any other kinda of ID, then it is very strange that it is missing...


                # Initialize spike trains and extract times from .res and appropriate clusters from .clu
                # based on user input for ignoring multi-unit activity
                self._spiketrains = []
                if keep_mua_units: # default
                    n_clu -= 1;
                    self._unit_ids = [x+1 for x in range(n_clu)] # generates list from 1,...,clu[0]-1
                    for s_id in self._unit_ids:
                        self._spiketrains.append(res[(clu == s_id).nonzero()])
                else:
                    # Ignoring IDs of 0 until get_unsorted_spike_train is implemented into base
                    # Also ignoring IDs of 1 since user called keep_mua_units=False
                    n_clu -= 2;
                    self._unit_ids = [x+1 for x in range(n_clu)] # generates list from 1,...,clu[0]-2
                    for s_id in self._unit_ids:
                        self._spiketrains.append(res[(clu == s_id+1).nonzero()]) # only reading cluster IDs 2,...,clu[0]-1
            else:
                self._spiketrains = []
                self._unit_ids = []
        #else: # If only the folder_path was passed; new auto-detecting file structure functionality which can also read from multiple shanks
            
            
        self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                        'resfile_path': str(Path(resfile_path).absolute()),
                        'clufile_path': str(Path(clufile_path).absolute()),
                        'keep_mua_units': keep_mua_units,
                        'exclude_shanks': exclude_shanks}


    def get_unit_ids(self):
        return list(self._unit_ids)
    
    
    def get_sampling_frequency(self):
        return self._sampling_frequency
    
    
    def shift_unit_ids(self,shift):
        self._unit_ids = [x + shift for x in self._unit_ids]
    
    
    def add_unit(self, unit_id, spike_times):
        '''This function adds a new unit with the given spike times.

        Parameters
        ----------
        unit_id: int
            The unit_id of the unit to be added.
        '''
        self._unit_ids.append(unit_id)
        self._spiketrains.append(spike_times)
        
        
#     def read_shanks(general_path, shanks, keep_mua_units=True):
#         '''
#         This function streamlines the iterative concatenation of multiple
#         NeuroscopeSortingExtractor objects into a single list, with each
#         element representing data specific to that index of the shanks list.

#         Parameters
#         ----------
#         general_path: str
#             The basic path to the location of multiple .res and .clu files
#         shanks: list
#             List of indices corresponding to which shanks to read from; this index
#             must be the value appended onto the .res.%i and .res.%i files
#         keep_mua_units: bool
#             Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
#         '''
#         nse_shank = []
#         for shankn in shanks:
#             # Get isolated cluster-based spike activity from .res and .clu files on a per-shank basis
#             #res_file = os.path.join(general_path, '.res.' + str(shankn))
#             #clu_file = os.path.join(general_path, '.clu.' + str(shankn))
#             res_file = general_path + '.res.' + str(shankn)
#             clu_file = general_path + '.clu.' + str(shankn)

#             if not os.path.isfile(res_file):
#                 print('spike times for shank{} not found'.format(shankn))
#             if not os.path.isfile(clu_file):
#                 print('spike clusters for shank{} not found'.format(shankn))

#             nse_shank.append(NeuroscopeSortingExtractor(res_file,clu_file,keep_mua_units=keep_mua_units))
            
#         return nse_shank
    

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, shank_id=None, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    
    @staticmethod
    def write_sorting(sorting, save_path):
        save_xml = "{}/{}.xml".format(save_path,save_path)
            
        # Create and save .res and .clu files from the current sorting object
        save_res = "{}/{}.res".format(save_path,save_path)
        save_clu = "{}/{}.clu".format(save_path,save_path)
        unit_ids = sorting.get_unit_ids()
        if len(unit_ids) > 0:
            spiketrains = [sorting.get_unit_spike_train(u) for u in unit_ids]
            res = np.concatenate(spiketrains).ravel()
            clu = np.concatenate([np.repeat(i+1,len(st)) for i,st in enumerate(spiketrains)]).ravel() # i here counts from 0
            res_sort = np.argsort(res)
            res = res[res_sort]
            clu = clu[res_sort]
        else:
            res = []
            clu = []
        
        unique_ids = np.unique(clu)
        n_clu = len(unique_ids)
            
        clu = np.insert(clu, 0, n_clu) # The +1 is necessary here b/c the convention for the base sorting object is from 1,...,nUnits

        np.savetxt(save_res, res, fmt='%i')
        np.savetxt(save_clu, clu, fmt='%i')
        
        # create parameters file if none exists
        if not os.path.isfile(save_xml):
            soup = BeautifulSoup("",'xml')

            new_tag = soup.new_tag('samplingrate')
            new_tag.string = str(sorting.get_sampling_frequency())
            soup.append(new_tag)

            # write parameters file
            f = open(save_xml, "w")
            f.write(str(soup))
            f.close()
