from spikeextractors import RecordingExtractor,SortingExtractor,MultiSortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id
import os

from os import listdir
from os.path import isfile, isdir, join, split

try:
    from bs4 import BeautifulSoup
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
    file_path : str
        Path to the .dat file to be extracted.
    numchan : int
        Optional. Number of sequential channels to read from binary data, starting from the first.
    """
    extractor_name = 'NeuroscopeRecordingExtractor'
    installed = HAVE_BS4_LXML  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = 'Please install bs4 and lxml to use this extractor!'  # error message when not installed
            
    def __init__(self, file_path, numchan=None):
        assert HAVE_BS4_LXML, self.installation_mesg
        assert isfile(file_path) and file_path[-4:] == '.dat', 'file_path must lead to a .dat file!'
        
        RecordingExtractor.__init__(self)
        self._recording_file = file_path
        
        folder_path,_ = split(Path(file_path).absolute())
        
        xml_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.xml']
        assert any(xml_files), 'No .xml file found in the folder_path.'
        assert len(xml_files)==1, 'More than one .xml file found in the folder_path.'
        xml_filepath = '{}/{}'.format(folder_path, xml_files[0])
        
        with open(xml_filepath, 'r') as xml_file:
            contents = xml_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            # Normally, this would be a .xml, but there were strange issues
            # in the write_recording method that require it to be a .lxml instead
            # which also requires all capital letters to be removed from the tag names
        
        n_bits = int(soup.nbits.string)
        dtype='int'+str(n_bits)
        
        numchan_from_file = int(soup.nchannels.string)
        
        if numchan is not None:
            if numchan > numchan_from_file:
                print('Warning: Requested more channels with "numchan" than are available in recording. Defaulting to maximum number.')
                numchan = numchan_from_file
        else: # numchan not passed
            numchan = numchan_from_file
        
        sampling_frequency = float(soup.samplingrate.string)
        
        BinDatRecordingExtractor.__init__(self, file_path, sampling_frequency=sampling_frequency,
                                          dtype=dtype, numchan=numchan)
        
        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'numchan': numchan}
        
        
    @staticmethod
    def write_recording(recording, save_path, dtype=None):
        """
        Convert and save the recording extractor to Neuroscope format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved.
        save_path: str
            Path to desired target folder. The name of the files will be the same as the final directory.
        dtype: str or numpy data type
            Optional. Data type to be used in writing; must be int16 or int32 (default).
                      Will throw a warning if stored recording type from get_traces() does not match.
        """
        if dtype is not None: # user specified the data type manually
            dtype = str(dtype) # if user passed numpy data type
            int_loc = dtype.find('int')
            assert int_loc != -1, 'Data type must be int16 or int32! Non-integer received.'
            n_bits = dtype[(int_loc+3):(int_loc+5)]
            assert n_bits in ['16','32'], 'Data type must be int16 or int32!'
            print('Warning: Recording data type must be int16 or int32! Coercing to int'+n_bits+'.')
        
        abs_save_path = Path(save_path).absolute()
        _, RECORDING_NAME = os.path.split(abs_save_path)
        XML_NAME = RECORDING_NAME
        save_xml_filpath = Path('{}/{}.xml'.format(save_path,XML_NAME))

        # write recording
        # recording_fn = Path('{}/{}'.format(save_path,RECORDING_NAME)) # .dat extension handled in BinDataRecordingExtractor
        recording_fn = Path('{}/{}'.format(save_path,RECORDING_NAME))

        # create parameters file if none exists
        if not os.path.isfile(save_xml_filpath):
            soup = BeautifulSoup("",'xml')

            new_tag = soup.new_tag('nbits')
            recording_dtype = str(recording.get_dtype())
            int_loc = recording_dtype.find('int')
            recording_n_bits = recording_dtype[(int_loc+3):(int_loc+5)]
            
            if dtype is None: # user did not specify data type
                if int_loc != -1 and recording_n_bits in ['16','32']:
                    n_bits = recording_n_bits
                else:
                    print('Warning: Recording data type must be int16 or int32! Defaulting to int32.')
                    n_bits = '32'
                    # the data typing methods rely on the actual type of the implicit data, not just the value passed into BinDatRecordingExtractor
                    recording._timeseries = recording._timeseries.astype('int32') 
                dtype = 'int' + n_bits # update dtype in pass to BinDatRecordingExtractor.write_recording
            
            new_tag.string = n_bits
            soup.append(new_tag)

            new_tag = soup.new_tag('nchannels')
            new_tag.string = str(len(recording.get_channel_ids()))
            soup.append(new_tag)

            new_tag = soup.new_tag('samplingrate')
            new_tag.string = str(recording.get_sampling_frequency())
            soup.append(new_tag)

            # write parameters file
            with open(save_xml_filpath, "w") as f:
                f.write(str(soup))
            
        BinDatRecordingExtractor.write_recording(recording, recording_fn, dtype=dtype)
        

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
        Optional. Path to a particular .res text file.
    clufile_path : str
        Optional. Path to a particular .clu text file.
    folder_path : str
        Optional. Path to the collection of .res and .clu text files. Will auto-detect format.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    """
    extractor_name = 'NeuroscopeSortingExtractor'
    installed = HAVE_BS4_LXML  # check at class level if installed or not
    is_writable = True
    mode = 'custom'
    installation_mesg = 'Please install bs4 and lxml to use this extractor!'  # error message when not installed

    def __init__(self, resfile_path=None, clufile_path=None, folder_path=None, keep_mua_units=True):
        assert HAVE_BS4_LXML, self.installation_mesg
        
        # None of the location arguments were passed
        assert not (folder_path is None and resfile_path is None and clufile_path is None), 'Either pass a single folder_path location, or a pair of resfile_path and clufile_path. None received.'
        assert type(keep_mua_units) == bool, 'Optional argument "keep_mua_units" must be boolean!'
        
        # At least one file_path passed
        if resfile_path is not None or clufile_path is not None:
            assert resfile_path is not None and clufile_path is not None, 'If passing resfile_path or clufile_path, both are required.'
            assert isfile(resfile_path) and isfile(clufile_path), \
                'The resfile_path and clufile_path must be .res and .clu files!'
            assert folder_path is None, 'Pass either a single folder_path location, or a pair of resfile_path and clufile_path. All received.'
        
        SortingExtractor.__init__(self)
        
        # If folder_path was the only location passed
        # Auto-detect .res and .clu file structure with inferred naming, error check everything along the way
        if folder_path is not None:
            assert isdir(folder_path), 'The folder_path must be a directory!'
            folder_path_passed = True # flag for setting kwargs for proper dumping
            
            single_res_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.res' and f[-9:-3] != '.temp.']
            single_clu_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.clu' and f[-9:-3] != '.temp.']
            multi_res_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-6:-1] == '.res.' and f[-11:-5] != '.temp.']
            multi_clu_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-6:-1] == '.clu.' and f[-11:-5] != '.temp.']
            
            assert any(single_res_files) or any(single_clu_files) or any(multi_res_files) or any(multi_clu_files), \
                'No .res or .clu file formats found in the folder_path.'
            assert len(single_res_files) == 1 and len(single_clu_files) == 1, \
                'NeuroscopeSortingExtractor expects a single pair of .res and .clu files in the folder_path. Consider passing the file locations instead.'
            if any(multi_res_files) or any(multi_clu_files):
                print('Warning: Multiple .res and .clu files found in the folder_path; use the NeuroscopeMultiSortingExtractor to read all of them. Reading detected .res and .clu pair instead.')
            
            resfile_path = '{}/{}'.format(folder_path,single_res_files[0])
            clufile_path = '{}/{}'.format(folder_path,single_clu_files[0])
        else:
            folder_path_passed = False
            folder_path = split(Path(resfile_path).absolute())[0]
            
        abspath_resfile_name = split(Path(resfile_path).absolute())[1]
        abspath_clufile_name = split(Path(clufile_path).absolute())[1]
        
        if abspath_resfile_name[-1].isdigit(): 
            res_sorting_name = abspath_resfile_name[0:-6]
        else:
            res_sorting_name = abspath_resfile_name[0:-4]
        if abspath_clufile_name[-1].isdigit(): 
            clu_sorting_name = abspath_clufile_name[0:-6]
        else:
            clu_sorting_name = abspath_clufile_name[0:-4]
        assert res_sorting_name == clu_sorting_name, 'The .res and .clu files do not share the same name!'+res_sorting_name+'--'+clu_sorting_name
        
        xml_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.xml']
        assert any(xml_files), 'No .xml file found in the folder.'
        assert len(xml_files) == 1, 'More than one .xml file found in the folder.'
        xml_filepath = '{}/{}'.format(folder_path, xml_files[0])
        
        with open(xml_filepath, 'r') as xml_file:
            contents = xml_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            # Normally, this would be a .xml, but there were strange issues
            # in the write_recording method that require it to be a .lxml instead
            # which also requires all capital letters to be removed from the tag names
        
        self._sampling_frequency = float(soup.samplingrate.string) # careful not to confuse it with the lfpsamplingrate
        
        res = np.loadtxt(resfile_path, dtype=np.int64, usecols=0, ndmin=1)
        clu = np.loadtxt(clufile_path, dtype=np.int64, usecols=0, ndmin=1)
        if len(res) > 0:
            # Extract the number of clusters read as the first line of the clufile then remove it from the clu list
            n_clu = clu[0]
            clu = np.delete(clu, 0)
            unique_ids = np.unique(clu)

            if not unique_ids==np.arange(n_clu+1): # some missing IDs somewhere
                if 0 not in unique_ids: # missing unsorted IDs
                    n_clu += 1
                if 1 not in unique_ids: # missing mua IDs
                    n_clu += 1
                # If it is any other ID, then it would be very strange if it were missing...

            # Initialize spike trains and extract times from .res and appropriate clusters from .clu based on user input for ignoring multi-unit activity
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
                    
        if folder_path_passed:
            self._kwargs = {'resfile_path': None,
                            'clufile_path': None,
                            'folder_path': str(Path(folder_path).absolute()),
                            'keep_mua_units': keep_mua_units}
        else:
            self._kwargs = {'resfile_path': str(Path(resfile_path).absolute()),
                            'clufile_path': str(Path(clufile_path).absolute()),
                            'folder_path': None,
                            'keep_mua_units': keep_mua_units}


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
        abs_save_path = Path(save_path).absolute()
        _, SORTING_NAME = os.path.split(abs_save_path)
        XML_NAME = SORTING_NAME
        save_xml_filpath = '{}/{}.xml'.format(save_path,XML_NAME)
        
        _, SORTING_NAME = os.path.split(save_path)
            
        # Create and save .res and .clu files from the current sorting object
        save_res = "{}/{}.res".format(save_path,SORTING_NAME)
        save_clu = "{}/{}.clu".format(save_path,SORTING_NAME)
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
        if not os.path.isfile(save_xml_filpath):
            soup = BeautifulSoup("",'xml')

            new_tag = soup.new_tag('samplingrate')
            new_tag.string = str(sorting.get_sampling_frequency())
            soup.append(new_tag)

            # write parameters file
            with open(save_xml_filpath, "w") as f:
                f.write(str(soup))


class NeuroscopeMultiSortingExtractor(MultiSortingExtractor):

    """
    Extracts spiking information from an arbitrary number of .res.%i and .clu.%i files in the general folder path.
    
    The .res is a text file with a sorted list of all spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to the total number of unique ids in the file (and may exclude 0 & 1 from this count)
    with the rest of the rows indicating which unit id the corresponding entry in the .res file refers to.
    
    In the original Neuroscope format:
        Unit ID 0 is the cluster of unsorted spikes (noise).
        Unit ID 1 is a cluster of multi-unit spikes.
        
    The function defaults to returning multi-unit activity as the first index, and ignoring unsorted noise.
    To return only the fully sorted units, set keep_mua_units=False.
        
    The sorting extractor always returns unit IDs from 1, ..., number of chosen clusters.

    Parameters
    ----------
    folder_path : str
        Optional. Path to the collection of .res and .clu text files. Will auto-detect format.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    exclude_inds : list
        Optional. List of indices to ignore. The set of all possible indices is chosen by default, extracted as the final integer of all the .res.%i and .clu.%i pairs.
    """
    extractor_name = 'NeuroscopeMultiSortingExtractor'
    installed = HAVE_BS4_LXML  # check at class level if installed or not
    is_writable = True
    mode = 'custom'
    installation_mesg = 'Please install bs4 and lxml to use this extractor!'  # error message when not installed

    def __init__(self, folder_path, keep_mua_units=True, exclude_shanks=None):
        assert HAVE_BS4_LXML, self.installation_mesg
        
        assert isdir(folder_path), 'The folder_path must be a directory!'
        assert type(keep_mua_units) == bool, 'Optional argument "keep_mua_units" must be boolean!'
        if exclude_shanks is not None: # dumping checks do not like having an empty list as default
            assert type(exclude_shanks) == list, 'Optional argument "exclude_shanks" must be a list!'
            assert all([type(x) == int and x >= 0 for x in exclude_shanks]), 'Optional argument "exclude_shanks" must contain positive integers only!'
            exclude_shanks_passed = True
        else:
            exclude_shanks = []
            exclude_shanks_passed = False
            
        abs_folder_path = Path(folder_path).absolute()
        
        xml_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.xml']
        assert any(xml_files), 'No .xml file found in the folder.'
        assert len(xml_files) == 1, 'More than one .xml file found in the folder.'
        xml_filepath = '{}/{}'.format(folder_path, xml_files[0])
        
        # None of the location arguments were passed
        #assert folder_path is None and resfile_path is None and clufile_path is None, 'Either pass a single folder_path location, or a pair of resfile_path and clufile_path. None received.' # ToDo: examine the logic of this assertion and where it is breaking down
        with open(xml_filepath, 'r') as xml_file:
            contents = xml_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            # Normally, this would be a .xml, but there were strange issues
            # in the write_recording method that require it to be a .lxml instead
            # which also requires all capital letters to be removed from the tag names
        
        self._sampling_frequency = float(soup.samplingrate.string) # careful not to confuse it with the lfpsamplingrate
        
        single_res_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.res' and f[-9:-3] != '.temp.']
        single_clu_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-4:] == '.clu' and f[-9:-3] != '.temp.']
        multi_res_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-6:-1] == '.res.' and f[-11:-5] != '.temp.']
        multi_clu_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) if f[-6:-1] == '.clu.' and f[-11:-5] != '.temp.']
        
        assert any(single_res_files) or any(single_clu_files) or any(multi_res_files) or any(multi_clu_files), \
            'No .res or .clu file formats found in the folder_path.'
        if any(single_res_files) or any(single_clu_files):
            print('Warning: Single .res and .clu pairs found in the folder_path. These will be ignored by NeuroscopeMultiSortingExtractor; please use the NeuroscopeSortingExtractor to read these.')
        assert any(multi_res_files) or any(multi_clu_files), 'No .res.%i or .clu.%i files found in the directory!'
            
        res_ids = [x[-1] for x in multi_res_files]
        clu_ids = [x[-1] for x in multi_clu_files]
        assert res_ids==clu_ids, 'Unmatched .clu.%i and .res.%i files detected!'
        if any([x not in res_ids for x in exclude_shanks]):
            print('Warning: Detected indices in exclude_shanks that are not in the directory. These will be ignored.')
        
        resfile_names = [x[0:-6] for x in multi_res_files]
        clufile_names = [x[0:-6] for x in multi_clu_files]
        assert resfile_names==clufile_names, 'Some of the .res.%i and .clu.%i files do not share the same name!'
        SORTING_NAME = resfile_names[0]
        
        all_shanks_list_se = []
        for shank_id in list(set(res_ids) - set(exclude_shanks)):
            resfile_path = '{}/{}.res.{}'.format(abs_folder_path,SORTING_NAME,shank_id)
            clufile_path = '{}/{}.clu.{}'.format(abs_folder_path,SORTING_NAME,shank_id)
    
            all_shanks_list_se.append(NeuroscopeSortingExtractor(resfile_path=resfile_path,
                                                                 clufile_path=clufile_path,
                                                                 keep_mua_units=keep_mua_units))
                    

        MultiSortingExtractor.__init__(self,sortings=all_shanks_list_se)
            
        if exclude_shanks_passed:
            self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                            'keep_mua_units': keep_mua_units,
                            'exclude_shanks': exclude_shanks}
        else:
            self._kwargs = {'folder_path': str(Path(folder_path).absolute()),
                            'keep_mua_units': keep_mua_units,
                            'exclude_shanks': None}
    
    
    @staticmethod
    def write_sorting(multisorting, save_path):
        abs_save_path = Path(save_path).absolute()
        _, SORTING_NAME = os.path.split(abs_save_path)
        XML_NAME = SORTING_NAME
        save_xml_filpath = '{}/{}.xml'.format(save_path,XML_NAME)
        
        counter = 1
        for sorting in multisorting.sortings:
            # Create and save .res.%i and .clu.%i files from the current sorting object
            save_res = "{}/{}.res.{}".format(save_path,SORTING_NAME,counter)
            save_clu = "{}/{}.clu.{}".format(save_path,SORTING_NAME,counter)
            counter += 1
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
        if not os.path.isfile(save_xml_filpath):
            soup = BeautifulSoup("",'xml')

            new_tag = soup.new_tag('samplingrate')
            new_tag.string = str(sorting.get_sampling_frequency())
            soup.append(new_tag)

                        # write parameters file
            with open(save_xml_filpath, "w") as f:
                f.write(str(soup))