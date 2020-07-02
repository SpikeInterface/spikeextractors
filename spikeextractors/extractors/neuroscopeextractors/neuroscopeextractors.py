from spikeextractors import RecordingExtractor,SortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_traces_args,check_valid_unit_id
from bs4 import BeautifulSoup
import os



class NeuroscopeRecordingExtractor(RecordingExtractor):
    
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
    installation_mesg = ""  # error message when not installed
            
    def __init__(self, folder_path):
        RecordingExtractor.__init__(self)
        self._recording_file = folder_path
        
        fpath_base, fname = os.path.split(folder_path)
        xml_filepath = os.path.join(folder_path, fname + '.xml')
        
        with open(xml_filepath, 'r') as xml_file:
            contents = xml_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            # Normally, this would be a .xml, but there were strange issues
            # in the write_recording method that require it to be a .lxml instead
            # which also requires all capital letters to be removed from the tag names
        
        n_bits = int(soup.nbits.string)
        num_channels = int(soup.nchannels.string)
        channel_ids = np.arange(num_channels)
        num_frames = int(soup.nsamples.string)
        sampling_frequency = float(soup.lfpsamplingrate.string)
        
        self._channel_ids = channel_ids
        self._num_frames = num_frames
        self._sampling_frequency = sampling_frequency
        
        dat_filepath = os.path.join(folder_path, fname + '.dat')
        self._recording = np.memmap(dat_filepath, mode='r', shape=(num_channels,num_frames), dtype='int'+str(n_bits)) # memmap reads row-wise
        
        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}
    
    def get_channel_ids(self):
        return self._channel_ids

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        return self._recording[channel_ids,start_frame:end_frame]
            
    @staticmethod
    def write_recording(recording, save_path):
        """ Convert and save the recording extractor to SHYBRID format

        parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved
        save_path: str
            Full path to desired target folder
        """
        RECORDING_NAME = save_path + '.dat'
        PARAMETERS_NAME = save_path + '.xml'

        # write recording
        recording_fn = os.path.join(save_path, RECORDING_NAME)
        BinDatRecordingExtractor.write_recording(recording, recording_fn,
                                                 time_axis=1, dtype=str(recording.get_dtype()))

        # create parameters file
        soup = BeautifulSoup("",'xml')
        
        new_tag = soup.new_tag('nbits')
        dataType = recording.get_dtype();
        assert any([dataType == x for x in ['int16', 'int32']]),"NeuroscopeRecordingExtractor only permits data of type 'int16' or 'int32'"
        nBits = str(dataType)[3:5]
        new_tag.string = str(nBits)
        soup.append(new_tag)
        
        new_tag = soup.new_tag('nchannels')
        new_tag.string = str(len(recording.get_channel_ids()))
        soup.append(new_tag)
        
        new_tag = soup.new_tag('nsamples')
        new_tag.string = str(recording.get_num_frames())
        soup.append(new_tag)
        
        new_tag = soup.new_tag('lfpsamplingrate')
        new_tag.string = str(recording.get_sampling_frequency())
        soup.append(new_tag)

        # write parameters file
        parameters_fn = os.path.join(save_path, PARAMETERS_NAME)
        f = open(parameters_fn, "w")
        f.write(str(soup))
        f.close()
        

class NeuroscopeSortingExtractor(SortingExtractor):

    """
    Extracts spiking information from pair of .res and .clu files. The .res is a text file with
    a sorted list of all spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to
    the total number of unit ids and the rest of the rows indicating which unit id the corresponding
    entry in the .res file refers to.
    
    In the original Neuroscope format:
        Unit ID 0 is the cluster of unsorted spikes (noise).
        Unit ID 1 is a cluster of multi-unit spikes.
        
    The function defaults to returning multi-unit activity, and ignoring unsorted noise.
    To return only the fully sorted units, set keep_mua_units=False.
        
    The sorting extractor always returns unit IDs from 1, ..., number of chosen clusters.

    Parameters
    ----------
    resfile_path : str
        Path to the .res text file.
    clufile_path : str
        Path to the .clu text file.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    """
    extractor_name = 'NeuroscopeSortingExtractor'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'custom'
    installation_mesg = ""  # error message when not installed

    def __init__(self, resfile_path, clufile_path, keep_mua_units=True):
        SortingExtractor.__init__(self)
        
        res = np.loadtxt(resfile_path, dtype=np.int64, usecols=0, ndmin=1)
        clu = np.loadtxt(clufile_path, dtype=np.int64, usecols=0, ndmin=1)
        if len(res) > 0:
            # Extract the number of clusters read as the first line of the clufile
            # then remove it from the clu list
            n_clu = clu[0]
            clu = np.delete(clu, 0)
            
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
            
        self._kwargs = {'resfile_path': str(Path(resfile_path).absolute()),
                        'clufile_path': str(Path(clufile_path).absolute()),
                        'keep_mua_units': keep_mua_units}


    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
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
        save_res = "{}.res".format(save_path)
        save_clu = "{}.clu".format(save_path)
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
        clu = np.insert(clu, 0, len(unit_ids)+1) # The +1 is necessary here b/c the convention for the base sorting object is from 1,...,nUnits

        np.savetxt(save_res, res, fmt='%i')
        np.savetxt(save_clu, clu, fmt='%i')
