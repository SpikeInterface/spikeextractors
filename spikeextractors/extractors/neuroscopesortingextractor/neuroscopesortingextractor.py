from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id


class NeuroscopeSortingExtractor(SortingExtractor):

    """
    Extracts sorting information from pair of .res and .clu files . The .res is a text file with
    a sorted list of all spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to
    the total number of unit ids and the rest of the rows indicating which unit id the corresponding
    entry in the .res file refers to.
    
    In the original Neuroscope format:
        Unit ID 0 is the cluster of unsorted spikes (noise).
        Unit ID 1 is a cluster of multi-unit spikes.
        
    The function defaults to returning multi-unit activity.
    To return only the fully sorted units, set keep_mua_units=False.
        
    The sorting extractor always returns unit IDs from 1, ..., number of chosen clusters.
        
    Until get_unsorted_spike_train is implemented into the base for sortingextractor objects,
    that group will be ignored by this function (consistent with the original implementation).

    Parameters
    ----------
    resfile : str
        Path to the .res text file.
    clufile : str
        Path to the .clu text file.
    """
    extractor_name = 'NeuroscopeSortingExtractor'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'custom'

    def __init__(self, resfile_path, clufile_path):
        SortingExtractor.__init__(self, resfile_path, clufile_path, keep_mua_units=True)
        self._kwargs = {'resfile_path': str(Path(resfile).absolute()),
                        'clufile_path': str(Path(clufile).absolute()),
                        'keep_mua_units': keep_mua_units}
        
        
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
                self._unit_ids = list(x+1 for x in range(n_clu)) # generates list from 1,...,clu[0]-1
                for s_id in self._unit_ids:
                    self._spiketrains.append(res[(clu == s_id).nonzero()])
            else:
                # Ignoring IDs of 0 until get_unsorted_spike_train is implemented into base
                # Also ignoring IDs of 1 since user called keep_mua_units=False
                n_clu -= 2;
                self._unit_ids = list(x+1 for x in range(n_clu)) # generates list from 1,...,clu[0]-2
                for s_id in self._unit_ids:
                    self._spiketrains.append(res[(clu == s_id+1).nonzero()]) # only reading cluster IDs 2,...,clu[0]-1
        else:
            self._spiketrains = []
            self._unit_ids = []
        self._kwargs = {'resfile': str(Path(resfile).absolute()),
                        'clufile': str(Path(clufile).absolute())}


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
            clu = np.concatenate([np.repeat(i+2,len(st)) for i,st in enumerate(spiketrains)]).ravel()
            res_sort = np.argsort(res)
            res = res[res_sort]
            clu = clu[res_sort]
        else:
            res = []
            clu = []
        # add fake 'unit 1'
        # Cody: Why? Commenting this out as it seems problematic for the testing method
        # clu = np.insert(clu, 0, 1)
        # res = np.insert(res, 0, 1)
        # clu = np.insert(clu, 0, len(unit_ids)+1) # the + 1 seems to make the format believe there is an additional cluster
        clu = np.insert(clu, 0, len(unit_ids))

        np.savetxt(save_res, res, fmt='%i')
        np.savetxt(save_clu, clu, fmt='%i')
