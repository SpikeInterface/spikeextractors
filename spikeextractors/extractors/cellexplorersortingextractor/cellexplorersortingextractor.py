from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id, get_sub_extractors_by_property
from typing import Union
import re
from scipy.io import loadmat

PathType = Union[str, Path, None]
DtypeType = Union[str, np.dtype, None]


class CellExplorerSortingExtractor(SortingExtractor):
    """
    Extracts spiking information from .mat files stored in the CellExplorer format.

    Parameters
    ----------
    matfile_path : str
        Optional. Path to the .mat file.
    """

    extractor_name = 'CellExplorerSortingExtractor'
    installed = True
    is_writable = True
    mode = 'custom'
    installation_mesg = ""

    def __init__(self, matfile_path: PathType):

        # At least one file_path passed
        if resfile_path is not None:
            assert clufile_path is not None, 'If passing resfile_path or clufile_path, both are required.'
            resfile_path = Path(resfile_path)
            clufile_path = Path(clufile_path)
            assert resfile_path.is_file() and clufile_path.is_file(), \
                'The resfile_path ({}) and clufile_path ({}) must be .res and .clu files!'.format(resfile_path,
                                                                                                  clufile_path)

            assert folder_path is None, 'Pass either a single folder_path location, ' \
                                        'or a pair of resfile_path and clufile_path. All received.'
            folder_path_passed = False
            folder_path = resfile_path.parent
        else:
            assert folder_path is not None, 'Either pass resfile_path and clufile_path, or folder_path'
            folder_path = Path(folder_path)
            assert folder_path.is_dir(), 'The folder_path must be a directory!'

            res_files = [f for f in folder_path.iterdir() if f.is_file()
                         and '.res' in f.suffixes 
                         and '.temp' not in f.suffixes
                         and not f.name.endswith('~')
                         and len(f.suffixes) == 1]
            clu_files = [f for f in folder_path.iterdir() if f.is_file()
                         and '.clu' in f.suffixes 
                         and not f.name.endswith('~')
                         and len(f.suffixes) == 1]

            assert len(res_files) > 0 or len(clu_files) > 0, \
                'No .res or .clu files found in the folder_path.'
            assert len(res_files) == 1 and len(clu_files) == 1, \
                'NeuroscopeSortingExtractor expects a single pair of .res and .clu files in the folder_path. ' \
                'For multiple .res and .clu files, use the NeuroscopeMultiSortingExtractor instead.'

            folder_path_passed = True  # flag for setting kwargs for proper dumping
            resfile_path = res_files[0]
            clufile_path = clu_files[0]

        SortingExtractor.__init__(self)

        res_sorting_name = resfile_path.name[:resfile_path.name.find('.res')]
        clu_sorting_name = clufile_path.name[:clufile_path.name.find('.clu')]

        assert res_sorting_name == clu_sorting_name, f'The .res and .clu files do not share the same name! ' \
                                                     f'{res_sorting_name}  -- {clu_sorting_name}'

        xml_files = [f for f in folder_path.iterdir() if f.is_file() if f.suffix == '.xml']
        assert len(xml_files) > 0, 'No .xml file found in the folder.'
        assert len(xml_files) == 1, 'More than one .xml file found in the folder.'
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath.absolute())).getroot()
        self._sampling_frequency = float(xml_root.find('acquisitionSystem').find(
            'samplingRate').text)  # careful not to confuse it with the lfpsamplingrate

        res = np.loadtxt(resfile_path, dtype=np.int64, usecols=0, ndmin=1)
        clu = np.loadtxt(clufile_path, dtype=np.int64, usecols=0, ndmin=1)

        if len(res) > 0:
            # Extract the number of clusters read as the first line of the clufile then remove it from the clu list
            n_clu = clu[0]
            clu = np.delete(clu, 0)
            unique_ids = np.unique(clu)
            if 0 not in unique_ids:  # missing unsorted IDs
                n_clu += 1
            if 1 not in unique_ids:  # missing mua IDs
                n_clu += 1

            # Initialize spike trains and extract times from .res and appropriate clusters from .clu based on
            # user input for ignoring multi-unit activity
            self._spiketrains = []
            if keep_mua_units:  # default
                n_clu -= 1
                self._unit_ids = [x + 1 for x in range(n_clu)]  # generates list from 1,...,clu[0]-1
                for s_id in self._unit_ids:
                    self._spiketrains.append(res[(clu == s_id).nonzero()])
            else:
                # Ignoring IDs of 0 until get_unsorted_spike_train is implemented into base
                # Also ignoring IDs of 1 since user called keep_mua_units=False
                n_clu -= 2
                self._unit_ids = [x + 1 for x in range(n_clu)]  # generates list from 1,...,clu[0]-2
                for s_id in self._unit_ids:
                    self._spiketrains.append(
                        res[(clu == s_id + 1).nonzero()])  # only reading cluster IDs 2,...,clu[0]-1

        if folder_path_passed:
            self._kwargs = {'resfile_path': None,
                            'clufile_path': None,
                            'folder_path': str(folder_path.absolute()),
                            'keep_mua_units': keep_mua_units}
        else:
            self._kwargs = {'resfile_path': str(resfile_path.absolute()),
                            'clufile_path': str(clufile_path.absolute()),
                            'folder_path': None,
                            'keep_mua_units': keep_mua_units}

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def shift_unit_ids(self, shift):
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
    def write_sorting(sorting: SortingExtractor, save_path: PathType):
        # if multiple groups, use the NeuroscopeMultiSortingExtactor write function
        if 'group' in sorting.get_shared_unit_property_names():
            NeuroscopeMultiSortingExtractor.write_sorting(sorting, save_path)
        else:
            save_path.mkdir(parents=True, exist_ok=True)

            if save_path.suffix == '':
                sorting_name = save_path.name
            else:
                sorting_name = save_path.stem
            xml_name = sorting_name
            save_xml_filepath = save_path / (str(xml_name) + '.xml')

            # create parameters file if none exists
            if save_xml_filepath.is_file():
                raise FileExistsError(f'{save_xml_filepath} already exists!')

            xml_root = et.Element('xml')
            et.SubElement(xml_root, 'acquisitionSystem')
            et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')
            xml_root.find('acquisitionSystem').find('samplingRate').text = str(sorting.get_sampling_frequency())
            et.ElementTree(xml_root).write(str(save_xml_filepath.absolute()), pretty_print=True)

            # Create and save .res and .clu files from the current sorting object
            save_res = save_path / f'{sorting_name}.res'
            save_clu = save_path / f'{sorting_name}.clu'

            res, clu = _extract_res_clu_arrays(sorting)

            np.savetxt(save_res, res, fmt='%i')
            np.savetxt(save_clu, clu, fmt='%i')
