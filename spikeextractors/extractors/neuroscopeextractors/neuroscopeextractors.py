from spikeextractors import RecordingExtractor, SortingExtractor, MultiSortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id, get_sub_extractors_by_property
from typing import Union
import os

try:
    from lxml import etree as et

    HAVE_LXML = True
except ImportError:
    HAVE_LXML = False

PathType = Union[str, Path, None]
DtypeType = Union[str, np.dtype, None]


class NeuroscopeRecordingExtractor(BinDatRecordingExtractor):
    """
    Extracts raw neural recordings from large binary .dat files in the neuroscope format.
    
    The recording extractor always returns channel IDs starting from 0.
    
    The recording data will always be returned in the shape of (num_channels,num_frames).

    Parameters
    ----------
    file_path : str
        Path to the .dat file to be extracted
    """
    extractor_name = 'NeuroscopeRecordingExtractor'
    installed = HAVE_LXML  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = 'Please install lxml to use this extractor!'  # error message when not installed

    def __init__(self, file_path: PathType):
        assert HAVE_LXML, self.installation_mesg
        file_path = Path(file_path)
        assert file_path.is_file() and file_path.suffix == '.dat', 'file_path must lead to a .dat file!'

        RecordingExtractor.__init__(self)
        self._recording_file = file_path
        file_path = Path(file_path)
        folder_path = file_path.parent

        xml_files = [f for f in folder_path.iterdir() if f.is_file() if f.suffix == '.xml']
        assert any(xml_files), 'No .xml file found in the folder_path.'
        assert len(xml_files) == 1, 'More than one .xml file found in the folder_path.'
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath.absolute())).getroot()
        n_bits = int(xml_root.find('acquisitionSystem').find('nBits').text)
        dtype = 'int' + str(n_bits)
        numchan_from_file = int(xml_root.find('acquisitionSystem').find('nChannels').text)
        sampling_frequency = float(xml_root.find('acquisitionSystem').find('samplingRate').text)

        BinDatRecordingExtractor.__init__(self, file_path, sampling_frequency=sampling_frequency,
                                          dtype=dtype, numchan=numchan_from_file)

        self._kwargs = {'file_path': str(Path(file_path).absolute())}

    @staticmethod
    def write_recording(recording: RecordingExtractor, save_path: PathType, dtype: DtypeType = None,
                        **write_binary_kwargs):
        """
        Convert and save the recording extractor to Neuroscope format.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be converted and saved.
        save_path: str
            Path to desired target folder. The name of the files will be the same as the final directory.
        dtype: dtype
            Optional. Data type to be used in writing; must be int16 or int32 (default).
                      Will throw a warning if stored recording type from get_traces() does not match.
        **write_binary_kwargs: keyword arguments for write_to_binary_dat_format function
            - chunk_size
            - chunk_mb
        """
        save_path = Path(save_path)

        if not save_path.is_dir():
            os.makedirs(save_path)

        if save_path.suffix == '':
            recording_name = save_path.name
        else:
            recording_name = save_path.stem
        xml_name = recording_name

        save_xml_filepath = save_path / (str(xml_name) + '.xml')
        recording_filepath = save_path / recording_name

        # create parameters file if none exists
        if save_xml_filepath.is_file():
            raise FileExistsError(f'{save_xml_filepath} already exists!')

        xml_root = et.Element('xml')
        et.SubElement(xml_root, 'acquisitionSystem')
        et.SubElement(xml_root.find('acquisitionSystem'), 'nBits')
        et.SubElement(xml_root.find('acquisitionSystem'), 'nChannels')
        et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')

        recording_dtype = str(recording.get_dtype())
        int_loc = recording_dtype.find('int')
        recording_n_bits = recording_dtype[(int_loc + 3):(int_loc + 5)]

        if dtype is None:  # user did not specify data type
            if int_loc != -1 and recording_n_bits in ['16', '32']:
                n_bits = recording_n_bits
            else:
                print('Warning: Recording data type must be int16 or int32! Defaulting to int32.')
                n_bits = '32'
            dtype = 'int' + n_bits  # update dtype in pass to BinDatRecordingExtractor.write_recording
        else:
            dtype = str(dtype)  # if user passed numpy data type
            int_loc = dtype.find('int')
            assert int_loc != -1, 'Data type must be int16 or int32! Non-integer received.'
            n_bits = dtype[(int_loc + 3):(int_loc + 5)]
            assert n_bits in ['16', '32'], 'Data type must be int16 or int32!'

        xml_root.find('acquisitionSystem').find('nBits').text = n_bits
        xml_root.find('acquisitionSystem').find('nChannels').text = str(recording.get_num_channels())
        xml_root.find('acquisitionSystem').find('samplingRate').text = str(recording.get_sampling_frequency())

        et.ElementTree(xml_root).write(str(save_xml_filepath.absolute()), pretty_print=True)

        recording.write_to_binary_dat_format(recording_filepath, dtype=dtype, **write_binary_kwargs)


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
    installed = HAVE_LXML  # check at class level if installed or not
    is_writable = True
    mode = 'custom'
    installation_mesg = 'Please install lxml to use this extractor!'  # error message when not installed

    def __init__(self, resfile_path: PathType = None, clufile_path: PathType = None, folder_path: PathType = None,
                 keep_mua_units: bool = True):
        assert HAVE_LXML, self.installation_mesg

        # None of the location arguments were passed
        assert not (folder_path is None and resfile_path is None and clufile_path is None), \
            'Either pass a single folder_path location, or a pair of resfile_path and clufile_path. None received.'

        # At least one file_path passed
        if resfile_path is not None:
            assert clufile_path is not None, 'If passing resfile_path or clufile_path, both are required.'
            resfile_path = Path(resfile_path)
            clufile_path = Path(clufile_path)
            assert resfile_path.is_file() and clufile_path.is_file(), \
                'The resfile_path and clufile_path must be .res and .clu files!'

            assert folder_path is None, 'Pass either a single folder_path location, ' \
                                        'or a pair of resfile_path and clufile_path. All received.'
            folder_path_passed = False
            folder_path = resfile_path.parent
        else:
            assert folder_path is not None, 'Either pass resfile_path and clufile_path, or folder_path'
            folder_path = Path(folder_path)
            assert folder_path.is_dir(), 'The folder_path must be a directory!'

            res_files = [f for f in folder_path.iterdir() if f.is_file()
                         and '.res' in f.name and '.temp.' not in f.name]
            clu_files = [f for f in folder_path.iterdir() if f.is_file()
                         and '.clu' in f.name and '.temp.' not in f.name]

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

            if not np.sort(unique_ids) == np.arange(n_clu + 1):  # some missing IDs somewhere
                if 0 not in unique_ids:  # missing unsorted IDs
                    n_clu += 1
                if 1 not in unique_ids:  # missing mua IDs
                    n_clu += 1
                # If it is any other ID, then it would be very strange if it were missing...

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
    def write_sorting(sorting: SortingExtractor, save_path: PathType):
        # if multiple groups, use the NeuroscopeMultiSortingExtactor write function
        if 'group' in sorting.get_shared_unit_property_names():
            NeuroscopeMultiSortingExtractor.write_sorting(sorting, save_path)
        else:
            if not save_path.is_dir():
                os.makedirs(save_path)

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


class NeuroscopeMultiSortingExtractor(MultiSortingExtractor):
    """
    Extracts spiking information from an arbitrary number of .res.%i and .clu.%i files in the general folder path.
    
    The .res is a text file with a sorted list of all spiketimes from all units displayed in sample (integer '%i') units.
    The .clu file is a file with one more row than the .res with the first row corresponding to the total number of
    unique ids in the file (and may exclude 0 & 1 from this count)
    with the rest of the rows indicating which unit id the corresponding entry in the .res file refers to.
    The group id is loaded as unit property 'group'.
    
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
    exclude_shanks : list
        Optional. List of indices to ignore. The set of all possible indices is chosen by default, extracted as the
        final integer of all the .res.%i and .clu.%i pairs.
    """
    extractor_name = 'NeuroscopeMultiSortingExtractor'
    installed = HAVE_LXML  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = 'Please install lxml to use this extractor!'  # error message when not installed

    def __init__(self, folder_path: PathType, keep_mua_units: bool = True, exclude_shanks: Union[list, None] = None):
        assert HAVE_LXML, self.installation_mesg

        folder_path = Path(folder_path)

        if not folder_path.is_dir():
            os.makedirs(folder_path)

        if exclude_shanks is not None:  # dumping checks do not like having an empty list as default
            assert all([isinstance(x, (int, np.integer)) and x >= 0 for x in
                        exclude_shanks]), 'Optional argument "exclude_shanks" must contain positive integers only!'
            exclude_shanks_passed = True
        else:
            exclude_shanks = []
            exclude_shanks_passed = False
        xml_files = [f for f in folder_path.iterdir() if f.is_file if f.suffix == '.xml']
        assert len(xml_files) > 0, 'No .xml file found in the folder.'
        assert len(xml_files) == 1, 'More than one .xml file found in the folder.'
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath.absolute())).getroot()
        self._sampling_frequency = float(xml_root.find('acquisitionSystem').find(
            'samplingRate').text)  # careful not to confuse it with the lfpsamplingrate

        res_files = [f for f in folder_path.iterdir() if f.is_file()
                     and '.res' in f.name and '.temp.' not in f.name]
        clu_files = [f for f in folder_path.iterdir() if f.is_file()
                     and '.clu' in f.name and '.temp.' not in f.name]

        assert len(res_files) > 0 or len(clu_files) > 0, \
            'No .res or .clu files found in the folder_path.'
        assert len(res_files) > 1 and len(clu_files) > 1, \
            'Single .res and .clu pairs found in the folder_path. ' \
            'For single .res and .clu files, use the NeuroscopeSortingExtractor instead.'
        assert len(res_files) == len(clu_files)

        res_ids = [int(x.name[-1]) for x in res_files]
        clu_ids = [int(x.name[-1]) for x in res_files]
        assert sorted(res_ids) == sorted(clu_ids), 'Unmatched .clu.%i and .res.%i files detected!'
        if any([x not in res_ids for x in exclude_shanks]):
            print('Warning: Detected indices in exclude_shanks that are not in the directory. These will be ignored.')

        resfile_names = [x.name[:x.name.find('.res')] for x in res_files]
        clufile_names = [x.name[:x.name.find('.clu')] for x in clu_files]
        assert np.all(r == c for (r, c) in zip(resfile_names, clufile_names)), \
            'Some of the .res.%i and .clu.%i files do not share the same name!'
        sorting_name = resfile_names[0]

        all_shanks_list_se = []
        for shank_id in list(set(res_ids) - set(exclude_shanks)):
            resfile_path = folder_path / f'{sorting_name}.res.{shank_id}'
            clufile_path = folder_path / f'{sorting_name}.clu.{shank_id}'

            all_shanks_list_se.append(NeuroscopeSortingExtractor(resfile_path=resfile_path,
                                                                 clufile_path=clufile_path,
                                                                 keep_mua_units=keep_mua_units))

        MultiSortingExtractor.__init__(self, sortings=all_shanks_list_se)

        if exclude_shanks_passed:
            self._kwargs = {'folder_path': str(folder_path.absolute()),
                            'keep_mua_units': keep_mua_units,
                            'exclude_shanks': exclude_shanks}
        else:
            self._kwargs = {'folder_path': str(folder_path.absolute()),
                            'keep_mua_units': keep_mua_units,
                            'exclude_shanks': None}

    @staticmethod
    def write_sorting(sorting: Union[MultiSortingExtractor, SortingExtractor], save_path: PathType):
        save_path = Path(save_path)
        if save_path.suffix == '':
            sorting_name = save_path.name
        else:
            sorting_name = save_path.stem
        xml_name = sorting_name
        save_xml_filepath = save_path / (str(xml_name) + '.xml')

        assert not save_path.is_file(), "'save_path' should be a folder"
        if not save_path.is_dir():
            os.makedirs(save_path)

        if save_xml_filepath.is_file():
            raise FileExistsError(f'{save_xml_filepath} already exists!')

        xml_root = et.Element('xml')
        et.SubElement(xml_root, 'acquisitionSystem')
        et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')
        xml_root.find('acquisitionSystem').find('samplingRate').text = str(sorting.get_sampling_frequency())
        et.ElementTree(xml_root).write(str(save_xml_filepath.absolute()), pretty_print=True)

        if isinstance(sorting, MultiSortingExtractor):
            counter = 1
            for sort in sorting.sortings:
                # Create and save .res.%i and .clu.%i files from the current sorting object
                save_res = save_path / f'{sorting_name}.res.{counter}'
                save_clu = save_path / f'{sorting_name}.clu.{counter}'
                counter += 1

                res, clu = _extract_res_clu_arrays(sort)

                np.savetxt(save_res, res, fmt='%i')
                np.savetxt(save_clu, clu, fmt='%i')

        elif isinstance(sorting, SortingExtractor):
            # assert units have group property
            assert 'group' in sorting.get_shared_unit_property_names()
            sortings, groups = get_sub_extractors_by_property(sorting, 'group', return_property_list=True)

            for (sort, group) in zip(sortings, groups):
                # Create and save .res.%i and .clu.%i files from the current sorting object
                save_res = save_path / f'{sorting_name}.res.{group}'
                save_clu = save_path / f'{sorting_name}.clu.{group}'

                res, clu = _extract_res_clu_arrays(sort)

                np.savetxt(save_res, res, fmt='%i')
                np.savetxt(save_clu, clu, fmt='%i')


def _extract_res_clu_arrays(sorting):
    unit_ids = sorting.get_unit_ids()
    if len(unit_ids) > 0:
        spiketrains = [sorting.get_unit_spike_train(u) for u in unit_ids]
        res = np.concatenate(spiketrains).ravel()
        clu = np.concatenate(
            [np.repeat(i + 1, len(st)) for i, st in enumerate(spiketrains)]).ravel()  # i here counts from 0
        res_sort = np.argsort(res)
        res = res[res_sort]
        clu = clu[res_sort]

        unique_ids = np.unique(clu)
        n_clu = len(unique_ids)
        clu = np.insert(clu, 0, n_clu)  # The +1 is necessary becuase the base sorting object is from 1,...,nUnits
    else:
        res, clu = [], []

    return res, clu
