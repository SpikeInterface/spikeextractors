from spikeextractors import RecordingExtractor, MultiRecordingTimeExtractor, SortingExtractor, MultiSortingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id, get_sub_extractors_by_property
from typing import Union, Optional
import re
import warnings
from typing import Optional

try:
    from lxml import etree as et

    HAVE_LXML = True
except ImportError:
    HAVE_LXML = False

PathType = Union[str, Path]
OptionalPathType = Optional[PathType]
DtypeType = Union[str, np.dtype, None]


def get_single_files(folder_path: Path, suffix: str):
    return [
        f for f in folder_path.iterdir() if f.is_file() and suffix in f.suffixes and not f.name.endswith("~")
        and len(f.suffixes) == 1
    ]


def get_shank_files(folder_path: Path, suffix: str):
    return [
        f for f in folder_path.iterdir() if f.is_file() and suffix in f.suffixes
        and re.search(r"\d+$", f.name) is not None and len(f.suffixes) == 2
    ]


class NeuroscopeRecordingExtractor(BinDatRecordingExtractor):
    """
    Extracts raw neural recordings from large binary .dat files in the neuroscope format.

    The recording extractor always returns channel IDs starting from 0.

    The recording data will always be returned in the shape of (num_channels,num_frames).

    Parameters
    ----------
    file_path : str
        Path to the .dat file to be extracted.
    gain : float, optional
        Numerical value that converts the native int dtype to microvolts. Defaults to 1.
    """

    extractor_name = "NeuroscopeRecordingExtractor"
    installed = HAVE_LXML
    is_writable = True
    mode = "file"
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(self, file_path: PathType, gain: Optional[float] = None):
        assert HAVE_LXML, self.installation_mesg
        file_path = Path(file_path)
        assert file_path.is_file() and file_path.suffix in [".dat", ".eeg"], \
            "file_path must lead to a .dat or .eeg file!"

        RecordingExtractor.__init__(self)
        self._recording_file = file_path
        file_path = Path(file_path)
        folder_path = file_path.parent

        xml_files = [f for f in folder_path.iterdir() if f.is_file() if f.suffix == ".xml"]
        assert any(xml_files), "No .xml file found in the folder_path."
        assert len(xml_files) == 1, "More than one .xml file found in the folder_path."
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath)).getroot()
        n_bits = int(xml_root.find('acquisitionSystem').find('nBits').text)
        dtype = f"int{n_bits}"
        numchan_from_file = int(xml_root.find('acquisitionSystem').find('nChannels').text)

        if file_path.suffix == ".dat":
            sampling_frequency = float(xml_root.find('acquisitionSystem').find('samplingRate').text)
        else:
            sampling_frequency = float(xml_root.find('fieldPotentials').find('lfpSamplingRate').text)

        BinDatRecordingExtractor.__init__(self, file_path, sampling_frequency=sampling_frequency,
                                          dtype=dtype, numchan=numchan_from_file)

        if gain is not None:
            self.set_channel_gains(channel_ids=self.get_channel_ids(), gains=gain)

        self._kwargs = dict(file_path=str(Path(file_path).absolute()), gain=gain)

    @staticmethod
    def write_recording(
        recording: RecordingExtractor,
        save_path: PathType,
        dtype: DtypeType = None,
        **write_binary_kwargs
    ):
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
        save_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == "":
            recording_name = save_path.name
        else:
            recording_name = save_path.stem
        xml_name = recording_name

        save_xml_filepath = save_path / f"{xml_name}.xml"
        recording_filepath = save_path / recording_name

        # create parameters file if none exists
        if save_xml_filepath.is_file():
            raise FileExistsError(f"{save_xml_filepath} already exists!")

        xml_root = et.Element('xml')
        et.SubElement(xml_root, 'acquisitionSystem')
        et.SubElement(xml_root.find('acquisitionSystem'), 'nBits')
        et.SubElement(xml_root.find('acquisitionSystem'), 'nChannels')
        et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')

        recording_dtype = str(recording.get_dtype())
        int_loc = recording_dtype.find('int')
        recording_n_bits = recording_dtype[(int_loc + 3):(int_loc + 5)]

        valid_dtype = ["16", "32"]
        if dtype is None:
            if int_loc != -1 and recording_n_bits in valid_dtype:
                n_bits = recording_n_bits
            else:
                print("Warning: Recording data type must be int16 or int32! Defaulting to int32.")
                n_bits = "32"
            dtype = f"int{n_bits}"  # update dtype in pass to BinDatRecordingExtractor.write_recording
        else:
            dtype = str(dtype)  # if user passed numpy data type
            int_loc = dtype.find('int')
            assert int_loc != -1, "Data type must be int16 or int32! Non-integer received."
            n_bits = dtype[(int_loc + 3):(int_loc + 5)]
            assert n_bits in valid_dtype, "Data type must be int16 or int32!"

        xml_root.find('acquisitionSystem').find('nBits').text = n_bits
        xml_root.find('acquisitionSystem').find('nChannels').text = str(recording.get_num_channels())
        xml_root.find('acquisitionSystem').find('samplingRate').text = str(recording.get_sampling_frequency())

        et.ElementTree(xml_root).write(str(save_xml_filepath), pretty_print=True)

        recording.write_to_binary_dat_format(recording_filepath, dtype=dtype, **write_binary_kwargs)


class NeuroscopeMultiRecordingTimeExtractor(MultiRecordingTimeExtractor):
    """
    Extracts raw neural recordings from several binary .dat files in the neuroscope format.

    The recording extractor always returns channel IDs starting from 0.

    The recording data will always be returned in the shape of (num_channels,num_frames).

    Parameters
    ----------
    folder_path : PathType
        Path to the .dat files to be extracted.
    gain : float, optional
        Numerical value that converts the native int dtype to microvolts. Defaults to 1.
    """

    extractor_name = "NeuroscopeMultiRecordingTimeExtractor"
    installed = HAVE_LXML
    is_writable = True
    mode = "folder"
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(self, folder_path: PathType, gain: Optional[float] = None):
        assert HAVE_LXML, self.installation_mesg

        folder_path = Path(folder_path)
        recording_files = [x for x in folder_path.iterdir() if x.is_file() and x.suffix == ".dat"]
        assert any(recording_files), "The folder_path must lead to at least one .dat file!"

        recordings = [NeuroscopeRecordingExtractor(file_path=x, gain=gain) for x in recording_files]
        MultiRecordingTimeExtractor.__init__(self, recordings=recordings)

        self._kwargs = dict(folder_path=str(folder_path.absolute()), gain=gain)

    @staticmethod
    def write_recording(
        recording: Union[MultiRecordingTimeExtractor, RecordingExtractor],
        save_path: PathType,
        dtype: DtypeType = None,
        **write_binary_kwargs
    ):
        """
        Convert and save the recording extractor to Neuroscope format.

        Parameters
        ----------
        recording: MultiRecordingTimeExtractor or RecordingExtractor
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
        save_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == "":
            recording_name = save_path.name
        else:
            recording_name = save_path.stem

        xml_name = recording_name
        save_xml_filepath = save_path / f"{xml_name}.xml"
        if save_xml_filepath.is_file():
            raise FileExistsError(f"{save_xml_filepath} already exists!")

        recording_dtype = str(recording.get_dtype())
        int_loc = recording_dtype.find("int")
        recording_n_bits = recording_dtype[(int_loc + 3):(int_loc + 5)]

        valid_int_types = ["16", "32"]
        if dtype is None:
            if int_loc != -1 and recording_n_bits in valid_int_types:
                n_bits = recording_n_bits
            else:
                warnings.warn("Recording data type must be int16 or int32! Defaulting to int32.")
                n_bits = "32"
            dtype = f"int{n_bits}"
        else:
            dtype = str(dtype)
            int_loc = dtype.find('int')
            assert int_loc != -1, "Data type must be int16 or int32! Non-integer received."
            n_bits = dtype[(int_loc + 3):(int_loc + 5)]
            assert n_bits in valid_int_types, "Data type must be int16 or int32!"

        xml_root = et.Element('xml')
        et.SubElement(xml_root, 'acquisitionSystem')
        et.SubElement(xml_root.find('acquisitionSystem'), 'nBits')
        et.SubElement(xml_root.find('acquisitionSystem'), 'nChannels')
        et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')
        xml_root.find('acquisitionSystem').find('nBits').text = n_bits
        xml_root.find('acquisitionSystem').find('nChannels').text = str(recording.get_num_channels())
        xml_root.find('acquisitionSystem').find('samplingRate').text = str(recording.get_sampling_frequency())
        et.ElementTree(xml_root).write(str(save_xml_filepath.absolute()), pretty_print=True)

        if isinstance(recording, MultiRecordingTimeExtractor):
            for n, record in enumerate(recording.recordings):
                epoch_id = str(n).zfill(2)  # Neuroscope seems to zero-pad length 2
                record.write_to_binary_dat_format(
                    save_path=save_path / f"{recording_name}-{epoch_id}.dat",
                    dtype=dtype,
                    **write_binary_kwargs
                )

        elif isinstance(recording, RecordingExtractor):
            recordings = [recording.get_epoch(epoch_name=epoch_name) for epoch_name in recording.get_epoch_names()]

            if len(recordings) == 0:
                recording.write_to_binary_dat_format(
                    save_path=save_path / f"{recording_name}.dat",
                    dtype=dtype,
                    **write_binary_kwargs
                )
            else:
                for n, subrecording in enumerate(recordings):
                    epoch_id = str(n).zfill(2)  # Neuroscope seems to zero-pad length 2
                    subrecording.write_to_binary_dat_format(
                        save_path=save_path / f"{recording_name}-{epoch_id}.dat",
                        dtype=dtype,
                        **write_binary_kwargs
                    )


class NeuroscopeSortingExtractor(SortingExtractor):
    """
    Extracts spiking information from pair of .res and .clu files.

    The .res is a text file with a sorted list of spiketimes from all units displayed in sample (integer '%i') units.
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
    resfile_path : PathType
        Optional. Path to a particular .res text file.
    clufile_path : PathType
        Optional. Path to a particular .clu text file.
    folder_path : PathType
        Optional. Path to the collection of .res and .clu text files. Will auto-detect format.
    keep_mua_units : bool
        Optional. Whether or not to return sorted spikes from multi-unit activity. Defaults to True.
    spkfile_path : PathType
        Optional. Path to a particular .spk binary file containing waveform snippets added to the extractor as features.
    gain : float
        Optional. If passing a spkfile_path, this value converts the data type of the waveforms to units of microvolts.
    """

    extractor_name = "NeuroscopeSortingExtractor"
    installed = HAVE_LXML
    is_writable = True
    mode = "custom"
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(
        self,
        resfile_path: OptionalPathType = None,
        clufile_path: OptionalPathType = None,
        folder_path: OptionalPathType = None,
        keep_mua_units: bool = True,
        spkfile_path: OptionalPathType = None,
        gain: Optional[float] = None
    ):
        assert HAVE_LXML, self.installation_mesg
        assert not (folder_path is None and resfile_path is None and clufile_path is None), \
            "Either pass a single folder_path location, or a pair of resfile_path and clufile_path! None received."

        if resfile_path is not None:
            assert clufile_path is not None, "If passing resfile_path or clufile_path, both are required!"
            resfile_path = Path(resfile_path)
            clufile_path = Path(clufile_path)
            assert resfile_path.is_file() and clufile_path.is_file(), \
                f"The resfile_path ({resfile_path}) and clufile_path ({clufile_path}) must be .res and .clu files!"

            assert folder_path is None, "Pass either a single folder_path location, " \
                                        "or a pair of resfile_path and clufile_path! All received."
            folder_path_passed = False
            folder_path = resfile_path.parent
        else:
            assert folder_path is not None, "Either pass resfile_path and clufile_path, or folder_path!"
            folder_path = Path(folder_path)
            assert folder_path.is_dir(), "The folder_path must be a directory!"

            res_files = get_single_files(folder_path=folder_path, suffix=".res")
            clu_files = get_single_files(folder_path=folder_path, suffix=".clu")

            assert len(res_files) > 0 or len(clu_files) > 0, \
                "No .res or .clu files found in the folder_path!"
            assert len(res_files) == 1 and len(clu_files) == 1, \
                "NeuroscopeSortingExtractor expects a single pair of .res and .clu files in the folder_path. " \
                "For multiple .res and .clu files, use the NeuroscopeMultiSortingExtractor instead."

            folder_path_passed = True  # flag for setting kwargs for proper dumping
            resfile_path = res_files[0]
            clufile_path = clu_files[0]

        SortingExtractor.__init__(self)

        res_sorting_name = resfile_path.name[:resfile_path.name.find('.res')]
        clu_sorting_name = clufile_path.name[:clufile_path.name.find('.clu')]

        assert res_sorting_name == clu_sorting_name, "The .res and .clu files do not share the same name! " \
                                                     f"{res_sorting_name}  -- {clu_sorting_name}"

        xml_files = [f for f in folder_path.iterdir() if f.is_file() if f.suffix == ".xml"]
        assert len(xml_files) > 0, "No .xml file found in the folder!"
        assert len(xml_files) == 1, "More than one .xml file found in the folder!"
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath)).getroot()
        self._sampling_frequency = float(xml_root.find('acquisitionSystem').find('samplingRate').text)

        with open(resfile_path) as f:
            res = np.array([int(line) for line in f], np.int64)
        with open(clufile_path) as f:
            clu = np.array([int(line) for line in f], np.int64)

        n_spikes = len(res)
        if n_spikes > 0:
            # Extract the number of unique IDs from the first line of the clufile then remove it from the list
            n_clu = clu[0]
            clu = np.delete(clu, 0)
            unique_ids = np.unique(clu)
            if 0 not in unique_ids:  # missing unsorted IDs
                n_clu += 1
            if 1 not in unique_ids:  # missing mua IDs
                n_clu += 1

            self._spiketrains = []
            if keep_mua_units:
                n_clu -= 1
                self._unit_ids = [x + 1 for x in range(n_clu)]  # from 1,...,clu[0]-1
                for s_id in self._unit_ids:
                    self._spiketrains.append(res[(clu == s_id).nonzero()])
            else:
                n_clu -= 2
                self._unit_ids = [x + 1 for x in range(n_clu)]  # from 1,...,clu[0]-2
                for s_id in self._unit_ids:
                    self._spiketrains.append(res[(clu == s_id + 1).nonzero()])  # from 2,...,clu[0]-1

        if spkfile_path is not None and Path(spkfile_path).is_file():
            n_bits = int(xml_root.find('acquisitionSystem').find('nBits').text)
            dtype = f"int{n_bits}"
            n_samples = int(xml_root.find('neuroscope').find('spikes').find('nSamples').text)
            wf = np.memmap(spkfile_path, dtype=dtype)
            n_channels = int(wf.size / (n_spikes * n_samples))
            wf = wf.reshape(n_spikes, n_samples, n_channels)

            for unit_id in self.get_unit_ids():
                if gain is not None:
                    self.set_unit_property(unit_id=unit_id, property_name='gain', value=gain)
                self.set_unit_spike_features(
                    unit_id=unit_id,
                    feature_name='waveforms',
                    value=wf[clu == unit_id + 1 - int(keep_mua_units), :, :]
                )

        if folder_path_passed:
            self._kwargs = dict(
                resfile_path=None,
                clufile_path=None,
                folder_path=str(folder_path.absolute()),
                keep_mua_units=keep_mua_units,
                gain=gain
            )
        else:
            self._kwargs = dict(
                resfile_path=str(resfile_path.absolute()),
                clufile_path=str(clufile_path.absolute()),
                folder_path=None,
                keep_mua_units=keep_mua_units,
                gain=gain
            )
        if spkfile_path is not None:
            self._kwargs.update(spkfile_path=str(spkfile_path.absolute()))
        else:
            self._kwargs.update(spkfile_path=spkfile_path)

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


class NeuroscopeMultiSortingExtractor(MultiSortingExtractor):
    """
    Extracts spiking information from an arbitrary number of .res.%i and .clu.%i files in the general folder path.

    The .res is a text file with a sorted list of spiketimes from all units displayed in sample (integer '%i') units.
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
    write_waveforms : bool
        Optional. If True, extracts waveform data from .spk.%i files in the path corresponding to
        the .res.%i and .clue.%i files and sets these as unit spike features. Defaults to False.
    gain : float
        Optional. If passing a spkfile_path, this value converts the data type of the waveforms to units of microvolts.
    """

    extractor_name = "NeuroscopeMultiSortingExtractor"
    installed = HAVE_LXML
    is_writable = True
    mode = "folder"
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(
        self,
        folder_path: PathType,
        keep_mua_units: bool = True,
        exclude_shanks: Optional[list] = None,
        write_waveforms: bool = False,
        gain: Optional[float] = None
    ):
        assert HAVE_LXML, self.installation_mesg

        folder_path = Path(folder_path)

        if exclude_shanks is not None:  # dumping checks do not like having an empty list as default
            assert all([isinstance(x, (int, np.integer)) and x >= 0 for x in
                        exclude_shanks]), "Optional argument 'exclude_shanks' must contain positive integers only!"
            exclude_shanks_passed = True
        else:
            exclude_shanks = []
            exclude_shanks_passed = False
        xml_files = [f for f in folder_path.iterdir() if f.is_file if f.suffix == ".xml"]
        assert len(xml_files) > 0, "No .xml file found in the folder!"
        assert len(xml_files) == 1, "More than one .xml file found in the folder!"
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath)).getroot()
        self._sampling_frequency = float(xml_root.find('acquisitionSystem').find('samplingRate').text)

        res_files = get_shank_files(folder_path=folder_path, suffix=".res")
        clu_files = get_shank_files(folder_path=folder_path, suffix=".clu")

        assert len(res_files) > 0 or len(clu_files) > 0, "No .res or .clu files found in the folder_path!"
        assert len(res_files) == len(clu_files)

        res_ids = [int(x.suffix[1:]) for x in res_files]
        clu_ids = [int(x.suffix[1:]) for x in clu_files]
        assert sorted(res_ids) == sorted(clu_ids), "Unmatched .clu.%i and .res.%i files detected!"
        if any([x not in res_ids for x in exclude_shanks]):
            warnings.warn("Detected indices in exclude_shanks that are not in the directory! These will be ignored.")

        resfile_names = [x.name[:x.name.find('.res')] for x in res_files]
        clufile_names = [x.name[:x.name.find('.clu')] for x in clu_files]
        assert np.all(r == c for (r, c) in zip(resfile_names, clufile_names)), \
            "Some of the .res.%i and .clu.%i files do not share the same name!"
        sorting_name = resfile_names[0]

        all_shanks_list_se = []
        for shank_id in list(set(res_ids) - set(exclude_shanks)):
            nse_args = dict(
                resfile_path=folder_path / f"{sorting_name}.res.{shank_id}",
                clufile_path=folder_path / f"{sorting_name}.clu.{shank_id}",
                keep_mua_units=keep_mua_units
            )

            if write_waveforms:
                spk_files = get_shank_files(folder_path=folder_path, suffix=".spk")
                assert len(spk_files) > 0, "No .spk files found in the folder_path, but 'write_waveforms' is True!"
                assert len(spk_files) == len(res_files), "Mismatched number of .spk and .res files!"

                spk_ids = [int(x.suffix[1:]) for x in spk_files]
                assert sorted(spk_ids) == sorted(res_ids), "Unmatched .spk.%i and .res.%i files detected!"

                spkfile_names = [x.name[:x.name.find('.spk')] for x in spk_files]
                assert np.all(s == r for (s, r) in zip(spkfile_names, resfile_names)), \
                    "Some of the .spk.%i and .res.%i files do not share the same name!"

                nse_args.update(spkfile_path=folder_path / f"{sorting_name}.spk.{shank_id}", gain=gain)

            all_shanks_list_se.append(NeuroscopeSortingExtractor(**nse_args))

        MultiSortingExtractor.__init__(self, sortings=all_shanks_list_se)

        if exclude_shanks_passed:
            self._kwargs = dict(
                folder_path=str(folder_path.absolute()),
                keep_mua_units=keep_mua_units,
                exclude_shanks=exclude_shanks,
                write_waveforms=write_waveforms,
                gain=gain
            )
        else:
            self._kwargs = dict(
                folder_path=str(folder_path.absolute()),
                keep_mua_units=keep_mua_units,
                exclude_shanks=None,
                write_waveforms=write_waveforms,
                gain=gain
            )

    @staticmethod
    def write_sorting(sorting: Union[MultiSortingExtractor, SortingExtractor], save_path: PathType):
        save_path = Path(save_path)
        if save_path.suffix == '':
            sorting_name = save_path.name
        else:
            sorting_name = save_path.stem
        xml_name = sorting_name
        save_xml_filepath = save_path / (str(xml_name) + '.xml')

        assert not save_path.is_file(), "Argument 'save_path' should be a folder!"
        save_path.mkdir(parents=True, exist_ok=True)

        if save_xml_filepath.is_file():
            raise FileExistsError(f"{save_xml_filepath} already exists!")

        xml_root = et.Element('xml')
        et.SubElement(xml_root, 'acquisitionSystem')
        et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')
        xml_root.find('acquisitionSystem').find('samplingRate').text = str(sorting.get_sampling_frequency())
        et.ElementTree(xml_root).write(str(save_xml_filepath.absolute()), pretty_print=True)

        if isinstance(sorting, MultiSortingExtractor):
            counter = 1
            for sort in sorting.sortings:
                # Create and save .res.%i and .clu.%i files from the current sorting object
                save_res = save_path / f"{sorting_name}.res.{counter}"
                save_clu = save_path / f"{sorting_name}.clu.{counter}"
                counter += 1

                res, clu = _extract_res_clu_arrays(sort)

                np.savetxt(save_res, res, fmt="%i")
                np.savetxt(save_clu, clu, fmt="%i")

        elif isinstance(sorting, SortingExtractor):
            # assert units have group property
            assert 'group' in sorting.get_shared_unit_property_names()
            sortings, groups = get_sub_extractors_by_property(sorting, 'group', return_property_list=True)

            for (sort, group) in zip(sortings, groups):
                # Create and save .res.%i and .clu.%i files from the current sorting object
                save_res = save_path / f"{sorting_name}.res.{group}"
                save_clu = save_path / f"{sorting_name}.clu.{group}"

                res, clu = _extract_res_clu_arrays(sort)

                np.savetxt(save_res, res, fmt="%i")
                np.savetxt(save_clu, clu, fmt="%i")


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
