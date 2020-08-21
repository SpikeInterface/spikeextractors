from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors.extractors.npzsortingextractor import NpzSortingExtractor
from spikeextractors import RecordingExtractor, SortingExtractor
import tempfile
from pathlib import Path
from copy import deepcopy
import importlib
import os
import shutil


class CacheRecordingExtractor(BinDatRecordingExtractor, RecordingExtractor):
    def __init__(self, recording, chunk_size=None, chunk_mb=500, save_path=None, verbose=False):
        RecordingExtractor.__init__(self)  # init tmp folder before constructing BinDatRecordingExtractor
        tmp_folder = self.get_tmp_folder()
        self._recording = recording
        if save_path is None:
            self._is_tmp = True
            self._tmp_file = tempfile.NamedTemporaryFile(suffix=".dat", dir=tmp_folder).name
        else:
            save_path = Path(save_path)
            if save_path.suffix != '.dat' and save_path.suffix != '.bin':
                save_path = save_path.with_suffix('.dat')
            if not save_path.parent.is_dir():
                os.makedirs(save_path.parent)
            self._is_tmp = False
            self._tmp_file = save_path
        self._dtype = recording.get_dtype()
        recording.write_to_binary_dat_format(save_path=self._tmp_file, dtype=self._dtype, chunk_size=chunk_size,
                                             chunk_mb=chunk_mb, verbose=verbose)
        # keep track of filter status when dumping
        self.is_filtered = self._recording.is_filtered
        BinDatRecordingExtractor.__init__(self, self._tmp_file, numchan=recording.get_num_channels(),
                                          recording_channels=recording.get_channel_ids(),
                                          sampling_frequency=recording.get_sampling_frequency(),
                                          dtype=self._dtype, is_filtered=self.is_filtered)
        # keep BinDatRecording kwargs
        self._bindat_kwargs = deepcopy(self._kwargs)
        self.set_tmp_folder(tmp_folder)
        self.copy_channel_properties(recording)
        self._kwargs = {'recording': recording, 'chunk_size': chunk_size, 'chunk_mb': chunk_mb}

    def __del__(self):
        if self._is_tmp:
            try:
                os.remove(self._tmp_file)
            except Exception as e:
                print("Unable to remove temporary file", e)

    @property
    def filename(self):
        return str(self._tmp_file)

    def move_to(self, save_path):
        save_path = Path(save_path)
        if save_path.suffix != '.dat' and save_path.suffix != '.bin':
            save_path = save_path.with_suffix('.dat')
        if not save_path.parent.is_dir():
            os.makedirs(save_path.parent)
        shutil.move(self._tmp_file, str(save_path))
        self._tmp_file = str(save_path)
        self._kwargs['file_path'] = str(Path(self._tmp_file).absolute())
        self._bindat_kwargs['file_path'] = str(Path(self._tmp_file).absolute())
        self._is_tmp = False
        tmp_folder = self.get_tmp_folder()
        # re-initialize with new file
        BinDatRecordingExtractor.__init__(self, self._tmp_file, numchan=self._recording.get_num_channels(),
                                          recording_channels=self._recording.get_channel_ids(),
                                          sampling_frequency=self._recording.get_sampling_frequency(),
                                          dtype=self._dtype, is_filtered=self.is_filtered)
        self.set_tmp_folder(tmp_folder)
        self.copy_channel_properties(self._recording)

    # override to make serialization avoid reloading and saving binary file
    def make_serialized_dict(self, include_properties=None, include_features=None):
        '''
        Makes a nested serialized dictionary out of the extractor. The dictionary be used to re-initialize an
        extractor with spikeextractors.load_extractor_from_dict(dump_dict)

        Returns
        -------
        dump_dict: dict
            Serialized dictionary
        include_properties: list or None
            List of properties to include in the dictionary
        include_features: list or None
            List of features to include in the dictionary
        '''
        class_name = str(BinDatRecordingExtractor).replace("<class '", "").replace("'>", '')
        module = class_name.split('.')[0]
        imported_module = importlib.import_module(module)

        if self._is_tmp:
            print("Warning: dumping a CacheRecordingExtractor. The path to the tmp binary file will be lost in "
                  "further sessions. To prevent this, use the 'CacheRecordingExtractor.move_to('path-to-file)' "
                  "function")

        dump_dict = {'class': class_name, 'module': module, 'kwargs': self._bindat_kwargs,
                     'key_properties': self._key_properties, 'version': imported_module.__version__, 'dumpable': True}
        return dump_dict


class CacheSortingExtractor(NpzSortingExtractor, SortingExtractor):
    def __init__(self, sorting, save_path=None):
        SortingExtractor.__init__(self)  # init tmp folder before constructing NpzSortingExtractor
        tmp_folder = self.get_tmp_folder()
        self._sorting = sorting
        if save_path is None:
            self._is_tmp = True
            self._tmp_file = tempfile.NamedTemporaryFile(suffix=".npz", dir=tmp_folder).name
        else:
            save_path = Path(save_path)
            if save_path.suffix != '.npz':
                save_path = save_path.with_suffix('.npz')
            if not save_path.parent.is_dir():
                os.makedirs(save_path.parent)
            self._is_tmp = False
            self._tmp_file = save_path
        NpzSortingExtractor.write_sorting(self._sorting, self._tmp_file)
        NpzSortingExtractor.__init__(self, self._tmp_file)
        # keep Npz kwargs
        self._npz_kwargs = deepcopy(self._kwargs)
        self.set_tmp_folder(tmp_folder)
        self.copy_unit_properties(sorting)
        self.copy_unit_spike_features(sorting)
        self._kwargs = {'sorting': sorting}

    def __del__(self):
        if self._is_tmp:
            try:
                os.remove(self._tmp_file)
            except Exception as e:
                print("Unable to remove temporary file", e)

    @property
    def filename(self):
        return str(self._tmp_file)

    def move_to(self, save_path):
        save_path = Path(save_path)
        if save_path.suffix != '.npz':
            save_path = save_path.with_suffix('.npz')
        if not save_path.parent.is_dir():
            os.makedirs(save_path.parent)
        shutil.move(self._tmp_file, str(save_path))
        self._tmp_file = str(save_path)
        self._kwargs['file_path'] = str(Path(self._tmp_file).absolute())
        self._npz_kwargs['file_path'] = str(Path(self._tmp_file).absolute())
        self._is_tmp = False
        tmp_folder = self.get_tmp_folder()
        # re-initialize with new file
        NpzSortingExtractor.__init__(self, self._tmp_file)
        # keep Npz kwargs
        self.set_tmp_folder(tmp_folder)
        self.copy_unit_properties(self._sorting)
        self.copy_unit_spike_features(self._sorting)

    # override to make serialization avoid reloading and saving npz file
    def make_serialized_dict(self, include_properties=None, include_features=None):
        '''
        Makes a nested serialized dictionary out of the extractor. The dictionary be used to re-initialize an
        extractor with spikeextractors.load_extractor_from_dict(dump_dict)

        Returns
        -------
        dump_dict: dict
            Serialized dictionary
        include_properties: list or None
            List of properties to include in the dictionary
        include_features: list or None
            List of features to include in the dictionary
        '''
        class_name = str(NpzSortingExtractor).replace("<class '", "").replace("'>", '')
        module = class_name.split('.')[0]
        imported_module = importlib.import_module(module)

        if self._is_tmp:
            print("Warning: dumping a CacheSortingExtractor. The path to the tmp binary file will be lost in "
                  "further sessions. To prevent this, use the 'CacheSortingExtractor.move_to('path-to-file)' "
                  "function")

        dump_dict = {'class': class_name, 'module': module, 'kwargs': self._npz_kwargs,
                     'key_properties': self._key_properties, 'version': imported_module.__version__, 'dumpable': True}
        return dump_dict
