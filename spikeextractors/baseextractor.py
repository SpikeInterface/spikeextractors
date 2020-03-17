import json
import os
from pathlib import Path
import importlib
import numpy as np
import datetime


class BaseExtractor:
    is_dumpable = True

    def __init__(self, *args, **kargs):
        self._kwargs = {}
        self._tmp_folder = None
        
        self.
    
    def _init_extractor(self, *args, **kargs):
        # this remplace the init in each class
        # this is call both at __init__ but also in __setstate__ (when pickle.load)
        raise NotImplementedError
    

    def make_serialized_dict(self):
        class_name = str(type(self)).replace("<class '", "").replace("'>", '')
        module = class_name.split('.')[0]
        imported_module = importlib.import_module(module)

        if self.is_dumpable:
            dump_dict = {'class': class_name, 'module': module, 'kwargs': self._kwargs,
                         'version': imported_module.__version__, 'dumpable': True}
            if 'Recording' in class_name:
                if 'group' in self.get_shared_channel_property_names():
                    groups = self.get_channel_groups()
                    dump_dict['groups'] = groups
                if 'location' in self.get_shared_channel_property_names():
                    locations = self.get_channel_locations()
                    dump_dict['locations'] = locations
            elif 'Sorting' in class_name:
                if 'group' in self.get_shared_unit_property_names():
                    groups = [self.get_unit_property(u, 'group') for u in self.get_unit_ids()]
                    dump_dict['groups'] = groups
        else:
            dump_dict = {'class': class_name, 'module': module, 'kwargs': {},
                         'version': imported_module.__version__, 'dumpable': False}

        return dump_dict

    def dump(self, file_name=None, folder_path=None):
        '''
        Dumps recording extractor to json file.
        The extractor can be re-loaded with spikeextractors.load_extractor_from_json(json_file)

        Parameters
        ----------
        file_name: str
            Filename
        folder_path: str or Path
            Path to output_folder
        '''
        if self.check_if_dumpable():
            if folder_path is None:
                folder_path = Path(os.getcwd())
            else:
                folder_path = Path(folder_path)
            if file_name is None:
                file_name = 'spikeinterface_recording.json'
            elif Path(file_name).suffix == '':
                file_name = file_name + '.json'
            dump_dict = self.make_serialized_dict()
            with open(str(folder_path / file_name), 'w', encoding='utf8') as f:
                json.dump(_check_json(dump_dict), f, indent=4)
        else:
            raise Exception(f"The extractor is not dumpable to to json")

    @staticmethod
    def load_extractor_from_json(json_file):
        '''
        Instantiates extractor from json file

        Parameters
        ----------
        json_file: str or Path
            Path to json file

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        '''
        with open(str(json_file), 'r') as f:
            d = json.load(f)
            extractor = _load_extractor_from_dict(d)
        return extractor

    @staticmethod
    def load_extractor_from_dict(d):
        '''
        Instantiates extractor from dictionary

        Parameters
        ----------
        d: dictionary
            Python dictionary

        Returns
        -------
        extractor: RecordingExtractor or SortingExtractor
            The loaded extractor object
        '''
        extractor = _load_extractor_from_dict(d)
        return extractor

    def check_if_dumpable(self):
        return _check_if_dumpable(self.make_serialized_dict())
    
    def __getstate__(self):
        return self.make_serialized_dict()

    def __setstate__(self, d):
        self._init_extractor(**d)


def _load_extractor_from_dict(dic):
    cls = None
    class_name = None
    probe_file = None
    kwargs = dic['kwargs']
    if np.any([isinstance(v, dict) for v in kwargs.values()]):
        # nested
        for k in kwargs.keys():
            if isinstance(kwargs[k], dict):
                if 'module' in kwargs[k].keys() and 'class' in kwargs[k].keys() and 'version' in kwargs[k].keys():
                    extractor = _load_extractor_from_dict(kwargs[k])
                    class_name = dic['class']
                    cls = _get_class_from_string(class_name)
                    kwargs[k] = extractor
                    break
    elif np.any([isinstance(v, list) and isinstance(v[0], dict) for v in kwargs.values()]):
        # multi
        for k in kwargs.keys():
            if isinstance(kwargs[k], list) and isinstance(kwargs[k][0], dict):
                extractors = []
                for kw in kwargs[k]:
                    if 'module' in kw.keys() and 'class' in kw.keys() and 'version' in kw.keys():
                        extr = _load_extractor_from_dict(kw)
                        extractors.append(extr)
                class_name = dic['class']
                cls = _get_class_from_string(class_name)
                kwargs[k] = extractors
                break
    else:
        class_name = dic['class']
        cls = _get_class_from_string(class_name)

    assert cls is not None and class_name is not None, "Could not load spikeinterface class"
    if not _check_same_version(class_name, dic['version']):
        print('Versions are not the same. This might lead to errors. Use ', class_name.split('.')[0],
              'version', dic['version'])

    if 'probe_file' in kwargs.keys():
        probe_file = kwargs.pop('probe_file')

    # instantiate extrator object
    extractor = cls(**kwargs)

    # load properties and probe file
    if 'Recording' in class_name:
        if 'groups' in dic.keys():
            groups = dic['groups']
            for i, ch in enumerate(extractor.get_channel_ids()):
                extractor.set_channel_property(ch, 'group', groups[i])
        if 'locations' in dic.keys():
            locations = dic['locations']
            for i, ch in enumerate(extractor.get_channel_ids()):
                extractor.set_channel_property(ch, 'location', np.array(locations[i]))
    elif 'Sorting' in class_name:
        if 'groups' in dic.keys():
            groups = dic['groups']
            for i, unit in enumerate(extractor.get_unit_ids()):
                extractor.set_unit_property(unit, 'group', groups[i])
    if probe_file is not None:
        assert 'Recording' in class_name, "Only recording extractors can have probe files"
        extractor = extractor.load_probe_file(probe_file=probe_file)
    return extractor


def _get_class_from_string(class_string):
    class_name = class_string.split('.')[-1]
    module = '.'.join(class_string.split('.')[:-1])
    imported_module = importlib.import_module(module)

    try:
        imported_class = getattr(imported_module, class_name)
    except:
        imported_class = None

    return imported_class


def _check_same_version(class_string, version):
    module = class_string.split('.')[0]
    imported_module = importlib.import_module(module)

    return imported_module.__version__ == version


def _check_if_dumpable(d):
    kwargs = d['kwargs']
    if np.any([isinstance(v, dict) and 'dumpable' in v.keys() for (k, v) in kwargs.items()]):
        for k, v in kwargs.items():
            if 'dumpable' in v.keys():
                return _check_if_dumpable(v)
    else:
        return d['dumpable']


def _check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _check_json(v)
        elif isinstance(v, Path):
            d[k] = str(v.absolute())
        elif isinstance(v, (np.int, np.int32, np.int64)):
            d[k] = int(v)
        elif isinstance(v, (np.float, np.float32, np.float64)):
            d[k] = float(v)
        elif isinstance(v, datetime.datetime):
            d[k] = v.isoformat()
        elif isinstance(v, (np.ndarray, list)):
            if isinstance(v[0], dict):
                # these must be extractors for multi extractors
                d[k] = [_check_json(v_el) for v_el in v]
            else:
                v_arr = np.array(v)
                if len(v_arr.shape) == 1:
                    if 'int' in str(v_arr.dtype):
                        v_arr = [int(v_el) for v_el in v_arr]
                        d[k] = v_arr
                    elif 'float' in str(v_arr.dtype):
                        v_arr = [float(v_el) for v_el in v_arr]
                        d[k] = v_arr
                    else:
                        print('Skipping field: only int or float can be serialized')
                elif len(v_arr.shape) == 2:
                    if 'int' in str(v_arr.dtype):
                        v_arr = [[int(v_el) for v_el in v_row] for v_row in v_arr]
                        d[k] = v_arr
                    elif 'float' in str(v_arr.dtype):
                        v_arr = [[float(v_el) for v_el in v_row] for v_row in v_arr]
                        d[k] = v_arr
                    else:
                        print('Skipping field: only int or float can be serialized')
                else:
                    print("Skipping field: only 1D and 2D arrays can be serialized")
    return d