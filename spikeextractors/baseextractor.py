import json
import os
from pathlib import Path
import importlib
import numpy as np
import spikeextractors


class BaseExtractor:
    is_dumpable = True

    def __init__(self):
        self._kwargs = {}
        self._dump_dict = {}

    def make_serialized_dict(self):
        class_name = str(type(self)).replace("<class '", "").replace("'>", '')
        module = class_name.split('.')[0]
        imported_module = importlib.import_module(module)

        dump_dict = {'class': class_name, 'module': module, 'kwargs': self._kwargs,
                     'version': imported_module.__version__}

        return dump_dict

    def dump(self, output_folder=None, output_filename=None):
        '''
        Dumps recording extractor to json file.
        The extractor can be re-loaded with spikeextractors.load_extractor_from_json(json_file)

        Parameters
        ----------
        output_folder: str or Path
            Path to output_folder
        output_filename: str
            Filename
        '''
        if output_folder is None:
            output_folder = Path(os.getcwd())
        else:
            output_folder = Path(output_folder)
        if output_filename is None:
            output_filename = 'spikeinterface_recording.json'
        elif Path(output_filename).suffix == '':
            output_filename = output_filename + '.json'
        dump_dict = self.make_serialized_dict()
        if self.is_dumpable:
            with open(str(output_folder / output_filename), 'w', encoding='utf8') as f:
                json.dump(dump_dict, f, indent=4)
        else:
            raise Exception(f"Object {str(type(self))} is not dumpable to json")

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


def _load_extractor_from_dict(dic, check_version=False):
    extractor = None
    kwargs = dic['kwargs']
    if np.any([isinstance(v, dict) for v in kwargs.values()]):
        for k in kwargs.keys():
            if isinstance(kwargs[k], dict):
                if 'module' in kwargs[k].keys() and 'class' in kwargs[k].keys() and 'version' in kwargs[k].keys():
                    extractor = _load_extractor_from_dict(kwargs[k])
                    class_name = dic['class']
                    cls = _get_class_from_string(class_name)
                    if not _check_same_version(class_name, dic['version']):
                        print('Versions are not the same. This might lead to errors.')
                        print('Use ', class_name.split('.')[0], 'version', dic['version'])
                    kwargs[k] = extractor
                    extractor = cls(**kwargs)
    else:
        class_name = dic['class']
        cls = _get_class_from_string(class_name)
        if not _check_same_version(class_name, dic['version']):
            print('Versions are not the same. This might lead to errors.')
            print('Use ', class_name.split('.')[0], 'version', dic['version'])
        extractor = cls(**dic['kwargs'])
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
