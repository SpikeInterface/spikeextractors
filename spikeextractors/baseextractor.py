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
        extractor = None
        with open(str(json_file), 'r') as f:
            d = json.load(f)

            extractor = _load_extractor_from_dict(d)

        # import spiketoolkit

        #
        # ordered_keys = sorted([k for k in chain.keys()])
        # extractor = None
        # for k in ordered_keys:
        #     d = chain[k]
        #     if int(k) == 0:
        #         # first step MUST BE an extractor
        #         assert d['type'] == 'extractor', "First element of the dictionary should be an extractor!"
        #         if d['name'] in spikeextractors.recording_extractor_dict.keys():
        #             reccls = spikeextractors.recording_extractor_dict[d['name']]
        #             extractor = reccls(**d['kwargs'])
        #         elif d['name'] in spikeextractors.sorting_extractor_dict.keys():
        #             sortcls = spikeextractors.sorting_extractor_dict[d['name']]
        #             extractor = sortcls(**d['kwargs'])
        #     else:
        #         if 'preprocessor' == d['type']:
        #             if d['name'] in spiketoolkit.preprocessing.preprocesser_dict.keys():
        #                 tlkclass = spiketoolkit.preprocessing.preprocesser_dict[d['name']]
        #                 extractor = tlkclass(extractor, **d['kwargs'])
        #         elif 'curator' == d['type']:
        #             if d['name'] in spiketoolkit.validation.curation_dict.keys():
        #                 tlkclass = spiketoolkit.validation.curation_dict[d['name']]
        #                 extractor = tlkclass(extractor, **d['kwargs'])
        #         else:
        #             if 'SubRecording' in d['name']:
        #                 extractor = spikeextractors.SubRecordingExtractor(extractor, **d['kwargs'])
        #             elif 'SubSorting' in d['name']:
        #                 extractor = spikeextractors.SubSortingExtractor(extractor, **d['kwargs'])
        #             elif 'Multi' in d['name']:
        #                 raise NotImplementedError("Multi recording and sorting cannot be dumped to json")
        return extractor


def _load_extractor_from_dict(dic, check_version=False):
    # tODO fix this!
    extractor = None
    if np.any([isinstance(v, dict) for v in dic.values()]):
        for k in dic.keys():
            if isinstance(dic[k], dict):
                print(dic[k].keys())
                if 'module' in dic[k].keys() \
                    and 'class' in dic[k].keys() and 'version' in dic[k].keys():
                    print('ciao')
                    extractor = _load_extractor_from_dict(dic[k])
                    cls = _get_class_from_string(dic['class'])
                    # TODO add version check
                    extractor = cls(extractor, **dic['kwargs'])
    else:
        cls = _get_class_from_string(dic['class'])
        # TODO add version check
        extractor = cls(**dic['kwargs'])
    return extractor


def _get_class_from_string(class_string):
    module = class_string.split('.')[0]
    class_name = class_string.split('.')[-1]
    imported_module = importlib.import_module(module)

    try:
        imported_class = getattr(imported_module, class_name)
    except:
        imported_class = None

    return imported_class
