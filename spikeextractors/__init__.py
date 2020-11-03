from .recordingextractor import RecordingExtractor
from .sortingextractor import SortingExtractor
from .cacheextractors import CacheRecordingExtractor, CacheSortingExtractor
from .subsortingextractor import SubSortingExtractor
from .subrecordingextractor import SubRecordingExtractor
from .multirecordingchannelextractor import concatenate_recordings_by_channel, MultiRecordingChannelExtractor
from .multirecordingtimeextractor import concatenate_recordings_by_time, MultiRecordingTimeExtractor
from .multisortingextractor import concatenate_sortings, MultiSortingExtractor

from .extractorlist import *

from . import example_datasets
from .extraction_tools import load_probe_file, save_to_probe_file, read_binary, write_to_binary_dat_format,\
    write_to_h5_dataset_format, get_sub_extractors_by_property, load_extractor_from_json, load_extractor_from_dict, \
    load_extractor_from_pickle

from distutils.version import StrictVersion

try:
    import h5py
    if StrictVersion(h5py.__version__) > '2.10.0':
        print("h5py version >= 2.10.0. Some extractors might not work properly. It is recommended to "
              "downgrade to version 2.10.0: \n>>> pip install h5py==2.10.0")
except ImportError:
    pass

from .version import version as __version__
