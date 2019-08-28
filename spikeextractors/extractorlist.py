from .extractors.mdaextractors.mdaextractors import MdaRecordingExtractor, MdaSortingExtractor
from .extractors.mearecextractors.mearecextractors import MEArecRecordingExtractor, MEArecSortingExtractor
from .extractors.biocamrecordingextractor.biocamrecordingextractor import BiocamRecordingExtractor
from .extractors.exdirextractors.exdirextractors import ExdirRecordingExtractor, ExdirSortingExtractor
from .extractors.intanrecordingextractor.intanrecordingextractor import IntanRecordingExtractor
from .extractors.hs2sortingextractor.hs2sortingextractor import HS2SortingExtractor
from .extractors.klustaextractors.klustaextractors import KlustaSortingExtractor, KlustaRecordingExtractor
from .extractors.kilosortextractors.kilosortextractors import KiloSortSortingExtractor, KiloSortRecordingExtractor
from .extractors.numpyextractors.numpyextractors import NumpyRecordingExtractor, NumpySortingExtractor
from .extractors.nwbextractors.nwbextractors import NwbRecordingExtractor
from .extractors.maxonerecordingextractor import MaxOneRecordingExtractor
from .extractors.openephysextractors.openephysextractors import OpenEphysRecordingExtractor, OpenEphysSortingExtractor
from .extractors.phyextractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor
from .extractors.bindatrecordingextractor.bindatrecordingextractor import BinDatRecordingExtractor
from .extractors.spykingcircusextractors.spykingcircusextractors import SpykingCircusSortingExtractor, \
    SpykingCircusRecordingExtractor
from .extractors.spikeglxrecordingextractor.spikeglxrecordingextractor import SpikeGLXRecordingExtractor
from .extractors.tridescloussortingextractor.tridescloussortingextractor import TridesclousSortingExtractor
from .extractors.npzsortingextractor.npzsortingextractor import NpzSortingExtractor


recording_extractor_full_list = [
    MdaRecordingExtractor,
    MEArecRecordingExtractor,
    BiocamRecordingExtractor,
    ExdirRecordingExtractor,
    OpenEphysRecordingExtractor,
    IntanRecordingExtractor,
    BinDatRecordingExtractor,
    KlustaRecordingExtractor,
    KiloSortRecordingExtractor,
    SpykingCircusRecordingExtractor,
    SpikeGLXRecordingExtractor,
    PhyRecordingExtractor,
    MaxOneRecordingExtractor
]

recording_extractor_dict = {recording_class.extractor_name: recording_class for recording_class in recording_extractor_full_list}

installed_recording_extractor_list = [rx for rx in recording_extractor_full_list if rx.installed]

sorting_extractor_full_list = [
    MdaSortingExtractor,
    MEArecSortingExtractor,
    ExdirSortingExtractor,
    HS2SortingExtractor,
    KlustaSortingExtractor,
    KiloSortSortingExtractor,
    OpenEphysSortingExtractor,
    PhySortingExtractor,
    SpykingCircusSortingExtractor,
    TridesclousSortingExtractor,
    NpzSortingExtractor,
]

installed_sorting_extractor_list = [sx for sx in sorting_extractor_full_list if sx.installed]

sorting_extractor_dict = {sorting_class.extractor_name: sorting_class for sorting_class in sorting_extractor_full_list}
