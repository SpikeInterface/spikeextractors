from .extractors.mdaextractors.mdaextractors import MdaRecordingExtractor, MdaSortingExtractor
from .extractors.mearecextractors.mearecextractors import MEArecRecordingExtractor, MEArecSortingExtractor
from .extractors.biocamrecordingextractor import BiocamRecordingExtractor
from .extractors.exdirextractors import ExdirRecordingExtractor, ExdirSortingExtractor
from .extractors.intanrecordingextractor import IntanRecordingExtractor
from .extractors.hs2sortingextractor import HS2SortingExtractor
from .extractors.klustasortingextractor import KlustaSortingExtractor
from .extractors.kilosortsortingextractor import KiloSortSortingExtractor
from .extractors.numpyextractors.numpyextractors import NumpyRecordingExtractor, NumpySortingExtractor
from .extractors.nwbextractors.nwbextractors import NwbRecordingExtractor
from .extractors.openephysextractors.openephysextractors import OpenEphysRecordingExtractor, OpenEphysSortingExtractor
from .extractors.physortingextractor.physortingextractor import PhySortingExtractor
from .extractors.bindatrecordingextractor import BinDatRecordingExtractor
from .extractors.spykingcircussortingextractor.spykingcircussortingextractor import SpykingCircusSortingExtractor
from .extractors.tridesclousextractor.tridesclousextractor import TridesclousSortingExtractor


recording_extractor_full_list = [
    MdaRecordingExtractor,
    MEArecRecordingExtractor,
    BiocamRecordingExtractor,
    ExdirRecordingExtractor,
    OpenEphysRecordingExtractor,
    IntanRecordingExtractor,
    BinDatRecordingExtractor
]

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
    TridesclousSortingExtractor
]

installed_sorting_extractor_list = [sx for sx in sorting_extractor_full_list if sx.installed]
