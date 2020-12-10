from .extractors.mdaextractors.mdaextractors import MdaRecordingExtractor, MdaSortingExtractor
from .extractors.mearecextractors.mearecextractors import MEArecRecordingExtractor, MEArecSortingExtractor
from .extractors.biocamrecordingextractor.biocamrecordingextractor import BiocamRecordingExtractor
from .extractors.exdirextractors.exdirextractors import ExdirRecordingExtractor, ExdirSortingExtractor
from .extractors.intanrecordingextractor.intanrecordingextractor import IntanRecordingExtractor
from .extractors.hdsortsortingextractor.hdsortsortingextractor import HDSortSortingExtractor
from .extractors.hs2sortingextractor.hs2sortingextractor import HS2SortingExtractor
from .extractors.klustaextractors.klustaextractors import KlustaSortingExtractor, KlustaRecordingExtractor
from .extractors.kilosortextractors.kilosortextractors import KiloSortSortingExtractor, KiloSortRecordingExtractor
from .extractors.numpyextractors.numpyextractors import NumpyRecordingExtractor, NumpySortingExtractor
from .extractors.nwbextractors.nwbextractors import NwbRecordingExtractor, NwbSortingExtractor
from .extractors.maxonerecordingextractor import MaxOneRecordingExtractor
from .extractors.mea1krecordingextractor import Mea1kRecordingExtractor
from .extractors.openephysextractors.openephysextractors import OpenEphysRecordingExtractor, OpenEphysSortingExtractor
from .extractors.phyextractors.phyextractors import PhyRecordingExtractor, PhySortingExtractor
from .extractors.bindatrecordingextractor.bindatrecordingextractor import BinDatRecordingExtractor
from .extractors.spykingcircusextractors.spykingcircusextractors import SpykingCircusSortingExtractor, \
    SpykingCircusRecordingExtractor
from .extractors.spikeglxrecordingextractor.spikeglxrecordingextractor import SpikeGLXRecordingExtractor
from .extractors.tridescloussortingextractor.tridescloussortingextractor import TridesclousSortingExtractor
from .extractors.npzsortingextractor.npzsortingextractor import NpzSortingExtractor
from .extractors.mcsh5recordingextractor.mcsh5recordingextractor import MCSH5RecordingExtractor
from .extractors.shybridextractors import SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor
from .extractors.nixioextractors.nixioextractors import NIXIORecordingExtractor, NIXIOSortingExtractor
from .extractors.neoextractors import (PlexonRecordingExtractor, PlexonSortingExtractor,
                                       NeuralynxRecordingExtractor, NeuralynxSortingExtractor, MCSRawRecordingExtractor)
from .extractors.neuroscopeextractors import NeuroscopeRecordingExtractor,NeuroscopeSortingExtractor,NeuroscopeMultiSortingExtractor
from .extractors.waveclussortingextractor import WaveClusSortingExtractor
from .extractors.yassextractors import YassSortingExtractor
from .extractors.combinatosortingextractor import CombinatoSortingExtractor

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
    MaxOneRecordingExtractor,
    Mea1kRecordingExtractor,
    MCSH5RecordingExtractor,
    MCSRawRecordingExtractor,
    SHYBRIDRecordingExtractor,
    NIXIORecordingExtractor,
    NeuroscopeRecordingExtractor,
    
    # neo based
    PlexonRecordingExtractor,
    NeuralynxRecordingExtractor
    
]

recording_extractor_dict = {recording_class.extractor_name: recording_class for recording_class in recording_extractor_full_list}
installed_recording_extractor_list = [rx for rx in recording_extractor_full_list if rx.installed]

sorting_extractor_full_list = [
    MdaSortingExtractor,
    MEArecSortingExtractor,
    ExdirSortingExtractor,
    HDSortSortingExtractor,
    HS2SortingExtractor,
    KlustaSortingExtractor,
    KiloSortSortingExtractor,
    OpenEphysSortingExtractor,
    PhySortingExtractor,
    SpykingCircusSortingExtractor,
    TridesclousSortingExtractor,
    NpzSortingExtractor,
    SHYBRIDSortingExtractor,
    NIXIOSortingExtractor,
    NeuroscopeSortingExtractor,
    NeuroscopeMultiSortingExtractor,
    WaveClusSortingExtractor,
    YassSortingExtractor,
    CombinatoSortingExtractor,

    # neo based
    PlexonSortingExtractor,
    NeuralynxSortingExtractor,
    
]

installed_sorting_extractor_list = [sx for sx in sorting_extractor_full_list if sx.installed]
sorting_extractor_dict = {sorting_class.extractor_name: sorting_class for sorting_class in sorting_extractor_full_list}

writable_sorting_extractor_list = [sx for sx in installed_sorting_extractor_list if sx.is_writable]
writable_sorting_extractor_dict = {sorting_class.extractor_name: sorting_class for sorting_class in writable_sorting_extractor_list}
