from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_valid_unit_id
from typing import Union
from scipy.io import loadmat, savemat

try:
    from lxml import etree as et

    HAVE_LXML = True
except ImportError:
    HAVE_LXML = False

PathType = Union[str, Path, None]


class CellExplorerSortingExtractor(SortingExtractor):
    """
    Extracts spiking information from .mat files stored in the CellExplorer format.

    Parameters
    ----------
    matfile_path : str
        Optional. Path to the .mat file.
    """

    extractor_name = 'CellExplorerSortingExtractor'
    installed = HAVE_LXML
    is_writable = True
    mode = 'custom'
    installation_mesg = "Please install lxml to use this extractor!"

    def __init__(self, matfile_path: PathType):
        assert HAVE_LXML, self.installation_mesg
        assert matfile_path.is_file(), f"The matfile_path ({matfile_path}) must exist!"

        SortingExtractor.__init__(self)

        matfile_path = Path(matfile_path)
        folder_path = matfile_path.parent
        xml_files = [f for f in folder_path.iterdir() if f.is_file() if f.suffix == '.xml']
        assert len(xml_files) > 0, 'No .xml file found in the folder.'
        assert len(xml_files) == 1, 'More than one .xml file found in the folder.'
        xml_filepath = xml_files[0]

        xml_root = et.parse(str(xml_filepath.absolute())).getroot()
        self._sampling_frequency = float(xml_root.find('acquisitionSystem').find(
            'samplingRate').text)  # careful not to confuse it with the lfpsamplingrate

        spikes_mat = loadmat(str(matfile_path.absolute()))['spikes']

        self._unit_ids = list(spikes_mat['UID'][0][0][0])
        self._spiketrains = [[y[0] for y in x] for x in spikes_mat['times'][0][0][0]]

        self._kwargs = {'matfile_path': str(matfile_path.absolute())}

    def get_unit_ids(self):
        return list(self._unit_ids)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def add_unit(self, unit_id, spike_times):
        """
        Add a new unit with the given spike times.

        Parameters
        ----------
        unit_id: int
            The unit_id of the unit to be added.
        """
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
        save_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == '':
            sorting_name = save_path.name
        else:
            sorting_name = save_path.stem
        save_xml_filepath = save_path / f"{str(sorting_name)}.xml"
        if save_xml_filepath.is_file():
            raise FileExistsError(f"{save_xml_filepath} already exists!")

        xml_root = et.Element('xml')
        et.SubElement(xml_root, 'acquisitionSystem')
        et.SubElement(xml_root.find('acquisitionSystem'), 'samplingRate')
        xml_root.find('acquisitionSystem').find('samplingRate').text = str(sorting.get_sampling_frequency())
        et.ElementTree(xml_root).write(str(save_xml_filepath.absolute()), pretty_print=True)

        mat_dict = dict(
            spikes=dict(
                UID=sorting.get_unit_ids(),
                times=[[[y] for y in x] for x in sorting.get_units_spike_train()]
            )
        )
        savemat(save_path, mat_dict)
