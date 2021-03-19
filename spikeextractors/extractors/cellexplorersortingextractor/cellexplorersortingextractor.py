from spikeextractors import SortingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_unit_spike_train
from typing import Union

try:
    from scipy.io import loadmat, savemat
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PathType = Union[str, Path]


class CellExplorerSortingExtractor(SortingExtractor):
    """
    Extracts spiking information from .mat files stored in the CellExplorer format.

    Spike times are stored in units of seconds.

    Parameters
    ----------
    spikes_matfile_path : PathType
        Path to the sorting_id.spikes.cellinfo.mat file.
    """

    extractor_name = "CellExplorerSortingExtractor"
    installed = HAVE_SCIPY
    is_writable = True
    mode = "file"
    installation_mesg = "To use the CellExplorerSortingExtractor install scipy: \n\n pip install scipy\n\n"

    def __init__(self, spikes_matfile_path: PathType):
        assert self.installed, self.installation_mesg
        SortingExtractor.__init__(self)

        spikes_matfile_path = Path(spikes_matfile_path)
        assert spikes_matfile_path.is_file(), f"The spikes_matfile_path ({spikes_matfile_path}) must exist!"
        folder_path = spikes_matfile_path.parent
        sorting_id = spikes_matfile_path.name.split(".")[0]

        session_info_matfile_path = folder_path / f"{sorting_id}.sessionInfo.mat"
        assert session_info_matfile_path.is_file(), "No sessionInfo.mat file found in the folder!"
        session_info_mat = loadmat(file_name=str(session_info_matfile_path.absolute()))
        assert session_info_mat['sessionInfo']['rates'][0][0]['wideband'], "The sesssionInfo.mat file must contain " \
            "a 'sessionInfo' struct with field 'rates' containing field 'wideband'!"
        self._sampling_frequency = float(
            session_info_mat['sessionInfo']['rates'][0][0]['wideband'][0][0][0][0]
        )  # careful not to confuse it with the lfpsamplingrate; reported in units Hz

        spikes_mat = loadmat(file_name=str(spikes_matfile_path.absolute()))
        assert np.all(np.isin(['UID', 'times'], spikes_mat['spikes'].dtype.names)), \
            "The spikes.cellinfo.mat file must contain a 'spikes' struct with fields 'UID' and 'times'!"

        self._unit_ids = np.asarray(spikes_mat['spikes']['UID'][0][0][0], dtype=int)
        # CellExplorer reports spike times in units seconds; SpikeExtractors uses time units of sampling frames
        # Rounding is necessary to prevent data loss from int-casting floating point errors
        self._spiketrains = [
            (np.array([y[0] for y in x]) * self._sampling_frequency).round().astype(int)
            for x in spikes_mat['spikes']['times'][0][0][0]
         ]

        self._kwargs = dict(spikes_matfile_path=str(spikes_matfile_path.absolute()))

    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def write_sorting(sorting: SortingExtractor, save_path: PathType):
        assert save_path.suffixes == ['.spikes', '.cellinfo', '.mat'], \
            "The save_path must correspond to the CellExplorer format of sorting_id.spikes.cellinfo.mat!"

        base_path = save_path.parent
        sorting_id = save_path.name.split(".")[0]
        session_info_save_path = base_path / f"{sorting_id}.sessionInfo.mat"
        spikes_save_path = save_path
        base_path.mkdir(parents=True, exist_ok=True)

        sampling_frequency = sorting.get_sampling_frequency()
        session_info_mat_dict = dict(
            sessionInfo=dict(
                rates=dict(
                    wideband=sampling_frequency
                )
            )
        )
        savemat(file_name=session_info_save_path, mdict=session_info_mat_dict)

        spikes_mat_dict = dict(
            spikes=dict(
                UID=sorting.get_unit_ids(),
                times=[[[y / sampling_frequency] for y in x] for x in sorting.get_units_spike_train()]
            )
        )
        # If, in the future, it is ever desired to allow this to write unit properties, they must conform
        # to the format here: https://cellexplorer.org/datastructure/data-structure-and-format/
        savemat(file_name=spikes_save_path, mdict=spikes_mat_dict)
