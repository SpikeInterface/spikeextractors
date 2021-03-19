from spikeextractors import RecordingExtractor, SortingExtractor
from spikeextractors.extractors.numpyextractors import NumpyRecordingExtractor
from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_unit_spike_train

try:
    import h5py
    HAVE_SCSX = True
except ImportError:
    HAVE_SCSX = False


class SpykingCircusRecordingExtractor(RecordingExtractor):
    """
    RecordingExtractor for a SpykingCircus output folder

    Parameters
    ----------
    folder_path: str or Path
        Path to the output Spyking Circus folder or result folder
    """
    extractor_name = 'SpykingCircusRecording'
    has_default_locations = False
    has_unscaled = False
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'folder'
    installation_mesg = ""  # error message when not installed

    def __init__(self, folder_path):
        RecordingExtractor.__init__(self)
        spykingcircus_folder = Path(folder_path)
        listfiles = spykingcircus_folder.iterdir()

        parent_folder = None
        result_folder = None
        for f in listfiles:
            if f.is_dir():
                if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                    parent_folder = spykingcircus_folder
                    result_folder = f

        if parent_folder is None:
            parent_folder = spykingcircus_folder.parent
            for f in parent_folder.iterdir():
                if f.is_dir():
                    if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                        result_folder = spykingcircus_folder

        assert isinstance(parent_folder, Path) and isinstance(result_folder, Path), "Not a valid spyking circus folder"

        params = None
        params_file = None
        for f in parent_folder.iterdir():
            if f.suffix == '.params':
                params = _load_params(f)
                params_file = f
                break
        assert params is not None, "Could not find the .params file"
        recording_name = params_file.stem

        file_format = params["file_format"].lower()
        if file_format == "numpy":
            recording_file = parent_folder / f"{recording_name}.npy"
            self._recording = NumpyRecordingExtractor(recording_file, params["sampling_frequency"])
        elif file_format == "raw_binary":
            recording_file = parent_folder / f"{recording_name}.dat"
            self._recording = BinDatRecordingExtractor(recording_file, sampling_frequency=params["sampling_frequency"],
                                                       numchan=params["nb_channels"], dtype=params["dtype"],
                                                       time_axis=0)
        else:
            raise Exception(f"'file_format' {params['file_format']} is not supported by the "
                            f"SpykingCircusRecordingExtractor")

        if params["mapping"].is_file():
            self._recording = self.load_probe_file(params["mapping"])

        self.params = params
        self._kwargs = {'folder_path': str(Path(folder_path).absolute())}

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        return self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame,
                                          return_scaled=return_scaled)


class SpykingCircusSortingExtractor(SortingExtractor):
    """
    SortingExtractor for SpykingCircus output folder or file

    Parameters
    ----------
    file_or_folder_path: str or Path
        Path to the output Spyking Circus folder, the result folder, or a specific hdf5 file in the result folder
    load_templates: bool
        If True, templates are loaded from Spyking Circus output
    """
    extractor_name = 'SpykingCircusSorting'
    installed = HAVE_SCSX  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = "To use the SpykingCircusSortingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, file_or_folder_path, load_templates=False):
        assert self.installed, self.installation_mesg
        SortingExtractor.__init__(self)
        file_or_folder_path = Path(file_or_folder_path)

        if file_or_folder_path.is_dir():
            listfiles = file_or_folder_path.iterdir()
            results = None
            parent_folder = None
            result_folder = None
            for f in listfiles:
                if f.is_dir():
                    if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                        parent_folder = file_or_folder_path
                        result_folder = f

            if parent_folder is None:
                parent_folder = file_or_folder_path.parent
                for f in parent_folder.iterdir():
                    if f.is_dir():
                        if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                            result_folder = file_or_folder_path
            # load files
            for f in result_folder.iterdir():
                if 'result.hdf5' in str(f):
                    results = f
                    result_extension = ''
                    base_name = f.name[:f.name.find("result")-1]
                if 'result-merged.hdf5' in str(f):
                    results = f
                    result_extension = '-merged'
                    base_name = f.name[:f.name.find("result")-1]
                    break
        else:
            assert file_or_folder_path.suffix in ['.h5', '.hdf5']
            result_folder = file_or_folder_path.parent
            parent_folder = result_folder.parent
            results = file_or_folder_path
            result_extension = results.stem[results.stem.find("result") + 6:]
            base_name = file_or_folder_path.name[:file_or_folder_path.name.find("result") - 1]

        assert isinstance(parent_folder, Path) and isinstance(result_folder, Path), "Not a valid spyking circus folder"

        # load params
        params = {}
        for f in parent_folder.iterdir():
            if f.suffix == '.params':
                params = _load_params(f)

        if "sampling_frequency" in params.keys():
            self._sampling_frequency = params["sampling_frequency"]

        if results is None:
            raise Exception(f"{file_or_folder_path} is not a spyking circus folder")

        f_results = h5py.File(results, 'r')
        self._spiketrains = []
        self._unit_ids = []
        for temp in f_results['spiketimes'].keys():
            self._spiketrains.append(np.array(f_results['spiketimes'][temp]).astype('int64'))
            self._unit_ids.append(int(temp.split('_')[-1]))

        if load_templates:
            try:
                import scipy
            except:
                raise ImportError("'scipy' is needed to load templates from Spyking Circus")

            filename = result_folder / f"{base_name}.templates{result_extension}.hdf5"
            with h5py.File(filename, 'r', libver='earliest') as f:
                temp_x = f.get('temp_x')[:].ravel()
                temp_y = f.get('temp_y')[:].ravel()
                temp_data = f.get('temp_data')[:].ravel()
                N_e, N_t, nb_templates = f.get('temp_shape')[:].ravel().astype(np.int32)
            templates = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(N_e * N_t, nb_templates))
            templates = np.array([templates[:, i].toarray().reshape(N_e, N_t) for i in range(templates.shape[1])])

            templates = templates[:len(templates)//2]
            for u_i, unit in enumerate(self.get_unit_ids()):
                self.set_unit_property(unit, 'template', templates[u_i])

        self._kwargs = {'file_or_folder_path': str(Path(file_or_folder_path).absolute()),
                        'load_templates': load_templates}

    def get_unit_ids(self):
        return list(self._unit_ids)

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):

        times = self._spiketrains[self.get_unit_ids().index(unit_id)]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def write_sorting(sorting, save_path):
        assert HAVE_SCSX, SpykingCircusSortingExtractor.installation_mesg
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / 'data.result.hdf5'
        elif save_path.suffix == '.hdf5':
            if not str(save_path).endswith('result.hdf5') or not str(save_path).endswith('result-merged.hdf5'):
                raise AttributeError("'save_path' is either a folder or an hdf5 file "
                                     "ending with 'result.hdf5' or 'result-merged.hdf5")
        else:
            save_path.mkdir()
            save_path = save_path / 'data.result.hdf5'
        F = h5py.File(save_path, 'w')
        spiketimes = F.create_group('spiketimes')

        for id in sorting.get_unit_ids():
            spiketimes.create_dataset('tmp_' + str(id), data=sorting.get_unit_spike_train(id))


def _load_params(params_file):
    params = {}
    with params_file.open('r') as f:
        for r in f.readlines():
            if 'sampling_rate' in r:
                sampling_frequency = r.split('=')[-1]
                if '#' in sampling_frequency:
                    sampling_frequency = sampling_frequency[:sampling_frequency.find('#')]
                sampling_frequency = sampling_frequency.strip(" ").strip("\n")
                sampling_frequency = float(sampling_frequency)
                params["sampling_frequency"] = sampling_frequency
            if 'file_format' in r:
                file_format = r.split('=')[-1]
                if '#' in file_format:
                    file_format = file_format[:file_format.find('#')]
                file_format = file_format.strip(" ").strip("\n")
                params["file_format"] = file_format
            if 'nb_channels' in r:
                nb_channels = r.split('=')[-1]
                if '#' in nb_channels:
                    nb_channels = nb_channels[:nb_channels.find('#')]
                nb_channels = nb_channels.strip(" ").strip("\n")
                params["nb_channels"] = int(nb_channels)
            if 'data_dtype' in r:
                dtype = r.split('=')[-1]
                if '#' in dtype:
                    dtype = dtype[:dtype.find('#')]
                dtype = dtype.strip(" ").strip("\n")
                params["dtype"] = dtype
            if 'mapping' in r:
                mapping = r.split('=')[-1]
                if '#' in mapping:
                    mapping = mapping[:mapping.find('#')]
                mapping = mapping.strip(" ").strip("\n")
                params["mapping"] = Path(mapping)
    return params
