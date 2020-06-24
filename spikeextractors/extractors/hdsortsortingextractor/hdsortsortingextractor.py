from pathlib import Path
from typing import Union
import numpy as np

from spikeextractors.extractors.matsortingextractor.matsortingextractor import MATSortingExtractor, HAVE_MAT
from spikeextractors.extraction_tools import check_valid_unit_id

PathType = Union[str, Path]


class HDSortSortingExtractor(MATSortingExtractor):
    extractor_name = "HDSortSortingExtractor"

    def __init__(self, file_path: PathType, keep_good_only: bool = True):
        super().__init__(file_path)

        if not self._old_style_mat:
            _units = self._data['Units']
            units = _parse_units(self._data, _units)

            # Extracting MutliElectrode field by field:
            _ME = self._data["MultiElectrode"]
            multi_electrode = dict((k, _ME.get(k)[()]) for k in _ME.keys())

            # Extracting sampling_frequency:
            sr = self._data["samplingRate"]
            self._sampling_frequency = float(_squeeze_ds(sr))

            # Remove noise units if necessary:
            if keep_good_only:
                units = [unit for unit in units if unit["ID"].flatten()[0].astype(int) % 1000 != 0]

            if 'sortingInfo' in self._data.keys():
                info = self._data["sortingInfo"]
                start_frame = _squeeze_ds(info['startTimes'])
                self.start_frame = int(start_frame)
            else:
                self.start_frame = 0
        else:
            _units = self._getfield('Units').squeeze()
            fields = _units.dtype.fields.keys()
            units = []

            for unit in _units:
                unit_dict = {}
                for f in fields:
                    unit_dict[f] = unit[f]
                units.append(unit_dict)

            sr = self._getfield("samplingRate")
            self._sampling_frequency = float(_squeeze_ds(sr))

            _ME = self._data["MultiElectrode"]
            multi_electrode = dict((k, _ME[k][0][0].T) for k in _ME.dtype.fields.keys())

            # Remove noise units if necessary:
            if keep_good_only:
                units = [unit for unit in units if unit["ID"].flatten()[0].astype(int) % 1000 != 0]

            if 'sortingInfo' in self._data.keys():
                info = self._getfield("sortingInfo")
                start_frame = _squeeze_ds(info['startTimes'])
                self.start_frame = int(start_frame)
            else:
                self.start_frame = 0

        # Parse through 'units':
        self._spike_trains = {}
        self._unit_ids = np.empty(0, np.int)
        for uc, unit in enumerate(units):
            uid = int(_squeeze_ds(unit["ID"]))

            self._unit_ids = np.append(self._unit_ids, uid)
            self._spike_trains[uc] = _squeeze(unit["spikeTrain"]).astype(np.int) - self.start_frame

            # For memory efficiency in case it's necessary:
            # X = self.allocate_array( "amplitudes_" + uid, array= unit["spikeAmplitudes"].flatten().T)
            # self.set_unit_spike_features(uid, "amplitudes", X)
            self.set_unit_spike_features(uid, "amplitudes", _squeeze(unit["spikeAmplitudes"]))
            self.set_unit_spike_features(uid, "detection_channel", _squeeze(unit["detectionChannel"]).astype(np.int))

            idx = unit["detectionChannel"].astype(int) - 1
            spikePositions = np.vstack((_squeeze(multi_electrode["electrodePositions"][0][idx]),
                                        _squeeze(multi_electrode["electrodePositions"][1][idx]))).T
            self.set_unit_spike_features(uid, "positions", spikePositions)

            if self._old_style_mat:
                template = unit["footprint"].T
            else:
                template = unit["footprint"]
            self.set_unit_property(uid, "template", template)
            self.set_unit_property(uid, "template_frames_cut_before", unit["cutLeft"].flatten())

        self._units = units
        self._multi_electrode = multi_electrode
        self._kwargs['keep_good_only'] = keep_good_only

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        uidx = np.where(np.array(self.get_unit_ids()) == unit_id)[0][0]

        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        start_frame = start_frame or 0
        end_frame = end_frame or np.infty
        st = self._spike_trains[uidx]
        return st[(st >= start_frame) & (st < end_frame)]

    def get_unit_ids(self):
        return self._unit_ids.tolist()

    @staticmethod
    def write_sorting(sorting, save_path, locations=None, noise_std_by_channel=None, start_frame=0):
        # First, find out how many channels there are
        if locations is not None:
            # write_locations must be a 2D numpy array with n_channels in first dim., (x,y) in second dim.
            n_channels = locations.shape[0]
        elif 'template' in sorting.get_shared_unit_property_names() or \
                'detection_channel' in sorting.get_shared_unit_property_names():
            # Without locations, check if there is a template to get the number of channels
            uid = int(sorting.get_unit_ids()[0])
            if "template" in sorting.get_unit_property_names(uid):
                template = sorting.get_unit_property(uid, "template")
                n_channels = template.shape[0]
            else:
                # If there is also no template, loop through all units and find max. detection_channel
                max_channel = 1
                for uid_ in sorting.get_unit_ids():
                    uid = int(uid_)
                    detection_channel = sorting.get_unit_spike_features(uid, "detection_channel")
                    max_channel = max([max_channel], np.append(detection_channel))
                n_channels = max_channel
        else:
            n_channels = 1

        # Now loop through all units and extract the data that we want to save:
        units = []
        for uid_ in sorting.get_unit_ids():
            uid = int(uid_)

            unit = {"ID": uid,
                    "spikeTrain": sorting.get_unit_spike_train(uid)}
            num_spikes = len(sorting.get_unit_spike_train(uid))

            if "amplitudes" in sorting.get_unit_spike_feature_names(uid):
                unit["spikeAmplitudes"] = sorting.get_unit_spike_features(uid, "amplitudes")
            else:
                # Save a spikeAmplitudes = 1
                unit["spikeAmplitudes"] = np.ones(num_spikes, np.double)

            if "detection_channel" in sorting.get_unit_spike_feature_names(uid):
                unit["detectionChannel"] = sorting.get_unit_spike_features(uid, "detection_channel")
            else:
                # Save a detectionChannel = 1
                unit["detectionChannel"] = np.ones(num_spikes, np.double)

            if "template" in sorting.get_unit_property_names(uid):
                unit["footprint"] = sorting.get_unit_property(uid, "template")
            else:
                # If this unit does not have a footprint, create an empty one:
                unit["footprint"] = np.zeros((3, n_channels), np.double)

            if "template_cut_left" in sorting.get_unit_property_names(uid):
                unit["cutLeft"] = sorting.get_unit_property(uid, "template_cut_left")
            else:
                unit["cutLeft"] = 1

            units.append(unit)

        # Save the electrode locations:
        if locations is None:
            # Create artificial locations if none are provided:
            x = np.zeros(n_channels, np.double)
            y = np.array(np.arange(n_channels), np.double)
            locations = np.vstack((x, y)).T

        multi_electrode = {"electrodePositions": locations, "electrodeNumbers": np.arange(n_channels)}

        if noise_std_by_channel is None:
            noise_std_by_channel = np.ones((1, n_channels))

        dict_to_save = {'Units': units,
                        'MultiElectrode': multi_electrode,
                        'noiseStd': noise_std_by_channel,
                        "samplingRate": sorting._sampling_frequency}

        # Save Units and MultiElectrode to .mat file:
        MATSortingExtractor.write_dict_to_mat(save_path, dict_to_save, version='7.3')


# For .mat v7.3: Function to extract all fields of a struct-array:
def _parse_units(file, _units):
    import h5py

    t_units = {}
    if isinstance(_units, h5py.Group):
        for name in _units.keys():
            value = _units[name]
            dict_val = []
            for val in value:
                if isinstance(file[val[0]], h5py.Dataset):
                    dict_val.append(file[val[0]][()])
                    t_units[name] = dict_val
                else:
                    break
        out = [dict(zip(t_units, col)) for col in zip(*t_units.values())]
    else:
        out = []
        for unit in _units:
            group = file[unit[()][0]]
            unit_dict = {}
            for k in group.keys():
                unit_dict[k] = group[k][()]
            out.append(unit_dict)

    return out


def _squeeze_ds(ds):
    while not isinstance(ds, (int, float, np.integer, np.float)):
        ds = ds[0]
    return ds


def _squeeze(arr):
    shape = arr.shape
    if len(shape) == 2:
        if shape[0] == 1:
            arr = arr[0]
        elif shape[1] == 1:
            arr = arr[:, 0]
    return arr
