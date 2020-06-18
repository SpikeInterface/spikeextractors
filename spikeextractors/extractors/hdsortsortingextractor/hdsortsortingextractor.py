from pathlib import Path
from typing import Union
import numpy as np

from spikeextractors.extractors.matsortingextractor.matsortingextractor import MATSortingExtractor, HAVE_MAT
from spikeextractors.extraction_tools import check_valid_unit_id

PathType = Union[str, Path]

class HDSortSortingExtractor(MATSortingExtractor):
    extractor_name = "HDSortSortingExtractor"
    installation_mesg = "To use the MATSortingExtractor install h5py and scipy: \n\n pip install h5py scipy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, remove_noise_units: bool = True, old_style_mat= False):
        super().__init__(file_path)
        assert(not self._kwargs['old_style_mat'])
        file_path = self._kwargs["file_path"]

        # Extracting units that are saved as struct arrays into a list of dicts:
        _units = self._data["Units"]
        units = _extract_struct_array(self._data, _units)

        # Extracting MutliElectrode field by field:
        _ME = self._data["MultiElectrode"]
        multi_electrode = dict( ( k, _ME.get(k)[()] ) for k in _ME.keys())

        # Extracting sampling_frequency:
        self._sampling_frequency = self._getfield("samplingRate").ravel()

        # Remove noise units if necessary:
        if remove_noise_units:
            units = [unit for unit in units if unit["ID"].flatten()[0].astype(int)%1000 != 0]
        
        # Parse through 'units':
        self._spike_trains = {}
        self._unit_ids = np.empty(0, np.int)
        for uc, unit in enumerate(units):
            uid = unit["ID"].flatten()[0].astype(int)

            self._unit_ids = np.append(self._unit_ids, uid)
            self._spike_trains[uc] = unit["spikeTrain"].flatten().T.astype(np.int)

            # For memory efficiency in case it's necessary:
            # X = self.allocate_array( "amplitudes_" + uid, array= unit["spikeAmplitudes"].flatten().T)
            # self.set_unit_spike_features(uid, "amplitudes", X)
            self.set_unit_spike_features(uid, "amplitudes", unit["spikeAmplitudes"].flatten().T)
            self.set_unit_spike_features(uid, "detection_channel", unit["detectionChannel"].flatten().astype(np.int))

            idx = unit["detectionChannel"].astype(int) - 1
            spikePositions = np.vstack( (multi_electrode["electrodePositions"][0][idx], multi_electrode["electrodePositions"][1][idx])).T
            self.set_unit_spike_features(uid, "positions", spikePositions)

            self.set_unit_property(uid, "template", unit["footprint"])
            self.set_unit_property(uid, "template_frames_cut_before", unit["cutLeft"].flatten())

        self._units = units
        self._multi_electrode = multi_electrode

        # Close the h5py.File:
        self._data.close()

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
    def write_sorting(sorting, save_path, locations=None, noise_std_by_channel=None):
        # First, find out how many channels there are
        if locations is not None:
            # write_locations must be a 2D numpy array with n_channels in first dim., (x,y) in second dim.
            n_channels = locations.shape[0]
        else:
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

        # Now loop through all units and extract the data that we want to save:
        units = []
        for uid_ in sorting.get_unit_ids():
            uid = int(uid_)

            unit = {"ID": uid,
                    "spikeTrain": sorting.get_unit_spike_train(uid)}

            if "amplitudes" in sorting.get_unit_spike_feature_names(uid):
                unit["spikeAmplitudes"] = sorting.get_unit_spike_features(uid, "amplitudes")
            else:
                # Save a spikeAmplitudes = 1
                print("asdf")
                unit["spikeAmplitudes"] = np.ones(unit["spikeTrain"].shape, np.double)

            if "detection_channel" in sorting.get_unit_spike_feature_names(uid):
                unit["detectionChannel"] = sorting.get_unit_spike_features(uid, "detection_channel")
            else:
                # Save a detectionChannel = 1
                unit["detectionChannel"] = np.ones(unit["spikeTrain"].shape, np.double)

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
            locations = np.vstack((x,y)).T

        multi_electrode = {"electrodePositions": locations, "electrodeNumbers": np.arange(n_channels)}

        if noise_std_by_channel is None:
            noise_std_by_channel = np.ones((1, n_channels))

        dict_to_save = {'Units': units,
                        'MultiElectrode': multi_electrode,
                        'noiseStd': noise_std_by_channel,
                        "samplingRate": sorting._sampling_frequency}

        # Save Units and MultiElectrode to .mat file:
        MATSortingExtractor.write_dict_v7_3(save_path, dict_to_save)


# For .mat v7.3: Function to extract all fields of a struct-array:
def _extract_struct_array(_data, ds):
    try:
        # Try loading structure fields directly as datasets
        t_units = {}
        for name in ds.keys():
            x = ds[name]
            r = [_data[xx[0]][()] for xx in x]
            t_units[name] = r
    
    except:
        # Sometimes, a .mat -v7.3 file contains #refs#, i.e. datasets don't correspond directly
        # to structure fields but instead point to a different datasets with a hashed name.
        # Here, we solve this by looping through all fields, read the reference and then load the
        # referenced dataset.
        t_units = {}
        for name in _data[ds[0][0]].keys():
            r = []
            for _ds in ds:
                reference = _ds[0]
                val = _data[reference][name][()]
                r.append(val.flatten())

            t_units[name] = np.array(r)

    # The data (t_units) is now a dict where each entry is a numpy array with
    # N elements. To get a struct array, we need now to "transpose" it, such that the
    # return value is a N-list of dicts where all have the same keys.
    return [dict(zip(d, col)) for col in zip(*d.values())]
