from pathlib import Path
import re
from typing import Union

from scipy.spatial.distance import cdist

import numpy as np
import scipy.io as sio
import os

from spikeextractors.extractors.matsortingextractor.matsortingextractor import MATSortingExtractor, HAVE_MAT
from spikeextractors.extraction_tools import check_valid_unit_id

PathType = Union[str, Path]

'''
    Run:
    
    import spikeextractors.extractors.hdsortsortingextractor as e
    fileName = '/Volumes/BACKUP_DRIVE/imported_experiments/200113_rr_p1c2/results_200113_rr_p1c2.mat'
    H = e.HDSortSortingExtractor(fileName)
'''

class HDSortSortingExtractor(MATSortingExtractor):
    extractor_name = "HDSortSortingExtractor"
    installation_mesg = "To use the MATSortingExtractor install h5py and scipy: \n\n pip install h5py scipy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, keep_good_only: bool = False):
        super().__init__(file_path)
        file_path = self._kwargs["file_path"]

        # For .mat v7.3: Extracting "MultiElectrode" requires that each field is loaded separately
        _ME = self._data["MultiElectrode"]
        MultiElectrode = dict( ( k, _ME.get(k)[()] ) for k in _ME.keys())

        # For .mat v7.3: Function to extract all fields of a struct-array:
        def extract_datasets(ds, name):
            x = ds[name]
            return [self._data[xx[0]][()] for xx in x]

        # Extracting all fields of all
        _Units = self._data["Units"]
        tUnits = dict((k, extract_datasets(_Units, k) ) for k in _Units.keys())

        # "Transpose" the dict elements:
        def transpose_dict(d):
            return [dict(zip(d, col)) for col in zip(*d.values())]

        Units = transpose_dict(tUnits)

        # Todo: figure out which one of the following two lines is the one to use:
        #self.set_units_property("sampling_frequency", self._getfield("samplingRate").ravel())
        self._sampling_frequency = self._getfield("samplingRate").ravel()


        # Parse through 'Units':
        self._spike_trains = {}
        self._unit_ids = np.empty(0)
        for uc, Unit in enumerate(Units):
            uid = Unit["ID"].flatten()[0].astype(int)

            self._unit_ids = np.append(self._unit_ids, uid)
            self._spike_trains[uc] = Unit["spikeTrain"].flatten().T

            self.set_unit_spike_features(uid, "amplitudes", Unit["spikeAmplitudes"].flatten().T)
            self.set_unit_spike_features(uid, "amplitudes_up", Unit["spikeAmplitudesUp"].flatten())

            self.set_unit_spike_features(uid, "detection_channel", Unit["detectionChannel"].flatten())
            self.set_unit_spike_features(uid, "detection_channel_up", Unit["detectionChannelUp"].flatten())

            idx = Unit["detectionChannel"].astype(int) - 1
            spikePositions = np.concatenate( (MultiElectrode["electrodePositions"][0][idx], MultiElectrode["electrodePositions"][1][idx]), 0).T
            self.set_unit_spike_features(uid, "positions", spikePositions)

            # Todo: save max_channel correctly, i.e. the maximal template location:
            self.set_unit_property(uid, "max_channel", Unit["detectionChannel"].flatten())

            self.set_unit_property(uid, "template", Unit["footprint"])
            self.set_unit_property(uid, "template_cut_left", Unit["cutLeft"].flatten())

        # This should be changed in the future, but for debugging and saving of the MultiElectrode,
        # it's handy to have Units and MultiElectrode as object attributes
        self.Units = Units
        self.MultiElectrode = MultiElectrode


    @check_valid_unit_id
    def get_unit_spike_features(self, unit_id, feature_name, start_frame=None, end_frame=None):
        # todo: what does this function do?
        if feature_name not in ("raw_traces", "filtered_traces", "cluster_features"):
            return super().get_unit_spike_features(unit_id, feature_name, start_frame, end_frame)

        mask = self._unit_masks[unit_id]
        if feature_name == "raw_traces":
            return self._raw_traces[:, :, mask] * self._kwargs["bit_scaling"]
        elif feature_name == "filtered_traces":
            return self._filt_traces[:, :, mask] * self._kwargs["bit_scaling"]
        else:
            return self._cluster_features[:, :, mask]

    @check_valid_unit_id
    def get_unit_spike_feature_names(self, unit_id):
        # todo: what does this function do?
        return super().get_unit_spike_feature_names(unit_id) + ["raw_traces", "filtered_traces", "cluster_features"]

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
    def write_sorting(sorting, save_path, write_primary_channels=False):
        units = []
        for uid_ in sorting.get_unit_ids():
            uid = int(uid_)
            #print('uid: {}'.format(uid))
            unit = {"ID": uid,
                    "spikeTrain": sorting.get_unit_spike_train(uid),
                    "spikeAmplitudes": sorting.get_unit_spike_features(uid, "amplitudes"),
                    "spikeAmplitudesUp": sorting.get_unit_spike_features(uid, "amplitudes_up"),
                    "detectionChannel": sorting.get_unit_spike_features(uid, "detection_channel"),
                    "detectionChannelUp": sorting.get_unit_spike_features(uid, "detection_channel_up"),
                    "footprint": sorting.get_unit_property(uid, "template"),
                    "cutLeft": sorting.get_unit_property(uid, "template_cut_left"),
                    }
            units.append(unit)

        # Save MultiElectrode (so far there are problems with the orientation of each vector)
        if hasattr(sorting, 'MultiElectrode'):
            MultiElectrode = sorting.MultiElectrode
            MultiElectrode["electrodePositions"] = MultiElectrode["electrodePositions"].T
            MultiElectrode["electrodeNumbers"] = MultiElectrode["electrodeNumbers"].T
            MultiElectrode["parentElectrodeIndex"] = MultiElectrode["parentElectrodeIndex"].T
            dict_to_save = {'Units': units, 'MultiElectrode': MultiElectrode}
        else:
            dict_to_save = {'Units': units}

        # Save Units and MultiElectrode to .mat file:
        placeholder = "asdf"
        matFileName = "result_" + placeholder + ".mat"
        sio.savemat(os.path.join(save_path, matFileName), dict_to_save)
