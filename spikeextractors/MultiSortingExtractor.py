from .SortingExtractor import SortingExtractor
from .RecordingExtractor import RecordingExtractor
import numpy as np


# Encapsulates a grouping of sorting extractors

class MultiSortingExtractor(SortingExtractor):
    def __init__(self, *, sortings, start_frames=None):
        SortingExtractor.__init__(self)
        self._SXs = sortings
        self._all_unit_ids = []
        self._unit_map = {}

        if start_frames is None:
            start_frames = []
            for i in range(len(self._SXs)):
                start_frames.append(0)
        self._start_frames = start_frames
        self._allzeros = np.all([start == 0 for start in start_frames])
        if self._allzeros:
            u_id  = 0
            for s_i, SX in enumerate(self._SXs):
                unit_ids = SX.get_unit_ids()
                for unit_id in unit_ids:
                    self._all_unit_ids.append(u_id)
                    self._unit_map[u_id] = {'sx': s_i, 'unit': unit_id}
                    u_id += 1
            # set unit properties
            for unit_id in self.get_unit_ids():
                sx = self._unit_map[unit_id]['sx']
                unit_id_sx = self._unit_map[unit_id]['unit']
                properties = self._SXs[sx].get_unit_property_names(unit_id_sx)
                for prop in properties:
                    self.set_unit_property(unit_id, prop, self._SXs[sx].get_unit_property(unit_id_sx, prop))
        else:
            for s_i, SX in enumerate(self._SXs):
                unit_ids = SX.get_unit_ids()
                for unit_id in unit_ids:
                    self._all_unit_ids.append(unit_id)
            self._all_unit_ids = list(set(self._all_unit_ids))

    def get_unit_ids(self):
        return self._all_unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        if unit_id not in self.get_unit_ids():
            raise ValueError("Non-valid unit_id")

        if self._allzeros:
            sx = self._unit_map[unit_id]['sx']
            unit_id_sx = self._unit_map[unit_id]['unit']
            return self._SXs[sx].get_unit_spike_train(unit_id_sx)
        else:
            spike_train = []
            for i, SX in enumerate(self._SXs):
                if unit_id in SX.get_unit_ids():
                    section_start_frame = max(0, start_frame - self._start_frames[i])
                    section_end_frame = max(0, end_frame - self._start_frames[i])
                    section_spike_train = self._start_frames[i] + SX.get_unit_spike_train(unit_id=unit_id,
                                                                                          start_frame=section_start_frame,
                                                                                          end_frame=section_end_frame)
                    spike_train.append(section_spike_train)
            if not spike_train:
                return np.asarray(spike_train)
            else:
                return np.asarray(np.sort(np.concatenate(spike_train)))

    def get_unit_property(self, unit_id, property_name):
        if self._allzeros:
            if unit_id not in self._unit_map.keys():
                raise ValueError("Non-valid unit_id")
            sx = self._unit_map[unit_id]['sx']
            unit_id_sx = self._unit_map[unit_id]['unit']
            return self._SXs[sx].get_unit_property(unit_id_sx, property_name)
        else:
            raise NotImplementedError()


    def get_unit_spike_features(self, unit_id, feature_name, start_frame=None, end_frame=None):
        if self._allzeros:
            if unit_id not in self._unit_map.keys():
                raise ValueError("Non-valid unit_id")
            sx = self._unit_map[unit_id]['sx']
            unit_id_sx = self._unit_map[unit_id]['unit']
            return self._SXs[sx].get_unit_spike_features(unit_id_sx, feature_name, start_frame=start_frame, end_frame=end_frame)
        else:
            raise NotImplementedError()


    def get_unit_spike_feature_names(self, unit_id=None):
        if self._allzeros:
            if unit_id is None:
                feature_names = []
                for unit_id in self.get_unit_ids():
                    curr_feature_names = self.get_unit_spike_feature_names(unit_id)
                    for curr_feature_name in curr_feature_names:
                        feature_names.append(curr_feature_name)
                feature_names = sorted(list(set(feature_names)))
                return feature_names
            if isinstance(unit_id, (int, np.integer)):
                if unit_id in self.get_unit_ids():
                    if unit_id not in self._unit_map.keys():
                        raise ValueError("Non-valid unit_id")
                    sx = self._unit_map[unit_id]['sx']
                    unit_id_sx = self._unit_map[unit_id]['unit']
                    feature_names = sorted(self._SXs[sx].get_unit_spike_feature_names(unit_id_sx))
                    return feature_names
                else:
                    raise ValueError("Non-valid unit_id")
            else:
                raise ValueError("unit_id must be an int")
        else:
            return self._unit_features

    def set_unit_spike_features(self, unit_id, feature_name, value):
        if self._allzeros:
            if unit_id not in self._unit_map.keys():
                raise ValueError("Non-valid unit_id")
            sx = self._unit_map[unit_id]['sx']
            unit_id_sx = self._unit_map[unit_id]['unit']
            self._SXs[sx].set_unit_spike_features(unit_id_sx, feature_name, value)
        else:
            raise NotImplementedError()
