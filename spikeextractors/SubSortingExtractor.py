from .SortingExtractor import SortingExtractor
import numpy as np


# Encapsulates a subset of a spike sorted data file

class SubSortingExtractor(SortingExtractor):

    def __init__(self, parent_sorting, *, unit_ids=None, renamed_unit_ids=None, start_frame=None, end_frame=None):
        SortingExtractor.__init__(self)
        self._parent_sorting = parent_sorting
        self._unit_ids = unit_ids
        self._renamed_unit_ids = renamed_unit_ids
        self._start_frame = start_frame
        self._end_frame = end_frame
        if self._unit_ids is None:
            self._unit_ids = self._parent_sorting.get_unit_ids()
        if self._renamed_unit_ids is None:
            self._renamed_unit_ids = self._unit_ids
        if self._start_frame is None:
            self._start_frame = 0
        if self._end_frame is None:
            self._end_frame = float("inf")
        self._original_unit_id_lookup = {}
        for i in range(len(self._unit_ids)):
            self._original_unit_id_lookup[self._renamed_unit_ids[i]] = self._unit_ids[i]
        self.copy_unit_properties(parent_sorting, unit_ids=self._renamed_unit_ids)
        self.copy_unit_spike_features(parent_sorting, unit_ids=self._renamed_unit_ids)

    def get_unit_ids(self):
        return self._renamed_unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        if (isinstance(unit_id, (int, np.integer))):
            if (unit_id in self.get_unit_ids()):
                original_unit_id = self._original_unit_id_lookup[unit_id]
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")
        sf = self._start_frame + start_frame
        ef = self._start_frame + end_frame
        if sf < self._start_frame:
            sf = self._start_frame
        if ef > self._end_frame:
            ef = self._end_frame
        return self._parent_sorting.get_unit_spike_train(unit_id=original_unit_id, start_frame=sf,
                                                         end_frame=ef) - self._start_frame

    def copy_unit_properties(self, sorting, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        if isinstance(unit_ids, int):
            sorting_unit_id = unit_ids
            if sorting is self._parent_sorting:
                sorting_unit_id = self.get_original_unit_ids(unit_ids)
            curr_property_names = sorting.get_unit_property_names(unit_id=sorting_unit_id)
            for curr_property_name in curr_property_names:
                value = sorting.get_unit_property(unit_id=sorting_unit_id, property_name=curr_property_name)
                self.set_unit_property(unit_id=unit_ids, property_name=curr_property_name, value=value)
        else:
            for unit_id in unit_ids:
                sorting_unit_id = unit_id
                if sorting is self._parent_sorting:
                    sorting_unit_id = self.get_original_unit_ids(unit_id)
                curr_property_names = sorting.get_unit_property_names(unit_id=sorting_unit_id)
                for curr_property_name in curr_property_names:
                    value = sorting.get_unit_property(unit_id=sorting_unit_id, property_name=curr_property_name)
                    self.set_unit_property(unit_id=unit_id, property_name=curr_property_name, value=value)

    def copy_unit_spike_features(self, sorting, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        if isinstance(unit_ids, int):
            sorting_unit_id = unit_ids
            if sorting is self._parent_sorting:
                sorting_unit_id = self.get_original_unit_ids(unit_ids)
            curr_feature_names = sorting.get_unit_spike_feature_names(unit_id=sorting_unit_id)
            for curr_feature_name in curr_feature_names:
                value = sorting.get_unit_spike_features(unit_id=sorting_unit_id, feature_name=curr_feature_name)
                self.set_unit_spike_features(unit_id=unit_ids, feature_name=curr_feature_name, value=value)
        else:
            for unit_id in unit_ids:
                sorting_unit_id = unit_id
                if sorting is self._parent_sorting:
                    sorting_unit_id = self.get_original_unit_ids(unit_id)
                curr_feature_names = sorting.get_unit_spike_feature_names(unit_id=sorting_unit_id)
                for curr_feature_name in curr_feature_names:
                    value = sorting.get_unit_spike_features(unit_id=sorting_unit_id, feature_name=curr_feature_name)
                    self.set_unit_spike_features(unit_id=unit_id, feature_name=curr_feature_name, value=value)

    def get_original_unit_ids(self, unit_ids):
        if isinstance(unit_ids, (int, np.integer)):
            if unit_ids in self.get_unit_ids():
                original_unit_ids = self._original_unit_id_lookup[unit_ids]
            else:
                raise ValueError("Non-valid unit_id")
        else:
            original_unit_ids = []
            for unit_id in unit_ids:
                if isinstance(unit_id, (int, np.integer)):
                    if unit_id in self.get_unit_ids():
                        original_unit_id = self._original_unit_id_lookup[unit_id]
                        original_unit_ids.append(original_unit_id)
                    else:
                        raise ValueError("Non-valid unit_id")
                else:
                    raise ValueError("unit_id must be an int")
        return original_unit_ids
