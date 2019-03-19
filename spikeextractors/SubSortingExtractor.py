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
            self._unit_ids = self._parent_sorting.getUnitIds()
        if self._renamed_unit_ids is None:
            self._renamed_unit_ids = self._unit_ids
        if self._start_frame is None:
            self._start_frame = 0
        if self._end_frame is None:
            self._end_frame = float("inf")
        self._original_unit_id_lookup = {}
        for i in range(len(self._unit_ids)):
            self._original_unit_id_lookup[self._renamed_unit_ids[i]] = self._unit_ids[i]
        self.copyUnitProperties(parent_sorting, unit_ids=self._renamed_unit_ids)
        self.copyUnitSpikeFeatures(parent_sorting, unit_ids=self._renamed_unit_ids)

    def getUnitIds(self):
        return self._renamed_unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        if (isinstance(unit_id, (int, np.integer))):
            if (unit_id in self.getUnitIds()):
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
        return self._parent_sorting.getUnitSpikeTrain(unit_id=original_unit_id, start_frame=sf,
                                                      end_frame=ef) - self._start_frame

    def copyUnitProperties(self, sorting, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.getUnitIds()
        if isinstance(unit_ids, int):
            sorting_unit_id = unit_ids
            if sorting is self._parent_sorting:
                sorting_unit_id = self.getOriginalUnitIds(unit_ids)
            curr_property_names = sorting.getUnitPropertyNames(unit_id=sorting_unit_id)
            for curr_property_name in curr_property_names:
                value = sorting.getUnitProperty(unit_id=sorting_unit_id, property_name=curr_property_name)
                self.setUnitProperty(unit_id=unit_ids, property_name=curr_property_name, value=value)
        else:
            for unit_id in unit_ids:
                sorting_unit_id = unit_id
                if sorting is self._parent_sorting:
                    sorting_unit_id = self.getOriginalUnitIds(unit_id)
                curr_property_names = sorting.getUnitPropertyNames(unit_id=sorting_unit_id)
                for curr_property_name in curr_property_names:
                    value = sorting.getUnitProperty(unit_id=sorting_unit_id, property_name=curr_property_name)
                    self.setUnitProperty(unit_id=unit_id, property_name=curr_property_name, value=value)

    def copyUnitSpikeFeatures(self, sorting, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.getUnitIds()
        if isinstance(unit_ids, int):
            sorting_unit_id = unit_ids
            if sorting is self._parent_sorting:
                sorting_unit_id = self.getOriginalUnitIds(unit_ids)
            curr_feature_names = sorting.getUnitSpikeFeatureNames(unit_id=sorting_unit_id)
            for curr_feature_name in curr_feature_names:
                value = sorting.getUnitSpikeFeatures(unit_id=sorting_unit_id, feature_name=curr_feature_name)
                self.setUnitSpikeFeatures(unit_id=unit_ids, feature_name=curr_feature_name, value=value)
        else:
            for unit_id in unit_ids:
                sorting_unit_id = unit_id
                if sorting is self._parent_sorting:
                    sorting_unit_id = self.getOriginalUnitIds(unit_id)
                print(sorting_unit_id)
                print(unit_id)
                curr_feature_names = sorting.getUnitSpikeFeatureNames(unit_id=sorting_unit_id)
                for curr_feature_name in curr_feature_names:
                    value = sorting.getUnitSpikeFeatures(unit_id=sorting_unit_id, feature_name=curr_feature_name)
                    self.setUnitSpikeFeatures(unit_id=unit_id, v=curr_feature_name, value=value)

    def getOriginalUnitIds(self, unit_ids):
        if isinstance(unit_ids, (int, np.integer)):
            if unit_ids in self.getUnitIds():
                original_unit_ids = self._original_unit_id_lookup[unit_ids]
            else:
                raise ValueError("Non-valid unit_id")
        else:
            original_unit_ids = []
            for unit_id in unit_ids:
                if isinstance(unit_id, (int, np.integer)):
                    if unit_id in self.getUnitIds():
                        original_unit_id = self._original_unit_id_lookup[unit_id]
                        original_unit_ids.append(original_unit_id)
                    else:
                        raise ValueError("Non-valid unit_id")
                else:
                    raise ValueError("unit_id must be an int")
        return original_unit_ids
