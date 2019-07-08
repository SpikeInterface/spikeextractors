from .SortingExtractor import SortingExtractor
from .RecordingExtractor import RecordingExtractor
import numpy as np


# Encapsulates a grouping of non-continuous sorting extractors

class MultiSortingExtractor(SortingExtractor):
    def __init__(self, sortings):
        SortingExtractor.__init__(self)
        self._sortings = sortings
        self._all_unit_ids = []
        self._unit_map = {}

        u_id  = 0
        for s_i, sorting in enumerate(self._sortings):
            unit_ids = sorting.get_unit_ids()
            for unit_id in unit_ids:
                self._all_unit_ids.append(u_id)
                self._unit_map[u_id] = {'sorting': s_i, 'unit_id': unit_id}
                u_id += 1

    def get_unit_ids(self):
        return list(self._all_unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        if unit_id not in self.get_unit_ids():
            raise ValueError("Non-valid unit_id")

        sorting = self._unit_map[unit_id]['sorting']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        return self._sortings[sorting].get_unit_spike_train(unit_id_sorting)

    def get_sampling_frequency(self):
        return self._sortings[0].get_sampling_frequency()

    def set_unit_property(self, unit_id, property_name, value):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting = self._unit_map[unit_id]['sorting']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        self._sortings[sorting].set_unit_property(unit_id_sorting, property_name, value)

    def get_unit_property(self, unit_id, property_name):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting = self._unit_map[unit_id]['sorting']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        return self._sortings[sorting].get_unit_property(unit_id_sorting, property_name)

    def get_unit_property_names(self, unit_id=None):
        if unit_id is None:
            property_names = []
            for sorting in self._sortings:
                property_names_sorting = sorting.get_unit_property_names(unit_id)
                for property_name in property_names_sorting:
                    property_names.append(property_name)
            property_names = list(set(property_names))
        else:
            sorting = self._unit_map[unit_id]['sorting']
            unit_id_sorting = self._unit_map[unit_id]['unit_id']
            property_names = self._sortings[sorting].get_unit_property_names(unit_id_sorting)
        return property_names

    def get_unit_spike_features(self, unit_id, feature_name, start_frame=None, end_frame=None):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting = self._unit_map[unit_id]['sorting']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        return self._sortings[sorting].get_unit_spike_features(unit_id_sorting, feature_name, start_frame=start_frame, end_frame=end_frame)

    def get_unit_spike_feature_names(self, unit_id=None):
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
                sorting = self._unit_map[unit_id]['sorting']
                unit_id_sorting = self._unit_map[unit_id]['unit_id']
                feature_names = sorted(self._sortings[sorting].get_unit_spike_feature_names(unit_id_sorting))
                return feature_names
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")

    def set_unit_spike_features(self, unit_id, feature_name, value):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting = self._unit_map[unit_id]['sorting']
        unit_id_sorting = self._unit_map[unit_id]['unit']
        self._sortings[sorting].set_unit_spike_features(unit_id_sorting, feature_name, value)

def concatenate_sortings(sortings):
    '''
    Concatenates sortings together. The sortings should be non-continuous

    Parameters
    ----------
    sortings: list
        The list of SortingExtractors to be concatenated
    Returns
    -------
    recording: MultiSortingExtractor
        The concatenated sorting extractors enscapsulated in the
        MultiSortingExtractor object (which is also a sorting extractor)
    '''
    return MultiSortingExtractor(
        sortings=sortings,
    )