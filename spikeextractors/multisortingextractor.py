from .sortingextractor import SortingExtractor
from .recordingextractor import RecordingExtractor
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
                self._unit_map[u_id] = {'sorting_id': s_i, 'unit_id': unit_id}
                u_id += 1
        self._kwargs = {'sortings': [sort.make_serialized_dict() for sort in sortings]}

    def get_unit_ids(self):
        return list(self._all_unit_ids)

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if unit_id not in self.get_unit_ids():
            raise ValueError("Non-valid unit_id")

        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        return self._sortings[sorting_id].get_unit_spike_train(unit_id_sorting, start_frame, end_frame)

    def set_sampling_frequency(self, sampling_frequency):
        for sorting in self._sortings:
            sorting.set_sampling_frequency(sampling_frequency)

    def get_sampling_frequency(self):
        return self._sortings[0].get_sampling_frequency()

    def set_unit_property(self, unit_id, property_name, value):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        self._sortings[sorting_id].set_unit_property(unit_id_sorting, property_name, value)

    def get_unit_property(self, unit_id, property_name):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        return self._sortings[sorting_id].get_unit_property(unit_id_sorting, property_name)

    def get_unit_property_names(self, unit_id):
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        property_names = self._sortings[sorting_id].get_unit_property_names(unit_id_sorting)
        return property_names

    def clear_unit_property(self, unit_id, property_name):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        self._sortings[sorting_id].clear_unit_property(unit_id_sorting, property_name)

    def get_unit_spike_features(self, unit_id, feature_name, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        return self._sortings[sorting_id].get_unit_spike_features(unit_id_sorting, feature_name, start_frame=start_frame, end_frame=end_frame)

    def get_unit_spike_feature_names(self, unit_id):
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_map.keys():
                    raise ValueError("Non-valid unit_id")
                sorting_id = self._unit_map[unit_id]['sorting_id']
                unit_id_sorting = self._unit_map[unit_id]['unit_id']
                feature_names = sorted(self._sortings[sorting_id].get_unit_spike_feature_names(unit_id_sorting))
                return feature_names
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")

    def set_unit_spike_features(self, unit_id, feature_name, value):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        self._sortings[sorting_id].set_unit_spike_features(unit_id_sorting, feature_name, value)
        
    def clear_unit_spike_features(self, unit_id, feature_name):
        if unit_id not in self._unit_map.keys():
            raise ValueError("Non-valid unit_id")
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        self._sortings[sorting_id].clear_unit_spike_features(unit_id_sorting, feature_name)        

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
