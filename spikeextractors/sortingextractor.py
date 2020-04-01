from abc import ABC, abstractmethod
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import shutil
from .extraction_tools import get_sub_extractors_by_property
from .baseextractor import BaseExtractor


class SortingExtractor(ABC, BaseExtractor):
    '''A class that contains functions for extracting important information
    from spiked sorted data given a spike sorting software. It is an abstract
    class so all functions with the @abstractmethod tag must be implemented for
    the initialization to work.


    '''
    def __init__(self):
        BaseExtractor.__init__(self)
        self._epochs = {}
        self._unit_properties = {}
        self._unit_features = {}
        self._sampling_frequency = None
        self.id = np.random.randint(low=0, high=9223372036854775807, dtype='int64')

    def __del__(self):
        if self._tmp_folder is not None:
            try:
                shutil.rmtree(self._tmp_folder)
            except Exception as e:
                print('Impossible to delete temp file:', self._tmp_folder, 'Error', e)

    @abstractmethod
    def get_unit_ids(self):
        '''This function returns a list of ids (ints) for each unit in the sorsted result.

        Returns
        ----------
        unit_ids: array_like
            A list of the unit ids in the sorted result (ints).
        '''
        pass

    @abstractmethod
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        '''This function extracts spike frames from the specified unit.
        It will return spike frames from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike frames are returned in the form of an
        array_like of spike frames. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording
        start_frame: int
            The frame above which a spike frame is returned  (inclusive)
        end_frame: int
            The frame below which a spike frame is returned  (exclusive)
        Returns
        ----------
        spike_train: numpy.ndarray
            An 1D array containing all the frames for each spike in the
            specified unit given the range of start and end frames
        '''
        pass

    def get_sampling_frequency(self):
        '''
        It returns the sampling frequency.

        Returns
        -------
        sampling_frequency: float
            The sampling frequency
        '''
        return self._sampling_frequency

    def set_sampling_frequency(self, sampling_frequency):
        '''
        It sets the sorting extractor sampling frequency.

        sampling_frequency: float
            The sampling frequency
        '''
        self._sampling_frequency = sampling_frequency

    def set_unit_spike_features(self, unit_id, feature_name, value, indexes=None):
        '''This function adds a unit features data set under the given features
        name to the given unit.

        Parameters
        ----------
        unit_id: int
            The unit id for which the features will be set
        feature_name: str
            The name of the feature to be stored
        value
            The data associated with the given feature name. Could be many
            formats as specified by the user.
        '''
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_features.keys():
                    self._unit_features[unit_id] = {}
                if indexes is None:
                    if isinstance(feature_name, str) and len(value) == len(self.get_unit_spike_train(unit_id)):
                        self._unit_features[unit_id][feature_name] = value
                    else:
                        if not isinstance(feature_name, str):
                            raise ValueError("feature_name must be a string")
                        else:
                            raise ValueError("feature values should have the same length as the spike train")
                else:
                    if isinstance(feature_name, str) and len(value) == len(indexes):
                        self._unit_features[unit_id][feature_name] = value
                        self._unit_features[unit_id][feature_name+'_idxs'] = indexes
                    else:
                        if not isinstance(feature_name, str):
                            raise ValueError("feature_name must be a string")
                        else:
                            raise ValueError("feature values should have the same length as indexes")
            else:
                raise ValueError(str(unit_id) + " is not a valid unit_id")
        else:
            raise ValueError(str(unit_id) + " must be an int")

    def get_unit_spike_features(self, unit_id, feature_name, start_frame=None, end_frame=None):
        '''This function extracts the specified spike features from the specified unit.
        It will return spike features from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike features are returned in the form of an
        array_like of spike features. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording
        feature_name: string
            The name of the feature to be returned
        start_frame: int
            The frame above which a spike frame is returned  (inclusive)
        end_frame: int
            The frame below which a spike frame is returned  (exclusive)
        Returns
        ----------
        spike_features: numpy.ndarray
            An array containing all the features for each spike in the
            specified unit given the range of start and end frames
        '''
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_features.keys():
                    self._unit_features[unit_id] = {}
                if isinstance(feature_name, str):
                    if feature_name in self._unit_features[unit_id].keys():
                        spike_train = self.get_unit_spike_train(unit_id)
                        if start_frame is None:
                            start_frame = 0
                        if end_frame is None:
                            end_frame = np.inf
                        if start_frame == 0 and end_frame == np.inf:
                            # keep memmap objects
                            return self._unit_features[unit_id][feature_name]
                        else:
                            if len(self._unit_features[unit_id][feature_name]) == len(spike_train):
                                spike_indices = np.where(np.logical_and(spike_train >= start_frame,
                                                                        spike_train < end_frame))
                            elif len(self._unit_features[unit_id][feature_name]) < len(spike_train):
                                if not feature_name.endswith('idxs'):
                                    # retrieve features on the correct idxs
                                    assert feature_name + '_idxs' in self.get_unit_spike_feature_names(unit_id=unit_id)
                                    feature_name_idxs = feature_name + '_idxs'
                                    value_idxs = np.array(self.get_unit_spike_features(unit_id=unit_id,
                                                                                       feature_name=feature_name_idxs))
                                    spike_train = spike_train[value_idxs]
                                    spike_indices = np.where(np.logical_and(spike_train >= start_frame,
                                                                            spike_train < end_frame))
                                else:
                                    # retrieve idxs features
                                    value_idxs = np.array(self.get_unit_spike_features(unit_id=unit_id,
                                                                                       feature_name=feature_name))
                                    spike_train = spike_train[value_idxs]
                                    spike_indices = np.where(np.logical_and(spike_train >= start_frame,
                                                                            spike_train < end_frame))
                            else:
                                raise ValueError(str(feature_name) + " dimensions are inconsistent for unit "
                                                 + str(unit_id))
                            return self._unit_features[unit_id][feature_name][spike_indices]
                    else:
                        raise ValueError(str(feature_name) + " has not been added to unit " + str(unit_id))
                else:
                    raise ValueError(str(feature_name) + " must be a string")
            else:
                raise ValueError(str(unit_id) + " is not a valid unit_id")
        else:
            raise ValueError(str(unit_id) + " must be an int")

    def clear_unit_spike_features(self, unit_id, feature_name):
        '''This function clears the unit spikes features for the given feature.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the sorting
        feature_name: string
            The name of the feature to be cleared
        '''
        if unit_id in self._unit_features:
            if feature_name in self._unit_features[unit_id]:
                del self._unit_features[unit_id][feature_name]

    def clear_units_spike_features(self, feature_name, unit_ids=None):
        '''This function clears the units' spikes features for the given feature.

        Parameters
        ----------
        feature_name: string
            The name of the feature to be cleared
        unit_ids: list
            A list of ids that specifies a set of units in the sorting. If None, all units are cleared
        '''
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        for unit_id in unit_ids:
            self.clear_unit_spike_features(unit_id, feature_name)

    def get_unit_spike_feature_names(self, unit_id):
        '''This function returns the list of feature names for the given unit
        Parameters
        ----------
        unit_id: int
            The unit id for which the feature names will be returned
        Returns
        ----------
        property_names
            The list of feature names.
        '''
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_features:
                    self._unit_features[unit_id] = {}
                feature_names = sorted(self._unit_features[unit_id].keys())
                return feature_names
            else:
                raise ValueError(str(unit_id) + " is not a valid unit_id")
        else:
            raise ValueError(str(unit_id) + " must be an int")

    def get_shared_unit_spike_feature_names(self, unit_ids=None):
        '''Get the intersection of unit feature names for a given set of units or for all units if unit_ids is None.
         Parameters
        ----------
        unit_ids: array_like
            The unit ids for which the shared feature names will be returned.
            If None (default), will return shared feature names for all units
        Returns
        ----------
        property_names
            The list of shared feature names
        '''
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        curr_feature_name_set = set(self.get_unit_spike_feature_names(unit_id=unit_ids[0]))
        for unit_id in unit_ids[1:]:
            curr_unit_feature_name_set = set(self.get_unit_spike_feature_names(unit_id=unit_id))
            curr_feature_name_set = curr_feature_name_set.intersection(curr_unit_feature_name_set)
        feature_names = sorted(list(curr_feature_name_set))
        return feature_names

    def set_unit_property(self, unit_id, property_name, value):
        '''This function adds a unit property data set under the given property
        name to the given unit.

        Parameters
        ----------
        unit_id: int
            The unit id for which the property will be set
        property_name: str
            The name of the property to be stored
        value
            The data associated with the given property name. Could be many
            formats as specified by the user
        '''
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_properties:
                    self._unit_properties[unit_id] = {}
                if isinstance(property_name, str):
                    self._unit_properties[unit_id][property_name] = value
                else:
                    raise ValueError(str(property_name) + " must be a string")
            else:
                raise ValueError(str(unit_id) + " is not a valid unit_id")
        else:
            raise ValueError(str(unit_id) + " must be an int")

    def set_units_property(self, *, unit_ids=None, property_name, values):
        '''Sets unit property data for a list of units

        Parameters
        ----------
        unit_ids: list
            The list of unit ids for which the property will be set
            Defaults to get_unit_ids()
        property_name: str
            The name of the property
        value: list
            The list of values to be set
        '''
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        for i, unit in enumerate(unit_ids):
            self.set_unit_property(unit_id=unit, property_name=property_name, value=values[i])

    def get_unit_property(self, unit_id, property_name):
        '''This function returns the data stored under the property name given
        from the given unit.

        Parameters
        ----------
        unit_id: int
            The unit id for which the property will be returned
        property_name: str
            The name of the property
        Returns
        ----------
        value
            The data associated with the given property name. Could be many
            formats as specified by the user
        '''
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_properties:
                    self._unit_properties[unit_id] = {}
                if isinstance(property_name, str):
                    if property_name in list(self._unit_properties[unit_id].keys()):
                        return self._unit_properties[unit_id][property_name]
                    else:
                        raise ValueError(str(property_name) + " has not been added to unit " + str(unit_id))
                else:
                    raise ValueError(str(property_name) + " must be a string")
            else:
                raise ValueError(str(unit_id) + " is not a valid unit_id")
        else:
            raise ValueError(str(unit_id) + " must be an int")

    def get_units_property(self, *, unit_ids=None, property_name):
        '''Returns a list of values stored under the property name corresponding
        to a list of units

        Parameters
        ----------
        unit_ids: list
            The unit ids for which the property will be returned
            Defaults to get_unit_ids()
        property_name: str
            The name of the property
        Returns
        ----------
        values
            The list of values
        '''
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        values = [self.get_unit_property(unit_id=unit, property_name=property_name) for unit in unit_ids]
        return values

    def get_unit_property_names(self, unit_id):
        '''Get a list of property names for a given unit.
         Parameters
        ----------
        unit_id: int
            The unit id for which the property names will be returned
        Returns
        ----------
        property_names
            The list of property names
        '''
        if isinstance(unit_id, (int, np.integer)):
            if unit_id in self.get_unit_ids():
                if unit_id not in self._unit_properties:
                    self._unit_properties[unit_id] = {}
                property_names = sorted(self._unit_properties[unit_id].keys())
                return property_names
            else:
                raise ValueError(str(unit_id) + " is not a valid unit id")
        else:
            raise TypeError(str(unit_id) + " must be an int")

    def get_shared_unit_property_names(self, unit_ids=None):
        '''Get the intersection of unit property names for a given set of units or for all units if unit_ids is None.
         Parameters
        ----------
        unit_ids: array_like
            The unit ids for which the shared property names will be returned.
            If None (default), will return shared property names for all units
        Returns
        ----------
        property_names
            The list of shared property names
        '''
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        curr_property_name_set = set(self.get_unit_property_names(unit_id=unit_ids[0]))
        for unit_id in unit_ids[1:]:
            curr_unit_property_name_set = set(self.get_unit_property_names(unit_id=unit_id))
            curr_property_name_set = curr_property_name_set.intersection(curr_unit_property_name_set)
        property_names = sorted(list(curr_property_name_set))
        return property_names

    def copy_unit_properties(self, sorting, unit_ids=None):
        '''Copy unit properties from another sorting extractor to the current
        sorting extractor.

        Parameters
        ----------
        sorting: SortingExtractor
            The sorting extractor from which the properties will be copied
        unit_ids: (array_like, int)
            The list (or single value) of unit_ids for which the properties will be copied
        '''
        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        if isinstance(unit_ids, int):
            curr_property_names = sorting.get_unit_property_names(unit_id=unit_ids)
            for curr_property_name in curr_property_names:
                value = sorting.get_unit_property(unit_id=unit_ids, property_name=curr_property_name)
                self.set_unit_property(unit_id=unit_ids, property_name=curr_property_name, value=value)
        else:
            for unit_id in unit_ids:
                curr_property_names = sorting.get_unit_property_names(unit_id=unit_id)
                for curr_property_name in curr_property_names:
                    value = sorting.get_unit_property(unit_id=unit_id, property_name=curr_property_name)
                    self.set_unit_property(unit_id=unit_id, property_name=curr_property_name, value=value)

    def clear_unit_property(self, unit_id, property_name):
        '''This function clears the unit property for the given property.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the sorting
        property_name: string
            The name of the property to be cleared
        '''
        if unit_id in self._unit_properties:
            if property_name in self._unit_properties[unit_id]:
                del self._unit_properties[unit_id][property_name]

    def clear_units_property(self, property_name, unit_ids=None):
        '''This function clears the units' properties for the given property.

        Parameters
        ----------
        property_name: string
            The name of the property to be cleared
        unit_ids: list
            A list of ids that specifies a set of units in the sorting. If None, all units are cleared
        '''
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        for unit_id in unit_ids:
            self.clear_unit_property(unit_id, property_name)

    def copy_unit_spike_features(self, sorting, unit_ids=None):
        '''Copy unit spike features from another sorting extractor to the current
        sorting extractor.

        Parameters
        ----------
        sorting: SortingExtractor
            The sorting extractor from which the spike features will be copied
        unit_ids: (array_like, int)
            The list (or single value) of unit_ids for which the spike features will be copied
        '''
        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        if isinstance(unit_ids, int):
            curr_feature_names = sorting.get_unit_spike_feature_names(unit_id=unit_ids)
            for curr_feature_name in curr_feature_names:
                value = sorting.get_unit_spike_features(unit_id=unit_ids, feature_name=curr_feature_name)
                self.set_unit_spike_features(unit_id=unit_ids, feature_name=curr_feature_name, value=value)
        else:
            for unit_id in unit_ids:
                curr_feature_names = sorting.get_unit_spike_feature_names(unit_id=unit_id)
                for curr_feature_name in curr_feature_names:
                    value = sorting.get_unit_spike_features(unit_id=unit_id, feature_name=curr_feature_name)
                    self.set_unit_spike_features(unit_id=unit_id, feature_name=curr_feature_name, value=value)

    def add_epoch(self, epoch_name, start_frame, end_frame):
        '''This function adds an epoch to your sorting extractor that tracks
        a certain time period in your sorting. It is stored in an internal
        dictionary of start and end frame tuples.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be added
        start_frame: int
            The start frame of the epoch to be added (inclusive)
        end_frame: int
            The end frame of the epoch to be added (exclusive). If set to None, it will include the entire
            sorting after the start_frame
        '''
        if isinstance(epoch_name, str):
            start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
            self._epochs[epoch_name] = {'start_frame': start_frame, 'end_frame': end_frame}
        else:
            raise TypeError("epoch_name must be a string")

    def remove_epoch(self, epoch_name):
        '''This function removes an epoch from your sorting extractor.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be removed
        '''
        if isinstance(epoch_name, str):
            if epoch_name in list(self._epochs.keys()):
                del self._epochs[epoch_name]
            else:
                raise ValueError("This epoch has not been added")
        else:
            raise ValueError("epoch_name must be a string")

    def get_epoch_names(self):
        '''This function returns a list of all the epoch names in your sorting

        Returns
        ----------
        epoch_names: list
            List of epoch names in the recording extractor
        '''
        epoch_names = list(self._epochs.keys())
        if not epoch_names:
            pass
        else:
            epoch_start_frames = []
            for epoch_name in epoch_names:
                epoch_info = self.get_epoch_info(epoch_name)
                start_frame = epoch_info['start_frame']
                epoch_start_frames.append(start_frame)
            epoch_names = [epoch_name for _, epoch_name in sorted(zip(epoch_start_frames, epoch_names))]
        return epoch_names

    def get_epoch_info(self, epoch_name):
        '''This function returns the start frame and end frame of the epoch
        in a dict.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be returned

        Returns
        ----------
        epoch_info: dict
            A dict containing the start frame and end frame of the epoch
        '''
        # Default (Can add more information into each epoch in subclass)
        if isinstance(epoch_name, str):
            if epoch_name in list(self._epochs.keys()):
                epoch_info = self._epochs[epoch_name]
                return epoch_info
            else:
                raise ValueError("This epoch has not been added")
        else:
            raise ValueError("epoch_name must be a string")

    def get_epoch(self, epoch_name):
        '''This function returns a SubSortingExtractor which is a view to the
        given epoch

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be returned

        Returns
        ----------
        epoch_extractor: SubRecordingExtractor
            A SubRecordingExtractor which is a view to the given epoch
        '''
        epoch_info = self.get_epoch_info(epoch_name)
        start_frame = epoch_info['start_frame']
        end_frame = epoch_info['end_frame']
        from .subsortingextractor import SubSortingExtractor
        return SubSortingExtractor(parent_sorting=self, start_frame=start_frame,
                                   end_frame=end_frame)

    def get_sub_extractors_by_property(self, property_name, return_property_list=False):
        '''Returns a list of SubSortingExtractors from this SortingExtractor based on the given
        property_name (e.g. group)

        Parameters
        ----------
        property_name: str
            The property used to subdivide the extractor
        return_property_list: bool
            If True the property list is returned

        Returns
        -------
        sub_list: list
            The list of subextractors to be returned

        '''
        if return_property_list:
            sub_list, prop_list = get_sub_extractors_by_property(self, property_name=property_name,
                                                                 return_property_list=return_property_list)
            return sub_list, prop_list
        else:
            sub_list = get_sub_extractors_by_property(self, property_name=property_name,
                                                      return_property_list=return_property_list)
            return sub_list

    @staticmethod
    def write_sorting(sorting, save_path):
        '''This function writes out the spike sorted data file of a given sorting
        extractor to the file format of this current sorting extractor. Allows
        for easy conversion between spike sorting file formats. It is a static
        method so it can be used without instantiating this sorting extractor.

        Parameters
        ----------
        sorting: SortingExtractor
            A SortingExtractor that can extract information from the sorted data
            file to be converted to the new format

        save_path: string
            A path to where the converted sorted data will be saved, which may
            either be a file or a folder, depending on the format
        '''
        raise NotImplementedError("The write_sorting function is not \
                                  implemented for this extractor")

