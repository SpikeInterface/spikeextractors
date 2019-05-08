from .SortingExtractor import SortingExtractor
import numpy as np


# A Sorting Extractor that allows for manual curation of a sorting result (Represents curation as a tree of units)

class CurationSortingExtractor(SortingExtractor):

    def __init__(self, parent_sorting):
        SortingExtractor.__init__(self)
        self._parent_sorting = parent_sorting
        self._original_unit_ids = list(np.copy(parent_sorting.get_unit_ids()))
        self._all_ids = list(np.copy(parent_sorting.get_unit_ids()))

        #Create and store roots with original unit ids and cached spiketrains
        self._roots = []
        for i, unit_id in enumerate(self._original_unit_ids):
            root = Unit(unit_id)
            root.set_spike_train(parent_sorting.get_unit_spike_train(unit_id))
            self._roots.append(root)
        '''
        Copies over properties and spike features from parent_sorting.
        Only spike features will be preserved with merges and splits, properties
        cannot be resolved in these cases.
        '''
        self.copy_unit_properties(parent_sorting)
        self.copy_unit_spike_features(parent_sorting)

    def get_unit_ids(self):
        unit_ids = []
        for root in self._roots:
            unit_ids.append(root.unit_id)
        return unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf

        valid_unit_id = False
        spike_train = np.asarray([])
        for root in self._roots:
            if(root.unit_id == unit_id):
                valid_unit_id = True
                full_spike_train = root.get_spike_train()
                inds = np.where((start_frame <= full_spike_train) & (full_spike_train < end_frame))
                spike_train = full_spike_train[inds]
        if(valid_unit_id):
            return spike_train
        else:
            raise ValueError(str(unit_id) + " is an invalid unit id")

    def print_curation_tree(self, unit_id):
        '''This function prints the current curation tree for the unit_id (roots are current unit ids).

        Parameters
        ----------
        unit_id: in
            The unit id whose curation history will be printed.
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)
        if(unit_id in root_ids):
            root_index = root_ids.index(unit_id)
            print(self._roots[root_index])
        else:
            raise ValueError("invalid unit id")


    def exclude_units(self, unit_ids):
        '''This function deletes roots from the curation tree according to the given unit_ids

        Parameters
        ----------
        unit_ids: list
            The unit ids to be excluded
        '''
        if(len(unit_ids) == 0):
            return
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        if(set(unit_ids).issubset(set(root_ids)) and len(unit_ids) > 0):
            indices_to_be_deleted = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                indices_to_be_deleted.append(root_index)
                if unit_id in self._unit_features:
                    del self._unit_features[unit_id]
            self._roots = [self._roots[i] for i,_ in enumerate(root_ids) if i not in indices_to_be_deleted]
        else:
            raise ValueError(str(unit_ids) + " has one or more invalid unit ids")

    def merge_units(self, unit_ids):
        '''This function merges two roots from the curation tree according to the given unit_ids. It creates a new unit_id and root
        that has the merged roots as children.

        Parameters
        ----------
        unit_ids: list
            The unit ids to be merged
        '''
        if(len(unit_ids) <= 1):
            return

        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        indices_to_be_deleted = []
        if(set(unit_ids).issubset(set(root_ids))):
            #Find all unique feature names and create all feature lists
            all_feature_names = []
            for unit_id in unit_ids:
                feature_names = self.get_unit_spike_feature_names(unit_id)
                all_feature_names.append(feature_names)

            shared_feature_names = set(all_feature_names[0])
            for feature_names in all_feature_names[1:]:
                shared_feature_names.intersection_update(feature_names)
            shared_feature_names = list(shared_feature_names)
            shared_features = []
            for i in range(len(shared_feature_names)):
                shared_features.append([])

            new_root_id = max(self._all_ids)+1
            self._all_ids.append(new_root_id)
            new_root = Unit(new_root_id)
            all_spike_trains = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                new_root.add_child(self._roots[root_index])
                all_spike_trains.append(self._roots[root_index].get_spike_train())
                for i, feature_name in enumerate(shared_feature_names):
                    features = self.get_unit_spike_features(unit_id, feature_name)
                    shared_features[i].append(features)
                del self._unit_features[unit_id]
                self._roots[root_index].set_spike_train(np.asarray([])) #clear spiketrain
                indices_to_be_deleted.append(root_index)

            all_spike_trains = np.concatenate(all_spike_trains)
            sort_indices = np.argsort(all_spike_trains)
            new_root.set_spike_train(np.asarray(all_spike_trains)[sort_indices])
            del all_spike_trains
            self._roots = [self._roots[i] for i,_ in enumerate(root_ids) if i not in indices_to_be_deleted]
            self._roots.append(new_root)
            for i, feature_name in enumerate(shared_feature_names):
                self.set_unit_spike_features(new_root_id, feature_name, np.concatenate(shared_features[i])[sort_indices])
        else:
            raise ValueError(str(unit_ids) + " has one or more invalid unit ids")

    def split_unit(self, unit_id, indices):
        '''This function splits a root from the curation tree according to the given unit_id and indices. It creates two new unit_ids
        and roots that have the split root as a child. This function splits the spike train of the root by the given indices.

        Parameters
        ----------
        unit_id: int
            The unit id to be split
        indices: list
            The indices of the unit spike train at which the spike train will be split.
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        if(unit_id in root_ids):
            indices_1 = np.sort(np.asarray(list(set(indices))))

            root_index = root_ids.index(unit_id)
            new_child = self._roots[root_index]
            original_spike_train = self._roots[root_index].get_spike_train()

            try:
                spike_train_1 = original_spike_train[indices_1]
            except IndexError:
                print(str(indices) + " out of bounds for the spike train of " + str(unit_id))

            indices_2 = list(set(range(len(original_spike_train))) - set(indices_1))
            spike_train_2 = original_spike_train[indices_2]
            del original_spike_train

            new_root_1_id = max(self._all_ids)+1
            self._all_ids.append(new_root_1_id)
            new_root_1 = Unit(new_root_1_id)
            new_root_1.add_child(new_child)
            new_root_1.set_spike_train(spike_train_1)

            new_root_2_id = max(self._all_ids)+1
            self._all_ids.append(new_root_2_id)
            new_root_2 = Unit(new_root_2_id)
            new_root_2.add_child(new_child)
            new_root_2.set_spike_train(spike_train_2)

            self._roots.append(new_root_1)
            self._roots.append(new_root_2)

            for feature_name in self.get_unit_spike_feature_names(unit_id):
                full_features = self.get_unit_spike_features(unit_id, feature_name)
                self.set_unit_spike_features(new_root_1_id, feature_name, full_features[indices_1])
                self.set_unit_spike_features(new_root_2_id, feature_name, full_features[indices_2])
            del self._unit_features[unit_id]
            del self._roots[root_index]
        else:
            raise ValueError(str(unit_id) + " non-valid unit id")

    def printCurationTree(self, unit_id):
        '''This function prints the current curation tree for the unit_id (roots are current unit ids).

        Parameters
        ----------
        unit_id: in
            The unit id whose curation history will be printed.
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)
        if(unit_id in root_ids):
            root_index = root_ids.index(unit_id)
            print(self._roots[root_index])
        else:
            raise ValueError("invalid unit id")


# The Unit class is a node in the curation tree. Each Unit contains its unit_id, children, and spike_train.
class Unit(object):
    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.children = []
        self.spike_train = np.asarray([])

    def set_spike_train(self, spike_train):
        self.spike_train = spike_train

    def get_spike_train(self):
        return self.spike_train

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def __str__(self, level=0):
        if(level == 0):
            ret = "\t"*(max(level-1, 0)) +repr(self.unit_id)+ "\n"
        else:
            ret = "\t"*(max(level-1, 0)) + "^-------" +repr(self.unit_id)+ "\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret
