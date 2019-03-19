from .SortingExtractor import SortingExtractor
import numpy as np


# A Sorting Extractor that allows for manual curation of a sorting result (Represents curation as a tree of units)

class CuratedSortingExtractor(SortingExtractor):

    def __init__(self, parent_sorting):
        SortingExtractor.__init__(self)
        self._parent_sorting = parent_sorting
        self._original_unit_ids = list(np.copy(parent_sorting.getUnitIds()))
        self._all_ids = list(np.copy(parent_sorting.getUnitIds()))

        #Create and store roots with original unit ids and cached spiketrains
        self._roots = []
        for i, unit_id in enumerate(self._original_unit_ids):
            root = Unit(unit_id)
            root.set_spike_train(parent_sorting.getUnitSpikeTrain(unit_id))
            self._roots.append(root)

        self.copyUnitProperties(parent_sorting)
        self.copyUnitSpikeFeatures(parent_sorting)

    def getUnitIds(self):
        unit_ids = []
        for root in self._roots:
            unit_ids.append(root.unit_id)
        return unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
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


    def excludeUnits(self, unit_ids):
        '''This function deletes roots from the curation tree according to the given unit_ids

        Parameters
        ----------
        unit_ids: list
            The unit ids to be excluded
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        if(set(unit_ids).issubset(set(root_ids)) and len(unit_ids) > 0):
            indices_to_be_deleted = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                indices_to_be_deleted.append(root_index)
            self._roots = [self._roots[i] for i,_ in enumerate(root_ids) if i not in indices_to_be_deleted]
        else:
            raise ValueError(str(unit_ids) + " has one or more invalid unit ids")

    def mergeUnits(self, unit_ids):
        '''This function merges two roots from the curation tree according to the given unit_ids. It creates a new unit_id and root
        that has the merged roots as children.

        Parameters
        ----------
        unit_ids: list
            The unit ids to be merged
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        indices_to_be_deleted = []
        if(set(unit_ids).issubset(set(root_ids)) and len(unit_ids) > 1):
            new_root_id = max(self._all_ids)+1
            self._all_ids.append(new_root_id)
            new_root = Unit(new_root_id)
            all_spike_trains = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                new_root.add_child(self._roots[root_index])
                all_spike_trains.append(self._roots[root_index].get_spike_train())
                self._roots[root_index].set_spike_train(np.asarray([])) #clear spiketrain
                indices_to_be_deleted.append(root_index)

            new_root.set_spike_train(np.asarray(np.sort(np.concatenate(all_spike_trains))))
            del all_spike_trains
            self._roots = [self._roots[i] for i,_ in enumerate(root_ids) if i not in indices_to_be_deleted]
            self._roots.append(new_root)
        else:
            raise ValueError(str(unit_ids) + " has one or more invalid unit ids")

    def splitUnit(self, unit_id, indices):
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

            del self._roots[root_index]
            self._roots.append(new_root_1)
            self._roots.append(new_root_2)
        else:
            raise ValueError(str(unit_id) + " non-valid unit id")


# The Unit class is a node in the curation tree. Each Unit contains its unit_id, children, and spike_train.
class Unit(object):
    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.children = []
        self.spike_train = np.asarray([])

    def set_spike_train(self, spike_train):
        self.spike_train = spike_train

    def get_spike_train(self):        return self.spike_train

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
