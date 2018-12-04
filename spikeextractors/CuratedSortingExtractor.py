from .SortingExtractor import SortingExtractor
import numpy as np


# A Sorting Extractor that allows for manual curation of a sorting result (Represents curation as a tree of units)

class CuratedSortingExtractor(SortingExtractor):

    def __init__(self, parent_sorting):
        SortingExtractor.__init__(self)
        self._parent_sorting = parent_sorting
        self._original_unit_ids = list(np.copy(parent_sorting.getUnitIds()))
        self._all_ids = list(np.copy(parent_sorting.getUnitIds()))
        original_spike_indices = []
        for unit_id in self._original_unit_ids:
            original_spike_indices.append(np.asarray(list(range(len(parent_sorting.getUnitSpikeTrain(unit_id))))))
        self._indices_dict = {}
        for i, unit_id in enumerate(self._original_unit_ids):
            self._indices_dict[unit_id] = original_spike_indices[i]
        del original_spike_indices

        self._roots = []
        for i, unit_id in enumerate(self._original_unit_ids):
            root = Unit(unit_id)
            self._roots.append(root)

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
                self._collectSpikeTrains(root)
                inds = np.where((start_frame <= root.spike_train) & (root.spike_train < end_frame))
                spike_train = root.spike_train[inds]
                self._resetSpikeTrains(root)
        if(valid_unit_id):
            return spike_train
        else:
            raise ValueError("invalid unit id")

    def _collectSpikeTrains(self, unit): #post order traversal
        for child in unit.get_children():
            self._collectSpikeTrains(child)
        if(len(unit.get_children()) == 0):
            unit.spike_train = self._parent_sorting.getUnitSpikeTrain(unit.unit_id)
        else:
            new_spike_train = []
            for child in unit.get_children():
                new_spike_train.append(child.spike_train)
            new_spike_train = np.asarray(new_spike_train)
            new_spike_train = np.sort(np.concatenate(new_spike_train).ravel())
            new_spike_train = new_spike_train[self._indices_dict[unit.unit_id]]
            unit.spike_train = new_spike_train

    def _resetSpikeTrains(self, unit): #post order traversal
        for child in unit.get_children():
            self._resetSpikeTrains(child)
        unit.spike_train = np.asarray([])

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
                del self._indices_dict[unit_id]
            self._roots = [self._roots[i] for i,_ in enumerate(root_ids) if i not in indices_to_be_deleted]
        else:
            raise ValueError("invalid unit ids")

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
        if(set(unit_ids).issubset(set(root_ids)) and len(unit_ids) > 1):
            new_root_id = max(self._all_ids)+1
            self._all_ids.append(new_root_id)
            new_root = Unit(new_root_id)
            indices_to_be_deleted = []
            self._indices_dict[new_root_id] = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                new_root.add_child(self._roots[root_index])
                for spike_index in self._indices_dict[unit_id]:
                    self._indices_dict[new_root_id].append(len(self._indices_dict[new_root_id]))
                indices_to_be_deleted.append(root_index)
            self._indices_dict[new_root_id] = np.asarray(self._indices_dict[new_root_id])
            self._roots = [self._roots[i] for i,_ in enumerate(root_ids) if i not in indices_to_be_deleted]
            self._roots.append(new_root)
        else:
            raise ValueError("invalid unit ids")

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
            root_index = root_ids.index(unit_id)
            root_spike_indices = self._indices_dict[unit_id]
            if(set(indices).issubset(set(root_spike_indices)) and len(indices) > 0 and len(indices) < len(root_spike_indices)):
                new_child = self._roots[root_index]

                new_root_1_id = max(self._all_ids)+1
                self._all_ids.append(new_root_1_id)
                new_root_1 = Unit(new_root_1_id)
                new_root_1.add_child(new_child)
                self._indices_dict[new_root_1_id] = np.asarray(indices)

                non_indices = [index for i, index in enumerate(root_spike_indices) if i not in indices]
                new_root_2_id = max(self._all_ids)+1
                self._all_ids.append(new_root_2_id)
                new_root_2 = Unit(new_root_2_id)
                new_root_2.add_child(new_child)
                self._indices_dict[new_root_2_id] = np.asarray(non_indices)

                del self._roots[root_index]
                self._roots.append(new_root_1)
                self._roots.append(new_root_2)
            else:
                raise ValueError("invalid indices")
        else:
            raise ValueError("invalid unit ids")


# The Unit class is a node in the curation tree. Each Unit contains its unit_id, children, and spike_train.
class Unit(object):
    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.children = []
        self.spike_train = np.asarray([])

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
