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
                raise ValueError("Non-valid channel_id")
        else:
            raise ValueError("channel_id must be an int")
        sf = self._start_frame + start_frame
        ef = self._start_frame + end_frame
        if sf < self._start_frame:
            sf = self._start_frame
        if ef > self._end_frame:
            ef = self._end_frame
        return self._parent_sorting.getUnitSpikeTrain(unit_id=original_unit_id, start_frame=sf,
                                                      end_frame=ef) - self._start_frame
