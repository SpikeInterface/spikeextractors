from abc import ABC, abstractmethod
import numpy as np

# Encapsulates a subset of a spike sorting output

class SubOutputExtractor(ABC):

    def __init__(self, parent_extractor, *, unit_ids=None, new_unit_ids=None, start_frame=None, end_frame=None):
        self._parent_extractor=parent_extractor
        self._unit_ids=unit_ids
        self._new_unit_ids=new_unit_ids
        self._start_frame=start_frame
        self._end_frame=end_frame
        if self._unit_ids is None:
            self._unit_ids=self._parent_extractor.getUnitIds()
        if self._new_unit_ids is None:
            self._new_unit_ids=self._unit_ids
        if self._start_frame is None:
            self._start_frame=0
        if self._end_frame is None:
            self._end_frame=self._parent_extractor.getNumFrames()
        self._original_unit_id_lookup={}
        for i in range(len(self._unit_ids)):
            self._original_unit_id_lookup[self._new_unit_ids[i]]=self._unit_ids[i]

    def getUnitIds(self):
        return self._new_unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        u=self._original_unit_id_lookup[unit_id]
        sf=self._start_frame+start_frame
        ef=self._end_frame+end_frame
        return np.array(getUnitSpikeTrain(unit_id=u,start_frame=sf,end_frame=ef))-self._start_frame
