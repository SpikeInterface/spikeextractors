from abc import ABC, abstractmethod
import numpy as np
from .MultiInputExtractor import MultiInputExtractor

# Encapsulates a subset of a spike sorting output

class MultiOutputExtractor(ABC):
    # We need the input extractors so that we can determine the number of frames in each
    def __init__(self, *, input_extractors, output_extractors, epoch_names):
        self._IXs=input_extractors
        self._IX=MultiInputExtractor(input_extractors=input_extractors)
        self._OXs=output_extractors
        self._all_unit_ids=[]
        for OX in self._OXs:
            unit_ids=OX.getUnitIds()
            for unit_id in unit_ids:
                self._all_unit_ids.append(unit_id)
        self._all_unit_ids=list(set(self._all_unit_ids)) # unique ids

    def getUnitIds(self):
        return self._all_unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self._IX.getNumFrames()
        IX1, i_sec1, i_start_frame = self._IX._find_section_for_frame(start_frame)
        IX2, i_sec2, i_end_frame = self._IX._find_section_for_frame(end_frame)
        if i_sec1==i_sec2:
            f0=self._IX._start_frames[i_sec1]
            return f0+self._OXs[i_sec1].getUnitSpikeTrain(unit_id=unit_id,start_frame=i_start_frame,end_frame=i_end_frame)
        list=[]
        list.append(
            self._IX._start_frames[i_sec1]+self._OXs[i_sec1].getUnitSpikeTrain(unit_id=unit_id,start_frame=i_start_frame,end_frame=self._IXs[i_sec1].getNumFrames())
        )
        for i_sec in range(i_sec1+1,i_sec2):
            list.append(
                self._IX._start_frames[i_sec]+self._OXs[i_sec].getUnitSpikeTrain(unit_id=unit_id)
            )
        list.append(
            self._IX._start_frames[i_sec2]+self._OXs[i_sec2].getUnitSpikeTrain(unit_id=unit_id,start_frame=0,end_frame=i_end_frame)
        )
        return np.concatenate(list)
