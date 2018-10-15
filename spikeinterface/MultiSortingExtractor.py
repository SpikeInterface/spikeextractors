from .SortingExtractor import SortingExtractor
from .RecordingExtractor import RecordingExtractor
import numpy as np

# Encapsulates a subset of a spike sorted dataset

class MultiSortingExtractor(SortingExtractor):
    # We need the sorting extractors so that we can determine the number of frames in each
    def __init__(self, sortings, recordings, epoch_names=None):
        self._RXs=recordings
        self._RX=MultiRecordingExtractor(recordings=recordings)
        self._SXs=sortings
        self._all_unit_ids=[]
        for SX in self._SXs:
            unit_ids=SX.getUnitIds()
            for unit_id in unit_ids:
                self._all_unit_ids.append(unit_id)
        self._all_unit_ids=list(set(self._all_unit_ids)) # unique ids

    def getUnitIds(self):
        return self._all_unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self._RX.getNumFrames()
        RX1, i_sec1, i_start_frame = self._RX._find_section_for_frame(start_frame)
        RX2, i_sec2, i_end_frame = self._RX._find_section_for_frame(end_frame)
        if i_sec1==i_sec2:
            f0=self._RX._start_frames[i_sec1]
            return f0+self._SXs[i_sec1].getUnitSpikeTrain(unit_id=unit_id,start_frame=i_start_frame,end_frame=i_end_frame)
        list=[]
        list.append(
            self._RX._start_frames[i_sec1]+self._SXs[i_sec1].getUnitSpikeTrain(unit_id=unit_id,start_frame=i_start_frame,end_frame=self._RXs[i_sec1].getNumFrames())
        )
        for i_sec in range(i_sec1+1,i_sec2):
            list.append(
                self._RX._start_frames[i_sec]+self._SXs[i_sec].getUnitSpikeTrain(unit_id=unit_id)
            )
        list.append(
            self._RX._start_frames[i_sec2]+self._SXs[i_sec2].getUnitSpikeTrain(unit_id=unit_id,start_frame=0,end_frame=i_end_frame)
        )
        return np.concatenate(list)
