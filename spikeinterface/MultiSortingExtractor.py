from .SortingExtractor import SortingExtractor
from .RecordingExtractor import RecordingExtractor
import numpy as np

class MultiSortingExtractor(SortingExtractor):
    def __init__(self, *, sorting_extractors, start_frames, epoch_names=None):
        self._SXs=sorting_extractors
        self._start_frames=start_frames
        self._all_unit_ids=[]
        for SX in self._SXs:
            unit_ids=SX.getUnitIds()
            for unit_id in unit_ids:
                self._all_unit_ids.append(unit_id)
        self._all_unit_ids=list(set(self._all_unit_ids)) # unique ids

    def getUnitIds(self):
        return self._all_unit_ids

    def _find_section_for_frame(self,frame):
        inds=np.where(np.array(self._start_frames[:-1])<=frame)[0]
        ind=inds[-1]
        return ind, frame-self._start_frames[ind]

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=np.Inf
        i_sec1, i_start_frame = self._find_section_for_frame(start_frame)
        i_sec2, i_end_frame = self._find_section_for_frame(end_frame)
        if i_sec1==i_sec2:
            f0=self._start_frames[i_sec1]
            return f0+self._SXs[i_sec1].getUnitSpikeTrain(unit_id=unit_id,start_frame=i_start_frame,end_frame=i_end_frame)
        list=[]
        list.append(
            self._start_frames[i_sec1]+self._SXs[i_sec1].getUnitSpikeTrain(unit_id=unit_id,start_frame=i_start_frame,end_frame=np.Inf)
        )
        for i_sec in range(i_sec1+1,i_sec2):
            list.append(
                self._start_frames[i_sec]+self._SXs[i_sec].getUnitSpikeTrain(unit_id=unit_id)
            )
        list.append(
            self._start_frames[i_sec2]+self._SXs[i_sec2].getUnitSpikeTrain(unit_id=unit_id,start_frame=0,end_frame=i_end_frame)
        )
        return np.concatenate(list)
