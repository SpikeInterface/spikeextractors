from .SortingExtractor import SortingExtractor
from .RecordingExtractor import RecordingExtractor
import numpy as np


# Encapsulates a subset of a spike sorted dataset

class MultiSortingExtractor(SortingExtractor):
    def __init__(self, *, sortings, start_frames=None):
        SortingExtractor.__init__(self)
        self._SXs = sortings
        self._all_unit_ids = []
        for SX in self._SXs:
            unit_ids = SX.getUnitIds()
            for unit_id in unit_ids:
                self._all_unit_ids.append(unit_id)

        if start_frames is None:
            start_frames = []
            for i in range(len(self._SXs)):
                start_frames.append(0)

        self._start_frames = start_frames
        self._all_unit_ids = list(set(self._all_unit_ids))  # unique ids

    def getUnitIds(self):
        return self._all_unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        spike_train = []
        for i, SX in enumerate(self._SXs):
            if (unit_id in SX.getUnitIds()):
                section_start_frame = max(0, start_frame - self._start_frames[i])
                section_end_frame = max(0, end_frame - self._start_frames[i])
                section_spike_train = self._start_frames[i] + SX.getUnitSpikeTrain(unit_id=unit_id,
                                                                                   start_frame=section_start_frame,
                                                                                   end_frame=section_end_frame)
                spike_train.append(section_spike_train)
        if (not spike_train):
            return np.asarray(spike_train)
        else:
            return np.asarray(np.sort(np.concatenate(spike_train)))
