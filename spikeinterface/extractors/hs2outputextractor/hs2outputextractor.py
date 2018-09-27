from spikeinterface import InputExtractor
from spikeinterface import OutputExtractor

import numpy as np
from os.path import join
import h5py
import ctypes


class HS2OutputExtractor(OutputExtractor):
    def __init__(self, *, dataset_directory, recording_files):
        OutputExtractor.__init__(self)
        self._dataset_directory = dataset_directory
        self._recording_files = recording_files
        if type(recording_files) == list:
            if len(recording_files) != 1:
                raise NotImplementedError(
                    "Reading multiple files not yet implemented.")
            ifile = join(dataset_directory, recording_files[0])
        else:
            ifile = join(dataset_directory, recording_files)
        self._rf = h5py.File(ifile, mode='r')
        self._unit_ids = set(self._rf['cluster_id'].value)

    def getUnitIds(self):
        return self._unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._rf['times'].value[np.where(
            self._rf['cluster_id'].value == unit_id)[0]]
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    def getUnitChannels(self, unit_id):
        return self._rf['ch'].value[np.where(self._rf['cluster_id'].value == unit_id)[0]]
