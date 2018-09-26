from abc import ABC, abstractmethod

class OutputExtractor(ABC):
    '''A class that contains functions for extracting important information
    from the output data of spike sorting software. It is an abstract class so
    all functions with the @abstractmethod tag must be implemented for the
    initialization to work.


    '''
    def __init__(self):
        pass

    @abstractmethod
    def getUnitIds(self):
        '''This function returns a list of ids (ints) for each unit in the recording

        Returns
        ----------
        unit_ids: list
            A list of unit ids (ints)
        '''
        pass

    @abstractmethod
    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        '''This function extracts spike frames from the specified unit.
        It will return spike frames within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame]
            [beginning_unit_spike_frame, beginning_unit_spike_frame+1, ..., end_frame-1]

        if both start_frame and end_frame are given, if only start_frame is
        given, or if only end_frame is given, respectively. Spike frames are
        returned in the form of a array_like of spike frames.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording.
        start_frame: int
            The frame above which a spike frame is returned.
        end_frame: int
            The frame below which a spike frame is returned.
        Returns
        ----------
        spike_train: numpy.ndarray
            An 1D array containing all the frames for each spike in the
            specified unit given the range of start and end frames.
        '''
        pass
