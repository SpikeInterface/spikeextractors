from abc import ABC, abstractmethod
import numpy as np

class InputExtractor(ABC):
    '''A class that contains functions for extracting important information
    from input data to spike sorting software. It is an abstract class so all
    functions with the @abstractmethod tag must be implemented for the
    initialization to work.


    '''
    def __init__(self):
        pass

    @abstractmethod
    def getRawTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        '''This function extracts and returns a trace from the raw data from the
        given channels ids and the given start and end frame. It will return
        raw traces from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_recording_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_recording_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Raw traces are returned in a 2D array that
        contains all of the raw traces from each channel with dimensions
        (num_channels x num_frames). In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        start_frame: int
            The starting frame of the trace to be returned (inclusive).
        end_frame: int
            The ending frame of the trace to be returned (exclusive).
        channel_ids: array_like
            A list or 1D array of channel ids (ints) from which each trace will be
            extracted.

        Returns
        ----------
        raw_traces: numpy.ndarray
            A 2D array that contains all of the raw traces from each channel.
            Dimensions are: (num_channels x num_frames)
        '''
        pass

    @abstractmethod
    def getNumChannels(self):
        '''This function returns the number of channels in the recording.

        Returns
        -------
        num_channels: int
            Number of channels in the recording.
        '''
        pass

    @abstractmethod
    def getNumFrames(self):
        '''This function returns the number of frames in the recording.

        Returns
        -------
        num_frames: int
            Number of frames in the recording (duration of recording).
        '''
        pass

    @abstractmethod
    def getSamplingFrequency(self):
        '''This function returns the sampling frequency in units of Hz.

        Returns
        -------
        fs: float
            Sampling frequency of the recordings in Hz.
        '''
        pass

    def frameToTime(self, frame):
        '''This function converts a user-inputted frame to a time with units of seconds.
        It should handle both scalars and lists.

        Parameters
        ----------
        frame: float
            The frame (or list of frames) to be converted to a time.

        Returns
        -------
        time: float
            The corresponding time or list of times in seconds.
        '''
        # Default implementation
        return frame/self.getSamplingFrequency()

    def timeToFrame(self, time):
        '''This function converts a user-inputted time (in seconds) to a frame index.

        Parameters
        -------
        time: float
            The time or list of times (in seconds) to be converted to frames.

        Returns
        -------
        frame: float
            The corresponding frame or list of frames.
        '''
        # Default implementation
        return time*self.getSamplingFrequency()

    def getRawSnippets(self, snippet_len, center_frames, channel_ids=None):
        '''This function returns raw data snippets from the given channels that
        are centered on the given frames and are the length of the given snippet
        length.

        Parameters
        ----------
        snippet_len: int
            The length of each snippet in frames.
        center_frames: array_like
            A list or array of frames that will be used as the center frame of
            each snippet.
        channel_ids: array_like
            A list or array of channel ids (ints) from which each trace will be
            extracted.

        Returns
        ----------
        raw_snippets: numpy.ndarray
            Returns a list of the raw snippets as numpy arrays.
            The length of the list is len(center_frames)
            Each array has dimensions: (num_channels x snippet_len)
            Out-of-bounds cases should be handled by filling in zeros in the snippet.
        '''
        # Default implementation
        if channel_ids is None:
            channel_ids = range(self.getNumChannels())

        num_snippets = len(center_frames)
        num_channels = len(channel_ids)
        num_frames = self.getNumFrames()
        snippet_half_len = int(snippet_len/2)
        raw_snippets = []
        for i in range(num_snippets):
            snippet_chunk = np.zeros((num_channels,snippet_len))
            if (0<=center_frames[i]) and (center_frames[i]<num_frames):
                snippet_range = np.array([int(center_frames[i])-snippet_half_len, int(center_frames[i])-snippet_half_len+snippet_len])
                snippet_buffer = np.array([0,snippet_len])
                # The following handles the out-of-bounds cases
                if snippet_range[0] < 0:
                    snippet_buffer[0] -= snippet_range[0]
                    snippet_range[0] -= snippet_range[0]
                if snippet_range[1] >= num_frames:
                    snippet_buffer[1] -= snippet_range[1] + num_frames
                    snippet_range[1] -= snippet_range[1] + num_frames
                snippet_chunk[:,snippet_buffer[0]:snippet_buffer[1]] = self.getRawTraces(start_frame=snippet_range[0],
                                                                                         end_frame=snippet_range[1],
                                                                                         channel_ids=channel_ids)
            raw_snippets.append(snippet_chunk)

        return raw_snippets

    def getChannelInfo(self, channel_id):
        '''This function returns the a dictionary containing information about
        the channel specified by the channel id. Important keys in the dict to
        fill out should be: 'group', 'position', and 'type'.

        Parameters
        ----------
        channel_id: int
            The channel id of the channel for which information is returned.

        Returns
        ----------
        channel_info_dict: dict
            A dictionary containing important information about the specified
            channel. Should include:

                    key = 'group', type = int
                        the group number it is in, for tetrodes

                    key = 'position', type = array_like
                        two/three dimensional

                    key = 'type', type = string
                        recording ('rec') or reference ('ref')
        '''
        raise NotImplementedError("The getChannelInfo function is not \
                                  implemented for this extractor")

    def getEpochNames(self):
        return []

    def getEpochInfo(self,epoch_name):
        raise NotImplementedError("The getEpochInfo function is not \
                                  implemented for this extractor")

    def getEpoch(self,epoch_name):
        from .SubInputExtractor import SubInputExtractor
        return SubInputExtractor(parent_extractor=self,epoch_name=epoch_name)

    @staticmethod
    def writeInput(self, input_extractor, save_path):
        '''This function writes out the input file of a given input extractor
        to the file format of this current input extractor. Allows for easy
        conversion between input file formats. It is a static method so it
        can be used without instantiating this input extractor.

        Parameters
        ----------
        input_extractor: InputExtractor
            An InputExtractor that can extract information from the input file
            to be converted to the new format.

        save_path: string
            A path to where the converted input data will be saved, which may
            either be a file or a folder, depending on the format.
        '''
        raise NotImplementedError("The writeInput function is not \
                                  implemented for this extractor")
