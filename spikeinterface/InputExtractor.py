from abc import ABC, abstractmethod

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
        given channels ids and the given start and end frame.

        Parameters
        ----------
        start_frame: int
            The starting frame of the trace to be returned
        end_frame: int
            The ending frame of the trace to be returned
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
        '''This function returns the number of channels in the recording

        Returns
        -------
        num_channels: int
            Number of channels in the recording
        '''
        pass

    @abstractmethod
    def getNumFrames(self):
        '''This function returns the number of frames in the recording

        Returns
        -------
        num_frames: int
            Number of frames in the recording (duration of recording)
        '''
        pass

    @abstractmethod
    def getSamplingFrequency(self):
        '''This function returns the sampling frequency in units of Hz

        Returns
        -------
        fs: float
            Sampling frequency of the recordings in Hz
        '''
        pass

    def frameToTime(self, frame):
        '''This function converts a user-inputted frame to a time with units of seconds.
        It should handle both scalars and lists.

        Parameters
        ----------
        frame: float
            The frame (or list of frames) to be converted to a time
            
        Returns
        -------
        time: float
            The corresponding time or list of times in seconds
        '''
        raise NotImplementedError("The frameToTime function is not \
                                  implemented for this extractor")

    def timeToFrame(self, time):
        '''This function converts a user-inputted time (in seconds) to a frame index

        Parameters
        -------
        time: float
            The time or list of times (in seconds) to be converted to frames
            
        Returns
        -------
        frame: float
            The corresponding frame or list of frames
        '''
        raise NotImplementedError("The timeToFrame function is not \
                                  implemented for this extractor")

    def getRawSnippets(self, snippet_len, center_frames, channel_ids):
        '''This function returns raw data snippets from the given channels that
        are centered on the given frames and are the length of the given snippet
        length.

        Parameters
        ----------
        snippet_len: int
            The length of each snippet in frames
        center_frames: array_like
            A list or array of frames that will be used as the center frame of
            each snippet.
        channel_ids: array_like
            A list or array of channel ids (ints) from which each trace will be
            extracted.

        Returns
        ----------
        raw_snippets: numpy.ndarray
            A 3D array that contains all of the raw snippets from each channel.
            Dimensions are: (num_channels x num_snippets x snippet_len)
        '''
        raise NotImplementedError("The getRawSnippets function is not \
                                  implemented for this extractor")


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
