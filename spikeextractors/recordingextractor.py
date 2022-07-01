from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

from .extraction_tools import load_probe_file, save_to_probe_file, write_to_binary_dat_format, \
    write_to_h5_dataset_format, get_sub_extractors_by_property, cast_start_end_frame
from .baseextractor import BaseExtractor


class RecordingExtractor(ABC, BaseExtractor):
    """A class that contains functions for extracting important information
    from recorded extracellular data. It is an abstract class so all
    functions with the @abstractmethod tag must be implemented for the
    initialization to work.
    """

    _default_filename = "spikeinterface_recording"

    def __init__(self):
        BaseExtractor.__init__(self)
        self._key_properties = {'group': None, 'location': None, 'gain': None, 'offset': None}
        self.is_filtered = False

    @abstractmethod
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        """This function extracts and returns a trace from the recorded data from the
        given channels ids and the given start and end frame. It will return
        traces from within three ranges:

            [start_frame, start_frame+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_recording_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_recording_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Traces are returned in a 2D array that
        contains all of the traces from each channel with dimensions
        (num_channels x num_frames). In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        channel_ids: array_like
            A list or 1D array of channel ids (ints) from which each trace will be extracted.
        start_frame: int
            The starting frame of the trace to be returned (inclusive).
        end_frame: int
            The ending frame of the trace to be returned (exclusive).
        return_scaled: bool
            If True, traces are returned after scaling (using gain/offset).
            If False, the raw traces are returned.

        Returns
        -------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (num_channels x num_frames)
        """
        pass

    @abstractmethod
    def get_num_frames(self):
        """This function returns the number of frames in the recording

        Returns
        -------
        num_frames: int
            Number of frames in the recording (duration of recording)
        """
        pass

    @abstractmethod
    def get_sampling_frequency(self):
        """This function returns the sampling frequency in units of Hz.

        Returns
        -------
        fs: float
            Sampling frequency of the recordings in Hz
        """
        pass

    @abstractmethod
    def get_channel_ids(self):
        """Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.

        Returns
        -------
        channel_ids: list
            Channel list
        """
        pass

    def get_num_channels(self):
        """This function returns the number of channels in the recording.

        Returns
        -------
        num_channels: int
            Number of channels in the recording
        """
        return len(self.get_channel_ids())

    def get_dtype(self, return_scaled=True):
        """This function returns the traces dtype

        Parameters
        ----------
        return_scaled: bool
            If False and the recording extractor has unscaled traces, it returns the dtype of unscaled traces.
            If True (default) it returns the dtype of the scaled traces

        Returns
        -------
        dtype: np.dtype
            The dtype of the traces
        """
        return self.get_traces(channel_ids=[self.get_channel_ids()[0]], start_frame=0, end_frame=1,
                               return_scaled=return_scaled).dtype

    def set_times(self, times):
        """This function sets the recording times (in seconds) for each frame

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame
        """
        assert len(times) == self.get_num_frames(), "'times' should have the same length of the " \
                                                    "number of frames"
        self._times = times.astype('float64')

    def copy_times(self, extractor):
        """This function copies times from another extractor.

        Parameters
        ----------
        extractor: BaseExtractor
            The extractor from which the epochs will be copied
        """
        if extractor._times is not None:
            self.set_times(deepcopy(extractor._times))

    def frame_to_time(self, frames):
        """This function converts user-inputted frame indexes to times with units of seconds.

        Parameters
        ----------
        frames: float or array-like
            The frame or frames to be converted to times

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds
        """
        # Default implementation
        if self._times is None:
            return np.round(frames / self.get_sampling_frequency(), 6)
        else:
            return self._times[frames]

    def time_to_frame(self, times):
        """This function converts a user-inputted times (in seconds) to a frame indexes.

        Parameters
        -------
        times: float or array-like
            The times (in seconds) to be converted to frame indexes

        Returns
        -------
        frames: float or array-like
            The corresponding frame indexes
        """
        # Default implementation
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype('int64')
        else:
            return np.searchsorted(self._times, times).astype('int64')

    def get_snippets(self, reference_frames, snippet_len, channel_ids=None, return_scaled=True):
        """This function returns data snippets from the given channels that
        are starting on the given frames and are the length of the given snippet
        lengths before and after.

        Parameters
        ----------
        reference_frames: array_like
            A list or array of frames that will be used as the reference frame of each snippet.
        snippet_len: int or tuple
            If int, the snippet will be centered at the reference frame and
            and return half before and half after of the length. If tuple,
            it will return the first value of before frames and the second value
            of after frames around the reference frame (allows for asymmetry).
        channel_ids: array_like
            A list or array of channel ids (ints) from which each trace will be
            extracted
        return_scaled: bool
            If True, snippets are returned after scaling (using gain/offset).
            If False, the raw traces are returned.

        Returns
        -------
        snippets: numpy.ndarray
            Returns a list of the snippets as numpy arrays.
            The length of the list is len(reference_frames)
            Each array has dimensions: (num_channels x snippet_len)
            Out-of-bounds cases should be handled by filling in zeros in the snippet
        """
        # Default implementation
        if isinstance(snippet_len, (tuple, list, np.ndarray)):
            snippet_len_before = int(snippet_len[0])
            snippet_len_after = int(snippet_len[1])
        else:
            snippet_len_before = int((snippet_len + 1) / 2)
            snippet_len_after = int(snippet_len - snippet_len_before)

        if channel_ids is None:
            channel_ids = self.get_channel_ids()

        num_snippets = len(reference_frames)
        num_channels = len(channel_ids)
        num_frames = self.get_num_frames()
        snippet_len_total = int(snippet_len_before + snippet_len_after)
        snippets = np.zeros((num_snippets, num_channels, snippet_len_total), dtype=self.get_dtype(return_scaled))

        for i in range(num_snippets):
            snippet_chunk = np.zeros((num_channels, snippet_len_total), dtype=self.get_dtype(return_scaled))
            if 0 <= reference_frames[i] < num_frames:
                snippet_range = np.array([int(reference_frames[i]) - snippet_len_before,
                                          int(reference_frames[i]) + snippet_len_after])
                snippet_buffer = np.array([0, snippet_len_total], dtype='int')
                # The following handles the out-of-bounds cases
                if snippet_range[0] < 0:
                    snippet_buffer[0] -= snippet_range[0]
                    snippet_range[0] -= snippet_range[0]
                if snippet_range[1] >= num_frames:
                    snippet_buffer[1] -= snippet_range[1] - num_frames
                    snippet_range[1] -= snippet_range[1] - num_frames
                snippet_chunk[:, snippet_buffer[0]:snippet_buffer[1]] = self.get_traces(channel_ids=channel_ids,
                                                                                        start_frame=snippet_range[0],
                                                                                        end_frame=snippet_range[1],
                                                                                        return_scaled=return_scaled)
            snippets[i] = snippet_chunk
        return snippets

    def set_channel_locations(self, locations, channel_ids=None):
        """This function sets the location key properties of each specified channel
        id with the corresponding locations of the passed in locations list.

        Parameters
        ----------
        locations: array_like
            A list of corresponding locations (array_like) for the given channel_ids
        channel_ids: array-like or int
            The channel ids (ints) for which the locations will be specified. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
            locations = [locations]
        # Only None upon initialization
        if self._key_properties['location'] is None:
            default_locations = np.empty((self.get_num_channels(), 3), dtype='float')
            default_locations[:] = np.nan
            self._key_properties['location'] = default_locations
        if len(channel_ids) == len(locations):
            for i in range(len(channel_ids)):
                if isinstance(locations[i], (list, np.ndarray, tuple)):
                    location = np.asarray(locations[i])
                    channel_idx = list(self.get_channel_ids()).index(channel_ids[i])
                    if len(location) == 2:
                        self._key_properties['location'][channel_idx, :2] = location
                    elif len(location) == 3:
                        self._key_properties['location'][channel_idx] = location
                    else:
                        raise TypeError("'location' must be 2d ior 3d")
                else:
                    raise TypeError("'location' must be an array like object")
        else:
            raise ValueError("channel_ids and locations must have same length")

    def get_channel_locations(self, channel_ids=None, locations_2d=True):
        """This function returns the location of each channel specified by channel_ids

        Parameters
        ----------
        channel_ids: array-like or int
            The channel ids (ints) for which the locations will be returned. If None, all channel ids are assumed.
        locations_2d: bool
            If True (default), first two dimensions are returned

        Returns
        -------
        locations: array_like
            Returns a list of corresponding locations (floats) for the given
            channel_ids
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        locations = self._key_properties['location']
        # Only None upon initialization
        if locations is None:
            locations = np.empty((self.get_num_channels(), 3), dtype='float')
            locations[:] = np.nan
            self._key_properties['location'] = locations
        locations = np.array(locations)
        channel_idxs = np.array([list(self.get_channel_ids()).index(ch) for ch in channel_ids])
        if locations_2d:
            locations = np.array(locations)[:, :2]
        return locations[channel_idxs]

    def clear_channel_locations(self, channel_ids=None):
        """This function clears the location of each channel specified by channel_ids.

        Parameters
        ----------
        channel_ids: array-like or int
            The channel ids (ints) for which the locations will be cleared. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        # Reset to default locations (NaN)
        default_locations =  np.array([[np.nan, np.nan, np.nan] for i in range(len(channel_ids))])
        self.set_channel_locations(default_locations, channel_ids)

    def set_channel_groups(self, groups, channel_ids=None):
        """This function sets the group key property of each specified channel
        id with the corresponding group of the passed in groups list.

        Parameters
        ----------
        groups: array-like or int
            A list of groups (ints) for the channel_ids
        channel_ids: array_like or None
            The channel ids (ints) for which the groups will be specified. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        if isinstance(groups, (int, np.integer)):
            groups = [groups]
        # Only None upon initialization
        if self._key_properties['group'] is None:
            self._key_properties['group'] = np.zeros(self.get_num_channels(), dtype='int')
        if len(channel_ids) == len(groups):
            for i in range(len(channel_ids)):
                if isinstance(groups[i], (int, np.integer)):
                    channel_idx = list(self.get_channel_ids()).index(channel_ids[i])
                    self._key_properties['group'][channel_idx] = int(groups[i])
                else:
                    raise TypeError("'group' must be an int")
        else:
            raise ValueError("channel_ids and groups must have same length")

    def get_channel_groups(self, channel_ids=None):
        """This function returns the group of each channel specified by
        channel_ids

        Parameters
        ----------
        channel_ids: array-like or int
            The channel ids (ints) for which the groups will be returned

        Returns
        -------
        groups: array_like
            Returns a list of corresponding groups (ints) for the given channel_ids
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        groups = self._key_properties['group']
        # Only None upon initialization
        if groups is None:
            groups = np.zeros(self.get_num_channels(), dtype='int')
            self._key_properties['group'] = groups
        groups = np.array(groups)
        channel_idxs = np.array([list(self.get_channel_ids()).index(ch) for ch in channel_ids])
        return groups[channel_idxs]

    def clear_channel_groups(self, channel_ids=None):
        """This function clears the group of each channel specified by channel_ids

        Parameters
        ----------
        channel_ids: array-like or int
            The channel ids (ints) for which the groups will be cleared. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        # Reset to default groups (0)
        default_groups = [0] * len(channel_ids)
        self.set_channel_groups(default_groups, channel_ids)

    def set_channel_gains(self, gains, channel_ids=None):
        """This function sets the gain key property of each specified channel
        id with the corresponding group of the passed in gains float/list.

        Parameters
        ----------
        gains: float/array_like
            If a float, each channel will be assigned the corresponding gain.
            If a list, each channel will be given a gain from the list
        channel_ids: array_like or None
            The channel ids (ints) for which the groups will be specified. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        if isinstance(gains, (int, np.integer, float)):
            gains = [gains] * len(channel_ids)
        # Only None upon initialization
        if self._key_properties['gain'] is None:
            self._key_properties['gain'] = np.ones(self.get_num_channels(), dtype='float')
        if len(channel_ids) == len(gains):
            for i in range(len(channel_ids)):
                if isinstance(gains[i], (int, np.integer, float)):
                    channel_idx = list(self.get_channel_ids()).index(channel_ids[i])
                    self._key_properties['gain'][channel_idx] = float(gains[i])
                else:
                    raise TypeError("'gain' must be an int or float")
        else:
            raise ValueError("channel_ids and gains must have same length")

    def get_channel_gains(self, channel_ids=None):
        """This function returns the gain of each channel specified by channel_ids.

        Parameters
        ----------
        channel_ids: array_like
            The channel ids (ints) for which the gains will be returned

        Returns
        -------
        gains: array_like
            Returns a list of corresponding gains (floats) for the given channel_ids
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        gains = self._key_properties['gain']
        # Only None upon initialization
        if gains is None:
            gains = np.ones(self.get_num_channels(), dtype='float')
            self._key_properties['gain'] = gains
        gains = np.array(gains)
        channel_idxs = np.array([list(self.get_channel_ids()).index(ch) for ch in channel_ids])
        return gains[channel_idxs]

    def clear_channel_gains(self, channel_ids=None):
        """This function clears the gains of each channel specified by channel_ids

        Parameters
        ----------
        channel_ids: array-like or int
            The channel ids (ints) for which the groups will be cleared. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        # Reset to default gains (1)
        default_gains = [1.] * len(channel_ids)
        self.set_channel_gains(default_gains, channel_ids)

    def set_channel_offsets(self, offsets, channel_ids=None):
        """This function sets the offset key property of each specified channel
        id with the corresponding group of the passed in gains float/list.

        Parameters
        ----------
        offsets: float/array_like
            If a float, each channel will be assigned the corresponding offset.
            If a list, each channel will be given an offset from the list
        channel_ids: array_like or None
            The channel ids (ints) for which the groups will be specified. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        if isinstance(offsets, (int, np.integer, float)):
            offsets = [offsets] * len(channel_ids)
        # Only None upon initialization
        if self._key_properties['offset'] is None:
            self._key_properties['offset'] = np.zeros(self.get_num_channels(), dtype='float')
        if len(channel_ids) == len(offsets):
            for i in range(len(channel_ids)):
                if isinstance(offsets[i], (int, np.integer, float)):
                    channel_idx = list(self.get_channel_ids()).index(channel_ids[i])
                    self._key_properties['offset'][channel_idx] = float(offsets[i])
                else:
                    raise TypeError("'offset' must be an int or float")
        else:
            raise ValueError("channel_ids and offsets must have same length")

    def get_channel_offsets(self, channel_ids=None):
        """This function returns the offset of each channel specified by channel_ids.

        Parameters
        ----------
        channel_ids: array_like
            The channel ids (ints) for which the gains will be returned

        Returns
        -------
        offsets: array_like
            Returns a list of corresponding offsets for the given channel_ids
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        offsets = self._key_properties['offset']
        # Only None upon initialization
        if offsets is None:
            offsets = np.zeros(self.get_num_channels(), dtype='float')
            self._key_properties['offset'] = offsets
        offsets = np.array(offsets)
        channel_idxs = np.array([list(self.get_channel_ids()).index(ch) for ch in channel_ids])
        return offsets[channel_idxs]

    def clear_channel_offsets(self, channel_ids=None):
        """This function clears the gains of each channel specified by channel_ids.

        Parameters
        ----------
        channel_ids: array-like or int
            The channel ids (ints) for which the groups will be cleared. If None, all channel ids are assumed.
        """
        if channel_ids is None:
            channel_ids = list(self.get_channel_ids())
        if isinstance(channel_ids, (int, np.integer)):
            channel_ids = [channel_ids]
        # Reset to default offets (0)
        default_offsets = [0.] * len(channel_ids)
        self.set_channel_offsets(default_offsets, channel_ids)

    def set_channel_property(self, channel_id, property_name, value):
        """This function adds a property dataset to the given channel under the property name.

        Parameters
        ----------
        channel_id: int
            The channel id for which the property will be added
        property_name: str
            A property stored by the RecordingExtractor (location, etc.)
        value:
            The data associated with the given property name. Could be many
            formats as specified by the user
        """
        if isinstance(channel_id, (int, np.integer)):
            if channel_id in self.get_channel_ids():
                if isinstance(property_name, str):
                    if property_name == 'location':
                        self.set_channel_locations(value, channel_id)
                    elif property_name == 'group':
                        self.set_channel_groups(value, channel_id)
                    else:
                        if channel_id not in self._properties.keys():
                            self._properties[channel_id] = {}
                        self._properties[channel_id][property_name] = value
                else:
                    raise TypeError(str(property_name) + " must be a string")
            else:
                raise ValueError(str(channel_id) + " is not a valid channel_id")
        else:
            raise TypeError(str(channel_id) + " must be an int")

    def get_channel_property(self, channel_id, property_name):
        """This function returns the data stored under the property name from
        the given channel.

        Parameters
        ----------
        channel_id: int
            The channel id for which the property will be returned
        property_name: str
            A property stored by the RecordingExtractor (location, etc.)

        Returns
        -------
        property_data
            The data associated with the given property name. Could be many
            formats as specified by the user
        """
        if not isinstance(channel_id, (int, np.integer)):
            raise TypeError(str(channel_id) + " must be an int")
        if channel_id not in self.get_channel_ids():
            raise ValueError(str(channel_id) + " is not a valid channel_id")
        if property_name == 'location':
            return self.get_channel_locations(channel_id)[0]
        if property_name == 'group':
            return self.get_channel_groups(channel_id)[0]
        if property_name == 'gain':
            return self.get_channel_gains(channel_id)[0]
        if property_name == 'offset':
            return self.get_channel_offsets(channel_id)[0]
        if channel_id not in self._properties.keys():
            raise ValueError('no properties found for channel ' + str(channel_id))
        if property_name not in self._properties[channel_id]:
            raise RuntimeError(str(property_name) + " has not been added to channel " + str(channel_id))
        if not isinstance(property_name, str):
            raise TypeError(str(property_name) + " must be a string")
        return self._properties[channel_id][property_name]

    def get_channel_property_names(self, channel_id):
        """Get a list of property names for a given channel.

        Parameters
        ----------
        channel_id: int
            The channel id for which the property names will be returned
            If None (default), will return property names for all channels

        Returns
        -------
        property_names
            The list of property names
        """
        if isinstance(channel_id, (int, np.integer)):
            if channel_id in self.get_channel_ids():
                if channel_id not in self._properties.keys():
                    self._properties[channel_id] = {}
                property_names = list(self._properties[channel_id].keys())
                if np.all(np.logical_not(np.isnan(self.get_channel_locations(channel_id)))):
                    property_names.extend(['location'])
                property_names.extend(['group'])
                property_names.extend(['gain'])
                property_names.extend(['offset'])
                return sorted(property_names)
            else:
                raise ValueError(str(channel_id) + " is not a valid channel_id")
        else:
            raise TypeError(str(channel_id) + " must be an int")

    def get_shared_channel_property_names(self, channel_ids=None):
        """Get the intersection of channel property names for a given set of channels or for all channels
        if channel_ids is None.

        Parameters
        ----------
        channel_ids: array_like
            The channel ids for which the shared property names will be returned.
            If None (default), will return shared property names for all channels

        Returns
        -------
        property_names
            The list of shared property names
        """
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        curr_property_name_set = set(self.get_channel_property_names(channel_id=channel_ids[0]))
        for channel_id in channel_ids[1:]:
            curr_channel_property_name_set = set(self.get_channel_property_names(channel_id=channel_id))
            curr_property_name_set = curr_property_name_set.intersection(curr_channel_property_name_set)
        property_names = list(curr_property_name_set)
        return sorted(property_names)

    def copy_channel_properties(self, recording, channel_ids=None):
        """Copy channel properties from another recording extractor to the current
        recording extractor.

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor from which the properties will be copied
        channel_ids: (array_like, (int, np.integer))
            The list (or single value) of channel_ids for which the properties will be copied
        """
        if channel_ids is None:
            self._key_properties = deepcopy(recording._key_properties)
            self._properties = deepcopy(recording._properties)
        else:
            if isinstance(channel_ids, (int, np.integer)):
                channel_ids = [channel_ids]
            # copy key properties
            groups = recording.get_channel_groups(channel_ids=channel_ids)
            locations = recording.get_channel_locations(channel_ids=channel_ids)
            gains = recording.get_channel_gains(channel_ids=channel_ids)
            offsets = recording.get_channel_offsets(channel_ids=channel_ids)
            self.set_channel_groups(groups)
            self.set_channel_locations(locations)
            self.set_channel_gains(gains)
            self.set_channel_offsets(offsets)

            # copy normal properties
            for channel_id in channel_ids:
                curr_property_names = recording.get_channel_property_names(channel_id=channel_id)
                for curr_property_name in curr_property_names:
                    if curr_property_name not in self._key_properties.keys():  # key property
                        value = recording.get_channel_property(channel_id=channel_id, property_name=curr_property_name)
                        self.set_channel_property(channel_id=channel_id, property_name=curr_property_name, value=value)

    def clear_channel_property(self, channel_id, property_name):
        """This function clears the channel property for the given property.

        Parameters
        ----------
        channel_id: int
            The id that specifies a channel in the recording
        property_name: string
            The name of the property to be cleared
        """
        if property_name == 'location':
            self.clear_channel_locations(channel_id)
        elif property_name == 'group':
            self.clear_channel_groups(channel_id)
        elif channel_id in self._properties.keys():
            if property_name in self._properties[channel_id]:
                del self._properties[channel_id][property_name]

    def clear_channels_property(self, property_name, channel_ids=None):
        """This function clears the channels' properties for the given property.

        Parameters
        ----------
        property_name: string
            The name of the property to be cleared
        channel_ids: list
            A list of ids that specifies a set of channels in the recording. If None all channels are cleared
        """
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        for channel_id in channel_ids:
            self.clear_channel_property(channel_id, property_name)

    def get_epoch(self, epoch_name):
        """This function returns a SubRecordingExtractor which is a view to the
        given epoch

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be returned

        Returns
        -------
        epoch_extractor: SubRecordingExtractor
            A SubRecordingExtractor which is a view to the given epoch
        """
        from .subrecordingextractor import SubRecordingExtractor

        epoch_info = self.get_epoch_info(epoch_name)
        start_frame = epoch_info['start_frame']
        end_frame = epoch_info['end_frame']
        return SubRecordingExtractor(parent_recording=self, start_frame=start_frame,
                                     end_frame=end_frame)

    def load_probe_file(self, probe_file, channel_map=None, channel_groups=None, verbose=False):
        """This function returns a SubRecordingExtractor that contains information from the given
        probe file (channel locations, groups, etc.) If a .prb file is given, then 'location' and 'group'
        information for each channel is added to the SubRecordingExtractor. If a .csv file is given, then
        it will only add 'location' to the SubRecordingExtractor.

        Parameters
        ----------
        probe_file: str
            Path to probe file. Either .prb or .csv
        channel_map : array-like
            A list of channel IDs to set in the loaded file.
            Only used if the loaded file is a .csv.
        channel_groups : array-like
            A list of groups (ints) for the channel_ids to set in the loaded file.
            Only used if the loaded file is a .csv.
        verbose: bool
            If True, output is verbose

        Returns
        -------
        subrecording = SubRecordingExtractor
            The extractor containing all of the probe information.
        """
        subrecording = load_probe_file(self, probe_file, channel_map=channel_map,
                                       channel_groups=channel_groups, verbose=verbose)
        return subrecording

    def save_to_probe_file(self, probe_file, grouping_property=None, radius=None,
                           graph=True, geometry=True, verbose=False):
        """Saves probe file from the channel information of this recording extractor.

        Parameters
        ----------
        probe_file: str
            file name of .prb or .csv file to save probe information to
        grouping_property: str (default None)
            If grouping_property is a shared_channel_property, different groups are saved based on the property.
        radius: float (default None)
            Adjacency radius (used by some sorters). If None it is not saved to the probe file.
        graph: bool
            If True, the adjacency graph is saved (default=True)
        geometry: bool
            If True, the geometry is saved (default=True)
        verbose: bool
            If True, output is verbose
        """
        save_to_probe_file(self, probe_file, grouping_property=grouping_property, radius=radius,
                           graph=graph, geometry=geometry, verbose=verbose)

    def write_to_binary_dat_format(self, save_path, time_axis=0, dtype=None, chunk_size=None, chunk_mb=500,
                                   n_jobs=1, joblib_backend='loky', return_scaled=True, verbose=False):
        """Saves the traces of this recording extractor into binary .dat format.

        Parameters
        ----------
        save_path: str
            The path to the file.
        time_axis: 0 (default) or 1
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype: dtype
            Type of the saved data. Default float32
        chunk_size: None or int
            Size of each chunk in number of frames.
            If None (default) and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        n_jobs: int
            Number of jobs to use (Default 1)
        joblib_backend: str
            Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
        return_scaled: bool
            If True, traces are returned after scaling (using gain/offset). If False, the raw traces are returned
        verbose: bool
            If True, output is verbose (when chunks are used)
        """
        write_to_binary_dat_format(self, save_path=save_path, time_axis=time_axis, dtype=dtype, chunk_size=chunk_size,
                                   chunk_mb=chunk_mb, n_jobs=n_jobs, joblib_backend=joblib_backend,
                                   return_scaled=return_scaled, verbose=verbose)

    def write_to_h5_dataset_format(self, dataset_path, save_path=None, file_handle=None,
                                   time_axis=0, dtype=None, chunk_size=None, chunk_mb=500, verbose=False):
        """Saves the traces of a recording extractor in an h5 dataset.

        Parameters
        ----------
        dataset_path: str
            Path to dataset in h5 file (e.g. '/dataset')
        save_path: str
            The path to the file.
        file_handle: file handle
            The file handle to dump data. This can be used to append data to an header. In case file_handle is given,
            the file is NOT closed after writing the binary data.
        time_axis: 0 (default) or 1
            If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
            If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
        dtype: dtype
            Type of the saved data. Default float32.
        chunk_size: None or int
            Size of each chunk in number of frames.
            If None (default) and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
        chunk_mb: None or int
            Chunk size in Mb (default 500Mb)
        verbose: bool
            If True, output is verbose (when chunks are used)
        """
        write_to_h5_dataset_format(self, dataset_path, save_path, file_handle, time_axis, dtype, chunk_size, chunk_mb,
                                   verbose)

    def get_sub_extractors_by_property(self, property_name, return_property_list=False):
        """Returns a list of SubRecordingExtractors from this RecordingExtractor based on the given
        property_name (e.g. group)

        Parameters
        ----------
        property_name: str
            The property used to subdivide the extractor
        return_property_list: bool
            If True the property list is returned

        Returns
        -------
        sub_list: list
            The list of subextractors to be returned
        OR
        sub_list, prop_list
            If return_property_list is True, the property list will be returned as well

        """
        if return_property_list:
            sub_list, prop_list = get_sub_extractors_by_property(self, property_name=property_name,
                                                                 return_property_list=return_property_list)
            return sub_list, prop_list
        else:
            sub_list = get_sub_extractors_by_property(self, property_name=property_name,
                                                      return_property_list=return_property_list)
            return sub_list

    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        """
        Returns an array with frames of TTL signals. To be implemented in sub-classes

        Parameters
        ----------
        start_frame: int
            The starting frame of the ttl to be returned (inclusive)
        end_frame: int
            The ending frame of the ttl to be returned (exclusive)
        channel_id: int
            The TTL channel id

        Returns
        -------
        ttl_frames: array-like
            Frames of TTL signal for the specified channel
        ttl_state: array-like
            State of the transition: 1 - rising, -1 - falling
        """
        raise NotImplementedError

    @staticmethod
    def write_recording(recording, save_path):
        """This function writes out the recorded file of a given recording
        extractor to the file format of this current recording extractor. Allows
        for easy conversion between recording file formats. It is a static
        method so it can be used without instantiating this recording extractor.

        Parameters
        ----------
        recording: RecordingExtractor
            An RecordingExtractor that can extract information from the recording
            file to be converted to the new format.

        save_path: string
            A path to where the converted recorded data will be saved, which may
            either be a file or a folder, depending on the format.
        """
        raise NotImplementedError("The write_recording function is not \
                                  implemented for this extractor")
