from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor
from pathlib import Path
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args, check_get_unit_spike_train, check_get_ttl_args

"""
The NumpyExtractors can be constructed and used to encapsulate custom file formats and data structures which
contain information about recordings or sorting results. NumpyExtractors are instantiated in-memory and function
like any other Recording/SortingExtractor.
"""

class NumpyRecordingExtractor(RecordingExtractor):
    extractor_name = 'NumpyRecording'
    is_writable = True
    has_default_locations = False
    has_unscaled = False

    def __init__(self, timeseries, sampling_frequency, geom=None):
        RecordingExtractor.__init__(self)
        if isinstance(timeseries, str):
            if Path(timeseries).is_file():
                assert Path(timeseries).suffix == '.npy', "'timeseries' file is not a numpy file (.npy)"
                self.is_dumpable = True
                self._timeseries = np.load(timeseries)
                self._kwargs = {'timeseries': str(Path(timeseries).absolute()),
                                'sampling_frequency': sampling_frequency, 'geom': geom}
            else:
                raise ValueError("'timeeseries' is does not exist")
        elif isinstance(timeseries, np.ndarray):
            self.is_dumpable = False
            self._timeseries = timeseries
            self._kwargs = {'timeseries': timeseries,
                            'sampling_frequency': sampling_frequency, 'geom': geom}
        else:
            raise TypeError("'timeseries' can be a str or a numpy array")
        self._sampling_frequency = float(sampling_frequency)
        self._geom = geom
        if geom is not None:
            self.set_channel_locations(self._geom)

        self._ttl_frames = None
        self._ttl_states = None

    def set_ttls(self, ttl_frames, ttl_states=None):
        self._ttl_frames = ttl_frames.astype('int64')
        if ttl_states is not None:
            self._ttl_states = ttl_states.astype('int64')
        else:
            self._ttl_states = np.ones_like(ttl_frames, dtype='int64')

    def get_channel_ids(self):
        return list(range(self._timeseries.shape[0]))

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        recordings = self._timeseries[:, start_frame:end_frame][channel_ids, :]
        return recordings

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        if self._ttl_frames is not None and self._ttl_states is not None:
            ttl_idxs = np.where((self._ttl_frames >= start_frame) & (self._ttl_frames < end_frame))[0]
            return self._ttl_frames[ttl_idxs], self._ttl_states[ttl_idxs]
        else:
            print("TTL frames have not been added to the extractor. You can add them with the `set_ttls()1 function")
            return None, None

    @staticmethod
    def write_recording(recording, save_path):
        save_path = Path(save_path)
        np.save(save_path, recording.get_traces())


class NumpySortingExtractor(SortingExtractor):
    extractor_name = 'NumpySorting'
    is_writable = False

    def __init__(self):
        SortingExtractor.__init__(self)
        self._units = {}
        self.is_dumpable = False

    def load_from_extractor(self, sorting, copy_unit_properties=False, copy_unit_spike_features=False):
        """This function loads the information from a SortingExtractor into this extractor.

        Parameters
        ----------
        sorting: SortingExtractor
            The SortingExtractor from which this extractor will copy information.
        copy_unit_properties: bool
            If True, the unit_properties will be copied from the given SortingExtractor to this extractor.
        copy_unit_spike_features: bool
            If True, the unit_spike_features will be copied from the given SortingExtractor to this extractor.
        """
        ids = sorting.get_unit_ids()
        for id in ids:
            self.add_unit(id, sorting.get_unit_spike_train(id))
        if sorting.get_sampling_frequency() is not None:
            self.set_sampling_frequency(sorting.get_sampling_frequency())
        if copy_unit_properties:
            self.copy_unit_properties(sorting)
        if copy_unit_spike_features:
            self.copy_unit_spike_features(sorting)

    def set_sampling_frequency(self, sampling_frequency):
        self._sampling_frequency = sampling_frequency

    def set_times_labels(self, times, labels):
        """This function takes in an array of spike times (in frames) and an array of spike labels and adds all the
        unit information in these lists into the extractor.

        Parameters
        ----------
        times: np.array
            An array of spike times (in frames).
        labels: np.array
            An array of spike labels corresponding to the given times.
        """
        units = np.sort(np.unique(labels))
        for unit in units:
            times0 = times[np.where(labels == unit)[0]]
            self.add_unit(unit_id=int(unit), times=times0)

    def add_unit(self, unit_id, times):
        """This function adds a new unit with the given spike times.

        Parameters
        ----------
        unit_id: int
            The unit_id of the unit to be added.
        times: np.array
            An array of spike times (in frames).
        """
        self._units[unit_id] = dict(times=times)

    def get_unit_ids(self):
        return list(self._units.keys())

    @check_get_unit_spike_train
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        times = self._units[unit_id]['times']
        inds = np.where((start_frame <= times) & (times < end_frame))[0]
        return np.rint(times[inds]).astype(int)
