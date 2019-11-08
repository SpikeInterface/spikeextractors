import os
import uuid
from datetime import datetime

import numpy as np
from hdmf.data_utils import DataChunkIterator

import spikeextractors as se

try:
    from pynwb import NWBHDF5IO
    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.ecephys import ElectrodeGroup
    from pynwb.device import Device
    from pynwb.misc import Units

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False


def check_nwb_install():
    assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"


def set_dynamic_table_property(dynamic_table, row_ids, property_name, values, default_value=np.nan,
                               description='no description'):
    check_nwb_install()
    if not isinstance(row_ids, list) or not all(isinstance(x, int) for x in row_ids):
        raise TypeError("'ids' must be an integer or a list of integers")
    ids = list(dynamic_table.id[:])
    if any([i not in ids for i in row_ids]):
        raise ValueError("'ids' contains values outside the range of existing ids")
    if not isinstance(property_name, str):
        raise TypeError("'property_name' must be a string")
    if len(row_ids) != len(values):
        raise ValueError("'ids' and 'values' should be lists of same size")

    if property_name in dynamic_table:
        for (row_id, value) in zip(row_ids, values):
            dynamic_table[property_name].data[ids.index(row_id)] = value
    else:
        col_data = [default_value] * len(ids)  # init with default val
        for (row_id, value) in zip(row_ids, values):
            col_data[ids.index(row_id)] = value

        dynamic_table.add_column(name=property_name, description=description, data=col_data)


def get_dynamic_table_property(dynamic_table, *, row_ids=None, property_name):
    all_row_ids = list(dynamic_table.id[:])
    if row_ids is None:
        row_ids = all_row_ids
    return [dynamic_table[property_name][all_row_ids.index(x)] for x in row_ids]


class NwbRecordingExtractor(se.RecordingExtractor):
    extractor_name = 'NwbRecordingExtractor'
    has_default_locations = True
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    extractor_gui_params = [
        {'name': 'file_path', 'type': 'file', 'title': "Path to file (.h5 or .hdf5)"},
        {'name': 'acquisition_name', 'type': 'string', 'value': None, 'default': None,
         'title': "Name of Acquisition Method"},
    ]
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, path, electrical_series_name='ElectricalSeries'):
        """

        Parameters
        ----------
        path: path to NWB file
        electrical_series_name: str, optional
        """
        check_nwb_install()
        se.RecordingExtractor.__init__(self)
        self._path = path
        with NWBHDF5IO(self._path, 'a') as io:
            nwbfile = io.read()
            if electrical_series_name is not None:
                self._electrical_series_name = electrical_series_name
            else:
                a_names = list(nwbfile.acquisition.keys())
                if len(a_names) > 1:
                    raise ValueError('More than one acquisition found. You must specify electrical_series.')
                if len(a_names) == 0:
                    raise ValueError('No acquisitions found in the .nwb file.')
                self._electrical_series_name = a_names[0]
            es = nwbfile.acquisition[self._electrical_series_name]
            if hasattr(es, 'timestamps') and es.timestamps:
                self.sampling_frequency = 1. / np.median(np.diff(es.timestamps))
                self.recording_start_time = es.timestamps[0]
            else:
                self.sampling_frequency = es.rate
                if hasattr(es, 'starting_time'):
                    self.recording_start_time = es.starting_time
                else:
                    self.recording_start_time = 0.

            self.num_frames = int(es.data.shape[0])

            # Fill channel properties dictionary from electrodes table
            self.channel_ids = es.electrodes.table.id[:]
            self._channel_properties = {}
            for i in self.channel_ids:
                self._channel_properties[i] = {}
                for col in nwbfile.electrodes.colnames:
                    if isinstance(nwbfile.electrodes[col][i], ElectrodeGroup):
                        pass
                    elif col == 'group_name':
                        self._channel_properties[i]['group'] = nwbfile.electrodes[col][i]
                    else:
                        self._channel_properties[i][col] = nwbfile.electrodes[col][i]

            # Fill epochs dictionary
            self._epochs = {}
            if nwbfile.epochs is not None:
                df_epochs = nwbfile.epochs.to_dataframe()
                self._epochs = {row['label']: {
                    'start_frame': self.time_to_frame(row['start_time']),
                    'end_frame': self.time_to_frame(row['stop_time'])}
                    for _, row in df_epochs.iterrows()}

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        check_nwb_install()
        if channel_ids is not None:
            if not isinstance(channel_ids, (list, np.ndarray)):
                raise TypeError("'channel_ids' must be a list or array of integers.")
            if not all([id in self.channel_ids for id in channel_ids]):
                raise ValueError("'channel_ids' contain values outside the range of valid ids.")
        else:
            channel_ids = self.channel_ids
        if start_frame is not None:
            if not isinstance(start_frame, (int, np.integer)):
                raise TypeError("'start_frame' must be an integer")
        else:
            start_frame = 0
        if end_frame is not None:
            if not isinstance(end_frame, (int, np.integer)):
                raise TypeError("'end_frame' must be an integer")
        else:
            end_frame = None #-1

        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            es = nwbfile.acquisition[self._electrical_series_name]
            if np.array(channel_ids).size > 1 and np.any(np.diff(channel_ids) < 0):
                sorted_idx = np.argsort(channel_ids)
                recordings = es.data[start_frame:end_frame, np.sort(channel_ids)].T
                traces = recordings[sorted_idx, :]
            else:
                traces = es.data[start_frame:end_frame, channel_ids].T
            # This DatasetView and lazy operations will only work within context
            # We're keeping the non-lazy version for now
            # es_view = DatasetView(es.data)  # es is an instantiated h5py dataset
            # traces = es_view.lazy_slice[start_frame:end_frame, channel_ids].lazy_transpose()
        return traces

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_num_frames(self):
        return self.num_frames

    def get_channel_ids(self):
        return self.channel_ids.tolist()

    # def set_channel_locations(self, channel_ids=None, locations=None):
    #     with NWBHDF5IO(self._path, 'r+') as io:
    #         nwbfile = io.read()
    #         set_dynamic_table_property(nwbfile.electrodes, channel_ids, 'x', locations[:, 0])
    #         set_dynamic_table_property(nwbfile.electrodes, channel_ids, 'y', locations[:, 1])
    #         set_dynamic_table_property(nwbfile.electrodes, channel_ids, 'z', locations[:, 2])
    #
    # def set_channel_groups(self, channel_ids=None, groups=None):
    #     # todo
    #     raise NotImplementedError()
    #
    # def get_channel_groups(self, channel_ids=None):
    #     with NWBHDF5IO(self._path, 'r') as io:
    #         nwbfile = io.read()
    #         return get_dynamic_table_property(nwbfile.electrodes, row_ids=channel_ids, property_name='group_name')
    #
    # def set_channel_gains(self, channel_ids, gains):
    #     self.set_channels_property(channel_ids=channel_ids,
    #                                property_name='gain',
    #                                values=gains,
    #                                default_value=1.)
    #
    # def set_channel_property(self, channel_id, property_name=None, value=None, default_value=np.nan,
    #                          description='no description'):
    #     self.set_channels_property(channel_ids=[channel_id], property_name=property_name, values=[value],
    #                                default_value=default_value, description=description)
    #
    # def set_channels_property(self, channel_ids, property_name, values, default_value=np.nan,
    #                           description='no description'):
    #     with NWBHDF5IO(self._path, 'r+') as io:
    #         nwbfile = io.read()
    #         set_dynamic_table_property(nwbfile.electrodes, row_ids=channel_ids, property_name=property_name,
    #                                    values=values, default_value=default_value, description=description)
    #         es = nwbfile.acquisition[self._electrical_series_name]
    #         self.electrodes_df = es.electrodes.table.to_dataframe()
    #         io.write(nwbfile)
    #
    # def get_channel_property(self, channel_id, property_name):
    #     return self.electrodes_df[property_name][channel_id]
    #
    # def get_channel_property_names(self, channel_id=None):
    #     return list(self.electrodes_df.columns)
    #
    # def get_shared_channel_property_names(self, channel_ids=None):
    #     return list(self.electrodes_df.columns)
    #
    # def copy_channel_properties(self, recording, channel_ids=None, default_values=None):
    #     if channel_ids is None:
    #         channel_ids = recording.get_channel_ids()
    #     else:
    #         if not isinstance(channel_ids, list) or not all(isinstance(x, int) for x in channel_ids):
    #             raise ValueError("'channel_ids' must be a list of integers")
    #         existing_ids = self.get_channel_ids()
    #         if any(x not in existing_ids for x in channel_ids):
    #             raise ValueError("'channel_ids' contains values outside the range of existing ids")
    #
    #     new_property_names = recording.get_shared_channel_property_names()
    #     curr_property_names = self.get_shared_channel_property_names()
    #     if default_values is None:
    #         default_values = [np.nan] * len(new_property_names)
    #     else:
    #         if len(default_values) != len(new_property_names):
    #             raise ValueError("'default_values' list must have length equal to" +
    #                             " number of properties to be copied.")
    #
    #     # Copies only properties that do not exist already in NWB file
    #     for i, pr in enumerate(new_property_names):
    #         if pr in curr_property_names:
    #             # todo
    #             raise NotImplementedError()
    #         else:
    #             for channel_id in channel_ids:
    #                 pr_value = recording.get_channel_property(channel_id=channel_id,
    #                                                           property_name=pr)
    #                 self.set_channel_property(channel_id=channel_id,
    #                                           property_name=pr,
    #                                           value=pr_value)
    #
    # def add_epoch(self, epoch_name, start_frame, end_frame):
    #     check_nwb_install()
    #     if not isinstance(epoch_name, str):
    #         raise TypeError("'epoch_name' must be a string")
    #     if not isinstance(start_frame, int):
    #         raise TypeError("'start_frame' must be an integer")
    #     if not isinstance(end_frame, int):
    #         raise TypeError("'end_frame' must be an integer")
    #
    #     with NWBHDF5IO(self._path, 'r+') as io:
    #         nwbfile = io.read()
    #         nwbfile.add_epoch(start_time=self.frame_to_time(start_frame),
    #                           stop_time=self.frame_to_time(end_frame),
    #                           tags=epoch_name)
    #         io.write(nwbfile)
    #
    # def remove_epoch(self, epoch_name=None):
    #     # todo
    #     raise NotImplementedError()
    #
    # def get_epoch_names(self):
    #     check_nwb_install()
    #     with NWBHDF5IO(self._path, 'r') as io:
    #         nwbfile = io.read()
    #         if nwbfile.epochs is None:
    #             return
    #         flat_list = [item for sublist in nwbfile.epochs['tags'][:] for item in sublist]
    #         aux = np.array(flat_list)
    #         epoch_names = np.unique(aux).tolist()
    #     return epoch_names
    #
    # def get_epoch_info(self, epoch_name):
    #     check_nwb_install()
    #     if not isinstance(epoch_name, str):
    #         raise TypeError("epoch_name must be a string")
    #     all_epoch_names = self.get_epoch_names()
    #     if epoch_name not in all_epoch_names:
    #         raise ValueError("This epoch has not been added")
    #
    #     epoch_info = {}
    #     with NWBHDF5IO(self._path, 'r') as io:
    #         nwbfile = io.read()
    #         flat_list = [item for sublist in nwbfile.epochs['tags'][:] for item in sublist]
    #         for i, tag in enumerate(flat_list):
    #             if tag == epoch_name:
    #                 epoch_info['start_frame'] = self.time_to_frame(nwbfile.epochs['start_time'][i])
    #                 epoch_info['end_frame'] = self.time_to_frame(nwbfile.epochs['stop_time'][i])
    #     return epoch_info

    @staticmethod
    def write_recording(recording, save_path, acquisition_name='ElectricalSeries', **nwbfile_kwargs):
        '''

        Parameters
        ----------
        recording: RecordingExtractor
        save_path: str
        acquisition_name: str (default 'ElectricalSeries')
        nwbfile_kwargs: optional, pynwb.NWBFile args
        '''
        check_nwb_install()
        n_channels = recording.get_num_channels()
        channel_ids = recording.get_channel_ids()

        if os.path.exists(save_path):
            read_mode = 'r+'
        else:
            read_mode = 'w'

        with NWBHDF5IO(save_path, mode=read_mode) as io:
            if read_mode == 'r+':
                nwbfile = io.read()
            else:
                kwargs = {'session_description': 'No description',
                          'identifier': str(uuid.uuid4()),
                          'session_start_time': datetime.now()}
                kwargs.update(**nwbfile_kwargs)
                nwbfile = NWBFile(**kwargs)

            # Tests if Device already exists
            aux = [isinstance(i, Device) for i in nwbfile.children]
            if any(aux):
                device = nwbfile.children[np.where(aux)[0][0]]
            else:
                device = nwbfile.create_device(name='Device')

            # Tests if ElectrodeGroup already exists
            aux = [isinstance(i, ElectrodeGroup) for i in nwbfile.children]
            if any(aux):
                electrode_group = nwbfile.children[np.where(aux)[0][0]]
                electrode_table = nwbfile.electrodes
            else:
                electrode_group = nwbfile.create_electrode_group(
                    name='electrode_group_name',
                    location="electrode_group_location",
                    device=device,
                    description="electrode_group_description"
                )

                # add electrodes with locations
                for m in range(n_channels):
                    location = recording.get_channel_property(m, 'location')
                    impedence = -1.0
                    while len(location) < 3:
                        location = np.append(location, [0])
                    nwbfile.add_electrode(
                        id=m,
                        x=float(location[0]), y=float(location[1]), z=float(location[2]),
                        imp=impedence,
                        location='unknown',
                        filtering='none',
                        group=electrode_group,
                    )
                electrode_table = nwbfile.electrodes

            # add/update electrode properties
            nwb_electrode_properties = electrode_table.colnames
            rx_channel_properties = recording.get_shared_channel_property_names()
            for pr in rx_channel_properties:
                pr_data = [recording.get_channel_property(ind, pr) for ind in channel_ids]
                # If new data column
                if pr not in nwb_electrode_properties:
                    nwbfile.add_electrode_column(
                        name=pr,
                        description='no description',
                        data=pr_data,
                    )
                # property 'group' of RX channels correspond to property 'group_name' of NWB electrodes
                elif pr == 'group':
                    nwbfile.electrodes['group_name'].data[:] = pr_data
                # If updated another existing property
                else:
                    nwbfile.electrodes[pr].data[:] = pr_data

            # Tests if ElectricalSeries already exists in acquisition
            aux = [isinstance(i, ElectricalSeries) for i in nwbfile.acquisition.values()]
            if not any(aux):
                electrode_table_region = nwbfile.create_electrode_table_region(
                    list(range(n_channels)),
                    'electrode_table_region'
                )

                rate = recording.get_sampling_frequency()
                if 'gain' in recording.get_shared_channel_property_names():
                    gains = np.array(recording.get_channel_gains())
                else:
                    gains = np.ones(n_channels)

                def data_generator(recording, num_channels):
                    #  generates data chunks for iterator
                    for id in range(0, num_channels):
                        data = recording.get_traces(channel_ids=[id]).flatten()
                        yield data

                data = data_generator(recording=recording, num_channels=n_channels)
                ephys_data = DataChunkIterator(data=data, iter_axis=1)
                acquisition_name = 'ElectricalSeries'

                # If traces are stored as 'int16', then to get Volts = data*channel_conversion*conversion
                ephys_ts = ElectricalSeries(
                    name=acquisition_name,
                    data=ephys_data,
                    electrodes=electrode_table_region,
                    starting_time=recording.frame_to_time(0),
                    rate=rate,
                    conversion=1e-6,
                    channel_conversion=gains,
                    comments='Generated from SpikeInterface::NwbRecordingExtractor',
                    description='acquisition_description'
                )
                nwbfile.add_acquisition(ephys_ts)

            io.write(nwbfile)


class NwbSortingExtractor(se.SortingExtractor):
    extractor_name = 'NwbSortingExtractor'
    exporter_name = 'NwbSortingExporter'
    exporter_gui_params = [
        {'name': 'save_path', 'type': 'file', 'title': "Save path"},
        {'name': 'identifier', 'type': 'str', 'value': None, 'default': None, 'title': "The session identifier"},
        {'name': 'session_description', 'type': 'str', 'value': None, 'default': None,
         'title': "The session description"},
    ]
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, path, electrical_series=None):
        """

        Parameters
        ----------
        path: path to NWB file
        electrical_series: pynwb.ecephys.ElectricalSeries object
        """
        check_nwb_install()
        se.SortingExtractor.__init__(self)
        self._path = path
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            # defines the electrical series from where the sorting came from
            # important to know the associated fs and t0
            if electrical_series is None:
                if len(nwbfile.acquisition) > 1:
                    raise Exception('More than one acquisition found. You must specify electrical_series.')
                if len(nwbfile.acquisition) == 0:
                    raise Exception('No acquisitions found in the .nwb file.')
                es = list(nwbfile.acquisition.values())[0]
            else:
                es = electrical_series

            # get rate
            if es.rate is not None:
                self._sampling_frequency = es.rate
            else:
                self._sampling_frequency = 1 / (es.timestamps[1] - es.timestamps[0])
            # get t0
            if hasattr(es, 'starting_time'):
                self._t0 = es.starting_time
            elif es.timestamps is not None:
                self._t0 = es.timestamps[0]
            else:
                self._t0 = 0.

    def set_sampling_frequency(self, sampling_frequency):
        raise ValueError("NWB file sampling rate can't be modified.")

    def get_unit_ids(self):
        '''This function returns a list of ids (ints) for each unit in the sorted result.

        Returns
        ----------
        unit_ids: array_like
            A list of the unit ids in the sorted result (ints).
        '''
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            unit_ids = list(nwbfile.units.id[:])
        return unit_ids

    def get_unit_property_names(self, unit_id=None):
        '''Get a list of property names for a given unit.

         Parameters
        ----------
        unit_id: int
            The unit id for which the property names will be returned.

        Returns
        ----------
        property_names
            The list of property names
        '''
        # units cannot have unique property keys in NWB
        return self.get_shared_unit_property_names(unit_ids=None)

    def get_shared_unit_property_names(self, unit_ids=None):
        '''Get the intersection of unit property names for a given set of units
        or for all units if unit_ids is None. In NWB files all units must have
        the same properties, so the argument 'unit_ids' is irrelevant and should
        be left as None.

         Parameters
        ----------
        unit_ids: array_like
            The unit ids for which the shared property names will be returned.
            If None (default), will return shared property names for all units,
        Returns
        ----------
        property_names
            The list of shared property names
        '''
        # units cannot have unique property keys in NWB
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            return [name for name in nwbfile.units.colnames if not name == 'spike_times']

    def get_unit_property(self, unit_id, property_name):
        return self.get_units_property(unit_ids=[unit_id], property_name=property_name)

    def get_units_property(self, *, unit_ids=None, property_name):
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            return get_dynamic_table_property(nwbfile.units, row_ids=unit_ids, property_name=property_name)

    def time_to_frame(self, time):
        return np.round((time - self._t0) * self.get_sampling_frequency()).astype('int')

    def get_unit_spike_train(self, unit_id, start_frame=0, end_frame=np.Inf):
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            # chosen unit and interval
            times = nwbfile.units['spike_times'][list(nwbfile.units.id[:]).index(unit_id)][:]
            # spike times are measured in samples
            frames = self.time_to_frame(times)
        return frames[(frames > start_frame) & (frames < end_frame)]

    def set_unit_property(self, unit_id, property_name, value, default_value=np.nan, description='no description'):
        if not isinstance(unit_id, int):
            raise ValueError('unit_id must be an int')
        self.set_units_property(
            unit_ids=[unit_id],
            property_name=property_name,
            values=[value],
            default_value=default_value,
            description=description
        )

    def set_units_property(self, *, unit_ids=None, property_name, values, default_value=np.nan,
                           description='no description'):
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        with NWBHDF5IO(self._path, 'r+') as io:
            nwbfile = io.read()
            set_dynamic_table_property(
                dynamic_table=nwbfile.units,
                row_ids=unit_ids,
                property_name=property_name,
                values=values,
                default_value=default_value,
                description=description
            )
            io.write(nwbfile)

    def copy_unit_properties(self, sorting, unit_ids=None, default_value=np.nan):
        property_names = sorting.get_shared_unit_property_names()

        # Copies only properties that do not exist already in NWB file
        for pr in property_names:
            pr_values = sorting.get_units_property(unit_ids=unit_ids, property_name=pr)
            self.set_units_property(unit_ids=unit_ids,
                                    property_name=pr,
                                    values=pr_values,
                                    default_value=default_value)

    def clear_unit_property(self, unit_id=None, property_name=None):
        '''NWB files do not allow for deleting properties.'''
        print(self.clear_unit_property.__doc__)

    def clear_units_property(self, unit_ids=None, property_name=None):
        '''NWB files do not allow for deleting properties.'''
        print(self.clear_units_property.__doc__)

    def get_nspikes(self):
        """Returns list with the number of spikes for each unit."""
        # todo: there is a way to do this without reading all of the data if you use spike_times_index
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r+') as io:
            nwbfile = io.read()
            nSpikes = [len(spkt) for spkt in nwbfile.units['spike_times'][:]]
        return nSpikes

    def get_unit_spike_features(self, unit_id, feature_name, start_frame=0, end_frame=np.Inf):
        '''This function extracts the specified spike features from the specified unit.
        It will return spike features from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike features are returned in the form of an
        array_like of spike features. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording.
        feature_name: string
            The name of the feature to be returned.
        start_frame: int
            The frame above which a spike frame is returned  (inclusive).
        end_frame: int
            The frame below which a spike frame is returned  (exclusive).
        Returns
        ----------
        spike_features: numpy.ndarray
            An array containing all the features for each spike in the
            specified unit given the range of start and end frames.
        '''
        check_nwb_install()
        if not isinstance(feature_name, str):
            raise TypeError("'feature_name' must be a string")
        full_feat_name = 'spike_feature_' + feature_name
        if full_feat_name not in self.get_shared_unit_spike_feature_names():
            raise ValueError(full_feat_name + " not present in NWB file")

        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            units = nwbfile.units
            # chosen unit and interval
            feat_vals = np.array(units[full_feat_name][list(units.id[:]).index(unit_id)][:])
            times = units['spike_times'][list(units.id[:]).index(unit_id)][:]
            # spike times are measured in samples
            frames = self.time_to_frame(times)
            mask = (frames >= start_frame) & (frames < end_frame)
        return feat_vals[mask]

    def set_unit_spike_features(self, unit_ids, feature_name, values,
                                default_value=np.nan):
        '''This function adds a unit features data set under the given features
        name to the given unit.

        Parameters
        ----------
        unit_id: int
            The unit id for which the features will be set
        feature_name: str
            The name of the feature to be stored
        value :
            The data associated with the given feature name. Could be many
            formats as specified by the user.
        default_value :
            Default value of property to be set.

        '''
        check_nwb_install()
        if not isinstance(unit_ids, list):
            raise TypeError("'unit_ids' must be a list of integers")
        existing_ids = self.get_unit_ids()
        if not isinstance(feature_name, str):
            raise TypeError("'feature_name' must be a string")
        if 'spike_feature_' + feature_name in self.get_shared_unit_spike_feature_names():
            raise ValueError('spike_feature_' + feature_name + " feature already exists")

        if default_value is None:
            default_value = np.nan
        nspikes_units = self.get_nspikes()
        new_values = [[default_value] * nSpikes for nSpikes in nspikes_units]
        with NWBHDF5IO(self._path, 'a') as io:
            nwbfile = io.read()
            for id in unit_ids:
                spikes_unit = nwbfile.units['spike_times'][existing_ids.index(id)]
                if len(spikes_unit) != len(values[str(id)]):
                    io.close()
                    raise ValueError("feature values should have the same length" +
                                     " as the spike train, error at unit #" + str(id))
                new_values[existing_ids.index(id)] = values[str(id)]

            flatten_new_values = [item for sublist in new_values for item in sublist]
            spikes_index = np.cumsum(nspikes_units)
            nwbfile.add_unit_column(name='spike_feature_' + feature_name,
                                    description='no description',
                                    data=flatten_new_values,
                                    index=spikes_index)
            io.write(nwbfile)

    def get_shared_unit_spike_feature_names(self, unit_ids=None):
        '''Get list of spike feature names for the units in the NWB file.
        Since in a NWB file all units must contain the same feature columns,
        'unit_ids' can be left in its default value of None.

         Parameters
        ----------
        unit_ids: array_like
            The unit ids for which the shared feature names will be returned.
            If None (default), will return shared feature names for all units,
        Returns
        ----------
        feature_names
            The list of shared feature names
        '''
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r+') as io:
            nwbfile = io.read()
            return [feat for feat in nwbfile.units.colnames if feat.startswith('spike_feature_')]

    def get_unit_spike_feature_names(self, unit_id=None):
        '''This function returns the list of feature names for the given unit.
        Since in a NWB file all units must contain the same feature columns,
        this method equals to 'get_shared_unit_spike_feature_names()'.

        Parameters
        ----------
        unit_id: int
            The unit id for which the feature names will be returned.
        Returns
        ----------
        feature_names
            The list of feature names.
        '''
        return self.get_shared_unit_spike_feature_names(unit_ids=None)

    def copy_unit_spike_features(self, sorting, unit_ids=None, default_values=None):
        '''Copy unit spike features from another sorting extractor to the current
        NWB file. NWB files require that new properties are set once
        for all units. Therefore, the spike feature values for ids not present
        in 'unit_ids' will be filled with 'default_values'.

        Parameters
        ----------
        sorting: SortingExtractor
            The sorting extractor from which the spike features will be copied
        unit_ids: list
            The list of unit_ids for which the spike features will be copied.
        default_values : list
            List of default values for each spike feature, for unit ids not
            present in 'unit_ids' list. Default to NaN for all properties.
        '''
        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        else:
            if not isinstance(unit_ids, list):
                raise ValueError("'unit_ids' must be a list of integers")
            if not all(isinstance(x, int) for x in unit_ids):
                raise ValueError("'unit_ids' must be a list of integers")
            existing_ids = self.get_unit_ids()
            if not all(x in existing_ids for x in unit_ids):
                raise ValueError("'unit_ids' contains values outside the range of existing ids")

        new_feature_names = sorting.get_shared_unit_spike_feature_names()
        curr_feature_names = self.get_shared_unit_spike_feature_names()
        if default_values is None:
            default_values = [np.nan] * len(new_feature_names)
        else:
            if len(default_values) != len(new_feature_names):
                raise ValueError("'default_values' list must have length equal to" +
                                " number of features to be copied.")
        # Copies only features that do not exist in NWB file
        nspikes_units = self.get_nspikes()
        for i, feat in enumerate(new_feature_names):
            full_feat_name = 'spike_feature_' + feat
            if full_feat_name in curr_feature_names:
                print(full_feat_name + " already exists in NWB file and can't be copied.")
            else:
                feat_values = {}
                for id in unit_ids:
                    vals = sorting.get_unit_spike_features(unit_id=id,
                                                           feature_name=feat)
                    feat_values[str(id)] = vals.tolist()
                self.set_unit_spike_features(unit_id=unit_ids,
                                             feature_name=feat,
                                             values=feat_values,
                                             default_value=default_values[i])

    def clear_unit_spike_features(self, unit_id=None, feature_name=None):
        '''NWB files do not allow removing features.'''
        print(self.clear_unit_spike_features.__doc__)

    def clear_units_spike_features(self, *, unit_ids=None, feature_name):
        '''NWB files do not allow removing features.'''
        print(self.clear_units_spike_features.__doc__)

    def frame_to_time(self, frame):
        return frame / self.get_sampling_frequency() + self._t0

    def add_epoch(self, epoch_name, start_frame, end_frame):
        '''This function adds an epoch to the NWB file.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be added
        start_frame: int
            The start frame of the epoch to be added (inclusive)
        end_frame: int
            The end frame of the epoch to be added (exclusive)
        '''
        check_nwb_install()
        if not isinstance(epoch_name, str):
            raise TypeError("'epoch_name' must be a string")
        if not isinstance(start_frame, int):
            raise TypeError("'start_frame' must be an integer")
        if not isinstance(end_frame, int):
            raise TypeError("'end_frame' must be an integer")

        with NWBHDF5IO(self._path, 'r+') as io:
            nwbfile = io.read()
            nwbfile.add_epoch(start_time=self.frame_to_time(start_frame),
                              stop_time=self.frame_to_time(end_frame),
                              tags=epoch_name)
            io.write(nwbfile)

    def remove_epoch(self, epoch_name=None):
        '''NWB files do not allow removing epochs.'''
        print(self.remove_epoch.__doc__)

    def get_epoch_names(self):
        '''This function returns a list of all the epoch names in the NWB file.

        Returns
        ----------
        epoch_names: list
            List of epoch names in the sorting extractor
        '''
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if nwbfile.epochs is None:
                print("No epochs in NWB file")
                return
            return [x[0] for x in nwbfile.epochs['tags'][:]]

    def get_epoch_info(self, epoch_name):
        '''This function returns the start frame and end frame of the epoch
        in a dict.

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be returned

        Returns
        ----------
        epoch_info: dict
            A dict containing the start frame and end frame of the epoch
        '''
        check_nwb_install()
        if not isinstance(epoch_name, str):
            raise TypeError("epoch_name must be a string")
        all_epoch_names = self.get_epoch_names()
        if epoch_name not in all_epoch_names:
            raise ValueError("This epoch has not been added")

        fs = self.get_sampling_frequency()
        epoch_info = {}
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            flat_list = [item for sublist in nwbfile.epochs['tags'][:] for item in sublist]
            for i, tag in enumerate(flat_list):
                if tag == epoch_name:
                    epoch_info['start_frame'] = self.time_to_frame(nwbfile.epochs['start_time'][i])
                    epoch_info['end_frame'] = self.time_to_frame(nwbfile.epochs['stop_time'][i])
        return epoch_info

    def get_epoch(self, epoch_name):
        '''This function returns a SubSortingExtractor which is a view to the
        given epoch

        Parameters
        ----------
        epoch_name: str
            The name of the epoch to be returned

        Returns
        ----------
        epoch_extractor: SubSortingExtractor
            A SubSortingExtractor which is a view to the given epoch
        '''
        epoch_info = self.get_epoch_info(epoch_name)
        start_frame = epoch_info['start_frame']
        end_frame = epoch_info['end_frame']
        from spikeextractors.subsortingextractor import SubSortingExtractor
        return SubSortingExtractor(parent_sorting=self, start_frame=start_frame,
                                   end_frame=end_frame)

    @staticmethod
    def write_sorting(sorting, save_path, **nwbfile_kwargs):
        """

        Parameters
        ----------
        sorting: SortingExtractor
        save_path: str
        nwbfile_kwargs: optional, pynwb.NWBFile args
        """
        check_nwb_install()

        ids = sorting.get_unit_ids()
        fs = sorting.get_sampling_frequency()

        if os.path.exists(save_path):
            read_mode = 'r+'
        else:
            read_mode = 'w'

        with NWBHDF5IO(save_path, mode=read_mode) as io:
            if read_mode == 'r+':
                io = NWBHDF5IO(save_path, 'r+')
                nwbfile = io.read()
            else:
                kwargs = {'session_description': 'No description',
                          'identifier': str(uuid.uuid4()),
                          'session_start_time': datetime.now()}
                kwargs.update(**nwbfile_kwargs)
                nwbfile = NWBFile(**kwargs)

            # Stores spike times for each detected cell (unit)
            for id in ids:
                spkt = sorting.get_unit_spike_train(unit_id=id) / fs
                if 'waveforms' in sorting.get_unit_spike_feature_names(unit_id=id):
                    # Stores average and std of spike traces
                    wf = sorting.get_unit_spike_features(unit_id=id,
                                                         feature_name='waveforms')
                    relevant_ch = most_relevant_ch(wf)
                    # Spike traces on the most relevant channel
                    traces = wf[:, relevant_ch, :]
                    traces_avg = np.mean(traces, axis=0)
                    traces_std = np.std(traces, axis=0)
                    nwbfile.add_unit(id=id,
                                     spike_times=spkt,
                                     waveform_mean=traces_avg,
                                     waveform_sd=traces_std)
                else:
                    nwbfile.add_unit(id=id, spike_times=spkt)

            io.write(nwbfile)


def most_relevant_ch(traces):
    """
    Calculates the most relevant channel for an Unit.
    Estimates the channel where the max-min difference of the average traces is greatest.

    traces : ndarray
        ndarray of shape (nSpikes, nChannels, nSamples)
    """
    n_channels = traces.shape[1]
    avg = np.mean(traces, axis=0)

    max_min = np.zeros(n_channels)
    for ch in range(n_channels):
        max_min[ch] = avg[ch, :].max() - avg[ch, :].min()

    relevant_ch = np.argmax(max_min)
    return relevant_ch
