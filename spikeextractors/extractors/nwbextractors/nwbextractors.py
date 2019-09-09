import spikeextractors as se
import os
import numpy as np
from datetime import datetime


class NwbRecordingExtractor(se.RecordingExtractor):
    def __init__(self, path, electrical_series_name='ElectricalSeries'):
        """

        Parameters
        ----------
        path: path to NWB file
        electrical_series_name: str, optional
        """
        try:
            from pynwb import NWBHDF5IO
            from pynwb import NWBFile
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        self._path = path
        se.RecordingExtractor.__init__(self)
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if electrical_series_name is not None:
                self._electrical_series_name = electrical_series_name
            else:
                a_names = list(nwbfile.acquisition.keys())
                if len(a_names) > 1:
                    raise Exception('More than one acquisition found. You must specify electrical_series.')
                if len(a_names) == 0:
                    raise Exception('No acquisitions found in the .nwb file.')
                self._electrical_series_name = a_names[0]
            es = nwbfile.acquisition[self._electrical_series_name]
            if hasattr(es, 'timestamps') and es.timestamps:
                self.sampling_frequency = 1 / np.median(np.diff(es.timestamps))
                self.recording_start_time = es.timestamps[0]
            else:
                self.sampling_frequency = es.rate
                if hasattr(es, 'starting_time'):
                    self.recording_start_time = es.starting_time
                else:
                    self.recording_start_time = 0.

            self.num_frames = len(es.data)
            if len(es.data.shape) == 1:
                self.num_channels = 1
            else:
                self.num_channels = es.data.shape[1]

            self.channel_ids = es.electrodes.table.id[:]

            self.geom = np.column_stack([es.electrodes.table[d][:] for d in ('x', 'y', 'z')])

            self.channel_groups = es.electrodes.table['group_name'][:]

            self.electrodes_df = es.electrodes.table.to_dataframe()

            if nwbfile.epochs is not None:
                df_epochs = nwbfile.epochs.to_dataframe()
                self._epochs = {row['label']:
                                {'start_frame': self.time_to_frame(row['start_time']),
                                 'end_frame': self.time_to_frame(row['stop_time'])}
                                for _, row in df_epochs.iterrows()}

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        import pynwb
        with pynwb.NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            es = nwbfile.acquisition[self._electrical_series_name]

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = -1
            if channel_ids is None:
                return es.data[start_frame:end_frame].T
            else:
                return es.data[start_frame:end_frame, np.isin(self.channel_ids, channel_ids)].T

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_num_frames(self):
        return self.num_frames

    def get_channel_ids(self):
        return self.channel_ids

    def get_channel_locations(self, channel_ids=None):
        if channel_ids is None:
            return self.geom
        else:
            return self.geom[np.isin(self.channel_ids, channel_ids), :]

    def get_channel_groups(self, channel_ids=None):
        if channel_ids is None:
            return self.channel_groups
        else:
            return self.channel_groups[np.isin(self.channel_ids, channel_ids)]

    def get_channel_property_names(self, channel_id=None):
        return list(self.electrodes_df.columns)

    def get_channel_property(self, channel_id, property_name):
        return self.electrodes_df[property_name][channel_id]

    def time_to_frame(self, time):
        return (time - self.recording_start_time) * self.get_sampling_frequency()

    def frame_to_time(self, frame):
        return frame / self.get_sampling_frequency() + self.recording_start_time

    @staticmethod
    def write_recording(recording, save_path, nwbfile_kwargs=None):
        """

        Parameters
        ----------
        recording: RecordingExtractor
        save_path: str
        nwbfile_kwargs: optional, dict with optional args of pynwb.NWBFile
        """
        try:
            import pynwb
            from pynwb import NWBHDF5IO
            from pynwb import NWBFile
            from pynwb.ecephys import ElectricalSeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        M = recording.get_num_channels()

        if os.path.exists(save_path):
            io = NWBHDF5IO(save_path, 'r+')
            nwbfile = io.read()
        else:
            io = NWBHDF5IO(save_path, mode='w')
            input_nwbfile_kwargs = {
                'session_start_time': datetime.now(),
                'identifier': '',
                'session_description': ''}
            if nwbfile_kwargs is not None:
                input_nwbfile_kwargs.update(nwbfile_kwargs)
            nwbfile = NWBFile(**input_nwbfile_kwargs)

        # Tests if Device already exists
        aux = [isinstance(i, pynwb.device.Device) for i in nwbfile.children]
        if any(aux):
            device = nwbfile.children[np.where(aux)[0][0]]
        else:
            device = nwbfile.create_device(name='Device')

        # Tests if ElectrodeGroup already exists
        aux = [isinstance(i, pynwb.ecephys.ElectrodeGroup) for i in nwbfile.children]
        if any(aux):
            electrode_group = nwbfile.children[np.where(aux)[0][0]]
        else:
            eg_name = 'electrode_group_name'
            eg_description = "electrode_group_description"
            eg_location = "electrode_group_location"
            electrode_group = nwbfile.create_electrode_group(
                name=eg_name,
                location=eg_location,
                device=device,
                description=eg_description
            )

            # add electrodes with locations
            for m in range(M):
                location = recording.get_channel_property(m, 'location')
                impedence = -1.0
                while len(location) < 3:
                    location = np.append(location, [0])
                nwbfile.add_electrode(
                    id=m,
                    x=float(location[0]), y=float(location[1]), z=float(location[2]),
                    imp=impedence,
                    location='electrode_location',
                    filtering='none',
                    group=electrode_group,
                )

            # add other existing electrode properties
            properties = recording.get_shared_channel_property_names()
            properties.remove('location')
            for pr in properties:
                pr_data = [recording.get_channel_property(ind, pr) for ind in range(M)]
                nwbfile.add_electrode_column(
                    name=pr,
                    description='',
                    data=pr_data,
                )

            electrode_table_region = nwbfile.create_electrode_table_region(
                list(range(M)),
                'electrode_table_region'
            )

        # Tests if Acquisition already exists
        aux = [isinstance(i, pynwb.ecephys.ElectricalSeries) for i in nwbfile.children]
        if any(aux):
            acquisition = nwbfile.children[np.where(aux)[0][0]]
        else:
            rate = recording.get_sampling_frequency()
            if 'gain' in recording.get_shared_channel_property_names():
                gains = np.array(recording.get_channel_gains())
            else:
                gains = np.ones(M)
            ephys_data = recording.get_traces().T
            ephys_data_V = 1e-6*gains*ephys_data
            acquisition_name = 'ElectricalSeries'
            ephys_ts = ElectricalSeries(
                name=acquisition_name,
                data=ephys_data_V,
                electrodes=electrode_table_region,
                starting_time=recording.frame_to_time(0),
                rate=rate,
                conversion=1.,
                comments='Generated from SpikeInterface::NwbRecordingExtractor',
                description='acquisition_description'
            )
            nwbfile.add_acquisition(ephys_ts)

        io.write(nwbfile)
        io.close()



class NwbSortingExtractor(se.SortingExtractor):
    def __init__(self, path, electrical_series=None):
        """

        Parameters
        ----------
        path: path to NWB file
        electrical_series: pynwb.ecephys.ElectricalSeries object
        """
        try:
            from pynwb import NWBHDF5IO
            from pynwb import NWBFile
            from pynwb.ecephys import ElectricalSeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        se.SortingExtractor.__init__(self)
        self._path = path
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            # defines the electrical series from where the sorting came from
            # important to know the associated fs and t0
            if electrical_series is None:
                a_names = list(nwbfile.acquisition.keys())
                if len(a_names) > 1:
                    raise Exception('More than one acquisition found. You must specify electrical_series.')
                if len(a_names) == 0:
                    raise Exception('No acquisitions found in the .nwb file.')
                acquisition_name = a_names[0]
                es = nwbfile.acquisition[acquisition_name]
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
        '''NWB file sampling rate can't be modified.'''
        print(self.set_sampling_frequency.__doc__)

    def get_unit_ids(self):
        '''This function returns a list of ids (ints) for each unit in the sorsted result.

        Returns
        ----------
        unit_ids: array_like
            A list of the unit ids in the sorted result (ints).
        '''
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            unit_ids = list(nwbfile.units.id[:])
        return unit_ids

    def get_unit_property_names(self, unit_id):
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
        property_names = self.get_shared_unit_property_names(unit_ids=None)
        return property_names

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
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            property_names = list(nwbfile.units.colnames)
        return property_names

    def get_unit_property(self, unit_id, property_name):
        '''This function returns the data stored under the property name given
        from the given unit.

        Parameters
        ----------
        unit_id: int
            The unit id for which the property will be returned
        property_name: str
            The name of the property
        Returns
        ----------
        value
            The data associated with the given property name. Could be many
            formats as specified by the user.
        '''
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")

        if not isinstance(unit_id, (int, np.integer)):
            raise ValueError("'unit_id' must be an integer")
        existing_ids = self.get_unit_ids()
        if not unit_id in existing_ids:
            raise ValueError("'unit_id' outside the range of existing ids")
        if not isinstance(property_name, str):
            raise Exception("'property_name' must be a string")

        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if property_name in list(nwbfile.units.colnames):
                val = nwbfile.units[property_name][existing_ids.index(unit_id)]
            else:
                raise Exception(property_name+" is not a valid property in dataset")
        return val


    def get_units_property(self, unit_ids, property_name):
        '''Returns a list of values stored under the property name corresponding
        to a list of units

        Parameters
        ----------
        unit_ids: list
            The unit ids for which the property will be returned
            Defaults to all ids
        property_name: str
            The name of the property
        Returns
        ----------
        values
            The list of values
        '''
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        if not isinstance(property_name, str):
            raise Exception("'property_name' must be a string")
        existing_ids = self.get_unit_ids()
        if unit_ids is None:
            unit_ids = existing_ids
        else:
            if not isinstance(unit_ids, list):
                raise ValueError("'unit_ids' must be a list of integers")
            if not all(isinstance(x, int) for x in unit_ids):
                raise ValueError("'unit_ids' must be a list of integers")
            if not all(x in existing_ids for x in unit_ids):
                raise ValueError("'unit_ids' contains values outside the range of existing ids")

        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if property_name in list(nwbfile.units.colnames):
                values = [nwbfile.units[property_name][existing_ids.index(id)] for id in unit_ids]
            else:
                raise Exception(property_name+" is not a valid property in dataset")
        return values


    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        '''This function extracts spike frames from the specified unit.
        It will return spike frames from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike frames are returned in the form of an
        array_like of spike frames. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording.
        start_frame: int
            The frame above which a spike frame is returned  (inclusive).
        end_frame: int
            The frame below which a spike frame is returned  (exclusive).
        Returns
        ----------
        spike_train: numpy.ndarray
            An 1D array containing all the frames for each spike in the
            specified unit given the range of start and end frames.
        '''
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = np.Inf
            # chosen unit and interval
            times0 = nwbfile.units['spike_times'][int(unit_id-1)][:]
            # spike times are measured in samples
            times = ((times0 - self._t0) * self._sampling_frequency).astype('int')
        return times[(times>start_frame) & (times<end_frame)]


    def set_unit_property(self, unit_id, property_name, value):
        """
        NWB files require that new properties are set once for all units.
        Please use method 'set_units_property()' instead.
        """
        print(self.set_unit_property.__doc__)

    def set_units_property(self, unit_ids, property_name, values, default_values=np.nan):
        '''This function adds a new property data set to the chosen units.
        NWB files require that new properties are set once for all units. Therefore,
        the 'property_name' for ids not present in 'unit_ids' will be filled with
        'default_values'.

        Parameters
        ----------
        unit_ids: list of ints
            The unit ids for which the property will be set.
        property_name: str
            The name of the property to be stored.
        values :
            The data associated with the given property name. Could be many
            formats as specified by the user.
        default_values :
            Default values of 'property_name' for unit ids not present in
            'unit_ids' list.
        '''
        if not isinstance(unit_ids, list):
            raise ValueError("'unit_ids' must be a list of integers")
        if not all(isinstance(x, int) for x in unit_ids):
            raise ValueError("'unit_ids' must be a list of integers")
        existing_ids = self.get_unit_ids()
        if not all(x in existing_ids for x in unit_ids):
            raise ValueError("'unit_ids' contains values outside the range of existing ids")
        if not isinstance(property_name, str):
            raise Exception("'property_name' must be a string")
        if property_name in self.get_shared_unit_property_names():
            raise Exception(property_name + " property already exists")
        if len(unit_ids)!=len(values):
            raise Exception("'unit_ids' and 'values' should be lists of same size")
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")

        nUnits = len(existing_ids)
        new_values = [default_values]*nUnits
        with NWBHDF5IO(self._path, 'r+') as io:
            nwbfile = io.read()
            for i, v in zip(unit_ids, values):
                new_values[existing_ids.index(i)] = v
            nwbfile.add_unit_column(name=property_name,
                                    description='',
                                    data=new_values)
            io.write(nwbfile)


    def copy_unit_properties(self, sorting, unit_ids=None, default_values=None):
        '''Copy unit properties from another sorting extractor to the current
        sorting extractor. NWB files require that new properties are set once
        for all units. Therefore, the 'property_name' for ids not present in
        'unit_ids' will be filled with 'default_values'.

        Parameters
        ----------
        sorting: SortingExtractor
            The sorting extractor from which the properties will be copied
        unit_ids: list
            The list of unit_ids for which the properties will be copied.
        default_values : list
            List of default values for each property, for unit ids not present in
            'unit_ids' list. Default to NaN for all properties.
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

        new_property_names = sorting.get_shared_unit_property_names()
        curr_property_names = self.get_shared_unit_property_names()
        if default_values is None:
            default_values = [np.nan]*len(new_property_names)
        else:
            if len(default_values)!=len(new_property_names):
                raise Exception("'default_values' list must have length equal to"+
                                " number of properties to be copied.")

        # Copies only properties that do not exist already in NWB file
        for i, pr in enumerate(new_property_names):
            if pr in curr_property_names:
                print(pr+" already exists in NWB file and can't be copied.")
            else:
                pr_values = sorting.get_units_property(unit_ids=unit_ids,
                                                       property_name=pr)
                self.set_units_property(unit_ids=unit_ids,
                                        property_name=pr,
                                        values=pr_values,
                                        default_values=default_values[i])


    def clear_unit_property(self, unit_id=None, property_name=None):
        '''NWB files do not allow for deleting properties.'''
        print(self.clear_unit_property.__doc__)

    def clear_units_property(self, unit_ids=None, property_name=None):
        '''NWB files do not allow for deleting properties.'''
        print(self.clear_units_property.__doc__)


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
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        if not isinstance(epoch_name, str):
            raise Exception("'epoch_name' must be a string")
        if not isinstance(start_frame, int):
            raise Exception("'start_frame' must be an integer")
        if not isinstance(end_frame, int):
            raise Exception("'end_frame' must be an integer")

        fs = self.get_sampling_frequency()
        with NWBHDF5IO(self._path, 'r+') as io:
            nwbfile = io.read()
            nwbfile.add_epoch(start_time=start_frame/fs,
                              stop_time=end_frame/fs,
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
            List of epoch names in the recording extractor
        '''
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if nwbfile.epochs is None:
                print("No epochs in NWB file")
                epoch_names = None
            else:
                flat_list = [item for sublist in nwbfile.epochs['tags'][:] for item in sublist]
                aux = np.array(flat_list)
                epoch_names = np.unique(aux).tolist()
        return epoch_names


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
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        if not isinstance(epoch_name, str):
            raise ValueError("epoch_name must be a string")
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
                    epoch_info['start_frame'] = int(nwbfile.epochs['start_time'][i]*fs)
                    epoch_info['end_frame'] = int(nwbfile.epochs['stop_time'][i]*fs)
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
        epoch_extractor: SubRecordingExtractor
            A SubRecordingExtractor which is a view to the given epoch
        '''
        epoch_info = self.get_epoch_info(epoch_name)
        start_frame = epoch_info['start_frame']
        end_frame = epoch_info['end_frame']
        from spikeextractors.subsortingextractor import SubSortingExtractor
        return SubSortingExtractor(parent_sorting=self, start_frame=start_frame,
                                   end_frame=end_frame)


    @staticmethod
    def write_sorting(sorting, save_path, nwbfile_kwargs=None):
        """

        Parameters
        ----------
        sorting: SortingExtractor
        save_path: str
        nwbfile_kwargs: optional, dict with optional args of pynwb.NWBFile
        """
        try:
            import pynwb
            from pynwb import NWBHDF5IO
            from pynwb import NWBFile
            from pynwb.ecephys import ElectricalSeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")

        ids = sorting.get_unit_ids()
        fs = sorting.get_sampling_frequency()

        if os.path.exists(save_path):
            io = NWBHDF5IO(save_path, 'r+')
            nwbfile = io.read()
        else:
            io = NWBHDF5IO(save_path, mode='w')
            input_nwbfile_kwargs = {
                'session_start_time': datetime.now(),
                'identifier': '',
                'session_description': ''}
            if nwbfile_kwargs is not None:
                input_nwbfile_kwargs.update(nwbfile_kwargs)
            nwbfile = NWBFile(**input_nwbfile_kwargs)

        # Tests if Units already exists
        aux = [isinstance(i, pynwb.misc.Units) for i in nwbfile.children]
        if any(aux):
            units = nwbfile.children[np.where(aux)[0][0]]
        else:
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
        io.close()


def most_relevant_ch(traces):
    """
    Calculates the most relevant channel for an Unit.
    Estimates the channel where the max-min difference of the average traces is greatest.

    traces : ndarray
        ndarray of shape (nSpikes, nChannels, nSamples)
    """
    nSpikes = traces.shape[0]
    nChannels = traces.shape[1]
    nSamples = traces.shape[2]

    #average and std of spike traces per channel
    avg = np.mean(traces, axis=0)
    std = np.std(traces, axis=0)

    max_min = np.zeros(nChannels)
    for ch in range(nChannels):
        max_min[ch] = avg[ch,:].max() - avg[ch,:].min()

    relevant_ch = np.argmax(max_min)
    return relevant_ch
