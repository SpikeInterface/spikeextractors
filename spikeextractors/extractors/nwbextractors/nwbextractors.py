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
            else:
                self.sampling_frequency = es.rate

            self.num_frames = len(es.data)
            if len(es.data.shape) == 1:
                self.num_channels = 1
            else:
                self.num_channels = es.data.shape[1]

            self.channel_ids = es.electrodes.table.id[:]

            self.geom = np.column_stack([es.electrodes.table[d][:] for d in ('x', 'y', 'z')])

            self.channel_groups = es.electrodes.table['group_name'][:]

            self.electrodes_df = es.electrodes.table.to_dataframe()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):

        with NWBHDF5IO(self._path, 'r') as io:
            self._nwbfile = io.read()
            es = self._nwbfile.acquisition[self._electrical_series_name]

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

    def get_num_channels(self):
        return self.num_channels

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
            return self.channel_groups[np.isin(self.channel_ids, channel_ids), :]

    def get_channel_property_names(self, channel_id=None):
        return list(self.electrodes_df.columns)

    def get_channel_property(self, channel_id, property_name):
        return self.electrodes_df[property_name][channel_id]


    @staticmethod
    def write_recording(recording, save_path, acquisition_name='ElectricalSeries', nwbfile_kwargs=None):
        try:
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

        device = nwbfile.create_device(name='device_name')
        eg_name = 'electrode_group_name'
        eg_description = "electrode_group_description"
        eg_location = "electrode_group_location"

        electrode_group = nwbfile.create_electrode_group(
            name=eg_name,
            location=eg_location,
            device=device,
            description=eg_description
        )

        #add electrodes with locations
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

        #add other existing electrode properties
        properties = recording.get_channel_property_names()
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

        rate = recording.get_sampling_frequency()
        ephys_data = recording.get_traces().T

        ephys_ts = ElectricalSeries(
            name=acquisition_name,
            data=ephys_data,
            electrodes=electrode_table_region,
            starting_time=recording.frame_to_time(0),
            rate=rate,
            resolution=1e-6,
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
        io = NWBHDF5IO(self._path, 'r')
        nwbfile = io.read()
        #defines the electrical series from where the sorting came from
        #important to know the associated fs and t0
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

        #get rate
        if es.rate is not None:
            self._sampling_frequency = es.rate
        else:
            self._sampling_frequency = 1 / (es.timestamps[1] - es.timestamps[0])
        #get t0
        if hasattr(es, 'starting_time'):
            self._t0 = es.starting_time
        elif es.timestamps is not None:
            self._t0 = es.timestamps[0]
        else:
            self._t0 = 0.
        io.close()

    def get_unit_ids(self):
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        io = NWBHDF5IO(self._path, 'r')
        nwbfile = io.read()
        unit_ids = list(nwbfile.units.id[:])
        io.close()
        return unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        try:
            from pynwb import NWBHDF5IO
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        io = NWBHDF5IO(self._path, 'r')
        nwbfile = io.read()
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        #chosen unit and interval
        times0 = nwbfile.units['spike_times'][int(unit_id-1)][start_frame:end_frame]
        #spike times are measured in samples
        times = (( times0 - self._t0) * self._sampling_frequency).astype('int')
        io.close()
        return times


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

        # Stores spike times for each detected cell (unit)
        for id in ids:
            spkt = sorting.get_unit_spike_train(unit_id=id) / fs
            nwbfile.add_unit(id=id, spike_times=spkt)
            # 'waveform_mean' and 'waveform_sd' are interesting args to include later

        io.write(nwbfile)
        io.close()
