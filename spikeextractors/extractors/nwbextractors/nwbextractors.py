import spikeextractors as se

import os
import numpy as np
from datetime import datetime

try:
    from pynwb import NWBHDF5IO
    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries
    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False

class CopyRecordingExtractor(se.RecordingExtractor):
    def __init__(self, other):
        se.RecordingExtractor.__init__(self)
        self._other = other
        self.copy_channel_properties(other)

    def get_channel_ids(self):
        return list(range(self._other.get_num_channels()))

    def get_num_frames(self):
        return self._other.get_num_frames()

    def get_sampling_frequency(self):
        return self._other.get_sampling_frequency()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        return self._other.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)


class NwbRecordingExtractor(CopyRecordingExtractor):
    extractor_name = 'NwbRecordingExtractor'
    has_default_locations = True
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    extractor_gui_params = [
        {'name': 'file_path', 'type': 'file', 'title': "Path to file (.h5 or .hdf5)"},
        {'name': 'acquisition_name', 'type': 'string', 'value': None, 'default': None, 'title': "Name of Acquisition Method"},
    ]
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path, acquisition_name=None):
        assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"
        self._path = file_path
        self._acquisition_name = acquisition_name
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            if acquisition_name is None:
                a_names = list(nwbfile.acquisition.keys())
                if len(a_names) > 1:
                    raise Exception('More than one acquisition found. You must specify acquisition_name.')
                if len(a_names) == 0:
                    raise Exception('No acquisitions found in the .nwb file.')
                acquisition_name = a_names[0]
            ts = nwbfile.acquisition[acquisition_name]
            self._nwb_timeseries = ts
            M = np.array(ts.data).shape[1]
            if M != len(ts.electrodes):
                raise Exception(
                    'Number of electrodes does not match the shape of the data {}<>{}'.format(M, len(ts.electrodes)))
            geom = np.zeros((M, 3))
            for m in range(M):
                geom[m, :] = [ts.electrodes[m][1], ts.electrodes[m][2], ts.electrodes[m][3]]
            if hasattr(ts, 'timestamps') and ts.timestamps:
                sampling_frequency = 1 / (ts.timestamps[1] - ts.timestamps[0])  # there's probably a better way
            else:
                sampling_frequency = ts.rate * 1000
            data = np.copy(np.transpose(ts.data))
            NRX = se.NumpyRecordingExtractor(timeseries=data, sampling_frequency=sampling_frequency, geom=geom)
            CopyRecordingExtractor.__init__(self, NRX)

    @staticmethod
    def write_recording(recording, save_path, acquisition_name='ElectricalSeries'):
        assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"
        M = recording.get_num_channels()

        nwbfile = NWBFile(
            session_description='',
            identifier='',
            session_start_time=datetime.now(),
        )
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

        if os.path.exists(save_path):
            os.remove(save_path)
        with NWBHDF5IO(save_path, 'w') as io:
            io.write(nwbfile)


class NwbSortingExtractor(se.SortingExtractor):
    extractor_name = 'NwbSortingExtractor'
    exporter_name = 'NwbSortingExporter'
    exporter_gui_params = [
        {'name': 'save_path', 'type': 'file', 'title': "Save path"},
        {'name': 'identifier', 'type': 'str', 'value': None, 'default': None, 'title': "The session identifier"},
        {'name': 'session_description', 'type': 'str', 'value': None, 'default': None, 'title': "The session description"},
    ]
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path):
        assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"
        se.SortingExtractor.__init__(self)

    @staticmethod
    def write_sorting(sorting, save_path, nwbfile_kwargs=None):
        """

        Parameters
        ----------
        sorting: SortingExtractor
        save_path: str
        nwbfile_kwargs: optional, dict with optional args of pynwb.NWBFile
        """
        assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"
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
            spkt = sorting.get_unit_spike_train(unit_id=id+1) / fs
            nwbfile.add_unit(id=id, spike_times=spkt)
            # 'waveform_mean' and 'waveform_sd' are interesting args to include later            

        io.write(nwbfile)
        io.close()
