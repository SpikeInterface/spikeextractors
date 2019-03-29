import spikeextractors as se

import os
import numpy as np
from datetime import datetime


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
    def __init__(self, path, acquisition_name=None):
        try:
            from pynwb import NWBHDF5IO
            from pynwb import NWBFile
            from pynwb.ecephys import ElectricalSeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        self._path = path
        self._acquisition_name = acquisition_name
        with NWBHDF5IO(path, 'r') as io:
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
                samplerate = 1 / (ts.timestamps[1] - ts.timestamps[0])  # there's probably a better way
            else:
                samplerate = ts.rate * 1000
            data = np.copy(np.transpose(ts.data))
            NRX = se.NumpyRecordingExtractor(timeseries=data, samplerate=samplerate, geom=geom)
            CopyRecordingExtractor.__init__(self, NRX)

    @staticmethod
    def write_recording(recording, save_path, acquisition_name):
        try:
            from pynwb import NWBHDF5IO
            from pynwb import NWBFile
            from pynwb.ecephys import ElectricalSeries
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To use the Nwb extractors, install pynwb: \n\n"
                                      "pip install pynwb\n\n")
        M = recording.get_num_channels()
        N = recording.get_num_frames()

        nwbfile = NWBFile(
            source='SpikeInterface::NwbRecordingExtractor',
            session_description='',
            identifier='',
            session_start_time=datetime.now(),
            experimenter='',
            lab='',
            institution='',
            experiment_description='',
            session_id=''
        )
        device = nwbfile.create_device(name='device_name', source="device_source")
        eg_name = 'electrode_group_name'
        eg_source = "electrode_group_source"
        eg_description = "electrode_group_description"
        eg_location = "electrode_group_location"

        electrode_group = nwbfile.create_electrode_group(
            name=eg_name,
            source=eg_source,
            location=eg_location,
            device=device,
            description=eg_description
        )

        for m in range(M):
            id = m
            location = recording.get_channel_property(m, 'location')
            impedence = -1.0
            while len(location) < 3:
                location = np.append(location, [0])
            nwbfile.add_electrode(
                id,
                x=float(location[0]), y=float(location[1]), z=float(location[2]),
                imp=impedence,
                location='electrode_location',
                filtering='none',
                group=electrode_group,
                description='electrode_description'
            )
        electrode_table_region = nwbfile.create_electrode_table_region(
            list(range(M)),
            'electrode_table_region'
        )

        rate = recording.get_sampling_frequency() / 1000
        ephys_data = recording.get_traces().T

        ephys_ts = ElectricalSeries(
            name=acquisition_name,
            source='acquisition_source',
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
