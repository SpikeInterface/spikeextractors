import os
import uuid
from datetime import datetime
from collections import abc
from pathlib import Path
import numpy as np
import distutils.version
from typing import Union
import warnings

import spikeextractors as se
from spikeextractors.extraction_tools import check_get_traces_args, check_valid_unit_id

try:
    import pynwb
    import pandas as pd
    from pynwb import NWBHDF5IO
    from pynwb import NWBFile
    from pynwb.ecephys import ElectricalSeries
    from pynwb.ecephys import ElectrodeGroup
    from hdmf.data_utils import DataChunkIterator
    from hdmf.backends.hdf5.h5_utils import H5DataIO

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False

PathType = Union[str, Path, None]


def check_nwb_install():
    assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"


def set_dynamic_table_property(dynamic_table, row_ids, property_name, values, index=False,
                               default_value=np.nan, table=False, description='no description'):
    check_nwb_install()
    if not isinstance(row_ids, list) or not all(isinstance(x, int) for x in row_ids):
        raise TypeError("'ids' must be a list of integers")
    ids = list(dynamic_table.id[:])
    if any([i not in ids for i in row_ids]):
        raise ValueError("'ids' contains values outside the range of existing ids")
    if not isinstance(property_name, str):
        raise TypeError("'property_name' must be a string")
    if len(row_ids) != len(values) and index is False:
        raise ValueError("'ids' and 'values' should be lists of same size")

    if index is False:
        if property_name in dynamic_table:
            for (row_id, value) in zip(row_ids, values):
                dynamic_table[property_name].data[ids.index(row_id)] = value
        else:
            col_data = [default_value] * len(ids)  # init with default val
            for (row_id, value) in zip(row_ids, values):
                col_data[ids.index(row_id)] = value
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=col_data,
                index=index,
                table=table
            )
    else:
        if property_name in dynamic_table:
            # TODO
            raise NotImplementedError
        else:
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=values,
                index=index,
                table=table
            )


def get_dynamic_table_property(dynamic_table, *, row_ids=None, property_name):
    all_row_ids = list(dynamic_table.id[:])
    if row_ids is None:
        row_ids = all_row_ids
    return [dynamic_table[property_name][all_row_ids.index(x)] for x in row_ids]


def get_nspikes(units_table, unit_id):
    """Return the number of spikes for chosen unit."""
    check_nwb_install()
    ids = np.array(units_table.id[:])
    indexes = np.where(ids == unit_id)[0]
    if not len(indexes):
        raise ValueError(f"{unit_id} is an invalid unit_id. Valid ids: {ids}.")
    index = indexes[0]
    if index == 0:
        return units_table['spike_times_index'].data[index]
    else:
        return units_table['spike_times_index'].data[index] - units_table['spike_times_index'].data[index - 1]


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


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default


def fill_kwargs_from_defaults(defaults: dict, values: dict = None):
    kwargs = {}
    if values is None:
        for default_property, default_value in defaults.items():
            kwargs.update({default_property: default_value})
    else:
        for default_property, default_value in defaults.items():
            kwargs.update({default_property: values.get(default_property, default_value)})
    return kwargs


class NwbRecordingExtractor(se.RecordingExtractor):
    extractor_name = 'NwbRecording'
    has_default_locations = True
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path, electrical_series_name='ElectricalSeries'):
        """
        Parameters
        ----------
        file_path: path to NWB file
        electrical_series_name: str, optional
        """
        assert HAVE_NWB, self.installation_mesg
        se.RecordingExtractor.__init__(self)
        self._path = str(file_path)
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if electrical_series_name is not None:
                self._electrical_series_name = electrical_series_name
            else:
                a_names = list(nwbfile.acquisition)
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
            num_channels = len(es.electrodes.table.id[:])

            # Channels gains - for RecordingExtractor, these are values to cast traces to uV
            if es.channel_conversion is not None:
                gains = es.conversion * es.channel_conversion[:] * 1e6
            else:
                gains = es.conversion * np.ones(num_channels) * 1e6

            # Extractors channel groups must be integers, but Nwb electrodes group_name can be strings
            if 'group_name' in nwbfile.electrodes.colnames:
                unique_grp_names = list(np.unique(nwbfile.electrodes['group_name'][:]))

            # Fill channel properties dictionary from electrodes table
            self.channel_ids = es.electrodes.table.id[es.electrodes.data]
            for es_ind, (channel_id, electrode_table_index) in enumerate(zip(self.channel_ids, es.electrodes.data)):
                self.set_channel_property(channel_id, 'gain', gains[es_ind])
                this_loc = []
                if 'rel_x' in nwbfile.electrodes:
                    this_loc.append(nwbfile.electrodes['rel_x'][electrode_table_index])
                    if 'rel_y' in nwbfile.electrodes:
                        this_loc.append(nwbfile.electrodes['rel_y'][electrode_table_index])
                    else:
                        this_loc.append(0)
                    self.set_channel_locations(this_loc, channel_id)

                for col in nwbfile.electrodes.colnames:
                    if isinstance(nwbfile.electrodes[col][electrode_table_index], ElectrodeGroup):
                        continue
                    elif col == 'group_name':
                        self.set_channel_groups(
                            int(unique_grp_names.index(nwbfile.electrodes[col][electrode_table_index])), channel_id)
                    elif col == 'location':
                        self.set_channel_property(channel_id, 'brain_area',
                                                  nwbfile.electrodes[col][electrode_table_index])
                    elif col in ['x', 'y', 'z', 'rel_x', 'rel_y']:
                        continue
                    else:
                        self.set_channel_property(channel_id, col, nwbfile.electrodes[col][electrode_table_index])

            # Fill epochs dictionary
            self._epochs = {}
            if nwbfile.epochs is not None:
                df_epochs = nwbfile.epochs.to_dataframe()
                self._epochs = {row['tags'][0]: {
                    'start_frame': self.time_to_frame(row['start_time']),
                    'end_frame': self.time_to_frame(row['stop_time'])}
                    for _, row in df_epochs.iterrows()}

            self._kwargs = {'file_path': str(Path(file_path).absolute()),
                            'electrical_series_name': electrical_series_name}
            self.make_nwb_metadata(nwbfile=nwbfile, es=es)

    def make_nwb_metadata(self, nwbfile, es):
        # Metadata dictionary - useful for constructing a nwb file
        self.nwb_metadata = dict()
        self.nwb_metadata['NWBFile'] = {
            'session_description': nwbfile.session_description,
            'identifier': nwbfile.identifier,
            'session_start_time': nwbfile.session_start_time,
            'institution': nwbfile.institution,
            'lab': nwbfile.lab
        }
        self.nwb_metadata['Ecephys'] = dict()
        # Update metadata with Device info
        self.nwb_metadata['Ecephys']['Device'] = []
        for dev in nwbfile.devices:
            self.nwb_metadata['Ecephys']['Device'].append({'name': dev})
        # Update metadata with ElectrodeGroup info
        self.nwb_metadata['Ecephys']['ElectrodeGroup'] = []
        for k, v in nwbfile.electrode_groups.items():
            self.nwb_metadata['Ecephys']['ElectrodeGroup'].append({
                'name': v.name,
                'description': v.description,
                'location': v.location,
                'device': v.device.name
            })
        # Update metadata with ElectricalSeries info
        self.nwb_metadata['Ecephys']['ElectricalSeries'] = []
        self.nwb_metadata['Ecephys']['ElectricalSeries'].append({
            'name': es.name,
            'description': es.description
        })

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            es = nwbfile.acquisition[self._electrical_series_name]
            es_channel_ids = np.array(es.electrodes.table.id[:])[es.electrodes.data[:]].tolist()
            channel_inds = [es_channel_ids.index(id) for id in channel_ids]
            if np.array(channel_ids).size > 1 and np.any(np.diff(channel_ids) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_idx = np.argsort(channel_inds)
                recordings = es.data[start_frame:end_frame, np.sort(channel_inds)].T
                traces = recordings[sorted_idx, :]
            else:
                traces = es.data[start_frame:end_frame, channel_inds].T
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

    @staticmethod
    def add_devices(recording: se.RecordingExtractor, nwbfile=None, metadata: dict = None):
        """
        Auxiliary static method for nwbextractor.

        Adds device information to nwbfile object.
        Will always ensure nwbfile has at least one device, but multiple
        devices within the metadata list will also be created.

        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['Device'] = [{'name': my_name,
                                                  'description': my_description}, ...]

        Missing keys in an element of metadata['Ecephys']['Device'] will be auto-populated with defaults.
        """
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        defaults = dict(
            name="Device",
            description="no description"
        )
        if metadata is None:
            metadata = dict(
                Ecephys=dict(
                    Device=[defaults]
                )
            )
        assert all([isinstance(x, dict) for x in metadata['Ecephys']['Device']]), \
            "Expected metadata['Ecephys']['Device'] to be a list of dictionaries!"

        for dev in metadata['Ecephys']['Device']:
            if dev.get('name', defaults['name']) not in nwbfile.devices:
                nwbfile.create_device(**dict(defaults, **dev))

    @staticmethod
    def add_electrode_groups(recording: se.RecordingExtractor, nwbfile=None, metadata: dict = None):
        """
        Auxiliary static method for nwbextractor.

        Adds electrode group information to nwbfile object.
        Will always ensure nwbfile has at least one electrode group.
        Will auto-generate a linked device if the specified name does not exist in the nwbfile.

        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['ElectrodeGroup'] = [{'name': my_name,
                                                          'description': my_description,
                                                          'location': electrode_location,
                                                          'device_name': my_device_name}, ...]

        Missing keys in an element of metadata['Ecephys']['ElectrodeGroup'] will be auto-populated with defaults.

        Group names set by RecordingExtractor channel properties will also be included with passed metadata,
        but will only use default description and location.
        """
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        if len(nwbfile.devices) == 0:
            se.NwbRecordingExtractor.add_devices(recording, nwbfile)
        defaults = dict(
            name="Electrode Group",
            description="no description",
            location="unknown",
            device_name="Device"
        )
        if metadata is None:
            metadata = dict(
                Ecephys=dict(
                    ElectrodeGroup=[]
                )
            )
        assert all([isinstance(x, dict) for x in metadata['Ecephys']['ElectrodeGroup']]), \
            "Expected metadata['Ecephys']['ElectrodeGroup'] to be a list of dictionaries!"

        for grp in metadata['Ecephys']['ElectrodeGroup']:
            device_name = grp.get('device_name', defaults['device_name'])
            if grp.get('name', defaults['name']) not in nwbfile.electrode_groups:
                if device_name not in nwbfile.devices:
                    new_device = dict(
                        Ecephys=dict(
                            Device=dict(
                                name=device_name
                            )
                        )
                     )
                    se.NwbRecordingExtractor.add_devices(recording, nwbfile, metadata=new_device)
                    warnings.warn(f"Device \'{device_name}\' not detected in "
                                  "attempted link to electrode group! Automatically generating.")
                electrode_group_kwargs = dict(defaults, **grp)
                electrode_group_kwargs.pop('device_name')
                electrode_group_kwargs.update(device=nwbfile.devices[device_name])
                nwbfile.create_electrode_group(**electrode_group_kwargs)

        if not nwbfile.electrode_groups:
            device_name = list(nwbfile.devices.keys())[0]
            device = nwbfile.devices[device_name]
            if len(nwbfile.devices) > 1:
                warnings.warn("More than one device found when adding electrode group "
                              f"via channel properties: using device \'{device_name}\'. To use a "
                              "different device, indicate it the metadata argument.")

            electrode_group_kwargs = dict(defaults)
            electrode_group_kwargs.pop('device_name')
            electrode_group_kwargs.update(device=device)
            for grp_name in np.unique(recording.get_channel_groups()).tolist():
                electrode_group_kwargs.update(name=str(grp_name))
                nwbfile.create_electrode_group(**electrode_group_kwargs)

    @staticmethod
    def add_electrodes(recording: se.RecordingExtractor, nwbfile=None, metadata: dict = None):
        """
        Auxiliary static method for nwbextractor.

        Adds channels from recording object as electrodes to nwbfile object.

        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['Electrodes'] = [{'name': my_name,
                                                      'description': my_description,
                                                      'data': [my_electrode_data]}, ...]
            where [my_electrode_data] is a list in one-to-one correspondence with
            the nwbfile electrode ids and RecordingExtractor channel ids.

        Missing keys in an element of metadata['Ecephys']['ElectrodeGroup'] will be auto-populated with defaults
        whenever possible.

        If 'my_name' is set to one of the required fields for nwbfile
        electrodes (id, x, y, z, imp, loccation, filtering, group_name),
        then the metadata will override their default values.

        Setting 'my_name' to 'group' is not supported as the linking to nwbfile.electrode_groups is handled
        automatically; please specify the string 'group_name' in this case.

        If no group information is passed via metadata, automatic linking to existing electrode groups,
        possibly including the default, will occur.
        """
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        if nwbfile.electrode_groups is None:
            se.NwbRecordingExtractor.add_electrode_groups(recording, nwbfile, metadata)
        # For older versions of pynwb, we need to manually add these columns
        if distutils.version.LooseVersion(pynwb.__version__) < '1.3.0':
            if nwbfile.electrodes is None or 'rel_x' not in nwbfile.electrodes.colnames:
                nwbfile.add_electrode_column('rel_x', 'x position of electrode in electrode group')
            if nwbfile.electrodes is None or 'rel_y' not in nwbfile.electrodes.colnames:
                nwbfile.add_electrode_column('rel_y', 'y position of electrode in electrode group')
        defaults = dict(
            id=np.nan,
            x=np.nan,
            y=np.nan,
            z=np.nan,
            # There doesn't seem to be a canonical default for impedence, if missing.
            # The NwbRecordingExtractor follows the -1.0 convention, other scripts sometimes use np.nan
            imp=-1.0,
            location="unknown",
            filtering="none",
            group_name="Electrode Group"
        )
        if metadata is None:
            metadata = dict(
                Ecephys=dict(
                    Electrodes=[]
                )
            )
        assert all([isinstance(x, dict) and set(x.keys()) == set(['name', 'description', 'data'])
                    and isinstance(x['data'], list) for x in metadata['Ecephys']['Electrodes']]), \
            "Expected metadata['Ecephys']['Electrodes'] to be a list of dictionaries!"

        if nwbfile.electrodes is None:
            nwb_elec_ids = []
        else:
            nwb_elec_ids = nwbfile.electrodes.id.data[:]
        for metadata_column in metadata['Ecephys']['Electrodes']:
            if nwbfile.electrodes is None or metadata_column['name'] not in nwbfile.electrodes.colnames:
                nwbfile.add_electrode_column(str(metadata_column['name']),
                                             str(metadata_column['description']))

        for j, channel_id in enumerate(recording.get_channel_ids()):
            if channel_id not in nwb_elec_ids:
                electrode_kwargs = dict(defaults)
                electrode_kwargs.update(id=channel_id)

                # recording.get_channel_locations defaults to np.nan if there are none
                location = recording.get_channel_locations(channel_ids=channel_id)[0]
                if all([not np.isnan(loc) for loc in location]):
                    # property 'location' of RX channels corresponds to rel_x and rel_ y of NWB electrodes
                    electrode_kwargs.update(
                        dict(
                            rel_x=float(location[0]),
                            rel_y=float(location[1])
                        )
                    )

                for metadata_column in metadata['Ecephys']['Electrodes']:
                    if metadata_column['name'] == 'group_name':
                        group_name = list_get(metadata_column['data'], j, defaults['group_name'])
                        if group_name not in nwbfile.electrode_groups:
                            warnings.warn(f"Electrode group for electrode {channel_id} was not "
                                          "found in the nwbfile! Automatically adding.")
                            missing_group_metadata = dict(
                                Ecephys=dict(
                                    ElectrodeGroup=dict(
                                        name=group_name,
                                        description="no description",
                                        location="unknown",
                                        device_name="Device"
                                    )
                                )
                            )
                            se.NwbRecordingExtractor.add_electrode_groups(recording, nwbfile, missing_group_metadata)
                        electrode_kwargs.update(
                            dict(
                                group=nwbfile.electrode_groups[group_name],
                                group_name=group_name
                            )
                        )
                    elif metadata_column['name'] != 'group':
                        if metadata_column['name'] in defaults:
                            electrode_kwargs.update({
                                metadata_column['name']: list_get(metadata_column['data'], j,
                                                                  defaults[metadata_column['name']])
                            })
                        else:
                            if j < len(metadata_column['data']):
                                electrode_kwargs.update({
                                    metadata_column['name']: metadata_column['data'][j]
                                })
                            else:
                                metadata_column_name = metadata_column['name']
                                warnings.warn(f"Custom column {metadata_column_name} "
                                              f"has incomplete data for channel id [{j}] and no "
                                              "set default! Electrode will not be added.")
                                continue

                if not any([x.get('name', '') == 'group_name' for x in metadata['Ecephys']['Electrodes']]):
                    group_id = recording.get_channel_groups(channel_ids=channel_id)[0]
                    if group_id in range(len(nwbfile.electrode_groups)):
                        group_name = list(nwbfile.electrode_groups.keys())[group_id]
                        electrode_kwargs.update(
                            dict(
                                group=nwbfile.electrode_groups[group_name],
                                group_name=group_name
                            )
                        )
                    else:
                        warnings.warn("No metadata was passed specifying the electrode group for "
                                      f"electrode {channel_id}, and the internal recording channel group was "
                                      f"assigned a value ({group_id}) outside the indices of the electrode "
                                      "groups in the nwbfile! Electrode will not be added.")
                        continue

                nwbfile.add_electrode(**electrode_kwargs)

        # property 'gain' should not be in the NWB electrodes_table
        # property 'brain_area' of RX channels corresponds to 'location' of NWB electrodes
        channel_prop_names = set(recording.get_shared_channel_property_names()) - set(nwbfile.electrodes.colnames) \
            - set(['gain', 'location'])
        for channel_prop_name in channel_prop_names:
            for channel_id in recording.get_channel_ids():
                val = recording.get_channel_property(channel_id, channel_prop_name)
                descr = 'no description'
                if channel_prop_name == 'brain_area':
                    channel_prop_name = 'location'
                    descr = 'brain area location'
                set_dynamic_table_property(
                    dynamic_table=nwbfile.electrodes,
                    row_ids=[channel_id],
                    property_name=channel_prop_name,
                    values=[val],
                    default_value=np.nan,
                    description=descr
                )

    @staticmethod
    def add_electrical_series(recording: se.RecordingExtractor, nwbfile=None, metadata: dict = None,
                              buffer_mb: int = 500):
        """
        Auxiliary static method for nwbextractor.

        Adds traces from recording object as ElectricalSeries to nwbfile object.

        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['ElectricalSeries'] = {'name': my_name,
                                                           'description': my_description}
        buffer_mb: int (optional, defaults to 500MB)
            maximum amount of memory (in MB) to use per iteration of the
            DataChunkIterator (requires traces to be memmap objects)

        Missing keys in an element of metadata['Ecephys']['ElectrodeGroup'] will be auto-populated with defaults
        whenever possible.
        """
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile!"
        assert buffer_mb > 10, "'buffer_mb' should be at least 10MB to ensure data can be chunked!"
        if not nwbfile.electrodes:
            se.NwbRecordingExtractor.add_electrodes(recording, nwbfile, metadata)
        defaults = dict(
            name="ElectricalSeries",
            description="no description"
        )
        if metadata is None:
            metadata = dict(
                Ecephys=dict(
                    ElectricalSeries=dict(defaults)
                )
            )
        assert isinstance(metadata['Ecephys']['ElectricalSeries'], dict), \
            "Expected metadata['Ecephys']['ElectricalSeries'] to be a dictionary!"

        es_name = metadata['Ecephys']['ElectricalSeries'].get('name', defaults['name'])
        channel_ids = recording.get_channel_ids()
        if es_name not in nwbfile.acquisition:
            table_ids = [list(nwbfile.electrodes.id[:]).index(id) for id in channel_ids]
            electrode_table_region = nwbfile.create_electrode_table_region(
                region=table_ids,
                description="electrode_table_region"
            )

            # channels gains - for RecordingExtractor, these are values to cast traces to uV
            # for nwb, the conversions (gains) cast the data to Volts
            # To get traces in Volts = data*channel_conversion*conversion
            gains = np.squeeze([recording.get_channel_gains(channel_ids=[ch])
                                if 'gain' in recording.get_channel_property_names(channel_id=ch) else 1
                                for ch in channel_ids])
            if len(np.unique(gains)) == 1:  # if all gains are equal
                scalar_conversion = np.unique(gains)[0] * 1e-6
                channel_conversion = None
            else:
                scalar_conversion = 1.
                channel_conversion = gains * 1e-6

            if isinstance(recording.get_traces(), np.memmap):
                n_bytes = np.dtype(recording.get_dtype()).itemsize
                buffer_size = int(buffer_mb * 1e6) // (recording.get_num_channels() * n_bytes)
                ephys_data = H5DataIO(
                    DataChunkIterator(
                        data=recording.get_traces().T,  # nwb standard is time as zero axis
                        buffer_size=buffer_size
                    ),
                    compression='gzip'
                )
            else:
                def data_generator(recording, channels_ids):
                    #  generates data chunks for iterator
                    for id in channels_ids:
                        data = recording.get_traces(channel_ids=[id]).flatten()
                        yield data
                ephys_data = H5DataIO(
                    DataChunkIterator(
                        data=data_generator(
                            recording=recording,
                            channels_ids=channel_ids
                        ),
                        iter_axis=1  # nwb standard is time as zero axis
                    ),
                    compression='gzip'
                )
            nwbfile.add_acquisition(
                ElectricalSeries(
                    name=es_name,
                    data=ephys_data,
                    electrodes=electrode_table_region,
                    starting_time=recording.frame_to_time(0),
                    rate=recording.get_sampling_frequency(),
                    conversion=scalar_conversion,
                    channel_conversion=channel_conversion,
                    comments='Generated from SpikeInterface::NwbRecordingExtractor',
                    description=metadata['Ecephys']['ElectricalSeries'].get('description', defaults['description'])
                )
            )

    @staticmethod
    def add_epochs(recording: se.RecordingExtractor, nwbfile=None,
                   metadata: dict = None):
        '''
        Auxiliary static method for nwbextractor.
        Adds epochs from recording object to nwbfile object.
        
        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
        '''
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        # add/update epochs
        for epoch_name in recording.get_epoch_names():
            epoch = recording.get_epoch_info(epoch_name)
            if nwbfile.epochs is None:
                nwbfile.add_epoch(
                    start_time=recording.frame_to_time(epoch['start_frame']),
                    stop_time=recording.frame_to_time(epoch['end_frame']),
                    tags=epoch_name
                )
            else:
                if [epoch_name] in nwbfile.epochs['tags'][:]:
                    ind = nwbfile.epochs['tags'][:].index([epoch_name])
                    nwbfile.epochs['start_time'].data[ind] = recording.frame_to_time(epoch['start_frame'])
                    nwbfile.epochs['stop_time'].data[ind] = recording.frame_to_time(epoch['end_frame'])
                else:
                    nwbfile.add_epoch(
                        start_time=recording.frame_to_time(epoch['start_frame']),
                        stop_time=recording.frame_to_time(epoch['end_frame']),
                        tags=epoch_name
                    )

    @staticmethod
    def add_all_to_nwbfile(recording: se.RecordingExtractor, nwbfile=None,
                           metadata: dict = None):
        '''
        Auxiliary static method for nwbextractor.
        Adds all recording related information from recording object and metadata
        to the nwbfile object.
        
        Parameters
        ----------
        recording: RecordingExtractor
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Check the auxiliary function docstrings for more information
            about metadata format.
        '''
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        se.NwbRecordingExtractor.add_devices(
            recording=recording,
            nwbfile=nwbfile,
            metadata=metadata
        )

        se.NwbRecordingExtractor.add_electrode_groups(
            recording=recording,
            nwbfile=nwbfile,
            metadata=metadata
        )

        # Add electrodes
        se.NwbRecordingExtractor.add_electrodes(
            recording=recording,
            nwbfile=nwbfile,
            metadata=metadata
        )

        # Add electrical series
        se.NwbRecordingExtractor.add_electrical_series(
            recording=recording,
            nwbfile=nwbfile,
            metadata=metadata
        )

        # Add epochs
        se.NwbRecordingExtractor.add_epochs(
            recording=recording,
            nwbfile=nwbfile,
            metadata=metadata
        )

    @staticmethod
    def write_recording(recording: se.RecordingExtractor, save_path: PathType = None,
                        nwbfile=None, metadata: dict = None):
        '''
        Writes all recording related information from recording object and metadata
        to either a saved nwbfile (with save_path specified) or directly to an
        nwbfile object (if nwbfile specified).
        
        Parameters
        ----------
        recording: RecordingExtractor
        save_path: PathType
            Required if an nwbfile is not passed. Must be the path to the nwbfile
            being appended, otherwise one is created and written.
        nwbfile: NWBFile
            Required if a save_path is not specified. If passed, this function
            will fill the relevant fields within the nwbfile. E.g., calling
            spikeextractors.NwbRecordingExtractor.write_recording(
                my_recording_extractor, my_nwbfile
            )
            will result in the appropriate changes to the my_nwbfile object.
        metadata: dict
            metadata info for constructing the nwb file (optional). Should be
            of the format
                metadata['Ecephys'] = {}
            with keys of the forms
                metadata['Ecephys']['Device'] = [{'name': my_name,
                                                  'description': my_description}, ...]
                metadata['Ecephys']['ElectrodeGroup'] = [{'name': my_name,
                                                          'description': my_description,
                                                          'location': electrode_location,
                                                          'device_name': my_device_name}, ...]
                metadata['Ecephys']['Electrodes'] = [{'name': my_name,
                                                      'description': my_description,
                                                      'data': [my_electrode_data]}, ...]
                metadata['Ecephys']['ElectricalSeries'] = {'name': my_name,
                                                           'description': my_description}
        '''
        assert HAVE_NWB, NwbRecordingExtractor.installation_mesg

        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        assert distutils.version.LooseVersion(pynwb.__version__) >= '1.3.3', \
            "'write_recording' not supported for version < 1.3.3. Run pip install --upgrade pynwb"

        assert save_path is None or nwbfile is None, \
            "Either pass a save_path location, or nwbfile object, but not both!"

        # Update any previous metadata with user passed dictionary
        if hasattr(recording, 'nwb_metadata'):
            metadata = update_dict(recording.nwb_metadata, metadata)

        if nwbfile is None:
            if os.path.exists(save_path):
                read_mode = 'r+'
            else:
                read_mode = 'w'

            with NWBHDF5IO(save_path, mode=read_mode) as io:
                if read_mode == 'r+':
                    nwbfile = io.read()
                else:
                    # Default arguments will be over-written if contained in metadata
                    nwbfile_kwargs = dict(session_description='no description',
                                          identifier=str(uuid.uuid4()),
                                          session_start_time=datetime.now())
                    if metadata is not None and 'NWBFile' in metadata:
                        nwbfile_kwargs.update(metadata['NWBFile'])
                    nwbfile = NWBFile(**nwbfile_kwargs)

                    se.NwbRecordingExtractor.add_all_to_nwbfile(
                        recording=recording,
                        nwbfile=nwbfile,
                        metadata=metadata
                    )

                # Write to file
                io.write(nwbfile)
        else:
            se.NwbRecordingExtractor.add_all_to_nwbfile(
                recording=recording,
                nwbfile=nwbfile,
                metadata=metadata
            )


class NwbSortingExtractor(se.SortingExtractor):
    extractor_name = 'NwbSortingExtractor'
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"

    def __init__(self, file_path, electrical_series=None, sampling_frequency=None):
        """
        Parameters
        ----------
        path: path to NWB file
        electrical_series: pynwb.ecephys.ElectricalSeries object
        """
        assert HAVE_NWB, self.installation_mesg
        se.SortingExtractor.__init__(self)
        self._path = str(file_path)
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if sampling_frequency is None:
                # defines the electrical series from where the sorting came from
                # important to know the sampling_frequency
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
            else:
                self._sampling_frequency = sampling_frequency

            # get all units ids
            units_ids = nwbfile.units.id[:]

            # store units properties and spike features to dictionaries
            all_pr_ft = list(nwbfile.units.colnames)
            all_names = [i.name for i in nwbfile.units.columns]
            for item in all_pr_ft:
                if item == 'spike_times':
                    continue
                # test if item is a unit_property or a spike_feature
                if item + '_index' in all_names:  # if it has index, it is a spike_feature
                    for u_id in units_ids:
                        ind = list(units_ids).index(u_id)
                        self.set_unit_spike_features(u_id, item, nwbfile.units[item][ind])
                else:  # if it is unit_property
                    for u_id in units_ids:
                        ind = list(units_ids).index(u_id)
                        if isinstance(nwbfile.units[item][ind], pd.DataFrame):
                            prop_value = nwbfile.units[item][ind].index[0]
                        else:
                            prop_value = nwbfile.units[item][ind]

                        if isinstance(prop_value, (list, np.ndarray)):
                            self.set_unit_property(u_id, item, prop_value)
                        else:
                            if prop_value == prop_value:  # not nan
                                self.set_unit_property(u_id, item, prop_value)

            # Fill epochs dictionary
            self._epochs = {}
            if nwbfile.epochs is not None:
                df_epochs = nwbfile.epochs.to_dataframe()
                self._epochs = {row['tags'][0]: {
                    'start_frame': self.time_to_frame(row['start_time']),
                    'end_frame': self.time_to_frame(row['stop_time'])}
                    for _, row in df_epochs.iterrows()}
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'electrical_series': electrical_series,
                        'sampling_frequency': sampling_frequency}

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
            unit_ids = [int(i) for i in nwbfile.units.id[:]]
        return unit_ids

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        check_nwb_install()
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            # chosen unit and interval
            times = nwbfile.units['spike_times'][list(nwbfile.units.id[:]).index(unit_id)][:]
            # spike times are measured in samples
            frames = self.time_to_frame(times)
        return frames[(frames > start_frame) & (frames < end_frame)]

    def time_to_frame(self, time):
        return np.round(time * self.get_sampling_frequency()).astype('int64')

    def frame_to_time(self, frame):
        return np.round(frame / self.get_sampling_frequency(), 6)

    @staticmethod
    def write_units(sorting: se.SortingExtractor, nwbfile,
                    property_descriptions: dict):
        '''
        Helper function for write_sorting.
        '''
        unit_ids = sorting.get_unit_ids()
        fs = sorting.get_sampling_frequency()

        all_properties = set()
        all_features = set()
        for unit_id in unit_ids:
            all_properties.update(sorting.get_unit_property_names(unit_id))
            all_features.update(sorting.get_unit_spike_feature_names(unit_id))

        if property_descriptions is not None:
            for pr in all_properties:
                if pr not in property_descriptions:
                    print(f"Warning: description for property {pr} not found in property_descriptions. "
                          f"Setting description to 'no description'")
        else:
            property_descriptions = {}

        # If no Units present in nwb file
        if nwbfile.units is None:
            # check that array properties have the same shape across units
            skip_properties = []
            property_shapes = {}
            for pr in all_properties:
                shapes = []
                for unit_id in unit_ids:
                    if pr in sorting.get_unit_property_names(unit_id):
                        prop_value = sorting.get_unit_property(unit_id, pr)
                        if isinstance(prop_value, (int, np.integer, float, np.float, str, bool)):
                            shapes.append(1)
                        elif isinstance(prop_value, (list, np.ndarray)):
                            if np.array(prop_value).ndim == 1:
                                shapes.append(len(prop_value))
                            else:
                                shapes.append(np.array(prop_value).shape)
                        elif isinstance(prop_value, dict):
                            print(f"Skipping property '{pr}' because dictionaries are not supported.")
                            skip_properties.append(pr)
                            break
                    else:
                        shapes.append(np.nan)
                property_shapes[pr] = shapes

            for pr in property_shapes.keys():
                if not np.all([elem == property_shapes[pr][0] for elem in property_shapes[pr]]):
                    print(f"Skipping property '{pr}' because it has variable size across units.")
                    skip_properties.append(pr)

            for pr in all_properties:
                # Special case of setting max_electrodes requires a table to be
                # passed to become a dynamic table region
                if pr not in skip_properties:
                    if pr in ['max_channel', 'max_electrode']:
                        if nwbfile.electrodes is None:
                            print('Warning: Attempted to make a custom column for max_channel '
                                  'or max_electrode, but there are no electrodes to reference! '
                                  'Column will not be added.')
                        else:
                            nwbfile.add_unit_column(pr, property_descriptions.get(pr, 'no description'),
                                                    table=nwbfile.electrodes)
                    else:
                        nwbfile.add_unit_column(pr, property_descriptions.get(pr, 'no description'))

            for unit_id in unit_ids:
                unit_kwargs = {}
                # spike trains withinin the SortingExtractor object are not scaled by sampling frequency
                spkt = sorting.get_unit_spike_train(unit_id=unit_id) / fs
                for pr in all_properties:
                    if pr not in skip_properties:
                        if pr in sorting.get_unit_property_names(unit_id):
                            prop_value = sorting.get_unit_property(unit_id, pr)
                            unit_kwargs.update({pr: prop_value})
                        else:  # Case of missing data for this unit and this property
                            unit_kwargs.update({pr: np.nan})
                nwbfile.add_unit(id=unit_id, spike_times=spkt, **unit_kwargs)

            # TODO
            # # Stores average and std of spike traces
            # This will soon be updated to the current NWB standard
            # if 'waveforms' in sorting.get_unit_spike_feature_names(unit_id=id):
            #     wf = sorting.get_unit_spike_features(unit_id=id,
            #                                          feature_name='waveforms')
            #     relevant_ch = most_relevant_ch(wf)
            #     # Spike traces on the most relevant channel
            #     traces = wf[:, relevant_ch, :]
            #     traces_avg = np.mean(traces, axis=0)
            #     traces_std = np.std(traces, axis=0)
            #     nwbfile.add_unit(
            #         id=id,
            #         spike_times=spkt,
            #         waveform_mean=traces_avg,
            #         waveform_sd=traces_std
            #     )

            # check that multidimensional features have the same shape across units
            feature_shapes = {}
            skip_features = []
            for ft in all_features:
                shapes = []
                for unit_id in unit_ids:
                    if ft in sorting.get_unit_spike_feature_names(unit_id):
                        feat_value = sorting.get_unit_spike_features(unit_id, ft)
                        if isinstance(feat_value[0], (int, np.integer, float, np.float, str, bool)):
                            break
                        elif isinstance(feat_value[0], (list, np.ndarray)):  # multidimensional features
                            if np.array(feat_value).ndim > 1:
                                shapes.append(np.array(feat_value).shape)
                                feature_shapes[ft] = shapes
                        elif isinstance(feat_value[0], dict):
                            print(f"Skipping feature '{ft}' because dictionaries are not supported.")
                            skip_features.append(ft)
                            break
                    else:
                        print(f"Skipping feature '{ft}' because not share across all units.")
                        skip_features.append(ft)
                        break

            nspikes = {k: get_nspikes(nwbfile.units, int(k)) for k in unit_ids}

            for ft in feature_shapes.keys():
                # skip first dimension (num_spikes) when comparing feature shape
                if not np.all([elem[1:] == feature_shapes[ft][0][1:] for elem in feature_shapes[ft]]):
                    print(f"Skipping feature '{ft}' because it has variable size across units.")
                    skip_features.append(ft)

            for ft in all_features:
                values = []
                if not ft.endswith('_idxs'):
                    for unit_id in sorting.get_unit_ids():
                        feat_vals = sorting.get_unit_spike_features(unit_id, ft)

                        if len(feat_vals) < nspikes[unit_id]:
                            skip_features.append(ft)
                            print(f"Skipping feature '{ft}' because it is not defined for all spikes.")
                            break
                            # this means features are available for a subset of spikes
                            # all_feat_vals = np.array([np.nan] * nspikes[unit_id])
                            # feature_idxs = sorting.get_unit_spike_features(unit_id, feat_name + '_idxs')
                            # all_feat_vals[feature_idxs] = feat_vals
                        else:
                            all_feat_vals = feat_vals
                        values.append(all_feat_vals)

                    if ft not in skip_features:
                        flatten_vals = [item for sublist in values for item in sublist]
                        nspks_list = [sp for sp in nspikes.values()]
                        spikes_index = np.cumsum(nspks_list).astype('int64')

                        set_dynamic_table_property(
                            dynamic_table=nwbfile.units,
                            row_ids=unit_ids,
                            property_name=ft,
                            values=flatten_vals,
                            index=spikes_index,
                        )

        else:  # there are already units in the nwbfile
            print("Warning: The nwbfile already contains units. "
                  "These units will not be over-written.")

    @staticmethod
    def write_sorting(sorting: se.SortingExtractor, save_path: PathType = None,
                      nwbfile=None, property_descriptions: dict = None,
                      **nbwbfile_kwargs):
        '''
        Parameters
        ----------
        sorting: SortingExtractor
        save_path: PathType
            Required if an nwbfile is not passed. Must be the path to the nwbfile
            being appended, otherwise one is created and written.
        nwbfile: NWBFile
            Required if a save_path is not specified. If passed, this function
            will fill the relevant fields within the nwbfile. E.g., calling
            spikeextractors.NwbRecordingExtractor.write_recording(
                my_recording_extractor, my_nwbfile
            )
            will result in the appropriate changes to the my_nwbfile object.
        property_descriptions: dict
            For each key in this dictionary which matches the name of a unit
            property in sorting, adds the value as a description to that
            custom unit column.
        nbwbfile_kwargs: dict
            Information for constructing the nwb file (optional).
            Only used if no nwbfile exists at the save_path, and no nwbfile
            was directly passed.
        '''
        assert HAVE_NWB, NwbSortingExtractor.installation_mesg
        assert save_path is None or nwbfile is None, \
            "Either pass a save_path location, or nwbfile object, but not both!"

        if nwbfile is None:
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
                    kwargs.update(**nbwbfile_kwargs)
                    nwbfile = NWBFile(**kwargs)

                se.NwbSortingExtractor.write_units(sorting, nwbfile,
                                                   property_descriptions)

                io.write(nwbfile)
        else:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
            se.NwbSortingExtractor.write_units(sorting, nwbfile,
                                               property_descriptions)
