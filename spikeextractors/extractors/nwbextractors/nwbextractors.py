import os
import uuid
from datetime import datetime
from collections import abc
from pathlib import Path
import numpy as np
import distutils.version
from typing import Union

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
    """Returns the number of spikes for chosen unit."""
    check_nwb_install()
    if unit_id not in units_table.id[:]:
        raise ValueError(str(unit_id) + " is an invalid unit_id. "
                                        "Valid ids: " + str(units_table.id[:].tolist()))
    nSpikes = np.diff([0] + list(units_table['spike_times_index'].data[:])).tolist()
    ind = np.where(np.array(units_table.id[:]) == unit_id)[0][0]
    return nSpikes[ind]


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
    def add_devices(recording: se.RecordingExtractor, nwbfile=None,
                    metadata: dict = None):
        '''
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
        '''
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        defaults = {'name': 'Device',
                    'description': 'no description'}

        if metadata is None:
            metadata = dict()

        if len(metadata.keys()) > 0:
            if 'Ecephys' in metadata \
                    and 'Device' in metadata['Ecephys'] \
                    and type(metadata['Ecephys']['Device']) is list \
                    and metadata['Ecephys']['Device']:
                for j, dev in enumerate(metadata['Ecephys']['Device']):
                    if type(dev) is dict:
                        # Will not overwrite the named device if already in nwbfile
                        if dev.get('name', defaults['name']) not in nwbfile.devices:
                            device_kwargs = fill_kwargs_from_defaults(defaults, dev)
                            nwbfile.create_device(**device_kwargs)
                    else:
                        print(f"Warning: Expected metadata['Ecephys']['Device'][{j}] to be"
                              " a dictionary with keys 'name' and 'description'!"
                              f" Device [{j}] will not be created.")
            else:
                print('HERE 2')
                print("Warning: metadata must be a list of dictionaries of the form"
                      " metadata['Ecephys']['Device'] = [{'name': my_name,"
                      " 'description': my_description}, ...]!")

        # If no device created above
        if not nwbfile.devices:
            device_kwargs = fill_kwargs_from_defaults(defaults)
            nwbfile.create_device(**device_kwargs)
            print("Warning: No device metadata provided. Generating default device!")

    @staticmethod
    def add_electrode_groups(recording: se.RecordingExtractor, nwbfile=None,
                             metadata: dict = None):
        '''
        Auxiliary static method for nwbextractor.
        Adds electrode group information to nwbfile object.
        Will always ensure nwbfile has at least one electrode group.
        Will auto-generate a linked device if the specified name 
        does not exist in the nwbfile.
        
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
        '''
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        defaults = {'name': 'Electrode Group',
                    'description': 'no description',
                    'location': 'unknown'}
        default_dev_name = 'Device'

        if metadata is None:
            metadata = dict()

        if len(metadata.keys()) > 0:
            if 'Ecephys' in metadata \
                    and 'ElectrodeGroup' in metadata['Ecephys'] \
                    and type(metadata['Ecephys']['ElectrodeGroup']) is list \
                    and metadata['Ecephys']['ElectrodeGroup']:
                for j, grp in enumerate(metadata['Ecephys']['ElectrodeGroup']):
                    if type(grp) is dict:
                        # Will not overwrite the named electrode_group if already in nwbfile
                        if grp.get('name', defaults['name']) not in nwbfile.electrode_groups:
                            # If named device link in electrode group does not exist, make it
                            if grp.get('device_name', default_dev_name) not in nwbfile.devices:
                                new_device = {'Ecephys': {'Device': {'name': grp.get('device_name', default_dev_name)}}}
                                se.NwbRecordingExtractor.add_devices(recording, nwbfile, metadata=new_device)
                                print("Warning: device name not detected in attempted link to electrode group! "
                                      "Automatically generating.")

                            electrode_group_kwargs = fill_kwargs_from_defaults(defaults, grp)
                            electrode_group_kwargs.update(
                                {'device': nwbfile.devices[grp.get('device_name', default_dev_name)]})
                            nwbfile.create_electrode_group(**electrode_group_kwargs)
                    else:
                        print(f"Warning: Expected metadata['Ecephy']['ElectrodeGroup'][{j}] to be"
                              " a dictionary with keys 'name', 'description', 'location', and 'device'!"
                              f"Electrode Group [{j}] will not be created.")
            else:
                print('HERE 1')

                print("Warning: metadata must be a list of dictionaries of the form"
                      " metadata['Ecephys']['ElectrodeGroup'] = [{'name': my_name,"
                      " 'description': my_description, 'location': electrode_location,"
                      " 'device_name': my_device_name}, ...]!")

        if not nwbfile.electrode_groups:
            if default_dev_name not in nwbfile.devices:
                new_device = {'Ecephys': {'Device': {'name': default_dev_name}}}
                se.NwbRecordingExtractor.add_devices(recording, nwbfile, metadata=new_device)

            electrode_group_kwargs = fill_kwargs_from_defaults(defaults)
            electrode_group_kwargs.update({'device': nwbfile.devices[default_dev_name]})

            if 'group' in recording.get_shared_channel_property_names():
                RX_groups_names = np.unique(recording.get_channel_groups()).tolist()
                for grp_name in RX_groups_names:
                    electrode_group_kwargs.update(
                        {'name': str(grp_name)})  # Over-write default name with internal value
                    nwbfile.create_electrode_group(**electrode_group_kwargs)
            else:
                nwbfile.add_electrode_groups(**electrode_group_kwargs)
                print("Warning: No electrode group metadata provided, and no"
                      " internal group properties set. Generating default device!")

    @staticmethod
    def add_electrodes(recording: se.RecordingExtractor, nwbfile=None,
                       metadata: dict = None):
        '''
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
        '''
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        defaults = {'id': np.nan,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                    # There doesn't seem to be a canonical default for impedence, if missing.
                    # The NwbRecordingExtractor follows the -1.0 convention, other scripts sometimes use np.nan
                    'imp': -1.0,
                    'location': 'unknown',
                    'filtering': 'none',
                    'group': np.nan}

        # If no electrode groups exist, make them
        if nwbfile.electrode_groups is None:
            se.NwbRecordingExtractor.add_electrode_groups(recording, nwbfile, metadata)

        if nwbfile.electrodes is not None:
            nwb_elec_ids = nwbfile.electrodes.id.data[:]
        else:
            nwb_elec_ids = []

        nwb_groups_names = list(nwbfile.electrode_groups.keys())
        channel_ids = list(recording.get_channel_ids())

        # For older versions of pynwb, we need to manually add these columns
        if distutils.version.LooseVersion(pynwb.__version__) < '1.3.0':
            if nwbfile.electrodes is None or 'rel_x' not in nwbfile.electrodes.colnames:
                nwbfile.add_electrode_column('rel_x', 'x position of electrode in electrode group')
            if nwbfile.electrodes is None or 'rel_y' not in nwbfile.electrodes.colnames:
                nwbfile.add_electrode_column('rel_y', 'y position of electrode in electrode group')

        if metadata is None:
            metadata = dict()

        if len(metadata.keys()) > 0:
            if 'Ecephys' in metadata \
                    and 'Electrodes' in metadata['Ecephys'] \
                    and type(metadata['Ecephys']['Electrodes']) is list \
                    and metadata['Ecephys']['Electrodes']:
                metadata_columns = metadata['Ecephys']['Electrodes']
                for j, custom_col in enumerate(metadata_columns):
                    if type(custom_col) is dict \
                            and set(custom_col.keys()) == set(['name', 'description', 'data']) \
                            and type(custom_col['data']) is list:
                        nwbfile.add_electrode_column(str(custom_col['name']),
                                                     str(custom_col['description']))
                    else:
                        print(f"Warning: Expected metadata['Ecephy']['Electrodes'][{j}] to be"
                              " a dictionary with keys 'name', 'description', and 'data',"
                              " with 'data' being a list of items!" +
                              " The custom column will not be added.")
            else:
                metadata_columns = []
                print('HERE')
                print("Warning: metadata must be a list of dictionaries of the form"
                      " metadata['Ecephys']['Electrodes'] = [{'name': my_name,"
                      " 'description': my_description, 'data': [my_electrode_data]}, ...]"
                      " where [my_electrode_data] is a list in one-to-one correspondence with"
                      " the nwbfile electrode ids and RecordingExtractor channel ids!")
        else:
            metadata_columns = []

        for j, channel_id in enumerate(channel_ids):
            # Will not overwrite the electrode id if already in nwbfile
            if channel_id not in nwb_elec_ids:
                electrode_kwargs = fill_kwargs_from_defaults(defaults)
                electrode_kwargs.update({'id': channel_id})

                # recording.get_channel_locations defaults to np.nan is there are none
                location = recording.get_channel_locations(channel_ids=channel_id)[0]
                if location[0] is not np.nan or location[1] is not np.nan:
                    electrode_kwargs.update({'rel_x': float(location[0]),
                                             'rel_y': float(location[1])})

                for metadata_column in metadata_columns:
                    if type(metadata_column) is dict \
                            and set(metadata_column.keys()) == set(['name', 'description', 'data']) \
                            and type(metadata_column['data']) is list:
                        if metadata_column['name'] == 'group':
                            if list_get(metadata_column['data'], j, np.nan) not in nwbfile.electrode_groups:
                                print(
                                    f"Warning: Electrode group for electrode {channel_id} was not found in the nwbfile."
                                    " Automatically adding!")
                                se.NwbRecordingExtractor.add_electrode_groups(recording, nwbfile, metadata)
                            electrode_kwargs.update({
                                metadata_column['name']: nwbfile.electrode_groups[list_get(metadata_column['data'],
                                                                                           j,
                                                                                           defaults['group'])]
                            })
                        else:
                            if metadata_column['name'] in defaults:
                                electrode_kwargs.update({
                                    metadata_column['name']: list_get(metadata_column['data'], j,
                                                                      defaults[metadata_column['name']])
                                })
                            else:
                                if j in range(len(metadata_column['data'])):
                                    electrode_kwargs.update({
                                        metadata_column['name']: metadata_column['data'][j]
                                    })
                                else:
                                    metadata_column_name = metadata_column['name']
                                    print(f"Warning: Custom column {metadata_column_name}"
                                          f" has incomplete data for channel id [{j}] and no"
                                          " set default! Electrode will not be added.")

                if not any([metadata_column.get('name', '') == 'group' for metadata_column in metadata_columns]):
                    grp_id = recording.get_channel_groups(channel_ids=channel_id)[0]
                    if grp_id in range(len(nwb_groups_names)):
                        electrode_kwargs.update({'group': nwbfile.electrode_groups[nwb_groups_names[grp_id]]})
                    else:
                        print("Warning: No metadata was passed specifying the electrode group for"
                              f" electrode {channel_id}, and the internal recording channel group was"
                              f" assigned a value ({grp_id}) outside the indices of the electrode"
                              " groups in the nwbfile! Electrode will not be added.")
                        continue

                nwbfile.add_electrode(**electrode_kwargs)

        # Add any additional custom columns from data specified via channel properties
        # property 'gain' should not be in the NWB electrodes_table
        # property 'location' of RX channels corresponds to rel_x and rel_ y of NWB electrodes
        # and rel_x, rel_y have already been added to this point
        channel_prop_names = set(recording.get_shared_channel_property_names()) - set(nwbfile.electrodes.colnames) \
                             - set(['gain', 'location'])
        for channel_prop_name in channel_prop_names:
            for channel_id in channel_ids:
                val = recording.get_channel_property(channel_id, channel_prop_name)
                descr = 'no description'
                # property 'brain_area' of RX channels corresponds to 'location' of NWB electrodes
                if channel_prop_name == 'brain_area':
                    if 'location' in nwbfile.electrodes.colnames:
                        continue
                    else:
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
    def add_electrical_series(recording: se.RecordingExtractor, nwbfile=None,
                              metadata: dict = None):
        '''
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
        '''
        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"
        defaults = {'name': 'ElectricalSeries',
                    'description': 'electrical_series_description'}

        if metadata is None:
            metadata = dict()

        if len(metadata.keys()) > 0:
            if 'Ecephys' in metadata \
                    and 'ElectricalSeries' in metadata['Ecephys'] \
                    and type(metadata['Ecephys']['ElectricalSeries']) is dict:
                es_name = metadata['Ecephys']['ElectricalSeries'].get('name',
                                                                      defaults['name'])
                es_descr = metadata['Ecephys']['ElectricalSeries'].get('description',
                                                                       defaults['description'])
            else:
                es_name = defaults['name']
                es_descr = defaults['description']
                print("Warning: metadata must be a dictionary of the form"
                      " metadata['Ecephys']['ElectricalSeries'] = {'name': my_name,"
                      " 'description': my_description}!")
        else:
            es_name = defaults['name']
            es_descr = defaults['description']

        if not nwbfile.electrodes:
            se.NwbRecordingExtractor.add_electrodes(recording, nwbfile, metadata)

        channel_ids = recording.get_channel_ids()
        rate = recording.get_sampling_frequency()
        if es_name not in nwbfile.acquisition:
            table_ids = [list(nwbfile.electrodes.id[:]).index(id) for id in channel_ids]
            electrode_table_region = nwbfile.create_electrode_table_region(
                region=table_ids,
                description='electrode_table_region'
            )

            # channels gains - for RecordingExtractor, these are values to cast traces to uV
            # for nwb, the conversions (gains) cast the data to Volts
            gains = np.squeeze([recording.get_channel_gains(channel_ids=[ch])
                                if 'gain' in recording.get_channel_property_names(channel_id=ch) else 1
                                for ch in channel_ids])
            if len(np.unique(gains)) == 1:  # if all gains are equal
                scalar_conversion = np.unique(gains)[0] * 1e-6
                channel_conversion = None
            else:
                scalar_conversion = 1.
                channel_conversion = gains * 1e-6

            def data_generator(recording, channels_ids):
                #  generates data chunks for iterator
                for id in channels_ids:
                    data = recording.get_traces(channel_ids=[id]).flatten()
                    yield data

            data = data_generator(recording=recording, channels_ids=channel_ids)
            ephys_data = DataChunkIterator(data=data, iter_axis=1)

            # To get traces in Volts = data*channel_conversion*conversion
            ephys_ts = ElectricalSeries(
                name=es_name,
                data=ephys_data,
                electrodes=electrode_table_region,
                starting_time=recording.frame_to_time(0),
                rate=rate,
                conversion=scalar_conversion,
                channel_conversion=channel_conversion,
                comments='Generated from SpikeInterface::NwbRecordingExtractor',
                description=es_descr
            )
            nwbfile.add_acquisition(ephys_ts)

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
        if metadata is None:
            metadata = dict()
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
                    if 'NWBFile' in metadata:
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
                    for id in units_ids:
                        ind = list(units_ids).index(id)
                        self.set_unit_spike_features(id, item, nwbfile.units[item][ind])
                else:  # if it is unit_property
                    for id in units_ids:
                        ind = list(units_ids).index(id)
                        if isinstance(nwbfile.units[item][ind], pd.DataFrame):
                            prop_value = nwbfile.units[item][ind].index[0]
                        else:
                            prop_value = nwbfile.units[item][ind]

                        if isinstance(prop_value, (list, np.ndarray)):
                            self.set_unit_property(id, item, prop_value)
                        else:
                            if prop_value == prop_value:  # not nan
                                self.set_unit_property(id, item, prop_value)

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
        return np.round(time * self.get_sampling_frequency()).astype('int')

    def frame_to_time(self, frame):
        return frame / self.get_sampling_frequency()

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
            for pr in all_properties:
                # Special case of setting max_electrodes requires a table to be
                # passed to become a dynamic table region
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
                    if pr in sorting.get_unit_property_names(unit_id):
                        unit_kwargs.update({pr: sorting.get_unit_property(unit_id, pr)})
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

            nspikes = {k: get_nspikes(nwbfile.units, int(k)) for k in unit_ids}
            for ft in all_features:
                values = []
                skip_feature = False
                if not ft.endswith('_idxs'):
                    for unit_id in sorting.get_unit_ids():
                        feat_vals = sorting.get_unit_spike_features(unit_id, ft)

                        if len(feat_vals) < nspikes[unit_id]:
                            # TODO address this case. This is very common when computing a subset of waveforms to compute templates, for example
                            skip_feature = True
                            break
                            # this means features are available for a subset of spikes
                            # all_feat_vals = np.array([np.nan] * nspikes[unit_id])
                            # feature_idxs = sorting.get_unit_spike_features(unit_id, feat_name + '_idxs')
                            # all_feat_vals[feature_idxs] = feat_vals
                        else:
                            all_feat_vals = feat_vals
                        values.append(all_feat_vals)

                    if skip_feature:
                        print(f"Feature '{ft}' is not defined for all spikes. Skipping.")
                        continue

                    flatten_vals = [item for sublist in values for item in sublist]
                    nspks_list = [sp for sp in nspikes.values()]
                    spikes_index = np.cumsum(nspks_list).tolist()

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
