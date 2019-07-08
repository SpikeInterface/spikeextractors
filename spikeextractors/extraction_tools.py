import numpy as np
from .RecordingExtractor import RecordingExtractor
from .SortingExtractor import SortingExtractor
from .SubRecordingExtractor import SubRecordingExtractor
from .SubSortingExtractor import SubSortingExtractor
import csv
import os
import sys
from pathlib import Path


def read_python(path):
    '''Parses python scripts in a dictionary

    Parameters
    ----------
    path: str or Path
        Path to file to parse

    Returns
    -------
    metadata:
        dictionary containing parsed file

    '''
    from six import exec_
    import re
    path = Path(path).absolute()
    assert path.is_file()
    with path.open('r') as f:
        contents = f.read()
    contents = re.sub(r'range\(([\d,]*)\)',r'list(range(\1))',contents)
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def write_python(path, dict):
    '''Saves python dictionary to file

    Parameters
    ----------
    path: str or Path
        Path to save file
    dict: dict
        dictionary to save
    '''
    with Path(path).open('w') as f:
        for k, v in dict.items():
            if isinstance(v ,str) and not v.startswith("'"):
                if 'path' in k and 'win' in sys.platform:
                    f.write(str(k) + " = r'" + str(v) + "'\n")
                else:
                    f.write(str(k) + " = '" + str(v) + "'\n")
            else:
                f.write(str(k) + " = " + str(v) + "\n")


def load_probe_file(recording, probe_file, channel_map=None, channel_groups=None):
    '''Loads channel information into recording extractor. If a .prb file is given,
    then 'location' and 'group' information for each channel is stored. If a .csv
    file is given, then it will only store 'location'

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to channel information
    probe_file: str
        Path to probe file. Either .prb or .csv
    Returns
    ---------
    subRecordingExtractor
    '''
    probe_file = Path(probe_file)
    if probe_file.suffix == '.prb':
        probe_dict = read_python(probe_file)
        if 'channel_groups' in probe_dict.keys():
            ordered_channels = np.array([], dtype=int)
            groups = sorted(probe_dict['channel_groups'].keys())
            for cgroup_id in groups:
                cgroup = probe_dict['channel_groups'][cgroup_id]
                for key_prop, prop_val in cgroup.items():
                    if key_prop == 'channels':
                        ordered_channels = np.concatenate((ordered_channels, prop_val))

            if list(ordered_channels) == recording.get_channel_ids():
                subrecording = recording
            else:
                if not np.all([chan in recording.get_channel_ids() for chan in ordered_channels]):
                    print('Some channel in PRB file are not in original recording')
                present_ordered_channels = [chan for chan in ordered_channels if chan in recording.get_channel_ids()]
                subrecording = SubRecordingExtractor(recording, channel_ids=present_ordered_channels)
            for cgroup_id in groups:
                cgroup = probe_dict['channel_groups'][cgroup_id]
                if 'channels' not in cgroup.keys() and len(groups) > 1:
                    raise Exception("If more than one 'channel_group' is in the probe file, the 'channels' field"
                                    "for each channel group is required")
                elif 'channels' not in cgroup.keys():
                    channels_in_group = subrecording.get_num_channels()
                else:
                    channels_in_group = len(cgroup['channels'])
                for key_prop, prop_val in cgroup.items():
                    if key_prop == 'channels':
                        for i_ch, prop in enumerate(prop_val):
                            if prop in subrecording.get_channel_ids():
                                subrecording.set_channel_property(prop, 'group', int(cgroup_id))
                    elif key_prop == 'geometry' or key_prop == 'location':
                        if isinstance(prop_val, dict):
                            if len(prop_val.keys()) != channels_in_group:
                                print('geometry in PRB does not have the same length as channel in group')
                            for (i_ch, prop) in prop_val.items():
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_property(i_ch, 'location', prop)
                        elif isinstance(prop_val, (list, np.ndarray)) and len(prop_val) == channels_in_group:
                            if 'channels' not in cgroup.keys():
                                raise Exception("'geometry'/'location' in the .prb file can be a list only if "
                                                "'channels' field is specified.")
                            channels_id_in_group = cgroup['channels']
                            if len(prop_val) != channels_in_group:
                                print('geometry in PRB does not have the same length as channel in group')
                            for (i_ch, prop) in zip(channels_id_in_group, prop_val):
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_property(i_ch, 'location', prop)
                    else:
                        if isinstance(prop_val, dict) and len(prop_val.keys()) == channels_in_group:
                            for (i_ch, prop) in prop_val.items():
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_property(i_ch, key_prop, prop)
                        elif isinstance(prop_val, (list, np.ndarray)) and len(prop_val) == channels_in_group:
                            for (i_ch, prop) in zip(subrecording.get_channel_ids(), prop_val):
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_property(i_ch, key_prop, prop)
                # create dummy locations
                if 'geometry' not in cgroup.keys() and 'location' not in cgroup.keys():
                    for i, chan in enumerate(subrecording.get_channel_ids()):
                        subrecording.set_channel_property(chan, 'location', [0, i])
        else:
            raise AttributeError("'.prb' file should contain the 'channel_groups' field")

    elif probe_file.suffix == '.csv':
        if channel_map is not None:
            assert np.all([chan in channel_map for chan in recording.get_channel_ids()]), \
                "all channel_ids in 'channel_map' must be in the original recording channel ids"
            subrecording = SubRecordingExtractor(recording, channel_ids=channel_map)
        else:
            subrecording = recording
        with probe_file.open() as csvfile:
            posreader = csv.reader(csvfile)
            row_count = 0
            loaded_pos = []
            for pos in (posreader):
                row_count += 1
                loaded_pos.append(pos)
            assert len(subrecording.get_channel_ids()) == row_count, "The .csv file must contain as many " \
                                                                     "rows as the number of channels in the recordings"
            for i_ch, pos in zip(subrecording.get_channel_ids(), loaded_pos):
                if i_ch in subrecording.get_channel_ids():
                    subrecording.set_channel_property(i_ch, 'location', list(np.array(pos).astype(float)))
            if channel_groups is not None and len(channel_groups) == len(subrecording.get_channel_ids()):
                for i_ch, chg in zip(subrecording.get_channel_ids(), channel_groups):
                    if i_ch in subrecording.get_channel_ids():
                        subrecording.set_channel_property(i_ch, 'group', chg)
    else:
        raise NotImplementedError("Only .csv and .prb probe files can be loaded.")

    return subrecording


def save_probe_file(recording, probe_file, format=None, radius=100, dimensions=None):
    '''Saves probe file from the channel information of the given recording
    extractor

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to save probe file from
    probe_file: str
        file name of .prb or .csv file to save probe information to
    format: str (optional)
        Format for .prb file. It can be either 'klusta' or 'spyking_circus'. Default is None.
    '''
    probe_file = Path(probe_file)
    if not probe_file.parent.is_dir():
        probe_file.parent.mkdir()

    if probe_file.suffix == '.csv':
        # write csv probe file
        with probe_file.open('w') as f:
            if 'location' in recording.get_channel_property_names():
                for chan in recording.get_channel_ids():
                    loc = recording.get_channel_property(chan, 'location')
                    if len(loc) == 2:
                        f.write(str(loc[0]))
                        f.write(',')
                        f.write(str(loc[1]))
                        f.write('\n')
                    elif len(loc) == 3:
                        f.write(str(loc[0]))
                        f.write(',')
                        f.write(str(loc[1]))
                        f.write(',')
                        f.write(str(loc[2]))
                        f.write('\n')
            else:
                raise AttributeError("Recording extractor needs to have "
                                     "'location' property to save .csv probe file")
    elif probe_file.suffix == '.prb':
        _export_prb_file(recording, probe_file, format, radius=radius, dimensions=dimensions)
    else:
        raise NotImplementedError("Only .csv and .prb probe files can be saved.")


def read_binary(file, numchan, dtype, frames_first=True, offset=0):
    '''
    Reads binary .bin or .dat file.

    Parameters
    ----------
    file: str
        File name
    numchan: int
        Number of channels
    dtype: dtype
        dtype of the file
    frames_first: bool
        If True frames are readed as the first dimension
    offset: int
        number of offset bytes

    Returns
    -------

    '''
    numchan = int(numchan)
    with Path(file).open() as f:
        nsamples = (os.fstat(f.fileno()).st_size - offset) // (numchan * np.dtype(dtype).itemsize)
        if frames_first:
            samples = np.memmap(f, np.dtype(dtype), mode='r', offset=offset,
                                shape=(nsamples, numchan))
            samples = np.transpose(samples)
        else:
            samples = np.memmap(f, np.dtype(dtype), mode='r', offset=offset,
                                shape=(numchan, nsamples))
    return samples


def write_binary_dat_format(recording, save_path, time_axis=0, dtype=None, chunksize=None):
    '''Saves the traces of a recording extractor in binary .dat format.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    save_path: str
        The path to the file.
    time_axis: 0 (default) or 1
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    dtype: dtype
        Type of the saved data. Default float32
    chunksize: None or int
        If not None then the copy done by chunk size.
        Thi avoid to much memory consumption for big files.
    Returns
    -------
    '''
    save_path = Path(save_path)
    if save_path.suffix == '':
        # when suffix is already raw/bin/dat do not change it.
        save_path = save_path.parent / (save_path.name + '.dat')

    if chunksize is None:
        traces = recording.get_traces()
        if dtype is not None:
            traces = traces.astype(dtype)
        if time_axis == 0:
            traces = traces.T
        with save_path.open('wb') as f:
            traces.tofile(f)
    else:
        assert time_axis ==0, 'chunked writting work only with time_axis 0'
        n_sample = recording.get_num_frames()
        n_chan = recording.get_num_channels()
        n_chunk = n_sample // chunksize
        if n_sample % chunksize > 0:
            n_chunk += 1
        with save_path.open('wb') as f:
            for i in range(n_chunk):
                traces = recording.get_traces(start_frame=i*chunksize,
                                              end_frame=min((i+1)*chunksize, n_sample))
                if dtype is not None:
                    traces = traces.astype(dtype)
                if time_axis == 0:
                    traces = traces.T
                f.write(traces.tobytes())
    return save_path


def get_sub_extractors_by_property(extractor, property_name, return_property_list=False):
    '''Divides Recording or Sorting Extractor based on the property_name (e.g. group)

    Parameters
    ----------
    extractor: RecordingExtractor or SortingExtractor
        The extractor to be subdivided in subextractors
    property_name: str
        The property used to subdivide the extractor
    return_property_list: bool
        If True the property list is returned

    Returns
    -------
    List of subextractors

    '''
    if isinstance(extractor, RecordingExtractor):
        if property_name not in extractor.get_channel_property_names():
            raise ValueError("'property_name' must be must be a property of the recording channels")
        else:
            sub_list = []
            recording = extractor
            properties = np.array([recording.get_channel_property(chan, property_name)
                                   for chan in recording.get_channel_ids()])
            prop_list = np.unique(properties)
            for prop in prop_list:
                prop_idx = np.where(prop == properties)
                chan_idx = list(np.array(recording.get_channel_ids())[prop_idx])
                sub_list.append(SubRecordingExtractor(recording, channel_ids=chan_idx))
            if return_property_list:
                return sub_list, prop_list
            else:
                return sub_list
    elif isinstance(extractor, SortingExtractor):
        if property_name not in extractor.get_unit_property_names():
            raise ValueError("'property_name' must be must be a property of the units")
        else:
            sub_list = []
            sorting = extractor
            properties = np.array([sorting.get_unit_property(unit, property_name)
                                   for unit in sorting.get_unit_ids()])
            prop_list = np.unique(properties)
            for prop in prop_list:
                prop_idx = np.where(prop == properties)
                unit_idx = list(np.array(sorting.get_unit_ids())[prop_idx])
                sub_list.append(SubSortingExtractor(sorting, unit_ids=unit_idx))
            if return_property_list:
                return sub_list, prop_list
            else:
                return sub_list
    else:
        raise ValueError("'extractor' must be a RecordingExtractor or a SortingExtractor")


def _export_prb_file(recording, file_name, format=None, adjacency_distance=None, graph=False, geometry=True, radius=100,
                     dimensions='all'):
    '''Exports .prb file

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to save probe file from
    file_name: str
        probe filename to be exported to
    format: str
        'klusta' | 'spiking_circus' (defualt=None)
    adjacency_distance: float
        distance to consider 2 channels adjacent (if 'location' is a property)
    graph: bool
        if True graph information is extracted and saved
    geometry:
        if True geometry is saved
    radius: float
        radius for template-matching (if format is 'spyking_circus')
    '''
    if format == 'klusta':
        graph = True
        geometry = False
    elif format == 'spyking_circus':
        graph = False
        geometry = True
    else:
        graph = True
        geometry = True

    file_name = Path(file_name)
    assert file_name is not None
    abspath = file_name.absolute()

    if geometry:
        if 'location' in recording.get_channel_property_names():
            positions = np.array([recording.get_channel_property(chan, 'location')
                                  for chan in recording.get_channel_ids()])
            if dimensions is not None:
                positions = positions[:, dimensions]
        else:
            print("'location' property is not available and it will not be saved.")
            positions = None
            geometry = False
    else:
        positions = None

    if 'group' in recording.get_channel_property_names():
        groups = np.array([recording.get_channel_property(chan, 'group') for chan in recording.get_channel_ids()])
        channel_groups = np.unique([groups])
    else:
        print("'group' property is not available and it will not be saved.")
        channel_groups = [0]
        groups = np.array([0] * recording.get_num_channels())

    n_elec = recording.get_num_channels()

    # find adjacency graph
    if graph:
        if positions is not None and adjacency_distance is not None:
            adj_graph = []
            for chg in channel_groups:
                group_graph = []
                elecs = list(np.where(groups == chg)[0])
                for i in range(len(elecs)):
                    for j in range(i, len(elecs)):
                        if elecs[i] != elecs[j]:
                            if np.linalg.norm(positions[elecs[i]] - positions[elecs[j]]) < adjacency_distance:
                                group_graph.append((elecs[i], elecs[j]))
                adj_graph.append(group_graph)
        else:
            # all connected by group
            adj_graph = []
            for chg in channel_groups:
                group_graph = []
                elecs = list(np.where(groups == chg)[0])
                for i in range(len(elecs)):
                    for j in range(i, len(elecs)):
                        if elecs[i] != elecs[j]:
                            group_graph.append((elecs[i], elecs[j]))
                adj_graph.append(group_graph)

    with abspath.open('w') as f:
        f.write('\n')
        if format == 'spyking_circus':
            f.write('total_nb_channels = ' + str(n_elec) + '\n')
            f.write('radius = ' + str(radius) + '\n')
        f.write('channel_groups = {\n')
        if len(channel_groups) > 0:
            for i_chg, chg in enumerate(channel_groups):
                f.write("     " + str(int(chg)) + ": ")
                elecs = list(np.where(groups == chg)[0])
                f.write("\n        {\n")
                f.write("           'channels': " + str(elecs) + ',\n')
                if graph:
                    if len(adj_graph) == 1:
                        f.write("           'graph':  " + str(adj_graph[0]) + ',\n')
                    else:
                        f.write("           'graph':  " + str(adj_graph[i_chg]) + ',\n')
                else:
                    f.write("           'graph':  [],\n")
                if geometry:
                    f.write("           'geometry':  {\n")
                    for i, pos in enumerate(positions[elecs]):
                        f.write('               ' + str(elecs[i]) + ': ' + str(list(pos)) + ',\n')
                    f.write('           }\n')
                f.write('       },\n')
            f.write('}\n')
        else:
            for elec in range(n_elec):
                f.write('    ' + str(elec) + ': ')
                f.write("\n        {\n")
                f.write("           'channels': [" + str(elec) + '],\n')
                f.write("           'graph':  [],\n")
                f.write('        },\n')
            f.write('}\n')
