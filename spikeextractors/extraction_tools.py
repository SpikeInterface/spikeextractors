import numpy as np
import csv
import os
import sys
from pathlib import Path
import warnings
import datetime
from functools import wraps
from .baseextractor import BaseExtractor
from tqdm import tqdm
from joblib import Parallel, delayed

try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


def read_python(path):
    """Parses python scripts in a dictionary

    Parameters
    ----------
    path: str or Path
        Path to file to parse

    Returns
    -------
    metadata:
        dictionary containing parsed file

    """
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
    """Saves python dictionary to file

    Parameters
    ----------
    path: str or Path
        Path to save file
    dict: dict
        dictionary to save
    """
    with Path(path).open('w') as f:
        for k, v in dict.items():
            if isinstance(v ,str) and not v.startswith("'"):
                if 'path' in k and 'win' in sys.platform:
                    f.write(str(k) + " = r'" + str(v) + "'\n")
                else:
                    f.write(str(k) + " = '" + str(v) + "'\n")
            else:
                f.write(str(k) + " = " + str(v) + "\n")


def load_probe_file(recording, probe_file, channel_map=None, channel_groups=None, verbose=False):
    """This function returns a SubRecordingExtractor that contains information from the given
    probe file (channel locations, groups, etc.) If a .prb file is given, then 'location' and 'group'
    information for each channel is added to the SubRecordingExtractor. If a .csv file is given, then
    it will only add 'location' to the SubRecordingExtractor.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to load channel information from.
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
    ---------
    subrecording: SubRecordingExtractor
        The extractor containing all of the probe information.
    """
    from .subrecordingextractor import SubRecordingExtractor
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
            if not np.all([chan in recording.get_channel_ids() for chan in ordered_channels]) and verbose:
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
                    channels_id_in_group = subrecording.get_channel_ids()
                else:
                    channels_in_group = len(cgroup['channels'])
                    channels_id_in_group = cgroup['channels']
                for key_prop, prop_val in cgroup.items():
                    if key_prop == 'channels':
                        for i_ch, prop in enumerate(prop_val):
                            if prop in subrecording.get_channel_ids():
                                subrecording.set_channel_groups(int(cgroup_id), channel_ids=prop)
                    elif key_prop == 'geometry' or key_prop == 'location':
                        if isinstance(prop_val, dict):
                            if len(prop_val.keys()) != channels_in_group and verbose:
                                print('geometry in PRB does not have the same length as channel in group')
                            for (i_ch, prop) in prop_val.items():
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_locations(prop, channel_ids=i_ch)
                        elif isinstance(prop_val, (list, np.ndarray)) and len(prop_val) == channels_in_group:
                            if 'channels' not in cgroup.keys():
                                raise Exception("'geometry'/'location' in the .prb file can be a list only if "
                                                "'channels' field is specified.")
                            if len(prop_val) != channels_in_group and verbose:
                                print('geometry in PRB does not have the same length as channel in group')
                            for (i_ch, prop) in zip(channels_id_in_group, prop_val):
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_locations(prop, channel_ids=i_ch)
                    else:
                        if isinstance(prop_val, dict) and len(prop_val.keys()) == channels_in_group:
                            for (i_ch, prop) in prop_val.items():
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_property(i_ch, key_prop, prop)
                        elif isinstance(prop_val, (list, np.ndarray)) and len(prop_val) == channels_in_group:
                            for (i_ch, prop) in zip(channels_id_in_group, prop_val):
                                if i_ch in subrecording.get_channel_ids():
                                    subrecording.set_channel_property(i_ch, key_prop, prop)
                # create dummy locations
                if 'geometry' not in cgroup.keys() and 'location' not in cgroup.keys():
                    if 'location' not in subrecording.get_shared_channel_property_names():
                        locs = np.zeros((subrecording.get_num_channels(), 2))
                        locs[:, 1] = np.arange(subrecording.get_num_channels())
                        subrecording.set_channel_locations(locs)
        else:
            raise AttributeError("'.prb' file should contain the 'channel_groups' field")

    elif probe_file.suffix == '.csv':
        if channel_map is not None:
            assert np.all([chan in channel_map for chan in recording.get_channel_ids()]), \
                "all channel_ids in 'channel_map' must be in the original recording channel ids"
            subrecording = SubRecordingExtractor(recording, channel_ids=channel_map)
        else:
            subrecording = SubRecordingExtractor(recording, channel_ids=recording.get_channel_ids())
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
                    subrecording.set_channel_locations(list(np.array(pos).astype(float)), i_ch)
            if channel_groups is not None and len(channel_groups) == len(subrecording.get_channel_ids()):
                for i_ch, chg in zip(subrecording.get_channel_ids(), channel_groups):
                    if i_ch in subrecording.get_channel_ids():
                        subrecording.set_channel_groups(chg, i_ch)
    else:
        raise NotImplementedError("Only .csv and .prb probe files can be loaded.")

    subrecording._kwargs['probe_file'] = str(probe_file.absolute())
    return subrecording


def save_to_probe_file(recording, probe_file, grouping_property=None, radius=None,
                       graph=True, geometry=True, verbose=False):
    """Saves probe file from the channel information of the given recording
    extractor.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to save probe file from
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
    probe_file = Path(probe_file)
    if not probe_file.parent.is_dir():
        probe_file.parent.mkdir()

    if probe_file.suffix == '.csv':
        # write csv probe file
        with probe_file.open('w') as f:
            if 'location' in recording.get_shared_channel_property_names():
                for chan in recording.get_channel_ids():
                    loc = recording.get_channel_locations(chan)[0]
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
        _export_prb_file(recording, probe_file, grouping_property=grouping_property, radius=radius, graph=graph,
                         geometry=geometry, verbose=verbose)
    else:
        raise NotImplementedError("Only .csv and .prb probe files can be saved.")


def read_binary(file, numchan, dtype, time_axis=0, offset=0):
    """
    Reads binary .bin or .dat file.

    Parameters
    ----------
    file: str
        File name
    numchan: int
        Number of channels
    dtype: dtype
        dtype of the file
    time_axis: 0 (default) or 1
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    offset: int
        number of offset bytes
    """
    numchan = int(numchan)
    with Path(file).open() as f:
        nsamples = (os.fstat(f.fileno()).st_size - offset) // (numchan * np.dtype(dtype).itemsize)
    if time_axis == 0:
        samples = np.memmap(file, np.dtype(dtype), mode='r', offset=offset, shape=(nsamples, numchan)).T
    else:
        samples = np.memmap(file, np.dtype(dtype), mode='r', offset=offset, shape=(numchan, nsamples))
    return samples


def write_to_binary_dat_format(recording, save_path=None, file_handle=None,
                               time_axis=0, dtype=None, chunk_size=None, chunk_mb=500, n_jobs=1, joblib_backend='loky',
                               return_scaled=True, verbose=False):
    """Saves the traces of a recording extractor in binary .dat format.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
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
    n_jobs: int
        Number of jobs to use (Default 1)
    joblib_backend: str
        Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
    return_scaled: bool
        If True, traces are written after scaling (using gain/offset). If False, the raw traces are written
    verbose: bool
        If True, output is verbose (when chunks are used)
    """
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == '':
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + '.dat')

    if chunk_size is not None or chunk_mb is not None:
        if time_axis == 1:
            print("Chunking disabled due to 'time_axis' == 1")
            chunk_size = None
            chunk_mb = None

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (recording.get_num_channels() * n_bytes)

    if n_jobs is None:
        n_jobs = 1
    if n_jobs == 0:
        n_jobs = 1

    if n_jobs > 1:
        if chunk_size is not None:
            chunk_size /= n_jobs

    if not recording.check_if_dumpable():
        if n_jobs > 1:
            n_jobs = 1
            print("RecordingExtractor is not dumpable and can't be processed in parallel")
        rec_arg = recording
    else:
        if n_jobs > 1:
            rec_arg = recording.dump_to_dict()
        else:
            rec_arg = recording

    if chunk_size is None:
        traces = recording.get_traces(return_scaled=return_scaled)
        if dtype is not None:
            traces = traces.astype(dtype)
        if time_axis == 0:
            traces = traces.T
        if save_path is not None:
            with save_path.open('wb') as f:
                traces.tofile(f)
        else:
            traces.tofile(file_handle)
    else:
        # chunk size is not None
        num_frames = recording.get_num_frames()
        num_channels = recording.get_num_channels()

        # chunk_size = num_bytes_per_chunk / num_bytes_per_frame
        chunks = divide_recording_into_time_chunks(
            num_frames=num_frames,
            chunk_size=chunk_size,
            padding_size=0
        )
        n_chunk = len(chunks)

        if verbose and n_jobs == 1:
            chunks_loop = tqdm(range(n_chunk), ascii=True, desc="Writing to binary .dat file")
        else:
            chunks_loop = range(n_chunk)
        if save_path is not None:
            if n_jobs == 1:
                if time_axis == 0:
                    shape = (num_frames, num_channels)
                else:
                    shape = (num_channels, num_frames)
                rec_memmap = np.memmap(str(save_path), dtype=dtype, mode='w+', shape=shape)
                for i in chunks_loop:
                    _write_dat_one_chunk(i, rec_arg, chunks, rec_memmap, dtype, time_axis, return_scaled,
                                         verbose=False)
            else:
                if time_axis == 0:
                    shape = (num_frames, num_channels)
                else:
                    shape = (num_channels, num_frames)
                rec_memmap = np.memmap(str(save_path), dtype=dtype, mode='w+', shape=shape)

                Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                    delayed(_write_dat_one_chunk)(i, rec_arg, chunks, rec_memmap, dtype, time_axis, return_scaled,
                                                  verbose,)
                    for i in chunks_loop)
        else:
            for i in chunks_loop:
                start_frame = chunks[i]['istart']
                end_frame = chunks[i]['iend']
                traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame,
                                              return_scaled=return_scaled)

                if dtype is not None:
                    traces = traces.astype(dtype)
                if time_axis == 0:
                    traces = traces.T
                file_handle.write(traces.tobytes())

    return save_path


def write_to_h5_dataset_format(recording, dataset_path, save_path=None, file_handle=None,
                               time_axis=0, dtype=None, chunk_size=None, chunk_mb=500, verbose=False):
    """Saves the traces of a recording extractor in an h5 dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    dataset_path: str
        Path to dataset in h5 filee (e.g. '/dataset')
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
    assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == '':
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + '.h5')

    num_channels = recording.get_num_channels()
    num_frames = recording.get_num_frames()

    if file_handle is not None:
        assert isinstance(file_handle, h5py.File)
    else:
        file_handle = h5py.File(save_path, 'w')

    if dtype is None:
        dtype_file = recording.get_dtype()
    else:
        dtype_file = dtype

    if time_axis == 0:
        dset = file_handle.create_dataset(dataset_path, shape=(num_frames, num_channels), dtype=dtype_file)
    else:
        dset = file_handle.create_dataset(dataset_path, shape=(num_channels, num_frames), dtype=dtype_file)

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (num_channels * n_bytes)

    if chunk_size is None:
        traces = recording.get_traces()
        if dtype is not None:
            traces = traces.astype(dtype_file)
        if time_axis == 0:
            traces = traces.T
        dset[:] = traces
    else:
        chunk_start = 0
        # chunk size is not None
        n_chunk = num_frames // chunk_size
        if num_frames % chunk_size > 0:
            n_chunk += 1
        if verbose:
            chunks = tqdm(range(n_chunk), ascii=True, desc="Writing to .h5 file")
        else:
            chunks = range(n_chunk)
        for i in chunks:
            traces = recording.get_traces(start_frame=i * chunk_size,
                                          end_frame=min((i + 1) * chunk_size, num_frames))
            chunk_frames = traces.shape[1]
            if dtype is not None:
                traces = traces.astype(dtype_file)
            if time_axis == 0:
                dset[chunk_start:chunk_start + chunk_frames] = traces.T
            else:
                dset[:, chunk_start:chunk_start + chunk_frames] = traces
            chunk_start += chunk_frames

    if save_path is not None:
        file_handle.close()
    return save_path


def get_sub_extractors_by_property(extractor, property_name, return_property_list=False):
    """Returns a list of SubExtractors from the Extractor based on the given
    property_name (e.g. group)

    Parameters
    ----------
    extractor: RecordingExtractor or SortingExtractor
        The extractor object to access SubRecordingExtractors from.
    property_name: str
        The property used to subdivide the extractor
    return_property_list: bool
        If True the property list is returned

    Returns
    -------
    sub_list: list
        The list of subextractors to be returned.
    OR
    sub_list, prop_list
        If return_property_list is True, the property list will be returned as well.
    """
    from spikeextractors import RecordingExtractor, SortingExtractor, SubRecordingExtractor, SubSortingExtractor

    if isinstance(extractor, RecordingExtractor):
        if property_name not in extractor.get_shared_channel_property_names():
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
        if property_name not in extractor.get_shared_unit_property_names():
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


def _export_prb_file(recording, file_name, grouping_property=None, graph=True, geometry=True,
                     radius=None, adjacency_distance=100, verbose=False):
    """Exports .prb file

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to save probe file from
    file_name: str
        probe filename to be exported to
    grouping_property: str (default None)
        If grouping_property is a shared_channel_property, different groups are saved based on the property.
    graph: bool
        If True, the adjacency graph is saved (default=True)
    geometry: bool
        If True, the geometry is saved (default=True)
    radius: float (default None)
        Adjacency radius (used by some sorters). If None it is not saved to the probe file.
    adjacency_distance: float
        Distance to consider two channels to adjacent (if 'location' is a property). If radius is given,
        then adjacency_distance is set to the radius.
    verbose : bool
        If True, output is verbose
    """
    file_name = Path(file_name)
    assert file_name is not None
    abspath = file_name.absolute()

    if radius is not None:
        adjacency_distance = radius

    if geometry:
        if 'location' in recording.get_shared_channel_property_names():
            positions = recording.get_channel_locations()
        else:
            if verbose:
                print("'location' property is not available and it will not be saved.")
            positions = None
            geometry = False
    else:
        positions = None

    if grouping_property is not None:
        if grouping_property in recording.get_shared_channel_property_names():
            grouping_property_groups = np.array([recording.get_channel_property(chan, grouping_property)
                                                 for chan in recording.get_channel_ids()])
            channel_groups = np.unique([grouping_property_groups])
        else:
            if verbose:
                print(f"{grouping_property} property is not available and it will not be saved.")
            channel_groups = [0]
            grouping_property_groups = np.array([0] * recording.get_num_channels())
    else:
        channel_groups = [0]
        grouping_property_groups = np.array([0] * recording.get_num_channels())

    n_elec = recording.get_num_channels()

    # find adjacency graph
    if graph:
        if positions is not None and adjacency_distance is not None:
            adj_graph = []
            for chg in channel_groups:
                group_graph = []
                elecs = list(np.where(grouping_property_groups == chg)[0])
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
                elecs = list(np.where(grouping_property_groups == chg)[0])
                for i in range(len(elecs)):
                    for j in range(i, len(elecs)):
                        if elecs[i] != elecs[j]:
                            group_graph.append((elecs[i], elecs[j]))
                adj_graph.append(group_graph)

    with abspath.open('w') as f:
        f.write('total_nb_channels = ' + str(n_elec) + '\n')
        if radius is not None:
            f.write('radius = ' + str(radius) + '\n')
        f.write('channel_groups = {\n')
        if len(channel_groups) > 0:
            for i_chg, chg in enumerate(channel_groups):
                f.write("     " + str(int(chg)) + ": ")
                elecs = list(np.where(grouping_property_groups == chg)[0])
                f.write("\n        {\n")
                f.write("           'channels': " + str(elecs) + ',\n')
                if graph:
                    if len(adj_graph) == 1:
                        f.write("           'graph':  " + str(adj_graph[0]) + ',\n')
                    else:
                        f.write("           'graph':  " + str(adj_graph[i_chg]) + ',\n')
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


def _check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, (int, np.integer)):
            d[k] = int(v)
        elif isinstance(v, float):
            d[k] = float(v)
        elif isinstance(v, datetime.datetime):
            d[k] = v.isoformat()

    return d


def load_extractor_from_json(json_file):
    """
    Instantiates extractor from json file

    Parameters
    ----------
    json_file: str or Path
        Path to json file

    Returns
    -------
    extractor: RecordingExtractor or SortingExtractor
        The loaded extractor object
    """
    return BaseExtractor.load_extractor_from_json(json_file)


def load_extractor_from_dict(d):
    """
    Instantiates extractor from dictionary

    Parameters
    ----------
    d: dictionary
        Python dictionary

    Returns
    -------
    extractor: RecordingExtractor or SortingExtractor
        The loaded extractor object
    """
    return BaseExtractor.load_extractor_from_dict(d)


def load_extractor_from_pickle(pkl_file):
    """
    Instantiates extractor from pickle file

    Parameters
    ----------
    pkl_file: str or Path
        Path to pickle file

    Returns
    -------
    extractor: RecordingExtractor or SortingExtractor
        The loaded extractor object
    """
    return BaseExtractor.load_extractor_from_pickle(pkl_file)


def check_get_unit_spike_train(func):
    @wraps(func)
    def check_validity(sorting, unit_id, start_frame=None, end_frame=None):
        # parse args and kwargs
        if unit_id is None:
            raise TypeError("get_unit_spike_train() missing 1 required positional argument: 'unit_id')")
        elif not (isinstance(unit_id, (int, np.integer))):
            raise ValueError("unit_id must be an integer")
        elif unit_id not in sorting.get_unit_ids():
            raise ValueError(f"{unit_id} is an invalid unit id")
        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        return func(sorting, unit_id, start_frame=start_frame, end_frame=end_frame)
    return check_validity


def check_get_traces_args(func):
    @wraps(func)
    def corrected_args(recording, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True, **kwargs):
        if channel_ids is not None:
            if isinstance(channel_ids, (int, np.integer)):
                channel_ids = list([channel_ids])
            else:
                channel_ids = channel_ids
            if np.any([ch not in recording.get_channel_ids() for ch in channel_ids]):
                print("Removing invalid 'channel_ids'",
                      [ch for ch in channel_ids if ch not in recording.get_channel_ids()])
                channel_ids = [ch for ch in channel_ids if ch in recording.get_channel_ids()]
        else:
            channel_ids = recording.get_channel_ids()
        if start_frame is not None:
            if start_frame < 0:
                start_frame = recording.get_num_frames() + start_frame
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame > recording.get_num_frames():
                print("'end_frame' set to", recording.get_num_frames())
                end_frame = recording.get_num_frames()
            elif end_frame < 0:
                end_frame = recording.get_num_frames() + end_frame
        else:
            end_frame = recording.get_num_frames()
        assert end_frame - start_frame > 0, "'start_frame' must be less than 'end_frame'!"
        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)

        if not recording.has_unscaled and not return_scaled:
            warnings.warn("The recording extractor does not have unscaled traces. Returning scaled traces")
            return_scaled = True

        traces = func(recording, channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame,
                      return_scaled=return_scaled, **kwargs)
        # scaling
        if recording.has_unscaled and return_scaled:
            channel_idxs = np.array([recording.get_channel_ids().index(ch) for ch in channel_ids])
            gains = recording.get_channel_gains()[channel_idxs, None]
            offsets = recording.get_channel_offsets()[channel_idxs, None]
            traces = (traces.astype("float32") * gains + offsets).astype("float32")

        return traces
    return corrected_args


def check_get_ttl_args(func):
    @wraps(func)
    def corrected_args(recording, start_frame=None, end_frame=None, channel_id=0, **kwargs):
        if start_frame is not None:
            if start_frame < 0:
                start_frame = recording.get_num_frames() + start_frame
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame > recording.get_num_frames():
                print("'end_frame' set to", recording.get_num_frames())
                end_frame = recording.get_num_frames()
            elif end_frame < 0:
                end_frame = recording.get_num_frames() + end_frame
        else:
            end_frame = recording.get_num_frames()
        assert end_frame - start_frame > 0, "'start_frame' must be less than 'end_frame'!"
        assert isinstance(channel_id, (int, np.integer)), "'channel_id' must be a single int"

        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        # pass recording as arg and rest as kwargs
        get_ttl_correct_arg = func(recording, start_frame=start_frame, end_frame=end_frame, channel_id=channel_id,
                                   **kwargs)
        return get_ttl_correct_arg
    return corrected_args


def cast_start_end_frame(start_frame, end_frame):
    if isinstance(start_frame, float):
        start_frame = int(start_frame)
    elif isinstance(start_frame, (int, np.integer, type(None))):
        start_frame = start_frame
    else:
        raise ValueError("start_frame must be an int, float (not infinity), or None")
    if isinstance(end_frame, float) and np.isfinite(end_frame):
        end_frame = int(end_frame)
    elif isinstance(end_frame, (int, np.integer, type(None))):
        end_frame = end_frame
    # else end_frame is infinity (accepted for get_unit_spike_train)
    if start_frame is not None:
        start_frame = int(start_frame)
    if end_frame is not None and np.isfinite(end_frame):
        end_frame = int(end_frame)
    return start_frame, end_frame


def divide_recording_into_time_chunks(num_frames, chunk_size, padding_size):
    chunks = []
    ii = 0
    while ii < num_frames:
        ii2 = int(min(ii + chunk_size, num_frames))
        chunks.append(dict(
            istart=ii,
            iend=ii2,
            istart_with_padding=int(max(0, ii - padding_size)),
            iend_with_padding=int(min(num_frames, ii2 + padding_size))
        ))
        ii = ii2
    return chunks


def _write_dat_one_chunk(i, rec_arg, chunks, rec_memmap, dtype, time_axis, return_scaled, verbose):
    chunk = chunks[i]

    if verbose:
        print(f"Writing chunk {i + 1} / {len(chunks)}")
    if isinstance(rec_arg, dict):
        recording = load_extractor_from_dict(rec_arg)
    else:
        recording = rec_arg

    start_frame = chunk['istart']
    end_frame = chunk['iend']
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, return_scaled=return_scaled)
    if dtype is not None:
        traces = traces.astype(dtype)
    if time_axis == 0:
        traces = traces.T
        rec_memmap[start_frame:end_frame, :] = traces
    else:
        rec_memmap[:, start_frame:end_frame] = traces
