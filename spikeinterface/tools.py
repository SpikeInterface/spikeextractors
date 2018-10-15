import numpy as np
from .RecordingExtractor import RecordingExtractor
import os
import os.path as op
import csv


def read_python(path):
    '''Parses python scripts in a dictionary

    Parameters
    ----------
    path: str
        Path to file to parse

    Returns
    -------
    metadata:
        dictionary containing parsed file

    '''
    from six import exec_
    path = os.path.realpath(os.path.expanduser(path))
    assert os.path.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def loadProbeFile(recording, probefile):
    '''Loads channel information in recording extractor

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to channel information
    probefile: str
        Path to probe file. Either .prb or .csv
    '''
    if probefile.endswith('.prb'):
        probe_dict = read_python(probefile)
        if 'channel_groups' in probe_dict.keys():
            numchannels = np.sum([len(cg['channels']) for key, cg in probe_dict['channel_groups'].items()])
            assert numchannels ==  recording.getNumChannels()

            for cgroup_id, cgroup in probe_dict['channel_groups'].items():
                for key_prop, prop_val in cgroup.items():
                    if key_prop == 'channels':
                        # if len(prop_val) == recording.getNumChannels():
                        for i_ch, prop in enumerate(prop_val):
                            recording.setChannelProperty(prop, 'group', int(cgroup_id))
                    elif key_prop == 'geometry':
                        # if len(prop_val) == recording.getNumChannels():
                        for (i_ch, prop) in prop_val.items():
                            recording.setChannelProperty(i_ch, 'location', prop)
        else:
            raise AttributeError("'.prb' file should contain the 'channel_groups' field")
    elif probefile.endswith('.csv'):
        with open(probefile) as csvfile:
            posreader = csv.reader(csvfile)
            for i_ch, pos in enumerate(posreader):
                recording.setChannelProperty(i_ch, 'location', pos)
    else:
        raise NotImplementedError("Only .csv and .prb probe files can be loaded.")


def saveProbeFile(recording, probefile, format=None):
    '''Saves probe file from channel information of a recording extractor

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to save probe file from
    probefile: str
        filename of .prb or .csv file to save probe information to
    format: str (optional)
        Format for .prb file. It can be either 'klusta' or 'spyking_circus'. Default is None.
    '''
    probefile = os.path.abspath(probefile)
    if not os.path.isdir(os.path.dirname(probefile)):
        os.makedirs(os.path.dirname(probefile))

    if probefile.endswith('.csv'):
        # write csv probe file
        with open(probefile, 'w') as f:
            if 'location' in recording.getChannelPropertyNames():
                for chan in range(recording.getNumChannels()):
                    loc = recording.getChannelProperty(chan, 'location')
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
    elif probefile.endswith('.prb'):
        _export_prb_file(recording, probefile, format)
    else:
        raise NotImplementedError("Only .csv and .prb probe files can be saved.")


def _export_prb_file(recording, filename, format=None, adjacency_distance=None, graph=False, geometry=True, radius=100):
    '''Exports .prb file

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to save probe file from
    filename: str
        probe filename
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

    assert filename is not None
    abspath = os.path.abspath(filename)

    if geometry:
        if 'location' in recording.getChannelPropertyNames():
            positions = np.array([recording.getChannelProperty(chan, 'location')
                                  for chan in range(recording.getNumChannels())])
        else:
            print("'location' property is not available and it will not be saved.")
            positions = None
            geometry = False
    else:
        positions = None

    if 'group' in recording.getChannelPropertyNames():
        groups = np.array([recording.getChannelProperty(chan, 'group') for chan in range(recording.getNumChannels())])
        channel_groups = np.unique([groups])
    else:
        print("'group' property is not available and it will not be saved.")
        channel_groups = [0]
        groups = np.array([0] * recording.getNumChannels())

    n_elec = recording.getNumChannels()

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
                                group_graph.append((elecs[i],  elecs[j]))
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
                            group_graph.append((elecs[i],  elecs[j]))
                adj_graph.append(group_graph)

    with open(filename, 'w') as f:
        f.write('\n')
        if format=='spyking_circus':
            f.write('total_nb_channels = ' + str(n_elec) + '\n')
            f.write('radius = ' + str(radius) + '\n')
        f.write('channel_groups = {\n')
        if len(channel_groups) > 0:
            for i_chg, chg in enumerate(channel_groups):
                f.write("     " + str(int(chg)) + ": ")
                elecs = list(np.where(groups==chg)[0])
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
                        f.write('               ' + str(elecs[i]) +': ' + str(tuple(pos)) + ',\n')
                    f.write('           }\n')
                f.write('       }\n}')
                f.write('\n')
        else:
            for elec in range(n_elec):
                f.write('    ' + str(elec) + ': ')
                f.write("\n        {\n")
                f.write("           'channels': [" + str(elec) + '],\n')
                f.write("           'graph':  [],\n")
                f.write('        },\n')
            f.write('}\n')