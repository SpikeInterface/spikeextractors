import numpy as np
from .RecordingExtractor import RecordingExtractor
import os.path as op


def read_python(path):
    from six import exec_
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata

def loadProbeFile(recording, probefile, format='auto'):
    '''

    Parameters
    ----------
    recording
    probefile

    Returns
    -------

    '''
    if format == 'auto':
        if probefile.endswith('.prb'):
            probe_dict = read_python(probefile)
            if 'channel_groups' in probe_dict.keys():
                numchannels = np.sum([len(cg['channels']) for key, cg in probe_dict['channel_groups'].items()])
                assert numchannels ==  recording.getNumChannels()
            else:
                raise AttributeError("'.prb' file should contain the 'channel_groups' field")
        elif probefile.endswith('.m'):
            pass
        elif probefile.endswith('.csv'):
            pass

    for chan in range(recording.getNumChannels()):
        recording.setProperty(chan, 'group', 1)

def saveProbeFile(recording, probefile):
    pass