from spikeextractors import RecordingExtractor
import os
import numpy as np
from pathlib import Path


class SpikeGLXRecordingExtractor(RecordingExtractor):

    extractor_name = 'SpikeGLXRecordingExtractor'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'datfile', 'type': 'str', 'title': "Path to file"},
        {'name': 'samplerate', 'type': 'float', 'title': "Sampling rate in HZ"},
        {'name': 'numchan', 'type': 'int', 'title': "Number of channels"},
        {'name': 'dtype', 'type': 'np.dtype', 'title': "The dtype of underlying data"},
        {'name': 'recording_channels', 'type': 'list', 'value':None, 'default':None, 'title': "List of recording channels"},
        {'name': 'frames_first', 'type': 'bool', 'value':True, 'default':True, 'title': "Frames first"},
        {'name': 'offset', 'type': 'int', 'value':0, 'default':0, 'title': "Offset in binary file"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, npx_file):
        RecordingExtractor.__init__(self)
        self._npxfile = Path(npx_file)
        numchan = 385
        dtype = 'int16'
        root = str(self._npxfile.stem).split('.')[0]
        # find metafile in same folder
        metafile = [x for x in self._npxfile.parent.iterdir() if 'meta' in str(x)
                    and root in str(x) and 'ap' in str(x)]
        if len(metafile) == 0:
            raise Exception("'meta' file for ap traces should be in the same folder.")
        else:
            metafile = metafile[0]
        tot_chan, ap_chan, samplerate, locations = _parse_spikeglx_metafile(metafile)
        frames_first = True
        self._timeseries = _read_binary(self._npxfile, tot_chan, dtype, frames_first, offset=0)
        self._samplerate = float(samplerate)

        if ap_chan < tot_chan:
            self._timeseries = self._timeseries[:ap_chan]
        self._channels = list(range(self._timeseries.shape[0]))

        if len(locations) > 0:
            for m in range(self._timeseries.shape[0]):
                self.set_channel_property(m, 'location', locations[m])

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._samplerate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = list(range(self._timeseries.shape[0]))
        else:
            channel_ids = [self._channels.index(ch) for ch in channel_ids]
        recordings = self._timeseries[:, start_frame:end_frame][channel_ids, :]
        return recordings

    @staticmethod
    def write_recording(recording, save_path, dtype=None, transpose=False):
        save_path = Path(save_path)
        if dtype == None:
            dtype = np.float32
        if not transpose:
            with save_path.open('wb') as f:
                np.transpose(np.array(recording.get_traces(), dtype=dtype)).tofile(f)
        elif transpose:
            with save_path.open('wb') as f:
                np.array(recording.get_traces(), dtype=dtype).tofile(f)


def _read_binary(file, numchan, dtype, frames_first, offset):
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


def _parse_spikeglx_metafile(metafile):
    tot_channels = None
    ap_channels = None
    x_pitch = 21
    y_pitch = 20
    locations = []
    with Path(metafile).open() as f:
        for line in f.readlines():
            if 'nSavedChans' in line:
                tot_channels = int(line.split('=')[-1])
            if 'snsApLfSy' in line:
                ap_channels = int(line.split('=')[-1].split(',')[0].strip())
            if 'imSampRate' in line:
                fs = float(line.split('=')[-1])
            if 'snsShankMap' in line:
                map = line.split('=')[-1]
                chans = map.split(')')[1:]
                for chan in chans:
                    chan = chan[1:]
                    if len(chan) > 0:
                        x_pos = int(chan.split(':')[1])
                        y_pos = int(chan.split(':')[2])
                        locations.append([x_pos*x_pitch, y_pos*y_pitch])
    return tot_channels, ap_channels, fs, locations