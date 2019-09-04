from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import read_binary
from .readSGLX import (readMeta, SampRate, makeMemMapRaw, Int2Volts,
    GainCorrectIM, GainCorrectNI)
import os
import numpy as np
from pathlib import Path


class SpikeGLXRecordingExtractor(RecordingExtractor):

    extractor_name = 'SpikeGLXRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'npx_file', 'type': 'path', 'title': "Path to file"},
        {'name': 'x_pitch', 'type': 'float', 'value':21.0, 'default':21.0, 'title': "x_pitch for Neuropixels probe (default 21)"},
        {'name': 'y_pitch', 'type': 'float', 'value':20.0, 'default':20.0, 'title': "y_pitch for Neuropixels probe (default 20)"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, npx_file, x_pitch=None, y_pitch=None):
        RecordingExtractor.__init__(self)
        dtype = 'int16'

        self._npxfile = Path(npx_file)
        self._basepath = self._npxfile.cwd()

        # Gets file type: 'imec0.ap', 'imec0.lf' or 'nidq'
        aux = self._npxfile.stem.split('.')[-1]
        if aux == 'nidq':
            self._ftype = aux
        else:
            self._ftype = self._npxfile.stem.split('.')[-2] + '.' + aux

        # Metafile
        self._metafile = self._npxfile.cwd().joinpath(self._npxfile.stem+'.meta')
        if not self._metafile.exists():
            raise Exception("'meta' file for '"+self._ftype+"' traces should be in the same folder.")

        # Read in metadata, returns a dictionary
        meta = readMeta(self._npxfile)

        # parameters common to NI and imec data
        self._samplerate = SampRate(meta)
        self._channels = list(range(int(meta['nSavedChans']))) #OriginalChans(meta).tolist()

        # Traces in 16-bit format
        rawData = makeMemMapRaw(self._npxfile, meta)
        selectData = rawData#[chanList, firstSamp:lastSamp+1]

        # apply gain correction
        if meta['typeThis'] == 'imec':
            convArray, gains = GainCorrectIM(selectData, self._channels, meta)
        elif meta['typeThis'] =='nidq':
            convArray, gains = GainCorrectNI(selectData, self._channels, meta)

        # set gains
        self.set_channel_gains(self._channels, gains)

        # convert to uV
        self._timeseries = 1e6*convArray

        
        ##------ OLD CODE - LEFT FOR REFERENCE ------##
        # frames_first = True
        # self._timeseries = read_binary(self._npxfile, tot_chan, dtype, frames_first, offset=0)
        # self._samplerate = float(samplerate)
        #if ap_chan < tot_chan:
        #    self._timeseries = self._timeseries[:ap_chan]
        #self._channels = list(range(self._timeseries.shape[0]))
        # tot_chan, ap_chan, locations = _parse_spikeglx_metafile(self._metafile, x_pitch, y_pitch)
        # if len(locations) > 0:
        #    for m in range(len(self._channels)):
        #        self.set_channel_property(m, 'location', locations[m])
        ##-------------------------------------------##

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

    def get_conversion_factor(self):
        return self._conversion_factor

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


def _parse_spikeglx_metafile(metafile, x_pitch, y_pitch):
    tot_channels = None
    ap_channels = None
    if x_pitch is None:
        x_pitch = 21
    if y_pitch is None:
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
    return tot_channels, ap_channels, locations
