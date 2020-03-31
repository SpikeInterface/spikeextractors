from spikeextractors import RecordingExtractor
from .readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_traces_args


class SpikeGLXRecordingExtractor(RecordingExtractor):
    extractor_name = 'SpikeGLXRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path, x_pitch=21, y_pitch=20):
        RecordingExtractor.__init__(self)
        self._npxfile = Path(file_path)
        self._basepath = self._npxfile.parents[0]

        # Gets file type: 'imec0.ap', 'imec0.lf' or 'nidq'
        if 'ap' in str(self._npxfile):
            self.is_filtered = True
        aux = self._npxfile.stem.split('.')[-1]
        if aux == 'nidq':
            self._ftype = aux
        else:
            self._ftype = self._npxfile.stem.split('.')[-2] + '.' + aux

        # Metafile
        self._metafile = self._basepath.joinpath(self._npxfile.stem+'.meta')
        if not self._metafile.exists():
            raise Exception("'meta' file for '"+self._ftype+"' traces should be in the same folder.")
        # Read in metadata, returns a dictionary
        meta = readMeta(self._npxfile)

        # Traces in 16-bit format
        rawData = makeMemMapRaw(self._npxfile, meta)
        self._timeseries = rawData  # [chanList, firstSamp:lastSamp+1]

        # sampling rate and ap channels
        self._sampling_frequency = SampRate(meta)
        tot_chan, ap_chan, locations = _parse_spikeglx_metafile(self._metafile, x_pitch, y_pitch)
        if ap_chan < tot_chan:
            self._channels = list(range(int(ap_chan)))
            self._timeseries = self._timeseries[0:ap_chan, :]
        else:
            self._channels = list(range(int(tot_chan)))  # OriginalChans(meta).tolist()

        # locations
        if len(locations) > 0:
           for m in range(len(self._channels)):
               self.set_channel_property(m, 'location', locations[m])

        # get gains
        if meta['typeThis'] == 'imec':
            gains = GainCorrectIM(self._timeseries, self._channels, meta)
        elif meta['typeThis'] =='nidq':
            gains = GainCorrectNI(self._timeseries, self._channels, meta)

        # set gains - convert from int16 to uVolt
        self.set_channel_gains(self._channels, gains*1e6)
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'x_pitch': x_pitch, 'y_pitch': y_pitch}


    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        channel_idxs = [self._channels.index(ch) for ch in channel_ids]
        recordings = self._timeseries[channel_idxs, start_frame:end_frame]
        return recordings

    @staticmethod
    def write_recording(recording, save_path, dtype=None, transpose=False):
        save_path = Path(save_path)
        if dtype is None:
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
