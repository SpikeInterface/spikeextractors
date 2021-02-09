from spikeextractors import RecordingExtractor
from .readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args


class SpikeGLXRecordingExtractor(RecordingExtractor):
    extractor_name = 'SpikeGLXRecordingExtractor'
    has_default_locations = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the SpikeGLXRecordingExtractor run:\n\n pip install mtscomp\n\n"  # error message when not installed

    def __init__(self, file_path: str, dtype: str = 'int16', x_pitch: int = 16, y_pitch: int = 20):
        RecordingExtractor.__init__(self)
        self._npxfile = Path(file_path)
        self._basepath = self._npxfile.parents[0]

        assert dtype in ['int16', 'float'], "'dtype' can be either 'int16' or 'float'"
        self._dtype = dtype
        # Gets file type: 'imec0.ap', 'imec0.lf' or 'nidq'
        assert 'imec0.ap' in self._npxfile.name or  'imec0.lf' in self._npxfile.name or 'nidq' in self._npxfile.name, \
            "'file_path' can be an imec0.ap, imec.lf, or nidq file"
        assert 'bin' in self._npxfile.name, "The 'npx_file should be either the 'ap' or the 'lf' bin file."
        if 'imec0.ap' in str(self._npxfile):
            lfp = False
            ap = True
            self.is_filtered = True
        elif 'imec0.lf' in str(self._npxfile):
            lfp = True
            ap = False
        else:
            lfp = False
            ap = False
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
        self._meta = meta

        # Traces in 16-bit format
        if '.cbin' in self._npxfile.name: # compressed binary format used by IBL
            try:
                import mtscomp
            except:
                raise Exception(self.installation_mesg)
            self._raw = mtscomp.Reader()
            self._raw.open(self._npxfile, self._npxfile.with_suffix('.ch'))
        else:
            self._raw = makeMemMapRaw(self._npxfile, meta)  # [chanList, firstSamp:lastSamp+1]

        # sampling rate and ap channels
        self._sampling_frequency = SampRate(meta)
        tot_chan, ap_chan, lfp_chan, locations = _parse_spikeglx_metafile(self._metafile,
                                                                          x_pitch=x_pitch,
                                                                          y_pitch=y_pitch)
        if ap:
            if ap_chan < tot_chan:
                self._channels = list(range(int(ap_chan)))
                self._timeseries = self._raw[0:ap_chan, :]
            else:
                self._channels = list(range(int(tot_chan)))  # OriginalChans(meta).tolist()
        elif lfp:
            if lfp_chan < tot_chan:
                self._channels = list(range(int(lfp_chan)))
                self._timeseries = self._raw[0:lfp_chan, :]
            else:
                self._channels = list(range(int(tot_chan)))
        else:
            # nidq
            self._channels = list(range(int(tot_chan)))
            self._timeseries = self._raw

        # locations
        if len(locations) > 0:
            self.set_channel_locations(locations)

        # get gains
        if meta['typeThis'] == 'imec':
            gains = GainCorrectIM(self._timeseries, self._channels, meta)
        elif meta['typeThis'] == 'nidq':
            gains = GainCorrectNI(self._timeseries, self._channels, meta)

        # set gains - convert from int16 to uVolt
        self.set_channel_gains(gains=gains*1e6, channel_ids=self._channels)
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'dtype': dtype,
                         'x_pitch': x_pitch, 'y_pitch': y_pitch}

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, dtype=None):
        channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
        if np.all(channel_ids == self.get_channel_ids()):
            recordings = self._timeseries[:, start_frame:end_frame]
        else:
            if np.all(np.diff(channel_idxs) == 1):
                recordings = self._timeseries[channel_idxs[0]:channel_idxs[0]+len(channel_idxs), start_frame:end_frame]
            else:
                # This block of the execution will return the data as an array, not a memmap
                recordings = self._timeseries[channel_idxs, start_frame:end_frame]
        if dtype is not None:
            assert dtype in ['int16', 'float'], "'dtype' can be either 'int16' or 'float'"
        else:
            dtype = self._dtype
        if dtype == 'int16':
            return recordings
        else:
            gains = np.array(self.get_channel_gains())[channel_idxs]
            return recordings * gains[:, None]

    @check_get_ttl_args
    def get_ttl_events(self, start_frame=None, end_frame=None, channel_id=0):
        channel = [channel_id]
        dw = 0
        dig = ExtractDigital(self._raw, firstSamp=start_frame, lastSamp=end_frame, dwReq=dw, dLineList=channel,
                             meta=self._meta)
        dig = np.squeeze(dig)
        diff_dig = np.diff(dig.astype(int))

        rising = np.where(diff_dig > 0)[0] + start_frame
        falling = np.where(diff_dig < 0)[0] + start_frame

        ttl_frames = np.concatenate((rising, falling))
        ttl_states = np.array([1] * len(rising) + [-1] * len(falling))
        sort_idxs = np.argsort(ttl_frames)
        return ttl_frames[sort_idxs], ttl_states[sort_idxs]


def _parse_spikeglx_metafile(metafile, x_pitch, y_pitch):
    tot_channels = None
    ap_channels = None
    lfp_channels = None

    locations = []
    with Path(metafile).open() as f:
        for line in f.readlines():
            if 'nSavedChans' in line:
                tot_channels = int(line.split('=')[-1])
            if 'snsApLfSy' in line:
                ap_channels = int(line.split('=')[-1].split(',')[0].strip())
                lfp_channels = int(line.split(',')[-2].strip())
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
    return tot_channels, ap_channels, lfp_channels, locations
