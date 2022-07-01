from .readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
import numpy as np
from pathlib import Path

from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args, check_get_ttl_args
import re

class SpikeGLXRecordingExtractor(RecordingExtractor):
    """
    RecordingExtractor from a SpikeGLX Neuropixels file

    Parameters
    ----------
    file_path: str or Path
        Path to the ap.bin, lf.bin, or nidq.bin file
    dtype: str
        'int16' or 'float'. If 'float' is selected, the returned traces are converted to uV
    x_pitch: int
        The x pitch of the probe (default 16)
    y_pitch: int
        The y pitch of the probe (default 20)
    """
    extractor_name = 'SpikeGLXRecording'
    has_default_locations = True
    has_unscaled = True
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the SpikeGLXRecordingExtractor run:\n\n pip install mtscomp\n\n"  # error message when not installed

    def __init__(self, file_path: str, x_pitch: int = 32, y_pitch: int = 20):
        RecordingExtractor.__init__(self)
        self._npxfile = Path(file_path)
        self._basepath = self._npxfile.parents[0]
        
        # Gets file type: 'imec0.ap', 'imec0.lf' or 'nidq'
        assert re.search(r'imec[0-9]*.(ap|lf){1}.bin$', self._npxfile.name) or  'nidq' in self._npxfile.name, \
               "'file_path' can be an imec.ap, imec.lf, imec0.ap, imec0.lf, or nidq file"
           
        if 'ap.bin' in str(self._npxfile):
            rec_type = "ap"
            self.is_filtered = True
        elif 'lf.bin' in str(self._npxfile):
            rec_type = "lf"
        else:
            rec_type = "nidq"
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
        self._raw = makeMemMapRaw(self._npxfile, meta)  # [chanList, firstSamp:lastSamp+1]

        # sampling rate and ap channels
        self._sampling_frequency = SampRate(meta)

        tot_chan, ap_chan, lfp_chan, locations, channel_ids, channel_names \
            = _parse_spikeglx_metafile(self._metafile,
                                       x_pitch=x_pitch,
                                       y_pitch=y_pitch,
                                       rec_type=rec_type)
        if rec_type in ("ap", "lf"):
            self._channels = channel_ids
            # locations
            if len(locations) > 0:
                self.set_channel_locations(locations)
            if len(channel_names) > 0:
                if len(channel_names) == len(self._channels):
                    for i, ch in enumerate(self._channels):
                        self.set_channel_property(ch, "channel_name", channel_names[i])

            if rec_type == "ap":
                if ap_chan < tot_chan:
                    self._timeseries = self._raw[0:ap_chan, :]
            elif rec_type == "lf":
                if lfp_chan < tot_chan:
                    self._timeseries = self._raw[0:lfp_chan, :]
        else:
            # nidq
            self._channels = list(range(int(tot_chan)))
            self._timeseries = self._raw

        # get gains
        if meta['typeThis'] == 'imec':
            gains = GainCorrectIM(self._timeseries, self._channels, meta)
        elif meta['typeThis'] == 'nidq':
            gains = GainCorrectNI(self._timeseries, self._channels, meta)

        # set gains - convert from int16 to uVolt
        self.set_channel_gains(gains=gains*1e6, channel_ids=self._channels)
        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'x_pitch': x_pitch, 'y_pitch': y_pitch}

    def get_channel_ids(self):
        return self._channels

    def get_num_frames(self):
        return self._timeseries.shape[1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
        if np.array_equal(channel_ids, self.get_channel_ids()):
            traces = self._timeseries[:, start_frame:end_frame]
        else:
            if np.all(np.diff(channel_idxs) == 1):
                traces = self._timeseries[channel_idxs[0]:channel_idxs[0]+len(channel_idxs), start_frame:end_frame]
            else:
                # This block of the execution will return the data as an array, not a memmap
                traces = self._timeseries[channel_idxs, start_frame:end_frame]

        return traces

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


def _parse_spikeglx_metafile(metafile, x_pitch, y_pitch, rec_type):
    tot_channels = None
    ap_channels = None
    lfp_channels = None

    y_offset = 20
    x_offset = 11

    locations = []
    channel_names = []
    channel_ids = []
    with Path(metafile).open() as f:
        for line in f.readlines():
            if 'nSavedChans' in line:
                tot_channels = int(line.split('=')[-1])
            if 'snsApLfSy' in line:
                ap_channels = int(line.split('=')[-1].split(',')[0].strip())
                lfp_channels = int(line.split(',')[-2].strip())
            if 'imSampRate' in line:
                fs = float(line.split('=')[-1])
            if rec_type in ("ap", "lf"):
                if 'snsChanMap' in line:
                    map = line.split('=')[-1]
                    chans = map.split(')')[1:]
                    for chan in chans:
                        chan_name = chan[1:].split(';')[0]
                        if rec_type == "ap":
                            if "AP" in chan_name:
                                channel_names.append(chan_name)
                                chan_id = int(chan_name[2:])
                                channel_ids.append(chan_id)
                        elif rec_type == "lf":
                            if "LF" in chan_name:
                                channel_names.append(chan_name)
                                chan_id = int(chan_name[2:])
                                channel_ids.append(chan_id)
                if 'snsShankMap' in line:
                    map = line.split('=')[-1]
                    chans = map.split(')')[1:]
                    for chan in chans:
                        chan = chan[1:]
                        if len(chan) > 0:
                            x_idx = int(chan.split(':')[1])
                            y_idx = int(chan.split(':')[2])
                            stagger = np.mod(y_idx + 0, 2) * x_pitch / 2
                            x_pos = (1 - x_idx) * x_pitch + stagger + x_offset
                            y_pos = y_idx * y_pitch + y_offset
                            locations.append([x_pos, y_pos])
    return tot_channels, ap_channels, lfp_channels, locations, channel_ids, channel_names
