import numpy as np

try:
    from sonpy import lib as sp

    # Data storage and function finder
    DataReadFunctions = {
        sp.DataType.Adc: sp.SonFile.ReadInts,
        sp.DataType.EventFall: sp.SonFile.ReadEvents,
        sp.DataType.EventRise: sp.SonFile.ReadEvents,
        sp.DataType.EventBoth: sp.SonFile.ReadEvents,
        sp.DataType.Marker: sp.SonFile.ReadMarkers,
        sp.DataType.AdcMark: sp.SonFile.ReadWaveMarks,
        sp.DataType.RealMark: sp.SonFile.ReadRealMarks,
        sp.DataType.TextMark: sp.SonFile.ReadTextMarks,
        sp.DataType.RealWave: sp.SonFile.ReadFloats
    }
except:
    pass

# Get the saved time and date
# f.GetTimeDate()


def get_channel_info(f, smrx_ch_ind):
    """
    Extract info from smrx files

    Parameters:
    -----------
    f: str
        SonFile object.
    smrx_ch_ind: int
        Index of smrx channel. Does not match necessarily with extractor id.
    """

    nMax = 1 + int(f.ChannelMaxTime(smrx_ch_ind) / f.ChannelDivide(smrx_ch_ind))
    frame_offset = f.FirstTime(chan=smrx_ch_ind, tFrom=0, tUpto=nMax) / f.ChannelDivide(smrx_ch_ind)
    ch_info = {
        'type': f.ChannelType(smrx_ch_ind),           # Get the channel kind
        'ch_number': f.PhysicalChannel(smrx_ch_ind),  # Get the physical channel number associated with this channel
        'title': f.GetChannelTitle(smrx_ch_ind),      # Get the channel title
        'ideal_rate': f.GetIdealRate(smrx_ch_ind),    # Get the requested channel ideal rate
        'rate': 1 / (f.GetTimeBase() * f.ChannelDivide(smrx_ch_ind)),    # Get the requested channel real rate
        'max_time': f.ChannelMaxTime(smrx_ch_ind),    # Get the time of the last item in the channel (in clock ticks)
        'divide': f.ChannelDivide(smrx_ch_ind),       # Get the waveform sample interval in file clock ticks
        'time_base': f.GetTimeBase(),                 # Get how many seconds there are per clock tick
        'frame_offset': frame_offset,                 # Get frame offset
        'scale': f.GetChannelScale(smrx_ch_ind),      # Get the channel scale
        'offset': f.GetChannelOffset(smrx_ch_ind),    # Get the channel offset
        'unit': f.GetChannelUnits(smrx_ch_ind),       # Get the channel units
        'y_range': f.GetChannelYRange(smrx_ch_ind),   # Get a suggested Y range for the channel
        'comment': f.GetChannelComment(smrx_ch_ind),  # Get the comment associated with a channel
        'size_bytes:': f.ChannelBytes(smrx_ch_ind),   # Get an estimate of the data bytes stored for the channel
    }

    return ch_info


def get_channel_data(f, smrx_ch_ind, start_frame=0, end_frame=None):
    """
    Extract info from smrx files

    Parameters:
    -----------
    f: str
        SonFile object.
    smrx_ch_ind: int
        Index of smrx channel. Does not match necessarily with extractor id.
    start_frame: int
        The starting frame of the trace to be returned (inclusive).
    end_frame: int
        The ending frame of the trace to be returned (exclusive).
    """

    if end_frame is None:
        end_frame = 1 + int(f.ChannelMaxTime(smrx_ch_ind) / f.ChannelDivide(smrx_ch_ind))

    nMax = 1 + int(f.ChannelMaxTime(smrx_ch_ind) / f.ChannelDivide(smrx_ch_ind))
    frame_offset = int(f.FirstTime(chan=smrx_ch_ind, tFrom=0, tUpto=nMax) / f.ChannelDivide(smrx_ch_ind))
    start_frame += frame_offset
    end_frame += frame_offset

    data = DataReadFunctions[f.ChannelType(smrx_ch_ind)](
        self=f,
        chan=smrx_ch_ind,
        nMax=nMax,
        tFrom=int(start_frame * f.ChannelDivide(smrx_ch_ind)),
        tUpto=int(end_frame * f.ChannelDivide(smrx_ch_ind))
    )

    return np.array(data)