from spikeextractors import RecordingExtractor
import numpy as np
from pathlib import Path
from spikeextractors.extraction_tools import check_get_traces_args

try:
    import h5py

    HAVE_MCSH5 = True
except ImportError:
    HAVE_MCSH5 = False


class MCSH5RecordingExtractor(RecordingExtractor):
    extractor_name = 'MCSH5Recording'
    has_default_locations = False
    has_unscaled = False
    installed = HAVE_MCSH5  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the MCSH5RecordingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path, stream_id=0, verbose=False):
        assert self.installed, self.installation_mesg
        self._recording_file = file_path
        self._verbose = verbose
        self._available_stream_ids = self.get_available_stream_ids()
        self.set_stream_id(stream_id)

        RecordingExtractor.__init__(self)
        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'stream_id': stream_id,
                        'verbose': verbose}

    def __del__(self):
        self._rf.close()

    def get_channel_ids(self):
        return list(self._channel_ids)

    def get_num_frames(self):
        return self._nFrames

    def get_sampling_frequency(self):
        return self._samplingRate

    def set_stream_id(self, stream_id):
        assert stream_id in self._available_stream_ids, "The specified stream ID is unavailable."
        self._stream_id = stream_id

        if hasattr(self, '_rf'):
            self._rf.close()

        self._rf, self._nFrames, self._samplingRate, self._nRecCh, \
        self._channel_ids, self._electrodeLabels, self._exponent, self._convFact \
            = openMCSH5File(self._recording_file, stream_id, self._verbose)

    def get_stream_id(self):
        assert hasattr(self, '_stream_id'), "Stream ID has not been set yet."
        return self._stream_id

    def get_available_stream_ids(self):
        if hasattr(self, '_available_stream_ids'):
            return self._available_stream_ids
        else:
            rf = h5py.File(self._recording_file, 'r')
            analog_stream_names = list(rf.require_group('/Data/Recording_0/AnalogStream').keys())
            return list(range(len(analog_stream_names)))

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        channel_idxs = []
        for m in channel_ids:
            assert m in self._channel_ids, 'channel_id {} not found'.format(m)
            channel_idxs.append(np.where(np.array(self._channel_ids) == m)[0][0])

        stream = self._rf.require_group('/Data/Recording_0/AnalogStream/Stream_' + str(self._stream_id))
        conv = self._convFact.astype(float) * (10.0 ** self._exponent)

        if np.array(channel_idxs).size > 1:
            if np.any(np.diff(channel_idxs) < 0):
                sorted_channel_ids = np.sort(channel_idxs)
                sorted_idx = np.array([list(sorted_channel_ids).index(ch) for ch in channel_idxs])
                signals = stream.get('ChannelData')[sorted_channel_ids, start_frame:end_frame][sorted_idx]
            else:
                signals =  stream.get('ChannelData')[np.sort(channel_idxs), start_frame:end_frame]
        else:
            signals = stream.get('ChannelData')[np.array(channel_idxs), start_frame:end_frame]
        if return_scaled:
            return signals * conv
        else:
            return signals


def openMCSH5File(filename, stream_id, verbose=False):
    """Open an MCS hdf5 file, read and return the recording info."""
    rf = h5py.File(filename, 'r')

    stream_name = 'Stream_' + str(stream_id)
    analog_stream_names = list(rf.require_group('/Data/Recording_0/AnalogStream').keys())
    assert stream_name in analog_stream_names, "Specified stream does not exist."

    stream = rf.require_group('/Data/Recording_0/AnalogStream/' + stream_name)
    data = np.array(stream.get('ChannelData'), dtype=np.int)
    timestamps = np.array(stream.get('ChannelDataTimeStamps'))
    info = np.array(stream.get('InfoChannel'))

    Unit = info['Unit'][0]
    Tick = info['Tick'][0] / 1e6
    exponent = info['Exponent'][0]
    convFact = info['ConversionFactor'][0]

    nRecCh, nFrames = data.shape
    channel_ids = info['ChannelID']
    assert len(np.unique(channel_ids)) == len(channel_ids), 'Duplicate MCS channel IDs found'
    electrodeLabels = info['Label']

    assert timestamps[0][0] < timestamps[0][2], 'Please check the validity of \'ChannelDataTimeStamps\' in the stream.'
    TimeVals = np.arange(timestamps[0][0], timestamps[0][2] + 1, 1) * Tick

    assert Unit == b'V', 'Unexpected units found, expected volts, found {}'.format(Unit.decode('UTF-8'))
    data_V = data * convFact.astype(float) * (10.0 ** (exponent))

    timestep_avg = np.mean(TimeVals[1:] - TimeVals[0:-1])
    timestep_std = np.std(TimeVals[1:] - TimeVals[0:-1])
    timestep_min = np.min(TimeVals[1:] - TimeVals[0:-1])
    timestep_max = np.min(TimeVals[1:] - TimeVals[0:-1])
    assert all(np.abs(np.array(
        (timestep_min, timestep_max)) - timestep_avg) / timestep_avg < 1e-6), 'Time steps vary by more than 1 ppm'
    samplingRate = 1. / timestep_avg

    if verbose:
        print('# MCS H5 data format')
        print('#')
        print('# File: {}'.format(rf.filename))
        print('# File size: {:.2f} MB'.format(rf.id.get_filesize() / 1024 ** 2))
        print('#')
        for key in rf.attrs.keys():
            print('# {}: {}'.format(key, rf.attrs[key]))
        print('#')
        print('# Signal range: {:.2f} to {:.2f} µV'.format(np.amin(data_V) * 1e6, np.amax(data_V) * 1e6))
        print('# Number of channels: {}'.format(nRecCh))
        print('# Number of frames: {}'.format(nFrames))
        print('# Time step: {:.2f} µs ± {:.5f} % (range {} to {})'.format(timestep_avg * 1e6,
                                                                          timestep_std / timestep_avg * 100,
                                                                          timestep_min * 1e6, timestep_max * 1e6))
        print('# Sampling rate: {:.2f} Hz'.format(samplingRate))
        print('#')
        print('# MCSH5RecordingExtractor currently only reads /Data/Recording_0/AnalogStream/Stream_0')

    return rf, nFrames, samplingRate, nRecCh, channel_ids, electrodeLabels, exponent, convFact
