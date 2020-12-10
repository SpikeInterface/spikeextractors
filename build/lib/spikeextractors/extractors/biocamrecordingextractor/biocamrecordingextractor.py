from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np
from pathlib import Path
import ctypes

try:
    import h5py

    HAVE_BIOCAM = True
except ImportError:
    HAVE_BIOCAM = False


class BiocamRecordingExtractor(RecordingExtractor):
    extractor_name = 'BiocamRecording'
    has_default_locations = True
    installed = HAVE_BIOCAM  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the BiocamRecordingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path, verbose=False, mea_pitch=42):
        assert HAVE_BIOCAM, self.installation_mesg
        self._mea_pitch = mea_pitch
        self._recording_file = file_path
        self._rf, self._nFrames, self._samplingRate, self._nRecCh, self._chIndices, \
        self._file_format, self._signalInv, self._positions, self._read_function = openBiocamFile(
            self._recording_file, self._mea_pitch, verbose)
        RecordingExtractor.__init__(self)
        self.set_channel_locations(self._positions)

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'mea_pitch': mea_pitch,
                        'verbose': verbose}

    def __del__(self):
        self._rf.close()

    def get_channel_ids(self):
        return list(range(self._nRecCh))

    def get_num_frames(self):
        return self._nFrames

    def get_sampling_frequency(self):
        return self._samplingRate

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        data = self._read_function(self._rf, start_frame, end_frame, self.get_num_channels())
        # transform to slice if possible
        if sorted(channel_ids) == channel_ids and np.all(np.diff(channel_ids) == 1):
            channel_ids = slice(channel_ids[0], channel_ids[0]+len(channel_ids))
        return data[:, channel_ids].T

    @staticmethod
    def write_recording(recording, save_path):
        # Convert to uV:
        # AnalogValue = MVOffset + DigitalValue * ADCCountsToMV
        # Where ADCCountsToMV is defined as:
        # ADCCountsToMV = SignalInversion * ((MaxVolt - MinVolt) / 2^BitDepth)
        # And MVOffset as:
        # MVOffset = SignalInversion * MinVolt
        # conversion back
        # DigitalValue = (AnalogValue - MVOffset)/ADCCountsToMV
        # we center at 2048

        assert HAVE_BIOCAM, "To use the BiocamRecordingExtractor install h5py: \n\n pip install h5py\n\n"
        M = recording.get_num_channels()
        N = recording.get_num_frames()
        rf = h5py.File(save_path, 'w')
        g = rf.create_group('3BData')
        dr = rf.create_dataset('3BData/Raw', (M * N,), dtype=int)
        dt = 50000
        for i in range(N // dt):
            dr[M * i * dt:M * (i + 1) * dt] = recording.get_traces(range(M), i * dt, (i + 1) * dt).T.flatten()
        dr[M * (N // dt) * dt:] = recording.get_traces(range(M), (N // dt) * dt, N).T.flatten()
        g.attrs['Version'] = 101
        rf.create_dataset('3BRecInfo/3BRecVars/MinVolt', data=[0])
        rf.create_dataset('3BRecInfo/3BRecVars/MaxVolt', data=[1])
        rf.create_dataset('3BRecInfo/3BRecVars/NRecFrames', data=[N])
        rf.create_dataset('3BRecInfo/3BRecVars/SamplingRate', data=[recording.get_sampling_frequency()])
        rf.create_dataset('3BRecInfo/3BRecVars/SignalInversion', data=[1])
        rf.create_dataset('3BRecInfo/3BMeaChip/NCols', data=[M])
        r = recording.get_channel_locations()[:, 0]
        c = recording.get_channel_locations()[:, 1]
        d = np.ndarray((1, len(r)), dtype=[('Row', '<i2'), ('Col', '<i2')])
        d['Row'] = r
        d['Col'] = c
        rf.create_dataset('3BRecInfo/3BMeaStreams/Raw/Chs', data=d)
        rf.close()


def openBiocamFile(filename, mea_pitch, verbose=False):
    """Open a Biocam hdf5 file, read and return the recording info, pick te correct method to access raw data, and return this to the caller."""
    rf = h5py.File(filename, 'r')
    # Read recording variables
    recVars = rf.require_group('3BRecInfo/3BRecVars/')
    # bitDepth = recVars['BitDepth'].value[0]
    # maxV = recVars['MaxVolt'].value[0]
    # minV = recVars['MinVolt'].value[0]
    nFrames = recVars['NRecFrames'][0]
    samplingRate = recVars['SamplingRate'][0]
    signalInv = recVars['SignalInversion'][0]
    # Read chip variables
    chipVars = rf.require_group('3BRecInfo/3BMeaChip/')
    nCols = chipVars['NCols'][0]
    # Get the actual number of channels used in the recording
    file_format = rf['3BData'].attrs.get('Version')
    if file_format == 100:
        nRecCh = len(rf['3BData/Raw'][0])
    elif (file_format == 101) or (file_format == 102):
        nRecCh = int(1. * rf['3BData/Raw'].shape[0] / nFrames)
    else:
        raise Exception('Unknown data file format.')

    if verbose:
        print('# 3Brain data format:', file_format, 'signal inversion', signalInv)
        print('#       signal range: ', recVars['MinVolt'][0], '- ', recVars['MaxVolt'][0])
        print('# channels: ', nRecCh)
        print('# frames: ', nFrames)
        print('# sampling rate: ', samplingRate)
    # get channel locations
    r = (rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Row'] - 1) * mea_pitch
    c = (rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Col'] - 1) * mea_pitch
    rawIndices = np.vstack((r, c)).T
    # assign channel numbers
    chIndices = np.array([(x - 1) + (y - 1) * nCols for (y, x) in rawIndices])
    # determine correct function to read data
    if verbose:
        print("# Signal inversion is " + str(signalInv) + ".")
        print("# If your spike sorting results look wrong, invert the signal.")
    if (file_format == 100) & (signalInv == 1):
        read_function = readHDF5t_100
    elif (file_format == 100) & (signalInv == -1):
        read_function = readHDF5t_100_i
    if ((file_format == 101) | (file_format == 102)) & (signalInv == 1):
        read_function = readHDF5t_101
    elif ((file_format == 101) | (file_format == 102)) & (signalInv == -1):
        read_function = readHDF5t_101_i
    else:
        raise RuntimeError("File format unknown.")
    return (rf, nFrames, samplingRate, nRecCh, chIndices, file_format, signalInv, rawIndices, read_function)


def readHDF5t_100(rf, t0, t1, nch):
    if t0 <= t1:
        return rf['3BData/Raw'][t0:t1]
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return rf['3BData/Raw'][t1:t0]


def readHDF5t_100_i(rf, t0, t1, nch):
    if t0 <= t1:
        return 4096 - rf['3BData/Raw'][t0:t1]
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return 4096 - rf['3BData/Raw'][t1:t0]


def readHDF5t_101(rf, t0, t1, nch):
    if t0 <= t1:
        return rf['3BData/Raw'][nch * t0:nch * t1].reshape((t1 - t0, nch), order='C')
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return rf['3BData/Raw'][nch * t1:nch * t0].reshape((t1 - t0, nch), order='C')


def readHDF5t_101_i(rf, t0, t1, nch):
    if t0 <= t1:
        return 4096 - rf['3BData/Raw'][nch * t0:nch * t1].reshape((t1 - t0, nch), order='C')
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return 4096 - rf['3BData/Raw'][nch * t1:nch * t0].reshape((t1 - t0, nch), order='C')
