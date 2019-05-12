from spikeextractors import RecordingExtractor

import numpy as np
import ctypes


try:
    import h5py
    HAVE_BIOCAM = True
except ImportError:
    HAVE_BIOCAM = False

class BiocamRecordingExtractor(RecordingExtractor):

    extractor_name = 'BiocamRecordingExtractor'
    has_locations = True
    installed = HAVE_BIOCAM  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording_file', 'type': 'path', 'title': "Path to file"},
    ]
    installation_mesg = "To use the BiocamRecordingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, recording_file, verbose=False):
        assert HAVE_BIOCAM, "To use the BiocamRecordingExtractor install h5py: \n\n pip install h5py\n\n"
        RecordingExtractor.__init__(self)
        self._recording_file = recording_file
        self._rf, self._nFrames, self._samplingRate, self._nRecCh, self._chIndices, \
        self._file_format, self._signalInv, self._positions, self._read_function = openBiocamFile(
            self._recording_file, verbose=verbose)
        for m in range(self._nRecCh):
            self.set_channel_property(m, 'location', self._positions[m])

    def __del__(self):
        self._rf.close()

    def get_channel_ids(self):
        return list(range(self._nRecCh))

    def get_num_frames(self):
        return self._nFrames

    def get_sampling_frequency(self):
        return self._samplingRate

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = range(self.get_num_channels())
        data = self._read_function(
            self._rf, start_frame, end_frame, self.get_num_channels())
        return data.reshape((end_frame - start_frame,
                             self.get_num_channels())).T[channel_ids]

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
        dr = rf.create_dataset('3BData/Raw', (M*N,), dtype=int)
        dt = 50000
        for i in range(N//dt):
            dr[M*i*dt:M*(i+1)*dt] = recording.get_traces(slice(0, M), i*dt, (i+1)*dt).T.flatten()
        dr[M*(N//dt)*dt:M*N] = recording.get_traces(slice(0, M), (N//dt)*dt, N).T.flatten()
        g.attrs['Version'] = 101
        rf.create_dataset('3BRecInfo/3BRecVars/MinVolt', data=[0])
        rf.create_dataset('3BRecInfo/3BRecVars/MaxVolt', data=[1])
        rf.create_dataset('3BRecInfo/3BRecVars/NRecFrames', data=[N])
        rf.create_dataset('3BRecInfo/3BRecVars/SamplingRate', data=[recording.get_sampling_frequency()])
        rf.create_dataset('3BRecInfo/3BRecVars/SignalInversion', data=[1])
        rf.create_dataset('3BRecInfo/3BMeaChip/NCols', data=[M])
        r = [recording.get_channel_property(i,'location')[-2] for i in range(recording.get_num_channels())]
        c = [recording.get_channel_property(i,'location')[-1] for i in range(recording.get_num_channels())]
        d = np.ndarray((1,len(r)),dtype=[('Row','<i2'),('Col','<i2')])
        d['Row'] = r
        d['Col'] = c
        rf.create_dataset('3BRecInfo/3BMeaStreams/Raw/Chs', data=d)
        rf.close()


def openBiocamFile(filename, verbose=False):
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
    elif file_format == 101:
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
    r = rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Row']
    c = rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Col']
    rawIndices = np.vstack((r, c)).T
    # assign channel numbers
    chIndices = np.array([(x - 1) + (y - 1) * nCols for (y, x) in rawIndices])
    # determine correct function to read data
    if verbose:
        print("# Signal inversion looks like " + str(signalInv) + ", guessing correct method for data access.")
        print("# If your results look wrong, signal polarity is may be wrong.")
    if file_format == 100:
        if signalInv == -1:
            read_function = readHDF5t_100
        else:
            read_function = readHDF5t_100_i
    else:
        if signalInv == -1:
            read_function = readHDF5t_101_i
        else:
            read_function = readHDF5t_101
    return (rf, nFrames, samplingRate, nRecCh, chIndices, file_format, signalInv, rawIndices, read_function)


def readHDF5(rf, t0, t1):
    return rf['3BData/Raw'][t0:t1].flatten().astype(ctypes.c_short)
    # return 4095 - rf['3BData/Raw'][t0:t1].flatten().astype(ctypes.c_short)


def readHDF5t_100(rf, t0, t1, nch):
    if t0 <= t1:
        d = rf['3BData/Raw'][t0:t1].flatten('C').astype(ctypes.c_short)
        # d = 2048 - rf['3BData/Raw'][t0:t1].flatten('C').astype(ctypes.c_short)
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return rf['3BData/Raw'][t1:t0].flatten('F').astype(ctypes.c_short)
        # return 2048 - rf['3BData/Raw'][t1:t0].flatten('F').astype(ctypes.c_short)


def readHDF5t_100_i(rf, t0, t1, nch):
    if t0 <= t1:
        d = rf['3BData/Raw'][t0:t1].flatten('C').astype(ctypes.c_short) #- 2048
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return rf['3BData/Raw'][t1:t0].flatten('F').astype(ctypes.c_short) #- 2048


def readHDF5t_101(rf, t0, t1, nch):
    if t0 <= t1:
        d = rf['3BData/Raw'][nch * t0:nch * t1].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short) #- 2048
        d[np.abs(d) > 1500] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        d = rf['3BData/Raw'][nch * t1:nch * t0].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short) #- 2048
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d


def readHDF5t_101_i(rf, t0, t1, nch):
    if t0 <= t1:
        d = rf['3BData/Raw'][nch * t0:nch * t1].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short)
        # d = 2048 - rf['3BData/Raw'][nch * t0:nch * t1].reshape(
        #     (-1, nch), order='C').flatten('C').astype(ctypes.c_short)
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        d = rf['3BData/Raw'][nch * t1:nch * t0].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short)
        # d = 2048 - rf['3BData/Raw'][nch * t1:nch * t0].reshape(
        #     (-1, nch), order='C').flatten('C').astype(ctypes.c_short)
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
