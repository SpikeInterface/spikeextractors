from spikeextractors import RecordingExtractor

import numpy as np
import ctypes


def _load_required_modules():
    try:
        import h5py
    except ModuleNotFoundError:
        raise ModuleNotFoundError("To use the BiocamRecordingExtractor install h5py: \n\n"
                                  "pip install h5py\n\n")
    return h5py


class BiocamRecordingExtractor(RecordingExtractor):
    def __init__(self, recording_file):
        RecordingExtractor.__init__(self)
        self._recording_file = recording_file
        self._rf, self._nFrames, self._samplingRate, self._nRecCh, self._chIndices, self._file_format, self._signalInv, self._positions, self._read_function = openBiocamFile(
            self._recording_file)
        for m in range(self._nRecCh):
            self.setChannelProperty(m, 'location', self._positions[m])

    def __del__(self):
        self._rf.close()

    def getChannelIds(self):
        return list(range(self._nRecCh))

    def getNumFrames(self):
        return self._nFrames

    def getSamplingFrequency(self):
        return self._samplingRate

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = range(self.getNumChannels())
        data = self._read_function(
            self._rf, start_frame, end_frame, self.getNumChannels())
        return data.reshape((end_frame - start_frame,
                             self.getNumChannels())).T[channel_ids]

    @staticmethod
    def writeRecording(recording, save_path):
        # Convert to uV:
        # AnalogValue = MVOffset + DigitalValue * ADCCountsToMV
        # Where ADCCountsToMV is defined as:
        # ADCCountsToMV = SignalInversion * ((MaxVolt - MinVolt) / 2^BitDepth)
        # And MVOffset as:
        # MVOffset = SignalInversion * MinVolt
        # conversion back
        # DigitalValue = (AnalogValue - MVOffset)/ADCCountsToMV
        # we center at 2048

        h5py = _load_required_modules()
        M = recording.getNumChannels()
        N = recording.getNumFrames()
        raw = recording.getTraces()
        scale = 200
        data_range = np.percentile(raw,[0,100,50])
        max_amp = np.max(np.abs(data_range[:2]-data_range[2]))
        # tt = ((raw-data_range[2])/max_amp*scale).astype(int)+2048
        # if raw.dtype != int:
        #     raise Exception('Cannot write dataset in the format with non-int datatype:', raw.dtype)
        rf = h5py.File(save_path, 'w')
        # writing out in 100 format: Time x Channels
        g = rf.create_group('3BData')
        # d = rf.create_dataset('3BData/Raw', data=raw.T + 2048, dtype=int)
        d = rf.create_dataset('3BData/Raw', data=((raw.T-data_range[2])/max_amp*scale).astype(int)+2048, dtype=int)
        g.attrs['Version'] = 100
        rf.create_dataset('3BRecInfo/3BRecVars/MinVolt', data=[0])
        rf.create_dataset('3BRecInfo/3BRecVars/MaxVolt', data=[1])
        rf.create_dataset('3BRecInfo/3BRecVars/NRecFrames', data=[N])
        rf.create_dataset('3BRecInfo/3BRecVars/SamplingRate', data=[recording.getSamplingFrequency()])
        rf.create_dataset('3BRecInfo/3BRecVars/SignalInversion', data=[1])
        rf.create_dataset('3BRecInfo/3BMeaChip/NCols', data=[M])
        rf.create_dataset('3BRecInfo/3BMeaStreams/Raw/Chs', data=np.vstack((np.arange(M), np.zeros(M))).T, dtype=int)
        rf.close()


def openBiocamFile(filename):
    """Open a Biocam hdf5 file, read and return the recording info, pick te correct method to access raw data, and return this to the caller."""
    h5py = _load_required_modules()
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

    print('# 3Brain data format:', file_format, 'signal inversion', signalInv)
    print('#       signal range: ', recVars['MinVolt'][0], '- ',
          recVars['MaxVolt'][0])
    # Compute channel locations
    r = rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Row']
    c = rf['3BRecInfo/3BMeaStreams/Raw/Chs'][()]['Col']
    rawIndices = np.vstack((r, c)).T
    # assign channel numbers
    chIndices = np.array([(x - 1) + (y - 1) * nCols for (y, x) in rawIndices])

    # determine correct function to read data
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
    return 4095 - rf['3BData/Raw'][t0:t1].flatten().astype(ctypes.c_short)


def readHDF5t_100(rf, t0, t1, nch):
    if t0 <= t1:
        d = 2048 - rf['3BData/Raw'][t0:t1].flatten('C').astype(ctypes.c_short)
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return 2048 - rf['3BData/Raw'][t1:t0].flatten(
            'F').astype(ctypes.c_short)


def readHDF5t_100_i(rf, t0, t1, nch):
    if t0 <= t1:
        d = rf['3BData/Raw'][t0:t1].flatten('C').astype(ctypes.c_short) - 2048
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        return rf['3BData/Raw'][t1:t0].flatten(
            'F').astype(ctypes.c_short) - 2048


def readHDF5t_101(rf, t0, t1, nch):
    if t0 <= t1:
        d = rf['3BData/Raw'][nch * t0:nch * t1].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short) - 2048
        d[np.abs(d) > 1500] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        d = rf['3BData/Raw'][nch * t1:nch * t0].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short) - 2048
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d


def readHDF5t_101_i(rf, t0, t1, nch):
    if t0 <= t1:
        d = 2048 - rf['3BData/Raw'][nch * t0:nch * t1].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short)
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
    else:  # Reversed read
        raise Exception('Reading backwards? Not sure about this.')
        d = 2048 - rf['3BData/Raw'][nch * t1:nch * t0].reshape(
            (-1, nch), order='C').flatten('C').astype(ctypes.c_short)
        d[np.where(np.abs(d) > 1500)[0]] = 0
        return d
