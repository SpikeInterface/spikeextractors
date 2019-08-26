from spikeextractors import RecordingExtractor

import numpy as np
#import ctypes

try:
    import h5py
    HAVE_MCS = True
except ImportError:
    HAVE_MCS = False

class MCSRecordingExtractor(RecordingExtractor):

    extractor_name = 'MCSRecordingExtractor'
    has_default_locations = False
    installed = HAVE_MCS  # check at class level if installed or not
#    _gui_params = [
#        {'name': 'recording_file', 'type': 'path', 'title': "Path to file"},
#    ]
    installation_mesg = "To use the MCSRecordingExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, recording_file, verbose=False, mea_pitch=200):
        assert HAVE_MCS, "To use the MCSRecordingExtractor install h5py: \n\n pip install h5py\n\n"
        self._mea_pitch = mea_pitch
        self._recording_file = recording_file
        self._rf, self._nFrames, self._samplingRate, self._nRecCh, \
        self._channel_ids, self._electrodeLabels, self._exponent, self._convFact \
        = openMCSFile(
            self._recording_file, self._mea_pitch, verbose)
        RecordingExtractor.__init__(self)
        # for m in range(self._nRecCh):
        #     self.set_channel_property(m, 'location', self._positions[m])
        # It would be useful to define electrode locations here.

    def __del__(self):
        self._rf.close()

    def get_channel_ids(self):
        return self._channel_ids

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
            channel_ids = self._channel_ids
        
        stream = self._rf.require_group('/Data/Recording_0/AnalogStream/Stream_0')
        data_V = np.array(stream.get('ChannelData'),dtype=np.int)*self._convFact.astype(float)*(10.0**(self._exponent))

        return data_V[start_frame:end_frame, channel_ids].T

    @staticmethod
    def write_recording(recording, save_path):
        # Not implemented
        # An informative example is in BiocamRecordingExtractor

        assert HAVE_MCS, "To use the MCSRecordingExtractor install h5py: \n\n pip install h5py\n\n"
        print('Method write_recording() not implemented in MCSRecordingExtractor.')


def openMCSFile(filename,  mea_pitch, verbose=False):
    """Open a MCS hdf5 file, read and return the recording info."""
    rf = h5py.File(filename, 'r')
    
    stream = rf.require_group('/Data/Recording_0/AnalogStream/Stream_0')
    data = np.array(stream.get('ChannelData'),dtype=np.int)
    timestamps = np.array(stream.get('ChannelDataTimeStamps'))
    info = np.array(stream.get('InfoChannel'))
    
    Unit = info['Unit'][0]
    Tick = info['Tick'][0]/1e6
    exponent = info['Exponent'][0]
    convFact = info['ConversionFactor'][0]
    
    nRecCh = data.shape[0]
    
    channel_ids = info['ChannelID']
    electrodeLabels = info['Label']
    
    TimeVals = np.arange(timestamps[0][0],timestamps[0][2]+1,1)*Tick
    
    assert Unit==b'V', 'Unexpected units found, expected volts, found {}'.format(Unit.decode('UTF-8'))
    data_V = data*convFact.astype(float)*(10.0**(exponent))
    
    nFrames = data.shape[1]
    samplingRate = 1./np.mean(TimeVals[1:]-TimeVals[0:-1])

    if verbose:
        print('# MCS H5 data format')
        print('# Signal range: {} to {} µV'.format(np.amin(data_V)*1e6,np.amax(data_V)*1e6))
        print('# Number of channels: {}'.format(nRecCh))
        print('# Number of frames: {}'.format(data.shape[1]))
        print('# Sampling rate: {} Hz'.format(samplingRate))

    return (rf, nFrames, samplingRate, nRecCh, channel_ids, electrodeLabels, exponent, convFact)