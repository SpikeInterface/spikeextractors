import os
import numpy as np
try:
    import nixio as nix
    HAVE_NIXIO = True
except ImportError:
    HAVE_NIXIO = False

from ...recordingextractor import RecordingExtractor

# error message when not installed
missing_nixio_msg = ("To use the NIXIORecordingExtractor install nixio:"
                     "\n\n pip install nixio\n\n")


class NIXIORecordingExtractor(RecordingExtractor):

    extractor_name = 'NIXIORecordingExtractor'
    installed = HAVE_NIXIO
    is_writable = True
    mode = 'file'

    def __init__(self, file_path):
        if not HAVE_NIXIO:
            raise ImportError(missing_nixio_msg)
        RecordingExtractor.__init__(self)
        self._file = nix.File.open(file_path, nix.FileMode.ReadOnly)

    def __del__(self):
        self._file.close()

    @property
    def _traces(self):
        blk = self._file.blocks[0]
        da = blk.data_arrays["traces"]
        return da

    def get_channel_ids(self):
        da = self._traces
        channel_dim = da.dimensions[0]
        channel_ids = [int(chid) for chid in channel_dim.labels]
        return channel_ids

    def get_num_frames(self):
        da = self._traces
        return da.shape[1]

    def get_sampling_frequency(self):
        da = self._traces
        timedim = da.dimensions[1]
        sampling_frequency = 1./timedim.sampling_interval
        return sampling_frequency

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        '''This function extracts and returns a trace from the recorded data
        from the given channels ids and the given start and end frame. It will
        return traces from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_recording_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_recording_frame - 1]

        If both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or
        end_frame are given, respectively. Traces are returned in a 2D array
        that contains all of the traces from each channel with dimensions
        (num_channels x num_frames). In this implementation, start_frame is
        inclusive and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        start_frame: int
            The starting frame of the trace to be returned (inclusive).
        end_frame: int
            The ending frame of the trace to be returned (exclusive).
        channel_ids: array_like
            A list or 1D array of channel ids (ints) from which each trace
            will be extracted.

        Returns
        ----------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (num_channels x num_frames)
        '''
        if channel_ids:
            channels = np.array([self._traces[cid] for cid in channel_ids])
        else:
            channels = self._traces[:]
        return channels[:, start_frame:end_frame]

    @staticmethod
    def write_recording(recording, save_path, overwrite=False):
        '''
        This is an example of a function that is not abstract so it is
        optional if you want to override it.  It allows other
        RecordingExtractor to use your new RecordingExtractor to convert their
        recorded data into your recording file format.
        '''
        if not HAVE_NIXIO:
            raise ImportError(missing_nixio_msg)
        if os.path.exists(save_path) and not overwrite:
            raise FileExistsError("File exists: {}".format(save_path))

        nf = nix.File.open(save_path, nix.FileMode.Overwrite)
        # use the file name to name the top-level block
        fname = os.path.basename(save_path)
        block = nf.create_block(fname, "spikeinterface.recording")
        da = block.create_data_array("traces", "spikeinterface.traces",
                                     data=recording.get_traces())
        labels = recording.get_channel_ids()
        if not labels:  # channel IDs not specified; just number them
            labels = list(range(recording.get_num_channels()))
        chandim = da.append_set_dimension()
        chandim.labels = labels
        sfreq = recording.get_sampling_frequency()
        timedim = da.append_sampled_dimension(sampling_interval=1./sfreq)
        timedim.unit = "s"

        nf.close()
