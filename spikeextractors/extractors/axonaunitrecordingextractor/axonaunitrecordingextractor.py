from spikeextractors.extraction_tools import check_get_traces_args
from spikeextractors import RecordingExtractor
from pathlib import Path
import numpy as np
from typing import Union

PathType = Union[Path, str]


try:
    import pyxona
    HAVE_PYXONA = True
except ImportError:
    HAVE_PYXONA = False


class AxonaUnitRecordingExtractor(RecordingExtractor):
    """
    Instantiates a RecordinExtractor from an Axon Unit mode file.

    Since the unit mode format only saves waveforms cutouts, the get_traces function fills in the rest of the
    recording with Gaussian uncorrelated noise

    Parameters
    ----------

    file_path: Path type
        The file path to the .set file
    noise_std: float
        Standard deviation of the Gaussian background noise
    """
    extractor_name = 'AxonaUnitRecording'
    has_default_locations = False
    has_unscaled = False
    installed = HAVE_PYXONA  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the AxonaUnitRecordingExtractor install pyxona: \n\n pip install pyxona\n\n"

    def __init__(self, file_path: PathType, noise_std: float = 3):
        RecordingExtractor.__init__(self)
        self._fileobj = pyxona.File(str(file_path))

        channel_ids = []
        channel_groups = []
        spike_trains = []
        channel_indexes = []
        waveforms = []
        for i, chan_grp in enumerate(self._fileobj.channel_groups):
            sample_rate = chan_grp.spike_train.sample_rate.magnitude
            if i == 0:
                self._fs = sample_rate
            spike_trains.append(np.array(chan_grp.spike_train.times.magnitude * sample_rate).astype(int))
            waveforms.append(chan_grp.spike_train.waveforms)
            channel_indexes.append([ch.index for ch in chan_grp.channels])
            for ch in chan_grp.channels:
                channel_ids.append(ch.index)
                channel_groups.append(chan_grp.channel_group_id)

        self._waveforms = waveforms
        self._spike_trains = spike_trains
        self._channel_indexes = channel_indexes
        self._channel_ids = channel_ids

        self._num_frames = int((self._fileobj._duration.magnitude * sample_rate))

        self._noise_std = noise_std

        # set groups
        self.set_channel_groups(channel_groups)

        self._kwargs = {'file_path': Path(file_path).absolute(), 'noise_std': noise_std}

    def get_channel_ids(self):
        return self._channel_ids

    def get_sampling_frequency(self):
        return self._fs

    def get_num_frames(self):
        return self._num_frames

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        num_frames_traces = end_frame - start_frame
        traces = self._noise_std * np.random.randn(len(channel_ids), num_frames_traces)

        channel_idxs = [self.get_channel_ids().index(ch) for ch in channel_ids]

        for (chan_idxs, spike_train, waveform) in zip(self._channel_indexes, self._spike_trains, self._waveforms):
            spike_times_idxs = np.where((spike_train > start_frame) & (spike_train <= end_frame))
            spike_times_i = spike_train[spike_times_idxs]

            waveforms_i = waveform[spike_times_idxs]
            wf_samples = waveforms_i.shape[2]

            for t, wf in zip(spike_times_i, waveforms_i):
                t = t - start_frame

                if t - wf_samples // 2 < 0:
                    traces[chan_idxs, :t + wf_samples // 2] = wf[:, wf_samples // 2 - t:]
                elif t + wf_samples // 2 > num_frames_traces:
                    traces[chan_idxs, t - wf_samples // 2:] = wf[:, :num_frames_traces - (t + wf_samples // 2)]
                else:
                    traces[chan_idxs, t - wf_samples // 2:t + wf_samples // 2] = wf
        return traces[channel_idxs]
