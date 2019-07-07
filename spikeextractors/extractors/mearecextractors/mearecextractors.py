from spikeextractors import RecordingExtractor
from spikeextractors import SortingExtractor

import numpy as np
from pathlib import Path

try:
    import MEArec as mr
    import quantities as pq
    import neo
    HAVE_MREX = True
except ImportError:
    HAVE_MREX = False

class MEArecRecordingExtractor(RecordingExtractor):

    extractor_name = 'MEArecRecordingExtractor'
    installed = HAVE_MREX  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording_path', 'type': 'path', 'title': "Path to file"},
        {'name': 'locs_2d', 'type': 'bool', 'title': "If True 3D locations are converted to 2D"},
        {'name': 'probe_path', 'type': 'path', 'value':None, 'default':None, 'title': "Path to probe file (.csv or .prb)"}
    ]
    installation_mesg = "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"  # error message when not installed

    def __init__(self, recording_path, locs_2d=True):
        RecordingExtractor.__init__(self)
        self._recording_path = recording_path
        self._fs = None
        self._positions = None
        self._recordings = None
        self._locs_2d = locs_2d
        self._locations = None
        self._initialize()

        if self._locations is not None:
            for chan, pos in enumerate(self._locations):
                self.set_channel_property(chan, 'location', pos)

    def _initialize(self):
        assert HAVE_MREX, "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"
        recgen = mr.load_recordings(recordings=self._recording_path, return_h5_objects=True, check_suffix=False)
        self._fs = recgen.info['recordings']['fs']
        self._recordings = recgen.recordings
        self._num_channels, self._num_frames = self._recordings.shape
        if len(np.array(recgen.channel_positions)) == self._num_channels:
            self._locations = np.array(recgen.channel_positions)
            if self._locs_2d:
                if 'electrodes' in recgen.info.keys():
                    if 'plane' in recgen.info['electrodes'].keys():
                        probe_plane = recgen.info['electrodes']['plane']
                        if probe_plane == 'xy':
                            self._locations = self._locations[:, :2]
                        elif probe_plane == 'yz':
                            self._locations = self._locations[:, 1:]
                        elif probe_plane == 'xz':
                            self._locations = self._locations[:, [0, 2]]
                if self._locations.shape[1] == 3:
                    print("Could not load plane information. Assuming probe is in yz plane")
                    self._locations = self._locations[:, 1:]

    def get_channel_ids(self):
        return list(range(self._num_channels))

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._fs

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = range(self.get_num_channels())
        if np.any(np.diff(channel_ids) < 0):
            sorted_idx = np.argsort(channel_ids)
            recordings = self._recordings[np.sort(channel_ids), start_frame:end_frame]
            return recordings[sorted_idx]
        else:
            return self._recordings[np.array(channel_ids), start_frame:end_frame]

    @staticmethod
    def write_recording(recording, save_path, check_suffix=True):
        '''
        Save recording extractor to MEArec format.
        Parameters
        ----------
        recording: RecordingExtractor
            Recording extractor object to be saved
        save_path: str
            .h5 or .hdf5 path
        '''
        assert HAVE_MREX, "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"
        save_path = Path(save_path)
        if save_path.is_dir():
            print("The file will be saved as recording.h5 in the provided folder")
            save_path = save_path / 'recording.h5'
        if (save_path.suffix == '.h5' or save_path.suffix == '.hdf5') or (not check_suffix):
            info = {'recordings': {'fs': recording.get_sampling_frequency()}}
            rec_dict = {'recordings': recording.get_traces()}
            if 'location' in recording.get_channel_property_names():
                positions = np.array([recording.get_channel_property(chan, 'location')
                                      for chan in recording.get_channel_ids()])
                rec_dict['channel_positions'] = positions
            recgen = mr.RecordingGenerator(rec_dict=rec_dict, info=info)
            mr.save_recording_generator(recgen, str(save_path), verbose=False)
        else:
            raise Exception("Provide a folder or an .h5/.hdf5 as 'save_path'")


class MEArecSortingExtractor(SortingExtractor):

    extractor_name = 'MEArecSortingExtractor'
    installed = HAVE_MREX  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording_path', 'type': 'str', 'title': "str, Path to file"},
    ]
    installation_mesg = "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"  # error message when not installed

    def __init__(self, recording_path):
        SortingExtractor.__init__(self)
        self._recording_path = recording_path
        self._num_units = None
        self._spike_trains = None
        self._unit_ids = None
        self._fs = None
        self._initialize()

    def _initialize(self):
        assert HAVE_MREX, "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"
        recgen = mr.load_recordings(recordings=self._recording_path, return_h5_objects=True, check_suffix=False)
        self._num_units = len(recgen.spiketrains)
        if 'unit_id' in recgen.spiketrains[0].annotations:
            self._unit_ids = [int(st.annotations['unit_id']) for st in recgen.spiketrains]
        else:
            self._unit_ids = list(range(self._num_units))
        self._spike_trains = recgen.spiketrains
        self._fs = recgen.info['recordings']['fs'] * pq.Hz  # fs is in kHz
        self._sampling_frequency = recgen.info['recordings']['fs']

        if 'soma_position' in self._spike_trains[0].annotations:
            for u, st in zip(self._unit_ids, self._spike_trains):
                self.set_unit_property(u, 'soma_location', st.annotations['soma_position'])

    def get_unit_ids(self):
        if self._unit_ids is None:
            self._initialize()
        return self._unit_ids

    def get_num_units(self):
        if self._num_units is None:
            self._initialize()
        return self._num_units

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        if self._spike_trains is None:
            self._initialize()
        times = (self._spike_trains[self.get_unit_ids().index(unit_id)].times.rescale('s') *
                 self._fs.rescale('Hz')).magnitude
        inds = np.where((start_frame <= times) & (times < end_frame))
        return np.rint(times[inds]).astype(int)

    @staticmethod
    def write_sorting(sorting, save_path, sampling_frequency, check_suffix=True):
        '''
        Save sorting extractor to MEArec format.
        Parameters
        ----------
        sorting: SortingExtractor
            Sorting extractor object to be saved
        save_path: str
            .h5 or .hdf5 path
        sampling_frequency: int
            Sampling frequency in Hz

        '''
        assert HAVE_MREX, "To use the MEArec extractors, install MEArec: \n\n pip install MEArec\n\n"
        save_path = Path(save_path)
        if save_path.is_dir():
            print("The file will be saved as sorting.h5 in the provided folder")
            save_path = save_path / 'sorting.h5'
        if (save_path.suffix == '.h5' or save_path.suffix == '.hdf5') or (not check_suffix):
            # create neo spike trains
            spiketrains = []
            for u in sorting.get_unit_ids():
                st = neo.SpikeTrain(times=sorting.get_unit_spike_train(u) / float(sampling_frequency) * pq.s,
                                    t_start=np.min(sorting.get_unit_spike_train(u) / float(sampling_frequency)) * pq.s,
                                    t_stop=np.max(sorting.get_unit_spike_train(u) / float(sampling_frequency)) * pq.s)
                st.annotate(unit_id=u)
                spiketrains.append(st)

            duration = np.max([st.t_stop.magnitude for st in spiketrains])
            info = {'recordings': {'fs': sampling_frequency}, 'spiketrains': {'duration': duration}}
            rec_dict = {'spiketrains': spiketrains}
            recgen = mr.RecordingGenerator(rec_dict=rec_dict, info=info)
            mr.save_recording_generator(recgen, str(save_path), verbose=False)
        else:
            raise Exception("Provide a folder or an .h5/.hdf5 as 'save_path'")
