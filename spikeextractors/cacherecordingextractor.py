from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
import tempfile
import os


class CacheRecordingExtractor(BinDatRecordingExtractor):

    extractor_name = 'CacheRecordingExtractor'

    def __init__(self, recording, dtype=None, chunksize=None):
        self._tmp_file = tempfile.NamedTemporaryFile(suffix=".dat").name
        if dtype is None:
            dtype = recording.get_traces(start_frame=0, end_frame=2).dtype
        recording.write_to_binary_dat_format(save_path=self._tmp_file, dtype=dtype, chunksize=chunksize)
        BinDatRecordingExtractor.__init__(self, self._tmp_file, numchan=recording.get_num_channels(),
                                          recording_channels=recording.get_channel_ids(),
                                          sampling_frequency=recording.get_sampling_frequency(),
                                          dtype=dtype)
        self.copy_channel_properties(recording)

    def __del__(self):
        os.remove(self._tmp_file)
