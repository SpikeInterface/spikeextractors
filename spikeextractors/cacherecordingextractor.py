from spikeextractors.extractors.bindatrecordingextractor import BinDatRecordingExtractor
from spikeextractors import RecordingExtractor
import tempfile
from pathlib import Path
import os
import shutil


class CacheRecordingExtractor(BinDatRecordingExtractor, RecordingExtractor):

    extractor_name = 'BinDatRecording'

    def __init__(self, recording, chunk_size=None):
        RecordingExtractor.__init__(self)  # init tmp folder before constructing BinDatRecordingExtractor
        tmp_folder = self.get_tmp_folder()
        self._recording = recording
        self._tmp_file = tempfile.NamedTemporaryFile(suffix=".dat", dir=tmp_folder).name
        dtype = recording.get_traces(start_frame=0, end_frame=2).dtype
        recording.write_to_binary_dat_format(save_path=self._tmp_file, dtype=dtype, chunk_size=chunk_size)
        BinDatRecordingExtractor.__init__(self, self._tmp_file, numchan=recording.get_num_channels(),
                                          recording_channels=recording.get_channel_ids(),
                                          sampling_frequency=recording.get_sampling_frequency(),
                                          dtype=dtype)
        self.set_tmp_folder(tmp_folder)
        self.copy_channel_properties(recording)

    def __del__(self):
        try:
            os.remove(self._tmp_file)
        except Exception as e:
            print("Unable to remove temporary file", e)

    @property
    def filename(self):
        return self._tmp_file

    def save_to_file(self, save_path):
        save_path = Path(save_path)
        if save_path.suffix != '.dat' and save_path.suffix != '.bin':
            save_path = save_path.with_suffix('.dat')
        shutil.move(self._tmp_file, str(save_path))
        self._tmp_file = str(save_path)
        self._kwargs['file_path'] = str(Path(self._tmp_file).absolute())
