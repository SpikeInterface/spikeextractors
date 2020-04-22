from pathlib import Path
from typing import Union

import numpy as np

from spikeextractors.extractors.matsortingextractor.matsortingextractor import MATSortingExtractor, HAVE_MAT
from spikeextractors.extraction_tools import check_valid_unit_id

PathType = Union[str, Path]


class JRCSortingExtractor(MATSortingExtractor):
    extractor_name = "JRCSortingExtractor"
    installation_mesg = "To use the MATSortingExtractor install h5py and scipy: \n\n pip install h5py scipy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, keep_good_only: bool = False):
        super().__init__(file_path)
        file_path = self._kwargs["file_path"]

        spike_times = self._getfield("spikeTimes").ravel() - 1  # int32
        spike_clusters = self._getfield("spikeClusters").ravel()  # uint32
        spike_amplitudes = self._getfield("spikeAmps").ravel()  # int16
        spike_sites = self._getfield("spikeSites").ravel()  # uint32
        spike_positions = self._getfield("spikePositions").T  # float32

        unit_centroids = self._getfield("clusterCentroids").astype(np.float).T
        unit_sites = self._getfield("clusterSites").astype(np.uint32).ravel()
        mean_waveforms = self._getfield("meanWfGlobal").T
        mean_waveforms_raw = self._getfield("meanWfGlobalRaw").T

        # try to extract the sample rate from the .prm file
        sample_rate = None
        prm_file = Path(file_path.name.replace("_res.mat", ".prm"))
        with prm_file.open("r") as fh:
            line = fh.readline()
            while line:
                line = line.split('%')[0].strip(" ;")
                if line.startswith("sampleRate"):
                    line = line.split("=", 1)
                    try:
                        sample_rate = float(line[1])
                    except (IndexError, ValueError):
                        raise AttributeError("Malformed or missing value for sample rate.")
                    break
                line = fh.readline()

        if sample_rate is None:
            raise AttributeError("Sample rate was not defined.")

        self.set_sampling_frequency(sample_rate)

        # traces, features
        raw_file = Path(file_path.name.replace("_res.mat", "_raw.jrc"))
        raw_shape = self._getfield("rawShape").ravel().astype(np.int)

        filt_file = Path(file_path.stem + "_filt.jrc")
        filt_shape = self._getfield("filtShape").ravel().astype(np.int)

        features_file = Path(file_path.stem + "_features.jrc")
        features_shape = self._getfield("featuresShape").ravel().astype(np.int)

        # nonpositive clusters are noise or deleted units
        if keep_good_only:
            good_mask = spike_clusters > 0
        else:
            good_mask = np.ones_like(spike_clusters, dtype=np.bool)

        self._unit_ids = np.unique(spike_clusters[good_mask])

        # load spike trains
        self._spike_trains = {}
        for uid in self._unit_ids:
            mask = (spike_clusters == uid)
            self._spike_trains[uid] = spike_times[mask]

            self.set_unit_spike_features(uid, "amplitudes", spike_amplitudes[mask])
            self.set_unit_spike_features(uid, "max_channels", spike_sites[mask])
            self.set_unit_spike_features(uid, "positions", spike_positions[mask, :])

            self.set_unit_property(uid, "centroid", unit_centroids[uid - 1, :])
            self.set_unit_property(uid, "max_channel", unit_sites[uid - 1])
            self.set_unit_property(uid, "template", mean_waveforms[:, :, uid - 1])
            self.set_unit_property(uid, "template_raw", mean_waveforms_raw[:, :, uid - 1])

        self._kwargs["keep_good_only"] = keep_good_only

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        start_frame = start_frame or 0
        end_frame = end_frame or np.infty

        st = self._spike_trains[unit_id]
        return st[(st >= start_frame) & (st < end_frame)]

    def get_unit_ids(self):
        return self._unit_ids.tolist()
