from pathlib import Path
import re
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
        spike_sites = self._getfield("spikeSites").ravel() - 1  # uint32
        spike_positions = self._getfield("spikePositions").T  # float32

        unit_centroids = self._getfield("clusterCentroids").astype(np.float).T
        unit_sites = self._getfield("clusterSites").astype(np.uint32).ravel()
        mean_waveforms = self._getfield("meanWfGlobal").T
        mean_waveforms_raw = self._getfield("meanWfGlobalRaw").T

        # try to extract various parameters from the .prm file
        self._bit_scaling = np.float32(0.30518)  # conversion factor for ADC units -> µV
        sample_rate = 30000.
        filter_type = "ndiff"
        ndiff_order = 2

        prm_file = Path(file_path.parent, file_path.name.replace("_res.mat", ".prm"))
        with prm_file.open("r") as fh:
            lines = [line.strip() for line in fh.readlines()]

        for line in lines:
            try:
                key, val = line.split('%', 1)[0].strip(" ;").split("=")
            except ValueError:
                continue

            key = key.strip()
            val = val.strip()

            if key == "sampleRate":
                try:
                    sample_rate = float(val)
                except (IndexError, ValueError):
                    pass
            elif key == "bitScaling":
                try:
                    self._bit_scaling = np.float32(val)
                except (IndexError, ValueError):
                    pass
            elif key == "filterType":
                filter_type = val
            elif key == "nDiffOrder":
                try:
                    ndiff_order = int(val)
                except (IndexError, ValueError):
                    pass
            elif key == "siteLoc":
                site_locs = []
                str_locs = map(lambda v: v.strip(" ]["), val.split(";"))
                for loc in str_locs:
                    x, y = map(float, re.split(r",?\s+", loc))
                    site_locs.append([x, y])

                site_locs = np.array(site_locs)
            elif key == "shankMap":
                val = val.strip("][")
                try:
                    shank_map = np.array(map(float, re.split(r"[,;]?\s+", val)))
                except:
                    shank_map = np.array([])

        self.set_sampling_frequency(sample_rate)
        if filter_type == "sgdiff":
            self._bit_scaling /= (2 * (np.arange(1, ndiff_order + 1) ** 2).sum())
        elif filter_type == "ndiff":
            self._bit_scaling /= 2

        # traces, features
        raw_file = Path(file_path.parent, file_path.name.replace("_res.mat", "_raw.jrc"))
        raw_shape = tuple(self._getfield("rawShape").ravel().astype(np.int))
        self._raw_traces = np.memmap(raw_file, dtype=np.int16, mode="r",
                                     shape=raw_shape, order="F")

        filt_file = Path(file_path.parent, file_path.name.replace("_res.mat", "_filt.jrc"))
        filt_shape = tuple(self._getfield("filtShape").ravel().astype(np.int))
        self._filt_traces = np.memmap(filt_file, dtype=np.int16, mode="r",
                                      shape=filt_shape, order="F")

        features_file = Path(file_path.parent, file_path.name.replace("_res.mat", "_features.jrc"))
        features_shape = tuple(self._getfield("featuresShape").ravel().astype(np.int))
        self._cluster_features = np.memmap(features_file, dtype=np.float32, mode="r",
                                           shape=features_shape, order="F")

        neighbors = _find_site_neighbors(site_locs, raw_shape[1], shank_map)  # get nearest neighbors for each site

        # nonpositive clusters are noise or deleted units
        if keep_good_only:
            good_mask = spike_clusters > 0
        else:
            good_mask = np.ones_like(spike_clusters, dtype=np.bool)

        self._unit_ids = np.unique(spike_clusters[good_mask])

        # load spike trains
        self._spike_trains = {}
        self._unit_masks = {}
        for uid in self._unit_ids:
            mask = (spike_clusters == uid)
            self._unit_masks[uid] = mask

            self._spike_trains[uid] = spike_times[mask]

            self.set_unit_spike_features(uid, "amplitudes", spike_amplitudes[mask])
            self.set_unit_spike_features(uid, "max_channels", spike_sites[mask])
            self.set_unit_spike_features(uid, "positions", spike_positions[mask, :])
            self.set_unit_spike_features(uid, "site_neighbors", neighbors[spike_sites[mask], :])

            self.set_unit_property(uid, "centroid", unit_centroids[uid - 1, :])
            self.set_unit_property(uid, "max_channel", unit_sites[uid - 1])
            self.set_unit_property(uid, "template", mean_waveforms[:, :, uid - 1])
            self.set_unit_property(uid, "template_raw", mean_waveforms_raw[:, :, uid - 1])

        self._kwargs["keep_good_only"] = keep_good_only


    @check_valid_unit_id
    def get_unit_spike_features(self, unit_id, feature_name, start_frame=None, end_frame=None):
        if feature_name not in ("raw_traces", "filtered_traces", "cluster_features"):
            return super().get_unit_spike_features(unit_id, feature_name, start_frame, end_frame)

        mask = self._unit_masks[unit_id]
        if feature_name == "raw_traces":
            return self._raw_traces[:, :, mask] * self._bit_scaling
        elif feature_name == "filtered_traces":
            return self._filt_traces[:, :, mask] * self._bit_scaling
        else:
            return self._cluster_features[:, :, mask]

    @check_valid_unit_id
    def get_unit_spike_feature_names(self, unit_id):
        return super().get_unit_spike_feature_names(unit_id) + ["raw_traces", "filtered_traces", "cluster_features"]

    @check_valid_unit_id
    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        start_frame = start_frame or 0
        end_frame = end_frame or np.infty

        st = self._spike_trains[unit_id]
        return st[(st >= start_frame) & (st < end_frame)]

    def get_unit_ids(self):
        return self._unit_ids.tolist()


def _find_site_neighbors(site_locs, n_neighbors, shank_map):
    from scipy.spatial.distance import cdist

    if np.unique(shank_map).size <= 1:
        pass

    n_sites = site_locs.shape[0]
    n_neighbors = int(min(n_neighbors, n_sites))

    neighbors = np.zeros((n_sites, n_neighbors), dtype=np.int)
    for i in range(n_sites):
        i_loc = site_locs[i, :][np.newaxis, :]
        dists = cdist(i_loc, site_locs).ravel()
        neighbors[i, :] = dists.argsort()[:n_neighbors]

    return neighbors
