from pathlib import Path

from .cacheextractors import CacheRecordingExtractor, CacheSortingExtractor
from .recordingextractor import RecordingExtractor
from .sortingextractor import SortingExtractor


def save_si_object(object_name: str, si_object, output_folder,
                   cache_raw=False, include_properties=True, include_features=False):
    """
    Save an arbitrary SI object to a temprary location.

    Parameters
    ----------
    object_name: str
        The unique name of the SpikeInterface object.
    si_object: RecordingExtractor or SortingExtractor
        The extractor to be saved.
    output_folder: str or Path
        The folder where the object is saved.
    cache_raw: bool
        If True, the Extractor is cached to a binary file (not recommended for RecordingExtractor objects)
        (default False).
    include_properties: bool
        If True, properties (channel or unit) are saved (default True).
    include_features: bool
        If True, spike features are saved (default False)
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if isinstance(si_object, RecordingExtractor):
        if not si_object.is_dumpable:
            cache = CacheRecordingExtractor(si_object, save_path=output_folder / "raw.dat")
        elif cache_raw:
            # save to json before caching to keep history (in case it's needed)
            json_file = output_folder / f"{object_name}.json"
            si_object.dump_to_json(output_folder / json_file)
            cache = CacheRecordingExtractor(si_object, save_path=output_folder / "raw.dat")
        else:
            cache = si_object

    elif isinstance(si_object, SortingExtractor):
        if not si_object.is_dumpable:
            cache = CacheSortingExtractor(si_object, save_path=output_folder / "sorting.npz")
        elif cache_raw:
            # save to json before caching to keep history (in case it's needed)
            json_file = output_folder / f"{object_name}.json"
            si_object.dump_to_json(output_folder / json_file)
            cache = CacheSortingExtractor(si_object, save_path=output_folder / "sorting.npz")
        else:
            cache = si_object
    else:
        raise ValueError("The 'si_object' argument shoulde be a SpikeInterface Extractor!")

    pkl_file = output_folder / f"{object_name}.pkl"
    cache.dump_to_pickle(
        output_folder / pkl_file,
        include_properties=include_properties,
        include_features=include_features
    )
