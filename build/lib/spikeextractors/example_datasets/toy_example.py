import spikeextractors as se
import numpy as np
from pathlib import Path
from .synthesize_random_waveforms import synthesize_random_waveforms
from .synthesize_random_firings import synthesize_random_firings
from .synthesize_timeseries import synthesize_timeseries


def toy_example(duration=10, num_channels=4, sampling_frequency=30000.0, K=10, dumpable=False, dump_folder=None,
                seed=None):
    '''
    Creates toy recording and sorting extractors.

    Parameters
    ----------
    duration: float
        Duration in s (default 10)
    num_channels: int
        Number of channels (default 4)
    sampling_frequency: float
        Sampling frequency (default 30000)
    K: int
        Number of units (default 10)
    dumpable: bool
        If True, objects are dumped to file and become 'dumpable'
    dump_folder: str or Path
        Path to dump folder (if None, 'test' is used
    seed: int
        Seed for random initialization

    Returns
    -------
    recording: RecordingExtractor
        The output recording extractor. If dumpable is False it's a NumpyRecordingExtractor, otherwise it's an
        MdaRecordingExtractor
    sorting: SortingExtractor
        The output sorting extractor. If dumpable is False it's a NumpyRecordingExtractor, otherwise it's an
        NpzSortingExtractor
    '''
    upsamplefac = 13
    waveforms, geom = synthesize_random_waveforms(K=K, M=num_channels, average_peak_amplitude=-100,
                                                  upsamplefac=upsamplefac, seed=seed)
    times, labels = synthesize_random_firings(K=K, duration=duration, sampling_frequency=sampling_frequency, seed=seed)
    labels = labels.astype(np.int64)
    SX = se.NumpySortingExtractor()
    SX.set_times_labels(times, labels)
    X = synthesize_timeseries(sorting=SX, waveforms=waveforms, noise_level=10, sampling_frequency=sampling_frequency,
                              duration=duration,
                              waveform_upsamplefac=upsamplefac, seed=seed)
    SX.set_sampling_frequency(sampling_frequency)

    RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
    RX.is_filtered = True

    if dumpable:
        if dump_folder is None:
            dump_folder = 'toy_example'
        dump_folder = Path(dump_folder)

        se.MdaRecordingExtractor.write_recording(RX, dump_folder)
        RX = se.MdaRecordingExtractor(dump_folder)
        se.NpzSortingExtractor.write_sorting(SX, dump_folder / 'sorting.npz')
        SX = se.NpzSortingExtractor(dump_folder / 'sorting.npz')

    return RX, SX
