import spikeextractors as se
import numpy as np
from .synthesize_random_waveforms import synthesize_random_waveforms
from .synthesize_random_firings import synthesize_random_firings
from .synthesize_timeseries import synthesize_timeseries


def toy_example1(duration=10, num_channels=4, samplerate=30000, K=10):
    upsamplefac = 13

    waveforms, geom = synthesize_random_waveforms(K=K, M=num_channels, average_peak_amplitude=-100,
                                                  upsamplefac=upsamplefac)
    times, labels = synthesize_random_firings(K=K, duration=duration, samplerate=samplerate)
    labels = labels.astype(np.int64)
    OX = se.NumpySortingExtractor()
    OX.setTimesLabels(times, labels)
    X = synthesize_timeseries(sorting=OX, waveforms=waveforms, noise_level=10, samplerate=samplerate, duration=duration,
                              waveform_upsamplefac=upsamplefac)

    IX = se.NumpyRecordingExtractor(timeseries=X, samplerate=samplerate, geom=geom)
    return (IX, OX)
