import numpy as np


def synthesize_timeseries(*, sorting, waveforms, noise_level=1, sampling_frequency=30000.0, duration=60, waveform_upsamplefac=13, seed=None):
    num_timepoints = np.int64(sampling_frequency * duration)
    waveform_upsamplefac = int(waveform_upsamplefac)
    W = waveforms

    M, TT, K = W.shape[0], W.shape[1], W.shape[2]
    T = int(TT / waveform_upsamplefac)
    Tmid = int(np.ceil((T + 1) / 2 - 1))

    N = num_timepoints

    if seed is not None:
        X = np.random.RandomState(seed=seed).randn(M, N) * noise_level
    else:
        X = np.random.randn(M, N) * noise_level

    unit_ids = sorting.get_unit_ids()
    for k0 in unit_ids:
        waveform0 = waveforms[:, :, k0 - 1]
        times0 = sorting.get_unit_spike_train(unit_id=k0)
        for t0 in times0:
            amp0 = 1
            frac_offset = int(np.floor((t0 - np.floor(t0)) * waveform_upsamplefac))
            tstart = np.int64(np.floor(t0)) - Tmid
            if (0 <= tstart) and (tstart + T <= N):
                X[:, tstart:tstart + T] = X[:, tstart:tstart + T] + waveform0[:,
                                                                    frac_offset::waveform_upsamplefac] * amp0

    return X
