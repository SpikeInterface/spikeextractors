import numpy as np


def exp_growth(amp1, amp2, dur1, dur2):
    t = np.arange(0, dur1)
    Y = np.exp(t / dur2)
    # Want Y[0]=amp1
    # Want Y[-1]=amp2
    Y = Y / (Y[-1] - Y[0]) * (amp2 - amp1)
    Y = Y - Y[0] + amp1;
    return Y


def exp_decay(amp1, amp2, dur1, dur2):
    Y = exp_growth(amp2, amp1, dur1, dur2)
    Y = np.flipud(Y)  # used to be flip, but that was not supported by older versions of numpy
    return Y


def smooth_it(Y, t):
    Z = np.zeros(Y.size)
    for j in range(-t, t + 1):
        Z = Z + np.roll(Y, j)
    return Z


def synthesize_single_waveform(*, N=800, durations=[200, 10, 30, 200], amps=[0.5, 10, -1, 0]):
    durations = np.array(durations).ravel()
    if (np.sum(durations) >= N - 2):
        durations[-1] = N - 2 - np.sum(durations[0:durations.size - 1])

    amps = np.array(amps).ravel()

    timepoints = np.round(np.hstack((0, np.cumsum(durations) - 1))).astype('int');

    t = np.r_[0:np.sum(durations) + 1]

    Y = np.zeros(len(t))
    Y[timepoints[0]:timepoints[1] + 1] = exp_growth(0, amps[0], timepoints[1] + 1 - timepoints[0], durations[0] / 4)
    Y[timepoints[1]:timepoints[2] + 1] = exp_growth(amps[0], amps[1], timepoints[2] + 1 - timepoints[1], durations[1])
    Y[timepoints[2]:timepoints[3] + 1] = exp_decay(amps[1], amps[2], timepoints[3] + 1 - timepoints[2],
                                                   durations[2] / 4)
    Y[timepoints[3]:timepoints[4] + 1] = exp_decay(amps[2], amps[3], timepoints[4] + 1 - timepoints[3],
                                                   durations[3] / 5)
    Y = smooth_it(Y, 3)
    Y = Y - np.linspace(Y[0], Y[-1], len(t))
    Y = np.hstack((Y, np.zeros(N - len(t))))
    Nmid = int(np.floor(N / 2))
    peakind = np.argmax(np.abs(Y))
    Y = np.roll(Y, Nmid - peakind)

    return Y


# Y=smooth_it(Y,3);
# Y=Y-linspace(Y(1),Y(end),length(Y));
#
# Y=[Y,zeros(1,N-length(Y))];
#
# Nmid=floor(N/2);
# [~,peakind]=max(abs(Y));
# Y=circshift(Y,[0,Nmid-peakind]);
#
# end
#
# function test_synth_waveform
# Y=synthesize_single_waveform(800);
# figure; plot(Y);
# end
#
# function Y=exp_growth(amp1,amp2,dur1,dur2)
# t=1:dur1;
# Y=exp(t/dur2);
# % Want Y(1)=amp1
# % Want Y(end)=amp2
# Y=Y/(Y(end)-Y(1))*(amp2-amp1);
# Y=Y-Y(1)+amp1;
# end
#
# function Y=exp_decay(amp1,amp2,dur1,dur2)
# Y=exp_growth(amp2,amp1,dur1,dur2);
# Y=Y(end:-1:1);
# end
#
# function Z=smooth_it(Y,t)
# Z=Y;
# Z(1+t:end-t)=0;
# for j=-t:t
#    Z(1+t:end-t)=Z(1+t:end-t)+Y(1+t+j:end-t+j)/(2*t+1);
# end;
# end

if __name__ == '__main__':
    Y = synthesize_single_waveform()
    import matplotlib.pyplot as plt

    plt.plot(Y)
