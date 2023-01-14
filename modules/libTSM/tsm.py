"""
This file is part of the
### Time-Scale Modification Toolbox ###
- a Python re-implementation of the Matlab TSM toolbox by Jonathan Driedger and Meinard Müller
Authors: Sebastian Rosenzweig, Edgar Suarez, El Mehdi Lemnaouar, Miguel Gonzales
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
"""

import numpy as np
import scipy as sc
from fractions import Fraction as frac
from .utils import win, stft, istft, cross_corr, normalize_length, hps, find_peaks


def PV_TSM(x, Fs, alpha, syn_hop=512, window=win(2048, 1), zero_pad=0, restore_energy=False, fft_shift=False,
           phase_locking=False):
    """
    Phase Vocoder Time scale modification algorithm, that rescales the time-axis of the input signal x
    according to the time-stretch function s without altering the pitch of x.

    Parameters
    ----------
    x : np.ndarray [shape=(N, )], real - valued
        Signal to be transformed

    Fs : int
        Sampling rate of the input audio signal x

    alpha : float or np.ndarray [shape=(S,2)],
        Time stretch function, given by a constant (float) or a set of S anchor points (int)

    syn_hop : int
        hop size of the synthesis window

    window : np.ndarray [shape=(W,)], real - valued
        the analysis and synthesis window for STFT

    zero_pad : int
        For FFT. Number of zeros padded to the window to increase the fft size

    restore_energy : bool
        For FFT. When True, rescales every windowed synthesis frame to compensate for synthesis energy leakage

    fft_shift: bool
        For FFT. When True, applies a circular shift to each frame of half its length, prior to computing the FFT

    phase_locking : bool
        when True, Applies identity phase locking

    Returns
    -------
    y : np.ndarray [shape=(L,1)], real - valued
        The time-scale modified output signal
    """

    # Pre-calculations
    w = np.concatenate((np.zeros(int(np.floor(zero_pad/2))), window, np.zeros(int(np.floor(zero_pad/2)))))
    win_len = len(w)
    win_len_half = np.around(win_len/2).astype(int)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    num_of_chan = x.shape[1]

    # Time-stretch function
    if np.isscalar(alpha):
        anchor_points = np.array([[0, 0], [int(x.shape[0]) - 1, int(np.ceil(alpha * x.shape[0])) - 1]])
    else:
        anchor_points = alpha.astype(int)

    output_length = anchor_points[-1, 1] + 1
    syn_win_pos = np.arange(0, output_length + win_len_half, syn_hop)   # positions of the synthesis winLenHalf windows
                                                                        # in the output

    fi = sc.interpolate.interp1d(anchor_points[:, 1], anchor_points[:, 0], kind='linear', fill_value='extrapolate')
    ana_win_pos = fi(syn_win_pos)
    ana_win_pos = np.round(ana_win_pos).astype(int)  # positions of the analysis windows in the input
    ana_hop = np.append([0], ana_win_pos[1:] - ana_win_pos[:-1])  # analysis hop sizes

    # Phase Vocoder
    y = np.zeros((output_length, num_of_chan))  # initialize output

    for c in range(num_of_chan):  # loop over channels

        x_c = x[:, c]

        # STFT
        X, f, t = stft(x_c, ana_hop=ana_win_pos, w=window, Fs=Fs, num_of_frames=-1, fft_shift=fft_shift,
                       zero_pad=zero_pad)

        # Phase adaptation
        Y = np.zeros(X.shape, dtype=complex)  # the spectrogram of the output
        Y[:, 0] = X[:, 0]  # phase initialization
        N = len(w)
        k = np.arange(0, N//2 + 1)  # center frequencies of the N/2+1 first bins of the spectrum in
                                    # 'oscillations per frame'
        omega = 2 * np.pi * k / N  # phase advances per sample for the frequencies k

        for i in range(1, X.shape[1]):
            dphi = omega * ana_hop[i]  # expected phase advances from the last to the current input frame
            ph_curr = np.angle(X[:, i])  # phases of the current input frame
            ph_last = np.angle(X[:, i-1])  # phases of the last input frame
            hpi = (ph_curr - ph_last) - dphi  # heterodyned phase increments
            #hpi = np.array([num - 2 * np.pi * round(num/(2*np.pi)) for num in hpi])  # reduce to the range -pi:pi,
            # np.round() deviates from Matlab's round()
            hpi = hpi - 2 * np.pi * np.round(hpi/(2*np.pi))  # reduce to the range -pi:pi
            ipa_sample = omega + hpi / ana_hop[i]  # instantaneous phase advances per sample
            ipa_hop = ipa_sample * syn_hop  # instantaneous phase advances per synthesis hopsize
            ph_syn = np.angle(Y[:, i-1])  # phases of the last synthesized frame

            # We now compute a phasor that rotates the phase angles of the current input frame by angles theta such
            # that no phase discontinuities occur when re-synthesizing the resulting spectrogram with the synthesis
            # hopsize
            if phase_locking == 0:  # standard phase vocoder: the phase continuity of every bin is preserved separately
                theta = ph_syn + ipa_hop - ph_curr  # phases of the last output frame Instantaneous phase advance
                                                    # Phases of the current input frame
                phasor = np.exp(1j * theta)

            else:   # Phase vocoder with identity phase locking: the phase relationships from the input frame are
                    # partially preserved by 'locking' the phases of bins in the region of influence of a peak in the
                    # sprectrum to the phase of the peak's bin
                p, irs, ire = find_peaks(X[:, i])     # Get the peaks in the spectrum together with their regions of
                                                     # influence
                theta = np.zeros(Y[:, i].shape)

                for n in range(0, len(p)):
                    theta[irs[n]:ire[n]] = ph_syn[p[n]] + ipa_hop[p[n]] - ph_curr[p[n]]  # Phases of the last
                    # output frame, Instantaneous phase advance, Phases of the current input frame

                phasor = np.exp(1j * theta)

            Y[:, i] = phasor * X[:, i]

        # ISTFT
        y_c = istft(Y, syn_hop=syn_hop, w=window, zero_pad=zero_pad, num_of_iter=1, orig_sig_len=output_length,
                    restore_energy=restore_energy, fft_shift=fft_shift)

        y[:, c] = y_c[:, 0]

    return y


def WSOLA_TSM(x, alpha, syn_hop=512, w=win(1024, 2), tol=512):
    """
    Waveform Similarity Overlap and Add algorithm, that rescales the time-axis of the input signal x
    according to the time-stretch function s without altering the pitch of x.

    Parameters
    ----------
    x : np.ndarray [shape=(N,num_of_chan)], real - valued
        Signal to be transformed

    alpha : float or np.ndarray [shape=(S,2)],
        Time stretch function, given by a constant (float) or a set of S anchor points (int)

    syn_hop : int
        hop size of the synthesis window

    w : np.ndarray [shape=(W,)], real - valued
        the analysis and synthesis window for STFT

    tol: int
        Amount of samples the window will be shifted to avoid phase discontinuities when overlap-adding to form the
        output signal

    Returns
    -------
    y : np.ndarray [shape=(L,num_of_chan)], real - valued
        The time-scale modified output signal
    """

    # Pre-calculations
    win_len = len(w)
    win_len_half = np.around(win_len/2).astype(int)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    num_of_chan = x.shape[1]

    # Time-stretch function
    if np.isscalar(alpha):
        anchor_points = np.array([[0, 0], [int(x.shape[0]) - 1, int(np.ceil(alpha * x.shape[0])) - 1]])
    else:
        anchor_points = alpha.astype(int)

    output_length = anchor_points[-1, 1] + 1
    syn_win_pos = np.arange(0, output_length + win_len_half, syn_hop)  # positions of the synthesis winLenHalf
                                                                       # windows in the output

    fi = sc.interpolate.interp1d(anchor_points[:, 1], anchor_points[:, 0], kind='linear',
                                 fill_value='extrapolate')
    ana_win_pos = fi(syn_win_pos)
    ana_win_pos = np.round(ana_win_pos).astype(int)  # positions of the analysis windows in the input
    ana_hop = np.append([0], ana_win_pos[1:] - ana_win_pos[:-1])  # analysis hop sizes

    # WSOLA
    y = np.zeros((output_length, num_of_chan))  # initialize output
    min_fac = np.min(syn_hop/ana_hop[1:])  # the minimal local stretching factor
    # to avoid that we access x outside its range, we need to zero pad it appropriately
    x = np.pad(x, [(win_len_half + tol, int(np.ceil(1 / min_fac)) * win_len + tol), (0, 0)])
    ana_win_pos += tol  # compensate for the extra 'tol' padded zeros at the beginning of x

    for c in range(num_of_chan):  # loop over channels
        x_c = x[:, c]
        y_c = np.zeros((output_length + 2 * win_len, 1))  # initialize the output signal
        ow = np.zeros((output_length + 2 * win_len, 1))  # keep track of overlapping windows
        delay = 0  # shift of the current analysis window position

        for i in range(len(ana_win_pos)-1):
            # OLA
            curr_syn_win_ran = np.arange(syn_win_pos[i], syn_win_pos[i] + win_len, dtype=int)  # range of current
            # synthesis window
            curr_ana_win_ran = np.arange(ana_win_pos[i] + delay, ana_win_pos[i] + win_len + delay, dtype=int)  # range
            # of the current analysis window, shift by 'del' offset
            y_c[curr_syn_win_ran, 0] += x_c[curr_ana_win_ran] * w  # overlap and add
            ow[curr_syn_win_ran, 0] += w  # update the sum of overlapping windows
            nat_prog = x_c[curr_ana_win_ran + syn_hop]  # 'natural progression' of the last copied audio segment
            next_ana_win_ran = np.arange(ana_win_pos[i + 1] - tol, ana_win_pos[i + 1] + win_len + tol, dtype=int)  #
            # range where the next analysis window could be located (including the tolerance region)
            x_next_ana_win_ran = x_c[next_ana_win_ran]  # corresponding segment in x

            # Cross Correlation
            cc = cross_corr(x_next_ana_win_ran, nat_prog, win_len)  # compute the cross correlation
            max_index = np.argmax(cc)  # pick the optimizing index in the cross correlation
            delay = tol - max_index  # infer the new 'delay'

        # process last frame
        y_c[syn_win_pos[-1]:syn_win_pos[-1] + win_len, 0] += x_c[ana_win_pos[i] + delay:ana_win_pos[i] + win_len
                                                                                        + delay] * w
        ow[syn_win_pos[-1]:syn_win_pos[-1] + win_len, 0] += w

        # re-normalize the signal by dividing by the added windows
        ow[ow < 10**(-3)] = 1  # avoid potential division by zero
        y_c /= ow

        # remove zero-padding at the beginning
        y_c = y_c[win_len_half:]

        # remove zero-padding at the end
        y_c = y_c[0:output_length]

        y[:, c] = y_c[:, 0]

    return y


def HPS_TSM(x, Fs, alpha, hps_ana_hop=256, hps_win=win(1024, 2), hps_zero_pad=0, hps_fil_len_harm=10,
            hps_fil_len_perc=10, pv_syn_hop=512, pv_win=win(2048, 2), pv_zero_pad=0, pv_restore_energy=False,
            pv_fft_shift=False, ola_syn_hop=128, ola_win=win(256, 2)):
    """
    Time Scale Modification algorithm based on Harmonic - Percussive separation. After separation is
    performed, the algorithm uses two phase vocoder TSM and WSOLA TSM algorithms for the Harmonic and percussive part
    separately.

    Parameters
    ----------
    x : np.ndarray [shape=(N, )], real - valued
        Signal to be transformed

    Fs :
        Sampling rate

    alpha : float or np.ndarray [shape=(S,2)],
        Time stretch function, given by a constant (float) or a set of S anchor points (int)

    hps_ana_hop : int
        hop size of the synthesis window

    hps_win : np.ndarray [shape=(W,)], real - valued
        the analysis and synthesis window for STFT

    hps_zero_pad : int
        For FFT. Number of zeros padded to the window to increase the fft size

    hps_fil_len_harm: int
        Length of the median filter in time direction.

    hps_fil_len_perc: int
        Length of the median filter in frequency direction.

    pv_syn_hop : int
        hop size for synthesize windows of phase vocoder

    pv_win : np.ndarray
        window for phase vocoder

    pv_zero_pad : int
        phase vocoder zero padding

    pv_restore_energy : bool
        restore energy of signal in phase vocoder

    pv_fft_shift : bool
        fft shift in phase vocoder

    ola_syn_hop : int
        synthesis hop size of OLA

    ola_win : np.ndarray
        window of OLA


    Returns
    -------
    y : np.ndarray [shape=(L,1)], real - valued
        The time-scale modified output signal
    """

    # Harmonic-Percussive Separation
    x_harm, x_perc = hps(x, Fs=Fs, ana_hop=hps_ana_hop, w=hps_win, zero_pad=hps_zero_pad,
                         fil_len_harm=hps_fil_len_harm, fil_len_perc=hps_fil_len_perc, masking_mode='binary')

    # Phase Vocoder for harmonic part
    y_harm = PV_TSM(x_harm, Fs=Fs, alpha=alpha, syn_hop=pv_syn_hop, window=pv_win, zero_pad=pv_zero_pad,
                    restore_energy=pv_restore_energy, fft_shift=pv_fft_shift, phase_locking=True)

    # OLA for percussive part
    y_perc = WSOLA_TSM(x_perc, alpha=alpha, syn_hop=ola_syn_hop, w=ola_win, tol=0)

    # Synthesis
    y = y_harm + y_perc

    return y.reshape(-1, 1)


def two_step_TSM(x, Fs, alpha, order='exact-coarse'):
    """
    Time Scale Modification algorithm, where the signal is stretched by the integer and decimal part
    of the time stretch function using two different algorithms.

    Parameters
    ----------
    x : np.ndarray [shape=(N, )], real - valued
        Signal to be transformed

    Fs : int
        Sampling rate of the input audio signal x

    alpha : float
        Scalar time stretch factor

    order : 'exact-coarse' or 'coarse-exact'
        Decides which of the two time stretching functions will be computed first, coarse corresponding to the integer
        part


    Returns
    -------
    y : np.ndarray [shape=(L,num_of_chan)], real - valued
        The time-scale modified output signal
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    alpha_rough = np.max([1, np.round(alpha)]).astype(int)
    alpha_exact = alpha / alpha_rough

    if order == 'exact-coarse':
        y_exact = HPS_TSM(x, Fs, alpha_exact)
        y = PV_int_TSM(y_exact[:, 0], Fs, alpha_rough)

    elif order == 'coarse-exact':
        y_coarse = PV_int_TSM(x, Fs, alpha_rough)
        y = HPS_TSM(y_coarse[:, 0], Fs, alpha_exact)
    else:
        raise Exception("Invalid order!")

    return y


def pitch_shift_TSM(x, Fs, p, **kwargs):
    """
    Pitch modification algorithm via time-scale modification. The input signal is stretched via TSM and then resampled.

    Parameters
    ----------
    x : np.ndarray [shape=(N, 1)], real - valued
        Signal to be transformed

    Fs : int
        Sampling rate of the input audio signal x

    p : float or np.ndarray [shape=(S,2)], real - valued
        Amount of pitch shifting to be applied, given in cents. Positive n indicates pitch rising, negative n a pitch
        lowering. If an array is given, the first column corresponds to time in seconds and the second column should
        contain the pitch shift in cents.

    **kwargs : Parameters for HPS_TSM

    Returns
    -------
    y : np.ndarray [shape=(L,1)], real - valued
        The time-scale modified output signal
    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    t = np.arange(0, len(x) / Fs, 1 / Fs).reshape(-1, 1)

    # convert shift in cents to anchor points
    if np.isscalar(p):
        alpha = 2 ** (p/1200)
        sig_len = x.shape[0] / Fs
        tau = np.array([[0, 0], [sig_len - 1/Fs, sig_len * alpha - 1/Fs]])
        anchor_points = np.round(tau * Fs).astype(int)

    else:
        # interpolate pitch shift function
        fi = sc.interpolate.interp1d(p[:, 0], p[:, 1], kind='linear', fill_value='extrapolate')
        drifts = fi(t[:, 0])
        p_interp = np.hstack((t, drifts.reshape(-1, 1)))

        # compute alpha
        alpha = np.hstack((t, 2 ** (p_interp[:, 1] / 1200).reshape(-1, 1)))

        # compute tau
        tau = np.zeros(alpha.shape)
        tau[:, 0] = alpha[:, 0]

        for i in range(1, len(t)):
            tau[i, 1] = alpha[i, 1] * 1 / Fs + tau[i - 1, 1]

        # compute anchor points
        anchor_points = np.round(tau * Fs).astype(int)

    # Time-Scale Modification
    y_tsm = HPS_TSM(x, Fs, anchor_points, **kwargs)

    # Resampling
    time_output = np.arange(0, len(y_tsm) / Fs, 1 / Fs)
    fi = sc.interpolate.interp1d(tau[:, 1], tau[:, 0], kind='linear', fill_value="extrapolate")
    time_input = fi(time_output)
    fi = sc.interpolate.interp1d(time_input, y_tsm[:, 0], kind='cubic', fill_value="extrapolate")
    y_ps = fi(t[:, 0])
    y_ps = y_ps.reshape(-1, 1)

    return y_ps


def PV_int_TSM(x, Fs, alpha, syn_hop=512, w=win(2048, 2), zero_pad=-1, restore_energy=False, fft_shift=True):
    """
    Phase Vocoder Time scale modification algorithm, that rescales the time-axis of the input signal x
    according to the time-stretch function s without altering the pitch of x. This algorithm is optimized for integer
    values of the time stretching function.

    Parameters
    ----------
    x : np.ndarray [shape=(N,)], real - valued
        Signal to be transformed

    Fs : int
        Sampling rate of the input audio signal x

    alpha : ndarray, int
        Time stretch factor

    syn_hop : int
        hop size of the synthesis window

    w : np.ndarray [shape=(W,)], real - valued
        the analysis and synthesis window for STFT

    zero_pad : int
        For FFT. Number of zeros padded to the window to increase the fft size

    restore_energy : bool
        For FFT. When True, rescales every windowed synthesis frame to compensate for synthesis energy leakage

    fft_shift: bool
        For FFT. When True, applies a circular shift to each frame of half its length, prior to computing the FFT


    Returns
    -------
    y : np.ndarray [shape=(L,1)], real - valued
        The time-scale modified output signal
    """

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    num_of_chan = x.shape[1]

    if zero_pad == -1:
        zero_pad = alpha * w.shape[0] // 2

    wn = np.hstack((np.zeros((int(np.floor(zero_pad/2)))),
                    w,
                    np.zeros((int(np.floor(zero_pad/2))))))
    win_len = wn.shape[0]
    win_len_half = int(np.round(win_len / 2))

    if (np.isscalar(alpha)) and (np.mod(alpha, 1) == 0) and (alpha >= 1):
        anchor_points = np.array([[0, 0], [int(x.shape[0]), int(np.ceil(alpha * x.shape[0]))]])
    else:
        raise Exception("alpha needs to be an integer >= 1 !")

    while np.mod(syn_hop, alpha) != 0:
        syn_hop = syn_hop + 1

    output_length = int(anchor_points[-1, 1])
    output_window_pos = np.arange(0, output_length + win_len_half, syn_hop)  # positions of the synthesis winLenHalf
    # windows in the output
    input_window_pos = output_window_pos // alpha

    y = np.zeros((output_length, num_of_chan))

    for c in range(num_of_chan):
        # stft
        X, f, t = stft(x[:, c], ana_hop=input_window_pos, w=w, Fs=Fs, num_of_frames=-1, fft_shift=fft_shift,
                       zero_pad=zero_pad)

        # Phase Adaption
        Y = np.abs(X) * np.exp(1j * alpha * np.angle(X))

        # istft
        y_c = istft(Y, syn_hop=syn_hop, w=w, Fs=Fs, zero_pad=zero_pad, num_of_iter=1, orig_sig_len=output_length,
                    restore_energy=restore_energy, fft_shift=fft_shift)

        y[:, c] = y_c[:, 0]

    return y
