

from utils import Fs, Fs_frame
from plot import plot_chromagram, plot_iirt

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
from scipy.ndimage import filters
import scipy.signal as signal
import librosa
import LibFMP



def normalize_feature_sequence(X, norm=2, threshold=0.0001, v=None):
    """
    Normalization taken from the function "normalizeFeature" in Matlab SyncToolbox
    
    The norm value in input is used as the order of normalization of the vector of ones. 
    The resulting normalization is a kind of l^p norm
    
    """

    K, N = X.shape
    X_norm = np.zeros((K, N))
    

    if v is None:
        v = np.ones(12, dtype=np.float64)
        v = v / np.linalg.norm(v, norm);
        
    for n in range(N):
        s = np.linalg.norm(X[:,n], norm)
        
        if s > threshold:
            X_norm[:, n] = X[:, n] / s
            
        else:
            X_norm[:, n] = v


    return X_norm


def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        M: Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average


def list_to_chromagram(note_list, num_frames, frame_rate):
    """Create a chromagram matrix from a list of note events

    Parameters
    ----------
    note_list : List
        A list of note events (e.g. gathered from a CSV file by LibFMP.C1.pianoroll.csv_to_list())

    num_frames : int
        Desired number of frames for the matrix

    frame_rate : float
        Frame rate for C (in Hz)

    Returns
    -------
    C : NumPy Array
        Chromagram matrix
    """
    C = np.zeros((12, num_frames))
    for l in note_list:
        start_frame = max(0, int(l[0] * frame_rate))
        end_frame = min(num_frames, int(l[1] * frame_rate  - 0))   ##-2

        C[int(l[2] % 12), start_frame:end_frame] = l[3]
    return C


def peak_picking_midi(midi_list):
    
    'It computes the pick peaking for the midi'
    
    peak_midi_list = copy.deepcopy(midi_list)
 
    for i in range(len(midi_list)):
        peak_midi_list[i][1] = peak_midi_list[i][0] + 0.0001    ##end must be almost equal to start, to obtain a peak
        #midi_list[i][3] = midi_list[i][3] / 1e6
        
    return peak_midi_list


def half_wave_rectification(D, dur, show_plot=False, matlabAlgorithm=False, filename=""):
    """Function that compute the half wave rectification of the input iirt-filtered signal 
    
    Args:
        D : iirt filtered signal
    
    Returns:
        half_wave_88pitch : half wave rectification of D
    
    """
    fs_index = np.zeros(128)
    fs_index[21:59] = 3
    fs_index[60:95] = 2
    fs_index[96:120] = 1
    
    down_step = [50, 10, 10]
    cutoff = [1/50, 1/10, 0.05]
    
    half_wave_88pitch = np.zeros(D.shape)

    if(matlabAlgorithm==True):
        for i in range(D.shape[0]):   ## for each pitch subband
            index = int(fs_index[i+24])
            Wp = cutoff[index-1]
            Rp = 1
            Rs = 20
            n, Wn = scipy.signal.cheb2ord(Wp, Wp+0.01, Rp, Rs)
            b,a = scipy.signal.cheby2(n, Rs, Wn)
            f_energy = scipy.signal.filtfilt(b,a,D[i, :])
             
            diff_row = np.diff(f_energy, prepend=0)    ##diff of the row, first sample = 0
            diff_row[diff_row<=0] = 0      ##set 0 the values <=0
            half_wave_88pitch[i,:] = diff_row
                
    else:
        diff_row = np.diff(D, prepend=0)    ##diff of the row, first sample = 0
        diff_row[diff_row<=0] = 0      ##set 0 the values <=0
        half_wave_88pitch = diff_row
                
    if(show_plot==True):
        
        fig = plt.figure(figsize=(20,8))
        
        plt.imshow(half_wave_88pitch, origin='lower', aspect = 'auto', cmap=prova_colorbar, extent=[0,dur, 24, 108+1])
      
        chroma_names=np.arange(24,108+1,3)
        plt.yticks(chroma_names)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pitch number')
        cbar = plt.colorbar()
        x_labels = np.arange(0,dur,5)
        plt.xticks(x_labels)
        cbar.set_label('Magnitude')  
        title = "Half-wave rectification  -  " + filename
        plt.title(title) 
        
    return half_wave_88pitch


def scipy_peaks_algorithm(half_wave_audio,  prominence=0.15, offset=0.155, size=100, distance=6):
    
    peak_matrix = np.zeros(half_wave_audio.shape)
    
    shape_array = half_wave_audio.shape[1]
    
    for i in range(half_wave_audio.shape[0]):
        
        peak_array = np.zeros(shape_array)
        current_half_wave = half_wave_audio[i,:]
        
        height = filters.median_filter(current_half_wave, size=size) + offset
        peaks, properties = signal.find_peaks(current_half_wave, prominence=prominence, distance=distance, height=height)

        peak_array[peaks] = current_half_wave[peaks]
        peak_matrix[i,:] = peak_array
        
    return peak_matrix


def compute_max_treshold(pitch_band, pitch_index):
    
    scalar = 2/3       #ORIGINAL VALUE FROM THE ALGORITHM
    #scalar = 2/7.5     

    fs_index = np.ones(128) * 4                 ## value 4 chosen in random way just to be sure not to have ones
    ##initialize parameters as in synctoolbox
    for i in range(fs_index.shape[0]):
        if ((i>=21) & (i<=59)):
            fs_index[i] = 2
        elif ((i>=60) & (i<=95)):
            fs_index[i] = 1     
        elif ((i>=96) & (i<=120)):
            fs_index[i] = 0 

    win_lengths = np.array([100,100,50]) 

    sample_first = 0
    pitch = pitch_index+24
    win_len = win_lengths[int(fs_index[pitch])]
    
    row_len = pitch_band.shape[0]
    row_abs_tresh = np.zeros(row_len)
        
    while sample_first <= row_len:

        sample_last = np.minimum(sample_first + win_len -1, row_len)      ##last peak
        
        windowed_pitch_band = pitch_band[sample_first : sample_last + 1]     ##windowed pitch band
        
        #if(pitch_index==0):
        #    #print(sample_first, sample_last)
        #    print(len(windowed_pitch_band))
        win_max = max(windowed_pitch_band)                               ##max of the window
        abs_thresh = scalar*win_max                                        ##absolute tresh of the window
        
        row_abs_tresh[sample_first:sample_last+1] = abs_thresh
        
        
        sample_first = sample_first+win_len;

    return row_abs_tresh


def peaks(half_wave_audio):
    """Given a half wave rectificated audio signal, this function returns the corresponding peaks values

    Args:
        half_wave_audio (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    scalar = 0.5    #ORIGINAL VALUE FROM ALGORITHM
    row_len = half_wave_audio.shape[1]   
    peaks_matrix = np.zeros(half_wave_audio.shape)
    
    for i in range(half_wave_audio.shape[0]):     ##for each pitch
        pitch_band = half_wave_audio[i, :]
    
        rel_tresh = scalar * np.sqrt(np.var(pitch_band))             ##relative treshold 
        row_rel_tresh = np.ones(row_len) * rel_tresh     ##array of relative treshold (copied for each sample)
        row_descent_thresh = 0.5*row_rel_tresh;
        
        row_abs_tresh = compute_max_treshold(pitch_band, pitch_index = i)
    
        dyold = 0;
        dy = 0;
        rise = 0;               # current amount of ascent during a rising portion of the signal W
        riseold = 0;            # accumulated amount of ascent from the last rising portion of W
        descent = 0;            # current amount of descent (<0) during a falling portion of the signal W
        searching_peak = True;
        candidate = 0;
        P = [];                 ##peaks indices of a single pitch band
        
        
        for j in range(pitch_band.shape[0]-1):         ##for each sample of the row
            dy = pitch_band[j+1] - pitch_band[j]
            
            if(dy>=0):
                rise = rise + dy
                
            else:
                descent = descent + dy
        
            if (dyold >= 0):
                if (dy < 0):        # slope change positive->negative
                    if ((rise >= row_rel_tresh[j]) & (searching_peak == True)):
                        candidate = j
                        searching_peak = False
                    riseold = rise
                    rise = 0
                    
                    
                if (dy < 0): 
                    if ((descent <= -row_rel_tresh[candidate]) & (searching_peak == False)):
                        if (pitch_band[candidate]>=row_abs_tresh[candidate]):
                            P.append(candidate)     # verified candidate as true peak

                        searching_peak = True


                    if (searching_peak == False):      # currently verifying a peak
                        if (pitch_band[candidate] - pitch_band[j] <= row_descent_thresh[j]):
                            rise = riseold + descent    # skip intermediary peak

                        if (descent <= -row_rel_tresh[candidate]):
                            if (pitch_band[candidate]>=row_abs_tresh[candidate]):
                                P.append(candidate)     # verified candidate as true peak

                        searching_peak = True

                    descent = 0

            dyold = dy
            
        peaks_matrix[i,P] = pitch_band[P]
        
    return peaks_matrix
        
    
def find_local_maxima(x, dur):
    
    """Helper function used to calculate the local maximum of each sample of the input
    Algorithm from Chroma SynchToolbox
    Little explanation:
    For each sample, we associate to it the maximum sample in a range of 20 samples before and 20 sample after it
    To do it, we shift two copies of the array for 20 times, one to the right and one to the left
    
    np.floor((maxFilterLength-1)/2) = 20      It`s the window of one second of samples
    maxFilterLength = 41                      It's the total window where we find the maximum
    Args:
        x : input signal
        
    
    Returns:
        y : array_maxima: values of local maxima
        normalized_vector: normalization of the input vector -> x / array_maxima
    
    """
    
    maxFilterLength = 41
    maxFilterTresh = 0.1
    maxFilterTreshArray = np.ones(x.shape[0])*maxFilterTresh
    shift = np.floor((maxFilterLength-1)/2)
    
    f_LN = x;
    f_left = x;
    f_right = x;
    

    for s in np.arange(shift):
            f_left = np.roll(f_left,1);
            f_left[0] = 0;
            f_right = np.roll(f_right,-1);
            f_right[-1] = 0;    
            temp = np.maximum(f_left, f_LN)
            f_LN = np.maximum(temp, f_right)
    
    f_LN = np.maximum(f_LN,maxFilterTreshArray)

    return f_LN



def hps(X, dur, filename ="",show_hps=False, save_fig=False):
    ''' 
    Harmonic Source separation
    X = iirt_audio
    '''
    
    fmp_algorithm_flag=False
    
    if(fmp_algorithm_flag == True):
        L_h = 23  #23
        L_p = 9   #9
        
        ##median filtering
        Y_h = signal.medfilt(X, [1, L_h])
        Y_p = signal.medfilt(X, [L_p, 1])
        
        ##masking
        eps = 0.00001
        M_soft_h = (Y_h + eps/2)/(Y_h + Y_p + eps)
        M_soft_p = (Y_p + eps/2)/(Y_h + Y_p + eps)
    
        ##harmonic and percussive feature
        X_h = X * M_soft_h
        X_p = X * M_soft_p
    
    else:
        X_h, X_p = librosa.decompose.hpss(X, margin=1)
    
    if(show_hps==True):
        plot_iirt(X_h, Fs,dur,"Harmonic Component of IIRT", save_fig=save_fig, save_name=filename, type_name="_harm")
        plot_iirt(X_p, Fs,dur,"Percussive Component of IIRT", save_fig=save_fig, save_name=filename, type_name="_perc")
    
    return X_h, X_p


def compute_cens(C, dur_chroma, Fs_smooth=1, ell=41, d=1, quant=True, transposition=0, show_plot=False):
    """
    C = iirt input (audio or symbolic)
    
    No downsampling, so d=1
    """
    chroma = np.zeros((12, C.shape[1]))
    
    for i in range(C.shape[0]):
        indChroma = np.mod(i+24,12)
        chroma[indChroma, :] += C[i, :]
    
    threshold = 0.001    ##from computCCell in matlab code
    
    chroma_norm = normalize_feature_sequence(chroma, norm=2, threshold=threshold)
    
    ####C_norm = LibFMP.C3.normalize_feature_sequence(C, norm='1')
    
    C_Q = quantize_matrix(chroma_norm) if quant else C_norm

    C_smooth, Fs_CENS = LibFMP.C3.smooth_downsample_feature_sequence(C_Q, Fs_smooth, filt_len=ell,
                                                                    down_sampling=d, w_type='hann')
    C_CENS = normalize_feature_sequence(C_smooth, norm=2)
    
    
    if (transposition!=0):
        C_CENS = np.roll(C_CENS, transposition, axis=0)
    
    if(show_plot==True):
        plot_chromagram(C_CENS, Fs,dur_chroma, title="", 
                        ann="", show_measures="", filename="", save_fig=False)

    return C_CENS


def quantize_matrix(C, quant_fct=None):
    """Quantize matrix values in a logarithmic manner (as done for CENS features)

    Notebook: C7/C7S2_CENS.ipynb

    Args:
        C: Input matrix
        quant_fct: List specifying the quantization function

    Returns:
        C_quant: Output matrix
    """
    C_quant = np.empty_like(C)
    if quant_fct is None:
        quant_fct = [(0.0, 0.05, 0), (0.05, 0.1, 1), (0.1, 0.2, 2), (0.2, 0.4, 3), (0.4, 1, 4)]
    for min_val, max_val, target_val in quant_fct:
        mask = np.logical_and(min_val <= C, C < max_val)
        C_quant[mask] = target_val
    return C_quant




