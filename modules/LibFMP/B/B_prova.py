#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import scipy
import scipy.signal as signal
sys.path.append('..')
import LibFMP.B
import LibFMP.C1
import LibFMP.C3
import sklearn
import pretty_midi
from scipy.ndimage import filters
import pandas as pd
from scipy.interpolate import interp1d

import IPython.display as ipd


# In[2]:


#def load_audio(fn, Fs, start, dur):
#    
#    audio,Fs = librosa.load(fn_wav, sr=Fs)
#    start = start
#    dur = dur
#    start = int(np.floor(start*Fs))
#    end = int(np.ceil(dur*Fs))
#    audio= audio[start: end + start]
#    return x


# In[3]:


#def load_midi(fn_midi, Fs_midi, start, dur):
#
#    start_midi = int(np.floor(start*Fs))
#    end_midi = int(np.ceil(dur*Fs_midi))
#
#
#    midi_data = pretty_midi.PrettyMIDI(fn_midi)
#    midi_list = []
#
#    for instrument in midi_data.instruments:
#        for note in instrument.notes:
#            start = note.start
#            if(start<dur):
#                end = note.end
#                pitch = note.pitch
#                velocity = note.velocity
#                midi_list.append([start, end, pitch, velocity, instrument.name])
#            
#    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))
#
#
#    midi = midi_data.synthesize(fs=Fs)
#
#    midi = x_midi[start_midi: end_midi + start_midi]
#
#    
#    
#    return midi,midi_data
#


# In[4]:


def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    colors[0:1, :] = 1
    #colors[1:end, :3] = colors[1:end, :3]-0.01
    
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    

def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    #grayscale[0:10,:] = 1
    
    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    
    
#view_colormap('copper_r')

prova_colorbar = grayscale_cmap('copper_r')


# In[5]:


Fs = 22050
features_per_second = 50
hop_s = Fs//features_per_second 



# In[6]:


def iirt(x, Fs, dur, show_plot=True, win_l=2048):
    """Function that compute the time-frequency representation of the signal using IIR filters 
    
    Args:
        x : input signal
    
    Returns:
        D : iirt of the signal
    
    """
    
    D = np.abs(librosa.iirt(x, sr=Fs, win_length=win_l, hop_length=hop_s))
    
    
    
    if(show_plot==True):
        figure = plt.subplots(figsize=(20,4))
        
        ylabels=np.arange(24,108+1)
        xlabels=np.arange(0,dur, dur/(D.shape[1]))
        
        img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_coords=xlabels, y_coords=ylabels, y_axis='frames', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title("IIRT Representation")
        
        
        
        #seconds=np.arange(0,dur,1)
        #plt.xticks(seconds)
    
    return D


# In[7]:


def chroma(x, Fs, dur, show_plot=True, win_l=2048):
    """Function that compute the Chromagram representation of the input signal 
    
    Args:
        x : input signal
    
    Returns:
        y : iirt of the signal
    
    """
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, win_length=win_l, hop_length=hop_s, n_fft=win_l)

    
    if(show_plot==True):
        plt.figure(figsize=(20, 4))
        
        xlabels=np.arange(0,dur, dur/(C.shape[1]))
        
        librosa.display.specshow(C, x_axis='time', 
                                 y_axis='chroma', sr=Fs, cmap='gray_r', x_coords=xlabels)
        plt.colorbar();
        
        #seconds=np.arange(0,dur,1)
        #plt.xticks(seconds)
        
        plt.title("Chromagram representation")
    
    return C


# In[8]:


def half_wave_rectification(x):
    
    """Helper function that compute the half wave rectification of the input 
    
    Args:
        x : input signal
    
    Returns:
        y : half wave rectification of the signal
    
    """
    N = x.shape[0]            
    y = np.zeros(x.shape)

    for i in range(N):
        if (i==0):     ##case: first sample of the half wave
            y[0] = 0
        else:
            if (x[i]-x[i-1])>0:               #case: other samples
                y[i] = x[i]-x[i-1]
            else:
                y[i] = 0
    
            
    return y


# In[9]:


def half_wave_88(D, dur, show_plot=False):
    
    """Function that compute the half wave rectification of the input iirt-filtered signal 
    
    Args:
        D : iirt filtered signal
    
    Returns:
        half_wave_88pitch : half wave rectification of D
    
    """
    
    fig = plt.figure(figsize=(20,6))
    half_wave_88pitch = np.zeros(D.shape)
    
    for i in range(D.shape[0]):
        
        current_stmsp = D[i, :]
        current_half_wave = half_wave_rectification(current_stmsp)
        half_wave_88pitch[i,:] = current_half_wave
        
    
        i = i+1
    if(show_plot==True):
        plt.imshow(half_wave_88pitch, origin='lower', aspect = 'auto', cmap=prova_colorbar, extent=[0,dur, 24, 108])
            
        plt.xlabel('Time (seconds)')
        plt.ylabel('88 Pitch')
        cbar = plt.colorbar()
        #plt.clim([0, 4.5])
        cbar.set_label('Magnitude')  
        plt.title("Half-wave rectification with 88 pitch") 
    
    return half_wave_88pitch


# In[10]:


def peaks_picking_88pitch(D, dur, show_plot=True, prominence=0.001, offset=0.001, size=5, distance=5):

    """Function that compute the onset feature of the input iirt-filtered signal with 88 pitch bands
    
    Args:
        D : iirt filtered signal
    
    Returns:
        half_wave_88pitch : onset feature of the input signal, with 88 pitch band
    
    """

    peak_matrix = np.zeros(D.shape)

    for i in range(D.shape[0]):

        current_stmsp = D[i, :]
        peak_array = np.zeros(current_stmsp.shape)           ## single row of the peak matrix
        
        current_half_wave = half_wave_rectification(current_stmsp)
        
        height = filters.median_filter(current_half_wave, size=size) + offset
        peaks, properties = signal.find_peaks(current_half_wave, prominence=prominence, distance=distance, height=height)

        #peaks = librosa.util.peak_pick(current_half_wave, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
        #peaks = peaks.astype(int)
        peak_array[peaks] = current_half_wave[peaks]
        peak_matrix[i,:] = peak_array


        i = i+1
    
    if(show_plot==True):
        figure = plt.figure(figsize=(20,10))
    
        plt.imshow(peak_matrix,origin='lower', aspect = 'auto', cmap=prova_colorbar, extent=[0,dur, 24, 108])
        #plt.clim([0, 0.5])
        
        chroma_names=np.arange(24,108+1,3)

        plt.yticks(chroma_names)
        
        plt.colorbar()
        plt.title("Onset features with 88 pitch")
        plt.xlabel("Time(sec)")
        plt.ylabel("88 Pitch")

    return peak_matrix
    
    


# In[11]:


def chroma_features(x, onset, dur, show_plot=True):

    """Function that sum the features belonging to pitches of the same pitch class of the input signal

    Args:
        x : input signal (with 88 or 12 pitch band)
        onset: boolean value. If True, it computes the onset chroma feature. If false, it computes the chroma half-wave rectification 
    
    Returns:
        y : chroma features
    
    """
    
    
    
    
    y = np.zeros((12, x.shape[1]))
    
    temp = np.zeros(x.shape)
    temp = x
    
    
    num_pitches = 12
    num_octaves = 8
    N_pitches = temp.shape[0]
    
    for i in range(num_pitches):   
        for j in range(num_octaves):
            index = int((j*12 + i))
            if ( index< N_pitches):
                y[i, :] = y[i, :]+np.log(1+10*temp[index, :])
                
    
    if(show_plot==True):
    
        fig = plt.figure(figsize=(20,6))
        
        plt.imshow(y, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur, 0, 11])
        
        chroma_names = 'C C# D D# E F F# G G# A A# B'.split()
        chroma_yticks=np.arange(12)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('12 Pitch ')
        cbar = plt.colorbar()
        plt.clim([0,5])
        cbar.set_label('Magnitude')
        if(onset==False):
            plt.title("Half-wave rectification with 12 pitch")   
        else:
            plt.title("Chroma Onset Features with 12 pitch - CO")
     
    
    return y


# In[12]:



def find_local_maxima(x):
    
    """Helper function used to calculate the local maximum of each sample of the input

    Args:
        x : input signal
        
    
    Returns:
        y : array_maxima: values of local maxima
        normalized_vector: normalization of the input vector -> x / array_maxima
    
    """
    
    tot_samples = x.shape[0]
    normalized_vector = np.zeros(x.shape[0])
    dur_win = 1
    num_samples_half_win = int(np.ceil((tot_samples / dur) * dur_win))       ##num campioni in un secondo di finestra
    num_samples_win = 2 * num_samples_half_win
    
    array_maxima = np.zeros(x.shape)
    
    local_max = 0
    
    for i in range(x.shape[0]):
        
        if (i<num_samples_win):
            left_margin = 0
            right_margin = i+num_samples_half_win
            
        
        elif ((i>=num_samples_half_win) & (i<=tot_samples-num_samples_half_win)):
            left_margin = i-num_samples_half_win
            right_margin = i+num_samples_half_win
            
        elif (i>tot_samples-num_samples_half_win):
            left_margin = i-num_samples_half_win
            right_margin = tot_samples
            
        temp_values = x[left_margin : right_margin]
        local_max = np.max(temp_values)     
        array_maxima[i] = local_max    
        
        if(x[i]>0):
            normalized_vector[i] = x[i] / array_maxima[i]
        
        else:
            normalized_vector[i] = 0


    return array_maxima, normalized_vector
    


# In[13]:


def chroma_onset_feature_vectors(CO, dur,show_plot=True):
    
    """Function that plot each feature vector, one for each pitch

    Args:
        CO : chroma onset features
    
    Returns:
        LNCO : locally adapteive normalized LNCO
    
    """

    N_pitch = CO.shape[0]
    
    if(show_plot==True):
        fig, ax = plt.subplots(N_pitch, 1, figsize=(20,70))
    
    
    num_samples = CO.shape[1]
    array_x = np.arange(0,num_samples) * (dur/num_samples)

    LNCO = np.zeros(CO.shape)
    
    for i in range(N_pitch):
        
        array_maxima, normalized_vector = find_local_maxima(CO[i,:])
        LNCO[i,:] = normalized_vector
        
        if(show_plot==True):
            ax[i].plot(array_x, CO[i,:])
            ax[i].plot(array_x,array_maxima, color="r")
            ax[i].set_title("Chroma Onset Feature vector -> Note: %s" % (LibFMP.C3.note_name(i+24)))
            ax[i].set_xlabel("Time(sec)")
            ax[i].set_ylabel("Magnitude")
            ax[i].grid()
        
        
    return LNCO


# In[14]:


def plot_LNCO(LNCO, dur):
    """Function that plot the LNCO
    
    Args: LNCO
    
    """
    fig = plt.figure(figsize=(20,6))
    plt.imshow(LNCO, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur, 0, 11])
    chroma_names = 'C C# D D# E F F# G G# A A# B'.split()
    chroma_yticks=np.arange(12)
    plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('12 Pitch ')
    plt.colorbar()
    plt.title("Locally adaptive normalized CO features - LNCO")


# In[15]:


def decay_LNCO(x, dur, show_plot=True):
    
    """Function that compute the decay_LNCO

    Args:
        x : LNCO
    
    Returns:
        decay_LNCO: decay LNCO
    
    """
    
    
    n = 10
    weights =(np.sqrt(np.arange(0.1, 1.1, 0.1)))
    weights = weights[::-1]
    
    decay_LNCO = np.zeros(x.shape)
    
    
    
    for i in range(x.shape[0]):    #for each pitch
        temp_dLNCO_row = np.zeros(x.shape[1])
        temp_LNCO_row = x[i,:]
        for j in range (n):  #for each shifted sequence
            delay_row = np.zeros(x.shape[1])
            delay_row[j:-1] = temp_LNCO_row[0:-1-j]
            temp_dLNCO_row = temp_dLNCO_row + (delay_row*weights[j])
            
        decay_LNCO[i,:] = temp_dLNCO_row
    
    decay_LNCO = sklearn.preprocessing.normalize(decay_LNCO, norm='max')
    
    
    if(show_plot==True):
        fig = plt.figure(figsize=(20,6))
        plt.imshow(decay_LNCO, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur, 0, 11])
        chroma_names = 'C C# D D# E F F# G G# A A# B'.split()
        chroma_yticks=np.arange(12)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('12 Pitch ')
        plt.colorbar()
        plt.title("Decay Locally adaptive normalized CO features - dLNCO")
    
    return decay_LNCO





# In[ ]:





# In[16]:


#fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Scherbakov.wav')

#fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Sibelius-Piano.wav')


fn_wav = os.path.join('..', 'data', 'C3', 'FMP_C3_F05_BurgmuellerFirstPart.mp3')


x, Fs = librosa.load(fn_wav, sr=Fs)



start_sec = 0
dur = 6.2
start = int(np.floor(start_sec*Fs))

end = int(np.ceil(dur*Fs))
x = x[start: end + start]
#dur = x.shape[0]/Fs



ipd.Audio(x, rate=Fs)


# In[17]:


#window=2048//4
window=882

D = iirt(x, Fs, show_plot=True, win_l=window, dur=dur)


# In[18]:


xlabels=np.arange(0,dur, dur/(D.shape[1]))
print(xlabels.shape)
print(D.shape)


# In[19]:


C_wave = chroma(x, Fs, show_plot=True, win_l=window, dur=dur)
print(C_wave.shape)


# In[20]:


half_wave_88pitch = half_wave_88(D, dur=dur, show_plot=False)


# In[21]:


half_wave_12pitch = chroma_features(D, onset=False, dur=dur, show_plot=False)


# In[22]:


ipd.Audio(x, rate=Fs)


# In[23]:


peak_matrix = peaks_picking_88pitch(D, dur=dur, show_plot=True,prominence=0.005, offset=0.0199 , size=150, distance=10 )   ##BURGMULLER

#peak_matrix = peaks_picking_88pitch(D, dur=dur, show_plot=True,prominence=0.37, offset=0.4 , size=100, distance=12)   ##BEETHOVEN 1

#peak_matrix = peaks_picking_88pitch(D, dur=dur, show_plot=True,prominence=0.095, offset=0.07 , size=100, distance=10)   ##BEETHOVEN 2


# In[24]:


CO = chroma_features(peak_matrix, onset=True, dur=dur, show_plot=True)


# In[25]:


LNCO = chroma_onset_feature_vectors(CO, dur=dur, show_plot=True)


# In[26]:


plot_LNCO(LNCO, dur=dur)


# In[27]:


d_LNCO = decay_LNCO(LNCO, dur=dur, show_plot=True)


# ## MIDI FILE 

# In[28]:


#fn_midi = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Midi-Piano.mp3')


#fn_midi = os.path.join('..', 'data', 'C3', 'arabesque.mp3')


# In[29]:


Fs_midi = 22050
start_sec = 0
start_midi = int(np.floor(start_sec*Fs))
dur_midi = 4.5
end_midi = int(np.ceil(dur_midi*Fs_midi))

fn_midi = os.path.join('..', 'data', 'C3', 'arabesque.mid')

midi_data = pretty_midi.PrettyMIDI(fn_midi)
midi_list = []

for instrument in midi_data.instruments:
    for note in instrument.notes:
        start = note.start
        if(start<dur_midi):
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity, instrument.name])
        
midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))





Fs = 22050
x_midi = midi_data.synthesize(fs=Fs)
#x_midi, Fs_midi = librosa.load(fn_midi, sr=Fs_midi)
x_midi = x_midi[start_midi: end_midi + start_midi]



ipd.Audio(x_midi, rate=Fs_midi)

#df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
#html = df.to_html(index=False)
#ipd.HTML(html)


# In[30]:


ipd.Audio(x, rate=Fs)


# In[31]:


#window_midi=2048
window_midi=882

D_midi = iirt(x_midi, Fs_midi, show_plot=True,win_l=window_midi, dur=dur_midi)


# In[32]:


C_midi = chroma(x_midi, Fs_midi, show_plot=True,win_l=window_midi, dur=dur_midi)


# In[33]:



peak_matrix_midi = peaks_picking_88pitch(D_midi, dur=dur_midi, show_plot=True, prominence=0.15, offset=0.155, size=100, distance=6)  ##BURG AUDIO

#peak_matrix_midi = peaks_picking_88pitch(D_midi, dur=dur_midi, show_plot=True, prominence=0.03, offset=0.002, size=100, distance=7)   ##BURG MIDI

CO_midi = chroma_features(peak_matrix_midi, dur=dur_midi, onset=True, show_plot=False)

LNCO_midi = chroma_onset_feature_vectors(CO_midi, dur=dur_midi, show_plot=False)


# In[34]:


d_LNCO = decay_LNCO(LNCO, dur=dur, show_plot=True)
d_LNCO_midi = decay_LNCO(LNCO_midi, dur=dur_midi, show_plot=True)


# ## SINCRONIZATION WAV FILE & ANNOTATION MIDI FILE

# In[35]:


C_wave = chroma(x, Fs_midi, show_plot=False, win_l=window, dur=dur)
C_midi = chroma(x_midi, Fs_midi, show_plot=False, win_l=window_midi, dur=dur_midi)

    

    
##for i in range(C_midi.shape[0]):
##    for j in range(C_midi.shape[1]):
##        if (C_midi[i][j]==0):
##            C_midi[i][j] = 0.000000000000000000000000000000000000000000000000000000001


# In[36]:


C_chroma = LibFMP.C3.compute_cost_matrix(C_wave, C_midi, metric='cosine')
D_chroma = LibFMP.C3.compute_accumulated_cost_matrix(C_chroma)
#P_chroma_jes = LibFMP.C3.compute_optimal_warping_path(D_chroma)

#if np.any(np.isnan(P_chroma_jes)):
#    print("we")

def replace_nan_values(C_chroma):

    for i in range(C_chroma.shape[0]):
        for j in range(C_chroma.shape[1]):
            if (np.isnan(C_chroma[i][j])):
                C_chroma[i][j] = 1

                
replace_nan_values(C_chroma)


step_sizes_sigma = np.array([[1, 0], [0, 1], [1, 1]])
weights_mul = np.array([1,1,1])

#weights_mul = np.array([2, 1.5, 1.5])
#step_sizes_sigma = np.array([[1, 1], [2, 1], [1, 2]])

#step_sizes_sigma = np.array([[1, 1], [3, 1], [1, 3]])


D_chroma, P_chroma = librosa.sequence.dtw(C_wave, C_midi, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul, global_constraints=False, band_rad=0.25)

P_chroma = P_chroma[::-1]


# In[37]:


C_dnlco = LibFMP.C3.compute_cost_matrix(d_LNCO, d_LNCO_midi, metric='euclidean')
D_dnlco = LibFMP.C3.compute_accumulated_cost_matrix(C_dnlco)
#P_dnlco = LibFMP.C3.compute_optimal_warping_path(D_dnlco)


#step_sizes_sigma = np.array([[1, 1], [2, 1], [1, 2]])
#weights_mul = np.array([2, 1.5, 1.5])

D_dnlco, P_dnlco = librosa.sequence.dtw(d_LNCO, d_LNCO_midi, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)

P_dnlco=P_dnlco[::-1]


# In[38]:


C_sum = C_dnlco + C_chroma



#D_sum = LibFMP.C3.compute_accumulated_cost_matrix(C_sum)
#P_sum = LibFMP.C3.compute_optimal_warping_path(D_sum)

D_sum, P_sum= librosa.sequence.dtw(C=C_sum, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)

P_sum = P_sum[::-1]


# In[39]:


fig, ax = plt.subplots(1,3,figsize=(20,6))

ax[0] = plt.subplot(1, 3, 1)
LibFMP.C3.plot_matrix_with_points(C_chroma, P_chroma, linestyle='-', ax=[ax[0]], aspect='equal', clim=[0, np.max(C_chroma)], title='C_chroma', xlabel='Midi File', ylabel='Audio File');

ax[1] = plt.subplot(1, 3, 2)
LibFMP.C3.plot_matrix_with_points(C_dnlco, P_dnlco, linestyle='-', ax=[ax[1]], aspect='equal', clim=[0, np.max(C_dnlco)], title='C_DNLCO', xlabel='Midi File', ylabel='Audio File');

ax[2] = plt.subplot(1, 3, 3)
LibFMP.C3.plot_matrix_with_points(C_sum, P_sum, linestyle='-', ax=[ax[2]], aspect='equal', clim=[0, np.max(C_sum)], title='C_sum', xlabel='Midi File', ylabel='Audio File');


# In[40]:


P_chroma_align = LibFMP.C3.compute_strict_alignment_path(P_chroma)
P_dnlco_align = LibFMP.C3.compute_strict_alignment_path(P_dnlco)
P_sum_align = LibFMP.C3.compute_strict_alignment_path(P_sum)



fig, ax = plt.subplots(1,3,figsize=(20,6))

ax[0] = plt.subplot(1, 3, 1)
LibFMP.C3.plot_matrix_with_points(C_chroma, P_chroma_align, linestyle='-', ax=[ax[0]], aspect='equal', clim=[0, np.max(C_chroma)], title='C_chroma with P aligned' , xlabel='Midi File', ylabel='Audio File');

ax[1] = plt.subplot(1, 3, 2)
LibFMP.C3.plot_matrix_with_points(C_dnlco, P_dnlco_align, linestyle='-', ax=[ax[1]], aspect='equal', clim=[0, np.max(C_dnlco)], title='C_DNLCO with P aligned' , xlabel='Midi File', ylabel='Audio File');

ax[2] = plt.subplot(1, 3, 3)
LibFMP.C3.plot_matrix_with_points(C_sum, P_sum_align, linestyle='-', ax=[ax[2]], aspect='equal', clim=[0, np.max(C_sum)], title='C_sum with P aligned' , xlabel='Midi File', ylabel='Audio File');


# In[41]:


import math
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def normalization_frames_to_second(vector,dur_x,dur_y):
    
    'Fucntion used to translate the alignment in terms of features in an alignment in terms of seconds '
    
    'dur_x = duration of the audio file'
    'dur_y = duration of the midi file'
    y = np.zeros(vector.shape)
    t_max_row = np.max(vector[:,0])
    t_max_col = np.max(vector[:,1])
    
    for i in np.arange(vector.shape[0]):
        y[i,0] = ( (vector[i,0] - 0 ) / ((t_max_row) - 0) ) * (dur_x - 0) + 0
        y[i,1] = ( (vector[i,1] - 0 ) / ((t_max_col) - 0) )  * (dur_y - 0) + 0
        
        y[i,0] = truncate(y[i,0], 3)
        y[i,1] = truncate(y[i,1], 3)
    
    return y


P_sum_align_seconds=normalization_frames_to_second(vector = P_sum_align, dur_x = dur, dur_y = dur_midi)
P_dnlco_align_seconds=normalization_frames_to_second(vector = P_dnlco_align, dur_x = dur, dur_y = dur_midi)

#print(P_sum_align)


# In[42]:


#df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
#html = df.to_html(index=False)
#ipd.HTML(html)


# In[43]:


#print("Audio   Midi\n")
#
#for i in range(P_sum_align_seconds.shape[0]):
#    print("%.3f " % P_sum_align_seconds[i, 0], " %.3f" % P_sum_align_seconds[i, 1])
    
    


# In[44]:


#result = np.where(P_sum_align_seconds[:,1] == 0.279)
#print(result[0])
#
#P_sum_align_seconds[11,1]


# In[45]:


def list_to_pitch_activations(note_list, num_frames, frame_rate):
    """Create a pitch activation matrix from a list of note events

    Parameters
    ----------
    note_list : List
        A list of note events (e.g. gathered from a CSV file by LibFMP.C1.pianoroll.csv_to_list())

    num_frames : int
        Desired number of frames for the matrix

    frame_rate : float
        Frame rate for P (in Hz)

    Returns
    -------
    P : NumPy Array
        Pitch activation matrix
        First axis: Indexed by [0:127], encoding MIDI pitches [1:128]
    F_coef_MIDI: MIDI pitch axis
    """
    P = np.zeros((128, num_frames))
    F_coef_MIDI = np.arange(128) + 1
    for l in note_list:
        start_frame = max(0, int(l[0] * frame_rate))
        #end_frame = min(num_frames, int((l[0] + l[1]) * frame_rate) + 1)
        end_frame = min(num_frames, int(l[1] * frame_rate) - 2)
        #end_frame = start_frame+5
        P[int(l[2]-1), start_frame:end_frame] = 1
    return P, F_coef_MIDI


# In[46]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def warping_midi(audio, midi_data, warping_path, dur_midi, show_alignment=True, start_trick=False):
    
    'Function used to align the two paths'
    midi_list_warped = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            if(start<=dur_midi):
                
                index_start_WP = find_nearest(warping_path[:,1], start)
                index_end_WP = find_nearest(warping_path[:, 1], end)
                
                if(start==0 and start_trick==True):   ##it works with the burgmuller
                    index_start_WP+=2
                    index_end_WP+=2
                
                midi_interp_start = [warping_path[index_start_WP,1], warping_path[index_start_WP-1,1]]
                audio_interp_start = [warping_path[index_start_WP,0], warping_path[index_start_WP-1,0]]
                
                midi_interp_end = [warping_path[index_end_WP,1], warping_path[index_end_WP-1,1]]
                audio_interp_end = [warping_path[index_end_WP,0], warping_path[index_end_WP-1,0]]
                
                
                new_start = interp1d(midi_interp_start, audio_interp_start, kind='linear', fill_value='extrapolate')(start)
                new_end = interp1d(midi_interp_end, audio_interp_end, kind='linear', fill_value='extrapolate')(end)
                #new_end = end + 0.2
                
                
                
                
                pitch = note.pitch
                velocity = note.velocity
                midi_list_warped.append([new_start, new_end, pitch, velocity, instrument.name])
        
    midi_list_warped = sorted(midi_list_warped, key=lambda x: (x[0], x[2]))

    
    if(show_alignment==True):
        H = hop_s
        num_frames = int(len(audio) / H)
        Fs_frame = Fs / H
        X_ann, F_coef_MIDI = list_to_pitch_activations(midi_list_warped,num_frames, Fs_frame)
        
        #plt.imshow(X_ann)
        
        harmonics = [1, 1/2, 1/3, 1/4, 1/5]
        fading_msec = 0.5
        x_pitch_ann, x_pitch_ann_stereo = LibFMP.B.sonify_pitch_activations_with_signal(X_ann, audio, Fs_frame, Fs, fading_msec=fading_msec, harmonics_weights=harmonics)
    
        ipd.display(ipd.Audio(x_pitch_ann_stereo, rate=Fs_midi) )

    return midi_list_warped


# In[47]:


print("BURGMULLER Arabesque - Synchronization between Audio and Midi using the warping path")
midi_list_warped = warping_midi(audio = x, midi_data= midi_data, warping_path = P_sum_align_seconds, dur_midi=dur_midi, show_alignment=True, start_trick=True)


# In[48]:


#df = pd.DataFrame(midi_list_warped, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
#html = df.to_html(index=False)
#ipd.HTML(html)


# In[49]:


##RESUME FUNCTION

def get_chroma_dnlco_audio(x,Fs, window, dur, prominence, offset, size, distance, 
                           show_iirt=False, show_chroma=False, show_onset=False, show_onset_chroma=False, show_LNCO=False, show_dLNCO=False):
    
    D = iirt(x, Fs, show_plot=show_iirt, win_l=window, dur=dur)
    C_wave = chroma(x, Fs, show_plot=show_chroma, win_l=window, dur=dur)
    peak_matrix = peaks_picking_88pitch(D, dur=dur, show_plot=show_onset,prominence=prominence, offset=offset , size=size, distance=distance )
    CO = chroma_features(peak_matrix, dur=dur, onset=True, show_plot=show_onset_chroma)
    LNCO = chroma_onset_feature_vectors(CO, dur=dur, show_plot=show_LNCO)
    d_LNCO = decay_LNCO(LNCO, dur=dur, show_plot=show_dLNCO)
    return C_wave, d_LNCO

def get_chroma_dnlco_midi(x_midi,Fs, window, dur_midi, prominence, offset, size, distance, 
                          show_iirt=False, show_chroma=False, show_onset=False, show_onset_chroma=False, show_LNCO=False, show_dLNCO=False):
    
    D_midi = iirt(x_midi, Fs_midi, show_plot=show_iirt,win_l=window_midi, dur=dur_midi)
    C_midi = chroma(x_midi, Fs_midi, show_plot=show_chroma,win_l=window_midi, dur=dur_midi)
    peak_matrix_midi = peaks_picking_88pitch(D_midi, dur=dur_midi, show_plot=show_onset, prominence=prominence, offset=offset, size=size, distance=distance)  ##BURG AUDIO
    CO_midi = chroma_features(peak_matrix_midi, dur=dur_midi, onset=True, show_plot=show_onset_chroma)
    LNCO_midi = chroma_onset_feature_vectors(CO_midi, dur=dur_midi, show_plot=show_LNCO)
    d_LNCO_midi = decay_LNCO(LNCO_midi, dur=dur_midi, show_plot=show_dLNCO)
    return C_midi, d_LNCO_midi


def get_alignment_path(audio, midi_data, d_LNCO, C_wave, d_LNCO_midi, C_midi, dur_audio, 
                       dur_midi, step_sizes_sigma, weights_mul,  align_flag=True, return_chroma_wp=False, show_wp=False):
    
    
    #WARPING PATH CHROMA
    C_chroma = LibFMP.C3.compute_cost_matrix(C_wave, C_midi, metric='cosine')
    D_chroma = LibFMP.C3.compute_accumulated_cost_matrix(C_chroma)
    replace_nan_values(C_chroma)
    D_chroma, P_chroma = librosa.sequence.dtw(C_wave, C_midi, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul, global_constraints=False, band_rad=0.25)
    P_chroma = P_chroma[::-1]
    
    #WARPING PATH DNLCO
    C_dnlco = LibFMP.C3.compute_cost_matrix(d_LNCO, d_LNCO_midi, metric='euclidean')
    D_dnlco = LibFMP.C3.compute_accumulated_cost_matrix(C_dnlco)
    D_dnlco, P_dnlco = librosa.sequence.dtw(d_LNCO, d_LNCO_midi, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
    P_dnlco=P_dnlco[::-1]
    
    ##WARPING PATH CHROMA + DNLCO
    C_sum = C_dnlco + C_chroma
    D_sum, P_sum= librosa.sequence.dtw(C=C_sum, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
    P_sum = P_sum[::-1]
    
    
    #ALIGN PATH
    P_chroma_align = LibFMP.C3.compute_strict_alignment_path(P_chroma)
    P_dnlco_align = LibFMP.C3.compute_strict_alignment_path(P_dnlco)
    P_sum_align = LibFMP.C3.compute_strict_alignment_path(P_sum)
    
    
    if (show_wp==True):
        ##PLOT
        fig, ax = plt.subplots(1,3,figsize=(20,6))
    
        ax[0] = plt.subplot(1, 3, 1)
        LibFMP.C3.plot_matrix_with_points(C_chroma, P_chroma_align, linestyle='-', ax=[ax[0]], aspect='equal', clim=[0, np.max(C_chroma)], title='C_chroma with P aligned' , xlabel='Midi File', ylabel='Audio File');
    
        ax[1] = plt.subplot(1, 3, 2)
        LibFMP.C3.plot_matrix_with_points(C_dnlco, P_dnlco_align, linestyle='-', ax=[ax[1]], aspect='equal', clim=[0, np.max(C_dnlco)], title='C_DNLCO with P aligned' , xlabel='Midi File', ylabel='Audio File');
    
        ax[2] = plt.subplot(1, 3, 3)
        LibFMP.C3.plot_matrix_with_points(C_sum, P_sum_align, linestyle='-', ax=[ax[2]], aspect='equal', clim=[0, np.max(C_sum)], title='C_sum with P aligned' , xlabel='Midi File', ylabel='Audio File');
        
        #TRANSLATION FRAMES TO SECONDS
    P_sum_align_seconds=normalization_frames_to_second(vector = P_sum_align, dur_x = dur_audio, dur_y= dur_midi)
    P_sum_seconds=normalization_frames_to_second(vector = P_sum, dur_x = dur_audio, dur_y= dur_midi)

    P_chroma_align_seconds = normalization_frames_to_second(vector = P_chroma_align, dur_x = dur_audio, dur_y= dur_midi)

    
    
    
    if(return_chroma_wp==True):
        return P_chroma_align_seconds
    
    else:
        return P_sum_align_seconds
    
    #if(align_flag==True):
    #    midi_list_warped = warping_midi(audio=audio, midi_data = midi_data, warping_path = P_sum_align_seconds, dur_midi=dur_midi, show_alignment=True)
    #else:
    #    midi_list_warped = warping_midi(audio=audio, midi_data = midi_data, warping_path = P_sum_seconds, dur_midi=dur_midi, show_alignment=True)


# ## BEETHOVEN FIFTH

# In[50]:


def load_audio(fn, Fs, start, dur):
    
    audio,Fs = librosa.load(fn, sr=Fs)
    start = start
    dur = dur
    start = int(np.floor(start*Fs))
    end = int(np.ceil(dur*Fs))
    audio= audio[start: end + start]
    return audio

def load_midi(fn_midi, Fs_midi, start, dur):

    start_midi = int(np.floor(start*Fs))
    end_midi = int(np.ceil(dur*Fs_midi))

    midi_data = pretty_midi.PrettyMIDI(fn_midi)
    midi_list = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            if(start<dur):
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                midi_list.append([start, end, pitch, velocity, instrument.name])
            
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))


    midi = midi_data.synthesize(fs=Fs)

    midi = midi[start_midi: end_midi + start_midi]
    
    
    return midi,midi_data


# In[51]:


fn_wav_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Sibelius-Piano_Fermata.wav')
#fn_wav_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Scherbakov.wav')
dur_b = 22
#dur_b=18

x_b= load_audio(fn = fn_wav_b, Fs=Fs, start=0, dur=dur_b)
ipd.Audio(x_b,rate=Fs)


# In[52]:


fn_midi_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Sibelius-Piano_Fermata.mid')
dur_midi_b = 22
x_midi_b, midi_data_b = load_midi(fn_midi = fn_midi_b, Fs_midi=Fs, start=0, dur=dur_midi_b )
ipd.Audio(x_midi_b, rate=22050)


# In[53]:


prominence_audio=0.165
offset_audio=0.02
size_audio=100
distance_audio=10

prominence_midi=0.1
offset_midi=0.05
size_midi=100
distance_midi=6

#step_sizes_sigma = np.array([[1, 0], [0, 1], [1, 1]])
#weights_mul = np.array([1,1,1])
weights_mul = np.array([2, 1.5, 1.5])
step_sizes_sigma = np.array([[1, 1], [2, 1], [1, 2]])

#step_sizes_sigma = np.array([[1, 1], [3, 1], [1, 3]])

C_wave_b, d_LNCO_b = get_chroma_dnlco_audio(x_b,Fs, window, dur_b, prominence_audio, offset_audio, size_audio, distance_audio, show_onset_chroma=False)
C_midi_b, d_LNCO_midi_b = get_chroma_dnlco_midi(x_midi_b,Fs, window, dur_midi_b, prominence_midi, offset_midi, size_midi, distance_midi, show_onset_chroma=False, show_onset=True)
warping_path_b = get_alignment_path(x_b, midi_data_b, d_LNCO_b, C_wave_b, d_LNCO_midi_b, C_midi_b, dur_audio = dur_b, dur_midi=dur_midi_b, step_sizes_sigma=step_sizes_sigma, weights_mul=weights_mul, align_flag=True, return_chroma_wp=False)


# In[54]:


print("Beethoven Fifth - Synchronization between Audio and Midi using the warping path")
midi_list_warped = warping_midi(audio=x_b, midi_data = midi_data_b, warping_path = warping_path_b, dur_midi=dur_midi_b, show_alignment=True)



#df = pd.DataFrame(midi_list_warped, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
#html = df.to_html(index=False)
#ipd.HTML(html)


# ## SCHERBAKOV -  BEETHOVEN 5th

# In[55]:


#fn_wav_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Sibelius-Piano_Fermata.wav')
fn_wav_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Scherbakov.wav')
#dur_b = 22
dur_b=18

x_b= load_audio(fn = fn_wav_b, Fs=Fs, start=0, dur=dur_b)
ipd.Audio(x_b,rate=Fs)


# In[56]:


fn_midi_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Beethoven_Fifth-MM1-20_Sibelius-Piano.mid')
dur_midi_b = 22
x_midi_b, midi_data_b = load_midi(fn_midi = fn_midi_b, Fs_midi=Fs, start=0, dur=dur_midi_b )
ipd.Audio(x_midi_b, rate=22050)


# In[57]:


#prominence_audio=0.006
#offset_audio=0.001
#size_audio=5
#distance_audio=10

prominence_audio=0.008
offset_audio=0.0065
size_audio=15
distance_audio=10

prominence_midi=0.1
offset_midi=0.05
size_midi=100
distance_midi=6

#step_sizes_sigma = np.array([[1, 0], [0, 1], [1, 1]])
#weights_mul = np.array([1,1,1])
weights_mul = np.array([2, 1.5, 1.5])
step_sizes_sigma = np.array([[1, 1], [2, 1], [1, 2]])

#step_sizes_sigma = np.array([[1, 1], [3, 1], [1, 3]])

C_wave_b, d_LNCO_b = get_chroma_dnlco_audio(x_b,Fs, window, dur_b, prominence_audio, offset_audio, size_audio, distance_audio, show_dLNCO=True)
C_midi_b, d_LNCO_midi_b = get_chroma_dnlco_midi(x_midi_b,Fs, window, dur_midi_b, prominence_midi, offset_midi, size_midi, distance_midi, show_dLNCO=True)
warping_path_b = get_alignment_path(x_b, midi_data_b, d_LNCO_b, C_wave_b, d_LNCO_midi_b, C_midi_b, dur_audio = dur_b,
                                    dur_midi=dur_midi_b, step_sizes_sigma=step_sizes_sigma, weights_mul=weights_mul, align_flag=True, return_chroma_wp=True, show_wp=True)


# In[58]:


print("Beethoven Fifth - Synchronization between Audio and Midi using the warping path")
midi_list_warped = warping_midi(audio=x_b, midi_data = midi_data_b, warping_path = warping_path_b, dur_midi=dur_midi_b, show_alignment=True)



#df = pd.DataFrame(midi_list_warped, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
#html = df.to_html(index=False)
#ipd.HTML(html)


# ## SCHUMANN - TRAEUMEREI

# In[59]:


fn_wav_b = os.path.join('..', 'data', 'C3', 'FMP_C3S3_Schumann_Op15No7_Traeumerei_HernandezRomero.wav')

dur_b=52

x_b= load_audio(fn = fn_wav_b, Fs=Fs, start=0, dur=dur_b)
ipd.Audio(x_b,rate=Fs)


# In[60]:


fn_midi_b = os.path.join('..', 'data', 'C3', 'schumann_traeumerei.mid')
dur_midi_b = 22
x_midi_b, midi_data_b = load_midi(fn_midi = fn_midi_b, Fs_midi=Fs, start=0, dur=dur_midi_b )
ipd.Audio(x_midi_b, rate=22050)


# In[61]:


#prominence_audio=0.006
#offset_audio=0.001
#size_audio=5
#distance_audio=10

prominence_audio=0.005
offset_audio=0.0065
size_audio=100
distance_audio=15

prominence_midi=0.001
offset_midi=0.1
size_midi=100
distance_midi=10

step_sizes_sigma = np.array([[1, 0], [0, 1], [1, 1]])
weights_mul = np.array([1,1,1])
#weights_mul = np.array([2, 1.5, 1.5])
#step_sizes_sigma = np.array([[1, 1], [2, 1], [1, 2]])

#step_sizes_sigma = np.array([[1, 1], [3, 1], [1, 3]])

C_wave_b, d_LNCO_b = get_chroma_dnlco_audio(x_b,Fs, window, dur_b, prominence_audio, offset_audio, size_audio, distance_audio, show_onset=True)
C_midi_b, d_LNCO_midi_b = get_chroma_dnlco_midi(x_midi_b,Fs, window, dur_midi_b, prominence_midi, offset_midi, size_midi, distance_midi, show_onset=True)
warping_path_b = get_alignment_path(x_b, midi_data_b, d_LNCO_b, C_wave_b, d_LNCO_midi_b, C_midi_b, dur_audio = dur_b,
                                    dur_midi=dur_midi_b, step_sizes_sigma=step_sizes_sigma, weights_mul=weights_mul, align_flag=True, return_chroma_wp=True, show_wp=True)


# In[62]:


print("Schumamm - Traeumerei, Synchronization between Audio and Midi using the warping path")
midi_list_warped = warping_midi(audio=x_b, midi_data = midi_data_b, warping_path = warping_path_b, dur_midi=dur_midi_b, show_alignment=True)

