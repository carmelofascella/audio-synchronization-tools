import numpy as np
from  matplotlib import pyplot as plt
import librosa

def visualizeAP(fs,anchorpoints):
    """
    Written by: Edgar Suarez, 08.2019.
    Plot a set of anchorpoints.
    
    Parameters
    ----------
    fs : int
        Sampling rate of the input audio signal
    
    anchorpoints : np.ndarray [shape=(Sx2)], real - valued
        Set of S anchorpoints. 
        An anchorpoint is a pair of sample positions [l k] where position l of the input is mapped to sample k on the TSM algorithm
    """
    apIn = anchorpoints[:,0]/fs
    apOut = anchorpoints[:,1]/fs
    
    plt.figure(figsize=(9, 6))

    plt.plot(apIn,apOut, '+r',alpha=0.8, markersize=10, label = 'Anchorpoints')
    plt.plot(apIn,apOut, 'k', label = 'Linear Interpolation')
    
    plt.title('Anchor Points')
    plt.xlabel('Time-axis Input (s)')
    plt.ylabel('Time-axis Output (s)')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
def visualizeSpec(x,fs,win=np.hanning(1024),hop=512,logComp=10,visFreqRange=[0],title=''):
    """
    Written by: Edgar Suarez, 08.2019.
    Computes and plots a signal's a magnitude Spectrogram.
    
    Parameters
    ----------
    x : np.ndarray [shape=(N, )], real - valued
        Input signal
    
    fs : int
        Sampling rate of the input audio signal

    win : np.ndarray [shape=(W,)], real - valued
        the analysis window for STFT  

    hop : int
        hop size of the analysis window
    
    log_comp : int
        Logarithmic Compresson factor, for enhanced spectrogram visualization
        
    visFreqRange : np.ndarray [shape=(2, )], real - valued
        Vector specifying the Frequency visualization range
    """
    W = len(win)
    H = hop
    left = -(H/2)/fs
    right = (len(x)+H/2)/fs
    lower = 0
    upper =  fs/2
    
    spec = librosa.stft(x, n_fft=W, hop_length=H, win_length=W, window=win, center=True)
    spec = np.log(1+logComp*np.abs(spec))  # logarithmic compression for better visualization
    
    plt.figure(figsize=(7, 4))
    plt.imshow(spec, cmap='gray_r', aspect='auto', origin='lower',extent=[left, right, lower, upper])
    plt.title(title)
    if len(visFreqRange)==2:
        plt.axis([left, right, visFreqRange[0], visFreqRange[1]])
    else:
        plt.axis([left,right,lower,upper])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')   
    plt.tight_layout()
        
def visualizeWav(x,fs,ampRange=[-0.8,0.8],timeRange=None,title=''):
    """
    Written by: Edgar Suarez, 08.2019.
    Computes and plots a signal's a magnitude Spectrogram.
    
    Parameters
    ----------
    x : np.ndarray [shape=(N, )], real - valued
        Input signal
    
    fs : int
        Sampling rate of the input audio signal
        
    ampRange : np.ndarray [shape=(2, )], real - valued
        Vector specifying the amplitude visualization range
        
    timeRange : np.ndarray [shape=(2, )], real - valued
        Vector specifying the time visualization range
    
    title : char
        Graph title
    """    
    if timeRange == None:
        timeRange = [0,len(x)/fs]
    
    t = np.arange(0,len(x))/fs
    
    plt.figure(figsize=(7, 3))
    plt.plot(t,x, 'b')
    plt.title(title)
    plt.axis([timeRange[0], timeRange[1], ampRange[0], ampRange[1]])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')   
    
    plt.tight_layout()