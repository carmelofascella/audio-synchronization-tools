from utils import W, H, tot_pitch, Fs, Fs_frame, dur_plot, font, figsize_full_feat, figsize_symb, temporal_resolution
from utils import save_figure
from plot import plot_iirt, plot_chromagram, plot_measures_on_feature
from os.path import dirname, join as pjoin
from mir_algorithms import normalize_feature_sequence, compute_local_average, half_wave_rectification, scipy_peaks_algorithm, find_local_maxima, hps, peak_picking_midi, compute_cens
import librosa
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import LibFMP


def iirt(x, Fs, W=W, H=H, show_plot=True, load_matlab_file=False, filename="", save_fig=False):
    """Function that computes the time-frequency representation of the input audio signal using IIR filters 
    
    Args:
        x : input audio signal
        Fs : 'sampling frequency of the input'
        W: number of samples of the window used in the IIRT filter
        H: hop size
        show_plot: boolean flag. If True, the plot is shown
    
    Returns:
        X : iirt of the signal
    
    """
    dur = x.shape[0] / Fs
    
    if(load_matlab_file==False):
        iirt_audio = librosa.iirt(x, sr=Fs, win_length=W, hop_length=H, center=True, tuning=0.0)
        title = "IIRT  -  " + filename
        
    else:
        data_dir = pjoin('..', 'data','matlab')
        mat_fname = pjoin(data_dir, 'arabesque_cut_WAV_IIRT.mat')
        mat_contents = sio.loadmat(mat_fname)
        teststruct = mat_contents['f_pitch']
        iirt_audio_temp = teststruct

        iirt_audio = np.zeros((tot_pitch, iirt_audio_temp.shape[1]))
        
        iirt_audio[:, :] = iirt_audio_temp[24-1:108, :]
        title = "IIRT (from MATLAB)  -  " + filename
        
    
    if(show_plot==True):
        plot_iirt(iirt_audio, Fs, dur, title, H, save_fig=save_fig, save_name=filename)

    return iirt_audio


def chroma(iirt_audio, Fs, W=W, H=H, show_plot=True, load_matlab_file=False, filename="", ann=[], show_measures=False, transposition=0, save_fig=False):
    """Function that compute the Chromagram representation starting from the IIRT of a signal
    
    Args:
        x : iirt signal
        load_matlab_file. If true, it load the chroma computed by Matlab. Else, it computes it
    
    Returns:
        y : normalized chroma
    
    """
    Fs_frame = Fs / H
    dur_chroma = iirt_audio.shape[1] / Fs_frame

    
    if(load_matlab_file==False):
    
        #chroma_audio = librosa.feature.chroma_cqt(C=iirt_audio, bins_per_octave=12, n_octaves=7,
        #                                   fmin=librosa.midi_to_hz(24), norm=None)
        
        chroma_audio = np.zeros((12, iirt_audio.shape[1]))
        
        
        for i in range(iirt_audio.shape[0]):
            indChroma = np.mod(i+24,12)
            chroma_audio[indChroma, :] += iirt_audio[i, :]
        
        threshold = 0.001    ##from computCCell in matlab code
        
        chroma_audio_norm = normalize_feature_sequence(chroma_audio, norm=2, threshold=threshold)
        
        ################################################################
        ##FEATURE ENHANCEMENT STEP  (see librosa Enhanced chroma)!!!
        #chroma_filter = np.minimum(chroma_audio_norm,
        #                   librosa.decompose.nn_filter(chroma_audio_norm,
        #                                               aggregate=np.median,
        #                                               metric='cosine'))            #1)to remove noise
        #
        #chroma_audio_norm = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))     #2)to suppress local discontinuities
        ################################################################
        
        title = "Chromagram  -  " + filename

    else:
        data_dir = pjoin('..', 'data','matlab')
        mat_fname = pjoin(data_dir, 'arabesque_cut_WAV_chroma.mat')
        mat_contents = sio.loadmat(mat_fname)
        teststruct = mat_contents['f_chroma_norm']
        chroma_audio_norm = teststruct
        title = "Chromagram (from MATLAB)  -  " + filename
    
    
    if (transposition!=0):
        chroma_audio_norm = np.roll(chroma_audio_norm, transposition, axis=0)
    
    if(show_plot==True):
        plot_chromagram(chroma_audio_norm, Fs,dur_chroma, H, title=title, 
                        ann=ann, show_measures=show_measures, filename=filename, save_fig=save_fig)
        
    return chroma_audio_norm



def compute_spectral_flux(X, Fs=1, N=W, H=H, gamma=100, M=10, norm=1, show_plot=True, filename="", ann=[], show_measures=False, save_fig=False, symb_flag=False, audio_flag=True):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        X: feature (iirt for both audio and symbolic)
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    Y = np.log(1 + gamma * np.abs(X))
    
    
    #SUPERFLUX for Audio File
    #if(audio_flag==True):
    #    novelty_spectrum = librosa.onset.onset_strength(S=Y, sr=Fs, hop_length=H, lag = 2, max_size = 3)     

    
    ##########################################################################
    #####CLASSIC SPECTRAL FLUX for Symbolic file
    
    Y_diff = np.diff(Y)
    
    if(symb_flag==True):   ##first sample problem in symbolic file 
        for i in range(Y.shape[0]):
            Y_diff[i,0] = Y[i,0] - Y[i,1]
    
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    
    ##########################################################################
    ##DECAYING PART
    len_novelty_spectrum = novelty_spectrum.shape[0]       

    novelty_spectrum_decay = np.zeros(len_novelty_spectrum)       
    
    filtercoef = np.sqrt(1./np.arange(1,10+1))
    num_coef = len(filtercoef)    
        

    v_shift = novelty_spectrum
    v_help = np.zeros((num_coef, len_novelty_spectrum))

    
    for n in range(num_coef):
        v_help[n, :] = filtercoef[n] * v_shift
        v_shift = np.roll(v_shift, 1)
        v_shift[0] = 0  
        
    novelty_spectrum_decay = np.max(v_help, axis=0)
                
    ##########################################################################    
    #Plot

    if (show_plot==True):
        title = 'Spectral Flux  -  ' + filename
        fig, ax, line = LibFMP.B.plot_signal(novelty_spectrum_decay, Fs_frame, color='k', figsize=(10,6))
        dur_sf = len_novelty_spectrum / Fs_frame
        plt.xlim([0,dur_plot+0.01])
        plt.ylabel("Magnitude", fontsize=15, fontfamily=font)
        plt.xlabel("Time (seconds)", fontsize=15, fontfamily=font)
        #plt.title(title, fontsize=17.5, fontfamily=font)
        x_labels = np.arange(0,dur_plot+0.01,2)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)  
        y_labels = np.arange(0, 1+0.001, 0.2)
        plt.yticks(y_labels, fontsize=12.5, fontfamily=font)  
        plt.ylim([0,max(novelty_spectrum_decay[0:int(Fs_frame*dur_plot)])+0.01])  ##just to plot
        
        if(show_measures==True):
            plot_measures_on_feature(ann)
            
        if(save_fig==True):
            name = "sflux_"+filename
            save_figure(name=name)
        
        plt.show()
            
    ###ATTENTION! Just to test that the decaying version is better
    return novelty_spectrum_decay


def iirt_MIDI(midi_list, len_midi, Fs_midi, dur_midi, H=H, show_plot=True, matlab_algorithm=False, load_matlab_file=False,
             filename ="", save_fig=False):

    num_frames = int(len_midi / H)
    Fs_frame = Fs_midi / H
    end_midi = int(np.ceil(dur_midi*Fs_midi))

    if(load_matlab_file==False):
        X_ann, _ = list_to_pitch_activations(midi_list,num_frames, Fs_frame, dur_midi=dur_midi,
                                             matlab_algorithm=matlab_algorithm, offset=0)
        title = "IIRT  " + filename

    else:
        data_dir = pjoin('..', 'data','matlab')
        mat_fname = pjoin(data_dir, 'arabesque_cut_MIDI_IIRT.mat')
        mat_contents = sio.loadmat(mat_fname)
        teststruct = mat_contents['f_pitch']
        X_ann_temp = teststruct
        X_ann = np.zeros((tot_pitch, num_frames))
        X_ann[:, :] = X_ann_temp[24-1:108, :]
        title = "IIRT (From Matlab)  -  " + filename
    

    if(show_plot==True):
        
        
        F_coef_MIDI = np.arange(24,108+1,1)
        
        LibFMP.B.plot_matrix(X_ann, Fs=Fs_frame, F_coef=F_coef_MIDI, figsize=figsize_symb, aspect = 'auto')
        plt.xlim([0,dur_plot +0.01])
        x_labels = np.arange(0,dur_plot+0.01,5)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)
        
        y_labels=np.arange(24,108+1,3)
        plt.yticks(y_labels, fontsize=12.5, fontfamily=font)
        
        plt.ylabel("Pitch Number", fontsize=15, fontfamily=font)
        plt.xlabel("Time (second)", fontsize=15, fontfamily=font)
        #plt.title(title, fontsize=17.5, fontfamily=font)
        
        
        if(save_fig==True):
            name = "iirt_"+filename
            save_figure(name=name)
        
        
        plt.show()
    
        
    return X_ann


def peak_iirt_MIDI(midi_list, len_midi, Fs_midi, dur_midi, H=H, show_plot=True, iirt_midi=0, filename ="", save_fig=False):
    num_frames = int(len_midi / H)
    Fs_frame = Fs_midi / H
    end_midi = int(np.ceil(dur_midi*Fs_midi))
    
    X_ann, _ = list_to_pitch_activations_peaks(midi_list,num_frames, Fs_frame, iirt_midi )
    
    if(show_plot==True):
        
        F_coef_MIDI = np.arange(24,108+1,1)
        
        #title = 'Peak Picking  ' + filename 
        title=""
        LibFMP.B.plot_matrix(X_ann, Fs=Fs_frame, F_coef=F_coef_MIDI
                             ,title=title, figsize=figsize_symb, aspect = 'auto')
        
        plt.xlim([0,dur_plot+0.01])
        x_labels = np.arange(0,dur_plot+0.01,5)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)
        
        plt.ylabel("Pitch Number", fontsize=15, fontfamily=font)
        plt.xlabel("Time (seconds)", fontsize=15, fontfamily=font)
        
        y_labels=np.arange(24,108+1,3)
        plt.yticks(y_labels, fontsize=12.5, fontfamily=font)

        
        if(save_fig==True):
            name = "peak_iirt_"+filename
            save_figure(name=name)
        


        plt.show()
    
    return X_ann



def chroma_MIDI(iirt_midi, len_midi, Fs_midi, H=H, show_plot=True, load_matlab_file=False, filename="", ann=[], show_measures=False, save_fig=False):
    
    """
    Function that computes the chromagram from a IIRT representation of a MIDI file
    The chroma is than normalized in a proper way: 
    
    """
    
    
    if(load_matlab_file==False):
        chroma_midi = np.zeros((12, iirt_midi.shape[1]))
        
        for i in range(iirt_midi.shape[0]):
            indChroma = np.mod(i+24,12)
            chroma_midi[indChroma, :] += iirt_midi[i, :]
        
        
        threshold=0.001
        chroma_midi_norm = normalize_feature_sequence(chroma_midi, norm=2, threshold=threshold)
        
        title = "Chromagram  -  " + filename
   
    else:
        data_dir = pjoin('..', 'data','matlab')
        mat_fname = pjoin(data_dir, 'arabesque_cut_MIDI_chroma.mat')
        mat_contents = sio.loadmat(mat_fname)
        chroma_midi_norm = teststruct
        
        title = "Chromagram (from Matlab)  -  " + filename
    
    if(show_plot==True):

        dur_midi = len_midi / Fs_midi
        LibFMP.B.plot_chromagram(chroma_midi_norm, Fs=Fs_frame, title="", figsize=figsize_symb, extent=[0, dur_midi+0.05, 0,11],
                                aspect = 'auto', colorbar=True, colorbar_aspect=30.0)
    
        
        plt.xlim([0,dur_plot+0.01])
        
        chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
        #chroma_yticks=np.arange(0.5,12, step=0.915)
        chroma_yticks=np.arange(0.5,12, step=1.01)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal', fontsize=12.5, fontfamily=font)
        
        x_labels = np.arange(0,dur_plot+0.01,5)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)  
        
        plt.ylim([0,11])
        
        #cbar = plt.colorbar()
        #cbar.ax.tick_params(labelsize='large')
        
        plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
        plt.ylabel('Pitch Class', fontsize=15, fontfamily=font)
        #plt.title(title, fontsize=20, fontfamily=font)
        
        
        if(show_measures==True):
            plot_measures_on_feature(ann)
        
        if(save_fig==True):    
            name = "chroma_"+filename 
            save_figure(name=name)
        
        
        plt.show()
        
    return chroma_midi_norm


def peaks_picking_88pitch(D, dur, show_plot=True, prominence=0.005, offset=0.0199, size=150, distance=10, matlabAlgorithm=False,
                         filename="", load_matlab_file=False, binary_peaks=False, save_fig=False):
    

    
    """Function that compute the onset feature of the input iirt-filtered signal with 88 pitch bands
    Algorithm from Matlab SYNCBOXTOOL
    
    Args:
        D : iirt filtered signal
        load_matlab_file: Boolean value, if true load the file computed by matlab. Else, it computes the peak matrix here
        matlabAlgorithm: Boolean value, if true it computes the peaks using the maltab-like algorithm. 
                         Else, it computes it using the scipy function
    
    Returns:
        half_wave_88pitch : onset feature of the input signal, with 88 pitch band
    
    """
    
    if(load_matlab_file==False):
    
        half_wave_audio = half_wave_rectification(D, dur, show_plot=False, filename=filename, matlabAlgorithm=matlabAlgorithm)
        
        
        if(matlabAlgorithm==True):
            peaks_matrix = peaks(half_wave_audio)    ##ALGORITHM TAKEN FROM MATLAB (it doesn't work so well)
            title = "Peak Peaking (using Matlab-like algorithm) -  " + filename
        
        else:
            peaks_matrix = scipy_peaks_algorithm(half_wave_audio,  prominence=prominence, offset = offset, 
                                        distance=distance)
            title = "Peak Peaking (using scipy.signal algorithm) -  " + filename
        
        
            
    else:
        data_dir = pjoin('..', 'data','matlab')
        mat_fname = pjoin(data_dir, 'arabesque_cut_WAV_PEAKS.mat')
        mat_contents = sio.loadmat(mat_fname)
        teststruct = mat_contents['f_peaks_export']
        teststruct = teststruct.T
        print(teststruct.shape)
        peaks_matrix = np.zeros((tot_pitch, teststruct.shape[1]))
        peaks_matrix[:,:] = teststruct[24-1:108, :]
        title = "Peak Peaking (from Matlab) -  " + filename
        
        
    if (binary_peaks==True):
        peaks_matrix[peaks_matrix>0] = 1
    
        
    
    if(show_plot==True):
        figure = plt.figure(figsize=figsize_full_feat)
    
        plt.imshow(peaks_matrix,origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur, 24, 108])
        
        
        
        plt.xlim([0,dur_plot+0.01])
        
        chroma_names=np.arange(24,108+1,3)

        plt.yticks(chroma_names, fontsize=12.5, fontfamily=font)
        x_labels = np.arange(0,dur_plot+0.01,2)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)   
        
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        
        ##just for plot!
        a = peaks_matrix[:,0:int(dur_plot*Fs_frame)]
        max_plot = a.max()
        plt.clim([0,max_plot - (max_plot/1.2)])
        
        #plt.title(title)
        plt.xlabel("Time (seconds)", fontsize=15, fontfamily=font)
        plt.ylabel("Pitch number ", fontsize=15, fontfamily=font)

        
        if(save_fig==True):
            name = "peak_iirt_"+filename
            save_figure(name=name)
        
        plt.show()
        
        
    return peaks_matrix
    
    
def chroma_features(x, onset_label, dur, show_plot=True, matlabAlgorithm=True, filename="", save_fig=False):

    """
    Function that sum up the features belonging to pitches of the chroma of the input signal
    If onset-label = True, it computes the Chroma Onset Feature (CO)
    The Matlab algorithm is taken from "pitch_OnsetPeaks_toDNLCO". It's used to 'chromatize' the peak picking iirt.

    Args:
        x : input signal (half_wave rectified signal, or pick picking matrix of the input)
        onset: boolean value. If True, it computes the onset chroma feature. If false, it computes the chroma half-wave rectification 
    
    Returns:
        y : chroma features
    
    """
    y = np.zeros((12, x.shape[1]))
    

    gamma = 10
    scaleEnergyBeforeLog = 10000


    ##Matlab algorithm from "pitch_OnsetPeaks_toDNLCO" It's used to chromatize the peak picking 
    if(matlabAlgorithm==True):
        for i in range(x.shape[0]):
            
            val_peaks = np.log(x[i,:] * scaleEnergyBeforeLog+ 1)
            indChroma = np.mod(i+24,12)     ##i+24 because to map index[0,85] in pitch[24,108]
            y[indChroma, :] += val_peaks
    
    
    else:
                
        for i in range(x.shape[0]):
            indChroma = np.mod(i+24,12)
            y[indChroma, :] += x[i, :]

            
    if(show_plot==True):
    
        fig = plt.figure(figsize=figsize_full_feat)
        
        plt.imshow(y, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur, 0, 11])
        plt.xlim([0, dur_plot+0.01])
        
        chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
        #chroma_yticks=np.arange(0.5,12, step=0.915)
        chroma_yticks=np.arange(0.5,12, step=1.01)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal', fontsize=12.5, fontfamily=font)
        plt.ylim([0,11])
        x_labels = np.arange(0,dur_plot+0.01,2)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)
        
        plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
        plt.ylabel('Pitch Class ', fontsize=15, fontfamily=font)
        
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        
        ##just to plot
        #a = x[:,0:int(dur_plot*Fs_frame)]
        #max_plot = a.max()
        #plt.clim([0,max_plot])

        
        
        if(onset_label==False):
            title = "Half-wave rectification 12pitch  -  " + filename   
            
        else:
            title = "Chroma Onset Features (LNCO)  -  " + filename
        
        #plt.title(title, fontsize=17.5, fontfamily=font)
        
        if(save_fig==True):
            name = "chromaonsetfeature_"+filename
            save_figure(name=name)
        
        
        plt.show()
        
    
    return y


def compute_lnco(CO, dur,show_plot=True, filename="", save_fig=False):
    
    """Function that plot each feature vector, one for each pitch

    Args:
        CO : chroma onset features
    
    Returns:
        LNCO : locally adapteive normalized LNCO
    
    """
    
        
    norm_vector = np.linalg.norm(CO, ord = 2, axis=0)       ##calculate norm column-wise (axis = 0 means column-wise norm)
                                                            
    
    array_maxima = find_local_maxima(norm_vector, dur)      ##calculate maxima (algorithm from Matlab)
    
    
    lnco = np.zeros(CO.shape)
    
    lnco = np.divide(CO,array_maxima)                       ##calculate LNCO
    

    if(show_plot):
        
        num_samples = lnco.shape[1]
        fig = plt.figure(figsize=(14,8))
        array_x = np.arange(0,num_samples) * (dur/num_samples)
        plt.plot(array_x,norm_vector, label="Norm")
        plt.plot(array_x,array_maxima, color="r", label="Max")
        
        plt.xlim([0,dur_plot+0.01])
        #plt.xlim([0,dur+0.01])
        
        x_labels = np.arange(0,dur_plot+0.01,2)
        
        y_ticks=np.arange(0,max(array_maxima)+0.05 , step=1)
        plt.yticks(ticks = y_ticks, fontsize=12.5, fontfamily=font)
        #plt.ylim([0,max(array_maxima)+0.05])
        plt.ylim([0,max(array_maxima[0:int(Fs_frame*dur_plot)])+0.05])  ##just to plot
        
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)
        plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
        plt.ylabel('Magnitude', fontsize=15, fontfamily=font)
        title = "LN  -  "  + filename
        #plt.title(title, fontsize=17.5, fontfamily=font)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
        
        
        leg = plt.legend(loc='upper left', fontsize="xx-large")
        leg.get_frame().set_edgecolor('k')
        
        if(save_fig==True):
            name = "ln_"+filename
            save_figure(name=name)
        
        
        plt.show()
        
        
        fig = plt.figure(figsize=figsize_full_feat)
        plt.imshow(lnco, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur, 0, 11])
        plt.xlim([0,dur_plot+0.01])
        chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
        #chroma_yticks=np.arange(0.5,12, step=0.915)
        chroma_yticks=np.arange(0.5,12, step=1.01)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal')
        plt.ylim([0,11])
        x_labels = np.arange(0,dur_plot+0.01,2)
        plt.xticks(x_labels)        
        plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
        plt.ylabel('Pitch Class ', fontsize=15, fontfamily=font)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        title = "LNCO  -  " + filename
        #plt.title(title)
        
        if(save_fig==True):
            name = "lnco_"+filename
            save_figure(name=name)
        
        plt.show()
        
    return lnco


def compute_dnlco(lnco, dur, show_plot=True, load_matlab_file=False, audio_file=True, filename="", ann=[], show_measures=[], transposition=0, save_fig=False):
    
    """Function that compute the DNLCO

    Args:
        lnco
        
        load_matlab_file = if True, we load the DLNCO computed by matlab. Else,we compute it here.
        audio_file = If true, it load the matlab file of the audio DLNCO, else the midi one.
    
    Returns:
        decay_lnco: decay LNCO
    
    """
    
    decay_lnco = np.zeros(lnco.shape)
    
    if (load_matlab_file==False):
    
    
    
        DLNCO_filtercoef = np.sqrt(1./np.arange(1,10+1))
        num_coef = len(DLNCO_filtercoef)
        
        for i in range(lnco.shape[0]):
            v_shift = lnco[i, :]
            v_help = np.zeros((num_coef, lnco.shape[1]))
            
            for n in np.arange(num_coef):
                v_help[n, :] = DLNCO_filtercoef[n] * v_shift
                v_shift = np.roll(v_shift, 1)
                v_shift[0] = 0
            
            decay_lnco[i,:] = np.max(v_help, axis=0)
        
        title = "DLNCO  -  "  + filename

    else:
        
        data_dir = pjoin('..', 'data','matlab')
        
        if(audio_file==True):
            mat_fname = pjoin(data_dir, 'arabesque_cut_WAV_DNLCO.mat')
            mat_contents = sio.loadmat(mat_fname)
            teststruct = mat_contents['f_DLNCO1']

        else:
            mat_fname = pjoin(data_dir, 'arabesque_cut_MIDI_DNLCO.mat')
            mat_contents = sio.loadmat(mat_fname)
            teststruct = mat_contents['f_DLNCO2'] 

        decay_lnco = teststruct
        
        title = "DLNCO (from Matlab)  -  "  + filename
        
    
    if(transposition!=0):
        decay_lnco = np.roll(decay_lnco, transposition, axis=0)
    
    if(show_plot==True):
        fig = plt.figure(figsize=figsize_full_feat)
        plt.imshow(decay_lnco, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur+0.05, 0, 11])
        plt.xlim([0,dur_plot+0.01])
        chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
        chroma_yticks=np.arange(0.5,11, step=0.915)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal', fontsize=12.5, fontfamily=font)
        x_labels = np.arange(0,dur_plot+0.01,2)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)        
        plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
        plt.ylabel('Pitch Class', fontsize=15, fontfamily=font)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        #plt.title(title, fontsize=17.5, fontfamily=font)
        
        if(show_measures==True):
            plot_measures_on_feature(ann)
            
        if(save_fig==True):    
            name = "dlnco_"+filename
            save_figure(name=name)
        
        plt.show()
    
    
    return decay_lnco


def list_to_pitch_activations_peaks(note_list, num_frames, frame_rate, iirt_midi):
    
    'The parameter peakheight_scalefactor was taken from the function "midird4_to_pitchOnsetPeaks" '
    
    peakheight_scalefactor = 1e6
    
    P = np.zeros((tot_pitch, num_frames))
    F_coef_MIDI = np.arange(tot_pitch) + 1
    
    ##inizialization parameters for computing iirt and chroma

    for l in note_list:
        if( (l[1]<np.inf)):
            
            start_frame = max(0, int(l[0] * frame_rate))  
            
            end_frame = start_frame+1

            P[int(l[2]-24), start_frame:end_frame] = iirt_midi[int(l[2]-24), start_frame:end_frame] / peakheight_scalefactor
            
    return P, F_coef_MIDI
            


def list_to_pitch_activations_warping(note_list, num_frames, frame_rate):
    
    P = np.zeros((tot_pitch, num_frames))
    F_coef_MIDI = np.arange(tot_pitch) + 1
    
    ##inizialization parameters for computing iirt and chroma

    for l in note_list:
        if( (l[1]<np.inf)):
            
            start_frame = max(0, int(l[0] * frame_rate))  
            
            end_frame = min(num_frames, int(l[1] * frame_rate)+1)   ##Originally +1, I use +1 to listen when the next note starts

            P[int(l[2]-24), start_frame:end_frame] = 1
            
    return P, F_coef_MIDI


def list_to_pitch_activations(note_list, num_frames, frame_rate, dur_midi=0, matlab_algorithm=False, offset=0):
    """Create a pitch activation matrix from a list of note events
    
    USED FOR THE SONIFICATION OF THE STEREO FILE; AND FOR THE PEAK PICKING OF MIDI
    
    Remember that we consider pitches from 24 to 108. That's why -24 in the indices
    Note that for the velocity computation it was used the algorithm from matlab "midird4_to_pitchSTMP"
    
    Parameters
    ----------
    
    note_list : List
        A list of note events
        l[0] = start
        l[1] = end
        l[2] = pitch
        l[3] = velocity

    num_frames : int
        Desired number of frames for the matrix

    frame_rate : float
        Frame rate for P (in Hz)
        
    If matlab_algorithm==True, we compute the same matlab algorithm for the velocities, in each activation point
    Else, we put the original value of the velocities

    Returns
    -------
    P : NumPy Array
        Pitch activation matrix
        First axis: Indexed by [0:127], encoding MIDI pitches [1:128]
    F_coef_MIDI: MIDI pitch axis
    """
    P = np.zeros((tot_pitch, num_frames))
    F_coef_MIDI = np.arange(tot_pitch) + 1
    
    if (matlab_algorithm==True):

        ##inizialization parameters for computing iirt and chroma
        stepsize_ms = temporal_resolution * 1000
        num_pitch_features = np.ceil(dur_midi*1000 / stepsize_ms);
        
        for l in note_list:
            
            #if( (l[1]<np.inf)):
                
            start_frame = max(0, int(l[0] * frame_rate))  
            
            end_frame = min(num_frames, int(l[1] * frame_rate  - 0))   ##-2
            
            
            firstStepsizeInterval = np.floor(l[0]*1000 /stepsize_ms)
            lastStepsizeInterval = min( np.floor(l[1]*1000 / stepsize_ms),num_pitch_features)
            
            firstWindowInvolved = int(firstStepsizeInterval)
            lastWindowInvolved = int(min(lastStepsizeInterval,num_pitch_features))
            
            for currentWindow in range(firstWindowInvolved,lastWindowInvolved):
                rightBorderCurrentWindow_ms = currentWindow * stepsize_ms
                leftBorderCurrentWindow_ms = rightBorderCurrentWindow_ms - 2 * stepsize_ms
                
                lastWindowInvolved = min(lastStepsizeInterval,num_pitch_features)
                contribution = (min(l[1]*1000,rightBorderCurrentWindow_ms) - max(l[0]*1000, leftBorderCurrentWindow_ms)) / (2 * stepsize_ms)
                P[int(l[2]-24), start_frame:end_frame] += (l[3] * contribution)
                  
                
    else:
        
        for l in note_list:
            
            start_frame = max(0, int(l[0] * frame_rate))  
            end_frame = min(num_frames, int(l[1] * frame_rate) ) + offset  ##Original +1. Try +2 to align to the maltab one
            P[int(l[2]-24), start_frame:end_frame] = l[3]            

            
    return P, F_coef_MIDI


def get_features_audio(x,Fs, dur_audio, param_peaks =[0.005,0.0199,150,10], ann=[],
                           show_iirt=False, show_chroma=False, show_peaks=False, show_onset_chroma=False, 
                           show_LNCO=False, show_dLNCO=False, show_spectral_flux=False, show_hps=False, show_dlncosf=False,
                       filename="", transposition=0,
                      hps_flag=False, show_measures=False, weight_dlnco=0.5, weight_sf=0.5, save_fig=False):
    
    iirt_audio = iirt(x, Fs, show_plot=show_iirt, filename=filename)
    
    if(hps_flag==True):
        
        iirt_harm,iirt_perc=hps(iirt_audio, dur_audio, show_hps=show_hps)
        chroma_audio = chroma(iirt_harm, Fs, show_plot=show_chroma, filename=filename, 
                              ann=ann, show_measures=show_measures, transposition=transposition, save_fig=save_fig)
        #chroma_audio = compute_cens(iirt_harm,dur_audio,transposition=transposition, show_plot=show_chroma)
        
        
        peak_iirt_audio = peaks_picking_88pitch(iirt_audio, dur_audio, show_plot=show_peaks, 
                                                prominence=param_peaks[0], offset=param_peaks[1], 
                                                size=param_peaks[2], distance=param_peaks[3],
                                                matlabAlgorithm=False, filename=filename)
        co_audio = chroma_features(peak_iirt_audio, onset_label=True, dur=dur_audio, show_plot=show_onset_chroma, matlabAlgorithm=True, filename=filename)
        lnco_audio = compute_lnco(co_audio, dur=dur_audio, show_plot=show_LNCO, filename=filename)
        dlnco_audio = compute_dnlco(lnco_audio, dur=dur_audio, show_plot=show_dLNCO, filename=filename,
                                    ann=ann, show_measures=show_measures, transposition=transposition, save_fig=save_fig)  
        
        spectral_flux_audio = compute_spectral_flux(iirt_audio, show_plot=show_spectral_flux, filename=filename,
                                                   ann=ann, show_measures=show_measures, save_fig=save_fig, audio_flag=True)


    else:

        chroma_audio = chroma(iirt_audio, Fs, show_plot=show_chroma, filename=filename,
                              ann=ann, show_measures=show_measures, transposition=transposition, save_fig=save_fig)
        #chroma_audio = compute_cens(iirt_audio,dur_audio,transposition=transposition, show_plot=show_chroma)
        
        peak_iirt_audio = peaks_picking_88pitch(iirt_audio, dur_audio, show_plot=show_peaks, 
                                                prominence=param_peaks[0], offset=param_peaks[1], 
                                                size=param_peaks[2], distance=param_peaks[3],
                                                matlabAlgorithm=False, filename=filename)
        co_audio = chroma_features(peak_iirt_audio, onset_label=True, dur=dur_audio, show_plot=show_onset_chroma, matlabAlgorithm=True, filename=filename)
        lnco_audio = compute_lnco(co_audio, dur=dur_audio, show_plot=show_LNCO, filename=filename)
        dlnco_audio = compute_dnlco(lnco_audio, dur=dur_audio, show_plot=show_dLNCO, filename=filename,
                                   ann=ann, show_measures=show_measures, transposition=transposition, save_fig=save_fig)
        
        spectral_flux_audio = compute_spectral_flux(iirt_audio, show_plot=show_spectral_flux, filename=filename,
                                                   ann=ann, show_measures=show_measures, save_fig=save_fig, audio_flag=True)
    
    sum_dlnco_sf_audio = compute_sum_dlnco_sf(dlnco_audio,spectral_flux_audio, dur=dur_audio, weight_dlnco=weight_dlnco, 
                                              weight_sf=weight_sf, filename=filename, show_dlncosf=show_dlncosf, save_fig=save_fig)  
    
    return chroma_audio, dlnco_audio, spectral_flux_audio, sum_dlnco_sf_audio


def get_features_symbolic(symb_list, len_symb,Fs_symb, dur_symb, 
                          show_iirt=False, show_chroma=False, show_peaks=False, show_onset_chroma=False,
                          show_LNCO=False, show_dLNCO=False, show_spectral_flux=False, show_hps=False, show_dlncosf=False,
                          filename="", hps_flag=False,
                         ann=[], show_measures=False, weight_dlnco=0.5, weight_sf=0.5, save_fig=False):
    
    iirt_symb = iirt_MIDI(symb_list, len_symb, Fs_symb, dur_symb, matlab_algorithm=False, show_plot=show_iirt, filename=filename)
    
    if(hps_flag==True):
        iirt_harm,iirt_perc=hps(iirt_symb, dur_symb, show_hps=show_hps)    
        chroma_symb = chroma_MIDI(iirt_harm, len_symb, Fs_symb, show_plot=show_chroma, 
                                  filename=filename, ann=ann, show_measures=show_measures, save_fig=save_fig)
        peak_symb_list = peak_picking_midi(symb_list)
        peak_iirt_symb = peak_iirt_MIDI(peak_symb_list, len_symb, Fs_symb, dur_symb,iirt_midi=iirt_symb, show_plot = show_peaks, filename=filename)
        co_symb = chroma_features(peak_iirt_symb, dur=dur_symb, onset_label=True, show_plot=show_onset_chroma, matlabAlgorithm=True, filename=filename)
        lnco_symb = compute_lnco(co_symb, dur=dur_symb, show_plot=show_LNCO, filename=filename)
        dlnco_symb = compute_dnlco(lnco_symb, dur=dur_symb, show_plot=show_dLNCO, filename=filename,
                                   ann=ann, show_measures=show_measures, save_fig=save_fig)
        
        spectral_flux_symb = compute_spectral_flux(peak_iirt_symb, show_plot=show_spectral_flux, filename=filename,
                                                   ann=ann, show_measures=show_measures, save_fig=save_fig,
                                                  audio_flag=False)
        
    else:
        chroma_symb = chroma_MIDI(iirt_symb, len_symb, Fs_symb, show_plot=show_chroma,
                                  filename=filename, ann=ann, show_measures=show_measures, save_fig=save_fig)
        
        #chroma_symb = compute_cens(iirt_symb, dur_symb,transposition=transposition, show_plot=show_chroma)
        peak_symb_list = peak_picking_midi(symb_list)
        peak_iirt_symb = peak_iirt_MIDI(peak_symb_list, len_symb, Fs_symb, dur_symb,iirt_midi=iirt_symb, show_plot = show_peaks, filename=filename)
        co_symb = chroma_features(peak_iirt_symb, dur=dur_symb, onset_label=True, show_plot=show_onset_chroma, matlabAlgorithm=True, filename=filename)
        lnco_symb = compute_lnco(co_symb, dur=dur_symb, show_plot=show_LNCO, filename=filename)
        dlnco_symb = compute_dnlco(lnco_symb, dur=dur_symb, show_plot=show_dLNCO, filename=filename,
                                  ann=ann, show_measures=show_measures, save_fig=save_fig)
        
        spectral_flux_symb = compute_spectral_flux(peak_iirt_symb, show_plot=show_spectral_flux, filename=filename,
                                                   ann=ann, show_measures=show_measures, save_fig=save_fig,
                                                  audio_flag=False)
        
        
    sum_dlnco_sf_symb = compute_sum_dlnco_sf(dlnco_symb,spectral_flux_symb, dur=dur_symb, weight_dlnco=weight_dlnco, 
                                             weight_sf=weight_sf, filename = filename, show_dlncosf=show_dlncosf, save_fig=save_fig)    
    
    
    return chroma_symb, dlnco_symb, spectral_flux_symb, sum_dlnco_sf_symb


def cut_feature(feature, ann, start_index, end_index, cut_matrix=True):
    """
    feature = feature to be cut
    ann = entire annotation measure to be cut
    
    Convert from time in seconds to frame of the feature, and cut.
    
    cut_matrix is True if we are working on Chroma or DLNCO, False if we are working on Spectral Flux
    
    """

    
    #start_index = iteration*num_measures
    #end_index = (iteration*num_measures) + num_measures
    #end_index = min(end_index, tot_measures-1)  ##NB: the end of one cut corresponds to the start of the next one
    

    start_time = ann[start_index][0]
    end_time = ann[end_index][0]   
    
    new_dur = end_time - start_time

    start_time_frame = int(np.floor(start_time * Fs))
    end_time_frame = int(np.floor(end_time * Fs))
    
    start_time_frame_feature = int(np.floor(start_time * Fs / H))
    end_time_frame_feature = int(np.floor(end_time * Fs / H))

    
    if(cut_matrix==True):
        cut_feature = feature[:,start_time_frame_feature : end_time_frame_feature]
    else:
        cut_feature = feature[start_time_frame_feature : end_time_frame_feature]
        
    
    return cut_feature, new_dur, start_time_frame, end_time_frame, start_time, end_time, start_time_frame_feature, end_time_frame_feature


def compute_sum_dlnco_sf(dlnco, sf, dur, weight_dlnco=0.5, weight_sf=0.5, filename ="", show_dlncosf = False, save_fig=True):
    "Dlnco + Spectral Flux = We sum each row of the DLNCO with the spectral flux. That is a weighted sum"
    
    sum_dlnco_sf = np.zeros((12, sf.shape[0]))
    
    weight_dlnco = weight_dlnco
    weight_sf = weight_sf
    
    for i in range(sum_dlnco_sf.shape[0]):
        sum_dlnco_sf[i, :] = (dlnco[i,:] * weight_dlnco) + (sf[:] * weight_sf)
    
    if(show_dlncosf==True):
        fig = plt.figure(figsize=figsize_full_feat)
        plt.imshow(sum_dlnco_sf, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,dur+0.05, 0, 11])
        plt.xlim([0,dur_plot+0.01])
        chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
        chroma_yticks=np.arange(0.5,11, step=0.915)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal', fontsize=12.5, fontfamily=font)
        x_labels = np.arange(0,dur_plot+0.01,2)
        plt.xticks(x_labels, fontsize=12.5, fontfamily=font)        
        plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
        plt.ylabel('Pitch Class', fontsize=15, fontfamily=font)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        title = "DLNCO + Spectral Flux - " + filename
        #plt.title(title, fontsize=17.5, fontfamily=font)
        
        if(save_fig==True):
            name = "dlncosf_" + filename
            save_figure(name)
            
        plt.show()
        

    
    return sum_dlnco_sf