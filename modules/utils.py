import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append('./modules/')
import LibFMP.B
import LibFMP.C1
import LibFMP.C3
import pretty_midi
import pandas as pd
import math
import soundfile as sf
import IPython.display as ipd
import pickle
import os
import warnings
from yaml.loader import SafeLoader
import yaml

yaml_path = "global.yaml"
with open(yaml_path) as f: # load yaml
    global_var = yaml.load(f, Loader=SafeLoader)

##INITIALIZATION VALUES
Fs = Fs_midi = Fs_symb = global_var["Fs"]
feature_per_second = global_var["features_per_second"]
temporal_resolution = 1 / feature_per_second
H = int(temporal_resolution * Fs)
W = int(H*2)
Fs_frame=Fs/H
tot_pitch = global_var["tot_pitch"]
step_sizes_sigma = np.array([[1, 0], [0, 1], [1, 1]])
weights_mul = np.array([1.5, 1.5, 2])
figsize=(12.5,7.5)
figsize_full_feat = (16, 8)   ##(20,8) to see well here
figsize_symb = (12,6)
font="serif"
mpl.rc('font',family='serif')
dur_plot = global_var["dur_plot"]


warnings.filterwarnings("ignore")

def grayscale_cmap(cmap, scalar):
    """
    Function that returns a grayscale version of the given colormap

    Args:
        cmap (_type_): input colormap
        scalar (int): scalar used in the luminance calculation

    Returns:
        LinearSegmentedColormap: grayscale version of the input colormap
    """
    
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance ( http://alienryderflex.com/hsp.html )
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** scalar, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    colors[0, :] = 1

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    

def view_colormap(cmap,scalar):
    """
    Function that plots a colormap with its grayscale equivalent

    Args:
        cmap (_type_): input colormap
        scalar (int): scalar used in the luminance calculation
    """
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap, scalar)
    grayscale = cmap(np.arange(cmap.N))
    #grayscale[0:10,:] = 1
    
    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    
    
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def load_audio_burgmuller(filename, Fs=Fs, start=0, dur=1):
    fn_wav = os.path.join('..', 'data', 'C3', filename)
    #dur = 6.2
    audio,Fs = librosa.load(fn_wav, sr=Fs)
    
    start = start
    dur = dur
    start = int(np.floor(start*Fs))
    end = int(np.ceil(dur*Fs))
    audio= audio[start: end + start]
    len_audio = len(audio)
    print("AUDIO FILE: " + filename)
    ipd.display(ipd.Audio(audio,rate=Fs))
        
    return audio, dur, len_audio


def load_audio(filename, Fs=Fs, start=0, dur=1, load_full_audio=True, show_audio=True):
    
    filename_audio=filename+ ".wav"
    fn_wav = os.path.join('.', 'data', 'Schubert', 'audio_wav', filename_audio)
    audio,Fs = librosa.load(fn_wav, sr=Fs)
    dur_audio = librosa.get_duration(audio)
    len_audio = len(audio)


    if(load_full_audio==False):
        start = start
        dur = dur
        start = int(np.floor(start*Fs))
        end = int(np.ceil(dur*Fs))
        audio= audio[start: end + start]
    
    if(show_audio==True):
        print("AUDIO FILE: " + filename_audio)
        ipd.display(ipd.Audio(audio,rate=Fs))
        
    return audio, dur_audio, len_audio


def load_symbolic(filename, len_audio=0, pitch_transpose=0,  load_file=False, show_audio=True):
    
    filename_xmltocsv = filename +  ".csv"
    fn_xmltocsv = os.path.join('.', 'data', 'Schubert', 'score_xmltocsv_time', filename_xmltocsv)  
    
    symb_list = csv_to_list(fn_xmltocsv)
    
    symb_list = transposition(symb_list,pitch_transpose)
    
    dur_symb = symb_list[-1][1]    ##release of the last note = dur midi file
    len_symb = int(np.ceil(dur_symb * Fs))
    
    
    filename_symbolic = filename + ".wav"
    fn_symbolic = os.path.join('.', 'data', 'Schubert', 'symbolic_file', filename_symbolic)
    
    if(load_file==False):
        x_symbolic = sonify_note_list(symb_list, len_audio)
        sf.write(fn_symbolic, x_symbolic, Fs_symb)
        
        
    else:
        x_symbolic,_ = librosa.load(fn_symbolic, sr=Fs_symb)
    
    if(show_audio==True):
        print("SYMBOLIC FILE: " + filename)
        ipd.display(ipd.Audio(x_symbolic,rate=Fs_symb))
                
    return x_symbolic, symb_list, dur_symb, len_symb



def transposition(score, offset):
    for i in range(len(score)):
        score[i][2] += offset
        
    return score

##DEPRECATED, BUT HANDLE IT FOR THE BURGMULLERRRR

def midi_to_list(fn_midi, dur_midi, Fs, path_output="", show_plot=False):
    """
    This function load a midiFile and transforms it in a list. 
    After that, it writes a cut version of the original midi file in "path_output", according to the "dur_midi" passed as input
    
    """
    
    midi_data = pretty_midi.PrettyMIDI(fn_midi)
    midi_list= []
    print (midi_data.key_signature_changes)
    cut_midi_data = pretty_midi.PrettyMIDI()
    cut_midi_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=cut_midi_program)
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start 
            if(start<dur_midi):
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                
                midi_list.append([start, end, pitch, velocity, instrument.name])

                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
                piano.notes.append(note)
                
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))
    
    
    cut_midi_data.instruments.append(piano)
    if(path_output!= ""):
        cut_midi_data.write(path_output)
    
    x_midi = cut_midi_data.synthesize(fs=Fs_midi)   #computed just to obtain the duration of the midi
    end_midi = int(np.ceil(dur_midi*Fs_midi))
    x_midi = x_midi[0:end_midi]
    len_midi = len(x_midi)

    
    print("MIDI Synthesized")
    ipd.display(ipd.Audio(x_midi, rate=Fs_midi) )
    if (show_plot==True):
        LibFMP.C1.visualize_piano_roll(midi_list, figsize=(12, 4), velocity_alpha=True);
    
    return midi_list, cut_midi_data, len_midi


def cut_midi(midi_data, dur_midi):

    cut_midi_data = pretty_midi.PrettyMIDI()

    cut_midi_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=cut_midi_program)

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start 
            if(start<dur_midi):
                end = note.end
                pitch = note.pitch
                velocity = note.velocity

                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
                piano.notes.append(note)
            
    cut_midi_data.instruments.append(piano)
    cut_midi_data.write('../data/C3/Cut files/arabesque_DIO_midi.midi')
    
    return cut_midi_data
        


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def normalization_frames_to_samples(vector):
    y = int(Fs/Fs_frame) * vector
    return y


def normalization_frames_to_second(vector):
    
    """
    Function used to translate the alignment in terms of features in an alignment in terms of seconds.
    
    vector: warping path to be aligned
    temporal_resolution = 1/features_per_second. It's a global variable
    
    """
    
    
    y = temporal_resolution * vector
    
    return y

def replace_nan_values(C_chroma):

    for i in range(C_chroma.shape[0]):
        for j in range(C_chroma.shape[1]):
            if (np.isnan(C_chroma[i][j])):
                C_chroma[i][j] = 1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def csv_to_annotation_list(csv):
    """Convert a csv score file to a list of note events

    Notebook: C1/C1S2_CSV.ipynb

    Args:
        csv: Either a path to a csv file or a data frame

    Returns:
        score: A list of note events where each note is specified as 
        [start, duration, pitch, velocity, label]
    """

    if isinstance(csv, str):
        df = pd.read_csv(csv, sep=";")
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    score = []
    for i, (start, measure) in df.iterrows():
        score.append([start, measure])
    return score


def transpose_midi(midi_data, transposition, start, dur, Fs_midi=22050, cut_midi=False):
    
    #audio_data = midi_data.synthesize(fs=22050)
    
    for instrument in midi_data.instruments:
    # Skip drum instruments - their notes aren't pitched!
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            note.pitch += transposition
            
    return midi_data


def shift_annotation(audio_ann, midi_ann, start_audio, end_audio, start_midi, end_midi, start_index, end_index):
    '''

    '''
    midi_ann_act = []
    audio_ann_act = []


    j = start_index

    while(j<=end_index):    
        midi_ann_act.append([midi_ann[j][0] - start_midi, j])
        audio_ann_act.append([audio_ann[j][0] - start_audio, j])
        j = j+1
    
    return audio_ann_act, midi_ann_act
    

def sonify_measures(audio, midi_warped, audio_ann, midi_ann_meas, dur_midi, title="", show_annotation=False, listen_mono=False):
    
    click_audio_mes = np.array(audio_ann)[:,0]         ##only the start time, not the number of the measure
    click_midi_mes = np.array(midi_ann_meas)[:,0]

    x_out = np.zeros((2,audio.shape[0] ))

    
    click_audio_meas_track = librosa.clicks(times=click_audio_mes, frames=None, sr=Fs, hop_length=None, click_freq=2648, 
                         click_duration=0.1, click=None, length=audio.shape[0])            
    

    click_midi_meas_track = librosa.clicks(times=click_midi_mes, frames=None, sr=Fs, hop_length=None, click_freq=1023, 
                         click_duration=0.1, click=None, length=audio.shape[0])  
    
    x_out[0,:] = 2*audio[:] + 0.25*click_audio_meas_track
    x_out[1,:] = 0.3*midi_warped[:] + 0.25*click_midi_meas_track
    
    
    if(listen_mono==True):
        x_mono = x_out.sum(axis=0) / 2
        print("Sonification of the annotation measures\n HIGHER Click= Hand-made Measure Annotation - LOWER Click: Warped Measured Annotation using " + title)
        ipd.display(ipd.Audio(x_mono, rate=Fs, normalize=True) )
    
    else:
        print("Sonification of the annotation measures\n LEFT= Hand-made Measure Annotation - RIGHT: Warped Measured Annotation using " + title)
        ipd.display(ipd.Audio(x_out, rate=Fs, normalize=True) )
    

def csv_to_list(filename):
    
    if isinstance(filename, str):
        df = pd.read_csv(filename, sep=";")
    elif isinstance(filename, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    score = []
    for i, (start, end, pitch, velocity, instrument) in df.iterrows():
        score.append([start, end, pitch, velocity, instrument])
    return score    

def get_midi_file(filename, transposition_flag=False, transposition=-2):
    
    midi_data = pretty_midi.PrettyMIDI(filename)
    dur_midi = midi_data.get_end_time()
    
    if(transposition_flag==True):
        new_midi_data = pretty_midi.PrettyMIDI()
        new_midi_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=new_midi_program)    
              
    
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                velocity = note.velocity
                pitch = note.pitch + transposition   
                
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
                piano.notes.append(note)            
        
        new_midi_data.instruments.append(piano)  
        
        x_midi = new_midi_data.synthesize(fs=22050)
    
    else:
        x_midi = midi_data.synthesize(fs=22050)   #computed just to obtain the duration of the midi
        
    len_midi = len(x_midi)
    
    ##HANDLE!! Create a new midi file already transposed in a folder!
    
    return new_midi_data, dur_midi, len_midi, x_midi


def get_annotation_measures(filename, x, x_symbolic, sonify=True):
    
    """
    Function which loads the annotation measures from a csv files into a list, for both files.
    """
    
    filename_csv = filename + '.csv'   
    fn_ann_audio_measure = os.path.join('.', 'data', 'Schubert', 'ann_audio_measure', filename_csv)  
    

    fn_ann_symbolic = os.path.join('.', 'data', 'Schubert', 'ann_symbolic_measure', filename_csv)
    
    ##COMPUTE ANNOTATIONS
    audio_ann = csv_to_annotation_list(fn_ann_audio_measure)
    symbolic_ann = csv_to_annotation_list(fn_ann_symbolic)
    
    if(sonify==True):
        click_audio_mes = np.array(audio_ann)[:,0]
        click_audio_meas_track = librosa.clicks(times=click_audio_mes, frames=None, sr=Fs, hop_length=None, click_freq=2000, 
                             click_duration=0.1, click=None, length=x.shape[0])  
        
        x_audio_out = np.zeros((2,x.shape[0]))
        x_audio_out = 0.5*x + 0.25*click_audio_meas_track
        print("AUDIO FILE with sonified measures: " + filename + ".wav")
        ipd.display(ipd.Audio(x_audio_out, rate=Fs) )
        
        
        click_symb_mes = np.array(symbolic_ann)[:,0]
        click_symb_meas_track = librosa.clicks(times=click_symb_mes, frames=None, sr=Fs_symb, hop_length=None, click_freq=2000, 
                             click_duration=0.1, click=None, length=x_symbolic.shape[0])  
        
        x_symb_out = np.zeros((2,x_symbolic.shape[0]))
        x_symb_out = 0.5*x_symbolic + 0.25*click_symb_meas_track
        print("SYMBOLIC FILE with sonified measures: " + filename)
        ipd.display(ipd.Audio(x_symb_out, rate=Fs_symb) )
        
    return audio_ann, symbolic_ann

def get_annotation_measures_audio(filename1, filename2, x1, x2, sonify=True):
    
    """
    Function which loads the annotation measures from a csv files into a list, for both files.
    """
    
    
    filename_csv1 = filename1 + '.csv'   
    fn_ann_audio_measure1 = os.path.join('.', 'data', 'Schubert', 'ann_audio_measure', filename_csv1)  
    

    filename_csv2 = filename2 + '.csv'   
    fn_ann_audio_measure2 = os.path.join('.', 'data', 'Schubert', 'ann_audio_measure', filename_csv2)  
    
    ##COMPUTE ANNOTATIONS
    audio_ann1 = csv_to_annotation_list(fn_ann_audio_measure1)
    audio_ann2 = csv_to_annotation_list(fn_ann_audio_measure2)
    

    
    if(sonify==True):
        click_audio_mes1 = np.array(audio_ann1)[:,0]
        click_audio_meas_track1 = librosa.clicks(times=click_audio_mes1, frames=None, 
                            sr=Fs, hop_length=None, click_freq=2000, 
                             click_duration=0.1, click=None, length=x1.shape[0])  
        
        x1_out = np.zeros((2,x1.shape[0]))
        x1_out = 0.5*x1 + 0.25*click_audio_meas_track1
        print("AUDIO FILE 1 with sonified measures: " + filename1)
        ipd.display(ipd.Audio(x1_out, rate=Fs) )
        
        
        click_audio_mes2 = np.array(audio_ann2)[:,0]
        click_audio_meas_track2 = librosa.clicks(times=click_audio_mes2, frames=None, sr=Fs, 
                            hop_length=None, click_freq=2000, 
                             click_duration=0.1, click=None, length=x2.shape[0])  
        
        x2_out = np.zeros((2,x2.shape[0]))
        x2_out = 0.5*x2 + 0.25*click_audio_meas_track2
        print("AUDIO FILE 2 with sonified measures: " + filename2)
        ipd.display(ipd.Audio(x2_out, rate=Fs) )
        
    
    return audio_ann1, audio_ann2


def sonify_measures_audio(sonification, audio1_ann_act, audio2_ann_act_warped, title, show_annotation, listen_mono=False):
    
    len_sonification = sonification[0].shape[0] 
    
    click_audio1_mes = np.array(audio1_ann_act)[:,0]         ##only the start time, not the number of the measure
    click_audio2_mes = np.array(audio2_ann_act_warped)[:,0]

    
    click_audio1_meas_track = librosa.clicks(times=click_audio1_mes, frames=None, sr=Fs, hop_length=None, click_freq=2648, 
                         click_duration=0.1, click=None, length=len_sonification)            
    

    click_audio2_meas_track = librosa.clicks(times=click_audio2_mes, frames=None, sr=Fs, hop_length=None, click_freq=1023, 
                         click_duration=0.1, click=None, length=len_sonification)    
    

    sonification[0] += 0.25*click_audio1_meas_track
    sonification[1] += 0.25*click_audio2_meas_track
    
    if(listen_mono==True):
        x_mono = sonification.sum(axis=0) / 2
        print("Sonification of the annotation measures\n HIGHER Click= Hand-made Measure Annotation - LOWER Click: Warped Measured Annotation using " + title)
        ipd.display(ipd.Audio(x_mono, rate=Fs, normalize=True) )
    
    else:
        print("Sonification of the annotation measures\n LEFT= Hand-made Measure Annotation - RIGHT: Warped Measured Annotation using " + title)
        ipd.display(ipd.Audio(sonification , rate=Fs, normalize=True) )


def save_figure(name):
    fn_path_figure = os.path.join('.', 'data', 'Schubert','figures', name)
    plt.savefig(fn_path_figure, bbox_inches='tight', quality=100, optimize =True, dpi=300)    

def save_audio(x, name):
    
    fn_out = os.path.join('.', 'data', 'Schubert','audio_output', name)
    sf.write(fn_out, x, Fs) 


def sonify_note_list(note_list, len_list):
    #dur_list = note_list[-1][1]
    #print(dur_list)
    #len_list = int(np.ceil(dur_list * Fs))
    num_frames = int(np.ceil(len_list / H))
    Fs_frame = Fs / H
    
    ##HANDLE!!!!
    X_ann, F_coef = list_to_pitch_activations(note_list,num_frames, Fs_frame, offset=-1)   


    harmonics = [1, 1/2, 1/3, 1/4, 1/5]
    fading_msec = 0.5

    x_pitch_ann = LibFMP.B.sonify_pitch_activations(X_ann, len_list, Fs_frame, Fs, min_pitch=24, Fc=440, harmonics_weights=harmonics, fading_msec=fading_msec)
    
    return x_pitch_ann


def color_argument_to_dict(colors, labels_set, default='gray'):
    """Creates a color dictionary

    Args:
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap, 3. list or np.ndarray of
            matplotlib color specifications, 4. dict that assigns labels  to colors
        labels_set: List of all labels
        default: Default color, used for labels that are in labels_set, but not in colors

    Returns:
        color_dict: Dictionary that maps labels to colors
    """

    if isinstance(colors, str):
        # FMP colormap
        if colors in FMP_COLORMAPS:
            color_dict = {l: c for l, c in zip(labels_set, FMP_COLORMAPS[colors])}
        # matplotlib colormap
        else:
            cm = plt.get_cmap(colors)
            num_labels = len(labels_set)
            colors = [cm(i / (num_labels + 1)) for i in range(num_labels)]
            color_dict = {l: c for l, c in zip(labels_set, colors)}

    # list/np.ndarray of colors
    elif isinstance(colors, (list, np.ndarray, tuple)):
        color_dict = {l: c for l, c in zip(labels_set, colors)}

    # is already a dict, nothing to do
    elif isinstance(colors, dict):
        color_dict = colors

    else:
        raise ValueError('`colors` must be str, list, np.ndarray, or dict')

    for key in labels_set:
        if key not in color_dict:
            color_dict[key] = default

    return color_dict


def check_line_annotations(annot, default_label=''):
    """Checks line annotation. If label is missing, adds an default label.

    Args:
        annot: A List of the form of [(time_position, label), ...], or [(time_position, ), ...], or [time_position, ...]
        default_label: The default label used if label is missing

    Returns:
        annot: A List of tuples in the form of [(time_position, label), ...]
    """
    if isinstance(annot[0], (list, np.ndarray, tuple)):
        len_annot = len(annot[0])
        assert all(len(a) == len_annot for a in annot)
        if len_annot == 1:
            annot = [(a[0], default_label) for a in annot]

    else:
        assert isinstance(annot[0], (int, float, complex)) or np.isscalar(annot[0])
        annot = [(a, default_label) for a in annot]

    return annot


def load_param_peaks(num_song, csv_audio_version1, csv_audio_version2, hps_flag):
    filename_pickle = num_song + "_" + csv_audio_version1 + "_" + csv_audio_version2
    
    
    if(csv_audio_version2=="symb"):   ##convention 
        scen_folder = "audio_symbolic"
    else:
        scen_folder = "audio_audio"
    
    
    if(hps_flag==False):
        type_folder = "standard"
    else:
        type_folder ="hps"
    
    fn_pickle = os.path.join('.', 'data', 'Schubert', 'peak_best_parameters',scen_folder, type_folder, filename_pickle)
    with open(fn_pickle, 'rb') as f:
        loaded_obj = pickle.load(f)

    return loaded_obj


class MyFraction:
    def __init__(self, numerator=1, denominator=1):
        self.numerator = numerator
        self.denominator = denominator

    def get_fraction(self):
        from fractions import Fraction
        return Fraction(numerator=self.numerator, denominator=self.denominator)

    def __repr__(self):
        return '{}/{}'.format(self.numerator, self.denominator)