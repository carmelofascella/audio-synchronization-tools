from utils import Fs, H, step_sizes_sigma, weights_mul, Fs_frame, font
from utils import truncate, normalization_frames_to_second, shift_annotation, sonify_measures, sonify_measures_audio, save_figure, save_audio, normalization_frames_to_samples, MyFraction, color_argument_to_dict
from feature_extraction import list_to_pitch_activations_warping, cut_feature
from cost_matrix import compute_cost_matrix_cosine_distance
from plot import plot_annotation_measures_fullfile, plot_cut_feature, plot_measures_on_costmatrix, plot_annotation_measures
from scipy.interpolate import interp1d
import numpy as np
import time
import copy
import LibFMP
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import libTSM as TSM



def warping_annotation(ann1, ann2, warping_path):
    
    """
    Function used to align the midi file to the audio file, using the warping_path    
    warping_path[:,0] = wp coordinates of the audio
    warping_path[:,1] = wp coordinates of the midi
    
    """
    
    ann2_warped = []
    
    
    interp = interp1d(warping_path[:, 1], warping_path[:, 0], fill_value='extrapolate')

    for i in range(len(ann2)):
            
        

        new_start_measure = float(max(0, interp(ann2[i][0])))
        
        if(i<len(ann2)-1):
            ann2_warped.append([new_start_measure, i])   ##HANDLE i must follow the offset of the measures!!
        
        if(i==len(ann2)-1):
            ann2_warped.append([ann1[i][0], i])     ##CHEAT. Anchor measure
           

    
    return ann2_warped


def compute_mean_abs_error(ann1, ann2, mae_list, treshold=0, num_mismatch=0, start_meas=0, title="", show_mae=True):

    act_mae = 0
    tot_dif = 0
    
    M = len(ann2) - 1      ##we don't consider the last one, because it's the following anchor point
    
    if(show_mae):
        print("\n__________________________________________________________________\n\n" + title + " MAE\n")
    
    for i in range(M):
        dif = np.abs(ann1[i][0]-ann2[i][0])    ##difference is in seconds
        
        if((dif <= treshold) & (i!=0)):    ##If the dif is under the treshol and if it's not the anchor point 
            num_mismatch += 1   
            
            
        if(show_mae):
            if(i==0):
                print("Time difference of measures " + str(start_meas + i) +" UNDER the treshold ->  " + str(truncate(dif*1000,4)) + "ms  <=  " + str(treshold*1000)+"ms" + "    ---> ANCHOR POINT")
            else:
                
                if((dif <= treshold)):
                    print("Time difference of measures " + str(start_meas + i) +" UNDER the treshold ->  " + str(truncate(dif*1000,4)) + "ms  <=  " + str(treshold*1000)+"ms")
                    
                else:
                    print("Time difference of measures " + str(start_meas + i) +" OVER  the treshold ->  " + str(truncate(dif*1000,4)) + "ms  >  " + str(treshold*1000)+"ms")
                

        mae_list.append(dif)
        tot_dif = tot_dif + dif


    if(M>1):
        act_mae = tot_dif / (M-1)       # -1 because we don't count the anchor point
        act_mae_ms = act_mae*1000 
    else:                               ##case where last iteration has only an anchor point to evaluate
        act_mae = 0
        act_mae_ms = 0
    
    if(show_mae):
        print("\nMean Absolute Error using " + title + " = %f seconds , %f milliseconds\n" %(act_mae,act_mae_ms))
    
    return act_mae, mae_list, num_mismatch


def calculate_offset_wp(wp,start_audio, start_midi):
    'Function used to translate the alignment path respect to the start of the actual frame'
    
    wp_offset = wp.copy()
    
    for i in range(len(wp)):
        wp_offset[i][0] += start_audio
        wp_offset[i][1] += start_midi
        
    return wp_offset

def warping(audio, midi_list, warping_path, end_midi, ind_note, start_midi=0, 
                  click_track=False, show_sonification=False, title="", elapsed_time=0, listen_mono=False):
    
    """
    Function used to align the midi file to the audio file, using the warping_path    
    warping_path[:,0] = wp coordinates of the audio
    warping_path[:,1] = wp coordinates of the symbolic
    
    """
    

    start = time.time()
    
    midi_list_warped = []
    click_times = []
    
    f = interp1d(warping_path[:, 1], warping_path[:, 0], fill_value='extrapolate')
    

    old_start = midi_list[ind_note][0]

    while((old_start< end_midi) & (ind_note<len(midi_list)) ):    ##end note
        old_start = midi_list[ind_note][0]
        old_end = midi_list[ind_note][1] 


        new_start = float(max(0, f(old_start - start_midi)))     ##-start_midi to let this segment start frmo time 0    
        new_end = float(max(0, f(old_end - start_midi )))        
        

        ##TAKE CARE of this. Inf problem
        if(np.isinf(new_end) ):
            new_end = new_start + 0.2
        
        ##HANDLE this problem (chroma burgmuller)
        if(new_start==new_end): 
            new_end+=0.1    ##0.1
        
        pitch = midi_list[ind_note][2]
        velocity = midi_list[ind_note][3]
        instrument = midi_list[ind_note][4]
        midi_list_warped.append([new_start, new_end, pitch, velocity, instrument])
        
        click_times.append(new_start)
        
        ind_note+=1
   
    
    midi_list_warped= sorted(midi_list_warped, key=lambda x: (x[0], x[2]))
    
    
    end = time.time()
    elapsed_time = elapsed_time + (end-start)
    

    num_frames = int(len(audio) / H)
    Fs_frame = Fs / H
    X_ann, F_coef_MIDI = list_to_pitch_activations_warping(midi_list_warped,num_frames, Fs_frame)

    
    harmonics = [1, 1/2, 1/3, 1/4, 1/5]
    fading_msec = 0.5
    x_pitch_ann, x_pitch_ann_stereo = LibFMP.B.sonify_pitch_activations_with_signal(X_ann, audio, Fs_frame, Fs, 
                                                                                  fading_msec=fading_msec, harmonics_weights=harmonics,
                                                                                   min_pitch=24)
    
    
    if(click_track==True):
        click_track = librosa.clicks(times=np.array(click_times), frames=None, sr=Fs, hop_length=None, click_freq=2000, 
                                 click_duration=0.1, click=None, length=x_pitch_ann_stereo[1,:].shape[0])
    
        x_pitch_ann_stereo[1,:] += 0.25*click_track[:]
        
    
    
    print("\nSynchronization between Audio File and Symbolic File using the " + title + " warping path")
    if(listen_mono==True):
        x_pitch_ann_stereo[0,:] = x_pitch_ann_stereo[0,:]*2
        x_mono = x_pitch_ann_stereo.sum(axis=0) / 2
        ipd.display(ipd.Audio(x_mono, rate=Fs, normalize=True) )
        
    else:
        ipd.display(ipd.Audio(x_pitch_ann_stereo, rate=Fs, normalize=True) )
    
    return midi_list_warped, x_pitch_ann, ind_note, elapsed_time


def get_alignment_path(x, x_symbolic, symb_list, audio_ann, symbolic_ann, anchor_points_distance, chroma_audio, 
                       chroma_symb, dlnco_audio, dlnco_symb, spectral_flux_audio, spectral_flux_symb,
                       sum_dlnco_sf_audio, sum_dlnco_sf_symb,
                       dur_audio, dur_symb, treshold=0, use_dlnco=True, sum_perc_features=False, show_wp=False,
                      filename = "",
                       show_cut_files=False, show_annotation=False, show_sonification=False, 
                       show_sonification_measures =False,
                      show_cut_features=False, show_mae=True, show_results=True, listen_mono=False, save_fig=False):
    
    c1="limegreen"
    c2="darkturquoise"
    
    
    #name_chroma1_cut = "chroma1_cut"
    #name_chroma2_cut = "chroma_symb_cut"
    name_chroma_cost = "chroma_symb_cost"
    #name_chroma_meas = "chroma_meas"
    if(sum_perc_features==False):
        if(use_dlnco==True):
            #percFeat_audio1 = copy.deepcopy(dlnco_audio1)
            #percFeat_audio2 = copy.deepcopy(dlnco_audio2)
            #percFeatTitle = "DLNCO"
            #sumFeatTitle = "Chroma + DLNCO"
            #name_perc1_cut = "dlnco1_cut"
            #name_perc2_cut = "dlnco2_cut"
            name_perc_cost = "dlnco_symb_cost"
            name_sum_cost = "chromadlnco_symb_cost"
            #name_perc_meas = "dlnco_meas"
            #name_sum_meas = "chromadlnco_meas"
            
        else:
            #percFeat_audio1 = copy.deepcopy(spectral_flux_audio1)
            #percFeat_audio2 = copy.deepcopy(spectral_flux_audio2)
            #percFeatTitle = "SPECTRAL FLUX"
            #sumFeatTitle = "Chroma + Spectral Flux"
            #name_perc1_cut ="sflux1_cut"
            #name_perc2_cut ="sflux2_cut"
            name_perc_cost = "sflux_symb_cost"
            name_sum_cost = "chromasflux_symb_cost"
            #name_perc_meas = "sflux_meas"
            #name_sum_meas = "chromasflux_meas"
        
    else:

        #percFeat_audio1 = copy.deepcopy(sum_dlnco_sf_audio1)
        #percFeat_audio2 =  copy.deepcopy(sum_dlnco_sf_audio2)
        #percFeatTitle = "DLNCO + SPECTRAL FLUX"
        #sumFeatTitle = "Chroma + DLNCO + Spectral Flux"
        #use_dlnco=True ## HANDLE THIS! It`s a problem for the cut of the features!!!!
        #name_perc1_cut ="dlncosflux1_cut"
        #name_perc2_cut ="dlncosflux2_cut"
        name_perc_cost = "dlncosflux_symb_cost"
        name_sum_cost = "chromadlncosflux_symb_cost"
        #name_perc_meas = "dlncosflux_meas"
        #name_sum_meas = "chromadlncosflux_meas"   
    
    
    if(show_results==True):
        print("\n\n\n############################################################################################\n")
        print('\033[1m' + "AUDIO  vs  SYMBOLIC.\n" +'\033[0m')
        print("File analyzed: " + '\033[1m'  + filename +'\033[0m')
        print("\nMeasures analyzed per each iteration: " + str(anchor_points_distance))
        print("Treshold of Accuray: " + str(treshold*1000) + " ms")
        print("\n_______________________________________________________________________________________________\n")
    
    filename_xml = filename + '.xml'
    filename_audio = filename + '.wav'
    
    ##CHOISE OF THE PERCUSSIVE FEATURE
    if(sum_perc_features==False):
        if(use_dlnco==True):
            percFeat_audio = copy.deepcopy(dlnco_audio)
            percFeat_symb = copy.deepcopy(dlnco_symb)
            percFeatTitle = "DLNCO"
            
        else:
            percFeat_audio = copy.deepcopy(spectral_flux_audio)
            percFeat_symb = copy.deepcopy(spectral_flux_symb)
            percFeatTitle = "SPECTRAL FLUX"
            
    else:

        percFeat_audio = copy.deepcopy(sum_dlnco_sf_audio)
        percFeat_symb =  copy.deepcopy(sum_dlnco_sf_symb)
        percFeatTitle = "DLNCO + SPECTRAL FLUX"
        use_dlnco=True ## HANDLE THIS! It`s a problem for the cut of the features!!!!
        
    ##PLOT ANNOTATION MEASURES
    if(show_annotation==True):
        plot_annotation_measures_fullfile(audio_ann, symbolic_ann, x, x_symbolic, 
                                          anchor_points_distance, filename_audio, filename_xml, save_fig=False)
    
    
    num_audio_measures = len(audio_ann) -1
    num_symb_measures = len(symbolic_ann) -1

    
    ##CONTROL the number of the measures
    if(num_audio_measures!=num_symb_measures):
        raise RuntimeError('The number of measures of the two files are different!')
    else:
        tot_measures = num_audio_measures

    if(show_mae==True):   
        print("__________________________________________________________________________________________________________")
    
    elapsed_time = 0
    
    tot_analized_measures=0
    iteration=0
    
    mae_chroma_list = []
    mae_percFeat_list = []
    mae_sum_list = []
    
    ind_note_chroma=0
    ind_note_percFeat=0
    ind_note_sum=0
    
    ind_ann_chroma=0
    ind_ann_percFeat=0
    ind_ann_sum=0
    
    num_mismatch_chroma = 0
    num_mismatch_perc = 0
    num_mismatch_sum = 0    
    
    #HANDLE the condition!!!
    while(tot_analized_measures < tot_measures):
        
        start1 = time.time()
        
        if(show_mae==True): 
            print('\033[1m' + "\nITERATION %d" %iteration + '\n\033[0m')
        
        ##COMPUTATION INDECES OF MEASURES
        start_index = iteration*anchor_points_distance
        end_index = (iteration*anchor_points_distance) + anchor_points_distance
        end_index = min(end_index, tot_measures)  ##NB: the end of one cut corresponds to the start of the next one

        

        
        
        ##CUT FEATURES
        chroma_audio_act, dur_audio, start_audio_frame, end_audio_frame, start_audio, end_audio, start_feature_audio, end_feature_audio = cut_feature(chroma_audio, audio_ann, start_index, end_index)
                
        chroma_symb_act, dur_symb, start_symb_frame, end_symb_frame, start_symb, end_symb,start_feature_symb, end_feature_symb= cut_feature(chroma_symb, symbolic_ann, start_index, end_index)
        
        percFeat_audio_act, _, _, _, _, _,_,_ = cut_feature(percFeat_audio, audio_ann, start_index, end_index,use_dlnco)
        
        percFeat_symb_act, _, _, _, _, _,_, _ = cut_feature(percFeat_symb, symbolic_ann, start_index, end_index,use_dlnco)
        
        
        end1 = time.time()
        elapsed_time = elapsed_time + (end1-start1)

        
        if((start_audio == end_audio) or (start_symb == end_symb)):   ##HANDLE: it works, but handle it!!
            break
        
        
        
        
        
        #SONIFICATION AUDIO AND SYMBOLIC FRAME 
        x_audio_act = x[start_audio_frame: end_audio_frame]
        x_symb_act = x_symbolic[start_symb_frame : end_symb_frame]
        
        start_meas = (iteration*anchor_points_distance)+1
        end_meas = min( ((iteration+1)*anchor_points_distance) , tot_measures )
        
        if(show_mae==True): 
            print("Analyzed Measures (%d -  %d)\n" %(start_meas, end_meas) )
        
        
        if(show_cut_files):
            print("AUDIO FILE. Start= %.2f sec | End = %.2f sec" %(start_audio, end_audio))
            ipd.display(ipd.Audio(x_audio_act, rate=Fs) )
            print("SYMBOLIC FILE. Start= %.2f sec | End = %.2f sec" %(start_symb, end_symb))
            ipd.display(ipd.Audio(x_symb_act, rate=Fs) )
            
        
        start2 = time.time()       

            
        #WARPING PATH CHROMA
        cost_matrix_chroma = compute_cost_matrix_cosine_distance(chroma_audio_act, chroma_symb_act)
        _, wp_chroma = librosa.sequence.dtw(chroma_audio_act, chroma_symb_act, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul, global_constraints=False, band_rad=0.25)
        wp_chroma = wp_chroma[::-1]
        
        #WARPING PATH PERCUSSIVE FEATURE
        cost_matrix_percFeat = LibFMP.C3.compute_cost_matrix(percFeat_audio_act, percFeat_symb_act, metric='euclidean')
        _, wp_percFeat = librosa.sequence.dtw(percFeat_audio_act, percFeat_symb_act, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
        wp_percFeat=wp_percFeat[::-1]
    
        
        ##WARPING PATH CHROMA + PERCUSSIVE FEATURE
        cost_matrix_sum = cost_matrix_chroma + cost_matrix_percFeat
        _, wp_sum= librosa.sequence.dtw(C=cost_matrix_sum, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
        wp_sum = wp_sum[::-1]
        
        
        
        
        #TRANSLATION FRAMES TO SECONDS
        wp_chroma_seconds = normalization_frames_to_second(vector = wp_chroma)
        wp_percFeat_seconds=normalization_frames_to_second(vector = wp_percFeat)
        wp_sum_seconds=normalization_frames_to_second(vector = wp_sum)
        
    
        ##ALIGN PATH
        wp_chroma_align = LibFMP.C3.compute_strict_alignment_path(wp_chroma)
        wp_percFeat_align = LibFMP.C3.compute_strict_alignment_path(wp_percFeat)
        wp_sum_align = LibFMP.C3.compute_strict_alignment_path(wp_sum)
        
        ##TRANSLATION ALIGN PATH Frames-> Seconds
        wp_chroma_align_seconds = normalization_frames_to_second(vector = wp_chroma_align)
        wp_percFeat_align_seconds=normalization_frames_to_second(vector = wp_percFeat_align)
        wp_sum_align_seconds=normalization_frames_to_second(vector = wp_sum_align)

        end2 = time.time()
        elapsed_time = elapsed_time + (end2-start2)

        
        x_label='Symbolic File' +'  -  Time (seconds)'
        y_label='Audio File' +'  -  Time (seconds)'
        x_label=""
        y_label=""
        figsize = (12,9)
    
    
        ##SHIFT ANNOTATION to let first measure start at 00:00
        audio_ann_act, symbolic_ann_act = shift_annotation(audio_ann, symbolic_ann, start_audio, end_audio, start_symb, end_symb, start_index, end_index)

        #PLOT CUT FEATURES 
        if(show_cut_features==True):
            
            plot_cut_feature(chroma_audio, start_audio, end_audio, title="Chroma Audio File",
                             ann = audio_ann_act, offset=start_audio,color_meas=c1, isMatrix=True)
            plot_cut_feature(chroma_symb, start_symb, end_symb, title="Chroma Symbolic File",
                             ann = symbolic_ann_act, offset=start_symb,color_meas=c2,isMatrix=True)
            plot_cut_feature(percFeat_audio, start_audio, end_audio, title=percFeatTitle +" Audio File",
                             ann = audio_ann_act, offset=start_audio,color_meas=c1, isMatrix=use_dlnco)
            plot_cut_feature(percFeat_symb, start_symb, end_symb, title=percFeatTitle+ " Symbolic File",
                             ann = symbolic_ann_act, offset=start_symb,color_meas=c2, isMatrix=use_dlnco)        

        off=0.01
        
        ###PLOT Warping Paths
        if (show_wp==True):
            wp_chroma_align_seconds_offset = calculate_offset_wp(wp_chroma_align_seconds, start_audio, start_symb)
            wp_percFeat_align_seconds_offset = calculate_offset_wp(wp_percFeat_align_seconds, start_audio, start_symb)
            wp_sum_align_seconds_offset = calculate_offset_wp(wp_sum_align_seconds, start_audio, start_symb)
            
            fig, im, _ = LibFMP.C3.plot_matrix_with_points(cost_matrix_chroma, wp_chroma_align_seconds_offset,  linestyle='-', aspect='auto', 
                                              clim=[1, np.max(cost_matrix_chroma)],
                                              title="", xlabel=x_label,
                                              ylabel=y_label, Fs=Fs_frame, Fs_F=Fs_frame, 
                                              extent=[start_symb-off,end_symb+off, start_audio-off, end_audio+off],
                                              figsize=figsize, colorbar=False);
            
            cbar = plt.colorbar(im)
            cbar.set_label('Cost', fontsize=15, fontfamily=font)
            cbar.ax.tick_params(labelsize='large')
            plot_measures_on_costmatrix(audio_ann_act, symbolic_ann_act, offset1=start_audio, offset2=start_symb,
                                       color_meas1 = c1, color_meas2=c2, fn1 = filename, fn2 = filename, symb_flag=True)
           
            if((name_chroma_cost!="") &  (save_fig==True) ):
                save_figure(name_chroma_cost)  
            plt.show()
            
            
            fig, im, _ = LibFMP.C3.plot_matrix_with_points(cost_matrix_percFeat, wp_percFeat_align_seconds_offset, linestyle='-', aspect='auto', 
                                              clim=[0, np.max(cost_matrix_percFeat)], 
                                              title="",
                                              xlabel=x_label, ylabel=y_label,
                                             Fs=Fs_frame, Fs_F=Fs_frame, 
                                              extent=[start_symb-off,end_symb+off, start_audio-off, end_audio+off],
                                               figsize=figsize, colorbar=False);
            
            cbar = plt.colorbar(im)
            cbar.set_label('Cost', fontsize=15, fontfamily=font)
            cbar.ax.tick_params(labelsize='large')
            plot_measures_on_costmatrix(audio_ann_act, symbolic_ann_act, offset1=start_audio, offset2=start_symb,
                                       color_meas1 = c1, color_meas2=c2, fn1 = filename, fn2 = filename, symb_flag=True)
            
            if((name_perc_cost!="") &  (save_fig==True) ):
                save_figure(name_perc_cost)  
            plt.show()
            
            fig, im, _ = LibFMP.C3.plot_matrix_with_points(cost_matrix_sum, wp_sum_align_seconds_offset, linestyle='-', aspect='auto', 
                                              clim=[1, np.max(cost_matrix_sum)], 
                                                title="", 
                                              xlabel=x_label, ylabel=y_label,
                                             Fs=Fs_frame, Fs_F=Fs_frame, 
                                              extent=[start_symb-off,end_symb+off, start_audio-off, end_audio+off],
                                               figsize=figsize, colorbar=False);
            
            cbar = plt.colorbar(im)
            cbar.set_label('Cost', fontsize=15, fontfamily=font)
            cbar.ax.tick_params(labelsize='large')
            plot_measures_on_costmatrix(audio_ann_act, symbolic_ann_act, offset1=start_audio, offset2=start_symb,
                                       color_meas1 = c1, color_meas2=c2, fn1 = filename, fn2 = filename, symb_flag=True)
            
            
            if((name_sum_cost!="") &  (save_fig==True) ):
                save_figure(name_sum_cost)  
            plt.show()
        
        
        

        start3 = time.time()
    
       ###WARPING ANNOTATION MEASURES
        symbolic_ann_meas_warped_chroma = warping_annotation(audio_ann_act, symbolic_ann_act,wp_chroma_align_seconds)
        symbolic_ann_meas_warped_percFeat = warping_annotation(audio_ann_act, symbolic_ann_act, wp_percFeat_align_seconds)
        symbolic_ann_meas_warped_sum= warping_annotation(audio_ann_act, symbolic_ann_act,wp_sum_align_seconds)
        
        end3 = time.time()
        elapsed_time = elapsed_time + (end3-start3)

       
       ###WARPING NOTES + SONIFICATION
        
        if(show_sonification == True):
            symb_list_warped_chroma, symb_warped_chroma, ind_note_chroma,elapsed_time = warping(audio = x_audio_act, midi_list= symb_list,
                                                                                        warping_path = wp_chroma_align_seconds,
                                            start_midi=start_symb, ind_note=ind_note_chroma, end_midi=end_symb, 
                                            click_track=True, show_sonification=show_sonification, 
                                            title="CHROMA", elapsed_time=elapsed_time, listen_mono=listen_mono)     
            
            symb_list_warped_percFeat, symb_warped_percFeat, ind_note_percFeat, elapsed_time  = warping(audio = x_audio_act, midi_list= symb_list, 
                                                                                    warping_path = wp_percFeat_align_seconds, 
                                           start_midi=start_symb, ind_note=ind_note_percFeat, end_midi=end_symb, 
                                            click_track=True, show_sonification=show_sonification,
                                            title=percFeatTitle,elapsed_time=elapsed_time, listen_mono=listen_mono) 
            
            symb_list_warped_sum, symb_warped_sum, ind_note_sum, elapsed_time = warping(audio = x_audio_act, midi_list= symb_list, 
                                                                               warping_path = wp_sum_align_seconds,
                                        start_midi=start_symb, ind_note=ind_note_sum, end_midi=end_symb,
                                        click_track=True, show_sonification=show_sonification, 
                                        title="SUM", elapsed_time=elapsed_time, listen_mono=listen_mono)    
            
        
        start4 = time.time()
       
        ##MEAN ABSOLUTE ERROR + SONIFICATION MEASURES
        
        mae_chroma, mae_chroma_list, num_mismatch_chroma = compute_mean_abs_error(audio_ann_act, symbolic_ann_meas_warped_chroma, 
                                                             mae_chroma_list, treshold, 
                                                             num_mismatch_chroma, start_meas, "CHROMA", show_mae)
        
        
        end4 = time.time()
        elapsed_time = elapsed_time + (end4-start4)
        
        if((show_sonification) & (show_sonification_measures)):
            sonify_measures(x_audio_act, symb_warped_chroma, audio_ann_act, symbolic_ann_meas_warped_chroma, dur_symb, "CHROMA"
                            , show_annotation=show_annotation, listen_mono=listen_mono)
        
        if((show_annotation) & (show_sonification)):
            plot_annotation_measures(audio_ann_act, symbolic_ann_act, symbolic_ann_meas_warped_chroma, 
                                     x_audio_act,x_symb_act, symb_warped_chroma, anchor_points_distance, 
                                     filename_audio, filename_xml, audio_scenario=False,
                                     offset1=start_audio, offset2 = start_symb,
                                    name_feat="", name="", save_fig=save_fig)
            

            
            
        start5 = time.time()
        
        mae_percFeat, mae_percFeat_list, num_mismatch_perc = compute_mean_abs_error(audio_ann_act, 
                                          symbolic_ann_meas_warped_percFeat, mae_percFeat_list,
                                          treshold, num_mismatch_perc, start_meas, percFeatTitle, show_mae)
        
        
        end5 = time.time()
        elapsed_time = elapsed_time + (end5-start5)
        
        if((show_sonification) & (show_sonification_measures)):
            sonify_measures(x_audio_act, symb_warped_percFeat, audio_ann_act, symbolic_ann_meas_warped_percFeat, dur_symb,
                            percFeatTitle, show_annotation=show_annotation, listen_mono=listen_mono)
        
        if((show_annotation) & (show_sonification)):
            plot_annotation_measures(audio_ann_act, symbolic_ann_act, symbolic_ann_meas_warped_percFeat,
                                     x_audio_act, x_symb_act, symb_warped_percFeat, anchor_points_distance, 
                                     filename_audio,filename_xml, audio_scenario=False,
                                     offset1=start_audio, offset2 = start_symb,
                                    name_feat="", name="", save_fig=save_fig)
        
            

            
            
        start6 = time.time()
        
        mae_sum, mae_sum_list, num_mismatch_sum = compute_mean_abs_error(audio_ann_act, symbolic_ann_meas_warped_sum, mae_sum_list, 
                                                treshold, num_mismatch_sum,start_meas, "CHROMA + " + percFeatTitle, show_mae)

        end6 = time.time()
        elapsed_time = elapsed_time + (end6-start6)
        
        if((show_sonification) & (show_sonification_measures)):
            sonify_measures(x_audio_act, symb_warped_sum, audio_ann_act, symbolic_ann_meas_warped_sum, dur_symb, 
                            "CHROMA + " + percFeatTitle, show_annotation=show_annotation, listen_mono=listen_mono)
        
        if((show_annotation) & (show_sonification)):
            plot_annotation_measures(audio_ann_act,symbolic_ann_act, symbolic_ann_meas_warped_sum,
                                     x_audio_act, x_symb_act, symb_warped_sum, anchor_points_distance,
                                     filename_audio, filename_xml, audio_scenario=False,
                                     offset1=start_audio, offset2 = start_symb,
                                    name_feat="", name="", save_fig=save_fig)

        
        tot_analized_measures += anchor_points_distance
        tot_analized_measures = min(tot_analized_measures,tot_measures)
        
        if(show_mae==True): 
            print("\nTot Measures analyzed at iteration %d ---> %d \n" %(iteration,tot_analized_measures))
            print("**********************************************************************************************\n\n\n")
          
        
        
        #name_chroma1_cut = ""
        #name_chroma2_cut = ""
        name_chroma_cost = ""
        #name_chroma_meas = ""
        #name_perc1_cut = ""
        #name_perc2_cut = ""
        name_perc_cost = ""
        name_sum_cost = ""
        #name_perc_meas = ""
        #name_sum_meas = ""
            
        iteration+=1
            
    
    anchor_points = iteration 
    tot_measures_counted = tot_measures - anchor_points
    
    mae_chroma_avg = np.mean(mae_chroma_list)
    mae_percFeat_avg = np.mean(mae_percFeat_list)
    mae_sum_avg = np.mean(mae_sum_list)
    
    mae_chroma_avg_ms = mae_chroma_avg * 1000
    mae_percFeat_avg_ms = mae_percFeat_avg * 1000
    mae_sum_avg_ms = mae_sum_avg * 1000
    
    chroma_fraction = MyFraction(num_mismatch_chroma,tot_measures_counted)
    perc_fraction = MyFraction(num_mismatch_perc,tot_measures_counted)
    sum_fraction = MyFraction(num_mismatch_sum,tot_measures_counted)
    
    chroma_mismatch_per = truncate(num_mismatch_chroma/tot_measures_counted * 100,2)
    perc_mismatch_per = truncate(num_mismatch_perc/tot_measures_counted * 100,2)
    sum_mismatch_per = truncate(num_mismatch_sum/tot_measures_counted * 100,2)
    
    mae_results=[]
    mismatch_results= []
    
    mae_results.append(mae_chroma_avg_ms)
    mae_results.append(mae_percFeat_avg_ms)
    mae_results.append(mae_sum_avg_ms)
    
    mismatch_results.append(chroma_mismatch_per)
    mismatch_results.append(perc_mismatch_per)
    mismatch_results.append(sum_mismatch_per)
    
    if(show_results==True):
        print("Number of Total Measures: " + str(tot_measures))
        print("Number of Anchor Points: " + str(anchor_points))
        print("Number of Measures Counted for statistics: " + str(tot_measures_counted))
        
    
        print("\nMAE Average - Chroma  = " + str(truncate(mae_chroma_avg,4)) + " seconds, " + str(truncate(mae_chroma_avg_ms,4)) + " milliseconds")
        print("MAE Average - " + percFeatTitle +" = " + str(truncate(mae_percFeat_avg,4)) + " seconds, " + str(truncate(mae_percFeat_avg_ms,4)) + " milliseconds")
        print("MAE Average - Sum = " + str(truncate(mae_sum_avg,4)) + " seconds, " + str(truncate(mae_sum_avg_ms,4)) + " milliseconds")
        
    
        print("\nCHROMA: Mismatches/Tot counted Measures = " +str(chroma_fraction)  + ".   Percentage of Accuracy: " +  str(chroma_mismatch_per) +" %")
        print(percFeatTitle + ": Mismatches/Tot counted Measures = " +str(perc_fraction)  + ".   Percentage of Accuracy: " +  str(perc_mismatch_per) +" %")
        print("CHROMA + " + percFeatTitle + ": Mismatches/Tot counted Measures = " +str(sum_fraction)  + ".   Percentage of Accuracy: " +  str(sum_mismatch_per) +" %")


    
    
    return mae_results, elapsed_time, mismatch_results

def get_alignment_path_audio(x1, x2, 
                             audio_ann1, audio_ann2, anchor_points_distance, 
                             chroma_audio1, chroma_audio2, 
                             dlnco_audio1, dlnco_audio2, 
                             spectral_flux_audio1, spectral_flux_audio2, 
                             sum_dlnco_sf_audio1, sum_dlnco_sf_audio2,
                             dur_audio1, dur_audio2, treshold, use_dlnco=True, use_classic_sf=False,
                             sum_perc_features=False, show_wp=False,
                             filename1  = "", filename2="",
                             show_cut_files=False, show_annotation=False, 
                             show_sonification=False, show_sonification_measures =False,
                             show_cut_features=False, show_mae=True, 
                             show_results=True, listen_mono=False, 
                             save_fig=False, interp_anchor_points=False):
    
    wp_chroma_full= []
    wp_perc_full = []
    wp_sum_full = []
    list_anchor_points = []
    
    if(show_results==True):
        print("\n\n\n############################################################################################\n")
        print('\033[1m' + "AUDIO  <--->  AUDIO.\n" +'\033[0m')
        print("File analyzed: " + '\033[1m' + filename1 + "  <--->  " + filename2 +'\033[0m') 
        print("\nMeasures analyzed per each iteration: " + str(anchor_points_distance))
        print("Treshold of Accuracy: " + str(treshold*1000) + " ms")
        print("\n______________________________________________________________________________________________\n")

       
    c1="limegreen"
    c2="darkturquoise"
    
    name_chroma1_cut = "chroma1_cut"
    name_chroma2_cut = "chroma2_cut"
    name_chroma_cost = "chroma_cost"
    name_chroma_meas = "chroma_meas"
    if(sum_perc_features==False):
        if(use_dlnco==True):
            percFeat_audio1 = copy.deepcopy(dlnco_audio1)
            percFeat_audio2 = copy.deepcopy(dlnco_audio2)
            percFeatTitle = "DLNCO"
            sumFeatTitle = "Chroma + DLNCO"
            name_perc1_cut = "dlnco1_cut"
            name_perc2_cut = "dlnco2_cut"
            name_perc_cost = "dlnco_cost"
            name_sum_cost = "chromadlnco_cost"
            name_perc_meas = "dlnco_meas"
            name_sum_meas = "chromadlnco_meas"
            
        else:

            percFeat_audio1 = copy.deepcopy(spectral_flux_audio1)
            percFeat_audio2 = copy.deepcopy(spectral_flux_audio2)
            percFeatTitle = "SPECTRAL FLUX"
            sumFeatTitle = "Chroma + Spectral Flux"
            name_perc1_cut ="sflux1_cut"
            name_perc2_cut ="sflux2_cut"
            name_perc_cost = "sflux_cost"
            name_sum_cost = "chromasflux_cost"
            name_perc_meas = "sflux_meas"
            name_sum_meas = "chromasflux_meas"

        
    else:

        percFeat_audio1 = copy.deepcopy(sum_dlnco_sf_audio1)
        percFeat_audio2 =  copy.deepcopy(sum_dlnco_sf_audio2)
        percFeatTitle = "DLNCO + SPECTRAL FLUX"
        sumFeatTitle = "Chroma + DLNCO + Spectral Flux"
        use_dlnco=True ## HANDLE THIS! It`s a problem for the cut of the features!!!!
        name_perc1_cut ="dlncosflux1_cut"
        name_perc2_cut ="dlncosflux2_cut"
        name_perc_cost = "dlncosflux_cost"
        name_sum_cost = "chromadlncosflux_cost"
        name_perc_meas = "dlncosflux_meas"
        name_sum_meas = "chromadlncosflux_meas"   
        
    
    ##PLOT ANNOTATION MEASURES
    if(show_annotation==True):
        plot_annotation_measures_fullfile(audio_ann1, audio_ann2, x1,x2, 
                                          anchor_points_distance, filename1, filename2, audio_scenario=True,
                                         save_fig=save_fig)
    
        
        
    num_audio_measures1 = len(audio_ann1)-1
    num_audio_measures2 = len(audio_ann2)-1

    
    ##CONTROL the number of the measures
    if(num_audio_measures1!=num_audio_measures2):
        raise RuntimeError('The number of measures of the two files are different!')
    else:
        tot_measures = num_audio_measures1

      
    if(show_mae):
        print("______________________________________________________________________________________________")
    
    elapsed_time = 0
    
    tot_analized_measures=0
    iteration=0
    
    mae_chroma_list = []
    mae_percFeat_list = []
    mae_sum_list = []
    
    ind_note_chroma=0
    ind_note_percFeat=0
    ind_note_sum=0
    
    ind_ann_chroma=0
    ind_ann_percFeat=0
    ind_ann_sum=0
    
    num_mismatch_chroma = 0
    num_mismatch_perc = 0
    num_mismatch_sum = 0
    
    
    #Main Iteration
    while(tot_analized_measures < tot_measures):
        
        start1 = time.time()
        
        if(show_mae):
            print('\033[1m' + "\nITERATION %d" %iteration + '\n\033[0m')

        ##COMPUTATION INDECES OF MEASURES
        start_index = iteration*anchor_points_distance
        end_index = (iteration*anchor_points_distance) + anchor_points_distance
        end_index = min(end_index, tot_measures)  ##NB: the end of one cut corresponds to the start of the next one

        
        ##CUT FEATURES
        chroma_audio_act1, dur_audio1, start_audio_frame1, end_audio_frame1, start_audio1, end_audio1, start_feature_audio1, end_feature_audio1 = cut_feature(chroma_audio1, audio_ann1, start_index, end_index)
                
        chroma_audio_act2, dur_audio2, start_audio_frame2, end_audio_frame2, start_audio2, end_audio2, start_feature_audio2, end_feature_audio2 = cut_feature(chroma_audio2, audio_ann2, start_index, end_index)
        
        percFeat_audio_act1, _, _, _, _, _,_,_ = cut_feature(percFeat_audio1, audio_ann1, start_index, end_index,use_dlnco)
        
        percFeat_audio_act2, _, _, _, _, _,_,_ = cut_feature(percFeat_audio2, audio_ann2, start_index, end_index,use_dlnco)
        
        
        end1 = time.time()
        elapsed_time = elapsed_time + (end1-start1)

        
        if((start_audio1 == end_audio1) or (start_audio2 == end_audio2)):   ##HANDLE: it works, but handle it!!
            break
        
        
        
        
        
        #SONIFICATION AUDIO AND SYMBOLIC FRAME 
        x_audio_act1 = x1[start_audio_frame1: end_audio_frame1]
        x_audio_act2 = x2[start_audio_frame2: end_audio_frame2]
        
        start_meas = (iteration*anchor_points_distance)+1
        end_meas = min( ((iteration+1)*anchor_points_distance) , tot_measures )
        
        if(show_mae):
            print("Analyzed Measures (%d -  %d)\n" %(start_meas, end_meas) )
        
        if(show_cut_files):
            print("AUDIO 1 FILE. Start= %.2f sec | End = %.2f sec" %(truncate(start_audio1,2), truncate(end_audio1,2)))
            ipd.display(ipd.Audio(x_audio_act1, rate=Fs) )
            print("AUDIO 2 FILE. Start= %.2f sec | End = %.2f sec" %(truncate(start_audio2,2), truncate(end_audio2,2)))
            ipd.display(ipd.Audio(x_audio_act2, rate=Fs) )
            
            name_audio_act1 = filename1 + "_" + str(iteration) +".wav"
            name_audio_act2 = filename2 + "_" + str(iteration) +".wav"
            save_audio(x_audio_act1.T, name_audio_act1)
            save_audio(x_audio_act2.T, name_audio_act2)
            
        
        start2 = time.time()       

            
        #WARPING PATH CHROMA
        cost_matrix_chroma = compute_cost_matrix_cosine_distance(chroma_audio_act1, chroma_audio_act2)
        _, wp_chroma = librosa.sequence.dtw(chroma_audio_act1, chroma_audio_act2, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul, global_constraints=False, band_rad=0.25)
        wp_chroma = wp_chroma[::-1]
        
        #WARPING PATH PERCUSSIVE FEATURE
        cost_matrix_percFeat = LibFMP.C3.compute_cost_matrix(percFeat_audio_act1, percFeat_audio_act2, metric='euclidean')
        _, wp_percFeat = librosa.sequence.dtw(percFeat_audio_act1, percFeat_audio_act2, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
        wp_percFeat=wp_percFeat[::-1]
    
        
        ##WARPING PATH CHROMA + PERCUSSIVE FEATURE
        cost_matrix_sum = cost_matrix_chroma + cost_matrix_percFeat
        _, wp_sum= librosa.sequence.dtw(C=cost_matrix_sum, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
        wp_sum = wp_sum[::-1]
        

        #TRANSLATION FRAMES TO SECONDS
        wp_chroma_seconds = normalization_frames_to_second(vector = wp_chroma)
        wp_percFeat_seconds=normalization_frames_to_second(vector = wp_percFeat)
        wp_sum_seconds=normalization_frames_to_second(vector = wp_sum)
        
    
        ##ALIGN PATH
        wp_chroma_align = LibFMP.C3.compute_strict_alignment_path(wp_chroma)
        wp_percFeat_align = LibFMP.C3.compute_strict_alignment_path(wp_percFeat)
        wp_sum_align = LibFMP.C3.compute_strict_alignment_path(wp_sum)
        
        ##TRANSLATION ALIGN PATH Frames-> Seconds
        wp_chroma_align_seconds = normalization_frames_to_second(vector = wp_chroma_align)
        wp_percFeat_align_seconds=normalization_frames_to_second(vector = wp_percFeat_align)
        wp_sum_align_seconds=normalization_frames_to_second(vector = wp_sum_align)
        
        
        #TRANSLATION ALIGN PATH Frames -> SAMPLES
        wp_chroma_samples = normalization_frames_to_samples(vector = wp_chroma_align)
        wp_percFeat_samples=normalization_frames_to_samples(vector = wp_percFeat_align)
        wp_sum_samples=normalization_frames_to_samples(vector = wp_sum_align)
        
        if(interp_anchor_points):
            wp_chroma_full = create_full_wp(wp_chroma_align_seconds, wp_chroma_full, start_audio1, start_audio2)
            wp_perc_full = create_full_wp(wp_percFeat_align_seconds, wp_perc_full, start_audio1, start_audio2)
            wp_sum_full = create_full_wp(wp_sum_align_seconds, wp_sum_full, start_audio1, start_audio2)
            
            list_anchor_points = create_list_anchor_points(wp_chroma_align_seconds, list_anchor_points, start_audio1, start_audio2)
        
        end2 = time.time()
        elapsed_time = elapsed_time + (end2-start2)

        
        ##SHIFT ANNOTATION to let first measure start at 00:00
        audio_ann_act1, audio_ann_act2 = shift_annotation(audio_ann1, audio_ann2, start_audio1, end_audio1, start_audio2, end_audio2, start_index, end_index)
                
            
        #PLOT CUT FEATURES 
        if(show_cut_features==True):

            plot_cut_feature(chroma_audio1, start_audio1, end_audio1, title=filename1 + "\nChroma " ,
                             ann = audio_ann_act1, offset=start_audio1,color_meas=c1, isMatrix=True,
                            name=name_chroma1_cut, save_fig=save_fig)
            plot_cut_feature(chroma_audio2, start_audio2, end_audio2, title=filename2 + "\nChroma ",
                             ann=audio_ann_act2, offset=start_audio2, color_meas=c2, isMatrix=True,
                            name=name_chroma2_cut, save_fig=save_fig)
            plot_cut_feature(percFeat_audio1, start_audio1, end_audio1, title= filename1 +"\n" +percFeatTitle,
                             ann=audio_ann_act1, offset=start_audio1, color_meas=c1, isMatrix=use_dlnco,
                            name=name_perc1_cut, save_fig=save_fig)
            plot_cut_feature(percFeat_audio2, start_audio2, end_audio2, title=filename2 +"\n" +percFeatTitle,
                             ann=audio_ann_act2, offset=start_audio2, color_meas=c2, isMatrix=use_dlnco,
                            name=name_perc2_cut, save_fig=save_fig)        

        off=0.01
        
        ###PLOT Warping Paths
        if ((show_wp==True) ):
            #save_fig=True
            x_label='Audio File: ' + filename2 +'  -  Time (seconds)'
            y_label='Audio File: ' + filename1 +'  -  Time (seconds)'
            figsize = (12,9)
            
            wp_chroma_align_seconds_offset = calculate_offset_wp(wp_chroma_align_seconds, start_audio1, start_audio2)
            wp_percFeat_align_seconds_offset = calculate_offset_wp(wp_percFeat_align_seconds, start_audio1, start_audio2)
            wp_sum_align_seconds_offset = calculate_offset_wp(wp_sum_align_seconds, start_audio1, start_audio2)
            #title = 'Chroma cost matrix and strict-aligned warping path'
            title=""
            fig, im, _ = LibFMP.C3.plot_matrix_with_points(cost_matrix_chroma, wp_chroma_align_seconds_offset,  linestyle='-', aspect='auto', 
                                              clim=[1, np.max(cost_matrix_chroma)],
                                              title= title, Fs=Fs_frame, Fs_F=Fs_frame, 
                                              extent=[start_audio2-off,end_audio2+off, start_audio1-off, end_audio1+off],
                                              figsize=figsize, colorbar=False)
            cbar = plt.colorbar(im)
            cbar.set_label('Cost', fontsize=15, fontfamily=font)
            cbar.ax.tick_params(labelsize='large')
            plt.xlabel(x_label, fontsize=15, fontfamily=font)
            plt.ylabel(y_label, fontsize=15, fontfamily=font)
            plot_measures_on_costmatrix(audio_ann_act1, audio_ann_act2, offset1=start_audio1, offset2=start_audio2,
                                       color_meas1 = c1, color_meas2=c2, fn1 = filename1, fn2 = filename2)

            if((name_chroma_cost!="") &  (save_fig==True)):
                save_figure(name_chroma_cost)          
                
            plt.show()
            
            #title = percFeatTitle + ' cost matrix and strict-aligned warping path'
            title = ""
            fig, im, _ = LibFMP.C3.plot_matrix_with_points(cost_matrix_percFeat, wp_percFeat_align_seconds_offset, linestyle='-', aspect='auto', 
                                              clim=[0, np.max(cost_matrix_percFeat)], 
                                              title=title,
                                             Fs=Fs_frame, Fs_F=Fs_frame, 
                                              extent=[start_audio2-off,end_audio2+off, start_audio1-off, end_audio1+off],
                                               figsize=figsize, colorbar=False)
            cbar = plt.colorbar(im)
            cbar.set_label('Cost',fontsize=15, fontfamily=font)
            cbar.ax.tick_params(labelsize='large')
            plt.xlabel(x_label, fontsize=15, fontfamily=font)
            plt.ylabel(y_label, fontsize=15, fontfamily=font)
            plot_measures_on_costmatrix(audio_ann_act1, audio_ann_act2, offset1=start_audio1, offset2=start_audio2,
                                       color_meas1 = c1, color_meas2=c2, fn1 = filename1, fn2 = filename2)
            
            if((name_perc_cost!="") &  (save_fig==True) ):
                save_figure(name_perc_cost)          
                
            plt.show()
            
            #title = sumFeatTitle + ' cost matrix and strict-aligned warping path'
            title = ""
            fig, im, _ = LibFMP.C3.plot_matrix_with_points(cost_matrix_sum, wp_sum_align_seconds_offset, linestyle='-', aspect='auto', 
                                              clim=[1, np.max(cost_matrix_sum)], title=title, 
                                             Fs=Fs_frame, Fs_F=Fs_frame, 
                                              extent=[start_audio2-off,end_audio2+off, start_audio1-off, end_audio1+off],
                                              figsize=figsize,colorbar=False);
            
            
            cbar = plt.colorbar(im)
            cbar.set_label('Cost', fontsize=15, fontfamily=font)
            cbar.ax.tick_params(labelsize='large')
            plot_measures_on_costmatrix(audio_ann_act1, audio_ann_act2, offset1=start_audio1, offset2=start_audio2,
                                       color_meas1 = c1, color_meas2=c2, fn1 = filename1, fn2 = filename2)
            
            plt.xlabel(x_label, fontsize=15, fontfamily=font)
            plt.ylabel(y_label, fontsize=15, fontfamily=font)
            
            if((name_sum_cost!="") &  (save_fig==True) ):
                save_figure(name_sum_cost)          
            plt.show()

            

        start3 = time.time()
    
        ##WARPING ANNOTATION MEASURES
        audio2_ann_meas_warped_chroma = warping_annotation(audio_ann_act1, audio_ann_act2,wp_chroma_align_seconds)
        audio2_ann_meas_warped_percFeat = warping_annotation(audio_ann_act1, audio_ann_act2, wp_percFeat_align_seconds)
        audio2_ann_meas_warped_sum = warping_annotation(audio_ann_act1, audio_ann_act2,wp_sum_align_seconds)
        
        end3 = time.time()
        elapsed_time = elapsed_time + (end3-start3)
       
    
        #WARPING NOTES + SONIFICATION
        if(show_sonification):
            sonification_chroma, x2_warped_chroma = warping_audio(x_audio_act1,x_audio_act2,wp_chroma_samples, wp_chroma_align_seconds, "CHROMA", iteration, listen_mono)
            
            sonification_perc, x2_warped_percFeat = warping_audio(x_audio_act1,x_audio_act2,wp_percFeat_samples, wp_percFeat_align_seconds, percFeatTitle, iteration, listen_mono)
            
            sonification_sum, x2_warped_sum = warping_audio(x_audio_act1,x_audio_act2,wp_sum_samples, wp_sum_align_seconds, sumFeatTitle, iteration, listen_mono)

        
        
        start4 = time.time()
       
        ##MEAN ABSOLUTE ERROR + SONIFICATION MEASURES

        mae_chroma, mae_chroma_list, num_mismatch_chroma = compute_mean_abs_error(audio_ann_act1, 
                                                                                  audio2_ann_meas_warped_chroma, 
                                                                                  mae_chroma_list, 
                                                                                  treshold, 
                                                                                  num_mismatch_chroma,
                                                                                  start_meas,
                                                                                  "CHROMA",
                                                                                  show_mae)
        
        end4 = time.time()
        elapsed_time = elapsed_time + (end4-start4)
        
        if(show_sonification_measures & show_sonification):
            sonify_measures_audio(sonification_chroma, audio_ann_act1, audio2_ann_meas_warped_chroma,
                                  "CHROMA", show_annotation, listen_mono=listen_mono)
        

        if((show_annotation) & (show_sonification)):
            plot_annotation_measures(audio_ann_act1, audio_ann_act2, audio2_ann_meas_warped_chroma,
                                     x_audio_act1, x_audio_act2, x2_warped_chroma, anchor_points_distance,
                                     filename1, filename2, audio_scenario=True,
                                    offset1 = start_audio1,offset2 = start_audio2,
                                     name_feat = "Chroma", name=name_chroma_meas, save_fig=save_fig)
        
        
        
        start5 = time.time()
        
        mae_percFeat, mae_percFeat_list, num_mismatch_perc = compute_mean_abs_error(audio_ann_act1, 
                                                                                    audio2_ann_meas_warped_percFeat,
                                                                                    mae_percFeat_list,
                                                                                    treshold,
                                                                                    num_mismatch_perc,
                                                                                    start_meas,
                                                                                    percFeatTitle,
                                                                                    show_mae)
        
        end5 = time.time()
        elapsed_time = elapsed_time + (end5-start5)
        
        if(show_sonification_measures & show_sonification):
            sonify_measures_audio(sonification_perc, audio_ann_act1, audio2_ann_meas_warped_percFeat,
                                  percFeatTitle, show_annotation, listen_mono=listen_mono)
        
        if((show_annotation) & (show_sonification)):
            plot_annotation_measures(audio_ann_act1, audio_ann_act2, audio2_ann_meas_warped_percFeat,
                                     x_audio_act1, x_audio_act2, x2_warped_percFeat, anchor_points_distance,
                                     filename1, filename2, audio_scenario=True,
                                    offset1 = start_audio1,offset2 = start_audio2,
                                     name_feat = percFeatTitle, name=name_perc_meas, save_fig=save_fig)
                
        
        start6 = time.time()
        
        mae_sum, mae_sum_list, num_mismatch_sum = compute_mean_abs_error(audio_ann_act1,
                                                                         audio2_ann_meas_warped_sum,
                                                                         mae_sum_list, 
                                                                         treshold,
                                                                         num_mismatch_sum,
                                                                         start_meas,
                                                                         sumFeatTitle,
                                                                         show_mae)
        
        end6 = time.time()
        elapsed_time = elapsed_time + (end6-start6)
        
        if(show_sonification_measures & show_sonification):
            sonify_measures_audio(sonification_sum, audio_ann_act1, audio2_ann_meas_warped_sum, 
                                  sumFeatTitle, show_annotation, listen_mono=listen_mono)
        
        if((show_annotation) & (show_sonification)):
            plot_annotation_measures(audio_ann_act1, audio_ann_act2, audio2_ann_meas_warped_sum,
                                     x_audio_act1, x_audio_act2, x2_warped_sum, anchor_points_distance, 
                                     filename1, filename2, audio_scenario=True,
                                    offset1 = start_audio1,offset2 = start_audio2,
                                     name_feat= sumFeatTitle, name=name_sum_meas, save_fig=save_fig)
                


        
        
        tot_analized_measures += anchor_points_distance
        tot_analized_measures = min(tot_analized_measures,tot_measures)
        
        if(show_mae):
            print("\nTot Measures analyzed at iteration %d ---> %d " %(iteration,tot_analized_measures))
 
        iteration +=1
        
        name_chroma1_cut = ""
        name_chroma2_cut = ""
        name_chroma_cost = ""
        name_chroma_meas = ""
        name_perc1_cut = ""
        name_perc2_cut = ""
        name_perc_cost = ""
        name_sum_cost = ""
        name_perc_meas = ""
        name_sum_meas = ""

        if(show_mae):        
            print("__________________________________________________________________________________________________________\n\n\n")
       

    ##Mean Absolute Error Calculation
    
    anchor_points = iteration 
    tot_measures_counted = tot_measures - anchor_points
    
    mae_chroma_avg = np.mean(mae_chroma_list)
    mae_percFeat_avg = np.mean(mae_percFeat_list)
    mae_sum_avg = np.mean(mae_sum_list)
    
    mae_chroma_avg_ms = mae_chroma_avg * 1000
    mae_percFeat_avg_ms = mae_percFeat_avg * 1000
    mae_sum_avg_ms = mae_sum_avg * 1000
    
    chroma_fraction = MyFraction(num_mismatch_chroma,tot_measures_counted)
    perc_fraction = MyFraction(num_mismatch_perc,tot_measures_counted)
    sum_fraction = MyFraction(num_mismatch_sum,tot_measures_counted)
    
    chroma_mismatch_per = truncate(num_mismatch_chroma/tot_measures_counted * 100,2)
    perc_mismatch_per = truncate(num_mismatch_perc/tot_measures_counted * 100,2)
    sum_mismatch_per = truncate(num_mismatch_sum/tot_measures_counted * 100,2)
    
    mae_results=[]
    mismatch_results= []
    
    mae_results.append(mae_chroma_avg_ms)
    mae_results.append(mae_percFeat_avg_ms)
    mae_results.append(mae_sum_avg_ms)
    
    mismatch_results.append(chroma_mismatch_per)
    mismatch_results.append(perc_mismatch_per)
    mismatch_results.append(sum_mismatch_per)
    
    if(show_results==True):
        print("Number of Total Measures: " + str(tot_measures))
        print("Number of Anchor Points: " + str(anchor_points))
        print("Number of Measures Counted for statistics: " + str(tot_measures_counted))
        
    
        print("\nMAE Average - Chroma  = " + str(truncate(mae_chroma_avg,4)) + " seconds, " + str(truncate(mae_chroma_avg_ms,4)) + " milliseconds")
        print("MAE Average - " + percFeatTitle +" = " + str(truncate(mae_percFeat_avg,4)) + " seconds, " + str(truncate(mae_percFeat_avg_ms,4)) + " milliseconds")
        print("MAE Average - Sum = " + str(truncate(mae_sum_avg,4)) + " seconds, " + str(truncate(mae_sum_avg_ms,4)) + " milliseconds")
        
    
        print("\nCHROMA: Mismatches/Tot counted Measures = " +str(chroma_fraction)  + ".   Percentage of Accuracy: " +  str(chroma_mismatch_per) +" %")
        print(percFeatTitle + ": Mismatches/Tot counted Measures = " +str(perc_fraction)  + ".   Percentage of Accuracy: " +  str(perc_mismatch_per) +" %")
        print("CHROMA + " + percFeatTitle + ": Mismatches/Tot counted Measures = " +str(sum_fraction)  + ".   Percentage of Accuracy: " +  str(sum_mismatch_per) +" %")

        
    if(interp_anchor_points==True):
        interpolate_anchor_points(list_anchor_points, wp_chroma_full, name="CHROMA")
        interpolate_anchor_points(list_anchor_points, wp_perc_full, name=percFeatTitle)
        interpolate_anchor_points(list_anchor_points, wp_sum_full, name="CHROMA + " + percFeatTitle)
    
    return mae_results, elapsed_time, mismatch_results


def create_full_wp(actual_wp, full_wp, start1, start2):
    '''
    Create the full warping path, iteration by iteration. 
    The first and the last points are not included because they are anchor points
    start1, start2 = offset of the wp
    '''
    
    for i in range(actual_wp.shape[0]):
        if((i>0) & (i<actual_wp.shape[0]-1)):                  ##delete anchor points from the wp
            full_wp.append([actual_wp[i,0] + start1, actual_wp[i,1] + start2])
        
    return full_wp


def create_list_anchor_points(actual_wp, list_anchor_points, start1, start2):
    '''
    Create the list of anchor points, adding only the first row of the warping path
    '''
    
    list_anchor_points.append([actual_wp[0,0] + start1, actual_wp[0,1] + start2])
    
    return list_anchor_points


def interpolate_anchor_points(list_anchor_points, full_wp, name):
    
    """
    Function used to align the midi file to the audio file, using the warping_path    
    warping_path[:,0] = wp coordinates of the audio
    warping_path[:,1] = wp coordinates of the midi
    
    """
    #print("LIST ANCHOR POINTS")
    #print(list_anchor_points )
        
    #print("FULL WARPING PATH")
    #print(full_wp)  
    
    full_wp = np.array(full_wp)
    
    list_anchor_points_gt = []
    list_anchor_points_warped = []
    
    
    interp = interp1d(full_wp[:, 1], full_wp[:, 0], fill_value='extrapolate')

    dif = 0
    tot_dif = 0
    L = len(list_anchor_points)
    for i in range(L):
        
        anchor_point_gt = list_anchor_points[i][0]
        list_anchor_points_gt.append(anchor_point_gt)
        anchor_point_warped = float(max(0, interp(list_anchor_points[i][1])))
        list_anchor_points_warped.append(anchor_point_warped)
    
        dif = np.abs(anchor_point_gt-anchor_point_warped) *1000   ##difference is in seconds. We multiply by 1000 to have in ms
        tot_dif = tot_dif + dif
        
    mae_anchors = tot_dif / L
    
    print("\n\nMAE ANCHORS : " + str(mae_anchors) + " ms")
    print("\nLIST ANCHOR POINTS")
    print(list_anchor_points_gt )
    
    print("\n\nWARPED LIST ANCHOR POINTS using " + name + " Warping Path")
    print(list_anchor_points_warped)
    
    
    ################################
    
    fig, ax = plt.subplots(1, 1, figsize=(15,3))
        

    #labels_set = sorted(set([label for label in list_anchor_points_gt]))
    label_keys_gt={}
    label_keys_wp={}
    colors_list_gt = []
    colors_list_wp = []
    col_anchor="red"
    col_wp_anchor = "purple"
    
    for i in range(len(list_anchor_points_gt)):
        colors_list_gt.append(col_anchor)    
        colors_list_wp.append(col_wp_anchor)
        
    colors_anchor = color_argument_to_dict(colors_list_gt, list_anchor_points_gt)
    colors_wp_anchor = color_argument_to_dict(colors_list_wp, list_anchor_points_gt)
    
    
    for key, value in colors_anchor.items():
        label_keys_gt[0] = {}
        label_keys_gt[0]['color'] = value
        
    for key, value in colors_wp_anchor.items():
        label_keys_wp[0] = {}
        label_keys_wp[0]['color'] = value
        
    pos_measures = []
    pos_measures_labels = []
    

    for pos in list_anchor_points_gt:
        linestyle="-"
        linewidth=3
        alpha=1
        pos = pos/4
        ax.axvline(pos,**label_keys_gt[0],  linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    
    for pos in list_anchor_points_warped:
        linestyle="--"
        linewidth=3
        alpha=1
        pos_plot=pos/4
        ax.axvline(pos_plot,**label_keys_wp[0],  linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        pos_measures.append(pos_plot)
        pos_measures_labels.append(truncate(pos,2))    
        
            
    plt.xticks(pos_measures, labels=pos_measures_labels, fontsize=12.5, fontfamily=font)
    #ax.set_xlim([0, 30])
    plt.show()

    
def warping_audio(x1,x2, wp, wp_click, title, iteration, listen_mono=False):
    
    wp = np.append(wp, [[len(x1)-1,len(x2)-1]],axis=0)
    wp = np.flip(wp, 1)   ## to let align x2 to x1
    
    
    x2_warped = TSM.HPS_TSM(x2, Fs, wp, hps_ana_hop=256, hps_win=TSM.win(1024, 2), hps_zero_pad=0, hps_fil_len_harm=11,
                       hps_fil_len_perc=11, pv_syn_hop=512, pv_win=TSM.win(2048, 2), pv_zero_pad=0, 
                       pv_restore_energy=False, pv_fft_shift=False, ola_syn_hop=128, ola_win=TSM.win(256, 2))
    
    
    x2_warped = x2_warped.T[0,:]
    
    x2_warped_padded = np.zeros((x2_warped.shape[0] + 1000))   ##HANDLE A CASA offset per far suonare il click

    x2_warped_padded[:x2_warped.shape[0]] = x2_warped
    
    y = np.zeros((2,x2_warped_padded.shape[0]))
    

    y[0,:x1.shape[0]] = x1
    y[1,:] = x2_warped_padded  
    
    
    print("Sonification --> Time Scale Modification using " + title +" warping path")

    
    
    
    if(listen_mono==True):
        x_mono = y.sum(axis=0) / 2
        ipd.display(ipd.Audio(x_mono, rate=Fs, normalize=True) )   
        
        
    else:
        print("Left: Original Audio 1 <---> Right: Audio 2 warped to Audio 1")
        ipd.display(ipd.Audio(y, rate=Fs, normalize=True))


    name_audio = "audio-to-audio_" + title.lower() + "_" + str(iteration)+".wav"
    save_audio(y.T, name_audio)
    
    return y, x2_warped_padded 



def get_dlnco_mae(audio_ann1, audio_ann2, num_measures, dlnco_audio1, dlnco_audio2,
                       dur_audio1, dur_audio2, param_peaks, show_wp=False,
                      filename1  = "", filename2="",
                       show_cut_files=False, show_annotation=False, show_sonification=False, 
                       show_sonification_measures =False,
                      show_cut_features=False, show_mae=True):
    
    
    #print("\n###############################################################\n")
    #print("Peak Parameters: " + str(param_peaks))
    
    
    

    percFeat_audio1 = copy.deepcopy(dlnco_audio1)
    percFeat_audio2 = copy.deepcopy(dlnco_audio2)
    percFeatTitle = "DLNCO"
    
    
    
    num_audio_measures1 = len(audio_ann1)-1
    num_audio_measures2 = len(audio_ann2)-1

    
    ##CONTROL the number of the measures
    if(num_audio_measures1!=num_audio_measures2):
        raise RuntimeError('The number of measures of the two files are different!')
    else:
        tot_measures = num_audio_measures1

    
    tot_analized_measures=0
    iteration=0
    

    mae_percFeat_list = []

    
    ind_note_percFeat=0
    ind_ann_percFeat=0
    
    #HANDLE the condition!!!
    while(tot_analized_measures < tot_measures):
        

        
        ##COMPUTATION INDECES OF MEASURES
        start_index = iteration*num_measures
        end_index = (iteration*num_measures) + num_measures
        end_index = min(end_index, tot_measures)  ##NB: the end of one cut corresponds to the start of the next one

        
        
        
        ##CUT FEATURES
        percFeat_audio_act1, dur_audio1, start_audio_frame1, end_audio_frame1, start_audio1, end_audio1, start_feature_audio1, end_feature_audio1 = cut_feature(percFeat_audio1, audio_ann1, start_index, end_index,cut_matrix=True)

        percFeat_audio_act2, dur_audio2, start_audio_frame2, end_audio_frame2, start_audio2, end_audio2, start_feature_audio2, end_feature_audio2 = cut_feature(percFeat_audio2, audio_ann2, start_index, end_index,cut_matrix=True)
        
        

        
        if((start_audio1 == end_audio1) or (start_audio2 == end_audio2)):   ##HANDLE: it works, but handle it!!
            break

        start_meas = (iteration*num_measures)+1
        end_meas = min( ((iteration+1)*num_measures) , tot_measures )
        

            
        #WARPING PATH PERCUSSIVE FEATURE
        cost_matrix_percFeat = LibFMP.C3.compute_cost_matrix(percFeat_audio_act1, percFeat_audio_act2, metric='euclidean')
        _, wp_percFeat = librosa.sequence.dtw(percFeat_audio_act1, percFeat_audio_act2, step_sizes_sigma=step_sizes_sigma,  weights_mul=weights_mul)
        wp_percFeat=wp_percFeat[::-1]
    

        

        #TRANSLATION FRAMES TO SECONDS
        wp_percFeat_seconds=normalization_frames_to_second(vector = wp_percFeat)
        
    
        ###ALIGN PATH
        wp_percFeat_align = LibFMP.C3.compute_strict_alignment_path(wp_percFeat)
        
        ##TRANSLATION ALIGN PATH Frames-> Seconds
        wp_percFeat_align_seconds=normalization_frames_to_second(vector = wp_percFeat_align)
        

        ##SHIFT ANNOTATION to let first measure start at 00:00
        audio_ann_act1, audio_ann_act2 = shift_annotation(audio_ann1, audio_ann2, start_audio1, end_audio1, start_audio2, end_audio2, start_index, end_index)
        
    

        audio2_ann_meas_warped_percFeat = warping_annotation(audio_ann_act1, audio_ann_act2, wp_percFeat_align_seconds, 
                                                              percFeatTitle)

        
        mae_percFeat, mae_percFeat_list, _ = compute_mean_abs_error(audio_ann_act1, 
                                          audio2_ann_meas_warped_percFeat, mae_percFeat_list,
                                                                    start_meas = start_meas,title=percFeatTitle, show_mae=False)

       
        tot_analized_measures += num_measures
        tot_analized_measures = min(tot_analized_measures,tot_measures)
        
 
        iteration +=1

       

    mae_percFeat_avg = np.mean(mae_percFeat_list)
    
    mae_percFeat_avg_ms = mae_percFeat_avg * 1000
    
    
    
    #print("MAE Average - DLNCO = " + str(truncate(mae_percFeat_avg,4)) + " seconds, " + str(truncate(mae_percFeat_avg_ms,4)) + " milliseconds")
    #print("\n###############################################################\n")
    
    return mae_percFeat_avg_ms