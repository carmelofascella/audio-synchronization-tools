from utils import  H, Fs, Fs_frame, dur_plot, font, figsize_full_feat
from utils import truncate, save_figure, check_line_annotations, color_argument_to_dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import librosa
import LibFMP

def plot_iirt(X, Fs,dur,title, H=H, save_fig=False, save_name="", type_name=""):
    
    figure, ax = plt.subplots(figsize=figsize_full_feat)
    
    y_labels=np.arange(24,108+1)


    librosa.display.specshow(X, y_axis='frames', x_axis='time', sr=Fs, hop_length=H,
                            y_coords=y_labels, cmap='gray_r')
    
    plt.xlim([0,dur_plot+0.01])
    
    seconds=np.arange(0,dur_plot+0.01,2)
    plt.xticks(seconds, fontsize=12.5, fontfamily=font)
    
    y_labels=np.arange(24,108+1,3)
    plt.yticks(y_labels, fontsize=12.5, fontfamily=font)
    
    plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
    plt.ylabel('Pitch number', fontsize=15, fontfamily=font)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize='large')
    
    
    ##just to plot
    a = X[:,0:int(dur_plot*Fs_frame)]
    max_plot = a.max()
    plt.clim(0,max_plot - (max_plot/1.5))
    
    if(save_fig==True):   
        name = "iirt" + type_name + "_" + save_name
        save_figure(name=name)
        
    else:
        plt.title(title)
    
    plt.show()



def plot_chromagram(X, Fs,dur, H=H, start=0, title="", ann=[], show_measures=False, filename="", save_fig=False):
    fig = plt.figure(figsize=figsize_full_feat)
    
    librosa.display.specshow(X, x_axis='time', 
                             y_axis='chroma', sr=Fs, cmap='gray_r', hop_length=H)
    
    ##HANDLE dur in realta e l end!
    
    #dur_plot=dur
    dur_plot = 10
    
    plt.xlim([start, start+dur_plot+0.01])
    
    chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
    chroma_yticks=np.arange(0.5,12, step=1)
    plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal', fontsize=12.5, fontfamily=font)
    
    x_labels = np.arange(0,dur_plot+0.01,2)
    plt.xticks(x_labels, fontsize=12.5, fontfamily=font)  
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize='large')
    
    plt.xlabel('Time (seconds)', fontsize=15, fontfamily=font)
    plt.ylabel('Pitch Class', fontsize=15, fontfamily=font)
    #plt.title(title, fontsize=17.5, fontfamily=font)

    if(show_measures==True):
        plot_measures_on_feature(ann)

        
    if(save_fig==True):   
        #name = "chroma_"+filename
        name = "cens_"+filename 
        save_figure(name=name)
    
    plt.show() 


def plot_zoom(dnlco,zoom_dur, figsize, filename="", plot_name=""):
    
    num_frames = int(zoom_dur*Fs_frame)
    dnlco_zoom = dnlco[:,0:num_frames]
    
    fig = plt.figure(figsize=figsize)
    plt.imshow(dnlco_zoom, origin='lower', aspect = 'auto', cmap='gray_r', extent=[0,zoom_dur, 0, 11])
    chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
    chroma_yticks=np.arange(12)
    plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('12 Pitch ')
    plt.colorbar()
    title = plot_name +"  -  " + filename
    plt.title(title)
    
    return dnlco_zoom


def plot_annotation_measures_fullfile(ann1, ann2, file_audio1, file_audio2, anchor_points_distance, filename1, filename2, audio_scenario=False, offset=0, save_fig=False):
    """
    Function used to plot the annotation measures from both files
    
    """
    figsize=(15,3)

    fig= plot_annotation_line(ann1, file_audio1, anchor_points_distance, offset_x = offset, 
                              figsize=figsize, full_flag=True )
    
    #plt.title("Audio File 1 with original Annotation Measures.\n File:  " + filename1)
    
    if(save_fig==True):
        name="full_song_measures1"
        save_figure(name)
    
    fig = plot_annotation_line(ann2, file_audio2, anchor_points_distance, offset_x = offset, figsize=figsize,
                              col_no_anchor = "royalblue", full_flag=True)
    
    #if(audio_scenario==False):
        #plt.title("Symbolic File with original annotation measures.\n File:  " + filename2)
        
    #else:
        #plt.title("Audio File 2 with original annotation measures.\n File:  " + filename2)

    if(save_fig==True):
        name = "full_song_measures2"
        save_figure(name)
        
    plt.show()

def plot_annotation_measures(ann1, ann2, ann2_warp, file_audio1, file_audio2, file_audio2_warped, anchor_points_distance, filename1="", filename2="", audio_scenario=False, offset1=0,offset2=0, name_feat="", name="", save_fig=False):
    """
    Function used to plot the annotation measures from both files
    offset1
    """
    figsize=(12,2.25)
    figsize_comp=(11.05,2.25)
    fig= plot_annotation_line(ann1, file_audio1, anchor_points_distance, offset_x = offset1,figsize=figsize)
    #plt.title("Audio File 1 with original Annotation Measures.\n File:  " + filename1)
    
    if((name!="") & (save_fig==True)):
        name1 = name +"1"
        save_figure(name1)
        
        
    fig= plot_annotation_line_comparison(ann2, ann2_warp, file_audio2, anchor_points_distance,
                                      offset_x = offset2, figsize=figsize_comp, off_warp = offset1)
    #plt.title("Audio File 2 with original Annotation Measures and Ground Trouth.\n File:  " + filename1)
    
    if((name!="") & (save_fig==True)):
        name1 = name +"2"
        save_figure(name1)
    
    
    
    fig = plot_annotation_line_warping(ann1,ann2_warp,file_audio2_warped,
                                       anchor_points_distance, offset_x = offset1, figsize=figsize)
    
    #if(audio_scenario==False):
    #    plt.title("Symbolic File with original and warped annotation measures.\n File:  " + filename2)
    #    
    #else:
    #    plt.title("Audio File 2 after TSM with Original and Warped Measures.\n WP = " + name_feat +  "\n File:  " + filename2)
    

    if((name!="") & (save_fig==True)):
        name2 = name +"_tsm2"
        save_figure(name2)
    plt.show()
    

    
def plot_cut_feature(feature, start, end, title, ann, offset=0, color_meas="", isMatrix=True, name="", save_fig=False):
    
    """
    is_Matrix==True if we are plotting Chroma and Dlnco
    is_Matrix==False if we are plotting Spectral Flux
    """
    font_off = 0
    
    if(isMatrix==True):
        plt.figure(figsize=(20, 4))
        librosa.display.specshow(feature, x_axis='time', 
                            y_axis='chroma', sr=Fs, cmap='gray_r', hop_length=H)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        
        chroma_names = ["C ", "C♯", "D ", "D♯", "E ", "F ", "F♯", "G ", "G♯", "A ", "A♯", "B " ]
        #chroma_yticks=np.arange(12)
        chroma_yticks=np.arange(0.5,12, step=1)
        plt.yticks(ticks = chroma_yticks ,labels = chroma_names, rotation = 'horizontal' , fontsize=12.5, fontfamily=font)
        plt.ylabel("Pitch class", fontsize=15, fontfamily=font)
    
    else:
        fig, ax, line = LibFMP.B.plot_signal(feature, Fs_frame, color='k',
            title="", figsize=(18,4))
        font_off = 5.5
        y_labels = np.arange(0, 0.4+0.001, 0.2)
        
        plt.yticks(y_labels, fontsize=12.5 + font_off, fontfamily=font)  
        plt.ylabel("Magnitude", fontsize=15+ font_off, fontfamily=font)
        
        ##just to plot
        plt.ylim([0,0.4+0.001])
    
    seconds=np.arange(start,end,1) 
    plt.xticks(seconds, fontsize=12.5 + font_off, fontfamily=font)
    plt.xlim([start, end+ 0.01]) 
    plt.xlabel('Time (seconds)', fontsize=15+ font_off, fontfamily=font)
    
    
    
    #plt.title(title + " CUT representation  (" + str(truncate(start,3)) + " , " + str(truncate(end,3)) + ") seconds" )
    
    plot_measures_on_feature(ann, offset, color_meas, font_off=font_off)
    
    if((name!="") & (save_fig==True)):
        name = name + "measures"
        save_figure(name)
    plt.show()


def plot_results(x_list, y_list1,y_list2,y_list3, y_label="", title="", x_label="", ms_flag_x=False, ms_flag_y=False, legend_label=""):
    
    fig = plt.figure(figsize=(15,5))

    #if(ms_flag_x==True):
    #    x_list = [element * 1000 for element in x_list]
        
        
    if(ms_flag_y==True):
        #y_list1 = [element * 1000 for element in y_list1]
        #y_list2 = [element * 1000 for element in y_list2]
        #y_list3 = [element * 1000 for element in y_list3]
        plt.ylabel("Time (milliseconds)", fontfamily=font)
    
    if(ms_flag_y==False):
        plt.ylabel("Time (seconds)", fontfamily=font)
                  
    if(y_label=="%"):
        plt.ylabel("Percentage (%)", fontfamily=font)
        
    if(x_label=="n"):
        plt.xlabel("Distance Anchor Points [n]", fontfamily=font)
        
    if(x_label=="tresh"):
        plt.xlabel("Tresholdstresh (milliseconds)", fontfamily=font)
                  

    min_y = min (y_list1 + y_list2 + y_list3) - 5
    max_y = max (y_list1 + y_list2 + y_list3) + 5
    
    plt.ylim([min_y,max_y])
    
    #plt.yticks(ticks = y_list)
    
    plt.xticks(ticks = x_list)
    plt.title(title)

    plt.grid()

    plt.plot(x_list, y_list1, color="r")
    plt.plot(x_list, y_list2, color="b")
    plt.plot(x_list, y_list3, color="g")
    leg = plt.legend((legend_label[0], legend_label[1], legend_label[2]), loc="upper right", fontsize = "large")
    leg.get_frame().set_edgecolor('k')
    
    plt.show()


def plot_results_audio(x_list, y_list1,y_list2,y_list3, y_list4, y_list5, y_list6, y_list7,
                       y_list8, y_list9, y_list10, y_list11,
                       y_label="", title="", x_label="",
                        ms_flag_y=False, legend_label="", name_save=""):
    

    figsize=(15,12)
    fig = plt.figure(figsize=figsize)

        
    if(ms_flag_y==True):
        plt.ylabel("Time (milliseconds)", fontsize=15, fontfamily=font)
    
    if(ms_flag_y==False):
        plt.ylabel("Time (seconds)", fontsize=15, fontfamily=font)
                  
    if(y_label=="%"):
        plt.ylabel("Percentage (%)", fontsize=15, fontfamily=font)
        

        
    if(x_label=="tresh"):
        plt.xlabel("Tresholds (milliseconds)", fontsize=15, fontfamily=font)
        y_ticks= np.arange(0, 100+0.1, step=10, dtype=int)
        plt.yticks(ticks = y_ticks, labels=y_ticks, fontsize=12.5, fontfamily=font)
        #x_labels = x_list
        #x_list = np.arange(len(x_list))
        #plt.xticks(ticks = x_list, labels=x_labels)
        plt.xscale("log")
        plt.xticks(ticks = x_list, labels=x_list, fontsize=12.5, fontfamily=font)
        plt.ylim([0,100])
                  
    if(x_label=="n"):
        plt.xlabel("Distance Anchor Points (measures)", fontsize=15, fontfamily=font)
        #min_y = 10
        max_y = max (y_list1 + y_list2 + y_list3 +y_list4 + y_list5 + y_list6 + y_list7 )  + 50
        #min_y = 0
        #max_y = 200
        
        
        #y_ticks= np.arange(min_y, max_y, step=50, dtype=int)
        #y_tick_labels= np.arange(0, max_y, step=50, dtype=int)
        plt.yscale('log')
        y_ticks = np.array([10,20,30,40,50,75,100, 125,150,175,200,250, 500,1000,1500])  #np.ceil(max_y)
        y_tick_labels = np.array([10,20,30,40,50,75,100,125,150,175,200,250, 500, 1000,1500])
        plt.yticks(ticks = y_ticks, labels=y_tick_labels, fontsize=12.5)
        plt.xticks(ticks = x_list, fontsize=12.5)
        
        plt.ylim([30,300])
    
    #plt.title(title, fontsize=17.5)

    plt.grid(linestyle="--", linewidth=0.5)
    
    plt.plot(x_list, y_list1, color="r", linewidth=3)
    plt.plot(x_list, y_list2, color="b", linewidth=3)
    plt.plot(x_list, y_list3, color="g", linewidth=3)
    plt.plot(x_list, y_list4, color="c", linewidth=3)
    plt.plot(x_list, y_list5, color="m", linewidth=3)
    plt.plot(x_list, y_list6, color="y", linewidth=3)
    plt.plot(x_list, y_list7, color="k", linewidth=3)
    
    plt.plot(x_list, y_list8, color="c", linewidth=3, linestyle='--')
    plt.plot(x_list, y_list9, color="m", linewidth=3, linestyle='--')
    plt.plot(x_list, y_list10, color="y", linewidth=3, linestyle='--')
    plt.plot(x_list, y_list11, color="k", linewidth=3, linestyle='--')
    
    
    plt.savefig('foto.png')    #HANDLE
    
    leg = plt.legend((legend_label[0], legend_label[1], legend_label[2], legend_label[3], legend_label[4], 
                legend_label[5], legend_label[6], legend_label[7], legend_label[8], legend_label[9], legend_label[10]),
                     loc="upper left", fontsize = "large")
    leg.get_frame().set_edgecolor('k')
    
    #if(name_save!=""):
    #    save_figure(name=name)
    
    plt.show()
    
    del fig
    

def plot_annotation_line(annot, file_audio, anchor_points_distance, offset_x=0, ax=None, label_keys={}, colors='FMP_1', figsize=(6, 1), direction='horizontal',
                         time_min=None, time_max=None, time_axis=True, nontime_axis=False, swap_time_ticks=False,
                         axis_off=False, dpi=72, filename="", col_no_anchor="limegreen", full_flag=False):
    """Creates a line plot for annotation data
    
    full_flag = to handle to plot of a full file, and to plot only the anchor points
    """
    
    assert direction in ['vertical', 'horizontal']
    annot = check_line_annotations(annot)
    
        
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    sec = np.arange(0, len(file_audio)) /Fs
    
    plt.plot(sec, file_audio) 
   
    
    
    labels_set = sorted(set([label for pos, label in annot]))
    
    colors_list=[]
    col_anchor="red"
    col_no_anchor = col_no_anchor
    

    for i in range(len(annot)):
        if ((i%anchor_points_distance == 0) or (i==len(annot)-1)):
            col = col_anchor
            
        else:
            col = col_no_anchor
            
        colors_list.append(col)    
    
    colors = color_argument_to_dict(colors_list, labels_set)

    for key, value in colors.items():
        
        label_keys[key] = {}
        label_keys[key]['color'] = value
        
    pos_measures = []
    pos_measures_labels = []
    i=0
    for pos, label in annot:

        if ((i%anchor_points_distance == 0) or (i==len(annot)-1)):
            linestyle="-"
            linewidth=8
            alpha=1
            if((full_flag==True) & (i<len(annot)-1)):    ##to not print the last indication
                linewidth=4
                pos_measures.append(pos)
                pos_measures_labels.append(truncate(pos + offset_x,1))
        
                
        else:
            
            linestyle="--"
            alpha=0.75
            if(full_flag==False):
                linewidth=3.5
            else:
                linewidth=0
            
          
        if direction == 'horizontal':
            ax.axvline(pos,**label_keys[label],  linewidth=linewidth, linestyle=linestyle, alpha=alpha)     #**label_keys[label],
        else:
            ax.axhline(pos,  linewidth=linewidth, linestyle=linestyle)

        if(full_flag==False):
            pos_measures.append(pos)
            pos_measures_labels.append(truncate(pos + offset_x,2))
        i+=1
        
        if(i==2):
            print(label_keys[label])
    
    if time_min is None:
        time_min = min(pos for pos, label in annot)
    if time_max is None:
        time_max = max(pos for pos, label in annot)

    if direction == 'horizontal':
        ax.set_xlim([0, time_max+0.01])
        if not time_axis:
            ax.set_xticks([])
        if not nontime_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()

    magnitude = np.array([truncate(min(file_audio),2), 0, truncate(max(file_audio),2)])

    if(full_flag==False):
        plt.xticks(pos_measures, labels=pos_measures_labels, fontsize=12.5, fontfamily=font)
    
    else:
        pos_measures = pos_measures[0:-1:2]
        pos_measures_labels = pos_measures[0:-1:2]
    

    plt.yticks(magnitude, fontsize=12.5, fontfamily=font)
    plt.xlabel("Time (seconds)", fontsize=15, fontfamily=font)
    plt.ylabel("Magnitude", fontsize=15, fontfamily=font)
    ##plt.title("Time position of the Measures  |  File: " + filename, fontsize=17.5, fontfamily=font)
    


    patch1 = mlines.Line2D([], [], color=None, linestyle='-',markersize=15, label='Audio Wave', linewidth=2.5)
    patch2 = mlines.Line2D([], [], color=col_anchor, linestyle='-',markersize=20, label='Anchor Points', linewidth=4)
    
    if(full_flag==False):
        patch3 = mlines.Line2D([], [], color=col_no_anchor, linestyle='--',markersize=15, label='Ground Trouth Meas.', linewidth=2.5)
        leg = plt.legend(handles=[patch1, patch2, patch3], loc = "upper right", fontsize = "large")
    else:
        leg = plt.legend(handles=[patch1, patch2], loc = "upper right", fontsize = "large")
        
    leg.get_frame().set_edgecolor('k')
  
    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_annotation_line_comparison(ann1, ann2, file_audio, anchor_points_distance, offset_x=0, ax=None, label_keys={}, colors='FMP_1', figsize=(6, 1), direction='horizontal',
                         time_min=None, time_max=None, time_axis=True, nontime_axis=False, swap_time_ticks=False,
                         axis_off=False, dpi=72, filename="", off_warp=0):
    """Creates a line plot for annotation data
    
    ann1 = annotation of the audio 2 file: ground trouth annotation
    ann2 = warped measures of audio 2, after syncrhonization

    """
    label_keys_wp={}
    
    assert direction in ['vertical', 'horizontal']
    ann1 = check_line_annotations(ann1)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    #sec = np.arange(0, len(file_audio)) /Fs
    #plt.plot(sec, file_audio)    

    labels_set = sorted(set([label for pos, label in ann1]))

    
    colors_list=[]
    colors_list_wp = []
    col_anchor="red"
    col_orig = "darkturquoise"
    col_wp= "darkviolet"
    
    
    for i in range(len(ann1)):
        if ((i%anchor_points_distance == 0) or (i==len(ann1)-1)):
            col = col_anchor
        else:
            col = col_orig
            
        colors_list.append(col)    
        colors_list_wp.append(col_wp)

    
    colors = color_argument_to_dict(colors_list, labels_set)
    colors_wp = color_argument_to_dict(colors_list_wp, labels_set)
    
    
    for key, value in colors.items():
        label_keys[key] = {}
        label_keys[key]['color'] = value
        
    for key, value in colors_wp.items():
        label_keys_wp[0] = {}
        label_keys_wp[0]['color'] = value

        
    pos_measures = []
    pos_measures_labels = []
    
    i=0
    for pos, label in ann1:

        
        if ((i%anchor_points_distance == 0) or (i==len(ann1)-1)):
            linestyle="-"
            linewidth=8
            alpha = 1
            
            pos_measures.append(pos)
            pos_measures_labels.append(truncate(pos + offset_x,2))
        else:
            linestyle="--"
            linewidth=3
            alpha=0.8

        
        if direction == 'horizontal':
            ax.axvline(pos,**label_keys[label],  linewidth=linewidth, linestyle=linestyle, alpha=alpha)     #**label_keys[label],
        else:
            ax.axhline(pos,  linewidth=linewidth, linestyle=linestyle)
        i+=1
        


    i=0
    for pos, label in ann2:
        
        
        if ((i>0) & (i<len(ann2)-1)):
            pos_measures.append(pos + (off_warp - offset_x))
            pos_measures_labels.append(truncate(pos + off_warp,2))
            pos = pos + (off_warp - offset_x)
            

            ax.axvline(pos,**label_keys_wp[0],  linewidth=3.3, linestyle="dashdot", alpha=0.9)     #**label_keys[label],
            
        i+=1
        
        


    if time_min is None:
        time_min = min(pos for pos, label in ann1)
    if time_max is None:
        time_max = max(pos for pos, label in ann1)
    
    #ax.set_xlim([0, time_max+0.01])    
    #if direction == 'horizontal':
    #    ax.set_xlim([0, time_max+0.01])
    #    if not time_axis:
    #        ax.set_xticks([])
    #    if not nontime_axis:
    #        ax.set_yticks([])
    #    if swap_time_ticks:
    #        ax.xaxis.tick_top()

        
    #magnitude = np.array([truncate(min(file_audio),2), 0, truncate(max(file_audio),2)])
        
    plt.xticks(pos_measures, labels=pos_measures_labels, fontsize=12.5, fontfamily=font)
    #plt.yticks(magnitude, fontsize=12.5, fontfamily=font)
    plt.xlabel("Time (seconds)", fontsize=15, fontfamily=font)
    #plt.ylabel("Magnitude", fontsize=15, fontfamily=font)
    #plt.title("Time position of the Measures  |  File: " + filename, , fontsize=17.5, fontfamily=font)
    
    
    #patch1 = mpatches.Patch(color=None, linewidth=0.0005, linestyle="-", label='Audio Wave')
    
    patch2 = mlines.Line2D([], [], color=col_anchor, linestyle='-',markersize=20, label='Anchor Points', linewidth=4)
    patch3 = mlines.Line2D([], [], color=col_orig, linestyle='--',markersize=15, label='Ground Trouth Meas.',linewidth=2.5)    
    patch4 = mlines.Line2D([], [], color=col_wp, linestyle='-.',markersize=15, label='Warped Meas.',linewidth=2.5)    
    
    leg = plt.legend(handles=[patch2, patch3, patch4], loc = "upper right" ,fontsize = "large")
    leg.get_frame().set_edgecolor('k')
  
    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()
        
    plt.xlim([0,time_max+0.01])

    return fig, ax


def plot_annotation_line_warping(ann1, ann2, file_audio, anchor_points_distance, offset_x=0, ax=None, label_keys={}, colors='FMP_1', figsize=(6, 1), direction='horizontal',
                         time_min=None, time_max=None, time_axis=True, nontime_axis=False, swap_time_ticks=False,
                         axis_off=False, dpi=72, filename=""):
    """Creates a line plot for annotation data
    
    ann1 = annotation of the ground trouth
    ann2= annotation of the warped measures

    """
    label_keys_gt={}
    
    assert direction in ['vertical', 'horizontal']
    ann2 = check_line_annotations(ann2)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    sec = np.arange(0, len(file_audio)) /Fs
    plt.plot(sec, file_audio)    

    labels_set = sorted(set([label for pos, label in ann2]))

    
    colors_list=[]
    colors_list_gt = []
    col_anchor="red"
    col_no_anchor = "darkviolet"
    col_gt = "limegreen"

    
    for i in range(len(ann2)):
        if ((i%anchor_points_distance == 0) or (i==len(ann1)-1)):
            col = col_anchor
        else:
            col = col_no_anchor
            
        colors_list.append(col)    
        colors_list_gt.append(col_gt)

    
    colors = color_argument_to_dict(colors_list, labels_set)
    colors_gt = color_argument_to_dict(colors_list_gt, labels_set)
    
    
    for key, value in colors.items():
        label_keys[key] = {}
        label_keys[key]['color'] = value
        
    for key, value in colors_gt.items():
        label_keys_gt[0] = {}
        label_keys_gt[0]['color'] = value

        
    pos_measures = []
    pos_measures_labels = []
    
    
    


    j=0
    for pos, label in ann1:
        
        if ((j%anchor_points_distance == 0) or (j==len(ann1)-1)):
            lab_col = label_keys[0]
            linewidth=8
            linestyle = "-"
            alpha=1
        else:
            lab_col = label_keys_gt[0]
            linewidth=3
            linestyle = "--"
            alpha=0.8
            
        ax.axvline(pos,**lab_col,  linewidth=linewidth, linestyle=linestyle, alpha=alpha)     #**label_keys[label],


        j+=1
        
    i=0  
    for pos, label in ann2:

        if ((i>0) or (i<len(ann2)-1)):
            linestyle="dashdot"
            linewidth=3.3

        pos_measures.append(pos)
        pos_measures_labels.append(truncate(pos + offset_x,2))
        ax.axvline(pos,**label_keys[label],  linewidth=linewidth, linestyle=linestyle, alpha=0.9)     #**label_keys[label],
        i+=1


    if time_min is None:
        time_min = min(pos for pos, label in ann2)
    if time_max is None:
        time_max = max(pos for pos, label in ann2)
        
    if direction == 'horizontal':
        ax.set_xlim([0, time_max+0.01])
        if not time_axis:
            ax.set_xticks([])
        if not nontime_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()

        
    magnitude = np.array([truncate(min(file_audio),2), 0, truncate(max(file_audio),2)])
        
    plt.xticks(pos_measures, labels=pos_measures_labels, fontsize=12.5, fontfamily=font)
    plt.yticks(magnitude, fontsize=12.5, fontfamily=font)
    plt.xlabel("Time (seconds)", fontsize=15, fontfamily=font)
    plt.ylabel("Magnitude", fontsize=15, fontfamily=font)
    #plt.title("Time position of the Measures  |  File: " + filename, , fontsize=17.5, fontfamily=font)
    

    label_gt="Source Meas."
    label_wav="TSM Audio Wave"
    label_anchor = "Anchor Points"
    
    
    patch1 = mlines.Line2D([], [], color=None, linestyle='-',markersize=15, label=label_wav,linewidth=2.5)
    patch2 = mlines.Line2D([], [], color=col_anchor, linestyle='-',markersize=20, label=label_anchor, linewidth=4)
    patch3 = mlines.Line2D([], [], color=col_gt, linestyle='--',markersize=15, label=label_gt,linewidth=2.5)
    patch4 = mlines.Line2D([], [], color=col_no_anchor, linestyle='-.',markersize=15, label='Warped Meas.',linewidth=2.5)
    
    leg = plt.legend(handles=[patch1, patch2, patch3, patch4], loc = "upper right" ,fontsize = "large")
    #bbox_to_anchor=(1.2, 1)
    leg.get_frame().set_edgecolor('k')
  
    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax

def plot_measures_on_feature(ann, offset=0, color_meas="", font_off=0):
    
    pos_measures=[]
    pos_measures_labels=[]

    if(color_meas==""):
        color_meas="red"
    
    i=0
    for pos, val in ann:
        if ((i== 0) or (i==len(ann)-1)):
            linestyle="-"
            linewidth=8
            alpha=1
            act_col = "red"
            
        else:
            linestyle="dashed"
            linewidth=3
            alpha=0.9
            act_col=color_meas
            
        pos = pos+offset
        plt.axvline(x=pos, ymin=0, ymax=1, color=act_col, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        pos_measures.append(pos)
        pos_measures_labels.append(truncate(pos,2))
        i+=1
    
    plt.xticks(pos_measures, labels=pos_measures_labels, fontsize=12.5+font_off, fontfamily=font)
    
    plt.xlabel('Time (seconds)', fontsize=15+font_off, fontfamily=font)
    
    
    patch1 = mlines.Line2D([], [], color="red", linestyle='-',markersize=20, label='Anchor Points', linewidth=4)
    patch2 = mlines.Line2D([], [], color=color_meas, linestyle='--',markersize=15, label='Measures',linewidth=2.5)
    
    
    if(font_off>0):
        fontsize="xx-large"
    else:
        fontsize="large"
        
    leg = plt.legend(handles=[patch1, patch2], loc = "upper right", fontsize = fontsize)
    leg.get_frame().set_edgecolor('k')


def plot_measures_on_costmatrix(ann_act1, ann_act2, offset1, offset2, color_meas1, color_meas2, fn1, fn2, symb_flag="False"):
    
    pos_measures=[]
    pos_measures_labels=[]
    
    for pos,val in ann_act1:
        pos = pos + offset1
        plt.axhline(y=pos, xmin=0, xmax=1, color=color_meas1, linestyle="dashed", linewidth=3.5, alpha=0.9)
        pos_measures.append(pos)
        pos_measures_labels.append(truncate(pos,2))
    plt.yticks(pos_measures, labels=pos_measures_labels, fontsize=12.5, fontfamily=font)
    
    pos_measures=[]
    pos_measures_labels=[]
    
    for pos, val in ann_act2:
        pos = pos+offset2
        plt.axvline(x=pos, ymin=0, ymax=1, color=color_meas2, linestyle="dashed", linewidth=3.5, alpha=0.9)
        pos_measures.append(pos)
        pos_measures_labels.append(truncate(pos,2))
        
    plt.xticks(pos_measures, labels=pos_measures_labels,fontsize=12.5, fontfamily=font)
    
    if(symb_flag==True):
        label_aligned_file = "Meas. Symbolic " + fn2
    else:
        label_aligned_file = "Meas. Audio: " + fn2
    
    
    patch1 = mlines.Line2D([], [], color="red", linestyle='-',markersize=15, label='Warping Path', linewidth=5)
    patch2 = mlines.Line2D([], [], color=color_meas1, linestyle='--',markersize=15, label='Meas. Audio: ' + fn1,linewidth=2.5)
    patch3 = mlines.Line2D([], [], color=color_meas2, linestyle='--',markersize=15, label=label_aligned_file,linewidth=2.5)
    
    leg = plt.legend(handles=[patch1, patch2, patch3], loc = "lower right", fontsize = "large")
    leg.get_frame().set_edgecolor('k')