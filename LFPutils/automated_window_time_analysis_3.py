"""   

    This script analyzes the recording sessions as broken down into time windows and creates peak amplitude graphs for each electrode window by window. For a given list it also generates normalized amplitude graphs and average average peak amplitudes using a error correction algorithm.

    Required Files:
        Evoked data, parameters dictionary.

    Input:
        main_path : Main path for the data.
        tw : Time window length.
        groupt_to_plot : The group on which the analysis will be performed on.
        trodes_to_plot : A list of electrodes defined by user. Normalized peak amplitudes and average peak amplitudes will be calculated according to this list.
        y_scale : Scale factor of the y-scale of the graph.
        normalization_t1, normalization_t2 : The start and the end points of the base interval. Corrected peak amplitudes will be calclated using this interval's mean.

    Output:
        Windowed evoked LFP graphs for a given electrode list
        Peak amplitude graphs for all electodes
        Normalized amplitude graph
        Average peak amplitude graph
        Numpy "npy" file that contains peak amplitudes values and peak amplitude errors containing all electordes
        Pickle file for window LFP values of all electordes
        Numpy "npy" file that contains peak amplitudes values and peak amplitude errors of only given electorde list
        Pickle file for window LFP values of only given electorde list

    Reviewed on June  2018
    Author: Mehmet Ozdas and Abdulkadir Gokce - Please contact them in case of any questions.

    
"""
import numpy as np
import numpy
import pickle
import os
import sys
import ipywidgets
import math
import shutil
from matplotlib.pyplot import *
from matplotlib.ticker import MultipleLocator
import pandas as pd

class Error:
    """
        The purpose of this class is to combine standard errors of different samples
        Usage: Initialize an object with the evoked_window and evoked_window_error by passing only the electrodes of interest
               Then call the combining_errors() function to obtain combined error via return value of the method
        combining_errors() function executes the algorithm for adjusted version of combination standard errors of two data set in NATURE June 8, 1963, Vol. 198, Page 1020
        For more info, contact with Mehmet Ozdas or Abdulkadir Gokce
    """
    def __init__(self, evoked_window, evoked_window_err):
        self.combined_error = np.zeros(len(evoked_window_err[1]))
        self.number_of_samples = 0
        self.mean = np.zeros(len(evoked_window[0][0]))
        self.evoked_window = evoked_window
        self.evoked_window_err = evoked_window_err

    def combining_errors(self):
        for trode in range(len(self.evoked_window[0])):
            x = self.evoked_window[:, trode, :]
            n1 = self.number_of_samples
            n2 = len(x)
            n = n1 + n2
            m1 = self.mean
            m2 = np.mean(self.evoked_window[:, trode, :], axis=0)
            e1 = self.combined_error
            e2 = self.evoked_window_err[trode, :]
            e3_square = (n1*(n1-1)*np.power(e1,2) + n2*(n2-1)*np.power(e2,2) + n1*n2*np.power(m1-m2,2)/n) / (n*(n-1))
            e3 = np.sqrt(e3_square)
            self.combined_error = e3
            self.number_of_samples = n
            self.mean = np.mean(np.mean(self.evoked_window[[j for j in range(trode+1)], :], axis=1), axis=0)
        return self.combined_error

def automated_window_lfp(main_path, tw, group_to_plot, trodes_to_plot, y_scale, normalization_t1, normalization_t2,FUS_on_time,rolling_window):

    """
    #Test values:    
    main_path = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_06_05_FUSs1_EphysM1_E-FUS_NBBB88/' 
    tw = 1 
    group_to_plot =1 
    trodes_to_plot = [0,3,9] 
    y_scale = 50 
    normalization_t1 = 5 
    normalization_t2 = 15
    """

    print(main_path)
    experiment_id = main_path.rpartition('_')[2].replace('/','')
    group = group_to_plot
    probe = 0
    stim_timestamps = np.zeros((0)) #Concatenated stimulation timestamps array of all iterated folders
    duration = 0 #Total duration of the iterated stimulation timestamps
    evoked_data = np.zeros((0))
        #########################
    rev_experiment_id=''.join(reversed(main_path))
    experiment_id=rev_experiment_id.split('_')[0] 
    experiment_id=''.join(reversed(experiment_id))
    experiment_id = experiment_id.replace("/", "")
    #########################
    dirs=os.listdir(main_path)
    dirs.sort()

    for i in range(1):
        for folder in (folder for folder in dirs if ((folder != 'log.txt') and (folder != 'notes.docx') and (folder != 'analysis_files') and (folder != 'other') and (folder != 'analyzed') and (folder != '.DS_Store')and (folder != '._.DS_Store') and (folder != 'analyzed'))):
            print(folder)
            p = pickle.load(open(main_path + folder + '/paramsDict.p', 'rb')) #Loading parameter dictionary
            len_time_window = tw * 60 * p['sample_rate']  # 30000 samples per second
            analyzed_path_for_folder = main_path + 'analyzed/' + experiment_id
            evoked_LFP_timerange = np.arange(-p['evoked_pre'],p['evoked_post'],1/p['sample_rate'])*1000  # 25
            peaks = []
            errs = []
            #lfp_averages=[]
            #remove old npy files if thet exist
            """
            Concatenate the several folders
            """
            
            filePath = main_path + folder + '/time.dat'
            evoked = pickle.load( open(main_path + folder + '/probe_{0}_group_{1}/probe_{0}_group_{1}_evoked.pickle'.format(probe, group), 'rb') )     
            evoked_waveforms = evoked['evoked']        
            stim_timestamps_folder = evoked['stim_timestamps']
            stim_timestamps_folder = stim_timestamps_folder + duration #Add previous total duration to stimulation timestamps of the folder to take account the presendence of the stimulations
            if len(stim_timestamps)!=0: #If stim_timestamps is not contatenated yet, set it equal to current folder's timestamp
                stim_timestamps = np.concatenate((stim_timestamps, stim_timestamps_folder))
                evoked_data=np.concatenate((evoked_data,evoked_waveforms))
            else:
                stim_timestamps = stim_timestamps_folder
                evoked_data = evoked_waveforms
            duration += len( np.fromfile(open(filePath, 'rb'), np.int32))#Increase the total duration of the experiment by the time of current folder        


        for i in range(len([1])):
            
            directory=analyzed_path_for_folder
            print(directory)
            files = os.listdir( directory )
            for item in files:
                if item.endswith(".npy"):
                    os.remove( os.path.join( directory, item ) )
            #4D array, 1st=probe number, 2nd=group number, 3rd=electrode_no, num_window=peak numbers in graph

            """
            This part removes the existing paths (if exist) and make directories for storing images&pdfs
            """  
            
            analyzed_path_for_group = analyzed_path_for_folder + '/probe_{:g}_group_{:g}/'.format(probe,group)
            
            tw_pdf_format = analyzed_path_for_group + 'tw_pdf_format/' # for saving images in pdf format
            if os.path.exists(tw_pdf_format): # if file exist, delete remove old files
                shutil.rmtree(tw_pdf_format)

            tw_svg_format = analyzed_path_for_group + 'tw_svg_format/'
            if os.path.exists(tw_svg_format):   # if file exist, delete remove old files
                shutil.rmtree(tw_svg_format)

            for trode in trodes_to_plot:
                
                tw_pdf_format = analyzed_path_for_group + 'tw_pdf_format/' # for saving images in pdf format
                if not os.path.exists(tw_pdf_format): # if not, create folder
                    os.mkdir(tw_pdf_format)
                tw_pdf_format = tw_pdf_format + 'electrode_{:g}/'.format(trode)
                if not os.path.exists(tw_pdf_format): # if not, create folder
                    os.mkdir(tw_pdf_format)
        
                tw_svg_format = analyzed_path_for_group + 'tw_svg_format/'
                if not os.path.exists(tw_svg_format): # create  a new fiel and store images here
                    os.mkdir(tw_svg_format)
                tw_svg_format = tw_svg_format + 'electrode_{:g}/'.format(trode)
                if not os.path.exists(tw_svg_format): # create  a new fiel and store images here
                    os.mkdir(tw_svg_format)
            
            if(len(trodes_to_plot)>1):
                tw_pdf_format = analyzed_path_for_group + 'tw_pdf_format/' # for saving images in pdf format
                if not os.path.exists(tw_pdf_format): # if not, create folder
                    os.mkdir(tw_pdf_format)
                tw_pdf_format = tw_pdf_format + 'list/'
                if not os.path.exists(tw_pdf_format): # if not, create folder
                    os.mkdir(tw_pdf_format)
        
                tw_svg_format = analyzed_path_for_group + 'tw_svg_format/'
                if not os.path.exists(tw_svg_format): # create  a new fiel and store images here
                    os.mkdir(tw_svg_format)
                tw_svg_format = tw_svg_format + 'list/'
                if not os.path.exists(tw_svg_format): # create  a new fiel and store images here
                    os.mkdir(tw_svg_format)
     
            """
            Create variables and empty Numpy arrays for later use
            """
            #data_location = main_path + '/' + folder + ('/probe_{:g}_group_{:g}'.format(probe,group)) + ('/probe_{:g}_group_{:g}_evoked.pickle'.format(probe,group))
            ##evoked_data = pickle.load(open(data_location, 'rb'))
            evoked = evoked_data#['evoked']
            #stim_timestamps = evoked_data['stim_timestamps']
            num_window = int(np.max(stim_timestamps) / len_time_window)
            windows = np.arange(0,num_window,1)
            windows=windows*tw  # multiply by time window for the graph
            evoked_window_avgs = np.zeros((num_window, len(evoked[0]), len(evoked[0][0])))
            evoked_window_err = np.zeros((num_window, len(evoked[0]), len(evoked[0][0])))
            evoked_window_amps = np.zeros((p['probes'], p['shanks'], p['nr_of_electrodes_per_group'], num_window))
            evoked_window_peak_errs = np.zeros((p['probes'], p['shanks'], p['nr_of_electrodes_per_group'], num_window))
            lfp_averages = numpy.zeros((p['probes'],p['shanks'],p['nr_of_electrodes_per_group'],num_window))
            

            #Most of the variable names that is used for denoting the values of the electrodes of interest start with 'list_' prefix
            evoked_window_correction = numpy.zeros((num_window, len(evoked[0])))
            evoked_window_amps_corrected = np.zeros((p['probes'], p['shanks'], p['nr_of_electrodes_per_group'], num_window))
            list_evoked_window = np.zeros((num_window, len(trodes_to_plot), len(evoked[0][0])))
            list_evoked_window_avg = np.zeros((num_window, len(evoked[0][0])))
            list_evoked_window_amps_normalized = np.zeros((p['probes'], p['shanks'], num_window))
            list_peaks_errs_combined = np.zeros((p['probes'], p['shanks'], num_window))
            list_peaks_errs_normalized = np.zeros((p['probes'], p['shanks'], num_window))
            list_window_errs_combined = np.zeros((num_window,len(evoked[0][0])))
            normalization_interval_avgs = np.zeros((int((normalization_t2-normalization_t1)/tw) + math.ceil((normalization_t2-normalization_t1)%tw), len(trodes_to_plot), len(evoked[0][0])))
            list_lfp_averages = numpy.zeros((p['probes'], p['shanks'], num_window))

            
            if(len(trodes_to_plot)>1): 
                 #This part calculates the normalization coefficient wrt user's time interval input
                 #The time interval is splitted to time windows (tw) and  mean of each subinterval is calculated seperately
                 #By doing so, the peak amplitude of each time window in the normalization coefficient interval coincides with original peak amplitude values. 
                 #Otherwise, peak values may be attenued or distorted and normalized figure may not resemble a true graph. 
                 for time in range( int( (normalization_t2 - normalization_t1) / tw ) ):
                     #Mean of each splitted time window
                     normalization_interval_avgs[time] = np.mean(evoked[np.all([stim_timestamps>(normalization_t1+time*tw)*60*p['sample_rate'],stim_timestamps<(normalization_t1+(time+1)*tw)*60*p['sample_rate']], axis=0)][:, trodes_to_plot, :], axis=0)
                 
                 #If the interval is not divisible by tw, calculate the remaining part separately
                 if(math.ceil((normalization_t2-normalization_t1)%tw)==1):
                     time = time + 1
                     normalization_interval_avgs[len(normalization_interval_avgs)-1] = np.mean(evoked[np.all([stim_timestamps>(normalization_t1+time*tw)*60*p['sample_rate'],stim_timestamps<normalization_t2*60*p['sample_rate']], axis=0)][:, trodes_to_plot, :], axis=0)
        
                 #The mean of the minimum values of the normalization interval 
                 normalization_coefficient = np.mean(np.min(np.mean(normalization_interval_avgs, axis=1), axis=1), axis=0)
            
            
            """
            Calculate  Numpy arrays for time window LFPs and plot figures 
            """
            #Calculate average windowed evoked LFP
            for window in range(num_window):
                 if ( (FUS_on_time - 10)/tw<=window):   
                    print("\nTime: {:g}".format(window-(FUS_on_time - 10)/tw))
                 #Finding all the evoked data for which the time stamp falls in the window of interest
                 evoked_window = evoked[np.all([stim_timestamps > window * len_time_window, stim_timestamps < (window + 1) * len_time_window], axis = 0)]
                 evoked_window_avgs[window] = np.mean(evoked_window, axis=0) #  mean of evoked window
                 evoked_window_std = np.std(evoked_window, axis=0) #Standard deviation of the data in the time window
                 evoked_window_err[window] = evoked_window_std / math.sqrt(len(evoked_window)) # standart error of evoked window
        
                 #Select and calculate the mean of the electrodes of interest
                 list_evoked_window[window] = evoked_window_avgs[window, trodes_to_plot, :]
                 list_evoked_window_avg = np.mean(list_evoked_window, axis=1)

            #Calculate peak amplitudes of windowed evoked LFP
            for window in range(num_window):
                for trode in range(p['nr_of_electrodes_per_group']):
                    evoked_window_amps[probe][group][trode][window] = np.min(evoked_window_avgs[window][trode]) 
                    evoked_window_correction = np.mean(evoked_window_avgs[:, :, 0:int(p['evoked_pre']*p['sample_rate'])], axis=2) #Average values between 'evoked_pre' and 0 for correction
                    #The mean of the -(pre evoked):0 interval is substracted from the minimum amplitude of each respective time window to obtain amplitude difference
                    evoked_window_amps_corrected[probe][group][trode][window] = np.min(evoked_window_avgs[window][trode]) - evoked_window_correction[window][trode] #Correction for the difference between peak and initial state
                    min_error = evoked_window_err[window][probe][np.where(evoked_window_avgs[window][trode] == np.min(evoked_window_avgs[window][trode]))]
                    if len(min_error) == 1:
                       evoked_window_peak_errs[probe][group][trode][window] = min_error


            #
               
            for window in range(num_window):  
                 #Plot the average amplitute graph of the user's list of electrodes
                 if(len(trodes_to_plot)>1 and (FUS_on_time - 10)/tw<=window ): 
                     #Normalized peak amplitudes for user's list wrt normalization coefficient
                     list_evoked_window_amps_normalized[probe][group][window] = (np.min(list_evoked_window_avg, axis=1)[window]/ normalization_coefficient)*100
                     #Combined error for multiple sample groups. See Error class above for further information.
                     list_window_errs_combined[window] = Error(evoked_window[:, trodes_to_plot, :], evoked_window_err[window, trodes_to_plot, :]).combining_errors()
                     figure() 
                     title('Average Amplitute Graph / List of Electrodes')
                     plot(evoked_LFP_timerange, list_evoked_window_avg[window],'k-')
                     xlabel('Time (ms)')
                     ylabel('Peak voltage (uV)')
                     ylim_min = np.floor(np.min(evoked) / 100) * y_scale 
                     ylim_max = np.ceil(np.max(evoked) / 100) * y_scale
                     ylim(ylim_min, ylim_max)
                     xlim_min = -p['evoked_pre']*1000
                     xlim_max = p['evoked_post']*1000
                     xlim(xlim_min, xlim_max)
                     fill_between(evoked_LFP_timerange, list_evoked_window_avg[window]-list_window_errs_combined[window], list_evoked_window_avg[window]+list_window_errs_combined[window])
                     print('Plotting window figure and saving / List')
                     savefig(analyzed_path_for_group + 'tw_pdf_format/' + 'list/' + 'average_evoked_LFP_at_time_window_{:g}_for_list'.format(window-((FUS_on_time - 10)/tw))+'.pdf', format = 'pdf')
                     savefig(analyzed_path_for_group + 'tw_svg_format/' + 'list/' +  'average_evoked_LFP_at_time_window_{:g}_for_list'.format(window-((FUS_on_time - 10)/tw))+'.svg', format = 'svg')
                     close()
                    
                 #Plot the figure for the average evoked LFP in this time window

                 for trode in trodes_to_plot:
                    if ((FUS_on_time - 10)/tw<=window):
                         figure()
                         title('Average Amplitute Graph / Electrode-{0}, Window-{1}'.format(trode,int(window-((FUS_on_time - 10)/tw))))
                         plot(evoked_LFP_timerange, evoked_window_avgs[window][trode],'k-')
                         xlabel('Time (ms)')
                         ylabel('Peak voltage (uV)')
                         ylim_min = np.floor(np.min(evoked) / 100) * y_scale
                         ylim_max = np.ceil(np.max(evoked) / 100) * y_scale
                         ylim(ylim_min, ylim_max)
                         xlim_min = -p['evoked_pre']*1000
                         xlim_max = p['evoked_post']*1000
                         xlim(xlim_min, xlim_max)
                         fill_between(evoked_LFP_timerange, evoked_window_avgs[window][trode]-evoked_window_err[window][trode], evoked_window_avgs[window][trode]+evoked_window_err[window][trode])
                         print('Plotting window figure and saving / Electrode-{}'.format(trode))
                         savefig(analyzed_path_for_group + 'tw_pdf_format/' + 'electrode_{:g}/'.format(trode) +'average_evoked_LFP_at_time_window_{:g}_for_group_{:g}_electrode_{:g}'.format(window-((FUS_on_time - 10)/tw), group, trode)+'.pdf', format = 'pdf')
                         savefig(analyzed_path_for_group + 'tw_svg_format/' + 'electrode_{:g}/'.format(trode) +'average_evoked_LFP_at_time_window_{:g}_for_group_{:g}_electrode_{:g}'.format(window-((FUS_on_time - 10)/tw), group, trode)+'.svg', format = 'svg')
                         close()
                 
                
                 if(len(trodes_to_plot)>1 and (FUS_on_time - 10)/tw<=window):
                    #If only one data point has the  minimum amplitude, peak error is equal to evoked window error at that data point
                    min_error = list_window_errs_combined[window][np.where(list_evoked_window_avg[window] == np.min(list_evoked_window_avg[window]))]
                    if len(min_error) == 1:
                        list_peaks_errs_combined[probe][group][window] = min_error
                    list_peaks_errs_normalized[probe][group][window] = -(list_peaks_errs_combined[probe][group][window]/normalization_coefficient)*100 #Normalize the error

            if(len(trodes_to_plot)>1 ):

                figure() #Plot the Normalized Amplitute Graph of the List of Electrodes
                title('Normalized Amplitute Graph / List of Electrodes')
                plot(windows-(FUS_on_time-10) , list_evoked_window_amps_normalized[probe][group],'k-')
                xlabel('Time (min)')
                ylabel('-% wrt iniatial 10 minute')
                ylim_min = -1*y_scale 
                ylim_max = 6*y_scale
                ylim(ylim_min, ylim_max)
                xlim_min = 0
                xlim_max = (num_window-1)*tw - (FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%10), (ylim_max-ylim_max%10), 10)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                errorbar(windows-(FUS_on_time-10), list_evoked_window_amps_normalized[probe][group], yerr = list_peaks_errs_normalized[probe][group])
                print('\nPlotting the Normalized Amplitute Graph / List of Electrodes')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' +'normalized_for_selected_electrodes_time_window.pdf', format = 'pdf')
                savefig(analyzed_path_for_group + 'tw_svg_format/' +'normalized_for_selected_electrodes_time_window.svg', format = 'svg')
                close()

                figure() #Plot the Normalized Amplitute with rolling mean Graph of the List of Electrodes
                title('Normalized Amplitute Graph with rolling mean / List of Electrodes')
                plot(windows-(FUS_on_time-10) , pd.rolling_mean(list_evoked_window_amps_normalized[probe][group],window=rolling_window),'k-')
                xlabel('Time (min)')
                ylabel('-% wrt iniatial 10 minute')
                ylim_min = -1*y_scale 
                ylim_max = 6*y_scale
                ylim(ylim_min, ylim_max)
                xlim_min = 0
                xlim_max = (num_window-1)*tw - (FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%10), (ylim_max-ylim_max%10), 10)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                errorbar(windows-(FUS_on_time-10), pd.rolling_mean(list_evoked_window_amps_normalized[probe][group],window=rolling_window), yerr = list_peaks_errs_normalized[probe][group])
                print('Plotting the Normalized Amplitute Graph with rolling mean / List of Electrodes')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' +'normalized_for_selected_electrodes_time_window_with_rm.pdf', format = 'pdf')
                savefig(analyzed_path_for_group + 'tw_svg_format/' +'normalized_for_selected_electrodes_time_window_with_rm.svg', format = 'svg')
                close()

                figure()#Plot the Average Automated Time Window Analysis of the List of Electrodes
                title('Automated Time Window Analysis / List of Electrodes')
                plot(windows-(FUS_on_time-10) , np.mean(evoked_window_amps_corrected[:, :, trodes_to_plot], axis=2)[probe][group], 'k-')
                xlabel('Time (min)')
                ylabel('Peak voltage (uV)')
                ylim_min = np.floor(np.min(evoked) / 100) * y_scale 
                ylim_max = np.ceil(np.max(evoked) / 100) * y_scale
                ylim(ylim_min, ylim_max)
                xlim_min =0
                xlim_max = (num_window-1)*tw - (FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%100), (ylim_max-ylim_max%100), 100)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                errorbar(windows-((FUS_on_time-10)), np.mean(evoked_window_amps_corrected[:, :, trodes_to_plot], axis=2)[probe][group], yerr = list_peaks_errs_combined[probe][group], color='olive')
                print('Plotting the Automated Time Window Analysis / List of Electrodes')
                savefig(analyzed_path_for_group + 'tw_svg_format/' + 'time_windows_list.svg', format = 'svg')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' + 'time_windows_list.pdf', format = 'pdf')
                list_lfp_averages[probe][group] = np.mean(evoked_window_amps_corrected[:, :, trodes_to_plot], axis=2)[probe][group]
                close()
                
                figure()#Plot the Average Automated Time Window Analysis with rolling mean of the List of Electrodes
                title('Automated Time Window Analysis with rolling mean / List of Electrodes')
                plot(windows-(FUS_on_time-10), pd.rolling_mean(np.mean(evoked_window_amps_corrected[:, :, trodes_to_plot], axis=2)[probe][group],window = rolling_window), 'k-')
                xlabel('Time (min)')
                ylabel('Peak voltage (uV)')
                ylim_min = np.floor(np.min(evoked_window_amps_corrected)-50) 
                ylim_max = np.ceil(np.max(evoked_window_amps_corrected)+50) 
                ylim(ylim_min, ylim_max)
                xlim_min =0
                xlim_max = (num_window-1)*tw - (FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%100), (ylim_max-ylim_max%100), 100)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go')
                errorbar(windows-((FUS_on_time-10)), pd.rolling_mean(np.mean(evoked_window_amps_corrected[:, :, trodes_to_plot], axis=2)[probe][group],window = rolling_window), yerr = list_peaks_errs_combined[probe][group], color='olive')
                print('Plotting the Automated Time Window Analysis with Rolling Mean/ List of Electrodes\n')
                savefig(analyzed_path_for_group + 'tw_svg_format/' + 'time_windows_list_with_rm.svg', format = 'svg')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' + 'time_windows_list_with_rm.pdf', format = 'pdf')
                list_lfp_averages[probe][group] = np.mean(evoked_window_amps_corrected[:, :, trodes_to_plot], axis=2)[probe][group]
                close()

            for trode in range(p['nr_of_electrodes_per_group']):
                figure()#Plot the Automated Time Window Analysis of Each Electrode
                title('Automated Time Window Analysis / Electrode-{}'.format(trode))
                plot(windows-(FUS_on_time-10), evoked_window_amps[probe][group][trode], 'k-')
                xlabel('Time (min)')
                ylabel('Peak voltage (uV)')
                ylim_min = np.floor(np.min(evoked) / 100) * y_scale
                ylim_max = np.ceil(np.max(evoked) / 100) * y_scale
                ylim(ylim_min, ylim_max)
                xlim_min =0
                xlim_max = (num_window-1)*tw-(FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%100), (ylim_max-ylim_max%100), 100)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                print('Plotting the Automated Time Window Analysis / Electrode-{}'.format(trode))
                errorbar(windows-((FUS_on_time-10)), evoked_window_amps[probe][group][trode], yerr = evoked_window_peak_errs[probe][group][trode])
                savefig(analyzed_path_for_group + 'tw_svg_format/' +'electrode_' + str(trode) + '_time_windows.svg', format = 'svg')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' +'electrode_' + str(trode) + '_time_windows.pdf', format = 'pdf')
                close()

                figure()#Plot the Corrected Automated Time Window Analysis of Each Electrode
                title('Automated Time Window Analysis / Electrode-{} - Corrected'.format(trode))
                plot(windows-(FUS_on_time-10), evoked_window_amps_corrected[probe][group][trode], 'k-')
                xlabel('Time (min)')
                ylabel('Peak voltage (uV)')
                ylim_min = np.floor(np.min(evoked) / 100) * y_scale
                ylim_max = np.ceil(np.max(evoked) / 100) * y_scale
                ylim(ylim_min, ylim_max)
                xlim_min =0
                xlim_max = (num_window-1)*tw-(FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%100), (ylim_max-ylim_max%100), 100)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                print('Plotting the Corrected Automated Time Window Analysis / Electrode-{}'.format(trode))
                errorbar(windows-((FUS_on_time-10)), evoked_window_amps_corrected[probe][group][trode], yerr = evoked_window_peak_errs[probe][group][trode], color='olive')
                savefig(analyzed_path_for_group + 'tw_svg_format/' +'electrode_' + str(trode) + '_time_windows_corr.svg', format = 'svg')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' +'electrode_' + str(trode) + '_time_windows_corr.pdf', format = 'pdf')
                lfp_averages[probe][group][trode]=evoked_window_amps_corrected[probe][group][trode]
                close()
                
                figure()#Plot the Automated Time Window Analysis with rolling mean of Each Electrode
                title('Automated Time Window Analysis with rolling mean/ Electrode-{}'.format(trode))
                plot(windows-(FUS_on_time-10), pd.rolling_mean(evoked_window_amps[probe][group][trode],window=rolling_window), 'k-')
                xlabel('Time (min)')
                ylabel('Peak voltage (uV)')
                ylim_min = np.floor(np.min(evoked_window_amps)-100) 
                ylim_max = np.ceil(np.max(evoked_window_amps) +100) 
                ylim(ylim_min, ylim_max)
                xlim_min =0
                xlim_max = (num_window-1)*tw-(FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%100), (ylim_max-ylim_max%100), 100)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                print('Plotting the Automated Time Window Analysis with Rolling Mean / Electrode-{}'.format(trode))
                errorbar(windows-((FUS_on_time-10)), pd.rolling_mean(evoked_window_amps[probe][group][trode],window=rolling_window),yerr = evoked_window_peak_errs[probe][group][trode])
                savefig(analyzed_path_for_group + 'tw_svg_format/' +'electrode_' + str(trode) + '_time_windows_with_rm.svg', format = 'svg')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' +'electrode_' + str(trode) + '_time_windows_with_rm.pdf', format = 'pdf')
                close()

                figure()#Plot the Corrected Automated Time Window Analysis with rolling mean of Each Electrode
                title('Automated Time Window Analysis with rolling mean/\n Electrode-{} - Corrected'.format(trode))
                plot(windows-(FUS_on_time-10), pd.rolling_mean(evoked_window_amps_corrected[probe][group][trode],window = rolling_window), 'k-')
                xlabel('Time (min)')
                ylabel('Peak voltage (uV)')
                ylim_min = np.floor(np.min(evoked_window_amps_corrected)- 100) 
                ylim_max = np.ceil(np.max(evoked_window_amps_corrected) + 100) 
                ylim(ylim_min, ylim_max)
                xlim_min =0
                xlim_max = (num_window-1)*tw-(FUS_on_time-10)
                xlim(xlim_min, xlim_max)
                y_ticks = np.arange((ylim_min-ylim_min%100), (ylim_max-ylim_max%100), 100)
                x_ticks = np.arange(xlim_min, xlim_max, tw)
                axes().set_xticks(x_ticks, minor=True)
                axes().set_yticks(y_ticks, minor=True)
                axes().grid(which='both')
                matplotlib.pyplot.plot(10, y_ticks[0], 'go') 
                print('Plotting the Corrected Automated Time Window Analysis with Rolling Mean/ Electrode-{}\n'.format(trode))
                errorbar(windows-((FUS_on_time-10)), pd.rolling_mean(evoked_window_amps_corrected[probe][group][trode],window = rolling_window),yerr = evoked_window_peak_errs[probe][group][trode], color='olive')
                savefig(analyzed_path_for_group + 'tw_svg_format/' +'electrode_' + str(trode) + '_time_windows_corr_with_rm.svg', format = 'svg')
                savefig(analyzed_path_for_group + 'tw_pdf_format/' +'electrode_' + str(trode) + '_time_windows_corr_with_rm.pdf', format = 'pdf')
                lfp_averages[probe][group][trode]=evoked_window_amps_corrected[probe][group][trode]
                close() 

        print('\nAutomated window LFP analysis is completed succesfully.\n')





