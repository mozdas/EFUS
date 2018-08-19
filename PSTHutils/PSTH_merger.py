"""
This script merges PSTH graphs of several analyzed experiments. Main path should be the parent directory which includes all the data folders that is used in the process. Script can also sort the electrodes wrt PSTH values in a given integration interval and calculate only for given number of top-valued electrodes.

    Reqired Files:
        psthParamters: Contains PSTH analysis parameters
        path_for_psth_all_electrodes: Contains electrode-wise PSTH values.
        
    Input:
        main_path: Parent directory to the data folders.
        integration_start, integration_end: A part of evoked LFP window whose PSTH values integrated to sort highest responding electrodes
        bumber_of_electrodes: Indicates how many electrodes will be used from the sorted electrodes.

    Outputs:
        Creates electrode-wise merged PSTH graphs. If there were a mismatched parameter, it would raise an error.

    Created on  July , 2018
    Author: Abdulkadir Gokce - Please contact him or Mehmet Ozdas in case of any questions.
"""

import os
import numpy as np
import pickle
import math
import shutil
import matplotlib.pyplot as plt


"""
#####Parameters for the script
#main_path ='/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/Analyzed-Cleaned/PBS/'
main_path ='/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/Analyzed-Cleaned/Muscimol/'
integration_start = 0
integration_end = 40
number_of_electrodes = 3
###########
"""

def merge_all_PSTHs(main_path, integration_start, integration_end, number_of_electrodes):

    dirs = os.listdir(main_path)
    to_be_skipped = ['other','average_PSTH']
    data_folders = [folder for folder in dirs if (folder not in to_be_skipped)]

    #Preallocate the analysis arrays
    window_durations = np.zeros(len(data_folders))
    bin_sizes = np.zeros(len(data_folders))
    pre_intervals = np.zeros(len(data_folders))
    post_intervals = np.zeros(len(data_folders))
    number_of_windows = np.zeros(len(data_folders))
    groups = np.zeros(len(data_folders))
    nr_of_electrodes_per_group = np.zeros(len(data_folders))

    for folder_number in range(len(data_folders)):

        experiment_id = data_folders[folder_number].rpartition('_')[2]

        PSTH_folder = main_path + data_folders[folder_number] + '/analyzed/PSTH'
        PSTH_folder_for_group = '{0}/probe_0_group_0'.format(PSTH_folder)
        psthParameters_path = '{0}/{1}_probe_0_group_0_psthParameters.pickle'.format(PSTH_folder_for_group, experiment_id)
        psthParamters = pickle.load(open(psthParameters_path,'rb')) #Load PSTH parameters for fist group

        for group in range(psthParamters['groups']-1): #Check whether all groups has the same PSTH parameters
            PSTH_folder_for_other_groups = '{0}/probe_0_group_{1}/'.format(PSTH_folder, group+1)
            psthParameters_other_groups_path = '{0}/{1}_probe_0_group_{2}_psthParameters.pickle'.format(PSTH_folder_for_other_groups, experiment_id, group+1)
            psthParamters_other_groups = pickle.load(open(psthParameters_other_groups_path,'rb'))
            if(((psthParamters['window_duration'] == psthParamters_other_groups['window_duration']) and (psthParamters['bin_size_ms'] == psthParamters_other_groups['bin_size_ms']) and (psthParamters['pre_interval_ms'] == psthParamters_other_groups['pre_interval_ms']) and (psthParamters['post_interval_ms'] == psthParamters_other_groups['post_interval_ms']) and (psthParamters['number_of_windows'] == psthParamters_other_groups['number_of_windows']))== False): #If not, raise an error
                print('Groups of Experiment {} were not analyzed with the same parameters. Rerun the analysis using the same parameters and then try again.'.format(experiment_id))
                return None

        #Save all parameters in respective arrays
        window_durations[folder_number] = psthParamters['window_duration']
        bin_sizes[folder_number] = psthParamters['bin_size_ms']
        pre_intervals[folder_number] = psthParamters['pre_interval_ms']
        post_intervals[folder_number] = psthParamters['post_interval_ms']
        number_of_windows[folder_number] = psthParamters['number_of_windows']
        groups[folder_number] = psthParamters['groups']
        nr_of_electrodes_per_group[folder_number] = psthParamters['nr_of_electrodes_per_group']



    #Check whether all elements of an array are the same. If not, raise an error.
    if(np.all(window_durations == window_durations[0]) == False):
        #problematic_folders=np.unique(PSTH_window_durations, return_index = True, return_counts = True)
        print('Not all the PSTH analyses have the same window duration. Rerun the analysis on the data whose window duration different than others and then try again.')
        return None
    elif(np.all(bin_sizes == bin_sizes[0]) == False):
        print('Not all the PSTH analyses have the same bin size. Rerun the analysis on the data whose bin size different than others and then try again.')
        return None
    elif(np.all(pre_intervals == pre_intervals[0]) == False):
        print('Not all the PSTH analyses have the same duration of pre interval. Rerun the analysis on the data whose duration of pre interval different than others and then try again.')
        return None
    elif(np.all(post_intervals == post_intervals[0]) == False):
        print('Not all the PSTH analyses have the same duration of post interval. Rerun the analysis on the data whose duration of post interval different than others and then try again.')
        return None
    elif(np.all(groups == groups[0]) == False):
        print('Not all the data has the same number of groups per probe')
        return None
    elif(np.all(nr_of_electrodes_per_group == nr_of_electrodes_per_group[0]) == False):
        print('Not all the data has the same number of electrodes per group.')
        return None

    #Clean and create directories
    if(os.path.exists(main_path + 'average_PSTH/')):
        shutil.rmtree(main_path + 'average_PSTH/')
    if not(os.path.exists(main_path + 'average_PSTH/')):
        os.mkdir(main_path + 'average_PSTH/')
    if not(os.path.exists(main_path + 'average_PSTH/pdf/')):
        os.mkdir(main_path + 'average_PSTH/pdf/')
    if not(os.path.exists(main_path + 'average_PSTH/svg/')):
        os.mkdir(main_path + 'average_PSTH/svg/')


    #After being sure about every parameter is the same, assinging them to single variables
    window_duration = window_durations[0]
    bin_size = bin_sizes[0]
    pre_interval = pre_intervals[0]
    post_interval = post_intervals[0]
    number_of_groups = int(groups[0])
    nr_of_electrodes = int(nr_of_electrodes_per_group[0])
    number_of_window = int(np.min(number_of_windows)) #Taking the minimum number of windows over all experiments.

    #Preallocate the analysis arrays
    psth_all = np.zeros(( len(data_folders), number_of_groups, nr_of_electrodes, number_of_window, math.ceil( (pre_interval+post_interval) / bin_size ) ))
    max_responding_electrodes = np.zeros((len(data_folders), number_of_electrodes))
    psth_max_responding_electrodes = np.zeros(( len(data_folders), number_of_electrodes, number_of_window, math.ceil( (pre_interval+post_interval) / bin_size ) ))

    for folder_number in range(len(data_folders)):
        experiment_id = data_folders[folder_number].rpartition('_')[2]

        PSTH_path = main_path + data_folders[folder_number] + '/analyzed/PSTH'
        PSTH_for_group_folders = os.listdir(PSTH_path)

        for group in range(number_of_groups):
            
            PSTH_group_path = PSTH_path + '/probe_0_group_{}'.format(group)
            path_for_psth_all_electrodes = PSTH_group_path + '/{0}_probe_0_group_{1}_psth_all_electrodes.npy'.format(experiment_id,group)
            psth_all[folder_number,group] = np.load(path_for_psth_all_electrodes)[:, :number_of_window, :] #Loading PSTH arrays from files. 


    integ_index_start = int(integration_start + pre_interval)
    integ_index_end = int(integration_end + pre_interval)
    #Reshaped PSTH of which groups of electrodes considered as different electordes
    psth_reshaped = psth_all.reshape(( len(data_folders), number_of_groups * nr_of_electrodes, number_of_window, math.ceil( (pre_interval+post_interval) / bin_size ) ))
    psth_integrated = np.sum(np.sum(psth_reshaped[:,:,:, integ_index_start:integ_index_end], axis=3), axis=2) #Integral along time and window axes
    psth_integrated_sorted = np.sort(psth_integrated, kind='mergesort', axis=1) #Sorting the electrodes wrt their integral values using above result


    for folder_number in range(len(data_folders)):
        for electrode in range(number_of_electrodes):
            #Finding indexes of the electrodes which have the highest ingtegral value over the given time interval.
            max_responding_electrodes[folder_number][electrode] = np.where(psth_integrated[folder_number] == psth_integrated_sorted[folder_number][nr_of_electrodes-electrode-1])[0]
         
    #electrodes = np.where( psth_integrated == psth_integrated.max(axis=2))
    for folder_number in range(len(data_folders)):
        #Selecting only the highest responding electrodes using the indexes finded above
        psth_max_responding_electrodes[folder_number] = psth_reshaped[:, max_responding_electrodes[folder_number].astype(int), :, : ][folder_number]

    #Taking average along electrode and folders axes
    psth_avg = np.mean(np.mean(psth_max_responding_electrodes, axis=1), axis=0)

    #Create an array that keeps the timings of the windows. Each window is represented by a PSTH graph. The first window start 10 minute before the FUS
    window_timings = np.arange( -window_duration, (number_of_window-1)*window_duration , window_duration)
    #Arrange the window timings in 3-tuple to draw 3 figures in a file
    window_timings_triplet = np.zeros((math.ceil(number_of_window/3), 3)) - 1
    for i in range(number_of_window):
        window_timings_triplet[int(i/3), i%3] = window_timings[i]
    #Create x-axis of the histogram
    bins = np.arange(-pre_interval, post_interval, bin_size)



    print('\nGenerating electrode-wise PSTH graphs')
    for i in range(len(window_timings_triplet)):
        fig=plt.figure()
        for j in range(3): #Draw 3 PSTH graphs in a single file
            if(window_timings_triplet[i,j] != -1): #If value is -1 , no graph to plot
                sp=plt.subplot(1, 3, j+1)
                sp.bar(bins, psth_avg[i*3 + j])
                plt.subplots_adjust(top=0.85) #Adjust the subplot to prevent overlapping
                plt.subplots_adjust(wspace=0.5)
                sp.set_ylabel('Spikes/Stim', fontsize=8)
                sp.set_xlabel('Time(ms)', fontsize=8)
                sp.tick_params(axis='both', which='major', labelsize=6)
                minute_to_plot = window_timings_triplet[i,j]
                if(window_timings_triplet[i,j] == -window_duration): #If the first window is printed
                    sp.set_title('FUS On {0} / FUS On '.format(minute_to_plot), fontsize=6)
                elif(window_timings_triplet[i,j] == -window_duration): #If the second window is printed:
                    sp.set_title('FUS On / FUS On +{0}'.format(window_duration), fontsize=6)
                else:
                    sp.set_title('FUS On +{0}/ FUS On +{1}'.format(minute_to_plot, minute_to_plot+window_duration ), fontsize=6)
                y_max = np.max(np.max(psth_avg, axis=1), axis=0) #y_scale of the graph
                sp.set_ylim(0, y_max)
                sp.set_xlim(-pre_interval, post_interval)

        fig.suptitle('Average PSTH / Figure-{0}'.format(i+1))
        plt.savefig(main_path+'average_PSTH/pdf/figure-{0}.pdf'.format( i+1), format='pdf')
        plt.savefig(main_path+'average_PSTH/svg/figure-{0}.svg'.format(i+1), format='svg')
        plt.close(fig)
        print('Finished: Figure-{0}'.format( i+1))

    print('\nMerging all PSTHs is completed successfully.')





