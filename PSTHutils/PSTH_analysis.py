"""
This function creates Post-stimulus Time Histogram graphs for the clustered spike data. It is useful for visualizing the effects of the experiment on the subject.

Required Files:
    '~.kwik', '~.kwx', '~spikeinfo.pickle', '~.clu.0' under 'analysis_files' folder.
    'paramsDict.p', 'time.dat', '~evoked.pickle' under the folder for raw data.

Inputs:
    psthParameters: A dictionary which stores the parameters required for the analysis. It can be either created through Jupyter Pipeline or using the below test scheme.

Outputs:
    Saves the PSTH graphs under the analyzed folder of the experiment.

    Created on  July , 2018
    Author: Abdulkadir Gokce - Please contact him or Mehmet Ozdas in case of any questions.
"""

import numpy as np
import h5py
import pickle
import os
import sys
import matplotlib.pyplot as plt
#from matplotlib.cm import viridis
import shutil
import math
from spikeSortingUtils.spikes_dict_utils import get_spiketimes_per_channel



"""
#Values for test

#mainPath = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_06_05_FUSs1_EphysM1_E-FUS_NBBB88/'
mainPath = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/test_1234567/'
#mainPath = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_13_FUSs1_EphysM1_E-FUS_NBBB68/'
#mainPath = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_24_FUSs1_EphysM1_E-FUS_NBBB72/'
#mainPath = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_27_FUSs1_EphysM1_E-FUS_NBBB75/'

psthParameters = {}
psthParameters['mainPath'] = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/test_1234567/'
psthParameters['experiment_id'] = '1234567' 
psthParameters['group'] = 0
psthParameters['psth_all_electrodes'] = True
psthParameters['psth_all_units'] = False
psthParameters['user_window'] = [False,0,0]
psthParameters['decision'] = 'all'
psthParameters['pre_interval_ms'] = 10
psthParameters['post_interval_ms'] = 60
psthParameters['bis_size_ms'] = 1
psthParameters['t_fus_on'] = 11

"""

def psth_analysis(psthParameters):
    mainPath = psthParameters['mainPath']
    experiment_id = psthParameters['experiment_id']
    probe = 0
    group = psthParameters['group']
    stim_timestamps = np.zeros((0)) #Concatenated stimulation timestamps array of all iterated folders
    duration = 0 #Total duration of the iterated stimulation timestamps
    dirs = os.listdir(mainPath)
    for folder in sorted((folder for folder in dirs if ((folder != 'log.txt') and (folder != 'notes.docx') and (folder != 'analysis_files') and (folder != 'analyzed') and (folder != 'other') and (folder != '.DS_Store') and (folder != '._.?DS_Store')))):
    #This removes and creates the directories and iterates over stim_timestamps to create a whole one


        print("Reading Stimulation Timestamps: {0}\n".format(mainPath + folder))
        p = pickle.load(open(mainPath + folder + '/paramsDict.p', 'rb')) #Load the parameters of LFP Analysis and Spike Sorting
        sample_rate=p['sample_rate']

        analyzed_path = mainPath + 'analyzed/PSTH/probe_{0}_group_{1}'.format(probe,group)
        #If the directories already exist, remove them with the files within them; if the directories don't exist, create them.
        if os.path.exists(analyzed_path):
           shutil.rmtree(analyzed_path)

        if not os.path.exists(mainPath + 'analyzed'):
               os.mkdir(analyzed + 'mainPath') 

        if not os.path.exists(mainPath + 'analyzed/PSTH'):
               os.mkdir(mainPath + 'analyzed/PSTH')

        if not os.path.exists(analyzed_path):
               os.mkdir(analyzed_path)   
    
        if(psthParameters['psth_all_electrodes'] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_electrodes_pdf/'):
               os.mkdir(analyzed_path + '/PSTH_all_electrodes_pdf/')            
            
        if(psthParameters['psth_all_electrodes'] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_electrodes_svg/'):
                os.mkdir(analyzed_path + '/PSTH_all_electrodes_svg/')

        if(psthParameters['psth_all_units'] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_units_pdf/'):
               os.mkdir(analyzed_path + '/PSTH_all_units_pdf/')

        if(psthParameters['psth_all_units'] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_units_svg/'):
                os.mkdir(analyzed_path + '/PSTH_all_units_svg/')

        if(psthParameters['user_window'][0] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_electrodes_user_window_pdf/'):
               os.mkdir(analyzed_path + '/PSTH_all_electrodes_user_window_pdf/')            
            
        if(psthParameters['user_window'][0] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_electrodes_user_window_svg/'):
                os.mkdir(analyzed_path + '/PSTH_all_electrodes_user_window_svg/')

        if(psthParameters['user_window'][0] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_units_user_window_pdf/'):
               os.mkdir(analyzed_path + '/PSTH_all_units_user_window_pdf/')

        if(psthParameters['user_window'][0] == True):
            if not os.path.exists(analyzed_path + '/PSTH_all_units_user_window_svg/'):
                os.mkdir(analyzed_path + '/PSTH_all_units_user_window_svg/')

        
        #Iterate over all the files in the main directory and load the stimulation time stamps. Then, concatenate timestamps while adding previous total duration to generate single timestamp array.
        filePath = mainPath + folder + '/time.dat'
        evoked = pickle.load( open(mainPath + folder + '/probe_{0}_group_{1}/probe_{0}_group_{1}_evoked.pickle'.format(probe, group), 'rb') )
        evoked_waveforms = evoked['evoked']
        stim_timestamps_folder = evoked['stim_timestamps']
        stim_timestamps_folder = stim_timestamps_folder + duration #Add previous total duration to stimulation timestamps of the folder to take account the presendence of the stimulations
        if len(stim_timestamps)!=0: #If stim_timestamps is not contatenated yet, set it equal to current folder's timestamp
            stim_timestamps = np.concatenate((stim_timestamps, stim_timestamps_folder))
        else:
            stim_timestamps = stim_timestamps_folder
        duration += len( np.fromfile(open(filePath, 'rb'), np.int32))#Increase the total duration of the experiment by the time of current folder

    print("Analyzing: {0}\n".format(mainPath))

            
    #Create an array that keeps the timings of the windows. Each window is represented by a PSTH graph. The first window start 10 minute before the FUS
    window_timings = np.arange( (psthParameters['t_fus_on'] - psthParameters['window_duration'])*sample_rate*60, duration, psthParameters['window_duration']*sample_rate*60) 
    number_of_windows = len(window_timings)
    #Save the below parameters for furhter usage so that several PSTH can be merged more easily
    psthParameters['number_of_windows'] = number_of_windows
    psthParameters['nr_of_electrodes_per_group'] = p['nr_of_electrodes_per_group']
    psthParameters['groups'] = p['shanks']

    #Convert paramaters from ms to s, multiply them with sampling rate of the measurement
    psthParameters['pre_interval'] = psthParameters['pre_interval_ms'] * (sample_rate / 1000)
    psthParameters['post_interval'] = psthParameters['post_interval_ms'] * (sample_rate / 1000)
    psthParameters['bin_size'] = psthParameters['bin_size_ms'] * (sample_rate / 1000)
    psthParameters['window_size'] = psthParameters['pre_interval'] + psthParameters['post_interval']

    #Create x-axis of the histogram
    bins = np.arange(-psthParameters['pre_interval_ms'], psthParameters['post_interval_ms'], psthParameters['bin_size_ms'])#.astype(int)
    #colors = viridis(np.linspace(0, 1, len(bins))) #Color map for graphs
    #Arrange the window timings in 3-tuple to draw 3 figures in a file
    window_timings_triplet = np.zeros((math.ceil(number_of_windows/3), 3)) - 1
    for i in range(number_of_windows):
        window_timings_triplet[int(i/3), i%3] = window_timings[i]

    if(psthParameters['psth_all_electrodes'] == True):
        #PSTH Analysis for all electrodes
        print('\nElectrode-wise PSTH Analysis')

        spikes_dict = get_spiketimes_per_channel(mainPath, group, decision = psthParameters['decision']) #Dictionary containing all spiketimes of interest per electrode
        psth_all_trodes = np.zeros(( p['nr_of_electrodes_per_group'], number_of_windows, math.ceil(psthParameters['window_size'] / psthParameters['bin_size']) ))
        for trode in range(p['nr_of_electrodes_per_group']):
            spike_times = np.array(spikes_dict['channel{}'.format(trode)]) #Create a NumPy array that contains one electrode's spike times 
            for window in range(number_of_windows):
                if(window < number_of_windows - 1): #If it is last window, just make the lowerbound indexing
                    stim_window = stim_timestamps[np.all([stim_timestamps >= window_timings[window], stim_timestamps < window_timings[window+1]], axis=0)]
                else:
                    stim_window = stim_timestamps[np.all([stim_timestamps >= window_timings[window]], axis=0)]
                for stim in stim_window:
                    #Spike timings in a stimulation window
                    spike_times_stim_window = spike_times[np.all([spike_times >= (stim - psthParameters['pre_interval']), spike_times < (stim + psthParameters['post_interval'])], axis=0)]
                    #For a stimulation window, find the positions of the fired spikes and put them in bins
                    bin_positions = ((spike_times_stim_window - (stim - psthParameters['pre_interval'])) / psthParameters['bin_size']).astype(int)
                    #Increase the value by 1 of the spike positions
                    psth_all_trodes[trode, window, bin_positions] += 1
                #Divide the PSTH value of the window by the number of stimulation in this window
                psth_all_trodes[trode][window] = psth_all_trodes[trode][window] / len(stim_window)
            print('PSTH Analysis for electrode-{} is done'.format(trode))
        


        
        print('\nGenerating electrode-wise PSTH graphs')
        for trode in range(len(psth_all_trodes)):
            for i in range(len(window_timings_triplet)):
                fig=plt.figure()
                for j in range(3): #Draw 3 PSTH graphs in a single file
                    if(window_timings_triplet[i,j] != -1): #If value is -1 , no graph to plot
                        sp=plt.subplot(1, 3, j+1)
                        sp.bar(bins, psth_all_trodes[trode][i*3 + j])
                        plt.subplots_adjust(top=0.85) #Adjust the subplot to prevent overlapping
                        plt.subplots_adjust(wspace=0.5)
                        sp.set_ylabel('Spikes/Stim', fontsize=8)
                        sp.set_xlabel('Time(ms)', fontsize=8)
                        sp.tick_params(axis='both', which='major', labelsize=6)
                        minute_to_plot = window_timings_triplet[i,j]/(sample_rate*60)
                        if(window_timings_triplet[i,j] == window_timings[number_of_windows - 1]): #If the last window is printed, use the total duration as uppenbound of the interval
                            sp.set_title('Interval: {0} - {1} Minutes'.format(minute_to_plot, round(duration/(sample_rate*60), 2)), fontsize=6)
                        else:
                            sp.set_title('Interval: {0} - {1} Minutes'.format(minute_to_plot, minute_to_plot + psthParameters['window_duration']), fontsize=6)
                        y_max = np.max(np.max(psth_all_trodes[trode], axis=1), axis=0) #y_scale of the graph
                        sp.set_ylim(0, y_max)
                        sp.set_xlim(-psthParameters['pre_interval_ms'], psthParameters['post_interval_ms'])

                fig.suptitle('PSTH Graph / Group-{0} Electrode-{1}, Figure-{2}'.format(group,trode,i+1))
                plt.savefig(analyzed_path+'/PSTH_all_electrodes_pdf/electrode-{0}_figure-{1}.pdf'.format(trode, i+1), format='pdf')
                plt.savefig(analyzed_path+'/PSTH_all_electrodes_svg/electrode-{0}_figure-{1}.svg'.format(trode, i+1), format='svg')
                plt.close(fig)
                print('Finished: Electrode-{0}, Figure-{1}'.format(trode, i+1))

        #Save PSTH Analysis results as npy file, use script for combining the result of several recordings
        np.save('{0}/{1}_probe_{2}_group_{3}_psth_all_electrodes.npy'.format(analyzed_path,experiment_id,probe,group), psth_all_trodes)     

    if(psthParameters['psth_all_units'] == True):
        #PSTH Analysis for all units
        print('\nUnit-wise PSTH Analysis')
        
        #Load the spike timestamps of 'good' units
        spike_timestamps = pickle.load(open(mainPath+'analysis_files/probe_{0}_group_{1}/probe_{0}_group_{1}_spikeinfo.pickle'.format(probe,group), 'rb'))
        psth_all_units_values = np.zeros(( len(spike_timestamps['units']), number_of_windows, math.ceil(psthParameters['window_size'] / psthParameters['bin_size']) ))
        psth_all_units_keys = ['']*len(spike_timestamps['units'])
        keys = sorted(list(spike_timestamps['units'].keys()))
        for unit in range(len(spike_timestamps['units'])):
            key = keys[unit]
            spike_times = spike_timestamps['units'][key][0]
            for window in range(number_of_windows):
                if(window < number_of_windows - 1):
                    stim_window = stim_timestamps[np.all([stim_timestamps >= window_timings[window], stim_timestamps < window_timings[window+1]], axis=0)]
                else:
                    stim_window = stim_timestamps[np.all([stim_timestamps >= window_timings[window]], axis=0)]
                for stim in stim_window:
                    #Spike timings in a stimulation window
                    spike_times_stim_window = spike_times[np.all([spike_times >= (stim - psthParameters['pre_interval']), spike_times < (stim + psthParameters['post_interval'])], axis=0)]
                    #For a stimulation window, find the positions of the fired spikes and put them in bins
                    bin_position = ((spike_times_stim_window - (stim - psthParameters['pre_interval'])) / psthParameters['bin_size']).astype(int)
                    #Increase the value by 1 of the spike positions
                    psth_all_units_values[unit, window, bin_position] += 1
                #Divide the PSTH value of the window by the number of stimulation in this window
                psth_all_units_values[unit][window] = psth_all_units_values[unit][window] / len(stim_window)
            psth_all_units_keys[unit] = key
            print('PSTH Analysis for {} is done'.format(key))
        
        print('\nGenerating unit-wise PSTH graphs')
        for unit in range(len(psth_all_units_values)):
            for i in range(len(window_timings_triplet)):
                fig=plt.figure()
                for j in range(3): #Draw 3 PSTH graphs in a single file
                    if(window_timings_triplet[i,j] != -1): #If value is -1 , no graph to plot
                        sp=plt.subplot(1, 3, j+1)
                        sp.bar(bins, psth_all_units_values[unit][i*3 + j])
                        plt.subplots_adjust(top=0.85) #Adjust the subplot to prevent overlapping
                        plt.subplots_adjust(wspace=0.5)
                        sp.set_ylabel('Spikes/Stim', fontsize=8)
                        sp.set_xlabel('Time(ms)', fontsize=8)
                        sp.tick_params(axis='both', which='major', labelsize=6)
                        minute_to_plot = window_timings_triplet[i,j]/(sample_rate*60)
                        if(window_timings_triplet[i,j] == window_timings[number_of_windows - 1]): #If the last window is printed, use the total duration as uppenbound of the interval
                            sp.set_title('Interval: {0} - {1} Minutes'.format(minute_to_plot, round(duration/(sample_rate*60), 2)), fontsize=6)
                        else:
                            sp.set_title('Interval: {0} - {1} Minutes'.format(minute_to_plot, minute_to_plot + psthParameters['window_duration']), fontsize=6)
                        y_max = np.max(np.max(psth_all_units_values[unit], axis=1), axis=0) #y_scale of the graph
                        sp.set_ylim(0, y_max)
                        sp.set_xlim(-psthParameters['pre_interval_ms'], psthParameters['post_interval_ms'])

                if(psth_all_units_keys[unit] == 'unit1'):
                    fig.suptitle('PSTH Graph / Group-{0} All MUA, Figure-{1}'.format(group,i+1))
                    plt.savefig(analyzed_path+'/PSTH_all_units_pdf/all_mua_figure-{0}.pdf'.format(i+1), format='pdf')
                    plt.savefig(analyzed_path+'/PSTH_all_units_svg/all_mua_figure-{0}.svg'.format(i+1), format='svg')
                elif(psth_all_units_keys[unit] == 'unit0'):
                    fig.suptitle('PSTH Graph / Group-{0} All Noise, Figure-{1}'.format(group,i+1))
                    plt.savefig(analyzed_path+'/PSTH_all_units_pdf/all_noise_figure-{0}.pdf'.format(i+1), format='pdf')
                    plt.savefig(analyzed_path+'/PSTH_all_units_svg/all_noise_figure-{0}.svg'.format(i+1), format='svg')
                else:
                    fig.suptitle('PSTH Graph / Group-{0} Unit-{1}, Figure-{2}'.format(group,psth_all_units_keys[unit],i+1))
                    plt.savefig(analyzed_path+'/PSTH_all_units_pdf/{0}_figure-{1}.pdf'.format(psth_all_units_keys[unit], i+1), format='pdf')
                    plt.savefig(analyzed_path+'/PSTH_all_units_svg/{0}_figure-{1}.svg'.format(psth_all_units_keys[unit], i+1), format='svg')
                plt.close(fig)
                print('Finished: {0}, Figure-{1}'.format(psth_all_units_keys[unit], i+1))

        #Save PSTH Analysis results as pickle file, use script for combining the result of several recordings
        pickle.dump({'keys':psth_all_units_keys, 'values':psth_all_units_values}, open('{0}/{1}_probe_{2}_group_{3}_psth_all_units.pickle'.format(analyzed_path,experiment_id,probe,group), 'wb'), protocol = -1) 

    if(psthParameters['user_window'][0] == True):

        if(psthParameters['psth_all_electrodes'] == True):
            print('\nElectrode-wise PSTH Analysis for user window')
            start_min = psthParameters['user_window'][1]
            start = start_min * sample_rate * 60
            end_min = psthParameters['user_window'][2]
            end = end_min * sample_rate * 60
            psth_all_trodes_user_window = np.zeros(( p['nr_of_electrodes_per_group'], math.ceil(psthParameters['window_size'] / psthParameters['bin_size']) ))
            for trode in range(p['nr_of_electrodes_per_group']):
                spike_times = np.array(spikes_dict['channel{}'.format(trode)]) #Create a NumPy array that contains one electrode's spike times 
                stim_window = stim_timestamps[np.all([stim_timestamps >= start, stim_timestamps < end], axis=0)]
                for stim in stim_window:
                    #Spike timings in a stimulation window
                    spike_times_stim_window = spike_times[np.all([spike_times >= (stim - psthParameters['pre_interval']), spike_times < (stim + psthParameters['post_interval'])], axis=0)]
                    #For a stimulation window, find the positions of the fired spikes and put them in bins
                    bin_positions = ((spike_times_stim_window - (stim - psthParameters['pre_interval'])) / psthParameters['bin_size']).astype(int)
                    #Increase the value by 1 of the spike positions
                    psth_all_trodes_user_window[trode, bin_positions] += 1
                #Divide the PSTH value of the window by the number of stimulation in this window
                psth_all_trodes_user_window[trode] = psth_all_trodes_user_window[trode] / len(stim_window)
                print('PSTH Analysis of user window for electrode-{} is done'.format(trode))

            print('\nGenerating electrode-wise PSTH graphs for user window')
            for trode in range(len(psth_all_trodes_user_window)):
                fig=plt.figure()
                plt.bar(bins, psth_all_trodes_user_window[trode])
                plt.ylabel('Spikes/Stim', fontsize=8)
                plt.xlabel('Time(ms)', fontsize=8)
                plt.tick_params(axis='both', which='major', labelsize=6)
                plt.title('Interval: {0} - {1} Minutes'.format(start_min, end_min), fontsize=6)
                y_max = np.max(psth_all_trodes_user_window[trode], axis=0) #y_scale of the graph
                plt.ylim(0, y_max)
                plt.xlim(-psthParameters['pre_interval_ms'], psthParameters['post_interval_ms'])

                plt.suptitle('PSTH Graph / Group-{0} Electrode-{1}, Interval-{2}:{3}'.format(group,trode, start_min, end_min))
                plt.savefig(analyzed_path+'/PSTH_all_electrodes_user_window_pdf/electrode-{0}_interval-{1}-{2}.pdf'.format(trode, start_min, end_min), format='pdf')
                plt.savefig(analyzed_path+'/PSTH_all_electrodes_user_window_svg/electrode-{0}_interval-{1}-{2}.svg'.format(trode, start_min, end_min), format='svg')
                plt.close(fig)
                print('Finished: Electrode-{0}, Interval-{1}:{2}'.format(trode, start_min, end_min))
        
            #Save PSTH Analysis results as npy file, use script for combining the result of several recordings
            np.save('{0}/{1}_probe_{2}_group_{3}_psth_all_electrodes_user_window_interval_{4}-{5}.npy'.format(analyzed_path,experiment_id,probe,group,start_min,end_min), psth_all_trodes_user_window) 

        if(psthParameters['psth_all_units'] == True):            
            print('\nUnit-wise PSTH Analysis for user window')
            psth_all_units_values_user_window = np.zeros(( len(spike_timestamps['units']), math.ceil(psthParameters['window_size'] / psthParameters['bin_size']) ))
            psth_all_units_keys_user_window = ['']*len(spike_timestamps['units'])
            for unit in range(len(spike_timestamps['units'])):
                key = keys[unit]
                spike_times = spike_timestamps['units'][key][0]
                stim_window = stim_timestamps[np.all([stim_timestamps >= start, stim_timestamps < end], axis=0)]
                for stim in stim_window:
                    #Spike timings in a stimulation window
                    spike_times_stim_window = spike_times[np.all([spike_times >= (stim - psthParameters['pre_interval']), spike_times < (stim + psthParameters['post_interval'])], axis=0)]
                    #For a stimulation window, find the positions of the fired spikes and put them in bins
                    bin_position = ((spike_times_stim_window - (stim - psthParameters['pre_interval'])) / psthParameters['bin_size']).astype(int)
                    #Increase the value by 1 of the spike positions
                    psth_all_units_values_user_window[unit, bin_position] += 1
                #Divide the PSTH value of the window by the number of stimulation in this window
                psth_all_units_values_user_window[unit] = psth_all_units_values_user_window[unit] / len(stim_window)
                psth_all_units_keys_user_window[unit] = key
                print('PSTH Analysis for unit-{} is done'.format(key))
            
            print('\nGenerating unit-wise PSTH graphs for user window')
            for unit in range(len(psth_all_units_values_user_window)):
                fig=plt.figure()
                plt.bar(bins, psth_all_units_values_user_window[unit])
                plt.ylabel('Spikes/Stim', fontsize=8)
                plt.xlabel('Time(ms)', fontsize=8)
                plt.tick_params(axis='both', which='major', labelsize=6)
                plt.title('Interval: {0}-{1} Minutes'.format(start_min, end_min), fontsize=6)
                y_max = np.max(psth_all_units_values_user_window[unit], axis=0) #PSTH Analysis for all electrodes
                plt.ylim(0, y_max)
                plt.xlim(-psthParameters['pre_interval_ms'], psthParameters['post_interval_ms'])

                if(psth_all_units_keys[unit] == 'unit1'):
                    plt.suptitle('PSTH Graph / Group-{0} All MUA, Interval-{2}:{3}'.format(group, start_min, end_min))
                    plt.savefig(analyzed_path+'/PSTH_all_units_user_window_pdf/all_mua_interval-{1}-{2}.pdf'.format(start_min, end_min), format='pdf')
                    plt.savefig(analyzed_path+'/PSTH_all_units_user_window_svg/all_mua_interval-{1}-{2}.svg'.format(start_min, end_min), format='svg')
                elif(psth_all_units_keys[unit] == 'unit0'):
                    plt.suptitle('PSTH Graph / Group-{0} All Noise, Interval-{2}:{3}'.format(group, start_min, end_min))
                    plt.savefig(analyzed_path+'/PSTH_all_units_user_window_pdf/all_noise_interval-{1}-{2}.pdf'.format(start_min, end_min), format='pdf')
                    plt.savefig(analyzed_path+'/PSTH_all_units_user_window_svg/all_noise_interval-{1}-{2}.svg'.format(start_min, end_min), format='svg')
                else:
                    plt.suptitle('PSTH Graph / Group-{0} Unit-{1}, Interval-{2}:{3}'.format(group, psth_all_units_keys[unit], start_min, end_min))
                    plt.savefig(analyzed_path+'/PSTH_all_units_user_window_pdf/{0}_interval-{1}-{2}.pdf'.format(psth_all_units_keys[unit], start_min, end_min), format='pdf')
                    plt.savefig(analyzed_path+'/PSTH_all_units_user_window_svg/{0}_interval-{1}-{2}.svg'.format(psth_all_units_keys[unit], start_min, end_min), format='svg')
                plt.close(fig)
                print('Finished: {0},  Interval-{1}:{2}'.format(psth_all_units_keys[unit], start_min, end_min))
            #Save PSTH Analysis results as pickle file, use script for combining the result of several recordings
            pickle.dump({'keys':psth_all_units_keys_user_window, 'values':psth_all_units_values_user_window}, open( '{0}/{1}_probe_{2}_group_{3}_psth_all_units_user_window_interval_{4}-{5}.pickle'.format(analyzed_path,experiment_id,probe,group,start_min,end_min), 'wb'), protocol = -1)

    pickle.dump(psthParameters, open('{0}/{1}_probe_{2}_group_{3}_psthParameters.pickle'.format(analyzed_path,experiment_id,probe,group), 'wb'), protocol = -1)
    #np.save('{0}/{1}_probe_{2}_group_{3}_window_timings.npy'.format(analyzed_path,experiment_id,probe,group), window_timings)

    print('\nPSTH Analysis is completed succesfully.')

    

