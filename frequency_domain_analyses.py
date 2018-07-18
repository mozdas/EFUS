"""
This file contains scripts for frequency domain analysis and uses SciPy's 'signal' and 'fftpack' packages.

    Periodogram : Estimates power spectral density using a periodogram.
    Fourier Transform: Return discrete Fourier transform of real or complex sequence. 
    Spectrogram: Compute a spectrogram with consecutive Fourier transforms.

    Required Files:
        This script uses down sampled (1000 Hz) evoked LFP data for analysis. It can be generated by checking the option when generating dictionay for the LFP Analysis and then running the LFP Pipeline.

    Input:
        main_path: Path for the data
        t_fus_on: The time when FUS is activated
        t_fus_off: The time when FUS is deactivated
        t_recovery: The time when recovery starts
        periodogram_analysis, fourier_transformation, spectrogram_analysis: Decision whether  a type of analysis will be done
        n_perseg, n_overlap, n_fft: Parameters for scipy.signal.spectrogram. See visit its documentation site for more info
    
    Output:
        Creates figures under 'analyzed' folder

    Created on  July , 2018
    Author: Abdulkadir Gokce - Please contact him or Mehmet Ozdas in case of any questions.
"""


import os
import shutil
import pickle
import numpy as np
from scipy import signal
import scipy.fftpack as fft
import math
import matplotlib.pyplot as plt

#######Parameters for test
main_path = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_27_FUSs1_EphysM1_E-FUS_NBBB75/'
#main_path = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_13_FUSs1_EphysM1_E-FUS_NBBB68/'
t_fus_on = 35
t_fus_of = 64
t_recovery = 95
periodogram_analysis = True
fourier_transformation = True
spectrogram_analysis = False
n_perseg = 16
n_overlap = 15
n_fft = 1024
#########



directories_to_skip = ['analyzed', 'analysis_files', 'other', 'log.txt', 'notes.txt', '.DS_Store', '._.DS_Store']
dirs = os.listdir(main_path)
window_titles = ['Baseline - Before FUS', 'During FUS', 'Post-FUS', 'Recovery' ]

for folder in (folder for folder in dirs if(folder not in directories_to_skip)):
    parameters_path = '{0}{1}/paramsDict.p'.format(main_path, folder) 
    p = pickle.load(open(parameters_path,'rb')) #Load the parameters dictionary
    

    #Preallocate the analysis arrays
    periodograms = np.zeros((p['probes'], p['shanks'], 4, p['nr_of_electrodes_per_group'], math.ceil( (p['evoked_pre']+p['evoked_post'])*500) ))
    frequencies = np.zeros((p['probes'], p['shanks'], 4, math.ceil( (p['evoked_pre']+p['evoked_post'])*500) ))
    fourier_transformed_powers = np.zeros((p['probes'], p['shanks'], 4, p['nr_of_electrodes_per_group'], math.ceil( (p['evoked_pre']+p['evoked_post'])*1000) ))
    spectrograms = np.zeros((p['probes'], p['shanks'], 4, p['nr_of_electrodes_per_group'], int(n_fft/2)+1, math.ceil(((p['evoked_pre']+p['evoked_post'])*1000-n_perseg+1)/(n_perseg-n_overlap)) ))
    f = np.zeros((p['probes'], p['shanks'], 4,  int(n_fft/2)+1 ))
    t = np.zeros((p['probes'], p['shanks'], 4,  math.ceil(((p['evoked_pre']+p['evoked_post'])*1000-n_perseg+1)/(n_perseg-n_overlap)) ))
    
    for probe in range(p['probes']):
        for group in range(p['shanks']):
            data_path = '{0}{1}/probe_{2}_group_{3}/probe_{4}_group_{5}_evoked_down_sampled.pickle'.format(main_path, folder, probe, group, probe, group)
            fid = open(data_path,'rb')
            down_sampled_data = pickle.load(fid) #Load the down sampled (1000 Hertz) evoked data
            fid.close()

            evoked = down_sampled_data['evoked'] #Down sampled evoked data
            stim_timestamps = down_sampled_data['stim_timestamps'] #Shifted stimulation timestamps
            sample_rate = 1000 #Sample rate of down sampled data. 

            #Construct windows list to slice evoked data in a loop
            windows = [10*60*sample_rate, t_fus_on*60*sample_rate, t_fus_of*60*sample_rate, t_recovery*60*sample_rate, stim_timestamps[-1]]

            for time in range(4):
                evoked_window = evoked[np.all( [stim_timestamps>windows[time], stim_timestamps<windows[time +1]], axis=0)] #Slicing the data that falls into the current window

                if(periodogram_analysis == True):
                    #Calling periodogram function, it returns sample frequencies and power spectral density/spectrum                
                    frequencies[probe, group, time], Pxx  = signal.periodogram(evoked_window, fs=sample_rate, axis=2, scaling='density') 
                    periodograms[probe, group, time] = np.mean(Pxx, axis=0) #Mean along the stimulus window axis
                    print('Periodogram for {} is completed'.format(window_titles[time]))

                if(fourier_transformation == True):
                    #Calling fourier transformation function and take fourier transformation of the data
                    fourier_transformed = fft.fft(evoked_window, axis=2)
                    #Calculate power to get rid of complex part and take the mean along axis 0
                    fourier_transformed_powers[probe, group, time] = np.mean([np.abs(fourier_transformed[:,trode,:]) for trode in range(p['nr_of_electrodes_per_group'])], axis=1)
                    fourier_frequency = fft.fftfreq(n=evoked_window.shape[-1], d=1/sample_rate) #Discrete Fourier Transform sample frequencies
                    print('Fourier Transformation for {} is completed'.format(window_titles[time]))

                if(spectrogram_analysis ==  True):
                    #Calling spectrogram function, it returns sample frequencies, segment times and power spectral density/spectrum
                    f[probe, group, time] , t[probe, group, time], Sxx = signal.spectrogram(evoked_window, fs=sample_rate, axis=2, nperseg=n_perseg, nfft=n_fft, noverlap=n_overlap)
                    spectrograms[probe, group, time] = np.mean(Sxx, axis=0) #Mean along the stimulus window axis
                    print('Spectrogram for {} is completed'.format(window_titles[time]))


        #Cleaning and/or creating directories
        analyzed_path = '{0}analyzed'.format(main_path)
        analyzed_folder_path = '{0}/{1}'.format(analyzed_path, folder)
        analyzed_group_path = '{0}/probe_{1}_group_{2}'.format(analyzed_folder_path, probe, group)
        save_path = '{0}/spectral_analysis'.format(analyzed_group_path)
        pdf_path = '{0}/pdf'.format(save_path)
        svg_path = '{0}/svg'.format(save_path)
        pdf_periodogram_path = '{0}/periodogram'.format(pdf_path)
        pdf_fourier_analysis_path = '{0}/fourier_analysis'.format(pdf_path)
        pdf_spectrogram_path = '{0}/spectrogram'.format(pdf_path)
        svg_periodogram_path = '{0}/periodogram'.format(svg_path)
        svg_fourier_analysis_path = '{0}/fourier_analysis'.format(svg_path)
        svg_spectrogram_path = '{0}/spectrogram'.format(svg_path)




        if (os.path.exists(save_path)):
            shutil.rmtree(save_path)

        if not (os.path.exists(analyzed_path)):
            os.mkdir(analyzed_path)

        if not (os.path.exists(analyzed_folder_path)):
            os.mkdir(analyzed_folder_path)

        if not (os.path.exists(analyzed_group_path)):
            os.mkdir(analyzed_group_path)

        if not (os.path.exists(save_path)):
            os.mkdir(save_path)

        if not (os.path.exists(pdf_path)):
            os.mkdir(pdf_path)

        if not (os.path.exists(svg_path)):
            os.mkdir(svg_path)

        if not (os.path.exists(pdf_periodogram_path)):
            os.mkdir(pdf_periodogram_path)

        if not (os.path.exists(pdf_fourier_analysis_path)):
            os.mkdir(pdf_fourier_analysis_path)

        if not (os.path.exists(pdf_spectrogram_path)):
            os.mkdir(pdf_spectrogram_path)

        if not (os.path.exists(svg_periodogram_path)):
            os.mkdir(svg_periodogram_path)

        if not (os.path.exists(svg_fourier_analysis_path)):
            os.mkdir(svg_fourier_analysis_path)

        if not (os.path.exists(svg_spectrogram_path)):
            os.mkdir(svg_spectrogram_path)

        

        #Draw figures
        for trode in range(p['nr_of_electrodes_per_group']):
            if(periodogram_analysis == True):
                for i in range(2): #To split 4 windows into 2 figures, use to different loops
                    fig = plt.figure()
                    for j in range(2):
                        sp = plt.subplot(1, 2, j+1)
                        sp.bar(frequencies[probe, group, i*2 + j], periodograms[probe, group, i*2 + j, trode], color='r')
                        sp.set_ylabel('Power Spectral Density (V^2/Hertz)', fontsize=6)
                        sp.set_xlabel('Frequency (Hertz)', fontsize=6)
                        sp.set_title(window_titles[i*2 + j], fontsize=6)
                        sp.set_ylim(0, np.max(np.max(periodograms, axis=4), axis=2)[probe, group, trode])
                        sp.set_xlim(0, 300)
                        sp.tick_params(axis='both', which='major', labelsize=6)
                        plt.subplots_adjust(top=0.85, wspace=0.5)
                        print('Periodogram subplot for {} is completed'.format(window_titles[i*2 + j]))

                    plt.savefig('{0}/electrode-{1}_figure-{2}.pdf'.format(pdf_periodogram_path, trode, i+1), format='pdf')
                    plt.savefig('{0}/electrode-{1}_figure-{2}.svg'.format(svg_periodogram_path, trode, i+1), format='svg')
                    plt.close()
                print('Periodogram figure completed')

            if(fourier_transformation == True):
                for i in range(2):
                    fig = plt.figure()
                    for j in range(2):
                        sp = plt.subplot(1, 2, j+1)
                        sp.bar(fourier_frequency, fourier_transformed_powers[probe, group, i*2 + j, trode], color='r')
                        sp.set_ylabel('Power Spectrum (V^2)', fontsize=6)
                        sp.set_xlabel('Frequency (Hertz)', fontsize=6)
                        sp.set_title(window_titles[i*2 + j], fontsize=6)
                        sp.set_ylim(0, np.max(np.max(fourier_transformed_powers, axis=4), axis=2)[probe, group, trode])
                        sp.set_xlim(-300, 300)
                        sp.tick_params(axis='both', which='major', labelsize=6)
                        plt.subplots_adjust(top=0.85, wspace=0.5)
                        print('Fourier Analysis subplot for {} is completed'.format(window_titles[i*2 + j]))

                    plt.savefig('{0}/electrode-{1}_figure-{2}.pdf'.format(pdf_fourier_analysis_path, trode, i+1), format='pdf')
                    plt.savefig('{0}/electrode-{1}_figure-{2}.svg'.format(svg_fourier_analysis_path, trode, i+1), format='svg')
                    plt.close()
            print('Periodogram figure completed')


            if(spectrogram_analysis ==  True):
                for i in range(2):
                    fig = plt.figure()
                    for j in range(2):
                        sp = plt.subplot(1, 2, j+1)
                        sp.pcolormesh(t[probe, group, time],f[probe, group, time], spectrograms[probe, group, time, trode])
                        #sp.pcolormesh(t[probe, group, i*2 + j], f[probe, group, i*2 + j], np.log10(spectrograms[probe, group, i*2 + j, trode])*1000)
                        sp.set_ylabel('Frequency (Hertz)', fontsize=6)
                        sp.set_xlabel('Time (ms)', fontsize=6)
                        sp.set_title(window_titles[i*2 + j], fontsize=6)
                        sp.set_xlim(-p['evoked_pre']*1000, p['evoked_post']*1000)
                        sp.tick_params(axis='both', which='major', labelsize=4)
                        plt.subplots_adjust(top=0.85, wspace=0.5)
                        print('Periodogram subplot for {} is completed'.format(window_titles[i*2 + j]))
                    plt.suptitle('Power Spectral Density')
                    plt.savefig('{0}/electrode-{1}_figure-{2}.pdf'.format(pdf_spectrogram_path, trode, i+1), format='pdf')
                    plt.savefig('{0}/electrode-{1}_figure-{2}.svg'.format(svg_spectrogram_path, trode, i+1), format='svg')
                    plt.close()
            print('Spectrogram figures completed')
                            

                
            
