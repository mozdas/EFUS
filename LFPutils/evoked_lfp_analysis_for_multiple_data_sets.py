import pickle
import os
import numpy as np
import math
from matplotlib.pyplot import *
from pandas import *

##mainPath = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_12_FUSs1_EphysM1_E-FUS_NBBB67/'
mainPath = sys.stdin.read().splitlines()[0]
print(mainPath)
dirs = os.listdir(mainPath)
dirs.sort()
analyzed_path = mainPath + 'analyzed/'
if not os.path.exists(analyzed_path):
    os.mkdir(analyzed_path)
writer_0 = ExcelWriter(analyzed_path + 'peak_data_prob_0.xlsx', engine = 'xlsxwriter')
a=0
c=0
stim_timestamps = np.zeros((0)) #Concatenated stimulation timestamps array of all iterated folders
duration = 0 #Total duration of the iterated stimulation timestamps
evoked_data = np.zeros((0))

rev_experiment_id=''.join(reversed(mainPath))
experiment_id=rev_experiment_id.split('_')[0] 
experiment_id=''.join(reversed(experiment_id))
experiment_id = experiment_id.replace("/", "")

for folder in (folder for folder in dirs if ((folder != 'log.txt') and (folder != 'notes.docx') and (folder != 'analysis_files') and (folder != 'other') and (folder != 'analyzed') and (folder != '.DS_Store'))):
    c=c+1
    doc=folder

for folder in (folder for folder in dirs if ((folder != 'log.txt') and (folder != 'notes.docx') and (folder != 'analysis_files') and (folder != 'other') and (folder != 'analyzed') and (folder != '.DS_Store'))):
        
    if(folder==doc):
        p = pickle.load(open(mainPath + folder + '/paramsDict.p', 'rb'))
        if (p['probes']==2) and (a==0):
            writer_1 = ExcelWriter(analyzed_path + 'peak_data_probe_1.xlsx', engine= 'xlswriter') 
            a=a+1
        
        peak_info = {key: {} for key in range(p['probes'])}
        
        analyzed_path_for_folder = analyzed_path + experiment_id
        if not os.path.exists(analyzed_path_for_folder):
            os.mkdir(analyzed_path_for_folder)

        for probe in range(p['probes']):
            peak_info[probe]['peak_locs'], peak_info[probe]['peak_stds'], peak_info[probe]['peak_times'], peak_info[probe]['peak_errs'], peak_info[probe]['peak_amps'] = (np.zeros(p['nr_of_electrodes']) for i in range(5))
            for group in range(p['nr_of_groups']):
                data= pickle.load(open(mainPath + folder + '/probe_{:g}_group_{:g}/probe_{:g}_group_{:g}_evoked.pickle'.format(probe,group,probe,group), 'rb'))

                """
                Concatenate the several folders
                """
                    
                filePath = mainPath + folder + '/time.dat'
                evoked = pickle.load( open(mainPath + folder + '/probe_{0}_group_{1}/probe_{0}_group_{1}_evoked.pickle'.format(probe, group), 'rb') )     
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
                analyzed_path_for_group = analyzed_path_for_folder + '/probe_{:g}_group_{:g}'.format(probe,group)
                if not os.path.exists(analyzed_path_for_group):
                    os.mkdir(analyzed_path_for_group)

                evoked_pdf_path = analyzed_path_for_group + '/evoked_pdf_format/' #for saving images in pdf format
                if not os.path.exists(evoked_pdf_path):
                    os.mkdir(evoked_pdf_path)

                evoked_svg_path = analyzed_path_for_group + '/evoked_svg_format/' # for saving images in svg
                if not os.path.exists(evoked_svg_path):
                    os.mkdir(evoked_svg_path)

                evoked = evoked_data
                evoked_avg = np.mean(evoked,0)
                evoked_std = np.std(evoked,0)
                evoked_err = evoked_std / math.sqrt(len(evoked))
                peak_info[probe]['peak_amps'][group*p['nr_of_electrodes_per_group']:(group+1)*p['nr_of_electrodes_per_group']] = np.min(evoked_avg,1)
                time = np.linspace(-p['evoked_pre']*1000, p['evoked_post']*1000, (p['evoked_post'] + p['evoked_pre']) * p['sample_rate'])
                    
                if p['probe_type']== 'linear':
                #Make a plot for electrode on the shank
                    figure()
                    y = np.linspace(p['bottom_ycoord'],p['top_ycoord'],p['nr_of_electrodes_per_group'])
                    pcolor(time,y,evoked_avg)
                    colorbar()
                    xlabel('Time (ms)')
                    ylabel('Height from tip (um)')
                    savefig(evoked_svg_path + 'probe_{:g}_group_{:g}_evoked.svg'.format(probe,group),format = 'svg') 
                    savefig(evoked_pdf_path + 'probe_{:g}_group_{:g}_evoked.pdf'.format(probe,group),format = 'pdf') 
                    close()

                #Calculate the peak parameters for each electrode and plot the average LFP waveform for each electrode
                for trode in range(p['nr_of_electrodes_per_group']):
                    real_trode = p['nr_of_electrodes_per_group']*group + trode
                    try:
                        peak_info[probe]['peak_locs'][real_trode] = np.where(evoked_avg[trode] == peak_info[probe]['peak_amps'][real_trode])[0]
                        peak_info[probe]['peak_stds'][real_trode] = evoked_std[trode][int(peak_info[probe]['peak_locs'][real_trode])]
                        peak_info[probe]['peak_errs'][real_trode] = evoked_err[trode][int(peak_info[probe]['peak_locs'][real_trode])]
                        peak_info[probe]['peak_times'][real_trode] = (peak_info[probe]['peak_locs'][real_trode] - p['evoked_pre']) / p['sample_rate']      

                    except ValueError:
                        pass
                    figure()
                    plot(time,evoked_avg[trode], 'k-')
                    fill_between(time, evoked_avg[trode] - evoked_err[trode] , evoked_avg[trode] + evoked_err[trode])    
                    xlabel('Time (ms)')
                    ylabel('voltage (uV)')
                    ylim_min = np.floor(np.min(evoked) / 100) * 70                         
                    ylim_max = np.ceil(np.max(evoked) /100) *70
                    ylim(ylim_min, ylim_max)
                    xlim_min = -p['evoked_pre']*1000
                    xlim_max = p['evoked_post']*1000
                    xlim(xlim_min,xlim_max)
                    y_ticks = np.arange((ylim_min - ylim_min%50), (ylim_max - ylim_max%50), 50)
                    x_ticks = np.arange(xlim_min,xlim_max,5)
                    axes().set_xticks(x_ticks, minor = True)
                    axes().set_yticks(y_ticks, minor = True)
                    axes().grid(which='both')
                    savefig(evoked_svg_path + 'electrode{:g}_evoked.svg'.format(trode), format = 'svg' )
                    savefig(evoked_pdf_path + 'electrode{:g}_evoked.pdf'.format(trode), format = 'pdf' )
                    close()    

        #Saving the peak parameters for each recording session into an excel file
        df0 = DataFrame({'Peak amplitudes': peak_info[0]['peak_amps'], 'Peak std': peak_info[0]['peak_stds'], 'Peak standard errors': peak_info[0]['peak_errs'], 'Peak times': peak_info[0]['peak_times']})
        df0.to_excel(writer_0, sheet_name = folder[0:30])
        if p['probes']==2:
            df1 = DataFrame({'Peak amplitudes': peak_info[1]['peak_amps'], 'Peak std': peak_info[1]['peak_stds'], 'Peak standard errors': peak_info[1]['peak_errs'], 'Peak times': peak_info[1]['peak_times']})    
            df1.to_excel(writer_1, sheet_name = folder[0:30]) 

writer_0.save()
if p['probes']==2:
    writer_1.save()
print('Done!')
