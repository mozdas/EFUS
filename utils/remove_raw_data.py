"""
    This functions removes the raw and the intermediate data files that are created and required for initial analyses.
    It cleans 'analysis_files' and '[data_folders]' folders and left several files for further analyses.
    WARNING : Make sure that you finalize Evoked LFP Analysis and Spike Sorting (Automated and Manuel via Klusta) on this main path. 
              Otherwise, this script will erase the required files for Evoked LFP Analysis and remove the clustering outputs of Klusta.
              To rerun Evoked LFP Analysis and Spike Sorting one must use raw data and start from the beginnig for analyses.
    This script will left required files for Automated Window LFP and PSTH algorithms for further analysis.
    After executing this script on a folder, Automated Window LFP and PSTH algorithms can be applied to this folder.


    Created on Friday, July 6th, 2018
    Author: Abdulkadir Gokce - Please contact him or Mehmet Ozdas in case of any questions.
"""

import os
import shutil

main_path = '/media/yaniklab/05d01d78-2bd6-4a4e-b573-df49ccacb71c/2018_04_12_FUSs1_EphysM1_E-FUS_NBBB67/'
#def remove_raw_data(main_path):

dirs = os.listdir(main_path)
directories_to_skip = ['analyzed', 'other', 'log.txt', 'notes.txt', '.DS_Store', '._.DS_Store'] #Folder that needs to be intact
for folder in (folder for folder in dirs if(folder not in directories_to_skip) ):

    if (folder == 'analysis_files'):
        analysis_files_path = main_path + folder + '/'
        analysis_files_dirs = os.listdir(analysis_files_path)

        for analysis_folder_for_group in analysis_files_dirs:
            analysis_files_group_path = analysis_files_path + analysis_folder_for_group + '/'
            if(os.path.isdir(analysis_files_group_path)): #Only iterate through folders of analysis files for groups, if there is a paramsDict.p, skipped it
                analysis_files_group_dirs = os.listdir(analysis_files_group_path)
                #Keeping the required files for PSTH Analysis
                files_to_keep = [analysis_folder_for_group+'.clu.0', analysis_folder_for_group+'.kwik', analysis_folder_for_group+'.kwx', analysis_folder_for_group+'_spikeinfo.pickle']        

                for file_to_remove in analysis_files_group_dirs:
                    if(file_to_remove not in files_to_keep):
                        file_to_remove_path = analysis_files_group_path + file_to_remove
                        if(os.path.isdir(file_to_remove_path)):#If it is a directory
                            shutil.rmtree(file_to_remove_path)
                        else:#If it is a file
                            os.remove(file_to_remove_path)

    else:
        raw_data_path = main_path + folder + '/'
        raw_data_dirs = os.listdir(raw_data_path)
        #Keeping the required files for Automated Window LFP and PSTH Analyses
        files_to_keep = ['info.rhd', 'paramsDict.p', 'time.dat']
        
        for file_to_remove in raw_data_dirs:
            if(file_to_remove not in files_to_keep): #To keep evoked pickle files, skipped the forder for each group that contains this pickle files
                file_to_remove_path = raw_data_path + file_to_remove
                if(os.path.isfile(file_to_remove_path)):
                    os.remove(file_to_remove_path)
