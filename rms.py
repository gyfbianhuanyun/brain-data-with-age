import numpy as np
import scipy.stats as sci
import time
import pandas as pd
import os
import csv

start = time.clock()

def file_name(file_dir):
    filelist = []
    for file in os.listdir(file_dir):
        filelist.append(file)
    return filelist

def brain_rms(project_name, subject_name):
    #Use csv data
    brain_region = pd.read_csv('/home/fsluser/PycharmProjects/fsl/AAL2_brain region.csv')
    name = brain_region.anatomical_name
    rest_image = pd.read_csv('/home/fsluser/Documents/{}/{}/rest_image.csv'.format(project_name, subject_name))

    #Extracting cerebrum data
    cerebrum_region = len(brain_region[brain_region.location =='cerebrum'])
    cerebrum_data = rest_image.ix[:, :cerebrum_region]

    correlate_list = []

    #Extract the data to find the correlation coefficient
    correlate_list = pd.DataFrame.corr(cerebrum_data)
    correlate_list = correlate_list.values
    tri_idx = np.ones((cerebrum_region, cerebrum_region))
    correlate_list = correlate_list[np.tril(tri_idx)==0]

    # obtain RMS(root mean square)
    rms = np.sqrt(np.mean(correlate_list ** 2))

    print(rms)
    return rms
'''
# for example
project_namelist = file_name('/home/fsluser/Documents')
print(project_namelist)
rms_list = []
subject_list = []
project_list = []
for project_name in project_namelist:
    file_namelist = file_name('/home/fsluser/Documents/{}'.format(project_name))
    print(project_name, file_namelist)
    print(len(file_namelist), len(project_namelist))
    for name in file_namelist:
        print(project_name, name)
        rms = brain_rms(project_name, name)
        rms_list.append(rms)
        subject_list.append(name)
        project_list.append(project_name)
rms_project = pd.DataFrame({'project': project_list, 'subject': subject_list, 'rms': rms_list})
rms_project.to_csv('/home/fsluser/PycharmProjects/fsl/project_rms.csv', index=False, sep=',')
'''



end = time.clock()
print('Total Running time:{}'.format(end - start))
