import numpy as np
import scipy.stats as sci
import time
import pandas as pd
import os


start = time.clock()


def file_name(file_dir):
    filelist = []
    for file in os.listdir(file_dir):
        filelist.append(file)
    return filelist


def brain_entropy(project_name, subject_name):
    # Use csv data
    brain_region = pd.read_csv('/home/fsluser/PycharmProjects'
                               '/fsl/AAL2_brain region.csv')
    name = brain_region.anatomical_name
    rest_image = pd.read_csv('/home/fsluser/Documents/{}/{}/'
                             'rest_image.csv'.format(project_name,
                                                     subject_name, names=name))

    # Extracting cerebrum data
    cerebrum_region = len(brain_region[brain_region.location == 'cerebrum'])
    cerebrum_data = rest_image.ix[:, :cerebrum_region]

    # Extract the data to find the correlation coefficient
    correlate_list = pd.DataFrame.corr(cerebrum_data)
    correlate_list = correlate_list.values
    tri_idx = np.ones((cerebrum_region, cerebrum_region))
    correlate_list = correlate_list[np.tril(tri_idx) == 0]

    # obtain frequency
    hist_list = np.histogram(correlate_list, bins=20, range=(-1, 1))

    # Calculate the probability
    p = hist_list[0].astype(float)/np.sum(hist_list[0])

    # Calculate the entropy
    entropy = sci.entropy(p, base=2)
    print(entropy)
    return entropy


'''
# for example

project_namelist = file_name('/home/fsluser/Documents')
print(project_namelist)
num = 0
entropy_list = []
subject_list = []
project_list = []
for project_name in project_namelist:
    project_name = 'SALD'
    file_namelist = file_name(f'/home/fsluser/Documents/{project_name}')
    print(project_name, file_namelist)
    print(len(file_namelist), len(project_namelist))
    for name in file_namelist:
        print(project_name, name)
        num = num + 1
        entropy = brain_entropy(project_name, name)
        entropy_list.append(entropy)
        subject_list.append(name)
        project_list.append(project_name)
entropy_project = pd.DataFrame({'project': project_list,
                                'subject': subject_list,
                                'entropy': entropy_list})
entropy_project.to_csv('/home/fsluser/PycharmProjects/fsl
                        /entropy_project/project_SALD_entropy.csv',
                        index=False, sep=',')
'''

end = time.clock()
print(f'Total Running time:{end - start}')
