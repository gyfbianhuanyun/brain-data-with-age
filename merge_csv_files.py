#Merge Documents
import pandas as pd

#Read the underlying file
brain_csv = pd.read_csv('/home/fsluser/PycharmProjects/fsl/entropy_project/project_SALD_entropy.csv')
#Read the target file
project_entropy = pd.read_csv('/home/fsluser/PycharmProjects/fsl/entropy_project/project_SALD_sum.csv')
#Merge files based on project name and subject name
data = pd.merge(brain_csv, project_entropy, on=['subject', 'project'], how='left')
#Delete the first column
data.drop(columns='Unnamed: 0', inplace=True)
#Save
data.to_csv(r'/home/fsluser/PycharmProjects/fsl/entropy_project/project_SALD_entropy.csv')
#View the results
print(data)
