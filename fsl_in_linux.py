import os
import nipype.interfaces.fsl as fsl
import shutil

#read file name & put it in the list
def file_name(file_dir):
    filelist = []
    for file in os.listdir(file_dir):
        filelist.append(file)
    return filelist

#fsf content change
#Because the file names are the same, the path is detailed.
#Change the path according to the actual situation
def file_wr(project_name, file_name, first_file):
    with open(f'/home/fsluser/Documents/{project_name}/{first_file}/func.feat/design.fsf') as file_o, \
            open(f'/home/fsluser/Documents/{project_name}/{file_name}/design.fsf', 'w') as file_1:
        for line in file_o:
            if f'/home/fsluser/Documents/{project_name}/{first_file}/func' in line:
                line = line.replace(f'/home/fsluser/Documents/{project_name}/{first_file}/func',
                                    f'/home/fsluser/Documents/{project_name}/{file_name}/func')
            if f'{first_file}_task-rest_bold' in line:
                line = line.replace(f'{first_file}_task-rest_bold',
                                    f'{file_name}_task-rest_bold')
            file_1.write(line)

#run feat preprocess
def feat_o(project_name, file_name):
    feat = fsl.FEAT()
    feat.inputs.fsf_file = f'/home/fsluser/Documents/{project_name}/{file_name}/design.fsf'
    feat.run()

#run 4D registration
def regis(project_name, file_name):
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = f'/home/fsluser/Documents/{project_name}/{file_name}/func.feat/filtered_func_data.nii.gz'
    applyxfm.inputs.in_matrix_file = f'/home/fsluser/Documents/{project_name}/{file_name}/func.feat/reg/example_func2standard.mat'
    applyxfm.inputs.reference = 'MNI152_T1_3mm_brain.nii.gz'
    applyxfm.inputs.apply_xfm = True
    applyxfm.run()

#run func
def func_fsl(project_name, file_namelist, first_name):
    for j in range(478, 493):
        subject_name = file_namelist[j]
        print(project_name, subject_name, j)
        print('step 1 write a .fsf')
        file_wr(project_name, subject_name, first_name)
        print('step 2 run FSL')
        feat_o(project_name, subject_name)
        print('step 3 run 4D registration')
        regis(project_name, subject_name)
        print('step 4 move the file to the target directory ')
        shutil.move('/home/fsluser/PycharmProjects/fsl/filtered_func_data_flirt.nii.gz',
                    f'/home/fsluser/Documents/{project_name}/{subject_name}/filtered_func_data_flirt.nii.gz')

'''
#for example

project_namelist = file_name('/home/fsluser/Documents')
print(project_namelist)
for i in range(0, 30):
    project_name = project_namelist[i]
    file_namelist = file_name(f'/home/fsluser/Documents/{project_name}')
    print(project_name, file_namelist)
    print(len(file_namelist), len(project_namelist))
    first_name = file_namelist[0]
    func_fsl(project_name, file_namelist, first_name)
'''

print('end')
