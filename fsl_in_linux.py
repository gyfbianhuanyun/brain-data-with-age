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
    with open('/home/fsluser/Documents/{}/{}/func.feat/design.fsf'.format(project_name, first_file)) as file_o, \
            open('/home/fsluser/Documents/{}/{}/design.fsf'.format(project_name, file_name), 'w') as file_1:
        for line in file_o:
            if '/home/fsluser/Documents/{}/{}/func'.format(project_name, first_file) in line:
                line = line.replace('/home/fsluser/Documents/{}/{}/func'.format(project_name, first_file),
                                    '/home/fsluser/Documents/{}/{}/func'.format(project_name, file_name))
            if '{}_task-rest_bold'.format(first_file) in line:
                line = line.replace('{}_task-rest_bold'.format(first_file),
                                    '{}_task-rest_bold'.format(file_name))
            file_1.write(line)

#run feat preprocess
def feat_o(project_name, file_name):
    feat = fsl.FEAT()
    feat.inputs.fsf_file = '/home/fsluser/Documents/{}/{}/design.fsf'.format(project_name, file_name)
    feat.run()

#run 4D registration
def regis(project_name, file_name):
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = '/home/fsluser/Documents/{}/{}/func.feat/filtered_func_data.nii.gz'.format(project_name, file_name)
    applyxfm.inputs.in_matrix_file = '/home/fsluser/Documents/{}/{}/func.feat/reg/example_func2standard.mat'.format(project_name, file_name)
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
                    '/home/fsluser/Documents/{}/{}/filtered_func_data_flirt.nii.gz'.format(project_name, subject_name))

'''
#for example

project_namelist = file_name('/home/fsluser/Documents')
print(project_namelist)
for i in range(0, 30):
    project_name = project_namelist[i]
    file_namelist = file_name('/home/fsluser/Documents/{}'.format(project_name))
    print(project_name, file_namelist)
    print(len(file_namelist), len(project_namelist))
    first_name = file_namelist[0]
    func_fsl(project_name, file_namelist, first_name)
'''

print('end')
