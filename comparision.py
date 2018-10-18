import nibabel as nib
import numpy as np
import time
import csv
import os

start = time.clock()

def file_name(file_dir):
    filelist = []
    for file in os.listdir(file_dir):
        filelist.append(file)
    return filelist

def comparision(project_name, file_name):
    #Load mask.nii
    mask_file = nib.load('aal2_3mm_o.nii')
    mask_data = mask_file.get_data()
    roi_list = np.unique(mask_data)#Read area
    roi_list = roi_list[1:]#Extract effective value
    n_roi = len(roi_list)# nrio = 120;area = 120

    #Load objective rest.nii
    rest_file = nib.load('/home/fsluser/Documents/{}/{}/filtered_func_data_flirt.nii.gz'.format(project_name, file_name))
    image = rest_file.get_data() #image = 61*73*61*185
    n_ts = image.shape[3] # nts = 185;scanning time = 185

    #mask3d -- 4d
    mask_4d = np.repeat(mask_data[:, :, :, np.newaxis], n_ts, axis=3)

    #Extraction data
    roi_ts = np.zeros((n_ts, n_roi))# roi_ts = 185*120; Create a target list

    #Calculate the value of each region of the brain
    for idx, roi in enumerate(roi_list):
        print('roi = {}'.format(roi))
        #Extraction region location extract = 61*73*61
        img_cpy = np.copy(image)
        img_cpy[mask_4d!=roi] = np.nan
        roi_ts[:, idx] = np.nanmean(img_cpy, axis=(0,1,2))
    with open('/home/fsluser/Documents/{}/{}/rest_image.csv'.format(project_name, file_name), 'w') as myfile:
        data = csv.writer(myfile)
        data.writerows(roi_ts)
'''
#for example 
project_namelist = file_name('/home/fsluser/Documents')
print(project_namelist)
for project_name in project_namelist:
    file_namelist = file_name('/home/fsluser/Documents/{}'.format(project_name))
    print(project_name, file_namelist)
    print(len(file_namelist), len(project_namelist))
    for name in file_namelist:
        print(project_name, name)
        comparision(project_name, name)
'''

end4 = time.clock()
print('Total Running time:{}'.format(end4 - start))


