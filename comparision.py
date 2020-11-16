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
    # Load mask.nii
    mask_file = nib.load('aal2_3mm_o.nii')
    mask_data = mask_file.get_data()
    # Read area
    roi_list = np.unique(mask_data)
    # Extract effective value
    roi_list = roi_list[1:]
    # nrio = 120;area = 120
    n_roi = len(roi_list)

    # Load objective rest.nii
    rest_file = nib.load(
        f'/home/fsluser/Documents/{project_name}'
        f'/{file_name}/filtered_func_data_flirt.nii.gz')
    image = rest_file.get_data()
    # nts = 185;scanning time = 185
    n_ts = image.shape[3]

    # mask3d -- 4d
    mask_4d = np.repeat(mask_data[:, :, :, np.newaxis], n_ts, axis=3)

    # Extraction data
    # roi_ts = 185*120; Create a target list
    roi_ts = np.zeros((n_ts, n_roi))

    # Calculate the value of each region of the brain
    for idx, roi in enumerate(roi_list):
        print(f'roi = {roi}')
        # Extraction region location extract = 61*73*61
        img_cpy = np.copy(image)
        img_cpy[mask_4d != roi] = np.nan
        roi_ts[:, idx] = np.nanmean(img_cpy, axis=(0, 1, 2))
    with open(f'/home/fsluser/Documents/{project_name}'
              f'/{file_name}/rest_image.csv', 'w') as myfile:
        data = csv.writer(myfile)
        data.writerows(roi_ts)


'''
# for example
project_namelist = file_name('/home/fsluser/Documents')
print(project_namelist)
for project_name in project_namelist:
    file_namelist = file_name(f'/home/fsluser/Documents/{project_name}')
    print(project_name, file_namelist)
    print(len(file_namelist), len(project_namelist))
    for name in file_namelist:
        print(project_name, name)
        comparision(project_name, name)
'''

end4 = time.clock()
print(f'Total Running time:{end4 - start}')
