#!/usr/bin/env python
# -*- coding:utf-8 -*-

# exter
import pandas as pd
import numpy as np
import voxelmorph as vxm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import datetime
import argparse
import os

# import from src
from src.generate_img_via_registration import generate_img_via_registration, calcu_extreme_point
from src.my_resample import my_crop

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--xlxs_path', help='filename and path (xlxs) of dataset')
parser.add_argument('--data_path', help='path of data ')
parser.add_argument('--criteria', type=str, default='SSIM', help='specify which criterion is used, can be: SSIM, NCC, PSNR, DSC, MAE, Forbenius, MEAN, default=SSIM')
parser.add_argument('--gpu', default=None, help='GPU ID numbers (default: None)')
parser.add_argument('--sheet_name', default='train_sessions', help='test_sessions or train_sessions')

args = parser.parse_args()

timestamp= '{:%Y-%m-%d_%H_%M_%S}'.format(datetime.datetime.now())

# load data from xlxs
path_database_xlxs = args.xlxs_path
path_of_img_seg = args.data_path
criteria = args.criteria
print('Path of data: {}, corresponding xlxs: {}'.format(path_of_img_seg, path_database_xlxs))
print('Criteria in use: {}'.format(criteria))

def load_metadata(path_database_xlxs, sheet_name = 'test_sessions'):
    df = pd.read_excel(path_database_xlxs, sheet_name=sheet_name)
    img_lists = df["Output collection GUID"].to_numpy()
    age_true = df["age from gustav"].to_numpy()
    ids = df["Individual's ID"].to_numpy()
    return img_lists, ids, age_true

def construct_registration_pair(subj, ids, img_lists,age_true):
    idexs_subj = ids == subj
    subj_list = img_lists[idexs_subj]
    subj_age = age_true[idexs_subj]
    subj_age, subj_list = zip(*sorted(zip(subj_age,subj_list)))
    gap = subj_age[-1]-subj_age[0]
    nb_synth = int(np.ceil(2*gap)) # number of generated images, decided by age gap between youngest and oldest brain

    # return a dictionary for subject
    sub = {}
    sub['id'] = subj
    sub['list'] = subj_list
    sub['true age'] = subj_age
    sub['age diff'] = gap
    return nb_synth, sub

# main
path_of_log = path_of_img_seg + criteria + '_crop/'
if not os.path.exists(path_of_log):
    os.makedirs(path_of_log)
    print('Creating path_of_log' , path_of_log)
sys.stdout  = open(path_of_log + timestamp +'logs.txt','w')
img_lists, ids, age_true = load_metadata(path_database_xlxs, sheet_name = args.sheet_name)
Subjs = [x for x in set(ids) if str(x)!='nan']

for subj in Subjs:
    nb_synth, sub = construct_registration_pair(subj, ids, img_lists, age_true)
    sub['criteria'] = criteria
    print('For subject {}, gap between youngest and oldest is {}, so {} images will be generated'.format(sub['id'], sub['age diff'],nb_synth))
    sub, gen_sq, gen_sq_seg = generate_img_via_registration(sub, nb_synth,
                              path_of_img_seg, criteria, save=True, gpu=args.gpu)

    # calculate extreme point for every img between youngest and oldest
    extreme_point_sub = []
    if len(sub['list']) >2:
        for count, mid_img_id in enumerate(sub['list'][1:-1]):
            # load data
            if os.path.exists(os.path.join(path_of_img_seg+ mid_img_id + crop_img)) == False:
                my_crop(os.path.join(path_of_img_seg + mid_img_id + '/mri/'))
            true_img_path = path_of_img_seg+ mid_img_id+ 'resampled_rescaled_align_norm.nii.gz'
            true_seg_path = path_of_img_seg+ mid_img_id+ 'resampled_align_aparc+aseg.nii.gz'
            true_img, _ = vxm.py.utils.load_volfile(true_img_path, add_batch_axis=True, add_feat_axis=False,
                                                            ret_affine=True)
            true_np = true_img.squeeze()
            true_seg = vxm.py.utils.load_volfile(true_seg_path, add_batch_axis=True, add_feat_axis=False)
            true_seg_np = true_seg.squeeze()
            extreme_point_sub.append(calcu_extreme_point(gen_sq, true_np, criteria, sub['true age'], sub['age diff']/nb_synth, gen_sq_seg=gen_sq_seg, true_seg=true_seg_np))

        # calcu and fit for every subject
        extreme_point_sub = np.array(extreme_point_sub)
        true_age_sub=np.array(sub['true age'][1:-1])
        sub['linear age'] = np.concatenate(([0],extreme_point_sub,[0]))

    # Save for every subject
    csv_name = path_of_img_seg + criteria + '_crop/'+ subj + '.csv'
    print('Saving to {} file'.format(csv_name))
    pd.DataFrame(sub).to_csv(csv_name, index=False)
    print('---------------------------------------------------------------------')

sys.stdout.close()
