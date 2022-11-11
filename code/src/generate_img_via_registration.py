#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import neurite as ne
import tensorflow.keras.backend as K
#import tensorflow_scientific as tfs
from voxelmorph.tf import tensorflow_scientific as tfs
import os
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from scipy.stats.mstats import gmean

import src.my_resample as my_resample


def DSC_loss(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def compute_label_dice(gt, pred):
    cls_lst = np.unique(gt)
    cls_lst = cls_lst[cls_lst!=0]
    dice_lst = []
    for cls in cls_lst:
        dice = DSC_loss(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

def NCC(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    #return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))
    mean_data0 = np.mean(data0)
    mean_data1 = np.mean(data1)
    up = np.sum((data0-mean_data0)*(data1-mean_data1))
    down = np.sqrt(np.sum(np.power(data0-mean_data0,2)))*np.sqrt(np.sum(np.power(data1-mean_data1,2)))
    return up/down


def my_integrate_vec(inputs, method = 'ss', int_steps = 7, out_time_pt = 10, interp_method = 'linear'):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    loc_shift = inputs[0]

    # necessary for multi_gpu models...
    if hasattr(inputs[0], '_keras_shape'):
        loc_shift._keras_shape = inputs[0]._keras_shape

    # map transform across batch
    vel = inputs[0].squeeze()
    vel = tf.convert_to_tensor(vel)
    out = raw_integrate_vec(vel,
                         method=method,
                         nb_steps=int_steps,
                         out_time_pt=out_time_pt,
                         ode_args={'rtol': 1e-6, 'atol': 1e-12},
                         odeint_fn= None,
                         interp_method =  interp_method) # interp_method (default:'linear'): 'linear', 'nearest'
    if out.ndim <=4:
        out = tf.expand_dims(out, 0) # add first dimension(batch) to make sure it can be used in the next step
    if hasattr(inputs[0], '_keras_shape'):
        out._keras_shape = inputs[0]._keras_shape
    return out

def raw_integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        out_time_pt = kwargs['out_time_pt']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep: # False, not change
            svec = K.permute_dimensions(vec, [-1, *range(0, vec.shape[-1] - 1)])
            assert 2 ** nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

            svec = svec / (2 ** nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + tf.map_fn(my_transform, svec[1::2, :], svec[0::2, :])

            disp = svec[0, :]
        else:
            if out_time_pt == 0:
                # pass
                vec = vec * 0
            else:
                vec = vec / (2 ** nb_steps)
                for _ in range(out_time_pt):
                    vec += my_transform(vec, vec)
            disp = vec
    else:
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda disp, _: my_transform(vec, disp, interp_method = kwargs['interp_method'])

        # process time point.
        out_time_pt = kwargs['out_time_pt'] if 'out_time_pt' in kwargs.keys() else 1
        out_time_pt = tf.cast(K.flatten(out_time_pt), tf.float32)
        len_out_time_pt = out_time_pt.get_shape().as_list()[0]
        assert len_out_time_pt is not None, 'len_out_time_pt is None :('
        z = out_time_pt[0:1]*0.0  # initializing with something like tf.zeros(1) gives a control flow issue.
        K_out_time_pt = K.concatenate([z, out_time_pt], 0)

        # enable a new integration function than tf.contrib.integrate.odeint
        odeint_fn = tfs.integrate.odes.odeint
        if 'odeint_fn' in kwargs.keys() and kwargs['odeint_fn'] is not None:
            odeint_fn = kwargs['odeint_fn']

        # process initialization
        if 'init' not in kwargs.keys() or kwargs['init'] == 'zero':
            disp0 = vec*0  # initial displacement is 0
        else:
            raise ValueError('non-zero init for ode method not implemented')

        # compute integration with odeint
        if 'ode_args' not in kwargs.keys():
            kwargs['ode_args'] = {}
        disp = odeint_fn(fn, disp0, K_out_time_pt, **kwargs['ode_args'])
    return disp

def my_transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes

    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)

def apply_ode(flow_output, moving, nb_gen=20, stopping_point = 2, segmentations = False, **kwargs):
    # using ode to calculate vecint layer=========
    out_time_pt = np.linspace(0, stopping_point, nb_gen+1)
    out_time_pt = out_time_pt[1:]

    pos_flow = my_integrate_vec(flow_output, method='ode', out_time_pt=out_time_pt, interp_method='linear')  # interp_method (default:'linear'): 'linear', 'nearest'
    pos_flow = vxm.networks.layers.RescaleTransform(2, name='my_diffflow')(pos_flow)

    sources = tf.convert_to_tensor(np.repeat(moving, len(out_time_pt) + 1, 0))
    input_samples = [sources, pos_flow]
    y_source = vxm.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='transformer')(input_samples)
    y_source = y_source.numpy()
    y_source_sq = y_source.squeeze()

    if segmentations:
        moving_seg = kwargs['moving_seg']
        sources_seg = tf.convert_to_tensor(np.repeat(moving_seg, len(out_time_pt) + 1, 0), dtype=float)
        input_samples_seg = [sources_seg, pos_flow]
        y_source_seg = vxm.layers.SpatialTransformer(interp_method='nearest', indexing='ij', name='transformer')(input_samples_seg)  # for segmentation
        y_source_seg = y_source_seg.numpy()
        y_source_sq_seg = y_source_seg.squeeze()
    else:
        y_source_sq_seg = []

    return y_source_sq, y_source_sq_seg, pos_flow

def apply_ss(flow_output, moving, iteration = 7, stopping_point = 2, segmentations = False, **kwargs):
    # using ode to calculate vecint layer=======
    pos_flow = my_integrate_vec(flow_output, method='ss', out_time_pt=0, interp_method='linear')
    iteration =iteration+1 # add one
    for out_time_pt in range(1,iteration):
        pos_flow = tf.concat([pos_flow, my_integrate_vec(flow_output, method='ss', out_time_pt=out_time_pt, interp_method='linear')],0)  # interp_method (default:'linear'): 'linear', 'nearest'

    pos_flow = vxm.networks.layers.RescaleTransform(2, name='my_diffflow')(pos_flow) # to the same solution of input

    sources = tf.convert_to_tensor(np.repeat(moving, iteration, 0))
    input_samples = [sources, pos_flow]
    y_source = vxm.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='transformer')(input_samples)
    y_source = y_source.numpy()
    y_source_sq = y_source.squeeze()

    if segmentations:
        moving_seg = kwargs['moving_seg']
        sources_seg = tf.convert_to_tensor(np.repeat(moving_seg, iteration, 0), dtype=float)
        input_samples_seg = [sources_seg, pos_flow]
        y_source_seg = vxm.layers.SpatialTransformer(interp_method='nearest', indexing='ij', name='transformer')(input_samples_seg)  # for segmentation
        y_source_seg = y_source_seg.numpy()
        y_source_sq_seg = y_source_seg.squeeze()
    else:
        y_source_sq_seg = []

    return y_source_sq, y_source_sq_seg, pos_flow

def save_generated_imgs(y_source_sq, fixed_affine, save_path, save = False, show = False, **kwargs):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Creating save_path', save_path)

    y_source_sq_seg = kwargs['y_source_sq_seg']
    for i in range(y_source_sq.shape[0]):
        template = y_source_sq[i, ...]
        # plot and save png
        mid_slices_moving = [np.take(template, template.shape[d] // 2, axis=d) for d in range(3)]
        mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
        mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)
        ne.plot.slices(mid_slices_moving, titles=[str(i)], cmaps=['gray'], do_colorbars=True, grid=[1, 3],
                       show=show,
                       save=save, save_name=save_path + '/generated_img' + str(i) + '.png');

        # save nii.gz
        save_name = save_path + '/generated_img' + str(i) + '.nii.gz'
        vxm.py.utils.save_volfile(template.squeeze(), save_name, fixed_affine)

        segmentations = kwargs['segmentations']
        if segmentations:
            # for segmentationn
            template_seg = y_source_sq_seg[i, ...]
            # plot and save png
            seg_slice = [np.take(template_seg, template_seg.shape[d] // 2, axis=d) for d in range(3)]
            seg_slice[1] = np.rot90(seg_slice[1], 1)
            seg_slice[2] = np.rot90(seg_slice[2], -1)
            ne.plot.slices(seg_slice, titles=[str(i)], cmaps=['gray'], do_colorbars=True, grid=[1, 3], show=show,
                           save=save, save_name=save_path + '/generated_seg' + str(i) + '.png');

            # save nii.gz
            save_name = save_path + '/generated_seg' + str(i) + '.nii.gz'
            vxm.py.utils.save_volfile(template_seg.squeeze(), save_name)

def apply_6_creteria(gen_sq, true_img, **kwargs):
    number_generated_imgs = gen_sq.shape[0]
    dice = []
    forbenius = []
    mae = []
    psnr = []
    ssim = []
    ncc = []
    for i in range(number_generated_imgs):  # exclude first point
        gen_img = gen_sq[i, ...]

        # if segmentation
        true_seg = kwargs['true_seg']
        gen_sq_seg = kwargs['gen_sq_seg']
        gen_seg = gen_sq_seg[i, ...]

        # 0. calculate dice
        dice_sg = compute_label_dice(true_seg, gen_seg)
        dice.append(dice_sg)

        # different criteria
        # 1. forbenius on the difference image
        dif_img = true_img - gen_img
        forbenius_sg = np.linalg.norm(dif_img)
        forbenius.append(forbenius_sg)

        # 2. MAE: mean absolute error
        mae_sg = np.mean(np.abs(dif_img))
        mae.append(mae_sg)

        # 3. PSNR: peak signal-to-noise
        if PSNR(true_img, gen_img) == float("inf"):
            psnr_sg = 100.0
            psnr.append(psnr_sg)
        else:
            psnr_sg = PSNR(true_img, gen_img)
            psnr.append(psnr_sg)

        # 4. SSIM: structural similarity index
        ssim_sg = SSIM(true_img, gen_img)
        ssim.append(ssim_sg)

        # # 5. NCC: normalized cross correlation
        ncc_sg = NCC(true_img, gen_img)
        ncc.append(ncc_sg)

    return [forbenius,mae,psnr,ssim,ncc,dice]

def apply_creteria(gen_sq, true_img, **kwargs):
    number_generated_imgs = gen_sq.shape[0]
    dice = []
    forbenius = []
    mae = []
    psnr = []
    ssim = []
    ncc = []
    g_mean = []
    for i in range(number_generated_imgs):  # exclude first point
        gen_img = gen_sq[i, ...]

        # if segmentation
        true_seg = kwargs['true_seg']
        gen_sq_seg = kwargs['gen_sq_seg']
        gen_seg = gen_sq_seg[i, ...]

        # 0. calculate dice
        dice_sg = compute_label_dice(true_seg, gen_seg)
        dice.append(dice_sg)

        # different criteria
        # 1. forbenius on the difference image
        dif_img = true_img - gen_img
        forbenius_sg = np.linalg.norm(dif_img)
        forbenius.append(forbenius_sg)

        # 2. MAE: mean absolute error
        mae_sg = np.mean(np.abs(dif_img))
        mae.append(mae_sg)

        # 3. PSNR: peak signal-to-noise
        if PSNR(true_img, gen_img) == float("inf"):
            psnr_sg = 100.0
            psnr.append(psnr_sg)
        else:
            psnr_sg = PSNR(true_img, gen_img)
            psnr.append(psnr_sg)

        # 4. SSIM: structural similarity index
        ssim_sg = SSIM(true_img, gen_img)
        ssim.append(ssim_sg)

        # # 5. NCC: normalized cross correlation
        ncc_sg = NCC(true_img, gen_img)
        ncc.append(ncc_sg)

        # gmean
        # g_mean.append(gmean([dice_sg, forbenius_sg,mae_sg, psnr_sg, ssim_sg, ncc_sg])) # gmean with dice
        g_mean.append(gmean([forbenius_sg, mae_sg, psnr_sg, ssim_sg, ncc_sg]))  # gmean without dice

    # wrap up criteria
    criteria = {}
    criteria['DSC'] = dice
    criteria['Forbenius'] = forbenius
    criteria['MAE'] = mae
    criteria['PSNR'] = psnr
    criteria['SSIM'] = ssim
    criteria['NCC'] = ncc
    criteria['GMEAN'] = g_mean
    return criteria

def calcu_stopping_point(gen_sq, true_img, criteria, **kwargs):
    true_seg = kwargs['true_seg']
    gen_sq_seg = kwargs['gen_sq_seg']
    cri = apply_creteria(gen_sq, true_img,gen_sq_seg=gen_sq_seg,true_seg=true_seg)
    # using single critrion to find extrame point
    if criteria in ['SSIM','NCC','PSNR','DSC']:
        stopping_point = np.argmax(cri[criteria])
    elif criteria in ['MAE','Forbenius', 'GMEAN']:
        stopping_point = np.argmin(cri[criteria])
    elif criteria in ['MEAN']:
        extrem_point = []
        for crit in ['SSIM','NCC','PSNR','DSC']:
            extrem_point.append(np.argmax(cri[crit]))
        for crit in ['MAE', 'Forbenius']:
            extrem_point.append(np.argmin(cri[crit]))
        cri_mean = np.mean(extrem_point)
        stopping_point = np.floor(cri_mean)
    else:
        print('criterion not exist!')

    return stopping_point*0.1

def calcu_extreme_point(gen_sq, true_img, criteria, ages, gap, **kwargs):
    true_seg = kwargs['true_seg']
    gen_sq_seg = kwargs['gen_sq_seg']
    cri = apply_creteria(gen_sq, true_img, gen_sq_seg=gen_sq_seg,true_seg=true_seg)
    # using single critrion to find extrame point
    if criteria in ['SSIM','NCC','PSNR','DSC']:
        stopping_point = np.argmax(cri[criteria])
    elif criteria in ['MAE','Forbenius', 'GMEAN']:
        stopping_point = np.argmin(cri[criteria])
    elif criteria in ['MEAN']:
        extrem_point = []
        for crit in ['SSIM','NCC','PSNR','DSC']:
            extrem_point.append(np.argmax(cri[crit]))
        for crit in ['MAE', 'Forbenius']:
            extrem_point.append(np.argmin(cri[crit]))
        cri_mean = np.mean(extrem_point)
        stopping_point = np.floor(cri_mean)
    else:
        print('criterion not exist!')

    return ages[0]+stopping_point*gap

def generate_img_via_registration(subject, nb_gen, path_of_img_seg, criteria, save = False, gpu = 0):
    crop_img = '/mri/crop_rescaled_align_norm.nii.gz'
    crop_seg = '/mri/crop_align_aparc+aseg.nii.gz'
    # path of mov and fix imgs and segs
    names = subject['list']
    # resample and save
    if os.path.exists(os.path.join(path_of_img_seg+names[0] + crop_img)) == False:
        my_resample.my_crop(os.path.join(path_of_img_seg+names[0] + '/mri/'))
    if os.path.exists(os.path.join(path_of_img_seg+names[-1] + crop_img)) == False:
        my_resample.my_crop(os.path.join(path_of_img_seg+names[-1] + '/mri/'))
    moving_img = path_of_img_seg+names[0] + crop_img# youngest as moving image
    moving_seg = path_of_img_seg+names[0] + crop_seg
    fixed_img =  path_of_img_seg+names[-1] + crop_img# oldest as moving image
    fixed_seg =  path_of_img_seg+names[-1] + crop_seg

    # pre-trained model
    model = './models/brains-dice-vel-0.5-res-16-256f.h5'

    # tensorflow device handling
    #device, nb_devices = vxm.tf.utils.setup_device(None) # use CPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    device, nb_devices = vxm.tf.utils.setup_device(gpu)  # use CPU

    # load moving and fixed images
    add_feat_axis = True
    moving = vxm.py.utils.load_volfile(moving_img, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(fixed_img, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    moving_seg = vxm.py.utils.load_volfile(moving_seg, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed_seg = vxm.py.utils.load_volfile(fixed_seg, add_batch_axis=True, add_feat_axis=add_feat_axis)


    with tf.device(device):
        # load model and predict
        model = vxm.networks.VxmDense.load(model)

        # extract input of VecInt layer
        flow_model = tf.keras.models.Model(model.inputs, model.get_layer('flow').output)

        input = [moving, fixed]
        flow_output= flow_model.predict(input)

        # extract ode outputs of fix and mov pair, then apply different criteria to find a stopping point
        gen_sq, gen_sq_seg = apply_ode(flow_output, moving,  nb_gen=20, stopping_point = 2, segmentations = True, moving_seg=moving_seg)

        # apply criteria on mov and fix and decide stopping point
        fixed_np = fixed.squeeze()
        fixed_seg_np = fixed_seg.squeeze()
        subject['stopping_point'] = calcu_stopping_point(gen_sq, fixed_np, criteria, gen_sq_seg=gen_sq_seg,true_seg=fixed_seg_np)
        print('Stopping point is {}'.format(subject['stopping_point']))

        # generate imgs and segs based on stopping_point and nb_gen
        gen_sq, gen_sq_seg = apply_ode(flow_output, moving, nb_gen=nb_gen, stopping_point = subject['stopping_point'], segmentations = True, moving_seg=moving_seg)

    if save:
        print('Save images (and Segmentations if provided)')
        save_path = path_of_img_seg + criteria + '_crop/' +subject['id']
        save_generated_imgs(gen_sq, fixed_affine, save_path, save=save, show=False, y_source_sq_seg=gen_sq_seg, segmentations = True)

    return subject, gen_sq, gen_sq_seg
