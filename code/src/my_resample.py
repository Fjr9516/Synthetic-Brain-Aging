#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
This script is for rescaling and resampling the aligned t1w mris
'''
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import neurite as ne
# # parse commandline args
# parser = argparse.ArgumentParser()
# parser.add_argument('--input-dir', default='./adni_genic_1/', help="directory of data, e.g. './adni_genic_1/'")
#
# args = parser.parse_args()


# resample by simpleitk
def resample_image(itk_image, out_size=[512, 512, 205], is_label=False):
    if is_label:
        interpolator_type = sitk.sitkNearestNeighbor
    else:
        interpolator_type = sitk.sitkLinear

    out_spacing = [origin_sz * origin_spc / out_sz for origin_sz, origin_spc, out_sz in
                   zip(itk_image.GetSize(), itk_image.GetSpacing(), out_size)]

    return sitk.Resample(itk_image, out_size, sitk.Transform(), interpolator_type, itk_image.GetOrigin(), out_spacing,
                         itk_image.GetDirection(), 0.0, itk_image.GetPixelIDValue())

def resample_spacing_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def my_resample(norm_file, out_size = [160, 160, 192]):
    # load img
    rescaled_align_norm = sitk.ReadImage(norm_file + "/rescaled_align_norm.nii.gz", sitk.sitkFloat32)

    # aseg_img  = sitk.ReadImage(norm_file + '/align_aseg.nii.gz', sitk.sitkInt16)
    aparc_aseg_img = sitk.ReadImage(norm_file + '/align_aparc+aseg.nii.gz', sitk.sitkInt16)

    resampled_rescaled_align_norm = resample_image(rescaled_align_norm, out_size=out_size, is_label=False)
    # resampled_align_aseg          = resample_image(aseg_img, out_size=out_size, is_label=True)
    resampled_align_aparc_aseg = resample_image(aparc_aseg_img, out_size=out_size, is_label=True)

    sitk.WriteImage(resampled_rescaled_align_norm, norm_file + '/resampled_rescaled_align_norm.nii.gz')
    # sitk.WriteImage(resampled_align_aseg, norm_file + '/resampled_align_aseg.nii.gz')
    sitk.WriteImage(resampled_align_aparc_aseg, norm_file + '/resampled_align_aparc+aseg.nii.gz')

def my_resample_back(norm_file, out_spacing = [1,1,1]):
    # load img
    rescaled_align_norm = sitk.ReadImage(norm_file + "/resampled_rescaled_align_norm.nii.gz", sitk.sitkFloat32)

    # aseg_img  = sitk.ReadImage(norm_file + '/align_aseg.nii.gz', sitk.sitkInt16)
    aparc_aseg_img = sitk.ReadImage(norm_file + '/resampled_align_aparc+aseg.nii.gz', sitk.sitkInt16)

    resampled_rescaled_align_norm = resample_spacing_image(rescaled_align_norm, out_spacing=out_spacing, is_label=False)
    # resampled_align_aseg          = resample_image(aseg_img, out_size=out_size, is_label=True)
    resampled_align_aparc_aseg = resample_spacing_image(aparc_aseg_img, out_spacing=out_spacing, is_label=True)

    sitk.WriteImage(resampled_rescaled_align_norm, norm_file + '/reresampled_rescaled_align_norm.nii.gz')
    # sitk.WriteImage(resampled_align_aseg, norm_file + '/resampled_align_aseg.nii.gz')
    sitk.WriteImage(resampled_align_aparc_aseg, norm_file + '/reresampled_align_aparc+aseg.nii.gz')


def my_crop(norm_file, uppoint=[48, 42, 13], out_size = [160, 160, 192]):
    # load img
    rescaled_align_norm = sitk.ReadImage(norm_file + "/rescaled_align_norm.nii.gz", sitk.sitkFloat32)

    # aseg_img  = sitk.ReadImage(norm_file + '/align_aseg.nii.gz', sitk.sitkInt16)
    aparc_aseg_img = sitk.ReadImage(norm_file + '/align_aparc+aseg.nii.gz', sitk.sitkInt16)

    # # print metadata
    # print('---- Print metadata for input image: ---- ')
    # print("Size:", rescaled_align_norm.GetSize())
    # print("Origin:", rescaled_align_norm.GetOrigin())
    # print("Spacing", rescaled_align_norm.GetSpacing())

    # crop: oasis: [48, 42, 13]
    # crop image
    crop_rescaled_align_norm = sitk.Crop(
        rescaled_align_norm,  # 輸入影像
        uppoint,  # 下方去除寬度
        [rescaled_align_norm.GetWidth() - (uppoint[0]+out_size[0]), rescaled_align_norm.GetHeight() - (uppoint[1]+out_size[1]),
         rescaled_align_norm.GetDepth() - (uppoint[2]+out_size[2])]) # 上方去除寬度
    # crop seg
    crop_align_aparc_aseg = sitk.Crop(
        aparc_aseg_img,  # 輸入影像
        uppoint,  # 下方去除寬度
        [aparc_aseg_img.GetWidth() - (uppoint[0]+out_size[0]), aparc_aseg_img.GetHeight() - (uppoint[1]+out_size[1]),
         aparc_aseg_img.GetDepth() - (uppoint[2]+out_size[2])]) # 上方去除寬度
    # crop_rescaled_align_norm = resample_image(rescaled_align_norm, out_size=out_size, is_label=False)
    # # resampled_align_aseg          = resample_image(aseg_img, out_size=out_size, is_label=True)
    # crop_align_aparc_aseg = resample_image(aparc_aseg_img, out_size=out_size, is_label=True)

    # # print metadata
    # print('---- Print metadata for output image: ---- ')
    # print("Size:", crop_rescaled_align_norm.GetSize())
    # print("Origin:", crop_rescaled_align_norm.GetOrigin())
    # print("Spacing", crop_rescaled_align_norm.GetSpacing())

    # SAVE
    # print('--- save ---')
    sitk.WriteImage(crop_rescaled_align_norm, norm_file + '/crop_rescaled_align_norm.nii.gz')
    # # sitk.WriteImage(resampled_align_aseg, norm_file + '/resampled_align_aseg.nii.gz')
    sitk.WriteImage(crop_align_aparc_aseg, norm_file + '/crop_align_aparc+aseg.nii.gz')

    # show
    # template = sitk.GetArrayViewFromImage(crop_rescaled_align_norm)
    # # plot and save png
    # mid_slices_moving = [np.take(template, template.shape[d] // 2, axis=d) for d in range(3)]
    # mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
    # mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)
    # ne.plot.slices(mid_slices_moving, cmaps=['gray'], do_colorbars=True, grid=[1, 3],
    #                show=True,save=False);

def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           axes_off=True,
           imshow_args=None,
           save=False,  # TODO jrfu: save plotted images
           save_name='./saved_images/'):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if save:
        plt.savefig(save_name)
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return (fig, axs)

# norm_files = ['/home/fjr/data/OASIS/OASIS-3/oasis3/OAS31167_d0064/mri/',
#               '/home/fjr/data/OASIS/OASIS-3/oasis3/OAS31167_d4564/mri/']
# for norm_file in norm_files:
#     my_crop(norm_file)