#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os, sys, glob
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import multiprocessing as mp
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import time

import torch
import mlflow
import mlflow.pytorch

import scipy
from skimage import measure, morphology
from skimage.morphology import convex_hull_image
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

LS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lung-segmentation')
sys.path.append(LS_PATH)
import predict
from data import utils as data_utils

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
import utils.exp_utils as utils


dir_path = os.path.dirname(os.path.realpath(__file__))
cf_file = utils.import_module("cf", os.path.join(dir_path, "configs.py"))
cf = cf_file.configs()
# Load lung segmentation model from MLFlow
remote_server_uri = "http://mlflow.10.7.13.202.nip.io/"
mlflow.set_tracking_uri(remote_server_uri)
model_name = "2-lungs-segmentation"
unet = mlflow.pytorch.load_model(f"models:/{model_name}/production")


def resample_array(src_imgs, src_spacing, target_spacing):

    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')

    return resampled_img


def resample_array_to_shape(img, spacing, target_shape=[128, 256, 256]):
    res_spacing = [spacing[ix] * img.shape[::-1][ix] / target_shape[::-1][ix] for ix in range(len(img.shape))]
    resampled_img = resize(img, target_shape, order=1, clip=True, preserve_range=True, mode='edge')
    return resampled_img, res_spacing


###########################################################
# START: preprocessing methods from DSB17 winning solution
###########################################################


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


def compute_lung_mask(case_path):
    case_pixels, spacing = data_utils.load_dicom_slices(case_path)
    # case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('float16')
    return newimg


###########################################################
# END: preprocessing methods from DSB17 winning solution
###########################################################


def preprocess_image(img_path, output_path=None):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    itk_img = sitk.ReadImage(img_path)
    original_spacing = np.array(itk_img.GetSpacing())
    img_arr = sitk.GetArrayFromImage(itk_img)
    original_shape = img_arr.shape
    ls_img_arr = np.copy(img_arr)
    print(f'processing {img_id}')

    # Resample and Normalize
    img_arr = resample_array(img_arr, itk_img.GetSpacing(), cf.target_spacing)

    # Compute lungs mask
    ls_img_arr, spacing = data_utils.prep_img_arr(ls_img_arr, original_spacing)
    mask = predict.predict(ls_img_arr, 1, unet, threshold=True, erosion=True)
    torch.cuda.empty_cache()
    mask, spacing = resample_array_to_shape(mask[0][0], spacing, target_shape=img_arr.shape)
    mask[mask > 0.5] = 1
    mask[mask != 1] = 0

    # Dilate arround lungs, normalize lum and remove bones (see winners DSB17)
    dilatedMask = process_mask(mask)
    Mask = mask
    extramask = dilatedMask.astype(np.uint8) - Mask.astype(np.uint8)
    bone_thresh = 210
    pad_value = 170
    img_arr[np.isnan(img_arr)] = -2000
    sliceim = lumTrans(img_arr)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim * extramask > bone_thresh
    sliceim[bones] = pad_value
    print(f'done processing {img_id}')
    if output_path is not None:
        np.save(output_path, sliceim)
    return sliceim, itk_img.GetOrigin(), original_spacing, original_shape


def pp_patient(inputs):

    ix, path = inputs
    pid = path.split('/')[-1]
    if not os.path.exists(os.path.join(cf.pp_dir, f'{pid}_rois.npy')):
        sliceim, _, original_spacing, _ = preprocess_image(os.path.join(path, f'{pid}_CT.nrrd'))
    else:
        print(f"{pid} image already exists, load it...")
        sliceim = np.load(os.path.join(cf.pp_dir, f'{pid}_rois.npy'))
        original_spacing = sitk.ReadImage(os.path.join(path, f'{pid}_CT.nrrd')).GetSpacing()
        
    df = pd.read_csv(os.path.join(cf.root_dir, 'characteristics.csv'), sep=';')
    df = df[df.patient_id == pid]

    final_rois = np.zeros_like(sliceim, dtype=np.uint8)
    mal_labels = []
    roi_paths = set([ii for ii in os.listdir(path) if '_nod_' in ii])

    rix = 1
    try:
        for roi_path in roi_paths:
            rid = os.path.splitext(roi_path)[0]
            mal_label = int(np.mean(df[df.nodule_id == rid].malignancy.values))
            roi = sitk.ReadImage(os.path.join(cf.raw_data_dir, pid, roi_path))
            roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)
            roi_arr = resample_array(roi_arr, roi.GetSpacing(), cf.target_spacing)
            assert roi_arr.shape == sliceim.shape, [roi_arr.shape, sliceim.shape, pid, roi.GetSpacing()]
            for ix in range(len(sliceim.shape)):
                npt.assert_almost_equal(roi.GetSpacing()[ix], original_spacing[ix])
            mal_labels.append(mal_label)
            final_rois[roi_arr > 0.5] = rix
            # final_rois[roi_arr > 0.5] = 1 # 1 output class
            rix += 1
    except Exception as e:
        print("Error {}:".format(pid), e)

    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]
    mal_labels = np.array(mal_labels)

    np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    if not os.path.exists(os.path.join(cf.pp_dir, f'{pid}_rois.npy')):
        np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), sliceim)

    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'spacing': original_spacing, 'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)
    print('done processing {}'.format(pid))


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print("aggregated meta info to df with length", len(df))


def select_paths(data_dir):
    import csv
    # Ignore some CT scans from csv file
    pids_to_ignore = []
    with open('pp_scans_checkup_bkp.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row['flip'] or row['crap'] or row['poor'] > 2 or row['warn']:
                print(f'ignoring pid {row["pid"]} (flip {row["flip"]}; crap {row["crap"]}; noise {row["poor"]}; warn {row["warn"]}; note: {row["note"]})')
                pids_to_ignore.append(row["pid"])
    paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(data_dir) if ii not in pids_to_ignore]
    return paths


if __name__ == "__main__":

    start_time = time.time()

    paths = select_paths(cf.raw_data_dir)

    """
    paths = []
    for ix, pid in enumerate(os.listdir(cf.raw_data_dir), 0):
        found = False
        for ex in os.listdir(cf.pp_dir):
            if pid in ex:
                found = True
        if not found:
            paths.append(os.path.join(cf.raw_data_dir, pid))
    """

    if not os.path.exists(cf.pp_dir):
        os.mkdir(cf.pp_dir)

    mp.set_start_method('spawn')
    pool = mp.Pool(processes=8)
    p1 = pool.map(pp_patient, enumerate(paths))
    pool.close()
    pool.join()

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)

    print(f"------ Ellapsed time : {time.time() - start_time} (s) ------")
