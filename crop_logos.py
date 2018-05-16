# The MIT License (MIT)
# Copyright (c) 2018 satojkovic

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from collections import defaultdict, deque
import glob
import common
import util
import skimage.io
from scipy.misc import imresize
import warnings
import cv2

MAX_DATA_AUG_PER_LINE = 30


def rect_coord(annot_part):
    return list(map(int, annot_part))


def center_wid_hgt(x1, y1, x2, y2):
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    wid = (x2 - x1)
    hgt = (y2 - y1)
    return cx, cy, wid, hgt


def is_skip(annot_part):
    x1, y1, x2, y2 = rect_coord(annot_part)
    _, _, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    if wid <= 0 or hgt <= 0:
        return True
    else:
        return False


def crop_image(img, rect):
    return img[rect[1]:rect[3], rect[0]:rect[2]]


def resize_img(img, size=(32, 32)):
    return imresize(img, size, interp='bicubic')


def save_transformed_imgs(imgs, annot, aug_params, line_no):
    fn, class_name, train_subset_class = util.parse_annot(annot)
    root, ext = os.path.splitext(fn)
    dst_dir = os.path.join(common.CROPPED_IMAGE_DIR, class_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i, img in enumerate(imgs):
        if aug_params[i]['rot_deg'] < 0:
            rot_deg = 'm' + format(np.abs(aug_params[i]['rot_deg']), '.1f')
        else:
            rot_deg = format(aug_params[i]['rot_deg'], '.1f')

        save_fn = '_'.join([
            str(line_no), str(i), root, class_name, train_subset_class,
            'shiftW' + str(aug_params[i]['shift_w']),
            'shiftH' + str(aug_params[i]['shift_h']), 'rot' + rot_deg,
            'scale' + format(aug_params[i]['scale_rate'], '.2f')
        ]) + ext
        skimage.io.imsave(os.path.join(dst_dir, save_fn), img)


def crop_and_aug_random(annot_train):
    # Data augmentation results
    aug_results = deque(maxlen=MAX_DATA_AUG_PER_LINE)
    aug_params = deque(maxlen=MAX_DATA_AUG_PER_LINE)
    aug_keys = ['shift_w', 'shift_h', 'rot_deg', 'rot_rad', 'scale_rate']
    cnt_per_line = defaultdict(int)

    for i, annot in enumerate(annot_train):
        # Get image file name
        fn, class_name, _ = util.parse_annot(annot)

        # Skip if width or height equal zero
        if is_skip(annot[3:]):
            print('Skip: ', fn)
            continue

        # Read image by skimage
        img = skimage.io.imread(os.path.join(common.TRAIN_IMAGE_DIR, fn))
        img = skimage.exposure.equalize_adapthist(img)

        # Crop logo area
        annot_rect = util.get_annot_rect(annot)
        cropped_img = crop_image(img, annot_rect)

        # Resize cropped image
        resized_cropped_img = resize_img(cropped_img)

        aug_results.append(resized_cropped_img)
        normal_params = {}
        for key in aug_keys:
            normal_params[key] = 0
        aug_params.append(normal_params)
        cnt_per_line[i] += 1

        # Save transformed images
        save_transformed_imgs(aug_results, annot, aug_params, i)

        # Clear data augmentation results
        aug_results.clear()
        aug_params.clear()


def crop_logo(annot_train, with_none=False):
    # make the root directory to save processed images
    if not os.path.exists(common.CROPPED_IMAGE_DIR):
        os.makedirs(common.CROPPED_IMAGE_DIR)

    # crop images and apply augmentation
    crop_and_aug_random(annot_train)

    # print results
    org_imgs = [img for img in os.listdir(common.TRAIN_IMAGE_DIR)]
    crop_and_aug_imgs = [
        fname
        for root, dirs, files in os.walk(common.CROPPED_IMAGE_DIR)
        for fname in glob.glob(os.path.join(root, '*.jpg'))
    ]
    print('cropped: %d' % (len(crop_and_aug_imgs)))


def main():
    with warnings.catch_warnings():
        # Supress low contrast warnings
        warnings.simplefilter("ignore")

        annot_train = np.loadtxt(common.ANNOT_FILE, dtype='a')
        print('train_annotation: %d' % (annot_train.shape[0]))

        # Crop logo
        crop_logo(annot_train)


if __name__ == '__main__':
    main()
