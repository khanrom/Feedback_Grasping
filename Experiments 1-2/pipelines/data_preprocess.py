"""
File adapted and modified from stevolopolis/GrTrainer_paperspace
"""

import glob
import torch
import os
import random
import math
import numpy as np

from PIL import Image
from torchvision import transforms
import yaml
import json

from types import SimpleNamespace
from tqdm import tqdm

with open('config.yaml', 'r') as file:
    p = yaml.safe_load(file)
    params = json.loads(json.dumps(p), object_hook=lambda d: SimpleNamespace(**d))


def rotate_grasp_label(grasp_list, degrees):
    """Returns rotated list of Grasp labels given the degrees."""
    # grasp_list.shape == (n, 5)
    # x, y, theta, w, h
    new_grasp_list = []
    for grasp in grasp_list:
        x = grasp[0] / params.Meta_data.IMG_DIM * 1024
        y = grasp[1] / params.Meta_data.IMG_DIM * 1024

        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d((1024 // 2, 1024 // 2))
        p = np.atleast_2d((x, y))

        coords = np.squeeze((R @ (p.T - o.T) + o.T).T)
        if degrees == 0 or degrees == 180:
            theta = grasp[2] * 180 - 90
        elif degrees == 90 or degrees == 270:
            if grasp[2] <= 0.5:
                theta = grasp[2] * 180
            elif grasp[2] > 0.5:
                theta = grasp[2] * 180 - 180

        w = grasp[3]
        h = grasp[4]

        new_grasp_list.append([coords[0] / 1024 * params.Meta_data.IMG_DIM,
                               coords[1] / 1024 * params.Meta_data.IMG_DIM,
                               (theta + 90) / 180,
                               w,
                               h])

    return np.array(new_grasp_list)


def grasps_to_bboxes(grasps):
    """Converts grasp boxes to bounding boxes."""
    # convert grasp representation to bbox
    x = grasps[:, 0]
    y = grasps[:, 1]
    theta = torch.deg2rad(grasps[:, 2] * 180 - 90)
    w = grasps[:, 3]
    h = grasps[:, 4] * 100

    x1 = x - w / 2 * torch.cos(theta) + h / 2 * torch.sin(theta)
    y1 = y - w / 2 * torch.sin(theta) - h / 2 * torch.cos(theta)
    x2 = x + w / 2 * torch.cos(theta) + h / 2 * torch.sin(theta)
    y2 = y + w / 2 * torch.sin(theta) - h / 2 * torch.cos(theta)
    x3 = x + w / 2 * torch.cos(theta) - h / 2 * torch.sin(theta)
    y3 = y + w / 2 * torch.sin(theta) + h / 2 * torch.cos(theta)
    x4 = x - w / 2 * torch.cos(theta) - h / 2 * torch.sin(theta)
    y4 = y - w / 2 * torch.sin(theta) + h / 2 * torch.cos(theta)

    xs = torch.stack((x1, x2, x3, x4), 1)
    ys = torch.stack((y1, y2, y3, y4), 1)
    min_x = torch.min(xs, dim=1)[0]
    max_x = torch.max(xs, dim=1)[0]
    min_y = torch.min(ys, dim=1)[0]
    max_y = torch.max(ys, dim=1)[0]

    return torch.stack((min_x, max_x, min_y, max_y), 1)


def bboxes_boundaries(bboxes):
    lower_bounds = torch.min(bboxes, dim=0)[0]
    upper_bounds = torch.max(bboxes, dim=0)[0]
    return lower_bounds[0], upper_bounds[1], lower_bounds[2], upper_bounds[3]


def point_in_bboxes(pt, bboxes):
    x_left = bboxes[:, 0] <= pt[0]
    x_right = bboxes[:, 1] >= pt[0]
    y_left = bboxes[:, 2] <= pt[1]
    y_right = bboxes[:, 3] >= pt[1]

    x_in_bboxes = torch.logical_and(x_left, x_right)
    y_in_bboxes = torch.logical_and(y_left, y_right)
    pt_in_bboxes = torch.logical_and(x_in_bboxes, y_in_bboxes)

    return pt_in_bboxes.nonzero()


def point_in_bbox(pt, bbox):
    """
    Returns True if the point is inside the bbox.
    pt = [x, y]
    bbox = [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if (bbox[0] <= pt[0] <= bbox[1]) and (bbox[2] <= pt[1] <= bbox[3]):
        return True
    else:
        return False


class DataPreProcessor:
    """
    This class has two main functions:
        - convert data (images/labels) to .npy files
        - convert grasp candidate labels to grasp maps
    Other helper functions in this class are taken from the
    DataLoader class in data_loader_v2.py.
    """

    def __init__(self):
        if params.Meta_data.MODE == 'train':
            self.raw_path = params.Meta_data.RAW_TRAIN_DATA_DIR
        else:
            self.raw_path = params.Meta_data.RAW_TEST_DATA_DIR
        if params.Meta_data.MODE == 'train':
            self.path = params.Meta_data.TRAIN_DATA_DIR
        else:
            self.path = params.Meta_data.TEST_DATA_DIR
        if self.path not in os.listdir():
            os.makedirs(self.path)
        self.subdir = self.path.split('/')[-1]

        # Create data subdirectory
        if 'img' not in os.listdir(os.path.join(self.path)):
            os.makedirs(os.path.join(self.path, 'img'))
        if 'label' not in os.listdir(os.path.join(self.path)):
            os.makedirs(os.path.join(self.path, 'label'))

        # Get list of obj hash names
        self.hash_names = self.get_hash_names()
        file = open(os.path.join(self.path, "hash_names.txt"), 'w')
        for hash_name in self.hash_names:
            file.write(hash_name + "\n")
        file.close()

    def get_hash_names(self):
        return [pn for pn in os.listdir(self.raw_path) if not pn.startswith('.')]

    def data2npy(self):
        """
        Save training data as .npy files for improved data loading efficiencies.
        Each training example is assigned with a global idx starting from 0.
        """
        global_id = 0
        id_to_cls_dict = {}
        for hash_name in tqdm(self.hash_names):
            id_list = [str(fn).split("_")[0] for fn in os.listdir(os.path.join(self.raw_path, hash_name))]
            for i in set(id_list):
                img_path = os.path.join(self.raw_path, hash_name, str(i)+"_"+hash_name)
                save_name = str(global_id)
                id_to_cls_dict[global_id] = [hash_name, i]
                self.img2npy(img_path, save_name)
                self.label2npy(img_path, save_name, data_augmentation=params.Meta_data.DATA_AUG)
                global_id += 1
        with open(os.path.join(self.path, 'id_to_cls.json'), 'w') as file:
            file.write(json.dumps(id_to_cls_dict))

    def img2npy(self, img_path, save_name):
        """Save images as .npy files after simply preprocessing"""
        # Convert RGB image to .npy
        img_rgb = Image.open(os.path.join(img_path + '_RGB.png'))
        img_rgb = img_rgb.resize((params.Meta_data.IMG_DIM, params.Meta_data.IMG_DIM))
        img_rgb = np.array(img_rgb)
        norm_img_rgb = (img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb))  # Normalize to [0, 1]

        # Convert Depth image to .npy
        img_d = Image.open(os.path.join(img_path + '_perfect_depth.tiff'))
        img_d = img_d.resize((params.Meta_data.IMG_DIM, params.Meta_data.IMG_DIM))
        img_d = np.array(img_d)
        norm_img_d = (img_d - np.min(img_d)) / (np.max(img_d) - np.min(img_d)) # Normalize to [0, 1]

        # Convert Mask image to .npy
        img_mask = Image.open(os.path.join(img_path + '_mask.png'))
        img_mask = img_mask.resize((params.Meta_data.IMG_DIM, params.Meta_data.IMG_DIM))
        img_mask = np.array(img_mask)

        # Save .npy file
        rgb_name = save_name + '_RGB.npy'
        new_rgb_path = os.path.join(self.path, "img", rgb_name)
        d_name = save_name + '_perfect_depth.npy'
        new_d_path = os.path.join(self.path, "img",d_name)
        mask_name = save_name + '_mask.npy'
        new_mask_path = os.path.join(self.path, "img", mask_name)
        np.save(open(new_rgb_path, 'wb'), norm_img_rgb.astype(np.float32))
        #np.save(open(new_rgb_path, 'wb'), img_rgb.astype(np.float32))
        np.save(open(new_d_path, 'wb'), norm_img_d.astype(np.float32))
        #np.save(open(new_d_path, 'wb'), img_d.astype(np.float32))
        np.save(open(new_mask_path, 'wb'), img_mask)

    def label2npy(self, img_path, save_name, data_augmentation=True):
        """Saves labels as .npy files after converting grasps into maps."""
        # Get Grasp label candidates and training label from '_grasps.txt' file
        grasp_file_path = os.path.join(img_path+'_grasps.txt')
        # List of Grasp candidates
        grasp_list = self.load_grasp_label(grasp_file_path)
        if data_augmentation:
            for degree in [0, 90, 180, 270]:
                # Augmentation on labels -- random rotations
                if degree != 0:
                    rotated_grasp_list = rotate_grasp_label(np.array(grasp_list), degree)
                else:
                    rotated_grasp_list = grasp_list
                rotated_grasp_map = self.grasp2map(rotated_grasp_list)

                label_map_name = save_name + '_%s_map_grasps.npy' % degree
                label_map_path = os.path.join(self.path, "label", label_map_name)
                label_txt_name = save_name + '_%s_txt_grasps.npy' % degree
                label_txt_path = os.path.join(self.path, "label", label_txt_name)
                np.save(open(label_map_path, 'wb'), rotated_grasp_map)
                np.save(open(label_txt_path, 'wb'), rotated_grasp_list)
        else:
            grasp_map = self.grasp2map(grasp_list)
            label_map_name = save_name + '_map_grasps.npy'
            label_map_path = os.path.join(self.path, "label", label_map_name)
            label_txt_name = save_name + '_txt_grasps.npy'
            label_txt_path = os.path.join(self.path, "label", label_txt_name)
            np.save(open(label_map_path, 'wb'), grasp_map)
            np.save(open(label_txt_path, 'wb'), grasp_list)

    def grasp2map(self, grasp_list):
        """
        Converts grasp candidates into grasp maps.
        Grasp label: [x, y, theta, w, h]
        """
        # Initiate map with shape <IMG_DIM, IMG_DIM, 6>
        grasp_map = np.zeros((params.Meta_data.IMG_DIM, params.Meta_data.IMG_DIM, 6), dtype=np.float32)
        bboxes = grasps_to_bboxes(torch.tensor(grasp_list))
        min_x, max_x, min_y, max_y = bboxes_boundaries(bboxes)
        for i in range(params.Meta_data.IMG_DIM):
            if not min_y <= i <= max_y:
                continue
            for j in range(params.Meta_data.IMG_DIM):
                if not min_x <= j <= max_x:
                    continue
                closest_candidate = [0, 0, 0, 0, 0]
                closest_dist = 999
                valid_bboxes_idx = point_in_bboxes([j, i], bboxes)
                for k in valid_bboxes_idx:
                    x_diff = grasp_list[k][0] - j
                    y_diff = grasp_list[k][1] - i
                    dist = abs(x_diff) + abs(y_diff)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_candidate = [x_diff, y_diff, grasp_list[k][2], grasp_list[k][3], grasp_list[k][4]]

                pixel_label = closest_candidate + [len(valid_bboxes_idx)]
                grasp_map[i, j, :] = pixel_label

        grasp_map[:, :, 5] /= np.max(grasp_map[:, :, 5])
        return grasp_map

    def load_grasp_label(self, file_path):
        """Returns a list of grasp labels from <file_path>."""
        grasp_list = []
        with open(file_path, 'r') as f:
            file = f.readlines()
            # dat format in each line: 'x;y;theta;w;h'
            for grasp in file:
                # remove '\n' from string
                grasp = grasp[:-1]
                label = grasp.split(';')
                label = self.noramlize_grasp(label)
                grasp_list.append(label)

        return grasp_list


    def get_grasp_label(self, grasp_list, metric='random'):
        """
        Returns the selected grasp label for training.

        Selection metrics:
            - random -- random label
            - median -- random label between 35-percentile and 65-percentile of grasp dimension sum (w+h)
            - widest -- random label between 70-percentile and 80-percentile of grasp dimension sum (w+h)
            |   - empirically the best choice (sufficiently big without outliers)
            - smallest -- random label between 0-percentile and 20-percentile of grasp dimension sum (w+h)

        """
        # Selection method: 'w+h' median
        if metric == 'median':
            grasp_list.sort(key=lambda x: x[3] + x[4])
            mid_idx = random.randint(int(len(grasp_list) * 0.35), int(len(grasp_list) * 0.65))
            return grasp_list[mid_idx]
        # Selection method: random
        if metric == 'random':
            idx = random.randint(0, len(grasp_list) - 1)
            return grasp_list[idx]
        # Selection method: widest grasp
        if metric == 'widest':
            grasp_list.sort(key=lambda x: x[3] + x[4])
            top_10 = random.randint(int(len(grasp_list) * 0.2), int(len(grasp_list) * 0.3))
            return grasp_list[-top_10]
        # Selection method: smallest grasp
        if metric == 'smallest':
            grasp_list.sort(key=lambda x: x[3] + x[4])
            top_10 = int(len(grasp_list) * 0.2)
            return grasp_list[top_10]

    def noramlize_grasp(self, label):
        """Returns normalize grasping labels."""
        norm_label = []
        for i, value in enumerate(label):
            if i == 4:
                # Height
                norm_label.append(float(value) / 100)
            elif i == 2:
                # Theta
                norm_label.append((float(value) + 90) / 180)
            elif i == 3:
                # Width
                norm_label.append(float(value) / 1024 * params.Meta_data.IMG_DIM)
            else:
                # Coordinates
                norm_label.append(float(value) / 1024 * params.Meta_data.IMG_DIM)
        #print('norm_label: ', norm_label)

        return norm_label


    def scan_img_id(self):
        """
        Returns a dictionary mapping the image ids from the 'data'
        folder to their corresponding classes.
        """
        img_id_dict = {}
        for img_path in glob.iglob('%s/*/*/*' % self.path):
            if not img_path.endswith('RGB.png'):
                continue

            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]
            img_id_with_var = img_var + '_' + img_id
            img_id_dict[img_id_with_var] = img_cls

        return img_id_dict
