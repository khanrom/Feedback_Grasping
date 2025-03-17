import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
import json
import torch

class JacquardDataset(Dataset):
    def __init__(self, data_dir, label_type="MAP", data_type="SEP", transform=None, depth_transform=None,
                 mask_transform=None, target_transform=None):
        """
        :param data_dir: directory of data
        :param label_type:  MAP or PARAM
        :param data_type:  SEP or COMB
        :param transform:  img transformation
        :param depth_transform:  depth transformation
        :param mask_transform:  mask transformation
        :param target_transform:  label transformation

        MAP: return label as 2D map of grasp points
        PARAM: return label a set of grasp parameters
        SEP: return img and depth as separate tensors
        COMB: return img and depth as a single tensor
        """
        self.data_dir = data_dir
        self.id_to_cls = json.load(open(os.path.join(data_dir, "id_to_cls.json")))
        self.transform = transform
        self.depth_transform = depth_transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.data_type = data_type

        assert self.data_type == "SEP" or self.data_type == "COMB", "unexpected data_type (Must be SEP or COMB)"
        assert self.label_type == "MAP" or self.label_type == "PARAM", "unexpected label_type (MAP be SEP or PARAM)"

    def __len__(self):
        return len(self.id_to_cls.keys())

    def __getitem__(self, idx):
        idx = int(idx)
        img_path = os.path.join(self.data_dir, 'img', str(idx))
        label_path = os.path.join(self.data_dir, 'label', str(idx))
        img = torch.tensor(np.transpose(np.load(img_path + "_RGB.npy"), (2, 0, 1)))
        depth = torch.tensor(np.load(img_path + "_perfect_depth.npy"))
        mask = torch.tensor(np.load(img_path+"_mask.npy"))

        if self.transform:
            img = self.transform(img)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.data_type == "SEP":
            X = (img, depth)
        else:
            X = torch.cat([img, depth.view(1, *depth.shape)], dim=0)

        if self.label_type == "MAP":
            # Normalize label batch
            label = np.transpose(np.load(label_path+"_map_grasps.npy"), (2, 0, 1))
            for i in range(label.shape[0]):
                if i == 0 or i == 1 or i == 3:
                    map = label[i,:,:]
                    map = (map - np.min(map)) / (np.max(map) - np.min(map))
                    label[i,:,:] = map
            label = torch.tensor(label)
        else:
            label = torch.tensor(np.transpose(np.load(label_path+"_txt_grasps.npy"), (2, 0, 1)))

        if self.target_transform:
            label = self.target_transform(label)

        return X, mask, label
