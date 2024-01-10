import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import itertools

class PatchDataset(Dataset):
    def __init__(self, image_list, patch_size=(32, 32, 32), stride=(16, 16, 16), labels=[0, 1], random_crop=False, resize_factor=1, augmentation=False):
        self.image_list = image_list
        self.patch_size = np.asarray(patch_size)
        self.randomized = random_crop
        self.stride = np.asarray(stride)
        self.labels = labels
        self.resize_factor = resize_factor
        self.augmentation = augmentation
        self.patch_indices = []
        self.data = self.load_data()
    
    def _load_data(self, image_dir, label_dir, image_ext='.tif', label_ext='.tif'):
        """ Load and pair images and labels from directories. """
        image_paths = self.get_sorted_file_paths(image_dir, extension=image_ext)
        label_paths = self.get_sorted_file_paths(label_dir, extension=label_ext)

        # Create a dictionary of label paths with filenames without extension as keys
        label_dict = {os.path.splitext(os.path.basename(path))[0]: path for path in label_paths}

        paired_images = []
        paired_labels = []

        for img_path in image_paths:
            img_filename_wo_ext = os.path.splitext(os.path.basename(img_path))[0]
            if img_filename_wo_ext in label_dict:
                paired_images.append(img_path)
                paired_labels.append(label_dict[img_filename_wo_ext])

        return paired_images, paired_labels

    def load_data(self):
        """ prepare array of 3D images """
        data = []
        for i, (image_path, label_path) in enumerate(tqdm(self.image_list, desc="Processing images")):
            image_3d, label_3d = [], []

            for img, lbl in zip(self._load_data(image_path, label_path)[0], self._load_data(image_path, label_path)[1]):
                image_3d.append(self._preprocess_image(img))
                label_3d.append(self._preprocess_mask(lbl))

            image = np.array(image_3d)
            label = np.array(label_3d)
            data.append({'image': image, 'label': label})
            indices_list = self.get_patch_indices(image.shape, self.patch_size, self.stride)
            self.patch_indices += ([(i, patch_index) for patch_index in indices_list])

        return data
    
    def get_patch_indices(self, image_shape, patch_size, stride):
        """ make list of patch indicies """
        indices_list = []
        
        max_indices = [image_shape[dim] - patch_size[dim] for dim in range(3)]
        if self.randomized:
            num_patchs = (1 + max_indices[0]//stride[0]) * (1 + max_indices[1]//stride[1]) * (1 + max_indices[2]//patch_size[2])
            for i in range(num_patchs):
                start_indices = [random.randint(0, max_indices[dim]) for dim in range(3)]
                indices_list.append(start_indices)
        else:
            for d_start_index, h_start_index, w_start_index in \
                itertools.product(list(range(0, max_indices[0], stride[0])), 
                                  list(range(0, max_indices[1], stride[1])),
                                  list(range(0, max_indices[2], stride[2]))):
                start_indices = [d_start_index, h_start_index, w_start_index]
                indices_list.append(start_indices)    
                
        return indices_list

    def _preprocess_image(self, image_path):
        image = np.array(Image.open(image_path))
        image = image - np.min(image)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)
        return image

    def _preprocess_mask(self, mask_path):
        mask = np.array(Image.open(mask_path))
        if mask.ndim > 2 and mask.shape[2] > 1:
            mask = mask[..., 0]
        mask = (mask > 127).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        X = torch.zeros((1, *self.patch_size), dtype=torch.float32)
        Y = torch.zeros((*self.patch_size, len(self.labels)), dtype=torch.int16)

        data_index, patch_index = self.patch_indices[idx]
        selected_image = self.data[data_index]
        image_data, label_data = selected_image['image'], selected_image['label']
        image_patch = self.extract_patch(image_data, patch_index)
        label_patch = self.extract_patch(label_data, patch_index)

        image_patch = self.normalize_mean_std(image_patch)
        image = torch.tensor(image_patch, dtype=torch.float32)
        label = torch.tensor(label_patch, dtype=torch.int16)
        
        X[0] = image
        Y = self.multi_class_labels(label)
        
        return X, Y
    
    def get_sorted_file_paths(self, directory, extension=None):
        """ Returns a list of sorted file paths in the given directory with a specific extension. """
        file_paths = [os.path.join(directory, fname) for fname in sorted(os.listdir(directory)) if fname.endswith(extension)]
        return file_paths

    def normalize_mean_std(self, tmp):
        """ normalize patch distribution """
        tmp_std = np.std(tmp) + 0.0001
        tmp_mean = np.mean(tmp)
        tmp = (tmp - tmp_mean) / tmp_std
        return tmp

    def extract_patch(self, data, start_indices):
        """ extract patch of data given indices """
        return data[start_indices[0]:start_indices[0] + self.patch_size[0], \
                    start_indices[1]:start_indices[1] + self.patch_size[1], \
                    start_indices[2]:start_indices[2] + self.patch_size[2]]

    def multi_class_labels(self, data):
        n_labels = len(self.labels)
        new_shape = tuple(np.append(data.shape, n_labels))
        y = torch.zeros(new_shape, dtype=torch.float32)
        for label_index in range(n_labels):
            y[..., label_index][data == self.labels[label_index]] = 1
        return y