import os
import sys
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import time

from . import tools


class Feeder(torch.utils.data.Dataset):
    """Feeder for skeleton-based movement disorder recognition to be used with SkateFormer.

    This feeder includes specialized augmentation techniques while maintaining
    the original ST-GCN structure and additionally returns a time index array
    (`index_t`) for SkateFormer’s temporal embedding when `index_t=True`.

    Args:
        data_path (str): Path to the .npy data file (N, C, T, V, M)
        label_path (str): Path to the label file
        random_choose (bool): If true, randomly choose a portion of the input sequence
        random_move (bool): If true, randomly pad zeros at the beginning or end of sequence
        window_size (int): The length of the output sequence
        augment (bool): If true, apply movement disorder-specific augmentations
        mixup (bool): If true, apply mixup augmentation
        debug (bool): If true, only use the first 100 samples
        mmap (bool): If true, use memory mapping for large datasets
    """
    
    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 augment=False,
                 mixup=False,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.augment = augment
        self.mixup = mixup
        
        self.load_data(mmap)

    def load_data(self, mmap):
        # Load label data
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f)

        # Load skeleton data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Get data and label
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # Random sampling / zero-padding
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # Apply data augmentation
        if self.augment:
            data_numpy = self.apply_augmentation(data_numpy)

        # Generate the time index for SkateFormer (simply 0..T-1)
        T = data_numpy.shape[1]
        index_t = np.arange(T)

        return data_numpy, label, index_t

    def apply_augmentation(self, data):
        """Apply various movement disorder-specific augmentations."""
        augmentation_prob = 0.5
        
        if np.random.rand() < augmentation_prob:
            aug_methods = [
                (self.pattern_intensity_augment, 0.3),
                (self.sustained_posture_augment, 0.3),
                (self.speed_variation_augment, 0.3),
                (self.tremor_augment, 0.2)
            ]
            
            for aug_func, prob in aug_methods:
                if np.random.rand() < prob:
                    data = aug_func(data)
                    
        return data

    def pattern_intensity_augment(self, data, intensity_range=(0.8, 1.2)):
        """Adjust movement intensity while preserving patterns."""
        movement = np.diff(data, axis=1)  # B, T, V, M -> diff along time
        intensity = np.random.uniform(*intensity_range)
        scales = np.ones(data.shape[1]) * intensity

        # Simple smoothing
        kernel_size = 5
        scales = np.convolve(scales, np.ones(kernel_size)/kernel_size, mode='same')
        
        augmented = data.copy()
        for t in range(1, data.shape[1]):
            augmented[:, t, :, :] = (
                augmented[:, t-1, :, :] + movement[:, t-1, :, :] * scales[t]
            )
        return augmented

    def sustained_posture_augment(self, data, max_duration=20):
        """Simulate sustained postures characteristic of Dystonia."""
        augmented = data.copy()
        num_segments = np.random.randint(1, 4)

        for _ in range(num_segments):
            start = np.random.randint(0, data.shape[1] - max_duration)
            duration = np.random.randint(10, max_duration)

            # Smooth transition
            transition = np.linspace(0, 1, 5)
            target_pose = data[:, start, :, :].copy()

            for t in range(duration):
                if t < 5:
                    weight = transition[t]
                elif t > duration - 5:
                    weight = transition[duration - t - 1]
                else:
                    weight = 1.0
                augmented[:, start + t, :, :] = (
                    weight * target_pose +
                    (1 - weight) * data[:, start + t, :, :]
                )
        return augmented

    def speed_variation_augment(self, data, speed_range=(0.8, 1.2)):
        """Adjust overall speed of motion."""
        speed_factor = np.random.uniform(*speed_range)
        old_length = data.shape[1]
        new_length = int(old_length * speed_factor)

        if new_length <= 1:
            return data

        old_time = np.linspace(0, 1, old_length)
        new_time = np.linspace(0, 1, new_length)

        # Interpolate across time
        augmented = data.copy()
        for c in range(data.shape[0]):
            for v in range(data.shape[2]):
                for m in range(data.shape[3]):
                    # Two-step to avoid direct dimension mismatch
                    # 1) Interp original onto new_time
                    # 2) Re-interp it back to old_time
                    new_vals = np.interp(new_time, old_time, data[c, :, v, m])
                    augmented[c, :, v, m] = np.interp(old_time, new_time, new_vals)
        return augmented

    def tremor_augment(self, data, frequency_range=(4, 12)):
        """Add tremor-like oscillations."""
        time = np.linspace(0, data.shape[1]/30, data.shape[1])  # assume 30 fps
        tremor = np.zeros_like(data)

        for v in range(data.shape[2]):
            if np.random.rand() < 0.3:  # 30% chance to affect this joint
                freq = np.random.uniform(*frequency_range)
                amp = np.random.uniform(0, 0.02)
                oscillation = amp * np.sin(2 * np.pi * freq * time)
                tremor[:, :, v, :] = oscillation[None, :, None]
        return data + tremor
