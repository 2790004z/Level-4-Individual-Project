# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools


class Feeder(torch.utils.data.Dataset):
    """Feeder for skeleton-based movement disorder recognition.
    
    This feeder includes specialized augmentation techniques for Dystonia and 
    Choreoathetosis recognition while maintaining the original ST-GCN structure.
    
    Args:
        data_path: Path to the .npy data file (N, C, T, V, M)
        label_path: Path to the label file
        random_choose: If true, randomly choose a portion of the input sequence
        random_move: If true, randomly pad zeros at the beginning or end of sequence
        window_size: The length of the output sequence
        augment: If true, apply movement disorder-specific augmentations
        mixup: If true, apply mixup augmentation
        debug: If true, only use the first 100 samples
        mmap: If true, use memory mapping for large datasets
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

        # Load movement data
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
        # Get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # Processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # Apply augmentation during training
        if self.augment:
            data_numpy = self.apply_augmentation(data_numpy)

        return data_numpy, label

    def apply_augmentation(self, data):
        """Apply movement disorder-specific augmentations."""
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
        # Calculate frame-to-frame movement
        movement = np.diff(data, axis=1)  # Temporal difference
        
        # Generate smooth intensity scaling
        intensity = np.random.uniform(*intensity_range)
        scales = np.ones(data.shape[1]) * intensity
        
        # Apply temporal smoothing to scales
        kernel_size = 5
        scales = np.convolve(scales, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Reconstruct movement with adjusted intensity
        augmented = data.copy()
        for t in range(1, data.shape[1]):
            augmented[:, t, :, :] = (augmented[:, t-1, :, :] + 
                                    movement[:, t-1, :, :] * scales[t])
        return augmented

    def sustained_posture_augment(self, data, max_duration=20):
        """Simulate sustained postures characteristic of Dystonia."""
        augmented = data.copy()
        num_segments = np.random.randint(1, 4)
        
        for _ in range(num_segments):
            # Select random segment for sustained posture
            start = np.random.randint(0, data.shape[1] - max_duration)
            duration = np.random.randint(10, max_duration)
            
            # Create smooth transition weights
            transition = np.linspace(0, 1, 5)
            
            # Apply sustained posture with smooth transitions
            target_pose = data[:, start, :, :].copy()
            for t in range(duration):
                if t < 5:  # Smooth start
                    weight = transition[t]
                elif t > duration - 5:  # Smooth end
                    weight = transition[duration - t - 1]
                else:  # Full sustain
                    weight = 1.0
                    
                augmented[:, start + t, :, :] = (
                    weight * target_pose + 
                    (1 - weight) * data[:, start + t, :, :]
                )
        
        return augmented

    def speed_variation_augment(self, data, speed_range=(0.8, 1.2)):
        """Adjust movement speed while preserving patterns."""
        speed_factor = np.random.uniform(*speed_range)
        
        # Interpolate temporal dimension
        old_length = data.shape[1]
        new_length = int(old_length * speed_factor)
        
        if new_length <= 1:
            return data
            
        old_time = np.linspace(0, 1, old_length)
        new_time = np.linspace(0, 1, new_length)
        
        # Interpolate each channel and joint separately
        augmented = np.zeros_like(data)
        for c in range(data.shape[0]):  # Channels
            for v in range(data.shape[2]):  # Joints
                for m in range(data.shape[3]):  # People
                    augmented[c, :, v, m] = np.interp(old_time, new_time,
                        np.interp(new_time, old_time, data[c, :, v, m]))
        
        return augmented

    def tremor_augment(self, data, frequency_range=(4, 12)):
        """Add tremor-like oscillations to movement patterns."""
        # Generate tremor signal
        time = np.linspace(0, data.shape[1]/30, data.shape[1])  # Assuming 30 fps
        tremor = np.zeros_like(data)
        
        # Apply tremor to random joints
        for v in range(data.shape[2]):  # Joints
            if np.random.rand() < 0.3:  # Only affect some joints
                freq = np.random.uniform(*frequency_range)
                amp = np.random.uniform(0, 0.02)
                oscillation = amp * np.sin(2 * np.pi * freq * time)
                tremor[:, :, v, :] = oscillation[None, :, None]
        
        return data + tremor