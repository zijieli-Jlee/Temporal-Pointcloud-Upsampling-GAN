import torch
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train_utils import sample_patch_feat, normalize_point_cloud, visualize_pointcloud, sample_patch_with_fps
from sampling import farthest_point_sampling as fps
import torch.multiprocessing
from functools import partial
import random
torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_feature(feature):
    mean = np.mean(feature)
    return (feature - mean), mean


class SiamData(Dataset):
    def __init__(self,
                 dataset_path,
                 case_num,
                 case_steps,
                 case_prefix='data',
                 case_to_start=1,
                 sample_num=4096,
                 jitter=0.003,
                 cache_size=2000):
        self.dataset_path = dataset_path
        self.case_num = case_num
        self.case_steps = case_steps
        self.case_prefix = case_prefix
        self.case_to_start = case_to_start
        self.sample_num = sample_num

        self.jitter = jitter
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return self.case_num * (self.case_steps - 2)

    def get_data_from_cache(self, key):
        if self.cache_size == 0:   # no cache used
            data_path = os.path.join(self.dataset_path, key)
            data = np.load(data_path)
        else:
            if not key in self.cache.keys():
                data_path = os.path.join(self.dataset_path, key)
                if len(self.cache.keys()) >= self.cache_size:
                    self.cache.pop(random.choice(list(self.cache.keys())))
                data = np.load(data_path)
                self.cache[key] = data
            else:
                data = self.cache[key]
        return data

    def __getitem__(self, idx):
        case_to_read = idx // self.case_steps + self.case_to_start
        step_to_read = idx % (self.case_steps - 2)
        left_key = f'case{case_to_read}/{self.case_prefix}_{step_to_read}.npz'
        center_key = f'case{case_to_read}/{self.case_prefix}_{step_to_read + 1}.npz'
        right_key = f'case{case_to_read}/{self.case_prefix}_{step_to_read + 2}.npz'

        left_data = self.get_data_from_cache(left_key)
        center_data = self.get_data_from_cache(center_key)
        right_data = self.get_data_from_cache(right_key)

        pos_center, m, h = normalize_point_cloud(center_data['pos'].astype(np.float32))
        pos_left = (left_data['pos'].astype(np.float32) - m) / h
        pos_right = (right_data['pos'].astype(np.float32) - m) / h

        vel_center = center_data['vel'].astype(np.float32) / h
        vel_left = left_data['vel'].astype(np.float32) / h
        vel_right = right_data['vel'].astype(np.float32) / h

        # data_dict = sample_patch_feat(pos, vel, h, return_next_frame_sample=True, next_frame_pos=pos_right)
        data_dict, patch_idx, fps_idx = sample_patch_with_fps(pos_center, h,
                                                              sample_num=self.sample_num,
                                                              return_free_surface_particles=False,
                                                              return_patch_and_fps_idx=True)
        highres_pos = data_dict['patch_pos']
        highres_vel = vel_center[patch_idx]
        highres_pos_left = pos_left[patch_idx]
        highres_pos_right = pos_right[patch_idx]
        highres_vel_left = vel_left[patch_idx]
        highres_vel_right = vel_right[patch_idx]

        lowres_pos = highres_pos[fps_idx]
        left_fps_idx = right_fps_idx = fps_idx

        lowres_pos += np.random.randn(lowres_pos.shape[0], lowres_pos.shape[1]) * self.jitter
        lowres_pos_left = highres_pos_left[left_fps_idx] + \
                          np.random.randn(lowres_pos.shape[0], lowres_pos.shape[1]).astype(np.float32) * self.jitter
        lowres_pos_right = highres_pos_right[right_fps_idx] + \
                           np.random.randn(lowres_pos.shape[0], lowres_pos.shape[1]).astype(np.float32) * self.jitter

        lowres_vel = vel_center[fps_idx]
        lowres_vel_left = vel_left[left_fps_idx]
        lowres_vel_right = vel_right[right_fps_idx]

        return highres_pos_left, highres_pos, highres_pos_right, \
               highres_vel_left, highres_vel, highres_vel_right, \
               lowres_pos_left, lowres_pos, lowres_pos_right,  \
               lowres_vel_left, lowres_vel, lowres_vel_right, h


def my_collate(desired_size, batch):
    filtered_batch = list(filter (lambda x: x[1].shape[0] == desired_size, batch))
    if len(filtered_batch) <= 1:
        filtered_batch = list(filter(lambda x: x[1].shape[0] == 4096, batch))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def get_dataloader(opt):
    train_dataset = SiamData(opt.train_dataset_path, opt.train_sequence_num, opt.sequence_length,
                             sample_num=9216 if opt.batch_size <= 4 and not opt.small_batch else 4096)
    test_dataset = SiamData(opt.test_dataset_path, opt.test_sequence_num, opt.sequence_length, sample_num=None, cache_size=0)
    wrapped_collate = partial(my_collate, 9216)
    tempo_train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=True,
                                        collate_fn=wrapped_collate)  # in some version of scipy, kDTree seems violating multi workers dataloader
    tempo_test_dataloader = DataLoader(test_dataset, batch_size=1,
                                        shuffle=True, num_workers=0, pin_memory=False)

    return tempo_train_dataloader, tempo_test_dataloader


def get_tempo_test_dataloader(opt, sample_num=None, batch_size=1):
    if sample_num is None:
        sample_num = 10240
    test_dataset = SiamData(opt.dataset_path, opt.sequence_num, opt.sequence_length, sample_num=sample_num)

    tempo_test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=False)

    return tempo_test_dataloader


def get_pos_test_dataloader(opt, sample_num=None, batch_size=1):
    if sample_num is None:
        sample_num = 11264          # gives roughly the same emd as reported in tranquil clouds
    test_dataset = SiamData(opt.dataset_path, opt.sequence_num, opt.sequence_length, sample_num=sample_num,
                            jitter=0.0)

    pos_test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=False)

    return pos_test_dataloader