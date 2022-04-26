import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from sampling import farthest_point_sampling
from torch.utils.data import Dataset, DataLoader
np.random.seed(0)

class MSRAction3D(Dataset):
    def __init__(self,
                 root,
                 frames_per_clip=16,
                 num_points=2048,
                 step_between_clips=1,
                 train=True,
                 return_idx=False,
                 return_lowres=True):
        super(MSRAction3D, self).__init__()
        self.num_points = num_points
        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.train = train
        self.num_classes = max(self.labels) + 1
        self.return_index = return_idx
        self.return_lowres = return_lowres


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.train:
            index, t = self.index_map[idx]
            label = self.labels[index]

            video = self.videos[index]

            clip = [video[t+i*self.step_between_clips].copy() for i in range(self.frames_per_clip)]
            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                p[:, 1] = -p[:, 1]
                clip[i] = p[r, :]
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

            for v in clip:
                v /= 300.
            c = np.mean(clip[len(clip)//2], axis=0)
            clip -= c

            highres_pos_lst = []
            lowres_pos_lst = []

            for i, v in enumerate(clip):
                highres_pos = v.astype(np.float32)
                highres_pos_lst.append(highres_pos)
                if self.return_lowres:
                    k = int(self.num_points * 0.0625)
                    fps_idx, _ = farthest_point_sampling(highres_pos, k)
                    lowres_pos = highres_pos[fps_idx]
                    lowres_pos_lst.append(lowres_pos)

            return highres_pos_lst, lowres_pos_lst, label
        else:
            index, t = self.index_map[idx]

            video = self.videos[index]
            label = self.labels[index]

            clip = [video[t + i * self.step_between_clips].copy() for i in range(self.frames_per_clip)]
            for i, p in enumerate(clip):
                if p.shape[0] > self.num_points:
                    r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
                else:
                    repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                    r = np.random.choice(p.shape[0], size=residue, replace=False)
                    r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
                p[:, 1] = -p[:, 1]
                clip[i] = p[r, :]

            c_lst = []
            for v in clip:
                v /= 300.
                c = np.mean(v, axis=0)
                v -= c
                c_lst.append(c)
            highres_pos_lst = []
            lowres_pos_lst = []

            for i, v in enumerate(clip):
                highres_pos = v.astype(np.float32)
                highres_pos_lst.append(highres_pos)
                if self.return_lowres:
                    k = int(self.num_points * 0.0625)
                    fps_idx, _ = farthest_point_sampling(highres_pos, k)
                    lowres_pos = highres_pos[fps_idx]
                    lowres_pos_lst.append(lowres_pos)
            if self.return_index:
                return highres_pos_lst, lowres_pos_lst, c_lst, label, index
            return highres_pos_lst, lowres_pos_lst, c_lst, label


def get_dataloader(opt):
    train_dataset = MSRAction3D(root=opt.data_dir, frames_per_clip=3, num_points=2048)

    action_train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=False)

    test_dataset = MSRAction3D(root=opt.data_dir, frames_per_clip=3, num_points=2048, train=False)
    action_test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size,
                                 shuffle=True, num_workers=1, pin_memory=False)
    return action_train_dataloader, action_test_dataloader


def get_test_dataloader(test_dataset):
    action_test_dataloader = DataLoader(test_dataset, batch_size=8,
                                        shuffle=False, num_workers=2, pin_memory=False)
    return action_test_dataloader


if __name__ == '__main__':
    data_set = MSRAction3D(root='./MSR-Action3D', frames_per_clip=2, num_points=2048)
    for i in range(10, 1000, 10):
        #print(data_set[0][0][0])
        print(np.min(data_set[0][0][0], axis=0))
        print(np.max(data_set[0][0][0], axis=0))
        print(np.mean(data_set[0][0][0], axis=0))