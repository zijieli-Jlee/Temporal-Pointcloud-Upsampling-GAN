import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time
import os, sys
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from tensorboardX import SummaryWriter
from itertools import cycle

from utils import load_checkpoint, save_checkpoint, ensure_dir
from train_utils import dump_pointcloud_visualization
from upsampling_network import SRNet
from discriminator import FluidTempoDis, FluidSpatialDis
from tempo_dataset import get_dataloader
from train_step_final import tempo_gan_step

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def get_arguments(parser):
    # basic training settings
    # should set the learning rate quite small, since this is a fine tune stage
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='Spcifies learing rate for optimizer. (default: 3e-4)'
    )
    parser.add_argument(
        '--resume', action='store_true', help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_resume', type=str, help='Path to checkpoint to resume training. (default: None)'
    )
    parser.add_argument(
        '--iters', type=int, default=80000, help='Number of training iterations. (default: 80k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5000, help='Save model checkpoints every x iterations. (default: 10k)'
    )
    # ==================================
    # model options
    parser.add_argument(
        '--in_node_feats', type=int, default=3, help='Dimension of intput node feature. (default: 3)'
    )
    parser.add_argument(
        '--node_embedding', type=int, default=128, help='Dimension of node embeddings. (default: 128)'
    )
    parser.add_argument(
        '--R', type=int, default=0.10, help='Cutoff radius for fixed radius graph. (default: 0.10)'
    )
    # ===================================
    # for dataset
    parser.add_argument(
        '--train_dataset_path', type=str, default='../../data/train_data_0.025_fine', help='Path to dataset.'
    )
    parser.add_argument(
        '--test_dataset_path', type=str, default='../../data/test_data_0.025_fine', help='Path to dataset.'
    )
    parser.add_argument(
        '--train_sequence_num', type=int, default=20, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_sequence_num', type=int, default=4, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--sequence_length', type=int, default=200, help='How many snapshots in each sequence.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Size of batch.'
    )
    parser.add_argument(
        '--small_batch', action='store_true'
    )

    parser.add_argument(
        '--w', type=float, default=0.5, help='weight for tempo loss.'
    )
    parser.add_argument(
        '--cutoff', type=float, default=0.025, help='Cutoff distance for calculating masking loss.'
    )

    parser.add_argument(
        '--use_vel', action='store_true'
    )
    parser.add_argument(
        '--freeze_D', action='store_true'
    )

    parser.add_argument(
        '--dump_visualization', action='store_true'
    )

    opt = parser.parse_args()
    print('Using following options')
    print(opt)
    return opt


def build_sr_model(opt):
    net = SRNet(in_feats=opt.in_node_feats,
                node_emb_dim=opt.node_embedding,
                )

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return net


def build_tempo_model():
    net = FluidTempoDis(3)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return net


def build_spatial_model():
    net = FluidSpatialDis()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return net


def fetch_and_to_gpu(data_tuple):
    data_lst = []
    for data in data_tuple:
        if len(data.shape) != 1:
            data = data.cuda()
        data_lst.append(data)
    return data_lst


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train temporal consistent GUN")
    opt = get_arguments(parser)

    # add code for datasets
    print('Preparing the data')
    train_dataloader, test_dataloader = get_dataloader(opt)

    # instantiate network
    print('Building network')
    sr_net = build_sr_model(opt)
    sr_net = sr_net.cuda()

    tempo_dis = build_tempo_model()
    tempo_dis = tempo_dis.cuda()

    spatial_dis = build_spatial_model()
    spatial_dis = spatial_dis.cuda()

    # create optimizers
    sr_optim = torch.optim.Adam(sr_net.parameters(), lr=opt.lr)
    sr_scheduler = torch.optim.lr_scheduler.StepLR(sr_optim, 10000, gamma=0.7, last_epoch=-1)

    tempo_dis_optim = torch.optim.Adam(tempo_dis.parameters(), lr=0.33*opt.lr)
    tempo_dis_scheduler = torch.optim.lr_scheduler.StepLR(tempo_dis_optim, 10000, gamma=0.7, last_epoch=-1)

    spatial_dis_optim = torch.optim.Adam(spatial_dis.parameters(), lr=0.33*opt.lr)
    spatial_dis_scheduler = torch.optim.lr_scheduler.StepLR(spatial_dis_optim, 10000, gamma=0.7, last_epoch=-1)

    start_n_iter = 0

    if opt.resume:
        path_to_resume = opt.path_to_resume
        print(f'Resuming checkpoint from: {path_to_resume}')
        ckpt = load_checkpoint(path_to_resume)
        sr_net.load_state_dict(ckpt['sr_net'])
        start_n_iter = ckpt['n_iter']
        sr_optim.load_state_dict(ckpt['sr_optim'])
        sr_scheduler.load_state_dict(ckpt['sr_sched'])

        tempo_dis.load_state_dict(ckpt['tempo_dis'])
        tempo_dis_optim.load_state_dict(ckpt['tempo_optim'])
        tempo_dis_scheduler.load_state_dict(ckpt['tempo_sched'])

        spatial_dis.load_state_dict(ckpt['spatial_dis'])
        spatial_dis_optim.load_state_dict(ckpt['spatial_optim'])
        spatial_dis_scheduler.load_state_dict(ckpt['spatial_sched'])
        print("last checkpoint restored")

    # use tensorboardX
    writer = SummaryWriter()
    checkpoint_dir = os.path.join(opt.log_dir, 'model_ckpt')
    ensure_dir(checkpoint_dir)

    sample_dir = os.path.join(opt.log_dir, 'samples')
    ensure_dir(sample_dir)

    # now we start the main loop
    n_iter = start_n_iter
    # for loop going through dataset
    start_time = time.time()
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        tempo_train_data_iter = iter(train_dataloader)
        while True:
            sr_net.train()
            tempo_dis.train()
            spatial_dis.train()

            # prepare the data
            try:
                data_tp = next(tempo_train_data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                del tempo_train_data_iter
                tempo_train_data_iter = iter(train_dataloader)
                data_tp = next(tempo_train_data_iter)

            data_lst = fetch_and_to_gpu(data_tp)

            [highres_pos_left, highres_pos, highres_pos_right,
             highres_vel_left, highres_vel, highres_vel_right,
             lowres_pos_left, lowres_pos, lowres_pos_right,
             lowres_vel_left, lowres_vel, lowres_vel_right,
             h] = data_lst

            highres_pos_lst = [highres_pos_left, highres_pos, highres_pos_right]
            lowres_pos_lst = [lowres_pos_left, lowres_pos, lowres_pos_right]
            highres_vel_lst = [highres_vel_left, highres_vel, highres_vel_right]
            lowres_vel_lst = [lowres_vel_left, lowres_vel, lowres_vel_right]

            prepare_time = start_time - time.time()
            n_iter += 1
            pbar.update(1)

            loss_dict = tempo_gan_step(sr_net, spatial_dis, tempo_dis,
                                       lowres_pos_lst, lowres_vel_lst,
                                       highres_pos_lst, highres_vel_lst,
                                       1., opt, n_iter,
                                       sr_optim, tempo_dis_optim, spatial_dis_optim,
                                       freeze_D=opt.freeze_D)

            sr_scheduler.step()
            tempo_dis_scheduler.step()
            spatial_dis_scheduler.step()

            # udpate tensorboardX
            for k, v in loss_dict.items():
                writer.add_scalar(k, v, global_step=n_iter)
            pbar.set_description(', '.join("{!s}={:.4f}".format(key,val) for (key, val) in loss_dict.items()))

            # maybe do a test pass every N=1 epochs
            # if False:
            if (n_iter - 1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                print('Testing')
                # bring models to evaluation mode
                sr_net.eval()

                if opt.dump_visualization:
                    tempo_test_data_iter = iter(test_dataloader)
                    for j in range(4):
                        try:
                            data_tp = next(tempo_test_data_iter)
                        except StopIteration:
                            # StopIteration is thrown if dataset ends
                            # reinitialize data loader
                            del tempo_train_data_iter
                            tempo_train_data_iter = iter(test_dataloader)
                            data_tp = next(tempo_test_data_iter)
                        # data preparation
                        # prepare the data
                        data_lst = fetch_and_to_gpu(data_tp)
                        [highres_pos_left, highres_pos, highres_pos_right,
                         highres_vel_left, highres_vel, highres_vel_right,
                         lowres_pos_left, lowres_pos, lowres_pos_right,
                         lowres_vel_left, lowres_vel, lowres_vel_right,
                         h] = data_lst

                        with torch.no_grad():
                            if opt.use_vel and opt.in_node_feats == 6:
                                feature = torch.cat([lowres_pos, lowres_vel*0.025], dim=2)
                            else:
                                feature = lowres_pos
                            _, _, pred_pos = sr_net(feature, lowres_pos, hard_masking=True)


                        dump_pointcloud_visualization(highres_pos[0].detach().cpu().numpy(),
                                                      os.path.join(sample_dir, f'gt_iter:{n_iter}_{j}.png'))
                        dump_pointcloud_visualization(lowres_pos[0].detach().cpu().numpy(),
                                                      os.path.join(sample_dir, f'input_iter:{n_iter}_{j}.png'))
                        dump_pointcloud_visualization(pred_pos[0].detach().cpu().numpy(),
                                                      os.path.join(sample_dir, f'pred_iter:{n_iter}_{j}.png'))

                # save checkpoint if needed
                cpkt = {
                    'sr_net': sr_net.state_dict(),
                    'tempo_dis': tempo_dis.state_dict(),
                    'spatial_dis': spatial_dis.state_dict(),

                    'n_iter': n_iter,

                    'sr_optim': sr_optim.state_dict(),
                    'tempo_optim': tempo_dis_optim.state_dict(),
                    'spatial_optim': spatial_dis_optim.state_dict(),

                    'sr_sched': sr_scheduler.state_dict(),
                    'tempo_sched': tempo_dis_scheduler.state_dict(),
                    'spatial_sched': spatial_dis_scheduler.state_dict()

                }

                save_checkpoint(cpkt, os.path.join(checkpoint_dir, f'tpugan_checkpoint{n_iter}.ckpt'))

            if n_iter >= opt.iters:
                print('exiting...')
                exit()

