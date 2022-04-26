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
from train_utils import dump_pointcloud_visualization, visualize_pointcloud
from msr_dataset import get_dataloader

from upsampling_network import NoMaskSRNet
from discriminator import ActionTempoDis, ActionSpatialDis
from train_step_final import tempo_gan_step_no_mask

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.set_num_threads(4)

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
        '--path_to_resume', type=str, default='', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--iters', type=int, default=80000, help='Number of training iterations. (default: 80k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5000, help='Save model checkpoints every x iterations. (default: 5k)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Size of batch.'
    )
    parser.add_argument(
        '--data_dir', type=str, default='../../data/MSR-Action3D', help='directory to data'
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
        '--R', type=int, default=2.0, help='Cutoff radius for fixed radius graph. (default: 2.0)'
    )

    parser.add_argument(
        '--w', type=float, default=2.0, help='weight for tempo loss.'
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
    net = NoMaskSRNet(in_feats=opt.in_node_feats,
                     node_emb_dim=opt.node_embedding,
                     upsample_ratio=16
                    )

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return net


def build_tempo_dis():
    net = ActionTempoDis(3, sn=True)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return net


def build_spatial_dis():
    net = ActionSpatialDis(sn=True)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return net


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train Action upsampling network")
    opt = get_arguments(parser)

    # add code for datasets
    print('Preparing the data')
    train_dataloader, test_dataloader = get_dataloader(opt)

    # instantiate network
    print('Building network')
    sr_net = build_sr_model(opt)
    sr_net = sr_net.cuda()
    tempo_dis = build_tempo_dis()
    tempo_dis = tempo_dis.cuda()
    spatial_dis = build_spatial_dis()
    spatial_dis = spatial_dis.cuda()

    # create optimizers
    sr_optim = torch.optim.Adam(sr_net.parameters(), lr=opt.lr)
    sr_scheduler = torch.optim.lr_scheduler.StepLR(sr_optim, opt.iters//10, gamma=0.72, last_epoch=-1)

    tempo_dis_optim = torch.optim.Adam(tempo_dis.parameters(), lr=opt.lr*0.33)
    tempo_dis_scheduler = torch.optim.lr_scheduler.StepLR(tempo_dis_optim, opt.iters//10,
                                                          gamma=0.72, last_epoch=-1)

    spatial_dis_optim = torch.optim.Adam(spatial_dis.parameters(), lr=opt.lr*0.33)
    spatial_dis_scheduler = torch.optim.lr_scheduler.StepLR(spatial_dis_optim, opt.iters//10,
                                                            gamma=0.72, last_epoch=-1)
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

    # typically we use tensorboardX to keep track of experiments
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

            highres_pos_lst = data_tp[0]
            lowres_pos_lst = data_tp[1]

            highres_pos_lst = [highres_pos.cuda() for highres_pos in highres_pos_lst]
            lowres_pos_lst = [lowres_pos.cuda() for lowres_pos in lowres_pos_lst]

            prepare_time = start_time - time.time()
            n_iter += 1
            pbar.update(1)

            loss_dict = tempo_gan_step_no_mask(sr_net,
                                            spatial_dis,
                                            tempo_dis,
                                            lowres_pos_lst,
                                            highres_pos_lst,
                                            opt,
                                            n_iter,
                                            sr_optim,
                                            tempo_dis_optim,
                                            spatial_dis_optim,
                                            opt.freeze_D
                                            )


            sr_scheduler.step()
            tempo_dis_scheduler.step()
            spatial_dis_scheduler.step()

            # udpate tensorboardX
            for k, v in loss_dict.items():
                writer.add_scalar(k, v, global_step=n_iter)
            pbar.set_description(', '.join("{!s}={:.4f}".format(key, val) for (key, val) in loss_dict.items()))

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(', '.join("{!s}={:.4f}".format(key,val) for (key, val) in loss_dict.items()))
            start_time = time.time()

            if (n_iter - 1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                print('Testing')
                # bring models to evaluation mode
                sr_net.eval()

                if opt.dump_visualization:
                    tempo_test_data_iter = iter(test_dataloader)
                    with torch.no_grad():
                        for j in range(4):
                            try:
                                data_tp = next(tempo_test_data_iter)
                            except StopIteration:
                                # StopIteration is thrown if dataset ends
                                # reinitialize data loader
                                del tempo_test_data_iter
                                tempo_test_data_iter = iter(test_dataloader)
                                data_tp = next(tempo_test_data_iter)
                            # data preparation
                            # prepare the data
                            highres_pos_lst = data_tp[0]
                            lowres_pos_lst = data_tp[1]
                            highres_pos = highres_pos_lst[0][0].cuda()
                            lowres_pos = lowres_pos_lst[0][0].cuda()
                            feature = lowres_pos
                            pred_pos, _ = sr_net(feature, lowres_pos)
                            pred_pos = pred_pos.squeeze(0)

                            dump_pointcloud_visualization(highres_pos.detach().cpu().numpy(),
                                                          os.path.join(sample_dir, f'gt_iter:{n_iter}_{j}.png'))
                            dump_pointcloud_visualization(lowres_pos.detach().cpu().numpy(),
                                                          os.path.join(sample_dir, f'input_iter:{n_iter}_{j}.png'))
                            dump_pointcloud_visualization(pred_pos.detach().cpu().numpy(),
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


