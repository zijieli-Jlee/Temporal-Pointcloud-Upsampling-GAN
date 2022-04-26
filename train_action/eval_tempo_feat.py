import os
import torch
import torch.nn.functional as F
import numpy as np
import sys

import datetime
import logging
import argparse

from pathlib import Path
from tqdm import tqdm
from msr_dataset import MSRAction3D

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import load_checkpoint
from discriminator import ActionTempoDis, ActionCls
from train_utils import visualize_pointcloud


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--pretrained_ckpt', type=str, help='path to pretrained ckpt')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--epoch', default=201, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='./', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    return parser.parse_args()


def model_wrapper(pos_lst, model:ActionCls):
    pred = model(pos_lst, 2.0)
    logits = F.log_softmax(pred, dim=-1)
    return logits


def test(model, loader, log_fun):
    model.eval()
    video_prob = {}
    video_label = {}

    with torch.no_grad():
        for clip, _, _, target, video_idx in loader:
            clip = [video.cuda() for video in clip]
            output = model_wrapper(clip, model)
            prob = torch.exp(output)
            batch_size = output.shape[0]
            target = target.numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * loader.dataset.num_classes
    class_correct = [0] * loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v == label)
    class_acc = [c / float(s) for c, s in zip(class_correct, class_count)]
    log_fun(' * Video Acc@1 %f' % total_acc)
    log_fun(' * Class Acc@1 %s' % str(class_acc))

    return total_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'transfer_tempo_dis'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    train_dataset = MSRAction3D(root=args.data_path, train=True, frames_per_clip=3, return_lowres=False)
    test_dataset = MSRAction3D(root=args.data_path, train=False, frames_per_clip=3, return_idx=True, return_lowres=False)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=64, shuffle=True,
                                                  num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=128, shuffle=False,
                                                 num_workers=4)

    '''MODEL LOADING'''
    classifier = ActionCls(3)

    pretrained_model = ActionTempoDis(3, sn=True)
    path_to_resume = args.pretrained_ckpt
    ckpt = load_checkpoint(path_to_resume)
    pretrained_model.load_state_dict(ckpt['tempo_dis'])

    print("last checkpoint restored")
    classifier.init_feature_extractor(pretrained_model)

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'Total trainable parameters for SA-MLP: {total_params}')
    classifier = classifier.cuda()
    del pretrained_model
    criterion = torch.nn.NLLLoss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(lambda p: p.requires_grad, classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_loss = []
        classifier = classifier.train()

        for batch_id, (clip, _, target) in \
                tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            target = target.cuda()
            pred = model_wrapper([video.cuda() for video in clip], classifier)
            loss = criterion(pred, target.long())
            mean_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()

        train_loss = np.mean(mean_loss)
        log_string('Train Loss: %f' % train_loss)
        if epoch % 10 == 0:
            with torch.no_grad():
                total_acc = test(classifier.eval(), testDataLoader, log_string)
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + f'/model_epoch:{epoch}.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'total_acc': total_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
