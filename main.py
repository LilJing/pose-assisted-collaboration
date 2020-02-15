from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import build_model
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time
from datetime import datetime


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor for rewards (default: 0.95)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T', help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy', type=float, default=0.001, metavar='T', help='parameter for entropy (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--workers', type=int, default=6, metavar='W', help='how many training processes to use (default: 6)')
parser.add_argument('--testers', type=int, default=1, metavar='W', help='how many test processes to collect data (default: 1)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C (default: 20)')
parser.add_argument('--test-eps', type=int, default=20, metavar='M', help='maximum length of an episode (default: 20)')
parser.add_argument('--env', default='UnrealMCRoomLarge-DiscreteColorGoal-v5', metavar='ENV', help='environment to train on')
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-vision-model-dir', default=None, metavar='LMD', help='folder to load trained vision models from')
parser.add_argument('--load-pose-model-dir', default=None, metavar='LMD', help='folder to load trained pose models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model', default='multi-cnn-lstm-discrete', metavar='M', help='viison model type to use')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
parser.add_argument('--render', dest='render', action='store_true', help='render test ')
parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use an optimizer without shared statistics.')
parser.add_argument('--train-mode', type=int, default=-1, metavar='TM', help='training mode')
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='choose number of observations to stack')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--lstm-out', type=int, default=256, metavar='LO', help='lstm output size')
parser.add_argument('--sleep-time', type=int, default=10, metavar='LO', help='seconds')
parser.add_argument('--step-size', type=int, default=10000, metavar='LO', help='step size for lr schedule')
parser.add_argument('--max-step', type=int, default=2000000, metavar='LO', help='max learning steps')
parser.add_argument('--global-model', default='gru', metavar='M', help='pose model type to use')
parser.add_argument('--test-type', default='modelgate', metavar='M', help='if use our gate model')
parser.add_argument('--rnn-layer', type=int, default=1, metavar='S', help='rnn layer of global pose model(default: 1)')


if __name__ == '__main__':

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpu_ids[-1]))
    env = create_env(args.env, args)

    shared_model = build_model(
        env.observation_space, env.action_space, args, device)
    print('model', shared_model)
    model_state = shared_model.state_dict()

    if args.load_vision_model_dir is not None:
        vision_saved_state = torch.load(
            args.load_vision_model_dir,
            map_location=lambda storage, loc: storage)
        for k, v in model_state.items():
            if 'header' in k or 'policy' in k or 'gate' in k:
                model_state[k] = vision_saved_state[k]
        shared_model.load_state_dict(model_state)

    if args.load_pose_model_dir is not None:
        pose_saved_state = torch.load(
            args.load_pose_model_dir,
            map_location=lambda storage, loc: storage)
        for k, v in model_state.items():
            if 'pose_actor' in k:
                model_state[k] = pose_saved_state[k]
            if 'pose_BiRNN' in k:
                key = k.replace('pose_BiRNN', 'global_net.pose_BiRNN')
                model_state[k] = pose_saved_state[key]
        shared_model.load_state_dict(model_state)

    params = shared_model.parameters()
    shared_model.share_memory()

    if args.shared_optimizer:
        print ('share memory')
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)
    env.close()

    processes = []
    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()

    for rank in range(0, args.testers):
        p = mp.Process(target=test, args=(rank, args, shared_model, train_modes, n_iters, device))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer, train_modes, n_iters, device))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)
    for p in processes:
        time.sleep(args.sleep_time)
        p.join()






