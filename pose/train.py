from __future__ import division
from setproctitle import setproctitle as ptitle
import os
import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from environment import create_env
from model import build_model
from player_util import Agent


def train(rank, args, shared_model, optimizer, train_modes, n_iters, env=None):
    n_steps = 0
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Agent:{}'.format(rank)))
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    training_mode = args.train_mode
    env_name = args.env

    train_modes.append(training_mode)
    n_iters.append(n_iter)

    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')
    if env == None:
        env = create_env(env_name)

    params = shared_model.parameters()
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.env.reset()
    # prepare model
    player.model = build_model(action_space=player.env.action_space,
                               pose_space=player.reset_cam_pose(),
                               args=args,)
    player.model = player.model.to(device)
    player.model.train()

    player.reset()
    reward_sum = torch.zeros(player.num_agents).to(device)
    count_eps = 0
    print('Start training...')
    while True:
        # sys to the shared model
        player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            player.reset()
            reward_sum = torch.zeros(player.num_agents).to(device)
            count_eps += 1

        player.update_rnn_hidden()
        fps_counter = 0
        t0 = time.time()
        for i in range(args.num_steps):
            player.action_train()
            reward_sum += player.reward
            fps_counter += 1
            n_steps += 1

            if player.done:
                for i, r_i in enumerate(reward_sum):
                    # add for Pose Only
                    if i not in player.env.random_ids:
                        continue
                    #
                    writer.add_scalar('train/reward_' + str(i), r_i, n_steps)
                break

        fps = fps_counter / (time.time() - t0)

        policy_loss, value_loss, entropies, pred_loss, values0 = player.optimize(params, optimizer, shared_model, gpu_id)
        writer.add_scalar('train/policy_loss_sum', policy_loss.sum(), n_steps)
        writer.add_scalar('train/value_loss_sum', value_loss.sum(), n_steps)
        writer.add_scalar('train/entropies_sum', entropies.sum(), n_steps)
        writer.add_scalar('train/values0', values0.sum(), n_steps)
        writer.add_scalar('train/pred_R_loss', pred_loss, n_steps)
        writer.add_scalar('train/fps', fps, n_steps)
        # writer.add_scalar('train/lr', lr[0], n_iter)
        n_iter += 1
        n_iters[rank] = n_iter
        if train_modes[rank] == -100:
            env.close()
            break
