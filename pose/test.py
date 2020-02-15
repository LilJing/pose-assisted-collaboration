from __future__ import division
import os
import numpy as np
import torch
import time
import logging
from tensorboardX import SummaryWriter
from setproctitle import setproctitle as ptitle

from environment import create_env
from utils import setup_logger
from player_util import Agent
from model import build_model


def test(args, shared_model, train_modes, n_iters):
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}/logger'.format(args.log_dir))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env(args.env)

    start_time = time.time()
    num_tests = 1
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.env.reset()
    player.gpu_id = gpu_id
    player.model = build_model(
        action_space=player.env.action_space,
        pose_space=player.reset_cam_pose(),
        args=args,).to(device)

    player.state = player.env.reset()
    player.reset()

    player.model.eval()
    max_score = -100
    reward_sum = np.zeros(player.num_agents)
    reward_total_sum = np.zeros(player.num_agents)
    reward_sum_ep = np.zeros(player.num_agents)
    fps_counter = 0
    t0 = time.time()

    len_sum = 0
    seed = args.seed
    print('Start testing...')
    while True:
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            
        player.action_test()
        fps_counter += 1
        reward_sum_ep += player.reward.numpy()

        if player.done:
            reward_sum += reward_sum_ep
            len_sum += player.eps_len
            fps = fps_counter / (time.time()-t0)
            n_iter = 0
            for n in n_iters:
                n_iter += n
            for i in range(player.num_agents):
                random_ids = player.env.random_ids
                if i not in random_ids:
                    continue
                
                writer.add_scalar('test/reward'+str(i), reward_sum_ep[i], n_iter)

            writer.add_scalar('test/fps', fps, n_iter)
            writer.add_scalar('test/eps_len', player.eps_len, n_iter)
            player.eps_len = 0
            fps_counter = 0
            reward_sum_ep = np.zeros(player.num_agents)
            t0 = time.time()
            count_eps += 1
            if count_eps % args.test_eps == 0:
                player.max_length = True
            else:
                player.max_length = False

        if player.done and not player.max_length:
            seed += 1
            player.reset()

        elif player.done and player.max_length:
            ave_reward_sum = reward_sum/args.test_eps
            reward_total_sum += ave_reward_sum
            reward_mean = reward_total_sum / num_tests
            len_mean = len_sum/args.test_eps
            reward_step = reward_sum / len_sum
            log['{}_log'.format(args.env)].info(
                "Time {0}, ave eps reward {1}, ave eps length {2}, reward mean {3}, reward step {4}".
                format(
                   time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    ave_reward_sum, len_mean, reward_mean, reward_step))
            if ave_reward_sum.mean() >= max_score:
                print ('save best! in %d iters'%n_iter)
                max_score = ave_reward_sum.mean()
                model_dir = os.path.join(args.log_dir, '{0}-best-{1}.dat'.format(args.env, n_iter))
            else:
                model_dir = os.path.join(args.log_dir, '{0}-new.dat'.format(args.env))

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, model_dir)
            else:
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, model_dir)

            num_tests += 1
            reward_sum = 0
            len_sum = 0
            seed += 1
            player.reset()
            time.sleep(args.sleep_time)

            if n_iter > args.max_step:
                env.close()
                for id in range(0, args.workers):
                    train_modes[id] = -100
                break


