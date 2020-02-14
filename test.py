from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger, check_path
from player_util import Agent
import logging
from tensorboardX import SummaryWriter
import os
from model import build_model
import torch.nn as nn
import time

def test(rank, args, shared_model, train_modes, n_iters, device):
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test Agent:{}'.format(rank)))
    ptitle('Test Agent: {}'.format(rank))
    torch.manual_seed(args.seed + rank)
    n_iter = 0

    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}/logger'.format(args.log_dir))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)

    if args.env_base is None:
        env = create_env(args.env, args)
    else:
        env = create_env(args.env_base, args)

    start_time = time.time()
    num_tests = 1
    n_step = 0

    player = Agent(None, env, args, None, None, device)
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device).to(device)

    player.state = player.env.reset()
    if 'Unreal' in args.env:
        player.cam_pos = player.env.env.env.env.cam_pose
        player.collect_state = player.env.env.env.env.current_states

    player.set_cam_info()
    player.state = torch.from_numpy(player.state).float().to(device)

    player.model.eval()
    max_score = -100
    reward_sum = np.zeros(player.num_agents)
    reward_total_sum = np.zeros(player.num_agents)
    reward_sum_ep = np.zeros(player.num_agents)

    success_rate_sum_ep = np.zeros(player.num_agents)

    fps_counter = 0
    t0 = time.time()
    cross_entropy_loss = nn.CrossEntropyLoss()

    len_sum = 0
    seed = args.seed

    count_eps = 0
    eps_length = 0
    rate = 0
    rates = [0, 0]
    step_rates = [0, 0]
    mean_rates = [0, 0]

    visible_steps = 0
    while True:
        if player.done:
            count_eps += 1

            t0 = time.time()
            eps_length = 0

            player.model.load_state_dict(shared_model.state_dict())

        player.action_test()
        eps_length += 1
        n_step += 1

        fps_counter += 1
        reward_sum_ep += player.reward
        success_rate_sum_ep += player.success_rate

        gate_ids, gate_probs, gt_gates = [], [], []
        for k1 in range(len(player.rewards)):
            for k2 in range(player.num_agents):
                _, max_id = torch.max(player.gates[k1][k2], 0)
                gate_probs.append(player.gates[k1][k2])
                gate_ids.append(max_id)
                gt_gates.append(player.gate_gts[k1][k2])

        gate_probs = torch.cat(gate_probs).view(-1, 2).to(device)
        gate_gt_ids = torch.Tensor(gt_gates).view(1, -1).squeeze().long().to(device)
        gate_loss = cross_entropy_loss(gate_probs, gate_gt_ids)

        visible_steps += sum(np.array(gt_gates).squeeze()) / 4

        gate_ids = np.array([gate_ids[i].cpu().detach().numpy() for i in range(4)])
        gt_gates = np.array([gt_gates[i].cpu().detach().numpy() for i in range(4)])
        one_step_rate = sum(gate_ids == gt_gates) / player.num_agents
        rate += one_step_rate
        for id in range(2):
            right_num = sum(gate_ids[i] == gt_gates[i] == id for i in range(4))
            num = sum(gt_gates[i] == id for i in range(4))
            step_rate = right_num / num if num != 0 else 0
            if step_rate > 0:
                rates[id] += step_rate
                step_rates[id] += 1
                mean_rates[id] = rates[id] / step_rates[id]

        mean_rate = rate / n_step

        if player.done:
            player.state = player.env.reset()
            player.state = torch.from_numpy(player.state).float().to(device)
            player.set_cam_info()

            reward_sum += reward_sum_ep

            len_sum += player.eps_len
            fps = fps_counter / (time.time()-t0)
            n_iter = 0
            for n in n_iters:
                n_iter += n
            for i in range(player.num_agents):
                writer.add_scalar('test/reward'+str(i), reward_sum_ep[i], n_iter)

            writer.add_scalar('test/fps', fps, n_iter)
            writer.add_scalar('test/eps_len', player.eps_len, n_iter)
            writer.add_scalar('test/unvisible_acc', mean_rates[0], n_iter)
            writer.add_scalar('test/visible_acc', mean_rates[1], n_iter)
            writer.add_scalar('test/mean_acc', mean_rate, n_iter)
            writer.add_scalar('test/gate_loss', gate_loss, n_iter)

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
            player.env.seed(seed)
            player.state = player.env.reset()
            player.set_cam_info()
            player.state = torch.from_numpy(player.state).float().to(device)

            player.eps_len += 2

        elif player.done and player.max_length:
            ave_reward_sum = reward_sum/args.test_eps
            reward_total_sum += ave_reward_sum
            reward_mean = reward_total_sum / num_tests
            len_mean = len_sum/args.test_eps
            reward_step = reward_sum / len_sum
            log['{}_log'.format(args.env)].info(
                "Time {0}, ave eps reward {1}, ave eps length {2}, reward mean {3}, reward step {4}".
                format(
                   time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    ave_reward_sum, len_mean, reward_mean, reward_step))

            if ave_reward_sum.mean() >= max_score:
                print ('save best! in %d iters'%n_step)
                max_score = ave_reward_sum.mean()
                model_dir = os.path.join(args.log_dir, '{0}-gate-all-model-best-{1}.dat'.format(args.env, n_step))
            else:
                model_dir = os.path.join(args.log_dir, '{0}-new.dat'.format(args.env))

            if args.gpu_ids[-1] >= 0:
                with torch.cuda.device(args.gpu_ids[-1]):
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, model_dir)
            else:
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, model_dir)

            num_tests += 1
            reward_sum = 0
            len_sum = 0
            seed += 1
            player.env.seed(seed)

            player.state = player.env.reset()
            if 'Unreal' in args.env:
                player.cam_pos = player.env.env.env.env.cam_pose
                player.collect_state = player.env.env.env.env.current_states
            player.set_cam_info()
            player.state = torch.from_numpy(player.state).float().to(device)
            player.input_actions = torch.Tensor(np.zeros((player.num_agents, 9)))

            time.sleep(args.sleep_time)

            if n_iter > args.max_step:
                env.close()
                for id in range(0, args.workers):
                    train_modes[id] = -100
                break

        player.clear_actions()

