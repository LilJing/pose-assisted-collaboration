from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from torch.nn import L1Loss
from environment import create_env
from utils import ensure_shared_grads, ensure_shared_grads_param
from model import build_model
from player_util import Agent
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import time
import torch.nn as nn

def train(rank, args, shared_model, optimizer, train_modes, n_iters, device, env=None):
    n_steps = 0
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Agent:{}'.format(rank)))
    ptitle('Training Agent: {}'.format(rank))
    torch.manual_seed(args.seed + rank)
    training_mode = args.train_mode
    env_name = args.env

    train_modes.append(training_mode)
    n_iters.append(n_iter)

    if env == None:
        env = create_env(env_name, args)

    params = shared_model.parameters()

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None, None, device)
    player.model = build_model(
        player.env.observation_space, player.env.action_space, args, device).to(device)

    player.state = player.env.reset()
    if 'Unreal' in args.env:
        player.cam_pos = player.env.env.env.env.cam_pose
        player.collect_state = player.env.env.env.env.current_states
    player.set_cam_info()
    player.state = torch.from_numpy(player.state).float()
    player.state = player.state.to(device)
    player.model = player.model.to(device)

    player.model.train()
    reward_sum = torch.zeros(player.num_agents).to(device)
    count_eps = 0
    cross_entropy_loss = nn.CrossEntropyLoss()

    while True:
        player.model.load_state_dict(shared_model.state_dict())
        player.update_lstm()
        fps_counter = 0
        t0 = time.time()
        for step in range(args.num_steps):
            player.action_train()
            n_steps += 1
            reward_sum += player.reward
            if player.done:
                break
        update_steps = len(player.rewards)

        fps = fps_counter / (time.time() - t0)

        if player.done:
            for i in range(player.num_agents):
                writer.add_scalar('train/reward_'+str(i), reward_sum[i], n_steps)
            count_eps += 1
            reward_sum = torch.zeros(player.num_agents).to(device)
            player.eps_len = 0
            player.state = player.env.reset()
            player.set_cam_info()
            player.state = torch.from_numpy(player.state).float().to(device)

        R = torch.zeros(player.num_agents, 1).to(device)

        if not player.done:
            state = player.state
            value_multi, _, _, _, _, _, _, _ , _= player.model(
                    (Variable(state, requires_grad=True),
                     Variable((player.cam_info), requires_grad=True), player.H_multi,
                     player.last_gate_ids, player.gt_gate))
            for i in range(player.num_agents):
                R[i][0] = value_multi[i].data

        gates, gt_gates = [], []
        for k1 in range(len(player.rewards)):
            for k2 in range(player.num_agents):
                gates.append(player.gates[k1][k2])
                gt_gates.append(player.gate_gts[k1][k2])

        gate_probs = torch.cat(gates).view(-1, 2).to(device)
        gate_gt_ids = torch.Tensor(gt_gates).view(1, -1).squeeze().long().to(device)
        gate_loss = cross_entropy_loss(gate_probs, gate_gt_ids)

        player.values.append(Variable(R).to(device))
        policy_loss = torch.zeros(player.num_agents, 1).to(device)
        value_loss = torch.zeros(player.num_agents, 1).to(device)
        pred_loss = torch.zeros(1, 1).to(device)
        entropies = torch.zeros(player.num_agents, 1).to(device)

        w_entropies = torch.Tensor([[float(args.entropy)] for i in range(player.num_agents)]).to(device)

        R = Variable(R, requires_grad=True).to(device)
        gae = torch.zeros(1, 1).to(device)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i] * Variable(gae)) - \
                (w_entropies * player.entropies[i])

            entropies += player.entropies[i]

        loss = policy_loss.sum() / update_steps / player.num_agents + 0.5 * value_loss.sum() / update_steps / player.num_agents + \
               5 * gate_loss

        player.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 50)
        ensure_shared_grads(player.model, shared_model, gpu=args.gpu_ids[-1] >= 0)

        writer.add_scalar('train/policy_loss_sum', policy_loss.sum(), n_steps)
        writer.add_scalar('train/value_loss_sum', value_loss.sum(), n_steps)
        writer.add_scalar('train/entropies_sum', entropies.sum(), n_steps)
        writer.add_scalar('train/fps', fps, n_steps)
        writer.add_scalar('train/gate_loss', gate_loss, n_steps)

        n_iter += 1
        n_iters[rank] = n_iter

        optimizer.step()

        player.clear_actions()

        if train_modes[rank] == -100:
            env.close()
            break
