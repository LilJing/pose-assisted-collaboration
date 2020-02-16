from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable

import sys
sys.path.append('../')
from utils import ensure_shared_grads

class Agent(object):
    def __init__(self, model, env, args, state, device):
        self.model = model
        self.env = env
        self.num_agents = len(env.action_space)
        if 'discrete' not in args.model:
            if type(env.action_space) == list:
                self.action_high = [env.action_space[i].high for i in range(self.num_agents)]
                self.action_low = [env.action_space[i].low for i in range(self.num_agents)]
            else:
                self.action_high = [env.action_space.high, env.action_space.high]
                self.action_low = [env.action_space.low, env.action_space.low]
        self.state = state

        self.lstm_out = args.lstm_out
        self.img_hxs = torch.zeros(self.num_agents, self.lstm_out)
        self.img_cxs = torch.zeros(self.num_agents, self.lstm_out)
        self.pose_hxs = torch.zeros(self.num_agents, self.lstm_out)
        self.pose_cxs = torch.zeros(self.num_agents, self.lstm_out)
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.preds = []
        self.done = True
        self.info = None
        self.pre_actions = None
        self.reward = 0
        self.device = device
        self.lstm_out = args.lstm_out
        self.reward_mean = None
        self.reward_std = 1
        self.num_steps = 0
        self.vk = 0
        self.render = args.render
        self.distance = args.distance
        self.angle = args.angle
        self.pose_reset_type = int(args.env.split('-v')[1])

    def wrap_action(self, action, high, low):
        action = np.squeeze(action)
        out = action * (high - low) / 2.0 + (high + low) / 2.0
        return out


    def action_train(self):
        self.gate_ids = self.env.gate_ids
        self.random_ids = self.env.random_ids

        value_multi, action_env_multi, (self.img_hxs, self.img_cxs), (self.pose_hxs, self.pose_cxs), entropy, log_prob, R_pred = self.model(
            (
                (Variable(self.state, requires_grad=True), (self.img_hxs, self.img_cxs)),
                (Variable(torch.Tensor(self.cam_info), requires_grad=True), (self.pose_hxs, self.pose_cxs)),
                (Variable(torch.Tensor(self.pre_actions), requires_grad=True),
                 Variable(torch.Tensor(self.gate_ids)),
                 Variable(torch.Tensor(self.random_ids)))
            )
        )
        if 'discrete' not in self.args.model:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]
        self.pre_actions = torch.zeros((self.num_agents, self.env.action_space[0].n))
        for i in range(self.num_agents):
            self.pre_actions[i][action_env_multi[i]] = 1
        # model return action_env_multi, entropy, log_prob

        state_multi, reward_multi, self.done, self.info = self.env.step(action_env_multi, False)

        # add to buffer
        self.local_reward = torch.tensor(reward_multi).float().to(self.device)
        self.state = torch.from_numpy(state_multi).float().to(self.device)
        if 'ns' in self.args.obs:
            self.set_noise_cam_info()
        elif 'gs' in self.args.obs:
            self.set_gs_noise_cam_info()
        elif self.distance:
            self.set_cam_info_d()
        elif self.angle:
            self.set_cam_info_angle()
        elif self.pose_reset_type == 2:
            self.set_cam_info_all()
        else:
            self.set_cam_info()
        self.eps_len += 1
        self.values.append(value_multi)

        self.reward = torch.zeros(len(self.local_reward)).float().to(self.device)
        self.reward += self.local_reward

        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward.unsqueeze(1))
        self.preds.append(R_pred)
        return self

    def action_test(self):
        self.gate_ids = self.env.gate_ids
        self.random_ids = self.env.random_ids
        with torch.no_grad():
            value_multi, action_env_multi, (self.img_hxs, self.img_cxs), (self.pose_hxs, self.pose_cxs), entropy, log_prob, R_pred = self.model(
                (
                    (Variable(self.state), (self.img_hxs, self.img_cxs)),
                    (Variable(torch.Tensor(self.cam_info)), (self.pose_hxs, self.pose_cxs)),
                    (Variable(torch.Tensor(self.pre_actions)),
                     Variable(torch.Tensor(self.gate_ids)),
                     Variable(torch.Tensor(self.random_ids)))
                ), True
            )

        if 'discrete' not in self.args.model:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i]) for i in range(self.num_agents)]
        self.pre_actions = torch.zeros((self.num_agents, self.env.action_space[0].n))
        for i in range(self.num_agents):
            self.pre_actions[i][action_env_multi[i]] = 1

        state_multi, reward_multi, self.done, self.info = self.env.step(action_env_multi, self.render)
        self.local_reward = torch.tensor(reward_multi).float().to(self.device)

        self.reward = torch.zeros(len(self.local_reward)).float().to(self.device)
        self.reward += self.local_reward

        self.state = torch.from_numpy(state_multi).float().to(self.device)

        if self.distance:
            self.set_cam_info_d()
        elif self.angle:
            self.set_cam_info_angle()
        elif self.pose_reset_type == 2:
            self.set_cam_info_all()
        else:
            self.set_cam_info()
        self.eps_len += 1
        return self

    def reset(self):
        self.state = torch.from_numpy(self.env.reset()).float().to(self.device)
        self.num_agents = self.state.shape[0]
        self.eps_len = 0
        self.reset_rnn_hidden()
        if 'PoseEnv' in self.args.env:
            self.pre_actions = torch.zeros(self.num_agents, self.env.action_space[0].n).to(self.device)
        self.reset_cam_pose()

    def reset_cam_pose(self):
        if 'Pose' not in self.args.env:
            self.cam_pos = self.env.env.env.env.cam_pose
        elif self.distance:
            self.cam_pos = self.env.cam_pose_d
            self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0)
            return self.set_cam_info_d()
        elif self.angle:
            self.cam_pos = self.env.cam_pose_angle
            self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0)
            return self.set_cam_info_angle()
        elif self.pose_reset_type == 2:
            self.cam_pos = self.env.cam_pose_all
            self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0)
            return self.set_cam_info_all()
        else:
            self.cam_pos = self.env.cam_pose
            self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0)
            return self.set_cam_info()

        if 'ns' in self.args.obs:
            self.noise_cam_pose = self.env.noise_cam_pose
            self.coordinate_delta = np.mean(np.array(self.noise_cam_pose)[:, :3], axis=0)
            return self.set_noise_cam_info()
        elif 'gs' in self.args.obs:
            self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0)
            return self.set_gs_noise_cam_info()

    def set_cam_info(self):
        if self.info:
            self.cam_pos = self.info['Cam_Pose']

        lengths = []
        for i in range(self.num_agents):
            length = np.sqrt(sum(np.array(self.cam_pos[i][:3] - self.coordinate_delta)) ** 2)
            lengths.append(length)
        pose_scale = max(lengths)

        cam = []
        for i in range(self.num_agents):

            sin_y = math.sin(self.cam_pos[i][4] / 180.0 * math.pi)
            sin_p = math.sin(self.cam_pos[i][5] / 180.0 * math.pi)
            cos_y = math.cos(self.cam_pos[i][4] / 180.0 * math.pi)
            cos_p = math.cos(self.cam_pos[i][5] / 180.0 * math.pi)

            tmp = np.concatenate([(self.cam_pos[i][:3] - self.coordinate_delta) / pose_scale,
                                  np.array([sin_y, cos_y, sin_p, cos_p])])

            cam.append(tmp)
        self.cam_info = np.array(cam)

        return np.array(cam)

    def set_cam_info_d(self):
        if self.info:
            self.cam_pos = self.info['cam pose d']
        cam = []
        for i in range(self.num_agents):
            sin_y = math.sin(self.cam_pos[i][4] / 180.0 * math.pi)
            sin_p = math.sin(self.cam_pos[i][5] / 180.0 * math.pi)
            cos_y = math.cos(self.cam_pos[i][4] / 180.0 * math.pi)
            cos_p = math.cos(self.cam_pos[i][5] / 180.0 * math.pi)

            tmp = np.concatenate([(self.cam_pos[i][:3] - self.coordinate_delta) / 1000.0,
                                  np.array([sin_y, cos_y, sin_p, cos_p]),  np.array([self.cam_pos[i][6]]) / 2000.0])

            cam.append(tmp)
        self.cam_info = np.array(cam)
        return np.array(cam)

    def set_cam_info_angle(self):
        if self.info:
            self.cam_pos = self.info['cam pose angle']
        cam = []
        for i in range(self.num_agents):
            sin_y = math.sin(self.cam_pos[i][4] / 180.0 * math.pi)
            sin_p = math.sin(self.cam_pos[i][5] / 180.0 * math.pi)
            cos_y = math.cos(self.cam_pos[i][4] / 180.0 * math.pi)
            cos_p = math.cos(self.cam_pos[i][5] / 180.0 * math.pi)

            sin_error_y = math.sin(self.cam_pos[i][6] / 180.0 * math.pi)
            sin_error_p = math.sin(self.cam_pos[i][7] / 180.0 * math.pi)
            cos_error_y = math.cos(self.cam_pos[i][6] / 180.0 * math.pi)
            cos_error_p = math.cos(self.cam_pos[i][7] / 180.0 * math.pi)

            tmp = np.concatenate([(self.cam_pos[i][:3] - self.coordinate_delta) / 1000.0,
                                  np.array([sin_y, cos_y, sin_p, cos_p]), np.array([sin_error_y, cos_error_y, sin_error_p, cos_error_p])])

            cam.append(tmp)
        self.cam_info = np.array(cam)
        return np.array(cam)

    def set_cam_info_all(self):
        if self.info:
            self.cam_pos = self.info['cam pose all']
        cam = []
        for i in range(self.num_agents):
            sin_y = math.sin(self.cam_pos[i][4] / 180.0 * math.pi)
            sin_p = math.sin(self.cam_pos[i][5] / 180.0 * math.pi)
            cos_y = math.cos(self.cam_pos[i][4] / 180.0 * math.pi)
            cos_p = math.cos(self.cam_pos[i][5] / 180.0 * math.pi)

            sin_error_y = math.sin(self.cam_pos[i][7] / 180.0 * math.pi)
            sin_error_p = math.sin(self.cam_pos[i][8] / 180.0 * math.pi)
            cos_error_y = math.cos(self.cam_pos[i][7] / 180.0 * math.pi)
            cos_error_p = math.cos(self.cam_pos[i][8] / 180.0 * math.pi)

            tmp = np.concatenate([(self.cam_pos[i][:3] - self.coordinate_delta) / 1000.0,
                                  np.array([sin_y, cos_y, sin_p, cos_p]), np.array([self.cam_pos[i][6]]) / 2000.0, np.array([sin_error_y, cos_error_y, sin_error_p, cos_error_p])])

            cam.append(tmp)
        self.cam_info = np.array(cam)
        return np.array(cam)

    def set_noise_cam_info(self):
        if self.info:
            self.noise_cam_pose = self.info['noise cam pose']

        cam = []
        for i in range(self.num_agents):

            sin_y = math.sin(self.cam_pos[i][4] / 180.0 * math.pi)
            sin_p = math.sin(self.cam_pos[i][5] / 180.0 * math.pi)
            cos_y = math.cos(self.cam_pos[i][4] / 180.0 * math.pi)
            cos_p = math.cos(self.cam_pos[i][5] / 180.0 * math.pi)
            tmp = np.concatenate([(self.noise_cam_pose[i][:3] - self.coordinate_delta) / 1000.0,
                                  np.array([sin_y, cos_y, sin_p, cos_p])])

            cam.append(tmp)
        self.cam_info = np.array(cam)

        return np.array(cam)

    def set_gs_noise_cam_info(self):
        if self.info:
            self.cam_pos = self.info['Cam_Pose']

        cam = []
        for i in range(self.num_agents):
            cam_loc_scale = (self.cam_pos[i][:3] - self.coordinate_delta) / 1000.0
            cam_gs_loc = [np.random.normal(cam_loc_scale[k], 0.005) for k in range(3)]
            cam_rot_h, cam_rot_v = self.cam_pos[i][4] , self.cam_pos[i][5]
            gs_rot_h, gs_rot_v  = np.random.normal(cam_rot_h, 10 * abs(cam_rot_h) / 180.0), np.random.normal(cam_rot_v, 8 * abs(cam_rot_v) / 180.0)

            sin_y = math.sin(cam_rot_h / 180.0 * math.pi)
            sin_p = math.sin(gs_rot_v / 180.0 * math.pi)
            cos_y = math.cos(cam_rot_h / 180.0 * math.pi)
            cos_p = math.cos(gs_rot_v / 180.0 * math.pi)
            tmp = np.concatenate([cam_gs_loc,  np.array([sin_y, cos_y, sin_p, cos_p])])
            cam.append(tmp)

        self.cam_info = np.array(cam)

        return np.array(cam)

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.preds = []
        return self

    def reward_normalizer(self, reward):
        self.num_steps += 1
        if self.num_steps == 1:
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt / self.num_steps
            self.vk = self.vk + delt * (reward - self.reward_mean)
            self.reward_std = math.sqrt(self.vk / (self.num_steps - 1))
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward

    def reset_rnn_hidden(self):
        self.img_cxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
        self.img_hxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
        self.pose_cxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
        self.pose_hxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))

    def update_rnn_hidden(self):
        self.img_cxs = Variable(self.img_cxs.data)
        self.img_hxs = Variable(self.img_hxs.data)
        self.pose_cxs = Variable(self.pose_cxs.data)
        self.pose_hxs = Variable(self.pose_hxs.data)

    def optimize(self, params, optimizer, shared_model, gpu_id):
        if 'Unreal' in self.args.env:
            self.gate_ids = self.env.env.env.env.gate_ids
        else:
            self.gate_ids = self.env.gate_ids
            self.random_ids = self.env.random_ids

        R = torch.zeros(self.num_agents, 1).to(self.device)
        if not self.done:
            # predict value
            state = self.state
            value_multi, *others = self.model(
                (
                    (Variable(state, requires_grad=True), (self.img_hxs, self.img_cxs)),
                    (Variable(torch.Tensor(self.cam_info), requires_grad=True), (self.pose_hxs, self.pose_cxs)),
                    (Variable(torch.Tensor(self.pre_actions), requires_grad=True),
                     Variable(torch.Tensor(self.gate_ids)),
                     Variable(torch.Tensor(self.random_ids)))
                )
            )
            for i in range(self.num_agents):
                R[i][0] = value_multi[i].data

        self.values.append(Variable(R).to(self.device))
        policy_loss = torch.zeros(self.num_agents, 1).to(self.device)
        value_loss = torch.zeros(self.num_agents, 1).to(self.device)
        pred_loss = torch.zeros(1, 1).to(self.device)
        entropies = torch.zeros(self.num_agents, 1).to(self.device)
        w_entropies = torch.Tensor([[float(self.args.entropy)] for i in range(self.num_agents)]).to(self.device)

        R = Variable(R, requires_grad=True).to(self.device)
        gae = torch.zeros(1, 1).to(self.device)
        for i in reversed(range(len(self.rewards))):
            R = self.args.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion
            delta_t = self.rewards[i] + self.args.gamma * self.values[i + 1].data - self.values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t
            policy_loss = policy_loss - \
                      (self.log_probs[i] * Variable(gae)) - \
                      (w_entropies * self.entropies[i])
            entropies += self.entropies[i]

        policy_loss = policy_loss[self.env.random_ids]
        value_loss = value_loss[self.env.random_ids]

        loss = policy_loss.sum() + 0.5 * value_loss.sum()
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 50)
        ensure_shared_grads(self.model, shared_model, gpu=gpu_id >= 0)

        optimizer.step()

        values0 = self.values[0].data
        self.clear_actions()
        return policy_loss, value_loss, entropies, pred_loss, values0
