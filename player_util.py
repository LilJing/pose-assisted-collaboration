from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable

class Agent(object):
    def __init__(self, model, env, args, state, cam_info, device):
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
        self.collect_state = None

        self.cam_info = cam_info
        self.cam_pos = None
        self.input_actions = None
        self.hxs = [None for i in range(self.num_agents)]
        self.cxs = [None for i in range(self.num_agents)]
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.gate_entropies = []
        self.preds = []
        self.done = True
        self.info = None
        self.reward = 0
        self.local_reward = 0
        self.global_reward = 0
        self.device = device
        self.lstm_out = args.lstm_out
        self.reward_mean = None
        self.reward_std = 1
        self.num_steps = 0
        self.vk = 0

        self.ds= []
        self.hori_angles = []
        self.verti_angles = []

        self.gt_ds = []
        self.gt_hori_angles = []
        self.gt_verti_angles = []

        self.gate_probs = []
        self.gate_entropies = []

        self.images = []
        self.zoom_angle_hs = []
        self.zoom_angle_vs = []
        self.collect_data_step = 0

        self.last_choose_whos = []
        self.last_gate_ids = torch.Tensor([1 for i in range(self.num_agents)]).to(self.device)
        self.pre_ids = []
        self.updates = []
        self.gt_ids = []
        self.last_choose_ids = []
        self.pre_ids = []
        self.single_actions = []
        self.pose_actions = []
        self.cam_poses = []
        self.lstm_features = []
        self.pose_features = []

        self.gate_gts = []
        self.gates = []
        self.time_step = 0
        self.times = 0

    def wrap_action(self, action, high, low):
        action = np.squeeze(action)
        out = action * (high - low)/2.0 + (high + low)/2.0
        return out

    def action_train(self):
        self.gt_gate = torch.Tensor(np.array(self.env.env.env.env.gate_ids)).to(self.device)
        if len(self.last_choose_whos) == 0:
            self.last_choose_whos = [0 for i in range(self.num_agents)]

        value_multi, action_env_multi, self.H_multi, entropy, log_prob, R_pred, gate_prob, gate_id , lstm_feature = self.model(
            (Variable(self.state, requires_grad=True), Variable((self.cam_info), requires_grad=True),
             self.H_multi, self.last_gate_ids, self.gt_gate))

        [self.Hx, self.Cx] = self.H_multi

        self.last_gate_ids = gate_id
        self.eps_len += 1
        self.values.append(value_multi)

        self.entropies.append(entropy)
        self.log_probs.append(log_prob)

        self.gates.append(gate_prob)

        if 'discrete' not in self.args.model:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]
        state_multi, reward_multi, self.done, self.info = self.env.step(action_env_multi)

        self.gate_gts.append(self.gt_gate)
        self.images.append(self.info['states'])
        self.gt_gate = self.info['gate ids']

        self.state = torch.from_numpy(state_multi).float().to(self.device)

        if 'Unreal' in self.args.env:
            self.cam_pos = self.env.env.env.env.cam_pose
            self.collect_state = self.env.env.env.env.current_states

        self.cam_poses.append(self.cam_pos)

        self.global_reward_mean = sum(reward_multi) / self.num_agents
        self.local_reward = torch.tensor(reward_multi).float().to(self.device)
        self.reward = self.local_reward
        self.rewards.append(self.reward.unsqueeze(1))
        self.set_cam_info()

        return self

    def action_test(self):
        self.gt_gate = torch.Tensor(np.array(self.env.env.env.env.gate_ids)).to(self.device)
        if 'Unreal' in self.args.env:
            self.cam_pos = self.env.env.env.env.current_cam_pos
            self.collect_state = self.env.env.env.env.current_states
            self.images = self.env.env.env.env.states
            self.target_poses = self.env.env.env.env.current_target_pos

        with torch.no_grad():
            self.update_lstm()
            if len(self.last_choose_whos) == 0:
                self.last_choose_whos = [0 for i in range(self.num_agents)]
            self.last_choose_ids.append(self.last_choose_whos)
            value_multi, action_env_multi, self.H_multi, entropy, log_prob, R_pred, gate_prob, gate_id, lstm_feature = self.model(
                (Variable(self.state), Variable(self.cam_info), self.H_multi, self.last_gate_ids, self.gt_gate))
            self.time_step += 1

            self.gates.append(gate_prob)
            self.last_gate_ids = gate_id

            [self.Hx, self.Cx] = self.H_multi

        if 'discrete' not in self.args.model:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]

        state_multi, self.reward, self.done, self.info = self.env.step(action_env_multi)
        self.success_rate = self.info['Success rate']
        self.success_ids = self.info['Success ids']

        self.gate_probs = gate_prob
        self.gate_ids = gate_id
        self.actions = action_env_multi
        self.lstm_features = lstm_feature

        self.gate_gts.append(self.gt_gate)
        self.gt_gate = self.info['gate ids']
        self.collect_data_step += 1

        if self.args.render:
            if 'gtgate' in self.args.test_type:
                self.env.env.env.env.to_render(self.gt_gate)
            else:
                if 'MCRoom' in self.args.env:
                    self.env.render()
                else:
                    self.env.env.env.env.to_render(gate_id)
                    self.env.render()

        self.state = torch.from_numpy(state_multi).float().to(self.device)

        self.set_cam_info()
        self.eps_len += 1
        self.rewards.append(self.reward)

        return self

    def set_cam_info(self):
        if self.info:
            self.cam_pos = self.info['camera poses']
        self.coordinate_delta = np.mean(np.array(self.cam_pos)[:, :3], axis=0)

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
        self.cam_info = torch.Tensor(np.array(cam)).to(self.device)
        return np.array(cam)

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.gate_entropies = []

        self.preds = []
        self.gate_probs = []
        self.gate_entropies = []

        self.pre_ids = []
        self.updates = []

        self.gt_ids = []
        self.pre_ids = []
        self.last_choose_ids = []
        self.single_actions = []
        self.pose_actions = []

        self.lstm_features = []
        self.pose_features = []
        self.images = []
        self.cam_poses = []

        self.gate_gts = []
        self.gates = []

        return self

    def reward_normalizer(self, reward):
        self.num_steps += 1
        if self.num_steps == 1:
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt/self.num_steps
            self.vk = self.vk + delt * (reward-self.reward_mean)
            self.reward_std = math.sqrt(self.vk/(self.num_steps - 1))
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward

    def update_lstm(self):

        if self.done:
            self.cxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
            self.hxs = Variable(torch.zeros(self.num_agents, self.lstm_out).to(self.device))
            self.H_multi = [self.hxs, self.cxs]
        else:
            self.Hx , self.Cx = Variable(self.Hx.data), Variable(self.Cx.data)
            self.H_multi = [self.Hx, self.Cx]



