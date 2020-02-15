from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import perception
import numpy as np
import math
from torch.nn import Parameter
from torch.autograd import Variable
from utils import norm_col_init, weights_init, normal


def build_model(action_space, pose_space, args):

    model = A3C_MULTI(action_spaces=action_space,
                      pose_space=pose_space,
                      lstm_out=args.lstm_out,
                      head_name=args.model,
                      rnn_layer = args.rnn_layer,
                      )
    model.train()
    return model


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2.0
    return out

def sample_action(action_type, mu_multi, sigma_multi, test=False, gpu_id=-1):
    if 'discrete' in action_type:
        logit = mu_multi
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        if test:
            action = prob.max(1)[1].data
        else:
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))
        action_env_multi = np.squeeze(action.cpu().numpy())
    else:  # continuous
        mu = torch.clamp(mu_multi, -1.0, 1.0)
        sigma = F.softplus(sigma_multi) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                eps = Variable(eps).cuda()
                pi = Variable(pi).cuda()
        else:
            eps = Variable(eps)
            pi = Variable(pi)
            action = (mu + sigma.sqrt() * eps).data
            act = Variable(action)
            prob = normal(act, mu, sigma, gpu_id, gpu=gpu_id >= 0)
            action = torch.clamp(action, -1.0, 1.0)
            entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)  # 0.5 * (log(2*pi*sigma) + 1
            log_prob = (prob + 1e-6).log()
            action_env_multi = action.cpu().numpy()
    return action_env_multi, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim, num=1):
        super(ValueNet, self).__init__()

        self.critic_linear = nn.Linear(input_dim, num)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        if 'discrete' in head_name:
            num_outputs = action_space.n
            self.continuous = False
        else:
            num_outputs = action_space.shape[0]
            self.continuous = True

        self.actor_linear = nn.Linear(input_dim, num_outputs)
        self.actor_linear2 = nn.Linear(input_dim, num_outputs)

        # init layers
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(self.actor_linear2.weight.data, 0.1)
        self.actor_linear2.bias.data.fill_(0)

    def forward(self, x, test=False):
        if self.continuous:
            mu = F.softsign(self.actor_linear(x))
            sigma = self.actor_linear2(x)
        else:
            mu = self.actor_linear(x)
            sigma = 0
        action, entropy, log_prob = sample_action(self.head_name, mu, sigma, test)
        return action, entropy, log_prob


class global_block_rnn(nn.Module):
    def __init__(self, in_dims, out_dim, global_feature_name, layer):
        super(global_block_rnn, self).__init__()
        self.global_feature_name = global_feature_name

        pose_feature_dim = in_dims['pose_feature_dim']

        num_layer = layer

        pose_out_dim = out_dim // 2
        self.pose_BiRNN = BiRNN(pose_feature_dim+1, pose_out_dim, num_layer, self.global_feature_name)

        self.out_dim = out_dim

    def forward(self, inputs):
        
        pose_features = inputs['pose_features']
        uncertainty = inputs['uncertainty']

        pose_reconstruction = []
        for i in range(len(pose_features)):
            tmp = [uncertainty[i].unsqueeze(0), pose_features[i]]
            tmp = torch.cat(tmp)
            pose_reconstruction.append(tmp)
        pose_reconstruction = torch.cat(pose_reconstruction).view(len(pose_features), -1).unsqueeze(0)
        global_features, global_features_hn = self.pose_BiRNN(pose_reconstruction)

        return global_features.squeeze()


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, head_name, device=torch.device('cpu')):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_name = head_name
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.feature_dim = hidden_size * 2
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection

        # Forward propagate LSTM
        if self.lstm:
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            out, hn = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)
        (_, batch, _) = hn.shape
        hn = hn.view(self.num_layers, 2, batch, self.hidden_size)[-1]  # get the last layer
        hn = hn.permute(1, 0, 2).view(batch, -1)  # transpose the direction dim and batch
        return out, hn


class A3C_MULTI(torch.nn.Module):
    def __init__(self, action_spaces, pose_space, lstm_out=128, head_name='gru', rnn_layer=1):
        super(A3C_MULTI, self).__init__()

        pose_feature_dim = len(pose_space[0])
        action_feature_dim = action_spaces[0].n
        self.num_agents = len(action_spaces)
        self.name = head_name

        inputs_dims = {
            'pose_feature_dim': pose_feature_dim,
            'action_feature_dim': action_feature_dim,
            'origin_pose_dim': len(pose_space[0])
        }

        self.global_net = global_block_rnn(in_dims=inputs_dims,
                                           out_dim=lstm_out,
                                           global_feature_name=head_name,
                                           layer = rnn_layer)

        fusion_dim = self.global_net.out_dim
        feature_dim = fusion_dim  

        #  create pose_actor
        self.pose_actor = PolicyNet(feature_dim, action_spaces[0], head_name)
        self.pose_critic = ValueNet(feature_dim, 1)

        self.apply(weights_init)

        self.train()

    def forward(self, inputs, test=False):
        R_stu = 0
        (states, (img_hxs, img_cxs)), (cam_info, (pose_hxs, pose_cxs)), (pre_actions, gate_ids,  random_ids) = inputs

        entropies = []
        log_probs = []
        actions = []
        values = []

        pose_features = cam_info

        self.gate = Variable(gate_ids, requires_grad=True)

        input_dict = {
            'pose_features': pose_features,
            'action_feature': pre_actions,
            'origin_pose': cam_info,
            'uncertainty': self.gate,
        }
        global_features = self.global_net(input_dict)

        for i in range(self.num_agents):
            # comment above for parallel faster
            value = self.pose_critic(global_features[i].unsqueeze(0))
            action, entropy, log_prob = self.pose_actor(global_features[i].unsqueeze(0), test)

            log_probs.append(log_prob)
            entropies.append(entropy)
            actions.append(action)
            values.append(value)

        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        if 'discrete' not in self.name:
            actions = actions.squeeze()
            entropies = entropies.sum(-1)
            log_probs = log_probs.sum(-1)
            log_probs = log_probs.unsqueeze(1)

        entropies = entropies.unsqueeze(1)

        return values, actions, (img_hxs, img_cxs), (pose_hxs, pose_cxs), entropies, log_probs, R_stu

