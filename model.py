from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init, weights_init_mlp, normal
import perception
import numpy as np
import math

def build_model(obs_space, action_space, args, device):
    name = args.model

    model = A3C_MULTI(args, obs_space, action_space, args.lstm_out, name, args.stack_frames, device)

    model.train()
    return model


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low)/2 + (self.action_high + self.action_low)/2.0
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
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.critic_linear = nn.Linear(input_dim, 1)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value

class HEAD(torch.nn.Module):
    def __init__(self, obs_space, action_space, lstm_out=128, head_name='cnn_lstm',  stack_frames=1):

        super(HEAD, self).__init__()
        self.head_name = head_name
        if 'cnn' in head_name:
            self.encoder = perception.CNN_net(obs_space, stack_frames)
        feature_dim = self.encoder.outdim
        self.head_cnn_dim = self.encoder.outdim
        if 'lstm' in head_name:
            self.lstm = nn.LSTMCell(feature_dim, lstm_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = lstm_out
        if 'gru' in head_name:
            self.lstm = nn.GRUCell(feature_dim, lstm_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = lstm_out
        self.head_dim = feature_dim

    def forward(self, inputs):
        X, (Hx, Cx) = inputs
        feature = self.encoder(X)
        feature = feature

        if 'lstm' in self.head_name:
            Hx, Cx = self.lstm(feature, (Hx, Cx))
            feature = Hx
        if 'gru' in self.head_name:
            Hx = self.lstm(feature, Hx)
            feature = Hx

        return feature, (Hx, Cx)

class Policy(torch.nn.Module):
    def __init__(self, outdim, action_space, lstm_out=128, head_name='cnn_lstm',  stack_frames=1):
        super(Policy, self).__init__()
        self.head_name = head_name
        if 'lstm' in self.head_name:
            feature_dim = lstm_out
        else:
            feature_dim = outdim

        #  create actor
        if 'discrete' in head_name:
            num_outputs = action_space.n
        else:
            num_outputs = action_space.shape[0]

        self.actor_linear = nn.Linear(feature_dim, num_outputs)
        self.actor_linear2 = nn.Linear(feature_dim, num_outputs)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.1)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(self.actor_linear2.weight.data, 0.1)
        self.actor_linear2.bias.data.fill_(0)

        # create critic
        if 'mc' in head_name:
            self.critic_linear = nn.Linear(feature_dim, num_outputs)
        else:
            self.critic_linear = nn.Linear(feature_dim, 1)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, feature):

        value = self.critic_linear(feature)
        if 'discrete' in self.head_name:
            mu = self.actor_linear(feature)
        else:
            mu = F.softsign(self.actor_linear(feature))
        sigma = self.actor_linear2(feature)

        return value, mu, sigma


class Gate(nn.Module):
    def __init__(self, head_dim, args):
        super(Gate, self).__init__()
        gate_input_dim = head_dim
        self.feature_dim = 256
        self.gate_fc1 = nn.Linear(gate_input_dim, self.feature_dim)
        self.gate_fc1.weight.data = norm_col_init(self.gate_fc1.weight.data, 0.1)
        self.gate_fc1.bias.data.fill_(0)

        self.gate_fc2 = nn.Linear(self.feature_dim, self.feature_dim)
        self.gate_fc2.weight.data = norm_col_init(self.gate_fc2.weight.data, 0.1)
        self.gate_fc2.bias.data.fill_(0)

        self.gate_fc3 = nn.Linear(self.feature_dim, 2)
        self.gate_fc3.weight.data = norm_col_init(self.gate_fc3.weight.data, 0.1)
        self.gate_fc3.bias.data.fill_(0)

    def forward(self, x):
        feature = torch.relu(self.gate_fc1(x))
        feature = torch.relu(self.gate_fc2(feature))
        gate_prob_value = self.gate_fc3(feature)

        return  gate_prob_value

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
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection

        # Forward propagate LSTM
        if self.lstm:
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
            out, hn = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2); hn: [num_layers * num_directions, bs, hidden_size]
        (_, batch, _) = hn.shape
        hn = hn.view(self.num_layers, 2, batch, self.hidden_size)[-1]  # get the last layer
        hn = hn.permute(1, 0, 2).view(batch, -1)  # transpose the direction dim and batch
        return out, hn


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        if 'discrete' in head_name:
            num_outputs = action_space.n - 2
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


class A3C_MULTI(torch.nn.Module):
    def __init__(self, args, obs_space, action_spaces, lstm_out=128, head_name='cnn_lstm',  stack_frames=1, device=None):
        super(A3C_MULTI, self).__init__()
        self.num_agents = len(obs_space)
        self.global_name = args.global_model

        obs_shapes = [obs_space[i].shape for i in range(self.num_agents)]
        self.head_name = head_name

        self.header = HEAD(obs_shapes[0], action_spaces[0], lstm_out, head_name, stack_frames)
        self.policy = Policy(self.header.head_dim, action_spaces[0], lstm_out, head_name, stack_frames)

        self.device = device
        self.Hx = torch.zeros(1, lstm_out)
        self.Cx = torch.zeros(1, lstm_out)

        self.pose_feature_dim = 7
        self.pose_out_dim = lstm_out // 2
        #
        self.rnn_layer = args.rnn_layer
        self.pose_BiRNN = BiRNN(self.pose_feature_dim + 1, self.pose_out_dim, self.rnn_layer, self.global_name, device)
        self.pose_actor = PolicyNet(lstm_out, action_spaces[0], head_name)
        self.gate = Gate(self.header.head_dim, args)

        self.device = device

        self.test_type = args.test_type

    def forward(self, inputs, test=False):
        R_stu = 0
        states, cam_info, H_states, last_gate_ids, gt_gate = inputs

        feature, (Hx, Cx) = self.header((states, H_states))
        Hiden_states = (Hx, Cx)
        gates = self.gate(feature.unsqueeze(1))
        values, single_mus, sigmas = self.policy(feature.unsqueeze(1))

        vision_actions = []
        entropies = []
        log_probs = []
        gate_ids = []
        gate_probs = []
        chooses = []
        pose_ids = []
        for i in range(self.num_agents):
            gate_prob = F.softmax(gates[i], dim=1)
            _, max_id = torch.max(gate_prob, 1)
            use = 'vision' if max_id[0] == 1 else 'pose'
            chooses.append(use)
            vision_action, entropy, log_prob = sample_action(self.head_name, single_mus[i], None, test)
            if 'gtgate' in self.test_type:
                pose_ids.append(gt_gate[i])
            else:
                pose_ids.append(max_id[0])
            gate_ids.append(max_id[0])
            vision_actions.append(vision_action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            gate_probs.append(gate_prob)

        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        gate_ids = torch.Tensor(gate_ids).to(self.device)
        pose_ids = torch.Tensor(pose_ids).to(self.device)
        gate_probs = torch.cat(gate_probs)
        entropies = entropies.unsqueeze(1)

        # pose controller
        pose_reconstruction = []
        for i in range(self.num_agents):
            tmp = torch.cat([pose_ids[i].unsqueeze(0), cam_info[i]]).unsqueeze(0)
            pose_reconstruction.append(tmp)

        pose_reconstruction = torch.cat(pose_reconstruction).unsqueeze(0)
        global_features, global_features_hn = self.pose_BiRNN(pose_reconstruction)
        global_features = global_features.squeeze()
        pose_actions = []
        for i in range(self.num_agents):
            global_features = global_features.squeeze()
            pose_action, _, _ = self.pose_actor(global_features[i].unsqueeze(0), test)
            pose_actions.append(pose_action.tolist())

        chooses = []
        final_actions = []
        use_gates = []
        for i in range(self.num_agents):
            if 'gtgate' in self.test_type:
                use_gate = int(gt_gate[i])
            elif 'modelgate' in self.test_type:
                use_gate = gate_ids[i]
            elif 'VisionOnly' in self.test_type:
                use_gate = 1 # use vision
            else:
                print('Error in testing type')
            use_gates.append(use_gate)

            if use_gate == 1:
                chooses.append('vision')
                final_actions.append(vision_actions[i])
            else:
                chooses.append('pose')
                final_actions.append(pose_actions[i])

        return values, final_actions, Hiden_states, entropies, log_probs, R_stu, gate_probs, gate_ids, feature
