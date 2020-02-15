import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from utils import weights_init


class CNN_simple(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_simple, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, obs_shape[0], obs_shape[1], obs_shape[2]))
        out = self.forward(dummy_state)
        self.outdim = out.size(-1)
        self.apply(weights_init)
        self.train()

    def forward(self, x):
        bs = x.shape[0]
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(bs, -1)
        return x


class FC_encoder(nn.Module):
    def __init__(self, obs_space, out_dim, stack_frames):
        super(FC_encoder, self).__init__()
        self.fc1 = nn.Linear(obs_space[0], 256)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, out_dim)
        self.lrelu4 = nn.LeakyReLU(0.1)
        self.outdim = out_dim  # * stack_frames

        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

    def forward(self, x):
        bs = x.shape[0]
        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))
        x = x.view(bs, self.outdim)
        return x
