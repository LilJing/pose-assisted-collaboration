import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
import json

from pose_env import Pose_Env
from environment import create_env
import time

def build_pose_env(reset_type):
    return Pose_Env(reset_type)

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return None
        # return self.action_space.sample()
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PoseEnv-v2', metavar='ENV',
                        help='environment to train on (default: BipedalWalker-v2)')
    parser.add_argument('--obs', default='img', metavar='UE', help='unreal env')
    parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
    parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
    parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
    parser.add_argument('--flip', dest='flip', action='store_true', help='flip image and action')
    parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='normalize image')
    parser.add_argument('--stack-frames', type=int, default=1, metavar='SF',
                        help='Choose number of observations to stack')
    parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
    parser.add_argument('--render', dest='render', action='store_true', help='render test')
    args = parser.parse_args()
    reset_type = args.env.split('-v')[1]
    env = build_pose_env(int(reset_type))

    # env = create_env(args.env, args)
    agents_num = len(env.action_space)
    agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]

    episode_count = 10000
    rewards = 0
    done = False

    ds = []
    ws = []
    hs = []
    areas = []
    num = 0
    Total_rewards = np.zeros(agents_num)
    for eps in range(1, episode_count):
        obs = env.reset()
        count_step = 0
        t0 = time.time()
        C_rewards = np.zeros(agents_num)
        while True:
            actions = [agents[i].act(obs[i]) for i in range(agents_num)]
            t0 = time.time()
            render = args.render
            obs, rewards, done, info = env.step(actions, render)
            # print('unreal time step', time.time() - t0)
            C_rewards += rewards
            count_step += 1
            # if args.render:
            #     img = env.render(mode='rgb_array')
            #     #  img = img[..., ::-1]  # bgr->rgb
            #     cv2.imshow('show', img)
            #     cv2.waitKey(1)
            if done:
                # print('Done')
                fps = count_step / (time.time() - t0)
                Total_rewards += C_rewards
                # print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                break
            # if len(ds) >= 5000:
            #     print('data done ..')
            #     dict = {'distance': ds, 'area': areas, 'width':ws, 'height':hs}
            #     with open("v0-test-10000.json", "w") as f:
            #         file = json.dump(dict, f)
            #     break
    # Close the env and write monitor result info to disk
    env.close()
