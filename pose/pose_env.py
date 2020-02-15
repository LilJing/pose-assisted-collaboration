import math
import random

import cv2
import json
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class Pose_Env:
    def __init__(self, reset_type):

        self.reset_type = reset_type

        with open("PoseEnvLarge.json", encoding='utf-8') as f: 
            setting = json.load(f)

        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_actions_player = setting['discrete_actions_player']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.max_obstacles = setting['max_obstacles']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        self.reset_area = setting['reset_area']
        self.cam_area = setting['cam_area']

        if setting.get('goal_list'):
            self.goal_list = setting['goal_list']
        self.camera_loc = setting['camera_loc']

        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']
        self.exp_distance = setting['exp_distance']
        self.safe_start = setting['start']
        self.start = setting['start']

        self.num_target = len(self.target_list)
        self.num_cam = len(self.cam_id)
        self.cam_height = [setting['height'] for i in range(self.num_cam)]

        # define action space
        self.action_space = [spaces.Discrete(len(self.discrete_actions)) for i in range(self.num_cam)]

        # define observation space
        self.observation_space = [np.zeros((1,80,80), int) for i in range(self.num_cam)]
        self.env = Env1()
        self.cam_focus_z = None
        self.action_type = 'Discrete'

        self.cam = dict()
        for i in range(len(self.cam_id)+1):
            self.cam[i] = dict(
                 location=[0, 0, 0],
                 rotation=[0, 0, 0],
            )

        self.gate_prob = random.normalvariate(0.7, 0.05)
        self.scale = 1000.0
        self.target_move_rule = setting['target_move_rule']

        self.walk_step = 0
        self.walk_len = np.random.randint(2, 10)
        self.walk_velocity = np.random.randint(20, 30)
        self.walk_distance = [np.random.randint(-self.walk_velocity, self.walk_velocity),
                              np.random.randint(-self.walk_velocity, self.walk_velocity),
                              np.random.randint(int(-self.walk_velocity / 2), int(self.walk_velocity / 2))]

        self.current_move_angle_h = 0
        self.current_move_angle_v = 0
        self.max_angle_noise = [5, 4]
        self.max_loc_noise = 30

        self.mix_target_rule = setting['mix target rules']
        self.reverse = False
        self.target_pos = self.start.copy()

        self.rerand_steps = 0
        self.test = False

    def set_location(self, cam_id, loc):
        self.cam[cam_id]['location'] = loc

    def get_rotation(self, cam_id):
        return self.cam[cam_id]['rotation']

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt)/np.pi*180-current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def get_angle(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180
        return angle_now

    def get_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        return d

    def get_verti_direction(self, current_pose, target_pose):
        person_height = target_pose[2]
        plane_distance = self.get_distance(current_pose, target_pose)
        height = current_pose[2] - person_height
        angle = np.arctan2(height, plane_distance) / np.pi * 180
        angle_now = angle + current_pose[-1]
        return angle_now

    def set_rotation(self, cam_id, rot):
        self.cam[cam_id]['rotation'] = rot

    def reset(self):

        # get state
        # random camera
        states = [np.zeros((1,80,80), int) for i in range(self.num_cam)]

        self.start_area = self.get_start_area(self.safe_start[0], 200)
        self.random_agent = GoalNavAgentTest(self.continous_actions_player, goal_list=self.goal_list)
        self.target_pos = np.array([np.random.randint(self.start_area[0],self.start_area[1]),
                           np.random.randint(self.start_area[2], self.start_area[3]),
                           self.safe_start[0][-1]])

        self.cam_pose = []
        self.gt_cam_pose= []
        self.noise_cam_pose = []
        self.distances = []
        self.cam_locs = []
        self.angle_errors = []
        self.gate_ids = np.ones(self.num_cam)

        random_num = np.random.randint(1, 4)
        self.random_ids = random.sample(range(self.num_cam), random_num)
        self.true_ids = list(set(range(self.num_cam)) - set(self.random_ids))

        for i in self.random_ids:
            self.gate_ids[i] = 0

        for i, cam in enumerate(self.cam_id):
            if self.test:
                cam_loc = self.camera_loc[i]
            else:
                cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                           np.random.randint(self.cam_area[i][2], self.cam_area[i][3]),
                           np.random.randint(self.cam_area[i][4], self.cam_area[i][5])
                           ]

            distance = np.linalg.norm(np.array(cam_loc) - np.array(self.target_pos))
            self.cam_locs.append(cam_loc)
            self.distances.append([distance])
            self.set_location(cam, cam_loc)
            cam_rot = self.get_rotation(cam)

            angle_h = self.get_direction(cam_loc+cam_rot, self.target_pos)
            angle_v = self.get_verti_direction(cam_loc+cam_rot, self.target_pos)
            cam_rot[1] += angle_h
            cam_rot[2] -= angle_v

            # random cameras' rotation
            if self.reset_type == 0 or i in self.random_ids:
                delta_h = np.random.randint(-180, 180)
                delta_v = np.random.randint(-80, 80)
                cam_rot[1] += delta_h
                cam_rot[2] += delta_v

            self.gt_cam_pose.append(cam_loc + cam_rot)

            if self.reset_type == 2:
                if i not in self.random_ids:  # add delta theta and d as input
                    error_h = np.random.randint(-20, 20)
                    error_v = np.random.randint(-15, 15)
                    cam_rot[1] += error_h
                    cam_rot[2] += error_v
                else:
                    error_h = 0
                    error_v = 0
                self.angle_errors.append([error_h, error_v])

            self.set_rotation(cam, cam_rot)
            self.cam_pose.append(cam_loc + cam_rot)

            # add random noise to all angles
            noise_cam_rot = [0, cam_rot[1]+np.random.randint(- self.max_angle_noise[0], self.max_angle_noise[0]),
                             cam_rot[2] + np.random.randint(- self.max_angle_noise[1], self.max_angle_noise[1])]

            noise_cam_loc = list(np.array(cam_loc) + np.array([np.random.randint(-self.max_loc_noise, self.max_loc_noise) for i in range(3)]))
            self.noise_cam_pose.append(noise_cam_loc + noise_cam_rot)

        self.env.env.env.cam_pose = self.cam_pose

        self.last_cam_pose = np.array(self.cam_pose).copy()
        self.last_target_pos = np.array(self.target_pos).copy()

        self.cam_angles = np.array([self.last_cam_pose[i][4] for i in range(self.num_cam)])

        self.count_steps = 0
        self.count_close = 0

        if self.reset_type == 2:
            self.cam_pose_d = []
            self.cam_pose_all = []
            self.cam_pose_angle = []
            for i in range(self.num_cam):
                self.cam_pose_d.append(self.cam_pose[i] + self.distances[i])
                self.cam_pose_all.append(self.cam_pose[i] + self.distances[i] + self.angle_errors[i])
                self.cam_pose_angle.append(self.cam_pose[i] + self.angle_errors[i])

        self.states = np.array(states)

        if self.mix_target_rule == 1:
            self.target_move_rule = random.sample(range(4), 1)[0]

        return np.array(states)

    def step(self, actions, render):
        info = dict(
            Done=False,
            Reward=[0 for i in range(self.num_cam)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )

        if render:
            Render(self.gt_cam_pose, self.target_pos, self.scale)

        # set angle bias to gt cameras
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.get_rotation(cam)
            self.gt_cam_pose[i][-3:] = cam_rot
            if self.reset_type == 2:  # add delta theta and d as input
                if i not in self.random_ids:
                    error_h = np.random.randint(-20, 20)
                    error_v = np.random.randint(-15, 15)
                    cam_rot[1] += error_h
                    cam_rot[2] += error_v
                else:
                    error_h = 0
                    error_v = 0
                self.angle_errors.append([error_h, error_v])

        self.distances = []
        for i, cam in enumerate(self.cam_id):
            distance = np.linalg.norm(np.array(self.cam_locs[i]) - np.array(self.target_pos))
            self.distances.append([distance])

        if self.reset_type == 2:
            self.cam_pose_d = []
            self.cam_pose_all = []
            self.cam_pose_angle = []
            for i in range(self.num_cam):
                self.cam_pose_d.append(self.cam_pose[i] + self.distances[i])
                self.cam_pose_all.append(self.cam_pose[i] + self.distances[i] + self.angle_errors[i])
                self.cam_pose_angle.append(self.cam_pose[i] + self.angle_errors[i])

        # target move
        if self.target_move_rule == 0:
            step = 50
            self.target_pos += np.array([np.random.randint(-1 * step, step), np.random.randint(-1 * step, step),
                                np.random.randint(int(-1 * step / 2), int(step / 2))])

        elif self.target_move_rule == 1:
            self.walk_step += 1
            if self.walk_step > self.walk_len:
                self.walk_len = np.random.randint(1, 10)
                self.walk_distance = [np.random.randint(-self.walk_velocity, self.walk_velocity),
                                      np.random.randint(-self.walk_velocity, self.walk_velocity),
                                      np.random.randint(int(-self.walk_velocity / 2), int(self.walk_velocity / 2))]
                self.walk_velocity = np.random.randint(30, 40)
                self.walk_step = 0

            for k in range(3):
                self.walk_distance[k] *= np.random.normal(1, 0.2)
                self.walk_distance[k] = int(self.walk_distance[k])

            self.target_pos += self.walk_distance

        elif self.target_move_rule == 2: # move by angle

            self.walk_velocity = np.random.randint(40, 50)
            self.current_move_angle_h += np.random.randint(-90, 90)
            self.current_move_angle_v += np.random.randint(-30, 30)
            delta_x = int(math.cos(self.current_move_angle_h / 180.0 * math.pi) * self.walk_velocity)
            delta_y = int(math.sin(self.current_move_angle_h / 180.0 * math.pi) * self.walk_velocity)
            delta_z = int(math.sin(self.current_move_angle_v / 180.0 * math.pi) * self.walk_velocity)

            self.walk_distance = np.array([delta_x, delta_y, delta_z])
            self.target_pos += self.walk_distance

        elif self.target_move_rule == 3:

            reach_corner =  False
            for i in range(3):
                if abs(self.target_pos[i]) >= self.scale:
                    reach_corner = True
                self.target_pos = self.random_agent.act(self.target_pos, reach_corner)

        elif self.target_move_rule == 4:   # move as goal list
            self.target_pos = self.random_agent.act(self.target_pos)

        actions = np.squeeze(actions)
        actions2cam = []
        self.noise_cam_pose = []
        for i in range(self.num_cam):
            if self.action_type == 'Discrete':
                actions2cam.append(self.discrete_actions[actions[i]])  # delta_yaw, delta_pitch
            else:
                actions2cam.append(actions[i])  # delta_yaw, delta_pitch

        self.angle_errors = []
        cal_target_observed = np.zeros(len(self.cam_id))
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.get_rotation(cam)
            cam_loc = self.cam_pose[i][:3]
            last_cam_rot = cam_rot

            # add noise to all angles
            noise_cam_rot = [0, cam_rot[1] + np.random.randint(- self.max_angle_noise[0], self.max_angle_noise[0]),
                             cam_rot[2] + np.random.randint(- self.max_angle_noise[1], self.max_angle_noise[1])]

            noise_cam_loc = list(np.array(cam_loc) + np.array(
                [np.random.randint(-self.max_loc_noise, self.max_loc_noise) for i in range(3)]))
            self.noise_cam_pose.append(noise_cam_loc + noise_cam_rot)

            cam_rot[1] += actions2cam[i][0]
            cam_rot[2] += actions2cam[i][1]
            cam_rot[2] = cam_rot[2] if cam_rot[2] < 80.0 else 80.0
            cam_rot[2] = cam_rot[2] if cam_rot[2] > - 80.0 else -80.0  # because person may walk under the camera

            cam_loc = self.cam_pose[i][:3]
            cam_rot = self.get_rotation(cam)

            angle_h = self.get_direction(cam_loc+cam_rot, self.target_pos)
            angle_v = self.get_verti_direction(cam_loc+cam_rot, self.target_pos)

            # test focus map
            if self.reset_type == 1 or i not in self.random_ids:
                cam_rot[1] += angle_h
                cam_rot[2] -= angle_v
                angle_h = self.get_direction(cam_loc+cam_rot, self.target_pos)
                angle_v = self.get_verti_direction(cam_loc+cam_rot, self.target_pos)
            #
            hori_reward = 1 - 2*abs(angle_h) / 45.0
            verti_reward = 1 - 2*abs(angle_v) / 30.0

            info['Reward'][i] = max(hori_reward + verti_reward, -2)/2

            self.set_rotation(cam, cam_rot)
            self.cam_pose[i][-3:] = cam_rot
            self.last_cam_pose[i][-3:] = last_cam_rot

            if abs(angle_h) <= 45.0 and abs(angle_v) <= 30.0:
                cal_target_observed[i] = 1

        self.count_steps += 1

        if self.count_close > 10:
            info['Done'] = True

        self.env.env.env.cam_pose = self.cam_pose
        info['Cam_Pose'] = self.cam_pose
        info['noise cam pose'] = self.noise_cam_pose

        if self.reset_type == 2:
            info['cam pose d'] = self.cam_pose_d
            info['cam pose all'] = self.cam_pose_all
            info['cam pose angle'] = self.cam_pose_angle

        # set your done condition
        if self.count_steps > self.max_steps:
            info['Done'] = True

        if info['Done']:
            self.count_steps = 0


        return self.states, info['Reward'], info['Done'], info

    def close(self):
        pass

    def seed(self, para):
        pass

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area


class Env1:
    def __init__(self):
        self.env = Env2()


class Env2:
    def __init__(self):
        self.env = Pose()


class Pose:
    def __init__(self):
        self.cam_pose = None
        self.cam_focus_z = None


def cal_focus_map(cam_pose_list):
    visible_len = 1000
    agent_num = len(cam_pose_list)
    coordinate_delta = np.min(np.array(cam_pose_list)[:, :3], axis=0)

    cam = []
    for i in range(agent_num):
        sin_y = math.sin(cam_pose_list[i][4] / 180.0 * math.pi)
        sin_p = math.sin(cam_pose_list[i][5] / 180.0 * math.pi)
        cos_y = math.cos(cam_pose_list[i][4] / 180.0 * math.pi)
        cos_p = math.cos(cam_pose_list[i][5] / 180.0 * math.pi)
        tmp = np.concatenate([(cam_pose_list[i][:3] - coordinate_delta),
                              np.array([cos_y * cos_p, sin_y * cos_p, sin_p])])
        cam.append(tmp)
    cam = np.array(cam)

    # calculate
    mini = 1
    no_focus = False
    for i in range(len(cam)):
        if cam[i,5].item() > 0:
            no_focus = True
        if abs(cam[i,5].item()) < 1e-5:
            cam[i, 5] = -1e-5
        if abs(cam[i,5].item()) < mini:
            mini = abs(cam[i,5].item())
    unit_z = max(-1.0 * mini / 0.1, -20)

    last_area = np.inf
    im = np.zeros([650, 650], dtype=np.uint8)
    im += 200
    points_list = []
    for i in range(0, visible_len):
        z = unit_z*i
        delta = (z-cam[:, 2])/cam[:, 5]
        x = cam[:, 0] + delta*cam[:, 3]
        y = cam[:, 1] + delta*cam[:, 4]
        points = np.array([x, y], dtype=int).T
        points = cv2.convexHull(points).squeeze()
        area = cv2.contourArea(points)
        if area > last_area:
            cv2.putText(im, 'last_area:{} {} {}'.format(last_area, unit_z, mini), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            break
        elif no_focus:
            cv2.putText(im, 'no focus', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            points_list.append(points[np.newaxis, :] / 5.0 + 200)
            break
        else:
            last_area = area
            points_list.append(points[np.newaxis, :]/5.0+200)

    for i in range(len(points_list)):
        color = 255-i*255.0/len(points_list)
        cv2.fillPoly(im, np.array(points_list[i], dtype=int), [color, color, color])
    cv2.putText(im, 'points:' + str(points_list[0]), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(im, 'cam[:,5]:{} '.format(len(points_list)) + str(cam[:,5]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return im

def Render(camera_pos, target_pos, area_length):
    length = 600
    area_length = 1000  # for random cam loc


    img = np.zeros((length + 1, length + 1, 3)) + 255
    num_cam = len(camera_pos)
    camera_position = [camera_pos[i][:2] for i in range(num_cam)]
    target_position = target_pos[:2]
    camera_position = length * (1 - np.array(camera_position) / area_length) / 2
    target_position = length * (1 - np.array(target_position) / area_length) / 2

    abs_angles = [camera_pos[i][4] for i in range(num_cam)]

    color_dict = {'red': [255, 0, 0], 'black': [0, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                  'darkred': [128, 0, 0], 'yellow': [255, 255, 0], 'deeppink': [255, 20, 147]}

    # plot camera
    for i in range(num_cam):
        img[int(camera_position[i][1])][int(camera_position[i][0])][0] = color_dict["black"][0]
        img[int(camera_position[i][1])][int(camera_position[i][0])][1] = color_dict["black"][1]
        img[int(camera_position[i][1])][int(camera_position[i][0])][2] = color_dict["black"][2]

    # plot target
    img[int(target_position[1])][int(target_position[0])][0] = color_dict['blue'][0]
    img[int(target_position[1])][int(target_position[0])][1] = color_dict['blue'][1]
    img[int(target_position[1])][int(target_position[0])][2] = color_dict['blue'][2]

    plt.cla()
    plt.imshow(img.astype(np.uint8))

    # get camera's view space positions
    visua_len = 50  # length of arrow
    for i in range(num_cam):
        theta = abs_angles[i] - 90.0
        dx = visua_len * math.sin(theta * math.pi / 180)
        dy = - visua_len * math.cos(theta * math.pi / 180)
        plt.arrow(camera_position[i][0], camera_position[i][1], dx, dy, width=0.1, head_width=8, head_length = 8, length_includes_head=True)

        plt.annotate(str(i), xy=(camera_position[i][0], camera_position[i][1]),
                     xytext=(camera_position[i][0], camera_position[i][1]), fontsize=10, color='blue')
        # plt.plot(camera_position[i][0], camera_position[i][1],'ro', color='blue')
    plt.plot(target_position[0], target_position[1], 'ro')
    plt.pause(0.01)

class GoalNavAgent(object):
    """The world's simplest agent!"""
    def __init__(self, goal_area, start_area, safe_start, nav='New'):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_area = goal_area
        self.goal = self.generate_goal(self.goal_area)
        if 'Base' in nav:
            self.discrete = True
        else:
            self.discrete = False
        if 'Old' in nav:
            self.max_len = 30
        else:
            self.max_len = 100
        self.velocity = 15
        self.walk_d = 0
        self.start_area = start_area
        self.safe_start =safe_start

    def act(self, pose, reach_corner):
        self.step_counter += 1
        self.target_pos = pose
        self.goal_pos = self.goal
        self.reach_corner = reach_corner

        # move target
        target_dy = self.goal_pos[1] - self.target_pos[1]
        target_dx = self.goal_pos[0] - self.target_pos[0]

        d = np.sqrt(target_dy * target_dy + target_dx * target_dx)
        # self.velocity *= np.random.normal(1, 0.1)

        if self.reach_corner or self.step_counter >= 10:
            self.goal = self.generate_goal(self.goal_area)

        if d > 20:
            if target_dy == 0 or target_dx == 0:
                # self.target_pos = np.array([np.random.randint(self.start_area[0], self.start_area[1]),
                #                             np.random.randint(self.start_area[2], self.start_area[3]),
                #                             self.safe_start[0][-1]])
                self.target_pos[1] += np.random.randint(-10, 10)
                self.target_pos[0] += np.random.randint(-10, 10)
            else:
                delta_y = int(target_dy / abs(target_dy)) * self.velocity
                delta_x = int(target_dx / abs(target_dx)) * self.velocity
                self.target_pos[1] += delta_y
                self.target_pos[0] += delta_x
                self.walk_d = np.sqrt(delta_x * delta_x + delta_y * delta_y)
            self.velocity = 20

        else:  # reach goal, and generate a new goal
            self.goal = self.generate_goal(self.goal_area)
            self.step_counter = 0

        return self.target_pos

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal = self.generate_goal(self.goal_area)
        self.pose_last = None

    def generate_goal(self, goal_area):  # NAV IN 2D
        x = np.random.randint(goal_area[0], goal_area[1])
        y = np.random.randint(goal_area[2], goal_area[3])
        goal = np.array([x, y])
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 10

class GoalNavAgentTest(object):
    """The world's simplest agent!"""

    def __init__(self, action_space, goal_list=None):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal_list = goal_list

        self.discrete = True

        self.max_len = 100
        self.velocity = 15
        self.walk_d = 0
        self.goal = self.generate_goal()

    def act(self, pose):
        self.step_counter += 1
        self.target_pos = pose
        self.goal_pos = self.goal

        # move target
        target_dy = self.goal_pos[1] - self.target_pos[1]
        target_dx = self.goal_pos[0] - self.target_pos[0]

        d = np.sqrt(target_dy * target_dy + target_dx * target_dx)

        if d > 20:
            if target_dy == 0 or target_dx == 0:
                self.target_pos[1] += np.random.randint(-10, 10)
                self.target_pos[0] += np.random.randint(-10, 10)
            else:
                delta_y = int(target_dy / abs(target_dy)) * self.velocity
                delta_x = int(target_dx / abs(target_dx)) * self.velocity
                self.target_pos[1] += delta_y
                self.target_pos[0] += delta_x
                self.walk_d = np.sqrt(delta_x * delta_x + delta_y * delta_y)
                self.velocity = 15

        else:  # reach goal, and generate a new goal
            self.goal = self.generate_goal()
            self.step_counter = 0

        return self.target_pos

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal = self.generate_goal()
        self.pose_last = None
        self.goal_id = 0

    def generate_goal(self):

        if len(self.goal_list) != 0:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        self.goal_id += 1

        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 50

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now