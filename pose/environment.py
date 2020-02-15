from __future__ import division

from pose_env import Pose_Env

def create_env(env_id):

    reset_type = env_id.split('-v')[1]
    env = Pose_Env(int(reset_type))

    return env






