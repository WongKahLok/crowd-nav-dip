import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.crowd_sim import CrowdSim_test, CrowdSim_train
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: needs the trained weights and a few seconds')


def write_map(path, white_rows=None, blocks=(), size=50):
    """A 50x50 map PNG: white = water, black = land.

    `blocks` are (col0, row0, col1, row1) in *image* coordinates, where row 0 becomes the
    top of the world once ``load_from_image`` has flipped it.
    """
    image = Image.new('L', (size, size), 0 if white_rows is not None else 255)
    px = image.load()
    for row in (white_rows or ()):
        for col in range(size):
            px[col, row] = 255
    for col0, row0, col1, row1 in blocks:
        for row in range(row0, row1):
            for col in range(col0, col1):
                px[col, row] = 0
    image.save(str(path))
    return path


class FixedPolicy(Policy):
    def __init__(self, action=ActionXY(0, 0)):
        super().__init__()
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.action = action

    def predict(self, state):
        self.last_state = state
        return self.action


@pytest.fixture
def env_cfg():
    cfg = configparser.RawConfigParser()
    cfg.read(ROOT / 'crowd_nav/configs/env.config')
    return cfg


@pytest.fixture
def make_env(env_cfg):
    def build(kind='map', obstacles=(), dynamic=()):
        env = CrowdSim_test() if kind == 'map' else CrowdSim_train()
        env.configure(env_cfg, SimpleNamespace(colregs=False))
        robot = Robot(env_cfg, 'robot')
        robot.set_policy(FixedPolicy())
        env.set_robot(robot)
        if kind == 'map':
            env.set_obstacles(list(obstacles))
            env.set_init_para(dict(x_min=0, y_min=0, x_max=50, y_max=50,
                                   grid_res=1, nearby_dist=5))
            env.set_local_goal_list([(18, 5), (5, 5)])
            env.reset('test', 0, list(dynamic))
        else:
            env.reset('test', 0)
            env.humans = []
            env.human_times = []
            robot.set(5, 5, 18, 5, 0, 0, 0)
        return env
    return build


@pytest.fixture
def make_sarl():
    def build(env, query=False, weights=True, with_om=False, cls=SARL):
        cfg = configparser.RawConfigParser()
        cfg.read(ROOT / 'crowd_nav/data/output-trained/policy.config')
        cfg.set('action_space', 'query_env', str(query))
        cfg.set('sarl', 'with_om', str(with_om))
        policy = cls()
        policy.configure(cfg)
        if weights:
            state = torch.load(ROOT / 'crowd_nav/data/output-trained/rl_model.pth', map_location='cpu')
            policy.model.load_state_dict(state, strict=True)
        policy.set_phase('test')
        policy.set_device(torch.device('cpu'))
        policy.set_env(env)
        policy.time_step = env.time_step
        env.robot.set_policy(policy)
        env.action_values = []
        env.attention_weights = []
        return policy
    return build


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
