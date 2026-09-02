import logging
import argparse
import configparser
import os
import math
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.theta_star import HybridEnv, ThetaStar
from crowd_nav.utils.theta_star import interpolate_path, get_local_goal
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import Timeout

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--colregs', default=False, action='store_true')
    parser.add_argument('--static_map', type=str, default=None)
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config  # was args.env_config: wrong config loaded when --model_dir is omitted

    if args.static_map is not None:
        if not os.path.exists(args.static_map):
            print('Static map not exists:', args.static_map)
            return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))
    env_loaded = HybridEnv(x_range=50, y_range=50, resolution=1, safety_distance=0)
    env_loaded.load_from_image(args.static_map)
    env_loaded_safety_dist = HybridEnv(x_range=50, y_range=50, resolution=1, safety_distance=1)
    env_loaded_safety_dist.load_from_image(args.static_map)
    planner = ThetaStar(start=(40, 5), goal=(32, 42), env=env_loaded_safety_dist)
    cost, path, expand = planner.plan()
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSimTest-v0')
    env.configure(env_config, args)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    env.set_obstacles(env_loaded.get_static_obstacles())
    env.set_init_para(env_loaded.get_env_para())
    env.set_local_goal_list(env_loaded.grid_2_drl(path))

    policy.set_phase(args.phase)
    policy.set_device(device)
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    local_goal_dist, monitor_dist = 1, 5
    dym_obs_list = [{'px': 27, 'py': 29, 'gx': 37, 'gy': 9}, {'px': 10, 'py': 20, 'gx': 40, 'gy': 20},
                    {'px': 28, 'py': 55, 'gx': 28, 'gy': 20}, {'px': 24.5, 'py': 67, 'gx': 24.5, 'gy': 20}]
    ob = env.reset(args.phase, args.test_case, dym_obs_list)
    path_dense = interpolate_path(env.global_path)[1:]
    
    one_goal = get_local_goal(robot.get_full_state(), ob, path_dense, local_goal_dist, monitor_dist)
    robot.gx, robot.gy = one_goal[0], one_goal[1]
    cnt_idx, cnt_max = 0, 200
    info = None  # outcome of the episode: set below, either by env.step() or the iteration cap
    while not math.dist((robot.px, robot.py), path_dense[-1])<robot.radius:
        ob = env.light_reset(args.phase, args.test_case)  # re-observe: dynamic humans + nearby static cells
        action = robot.act(ob)  # SARL: score all 81 candidate actions, pick the best
        ob, _, done, info = env.step(action)  # advance the world by one time_step; done=True on collision/timeout/reach-goal
        if done:
            break  # stop right away instead of continuing and overwriting info on the next tick
        one_goal = get_local_goal(robot.get_full_state(), ob, path_dense, local_goal_dist, monitor_dist)
        robot.gx, robot.gy = one_goal[0], one_goal[1]  # slide the local goal forward along the path
        cnt_idx += 1
        if cnt_idx>=cnt_max:
            info = Timeout()  # ran out of control-loop iterations, distinct from env's own time_limit-based Timeout
            break

    if args.visualize:
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        print('test results:', info)


if __name__ == '__main__':
    main()
