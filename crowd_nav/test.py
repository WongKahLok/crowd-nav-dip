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
from crowd_nav.utils.theta_star import interpolate_path
from crowd_nav.utils.local_goal import LocalGoalTracker
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import ReachGoal, Timeout

# start/goal of the other ships, in world coordinates. Every pair is checked against the
# static map and gets a land free route before it is handed to the simulator.
DYNAMIC_OBSTACLES = [{'px': 27, 'py': 29, 'gx': 37, 'gy': 9},
                     {'px': 10, 'py': 20, 'gx': 30, 'gy': 10},
                     {'px': 28, 'py': 46, 'gx': 28, 'gy': 20},
                     {'px': 24.5, 'py': 43, 'gx': 27, 'gy': 20},
                     {'px': 15, 'py': 35, 'gx': 35, 'gy': 25},
                     {'px': 40, 'py': 30, 'gx': 20, 'gy': 15}]


def build_dynamic_obstacles(env, obstacle_list, robot_route, seed=None):
    """
    Prepare the dynamic obstacles for the map that is actually loaded.

    The pairs of `obstacle_list` are a preferred scenario: each one is checked against
    the static map and gets the Theta* route it has to follow, because an obstacle that
    heads straight for its goal sails over whatever land lies in between. Pairs that do
    not fit this map (they were picked for a different one) are replaced by obstacles
    generated on free water that cross the ship's own path.

    :param env: HybridEnv holding the (inflated) static map
    :param obstacle_list: preferred start/goal pairs
    :param robot_route: the ship's global path in world coordinates, start first
    :param seed: makes the generated obstacles reproducible
    :return: usable obstacles, each with an extra 'path' entry
    """
    prepared = []
    for idx, obstacle in enumerate(obstacle_list):
        start = (obstacle['px'], obstacle['py'])
        goal = (obstacle['gx'], obstacle['gy'])
        if env.is_on_land(start) or env.is_on_land(goal):
            logging.warning('Dynamic obstacle %d does not fit this map: %s -> %s is not '
                            'on free water, replacing it', idx, start, goal)
            continue
        path = env.plan_world_path(start, goal)
        if path is None:
            logging.warning('Dynamic obstacle %d does not fit this map: no route from %s '
                            'to %s, replacing it', idx, start, goal)
            continue
        prepared.append(dict(obstacle, path=path))

    missing = len(obstacle_list) - len(prepared)
    if missing > 0:
        logging.info('Generating %d dynamic obstacle(s) for this map', missing)
        prepared += env.generate_crossing_obstacles(missing, robot_route, forb_list=prepared,
                                                    rng=np.random.RandomState(seed))
    if not prepared:
        raise RuntimeError('No dynamic obstacle could be placed on this map. The policy '
                           'needs at least one observed agent, check the static map and '
                           'the start/goal of the ship.')
    logging.info('%d dynamic obstacle(s) on the water', len(prepared))
    return prepared


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
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--local_goal_dist', type=float, default=1.0)
    parser.add_argument('--monitor_dist', type=float, default=5.0)
    parser.add_argument('--monitor_exit_dist', type=float, default=None,
                        help='obstacles stop being monitored beyond this, default 1.3 * monitor_dist')
    parser.add_argument('--lookahead_rate', type=float, default=1.0, help='m/s the look-ahead may change')
    parser.add_argument('--shore_clearance', type=float, default=0.5,
                        help='room between the ship-to-goal segment and the land, on top of the radius')
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error('max_steps must be positive')

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
        policy.get_model().load_state_dict(torch.load(model_weights, map_location=device))
    env_loaded = HybridEnv(x_range=50, y_range=50, resolution=1, safety_distance=0)
    env_loaded.load_from_image(args.static_map)
    env_loaded_safety_dist = HybridEnv(x_range=50, y_range=50, resolution=1, safety_distance=1)
    env_loaded_safety_dist.load_from_image(args.static_map)
    planner = ThetaStar(start=(40, 5), goal=(32, 42), env=env_loaded_safety_dist)
    cost, path, expand = planner.plan()
    if not path:
        logging.error('No route for the ship on %s, check its start and goal', args.static_map)
        return
    robot_route = env_loaded.grid_2_drl(path)[::-1]   # start first
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSimTest-v0')
    env.configure(env_config, args)
    env.time_limit = args.max_steps * env.time_step
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
    dym_obs_list = build_dynamic_obstacles(env_loaded_safety_dist, DYNAMIC_OBSTACLES,
                                           robot_route, seed=args.test_case or 0)
    ob = env.reset(args.phase, args.test_case, dym_obs_list)
    path_dense = interpolate_path(env.global_path)[1:]
    tracker = LocalGoalTracker.from_env(
        env, path_dense, goal_dist=args.local_goal_dist, monitor_dist=args.monitor_dist,
        monitor_exit=args.monitor_exit_dist, lookahead_rate=args.lookahead_rate,
        shore_clearance=args.shore_clearance)
    robot.gx, robot.gy = tracker.update(robot.get_full_state(), ob)
    info = None
    emergency_steps = 0
    for _ in range(args.max_steps):
        ob = env.light_reset(args.phase, args.test_case)
        action = robot.act(ob)
        emergency_steps += bool(getattr(policy, 'emergency', False))
        ob, _, done, info = env.step(action)
        if done:
            break
        robot.gx, robot.gy = tracker.update(robot.get_full_state(), ob)
    else:
        info = Timeout()
    if hasattr(policy, 'emergency'):
        logging.info('Emergency fallback used in %d of %d steps', emergency_steps, len(env.states))

    if args.visualize:
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and isinstance(info, ReachGoal):
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        print('test results:', info)


if __name__ == '__main__':
    main()
