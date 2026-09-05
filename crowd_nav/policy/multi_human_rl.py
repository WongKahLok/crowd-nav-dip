import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import JointState, ObservableState
from crowd_nav.policy.cadrl import CADRL


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action)
                else:
                    next_human_states, reward, done, info = self.env.predict_linear(state, action)
                value = reward
                if not done:
                    next_input = self.transform(JointState(next_self_state, next_human_states)).unsqueeze(0)
                    with torch.no_grad():
                        next_value = self.model(next_input).item()
                    value += pow(self.gamma, self.time_step * state.self_state.v_pref) * next_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        humans = state.human_states
        if not humans:
            # A zero interaction row carries the robot features without adding an agent.
            nav = state.self_state
            dummy = ObservableState(nav.px, nav.py, 0, 0, 0)
            state_tensor = self.rotate(torch.tensor([nav + dummy], dtype=torch.float32, device=self.device))
            state_tensor[:, self.self_state_dim:] = 0
        else:
            state_tensor = self.rotate(torch.tensor(
                [state.self_state + human for human in humans], dtype=torch.float32, device=self.device))
        if self.with_om:
            maps = self.build_occupancy_maps(humans).to(self.device)
            state_tensor = torch.cat([state_tensor, maps], dim=1)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        if len(human_states) <= 1:
            return torch.zeros((1, self.cell_num ** 2 * self.om_channel_size))
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

