from gym.envs.registration import register

register(
    id='CrowdSimTrain-v0',
    entry_point='crowd_sim.envs:CrowdSim_train',
)
register(
    id='CrowdSimTest-v0',
    entry_point='crowd_sim.envs:CrowdSim_test',
)

