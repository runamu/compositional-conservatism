import numpy as np
import torch

# print(env.model.nq, env.model.nv)
# env._get_obs()

def set_state_fn_halfcheetah(env, state):
    qpos = state[:9]
    qvel = state[9:]
    env.set_state(qpos, qvel)

def set_state_fn_hopper(env, state):
    qpos = state[:6]
    qvel = state[6:]
    env.set_state(qpos, qvel)

def set_state_fn_walker2d(env, state):
    qpos = state[:9]
    qvel = state[9:]
    env.set_state(qpos, qvel)


def get_set_state_fn(task):
    # gym
    if 'halfcheetah' in task:
        return set_state_fn_halfcheetah
    elif 'hopper' in task:
        return set_state_fn_hopper
    elif 'walker2d' in task:
        return set_state_fn_walker2d
    # neorl
    elif 'HalfCheetah' in task:
        return set_state_fn_halfcheetah
    elif 'Hopper' in task:
        return set_state_fn_hopper
    elif 'Walker2d' in task:
        return set_state_fn_walker2d
    else:
        return None