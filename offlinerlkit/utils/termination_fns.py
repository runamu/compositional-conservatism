import numpy as np
import torch

'''
For neorl envs, codes are adapted from https://github.com/yihaosun1124/mobile/blob/main/utils/termination_fns.py (numpy -> torch)
'''

def obs_unnormalization(termination_fn, obs_mean, obs_std):
    def thunk(obs, act, next_obs):
        obs = obs*obs_std + obs_mean
        next_obs = next_obs*obs_std + obs_mean
        return termination_fn(obs, act, next_obs)
    return thunk

def termination_fn_dummy(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = torch.tensor([False] * obs.shape[0], device=next_obs.device)

    done = done[:, None]
    return done

def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    # always torch
    not_done = torch.logical_and(torch.all(next_obs > -100, dim=-1), torch.all(next_obs < 100, dim=-1))

    # if isinstance(next_obs, torch.Tensor):
    #     not_done = torch.logical_and(torch.all(next_obs > -100, dim=-1), torch.all(next_obs < 100, dim=-1))
    # else:
    #     not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))

    done = ~not_done
    done = done[:, None]
    return done

def termination_fn_neorl_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = torch.tensor([False] * obs.shape[0], dtype=torch.bool).unsqueeze(1)
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]

    # always torch
    finite_check = torch.isfinite(next_obs).all(dim=-1)
    bound_check = torch.abs(next_obs[:, 1:]).lt(100).all(dim=-1)
    angle_check = torch.abs(angle).lt(0.2)
    not_done = finite_check * bound_check * (height > .7) * angle_check

    # if isinstance(next_obs, torch.Tensor):
    #     # not_done = torch.logical_and(torch.all(next_obs > -100, dim=-1), torch.all(next_obs < 100, dim=-1)) \
    #     #             * (height > .7) \
    #     #             * torch.logical_and(torch.all(angle > -.2, dim=-1), torch.all(angle < .2, dim=-1))
    #     finite_check = torch.isfinite(next_obs).all(dim=-1)
    #     bound_check = torch.abs(next_obs[:, 1:]).lt(100).all(dim=-1)
    #     angle_check = torch.abs(angle).lt(0.2)
    #     not_done = finite_check * bound_check * (height > .7) * angle_check
    # else:
    #     not_done =  np.isfinite(next_obs).all(axis=-1) \
    #                 * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
    #                 * (height > .7) \
    #                 * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_neorl_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 1:2]
    angle = next_obs[:, 2:3]
    state = next_obs[:, 3:]

    min_state, max_state = (-100.0, 100.0)
    min_z, max_z = (0.7, float('inf'))
    min_angle, max_angle = (-0.2, 0.2)

    healthy_state = torch.all(torch.logical_and(min_state < state, state < max_state), dim=-1, keepdim=True)
    healthy_z = torch.logical_and(min_z < z, z < max_z)
    healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)

    is_healthy = torch.logical_and(torch.logical_and(healthy_state, healthy_z), healthy_angle)

    # done = torch.logical_not(is_healthy).unsqueeze(1)
    done = torch.logical_not(is_healthy)
    return done


def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
    else:
        done = torch.tensor([False]).repeat(len(obs))
        done = done[:, None]

    return done

def termination_fn_antangle(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_ant(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]

    # always torch
    not_done = (torch.logical_and(torch.all(next_obs > -100, dim=-1), torch.all(next_obs < 100, dim=-1))
                    * (height > 0.8)
                    * (height < 2.0)
                    * (angle > -1.0)
                    * (angle < 1.0))

    # if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
    #     not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
    #                 * (height > 0.8) \
    #                 * (height < 2.0) \
    #                 * (angle > -1.0) \
    #                 * (angle < 1.0)
    # else:
    #     not_done = (torch.logical_and(torch.all(next_obs > -100, dim=-1), torch.all(next_obs < 100, dim=-1))
    #                 * (height > 0.8)
    #                 * (height < 2.0)
    #                 * (angle > -1.0)
    #                 * (angle < 1.0))

    done = ~not_done
    done = done[:, None]
    return done

def termination_fn_neorl_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    min_z, max_z = (0.8, 2.0)
    min_angle, max_angle = (-1.0, 1.0)
    min_state, max_state = (-100.0, 100.0)

    z = next_obs[:, 1:2]
    angle = next_obs[:, 2:3]
    state = next_obs[:, 3:]

    healthy_state = torch.all(torch.logical_and(min_state < state, state < max_state), dim=-1, keepdim=True)
    healthy_z = torch.logical_and(min_z < z, z < max_z)
    healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
    is_healthy = torch.logical_and(torch.logical_and(healthy_state, healthy_z), healthy_angle)
    done = torch.logical_not(is_healthy)
    return done

def termination_fn_point2denv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_point2dwallenv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_pendulum(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros((len(obs), 1))
    return done

def termination_fn_humanoid(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:,0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:,None]
    return done


def terminaltion_fn_hammer(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = torch.tensor([False] * obs.shape[0], dtype=torch.bool)

    done = done.unsqueeze(1)
    return done



def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    obj_pos = next_obs[:, 24:27]
    done = obj_pos[:, 2] < 0.075

    done = done.unsqueeze(1)
    return done


def termination_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = torch.zeros(obs.shape[0], dtype=torch.bool)

    done = done.unsqueeze(1)
    return done


# from mopo codes in romi
def termination_fn_maze(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
        done = np.zeros((obs.shape[0], 1)).astype(bool)
    else:
        done = torch.zeros((obs.shape[0], 1), dtype=torch.bool)

    return done

# from mopo codes in romi
def termination_fn_antmaze(obs, act, next_obs, env):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
        done = np.linalg.norm(next_obs[:, :2] - env.target_goal, axis=1) <= 0.5
    else:
        done = torch.linalg.norm(next_obs[:, :2] - torch.tensor(env.target_goal, device=next_obs.device), dim=1) <= 0.5
    done = done[:, None]

    return done

def termination_fn_slider(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
        done = np.zeros((obs.shape[0], 1)).astype(bool)
    else:
        done = torch.zeros((obs.shape[0], 1), dtype=torch.bool, device=next_obs.device)
    return done

def termination_fn_adroit(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    if type(obs) == np.ndarray and type(act) == np.ndarray and type(next_obs) == np.ndarray:
        done = np.zeros((obs.shape[0], 1)).astype(bool)
    else:
        done = torch.zeros((obs.shape[0], 1), dtype=torch.bool, device=next_obs.device)
    return done

# # make it static
# def static_get_termination_fn(task):


def get_termination_fn(task):
    if 'halfcheetahvel' in task:
        return termination_fn_halfcheetahveljump
    elif 'halfcheetah' in task:
        return termination_fn_halfcheetah
    elif 'hopper' in task:
        return termination_fn_hopper
    elif 'antangle' in task:
        return termination_fn_antangle
    elif 'ant' in task:
        return termination_fn_ant
    elif 'walker2d' in task:
        return termination_fn_walker2d
    elif 'HalfCheetah-v3' in task:
        return termination_fn_neorl_halfcheetah
    elif 'Hopper-v3' in task:
        return termination_fn_neorl_hopper
    elif 'Walker2d-v3' in task:
        return termination_fn_neorl_walker2d
    elif 'point2denv' in task:
        return termination_fn_point2denv
    elif 'point2dwallenv' in task:
        return termination_fn_point2dwallenv
    elif 'pendulum' in task:
        return termination_fn_pendulum
    elif 'humanoid' in task:
        return termination_fn_humanoid
    elif 'maze2d' in task:
        return termination_fn_maze
    elif 'antmaze' in task:
        return termination_fn_antmaze
    elif 'pen' in task:
        return termination_fn_pen
    elif 'door' in task:
        return termination_fn_door
    elif 'slider' in task:
        return termination_fn_slider
    elif 'hammer' in task:
        return terminaltion_fn_hammer
    elif 'adroit' in task:
        return termination_fn_adroit
    else:
        raise np.zeros
