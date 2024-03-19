import d4rl
import numpy as np
import torch

# neorl
REF_EXPERT_SCORE = {
    "HalfCheetah-v3-low" : 12284 ,
    "HalfCheetah-v3-medium" : 12284 ,
    "HalfCheetah-v3-high" : 12284 ,
    "Hopper-v3-low" : 3294 ,
    "Hopper-v3-medium" : 3294 ,
    "Hopper-v3-high" : 3294 ,
    "Walker2d-v3-low" : 5143 ,
    "Walker2d-v3-medium" : 5143 ,
    "Walker2d-v3-high" : 5143 ,
}
REF_RANDOM_SCORE = {
    "HalfCheetah-v3-low" : -298 ,
    "HalfCheetah-v3-medium" : -298 ,
    "HalfCheetah-v3-high" : -298 ,
    "Hopper-v3-low" : 5 ,
    "Hopper-v3-medium" : 5 ,
    "Hopper-v3-high" : 5 ,
    "Walker2d-v3-low" : 1 ,
    "Walker2d-v3-medium" : 1 ,
    "Walker2d-v3-high" : 1 ,
}

def get_normalized_score_neorl(task, ep_reward):
    ref_expert_score = REF_EXPERT_SCORE[task]
    ref_random_score = REF_RANDOM_SCORE[task]
    norm_ep_rew = (ep_reward - ref_random_score) / (ref_expert_score - ref_random_score) * 100
    return norm_ep_rew

def get_normalized_std_score_neorl(task, ep_reward_std):
    ref_expert_score = REF_EXPERT_SCORE[task]
    ref_random_score = REF_RANDOM_SCORE[task]
    norm_ep_rew_std = ep_reward_std / (ref_expert_score - ref_random_score) * 100
    return norm_ep_rew_std

def get_normalized_std_score(env, ep_reward_std):
    env_name = env.unwrapped.spec.id
    ref_min_score = d4rl.infos.REF_MIN_SCORE[env_name]
    ref_max_score = d4rl.infos.REF_MAX_SCORE[env_name]
    norm_ep_rew_std = env.get_normalized_score(ep_reward_std) * 100 + ref_min_score / (ref_max_score - ref_min_score) * 100
    return norm_ep_rew_std

def get_eval_task(task):
    if 'eval' not in task:
        eval_task = task.split('-')[0] + '-eval-' + '-'.join(task.split('-')[1:])
        if 'antmaze' in task:
            assert 'v0' in task
            # If we allow training on v2
            # eval_task = '-'.join(eval_task.split('-')[:-1] + ['v0'])
        return eval_task
    else:
        return task

def get_expert_task(task):
    if 'maze' in task:
        return task
    if any(keyword in task for keyword in ['slider', 'adroit', 'reach', 'push']):
        return task
    expert_task = task.split('-')[0] + '-expert-' + task.split('-')[-1]
    return expert_task

def get_combinations(A, B):
    # Repeat A by the number of B and tile B by the number of A
    A_combinations = np.repeat(A, len(B), axis=0)
    B_combinations = np.tile(B, (len(A), 1))
    return A_combinations.tolist(), B_combinations.tolist()

def romi_antmaze_timeout(dataset):
    threshold = np.mean(np.linalg.norm(dataset['observations'][1:, :2] - dataset['observations'][:-1, :2], axis=1))
    print('threshold', threshold)
    for i in range(dataset['observations'].shape[0]):
        dataset['timeouts'][i] = False
    for i in range(dataset['observations'].shape[0] - 1):
        gap = np.linalg.norm(dataset['observations'][i + 1, :2] - dataset['observations'][i, :2])
        if gap > threshold * 10:
            dataset['timeouts'][i] = True
    return dataset

def romi_processed_qlearning_dataset(env, env_name, timeout_frame=False, done_frame=False, return_dataset=False):
    dataset = env.get_dataset()
    if 'antmaze' in env_name: # handle wrong timeout
        dataset = romi_antmaze_timeout(dataset)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not timeout_frame) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if (not done_frame) and done_bool:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    if return_dataset:
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }, dataset
    else:
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }

def print_grad_norm(module):
    for name, param in module.named_parameters():
        if param.grad is not None:
            if torch.all(param.grad == 0):
                print("{:<60} {:<10}".format(name, "all_zero"))
            else:
                print("{:<60} {:.10f}".format(name, param.grad.data.norm(2)))
        else:
            print("{:<60} {:<10}".format(name, "None"))
    print()

def fix_anchor_mode(anchor_mode):
    fix_dict = {
        'noised_obs': 'noised',
        'closest_obs': 'closest_all',
        'closest_obs_sample': 'closest',
        'top_10pct': 'top_10',
    }
    if anchor_mode in fix_dict:
        return fix_dict[anchor_mode]
    else:
        return anchor_mode
