import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from distutils.util import strtobool
import os
import random
import socket

import numpy as np
import torch

if __name__ == '__main__':
    import gym
    import d4rl.gym_mujoco

    import time
    from offlinerlkit.modules import EnsembleDynamicsModel, AnchorActor
    from offlinerlkit.dynamics import ReverseEnsembleDynamics, EnsembleDynamics
    from offlinerlkit.nets import MLP
    from offlinerlkit.utils.scaler import StandardScaler
    from offlinerlkit.utils.termination_fns import get_termination_fn, obs_unnormalization
    from offlinerlkit.utils.load_dataset import qlearning_dataset, load_neorl_dataset
    from offlinerlkit.buffer import ReplayBuffer
    from offlinerlkit.utils.logger import Logger, make_log_dirs
    from offlinerlkit.policy_trainer import AnchorSeekerTrainer

    import romi.continuous_bcq.ReverseBC as ReverseBC
    from offlinerlkit.policy import DivergentPolicy

"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-v2: rollout-length=5, penalty-coef=0.5
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-replay-v2: rollout-length=5, penalty-coef=2.5
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=2.5
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=2.5
hopper-medium-expert-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=2.5
"""

"""
suggested hypers from the mopo paper

halfcheetah-medium-v2: rollout-length=1, penalty-coef=1.0
hopper-medium-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-v2: rollout-length=5, penalty-coef=5.0
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=1.0
hopper-medium-replay-v2: rollout-length=5, penalty-coef=1.0
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=1.0
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=1.0
hopper-medium-expert-v2: rollout-length=5, penalty-coef=1.0
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=2.0
"""

def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--algo_name", type=str, default="reverse")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load_policy_path", "-lpp", type=str, default=None)
    parser.add_argument("--hidden_dims", "-hd", type=int, nargs='*', default=[256, 256])

    parser.add_argument("--policy_train", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--split_validation", "-sv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--split_k_means", "-skm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--split_n_clusters", "-snc", type=int, default=5, help="the number of clusters in k-means clustering")
    parser.add_argument("--max_holdout_size", "-mhs", type=int, default=1000)
    parser.add_argument("--reverse_policy_mode", "-rpm", type=str, default="cvae", help="reverse policy mode, 'divergent' or 'cvae' or 'uncertainty'")

    ## divergent policy
    parser.add_argument("--action_mode", "-acm", type=str, default="cvae", help="when using divergent policy, how to sample actions, 'gaussian' or 'sample' or 'dissim'")
    parser.add_argument("--scale_coef", "-sc", type=float, default=None)
    parser.add_argument("--noise_coef", "-nc", type=float, default=None)

    # parser.add_argument("--foward_align_train", "-fat", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    # parser.add_argument("--uncertainty_level", "-ul", type=str, default=None, help="uncertainty level for the reverse imagination data generation")
    # parser.add_argument("--toggle_fake_buffer_size", "-tfbs", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="set fakebuffer size as realbuffer size")
    # parser.add_argument("--semi_fake_buffer", "-sfb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="add fakebuffer to realbuffer")

    parser.add_argument("--anchor_seeker_hidden_dims", "-ashd", type=int, nargs='*', default=[100, 100])
    parser.add_argument("--asp_layernorm", "-aln", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False)

    parser.add_argument("--dynamics_lr", "-dlr", type=float, default=1e-3)
    parser.add_argument("--policy_lr", "-plr", type=float, default=1e-3)

    parser.add_argument("--dynamics_hidden_dims", "-dhd", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics_weight_decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n_ensemble", "-ne", type=int, default=7)
    parser.add_argument("--n_elites", type=int, default=5)
    parser.add_argument("--rollout_epoch", "-re", type=int, default=100)
    parser.add_argument("--rollout_batch_size", "-rbs", type=int, default=3000)
    parser.add_argument("--rollout_length", "-rl", type=int, default=5)
    parser.add_argument("--model_retain_epochs", type=int, default=5)
    parser.add_argument("--real_ratio", "-rr", type=float, default=0.05)
    parser.add_argument("--load_dynamics_path", "-ldp", type=str, default=None)
    parser.add_argument("--load_reverse_dynamics_path", "-lrdp", type=str, default=None)

    parser.add_argument("--rollout_augmentation", "-ra", type=lambda x: bool(strtobool(x)), nargs="?", const=True, default=False) ### rollout only in the eval phase
    parser.add_argument("--holdout_ratio", "-hr", type=float, default=0.1) ## default in ensemble_dynamics.py
    parser.add_argument("--logvar_loss_coef", "-llc", type=float, default=0.01) ## default in ensemble_dynamics.py
    parser.add_argument("--step_loss_coef", "-slc", type=float, default=0.0)
    parser.add_argument("--max_epochs_since_update", "-mesu", type=int, default=5) ## default in ensemble_dynamics.py

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--asp_epoch", '-aspe',type=int, default=50)
    parser.add_argument("--asp_lr", "-asplr", type=float, default=1e-3)
    parser.add_argument("--asp_batch_size", '-bbs', type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", type=str, default="oos-debug",
        help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default='rloos_',
        help="the entity (team) of wandb's project")
    parser.add_argument("--wandb_tags", type=str, nargs='*', default=None,
        help="tags for wandb")

    if argv is not None:
        args, unknown_args = parser.parse_known_args(argv)
        args._original_parser = parser
        return args
    return parser.parse_args(argv)


def train(args):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S_%Z")

    if args.load_policy_path:
        import json
        hyperparams_path = os.path.join(args.load_policy_path, "../record/hyper_param.json")
        with open(hyperparams_path, "r") as f:
            policy_hyperparams = json.load(f)
        assert args.task == policy_hyperparams["task"]
        args.policy_args = policy_hyperparams

        args.hidden_dims=policy_hyperparams["hidden_dims"] if "hidden_dims" in policy_hyperparams else args.hidden_dims
        args.rollout_length=policy_hyperparams["rollout_length"] if "rollout_length" in policy_hyperparams else args.rollout_length
        args.rollout_augmentation=policy_hyperparams["rollout_augmentation"] if "rollout_augmentation" in policy_hyperparams else args.rollout_augmentation
        args.load_reverse_dynamics_path=policy_hyperparams["load_reverse_dynamics_path"] if "load_reverse_dynamics_path" in policy_hyperparams else args.load_reverse_dynamics_path

    if args.load_reverse_dynamics_path and not args.load_reverse_dynamics_path=='dummy':
        import json
        hyperparams_path = os.path.join(args.load_reverse_dynamics_path, "../record/hyper_param.json")
        with open(hyperparams_path, "r") as f:
            dynamics_hyperparams = json.load(f)
        # assert args.task == dynamics_hyperparams["task"]
        args.reverse_dynamics_args = dynamics_hyperparams
        args.dynamics_hidden_dims=dynamics_hyperparams["dynamics_hidden_dims"]
        args.n_ensemble=dynamics_hyperparams["n_ensemble"]
        args.n_elites=dynamics_hyperparams["n_elites"]

    if 'v3' in args.task:
        import neorl
        task, version, data_type = tuple(args.task.split("-"))
        env = neorl.make(task+'-'+version)
        dataset = load_neorl_dataset(env, data_type)
    else:
        # create env and dataset
        env = gym.make(args.task)
        dataset = qlearning_dataset(env)

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    highs = env.action_space.high
    neg_lows = -env.action_space.low
    assert np.all(highs == highs[0]) and np.all(neg_lows == highs[0])
    args.max_action = env.action_space.high[0]
    args.entropy_weight = 0.5

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create buffer
    real_buffer = ReplayBuffer(
        args,
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)

    fake_buffer = ReplayBuffer(
        args,
        buffer_size=args.rollout_batch_size*args.rollout_length*args.rollout_epoch,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    obs_mean_np, obs_std_np = np.zeros_like(dataset['observations'][0]), np.ones_like(dataset['observations'][0]) # dummy
    obs_mean, obs_std = torch.tensor(obs_mean_np, dtype=torch.float32, device=args.device), torch.tensor(obs_std_np, dtype=torch.float32, device=args.device)

    # create forward dynamics
    dynamics_model = EnsembleDynamicsModel(
        args,
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    dynamics_scaler = StandardScaler()
    termination_fn = obs_unnormalization(get_termination_fn(task=args.task), obs_mean, obs_std)
    dynamics = EnsembleDynamics(
        args,
        dynamics_model,
        dynamics_optim,
        dynamics_scaler,
        termination_fn,
    )

    if args.load_dynamics_path and not args.load_dynamics_path=='dummy':
        dynamics.load(args.load_dynamics_path)
        dynamics.model.eval()
        dynamics.model.requires_grad_(False)

    # create reverse dynamics
    load_reverse_dynamics_model = True if args.load_reverse_dynamics_path else False
    reverse_dynamics_model = EnsembleDynamicsModel(
        args,
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        with_reward=False, ######### for old dynamics path: True ????
        device=args.device
    )
    reverse_dynamics_optim = torch.optim.Adam(
        reverse_dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(reverse_dynamics_optim, args.rollout_epoch)
    reverse_dynamics_scaler = StandardScaler()
    termination_fn = obs_unnormalization(get_termination_fn(task=args.task), obs_mean, obs_std)
    reverse_dynamics = ReverseEnsembleDynamics(
        args,
        dynamics, ## for step loss
        reverse_dynamics_model,
        reverse_dynamics_optim,
        reverse_dynamics_scaler,
        termination_fn,
    )

    if args.load_reverse_dynamics_path and not args.load_reverse_dynamics_path=='dummy':
        reverse_dynamics.load(args.load_reverse_dynamics_path)

    # create reverse policy
    if args.reverse_policy_mode == 'divergent':
        policy = DivergentPolicy(
            args,
            env,
            dynamics,
            reverse_dynamics,
            real_buffer,
            args.action_dim,
            args.max_action,
            scale_coef=args.scale_coef,
            noise_coef=args.noise_coef,
            device=args.device,
            seed=args.seed,
            )
    elif args.reverse_policy_mode == 'cvae':
        policy = ReverseBC.ReverseBC(
            args,
            dynamics,
            reverse_dynamics,
            np.prod(args.obs_shape),
            args.action_dim,
            args.max_action,
            args.device,
            args.entropy_weight,
            args.policy_lr,
            )
    else:
        raise NotImplementedError

    # create anchor seeker
    anchor_seeking_backbone = MLP(input_dim=np.prod(args.obs_shape) , hidden_dims=args.anchor_seeker_hidden_dims, layernorm=args.asp_layernorm)
    anchor_seeking_policy = AnchorActor(args, anchor_seeking_backbone, owner = 'actor')

    # log
    log_dirs = make_log_dirs(
        args.task, args.algo_name, args.seed, vars(args),
        record_params=["action_mode", "scale_coef", "noise_coef"] if args.reverse_policy_mode == 'divergent' else [],
        timestamp=timestamp,
        init_run=True,
    )

    # logs
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "reverse_dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    if args.load_reverse_dynamics_path and not args.load_reverse_dynamics_path=='dummy':
        del args.reverse_dynamics_args

    if args.load_policy_path:
        del args.policy_args
        policy.load(args.load_policy_path)

    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = AnchorSeekerTrainer(
        args=args,
        anchor_seeking_policy=anchor_seeking_policy,
        reverse_dynamics=reverse_dynamics,
        dynamics=dynamics,
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_epoch, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
    )

    # train dynamics
    if not load_reverse_dynamics_model:
        reverse_dynamics.train(
            real_buffer.sample_all(),
            logger,
            holdout_ratio=args.holdout_ratio,
            logvar_loss_coef=args.logvar_loss_coef,
            max_epochs_since_update=args.max_epochs_since_update,
        )

    if args.policy_train:
        # cvae
        policy_trainer.train()

    # generate data
    load_reverse_imagination_path = policy_trainer.generate()

    # train anchor seeker
    policy_trainer.anchor_seeker_pretrain_reverse(load_reverse_imagination_path, args.asp_epoch, args.asp_batch_size, \
                                    args.asp_lr,logger)

    return log_dirs

if __name__ == "__main__":
    log_dirs = train(get_args())

