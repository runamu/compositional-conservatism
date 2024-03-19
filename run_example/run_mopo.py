import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from distutils.util import strtobool
import random
from glob import glob
import re
import numpy as np
import torch

if __name__ == '__main__':
    import gym
    import d4rl
    import d4rl.gym_mujoco

    import time
    from offlinerlkit.nets import MLP
    from offlinerlkit.modules import TanhDiagGaussian, EnsembleDynamicsModel
    from offlinerlkit.modules.actor_module import AnchorActor, TransdActorProb
    from offlinerlkit.modules.critic_module import TransdCritic
    from offlinerlkit.modules import AnchorHandler
    from offlinerlkit.dynamics import EnsembleDynamics
    from offlinerlkit.utils.scaler import StandardScaler
    from offlinerlkit.utils.termination_fns import get_termination_fn
    from offlinerlkit.utils.load_dataset import qlearning_dataset, load_neorl_dataset
    from offlinerlkit.buffer import ReplayBuffer
    from offlinerlkit.utils.logger import Logger, make_log_dirs
    from offlinerlkit.policy_trainer import MBPolicyTrainer
    from offlinerlkit.policy import MOPOPolicy


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
    parser.add_argument("--algo_name", type=str, default="mopo")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--dynamics_lr", "-dlr", type=float, default=1e-3)
    parser.add_argument("--dynamics_weight_decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--dynamics_hidden_dims", "-dhd", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--n_ensemble", "-ne", type=int, default=7)
    parser.add_argument("--n_elites", type=int, default=5)
    parser.add_argument("--load_dynamics_path", "-ldp", type=str, default=None)

    parser.add_argument("--rollout_augmentation", "-ra", type=lambda x: bool(strtobool(x)), nargs="?", const=True, default=True)
    parser.add_argument("--rollout_freq", "-rf", type=int, default=1000)
    parser.add_argument("--rollout_epoch", "-re", type=int, default=100)
    parser.add_argument("--rollout_batch_size", "-rbs", type=int, default=50000)
    parser.add_argument("--rollout_length", "-rl", type=int, default=1)
    parser.add_argument("--model_retain_epochs", type=int, default=5)
    parser.add_argument("--real_ratio", "-rr", type=float, default=0.05)

    parser.add_argument("--penalty_coef", "-pc", type=float, default=2.5)
    parser.add_argument("--uncertainty_mode", "-um", type=str, default='aleatoric')

    parser.add_argument("--hidden_dims", "-hd", type=int, nargs='*', default=[100, 100])
    parser.add_argument("--tr_hidden_dims", "-thd", type=int, nargs='*', default=[64, 64], help="transduction hidden dims")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_alpha", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--target_entropy", type=int, default=None)
    parser.add_argument("--alpha_lr", type=float, default=1e-4)
    parser.add_argument("--actor_lr", "-aclr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", "-clr", type=float, default=3e-4)

    parser.add_argument("--actor_horizon_len", "-ahl", type=int, default=None)
    parser.add_argument("--critic_horizon_len", "-chl", type=int, default=None)
    parser.add_argument("--embedding_dim", "-ed", type=int, default=4)
    parser.add_argument("--layernorm", "-ln", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    parser.add_argument("--anchor_mode", "-am", type=str, default='rollout', help="anchor seeking mode")
    parser.add_argument("--closest_obs_sample_size", "-coss", type=int, default=500)
    parser.add_argument("--anchor_seeker_hidden_dims", "-ashd", type=int, nargs='*', default=[100, 100])
    parser.add_argument("--asp_layernorm", "-aln", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False)
    parser.add_argument("--load_anchor_seeker_path", "-lasp", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step_per_epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
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

    if args.load_dynamics_path:
        import json
        hyperparams_path = os.path.join(args.load_dynamics_path, "../record/hyper_param.json")
        with open(hyperparams_path, "r") as f:
            dynamics_hyperparams = json.load(f)
        assert args.task == dynamics_hyperparams["task"]
        args.dynamics_args = dynamics_hyperparams
        args.dynamics_hidden_dims=dynamics_hyperparams["dynamics_hidden_dims"]
        args.n_ensemble=dynamics_hyperparams["n_ensemble"]
        args.n_elites=dynamics_hyperparams["n_elites"]

    if args.load_anchor_seeker_path:
        import json
        hyperparams_path = os.path.join(args.load_anchor_seeker_path, "../record/hyper_param.json")
        with open(hyperparams_path, "r") as f:
            anchor_seeker_hyperparams = json.load(f)
        assert args.task == anchor_seeker_hyperparams["task"]
        args.anchor_seeker_hidden_dims=anchor_seeker_hyperparams["anchor_seeker_hidden_dims"] if "anchor_seeker_hidden_dims" in anchor_seeker_hyperparams else args.anchor_seeker_hidden_dims
        args.actor_horizon_len=anchor_seeker_hyperparams["actor_horizon_len"]
        args.critic_horizon_len=anchor_seeker_hyperparams["critic_horizon_len"]

    if not args.rollout_augmentation:
        assert args.real_ratio == 1.0,  "If 'rollout_augmentation' is False, 'real_ratio' must be 1"

    assert args.actor_horizon_len == args.critic_horizon_len, "actor_horizon_len and critic_horizon_len must be same"

    # create env and dataset
    if 'v3' in args.task: # neorl
        import neorl
        task, version, data_type = tuple(args.task.split("-"))
        env = neorl.make(task+'-'+version)
        dataset = load_neorl_dataset(env, data_type)
    else:
        env = gym.make(args.task)
        dataset = qlearning_dataset(env)

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    highs = env.action_space.high
    neg_lows = -env.action_space.low
    assert np.all(highs == highs[0]) and np.all(neg_lows == highs[0])
    args.max_action = env.action_space.high[0]

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
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    obs_mean_np, obs_std_np = np.zeros_like(dataset['observations'][0][None, :]), np.ones_like(dataset['observations'][0][None, :]) # dummy
    obs_mean, obs_std = torch.tensor(obs_mean_np, dtype=torch.float32, device=args.device), torch.tensor(obs_std_np, dtype=torch.float32, device=args.device)

    # create dynamics
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
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        args,
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=args.penalty_coef,
        uncertainty_mode=args.uncertainty_mode,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)
        dynamics.model.eval()
        dynamics.model.requires_grad_(False)

    # create policy model
    # backbone for fg
    actor_fg_input_dim = np.prod(args.obs_shape)
    actor_fg_output_dim = args.embedding_dim * args.tr_hidden_dims[-1] # hid_dim = 256

    critic_fg_input_dim = np.prod(args.obs_shape) + args.action_dim
    critic_fg_output_dim = args.embedding_dim * args.tr_hidden_dims[-1] # hid_dim = 256

    # backbone for SAC's
    action_backbone_input = args.tr_hidden_dims[-1]
    critic_backbone_input = args.tr_hidden_dims[-1]
    #actor_backbone_output = args.action_dim * 2
    dist_input = args.hidden_dims[-1]
    dist_output = args.action_dim # for mu and sigma each
    dist = TanhDiagGaussian(
        latent_dim=dist_input,
        output_dim=dist_output,
        unbounded=True,
        conditioned_sigma=True
    )
    # actor backbone
    actor_backbone = MLP(input_dim=action_backbone_input, hidden_dims=args.hidden_dims, layernorm=args.layernorm)

    # critic backbone
    critic1_backbone = MLP(input_dim=critic_backbone_input, hidden_dims=args.hidden_dims, layernorm=args.layernorm)
    critic2_backbone = MLP(input_dim=critic_backbone_input, hidden_dims=args.hidden_dims, layernorm=args.layernorm)

    # anchor seeker backbone
    actor_anchor_seeking_backbone = MLP(input_dim=np.prod(args.obs_shape) , hidden_dims=args.anchor_seeker_hidden_dims, layernorm=args.asp_layernorm)
    critic_anchor_seeking_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.anchor_seeker_hidden_dims, layernorm=args.asp_layernorm)

    # anchor seeker for ablation exp
    anchor_handler = AnchorHandler(args, args.device)
    anchor_handler.set_dataset(dataset['observations'])

    anchor_handler.set_obs_std(obs_std_np, obs_std)
    anchor_handler.set_obs_mean(obs_mean_np, obs_mean)
    anchor_handler.normalise_obs()

    # actor (anchor backbone's hidden dimension is same as args.hidden dim, if want to customize, then should be changed)
    actor_anchor_seeking_policy = AnchorActor(args, actor_anchor_seeking_backbone, owner = 'actor') if args.anchor_mode == 'rollout' else None
    actor = TransdActorProb(
        args,
        actor_anchor_seeking_policy,
        anchor_handler,
        dynamics,
        input_dim=actor_fg_input_dim,
        backbone=actor_backbone,
        dist=dist,
        tr_hidden_dims=args.tr_hidden_dims,
        fg_output_dim=actor_fg_output_dim,
        embedding_dim=args.embedding_dim,
        action_dim=args.action_dim,
        layernorm=args.asp_layernorm,
        unbounded=False,
        conditioned_sigma=True,
        horizon_length=args.actor_horizon_len,
        max_mu=args.max_action,
        device=args.device,
    )

    critic_anchor_seeking_policy = AnchorActor(args, critic_anchor_seeking_backbone, owner = 'critic') if args.anchor_mode == 'rollout' else None
    critic1 = TransdCritic(
        args,
        critic_anchor_seeking_policy,
        anchor_handler,
        dynamics=dynamics,
        backbone=critic1_backbone,
        input_dim=critic_fg_input_dim,
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        fg_output_dim=critic_fg_output_dim,
        embedding_dim=args.embedding_dim,
        horizon_length=args.critic_horizon_len,
        tr_hidden_dims=args.tr_hidden_dims,
        layernorm=args.asp_layernorm,
        device=args.device,
        )
    critic2 = TransdCritic(
        args,
        critic_anchor_seeking_policy,
        anchor_handler,
        dynamics=dynamics,
        backbone=critic2_backbone,
        input_dim=critic_fg_input_dim,
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        fg_output_dim=critic_fg_output_dim,
        embedding_dim=args.embedding_dim,
        horizon_length=args.critic_horizon_len,
        tr_hidden_dims=args.tr_hidden_dims,
        layernorm=args.asp_layernorm,
        device=args.device,
        )

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = MOPOPolicy(
        args,
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        device=args.device,
    ).to(args.device)

    # anchor seeking policy
    if args.load_anchor_seeker_path:
        policy.actor.anchor_seeking_policy.pretrain_load(args.load_anchor_seeker_path)
        policy.actor.anchor_seeking_policy.freeze()

        for obj in [policy.critic1, policy.critic2, policy.critic1_old, policy.critic2_old]:
            obj.anchor_seeking_policy.pretrain_load(args.load_anchor_seeker_path)
            obj.anchor_seeking_policy.freeze()

    # log
    log_dirs = make_log_dirs(
        args.task, args.algo_name, args.seed, vars(args),
        record_params=["penalty_coef", "rollout_length"],
        timestamp=timestamp,
        init_run=True,
    )

    # logs
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    if args.load_dynamics_path:
        del args.dynamics_args
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MBPolicyTrainer(
        args=args,
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,

    )

    # train
    if not args.load_dynamics_path:
        dynamics.train(
            real_buffer.sample_all(),
            logger,
        )
    else:
        ckpt_files = glob(os.path.join(os.path.dirname(logger.checkpoint_dir), 'checkpoint_last', "policy_*.pth"))

        if len(ckpt_files)>0:
            logger.log('Loading policy...')
            ckpt_files_sorted = sorted(ckpt_files, key=lambda x: int(re.search(r'policy_(\d+).pth$', x).group(1)))[0] # only small epoch
            policy.load(ckpt_files_sorted, logger, lr_scheduler, policy_trainer)

        # train policy
        policy_trainer.train()

    return log_dirs

if __name__ == "__main__":
    log_dirs = train(get_args())
