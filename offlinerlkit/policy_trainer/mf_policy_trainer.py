import time
import os
import numpy as np
import torch
import gym
import copy
from typing import Optional, Dict, List, Union
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.util_fns import get_normalized_std_score, print_grad_norm, get_normalized_score_neorl, get_normalized_std_score_neorl
from offlinerlkit.policy import CQLPolicy, IQLPolicy
import wandb
import random
from offlinerlkit.utils.set_state_fns import get_set_state_fn

# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        args,
        policy: Union[CQLPolicy, IQLPolicy],
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.args = args
        self.policy = policy
        if args.track:
            wandb.config.update({"policy_parameters": sum(p.numel() for p in self.policy.parameters() if p.requires_grad)})

        self.eval_env = eval_env
        _ = self.eval_env.reset()
        self.buffer = buffer
        self.logger = logger
        self.track = args.track
        self.task = args.task
        self.all_eval_goals = None
        self.eval_goals = None

        self._start_epoch = 1
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.num_timesteps = None
        self.set_state_fn = get_set_state_fn(task=self.args.task)

    def train(
        self,
    ) -> Dict[str, float]:
        start_time = time.time()

        self.num_timesteps = 1000 * (self._start_epoch - 1)
        if self._start_epoch == 1:
            self.last_10_performance = deque(maxlen=10)

        self.best_epoch, self.best_last10_epoch, self.best_metric, self.best_last10_metric = None, None, None, None
        checkpoint_last = os.path.join(os.path.dirname(self.logger.checkpoint_dir), "checkpoint_last")

        # train loop
        for e in range(self._start_epoch, self._epoch + 1):
            train_start_time = time.time()
            self.policy.train()
            for it in range(self._step_per_epoch):
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                self.num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            train_time = time.time() -  train_start_time
            self.logger.logkv("time/train_one_epoch", train_time)
            self.evaluate_current_policy_mujoco()
            if e % 50 == 0:
                self.save_checkpoints(checkpoint_last, e)

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self.logger.close()

    def save_checkpoints(self, checkpoint_last, epoch):
        # save random state
        random_states = {}
        random_states["random"] = random.getstate()
        random_states["np"] = np.random.get_state() # dictionary
        random_states["torch"] = torch.get_rng_state() # Tensor
        random_states["torch_cuda"] = torch.cuda.get_rng_state_all() # List[Tensor]

        # save checkpoint
        self.policy.save(checkpoint_last, random_states=random_states, epoch=epoch, logger=self.logger, lr_scheduler=self.lr_scheduler, last_10_performance=self.last_10_performance)

    def evaluate_current_policy_mujoco(self):
        # d4rl benchmark
        norm_ep_rew_mean = self._evaluate_and_log()
        self.last_10_performance.append(norm_ep_rew_mean)
        self.logger.logkv(f"eval/last_10_performance", np.mean(self.last_10_performance))
        self.logger.dumpkvs()

    def _evaluate_and_log(self):
        eval_start_time = time.time()
        eval_info = self._evaluate()
        prefix = "eval"

        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        if hasattr(self.eval_env, "get_normalized_score"):
            mean_normalize_fn = lambda x: self.eval_env.get_normalized_score(x) * 100 # scaled
            std_normalize_fn = lambda x: get_normalized_std_score(self.eval_env, x)
        else:
            mean_normalize_fn = lambda x: get_normalized_score_neorl(self.task, x)
            std_normalize_fn = lambda x: get_normalized_std_score_neorl(self.task, x)
        norm_ep_rew_mean = mean_normalize_fn(ep_reward_mean)
        norm_ep_rew_std = std_normalize_fn(ep_reward_std)

        self.logger.logkv(f"{prefix}/normalized_episode_reward", norm_ep_rew_mean)
        self.logger.logkv(f"{prefix}/normalized_episode_reward_std", norm_ep_rew_std)
        self.logger.logkv(f"{prefix}/episode_length", ep_length_mean)
        self.logger.logkv(f"{prefix}/episode_length_std", ep_length_std)

        self.logger.set_timestep(self.num_timesteps)
        eval_time = time.time() - eval_start_time
        self.logger.logkv("time/eval_one_epoch", eval_time)

        return norm_ep_rew_mean

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1, -1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs
            if terminal or episode_length >= self.eval_env._max_episode_steps:
                eval_ep_info_buffer.append({
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                })
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
        }
