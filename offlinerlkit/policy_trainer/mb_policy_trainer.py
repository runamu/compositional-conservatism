import time
import os
from collections import defaultdict
import numpy as np
import torch
import gym
from typing import Optional, Dict, List, Tuple, Union
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.util_fns import get_normalized_std_score, get_normalized_score_neorl, get_normalized_std_score_neorl
from offlinerlkit.policy import MOPOPolicy, MOBILEPolicy
import wandb
import random

# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        args,
        policy: Union[MOPOPolicy, MOBILEPolicy],
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        reverse_policy = None,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
    ) -> None:

        self.args = args
        self.policy = policy
        if args.track:
            wandb.config.update({"param_size": sum(p.numel() for p in self.policy.parameters() if p.requires_grad)}) if args.track else None

        self.eval_env = eval_env
        self.eval_env.reset()
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger
        self.reverse_policy = reverse_policy
        self.track = args.track
        self.task = args.task
        self.all_eval_goals, self.eval_goals, self.num_timesteps = None, None, None

        self._rollout_freq, self._rollout_batch_size, self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self.renewed_run = True
        self._start_epoch = 1
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

    def train(
        self,
    ) -> Dict[str, float]:
        start_time = time.time()

        # reverse_dynamics, dynamics freeze
        self.policy.dynamics.model.eval()
        self.policy.dynamics.model.requires_grad_(False)

        self.num_timesteps = 1000 * (self._start_epoch - 1)

        if self._start_epoch == 1:
            self.last_10_performance = deque(maxlen=10)
        checkpoint_last = os.path.join(os.path.dirname(self.logger.checkpoint_dir), "checkpoint_last")

        self.fake_buffer.load_replay_buffer(self.logger._result_dir, self._start_epoch-1, self.logger) if self.renewed_run and self._start_epoch != 1 else None # renewed_run: load the replay buffer from the previous epoch

        # train loop
        for e in range(self._start_epoch, self._epoch + 1):
            train_start_time = time.time()
            self.policy.train()
            for it in range(self._step_per_epoch):
                if self.args.rollout_augmentation and self._rollout_length > 0:
                    if self.num_timesteps % self._rollout_freq == 0:
                        init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"]
                        rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                        self.fake_buffer.add_batch(**rollout_transitions)

                        self.logger.log(
                            "num rollout transitions: {}, reward mean: {:.4f}, reward std: {:.4f}".\
                                format(rollout_info["num_transitions"], rollout_info["reward_mean"], rollout_info["reward_std"])
                        )
                        for _key, _value in rollout_info.items():
                            self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                if self._rollout_length > 0:
                    fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                    batch = {"real": real_batch, "fake": fake_batch}
                else:
                    batch = {"real": real_batch}
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
                self.fake_buffer.save_replay_buffer(self.logger._result_dir, e, self.logger)
                self.save_checkpoints(checkpoint_last, e)
                self.fake_buffer.delete_replay_buffer(self.logger._result_dir, e, self.logger)

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
