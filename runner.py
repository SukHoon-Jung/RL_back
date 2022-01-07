import os
import shutil
import time

import gym
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import numpy as np


# <class 'stable_baselines3.td3.policies.TD3Policy'>
# <class 'stable_baselines3.sac.policies.SACPolicy'>
class TimeRecode:
    def __init__(self, writer, interval=2):
        self.total_sec=0
        self.tick = 0
        self.writer = writer
        self.interval=interval

    def start(self):
        self.start_tm = time.time()

    def recode(self, dict):
        if dict is None: return False
        self.total_sec += int(time.time()-self.start_tm)
        now = (self.total_sec/60) / self.interval

        if now >= self.tick:
            for key, val in dict.items():
                self.writer.add_scalar("TvIME" + "/" + key, val, now * self.interval)
            self.tick = now + 1

        return True




class IterRun:

    unit = 2050
    boost_step = 3* unit
    boosted = False

    gradient_steps = 4
    def __init__(self, MODEL, env_name, arc=[256, 128, 32], nproc=1):
        self.env_name = env_name
        self.model_cls = MODEL
        self.name = MODEL.__name__
        self.test_env = DummyVecEnv([lambda: gym.make(env_name, title=self.name, plot_dir="./sFig/{}".format(self.name))])
        self.env = self.make_env(nproc)
        self.writer = self.tensorboard("./summary_all/{}/".format(self.name))
        self.save = f"ckpt_{self.name}"
        self.buffer = None
        self.arch = arc
        self.iter = 1
        self.train_start = time.time ()
        self.time_recoder = TimeRecode(self.writer)
        self.nproc =nproc

    def make_env(self, nproc=1):
        if nproc <2:
            env =  DummyVecEnv([lambda: gym.make(self.env_name)])
        else:
            env = make_vec_env (self.env_name, n_envs=nproc, vec_env_cls=SubprocVecEnv)
        return VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)




    def init_boost(self, min_profit=-500):
        print("-----  BOOST UP", self.name)

        env = self.make_env(self.nproc)
        test_env =  DummyVecEnv([lambda: gym.make(self.env_name)])

        minimum = -1e8
        suit_model =None
        for iter in range(1,3):
            model = self._create(init=True, env=env)
            self.train_start = time.time ()
            model.learn(total_timesteps= self.boost_step + iter*self.unit)
            eval = self.evaluation(model, test_env)
            profit = eval["1_Reward"]

            print(" - - - - - BOOST PROFIT: ", self.name, profit)
            if (minimum < profit):
                minimum = profit
                suit_model = model
            if profit > min_profit: break

        self.buffer = suit_model.replay_buffer
        suit_model.learning_starts = 0
        suit_model.save(self.save)
        self.boosted = True
        del suit_model


    def tensorboard(self, dir):

        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)
        return SummaryWriter(dir)

    def _create(self, init=False, env=None):
        policy_kwargs = dict(net_arch=self.arch)
        noise_std = 0.3
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=noise_std * np.ones(1)
        )

        if init:
            tensorboard_log =None
            learning_starts =self.boost_step
        else:
            env = self.env
            tensorboard_log = "./summary/"
            learning_starts = 100

        model = self.model_cls("MlpPolicy", env, verbose=1, action_noise=noise, gradient_steps= self.gradient_steps,
                      policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_starts=learning_starts)
        return model

    def train_eval(self, steps =None):
        self.time_recoder.start()
        if steps is None: steps = self.unit*3
        try:
            model = self.model_cls.load(self.save, env=self.env)
            if self.buffer: model.replay_buffer = self.buffer
            print("LOADED", self.save, self.iter)
            print("BUFFER REUSE:", model.replay_buffer.size())
        except:
            if self.boosted: raise Exception("INIT ERROR")
            model = self._create()

        model.learn(total_timesteps=steps, tb_log_name=self.name)
        self.buffer = model.replay_buffer
        fps = int(model._n_updates / (time.time()-self.train_start))
        model.save(self.save)


        train = {
            "1_Actor_loss": model.last_log["train/actor_loss"],
            "2_Critic_Loss": model.last_log["train/critic_loss"],
            "3_Reward": model.last_log["rollout/ep_rew_mean"],
            "4_FPS": fps}

        print("===========   EVAL   =======   ",self.name, self.iter, ",FPS: ", fps)
        eval = self.evaluation(model, self.test_env)
        self.board("Eval", eval)
        self.board("Train",train)
        self.time_recoder.recode(eval)
        del model

        self.iter += 1

    def board(self, prefix, dict):
        for key, val in dict.items():
            self.writer.add_scalar(prefix + "/" + key, val, self.iter)

    def evaluation(self, model, env = None):
        if not env: env = self.test_env
        train_evn = model.get_env()
        obs = env.reset()
        obs = train_evn.normalize_obs(obs)
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            obs = train_evn.normalize_obs (obs)
            done = done[0]
            rewards.append(reward[0])

        info = info[0]
        env.training = True
        env.norm_reward = True
        rslt = {
            "1_Reward": sum(rewards),
            "2_Profit": info['profit'],
            "3_Risk": info['risk'],
            "4_Neg": info['neg'],
            "5_Trade": info['cnt']
        }

        return rslt
