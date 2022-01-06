import os
import shutil
import time

import gym
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

env_name = 'Stacked-v0'
entry = 'sim.env:StackedEnv'

class TimeRecode:
    def __init__(self, writer, interval=10):
        self.start = time.time()
        self.tick = 1
        self.time_interval = interval
        self.writer = writer
    def recode(self, dict):
        if dict is None: return False

        now = int((time.time()-self.start)/60) / self.time_interval
        if now >= self.tick :
            for key, val in dict.items():
                self.writer.add_scalar("TvIME" + "/" + key, val, now * self.time_interval)

            self.tick = now + 1
            return True

        return False




class IterRun:

    unit = 2050
    boost_step = 3* unit
    boosted = False

    gradient_steps = 2
    def __init__(self, MODEL, arc=[256, 128, 32]):
        self.model_cls = MODEL
        self.name = MODEL.__name__
        env = DummyVecEnv([lambda: gym.make(env_name, title=self.name, plot_dir="./sFig/{}".format(self.name))])
        self.env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        self.writer = self.tensorboard("./summary_all/{}/".format(self.name))
        self.save = f"ckpt_{self.name}"
        self.buffer = None
        self.arch = arc
        self.iter = 1
        self.time_recoder = TimeRecode(self.writer)


    def init_boost(self, min_profit=-500):
        print("-----  BOOST UP", self.name)

        env = DummyVecEnv([lambda: gym.make(env_name)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        minimum = -1e8
        suit_model =None
        for iter in range(1,3):
            model = self._create(init=True, env=env)
            model.learn(total_timesteps= self.boost_step + iter*self.unit)
            eval = self.evaluation(model, env)
            profit = eval["1_Reward"]
            print(" - - - - - BOOST PROFIT:   ",profit)
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
        if steps is None: steps = self.unit*3



        try:
            model = self.model_cls.load(self.save, env=self.env)
            if self.buffer: model.replay_buffer = self.buffer
            print("LOADED", self.save)
            print("BUFFER REUSE:", model.replay_buffer.size())
        except:
            if self.boosted: raise Exception("INIT ERROR")
            model = self._create()

        model.learn(total_timesteps=steps, tb_log_name=self.name)
        self.buffer = model.replay_buffer


        model.save(self.save)

        train = {
            "1_Actor_loss": model.last_log["train/actor_loss"],
            "2_Critic_Loss": model.last_log["train/critic_loss"],
            "3_Reward": model.last_log["rollout/ep_rew_mean"],
            "4_FPS": model.last_log["time/fps"]}

        eval = self.evaluation(model, self.env)
        self.board("Eval", eval)
        self.board("Train",train)
        self.time_recoder.recode(eval)
        del model

        self.iter += 1

    def board(self, prefix, dict):
        for key, val in dict.items():
            self.writer.add_scalar(prefix + "/" + key, val, self.iter)

    def evaluation(self, model, env):
        env.training = False
        env.norm_reward = False
        obs = env.reset()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
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
