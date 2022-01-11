import os
import shutil
import time

import gym
from torch.utils.tensorboard import SummaryWriter

from runner.callbacks import LearnEndCallback
from sim.env.Dict_env import DictEnv
from sim.env.Dict_env2 import DictEnv2
from sim.env.Dict_envtest import DictEnvTest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import numpy as np


# <class 'stable_baselines3.td3.policies.TD3Policy'>
# <class 'stable_baselines3.sac.policies.SACPolicy'>


ENV = DictEnv2
class IterRun:
    MIN_TRADE = 30
    BOOST_SEARCH = 3
    unit = 2050
    gradient_steps = 2
    def __init__(self, MODEL, arc=[256, 128, 32], nproc=1, retrain=False, batch_size=128, seed=None):
        self.seed = seed
        self.model_cls = MODEL
        self.name = MODEL.__name__
        self.test_env =  ENV(title=self.name, verbose=True, plot_dir="./sFig/{}".format(self.name))
        self.env = self.make_env()
        self.writer = self.tensorboard("./summary_all/{}/".format(self.name))
        self.save = f"ckpt_{self.name}"
        self.buffer = None
        self.arch = arc
        self.iter = 1


        self.time_recoder = TimeRecode(self.writer)
        self.nproc =nproc
        self.batch_size = batch_size
        if retrain:
            pass
        elif self.seed is None: self.init_boost (self.MIN_TRADE)
        else : self.set_same()

    def make_env(self):
        env = DummyVecEnv([lambda: ENV(verbose=False)])
        return VecNormalize(env, norm_obs_keys=["obs", "stat"])

    def set_same(self):
        model = self.unit_model()
        self.buffer = model.replay_buffer
        model.save(self.save)
        print("-----  CREATE", self.name, " SEED ", self.seed)

    def unit_model(self):
        env = self.make_env()
        model = self._create(env=env, learning_starts=self.unit)
        self.train_start = time.time()
        learn_steps = self.unit * 2
        model.learn(total_timesteps=learn_steps)
        model.learning_starts = 0
        return model

    def init_boost(self, MIN_TRADE, min_reward=-1000):
        print("-----  BOOST UP", self.name)

        test_env = ENV(verbose=False)
        minimum = -1e8
        suit_model = None
        bad_model = None
        max_cont = -1e8

        for iter in range(self.BOOST_SEARCH):
            model = self.unit_model()
            eval = self.evaluation(model, test_env)
            reward = eval["1_Reward"]
            count = eval['4_Trade']
            if (max_cont < count):
                max_cont = count
                bad_model = model
            if count < MIN_TRADE:
                print (" - - - - - BOOST FAIL: ", self.name, reward, " by Count:", count)
                continue
            print(" - - - - - BOOST PROFIT: ", self.name, reward)
            if (minimum < reward):
                minimum = reward
                suit_model = model
            if reward > min_reward: break
        if suit_model is None:
            suit_model = bad_model
            print(" - - - - - BOOST Selection Failed: ", self.name, "Bad Model Count", max_cont)

        print (" - - - - - BOOST Selected: ", self.name, minimum, "Seed:", model.seed)
        self.seed = model.seed
        self.buffer = suit_model.replay_buffer
        suit_model.save(self.save)

        del suit_model


    def _create(self, env=None, learning_starts = 100):
        policy_kwargs = dict(net_arch=self.arch)
        noise_std = 0.3
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=noise_std * np.ones(1)
        )
        if env is None:env = self.env
        seed = self.seed or np.random.randint(1e8)
        model = self.model_cls("MultiInputPolicy", env, verbose=1, action_noise=noise, seed =seed,
                               gradient_steps= self.gradient_steps,
                               batch_size = self.batch_size, policy_kwargs=policy_kwargs,
                               learning_starts=learning_starts)
        return model



    def train_eval(self, steps =None):

        self.time_recoder.start()
        self.seed = np.random.randint (1e8)
        steps = steps or self.unit*1
        model = self.model_cls.load(self.save, env=self.env)
        model.replay_buffer = self.buffer
        model.set_random_seed (self.seed)
        print (self.seed)

        print("LOADED", self.save, self.iter, model.seed)
        print("BUFFER REUSE:", model.replay_buffer.size() * self.nproc)

        CB = LearnEndCallback()
        model.learn(total_timesteps=steps, tb_log_name=self.name, callback=CB)
        self.buffer = model.replay_buffer

        print("===========   EVAL   =======   ", self.name, self.iter, ",FPS: ", CB.fps)

        train = {
            "1_Actor_loss": CB.last_aloss,
            "2_Critic_Loss": CB.last_closs,
            "3_FPS": CB.fps}

        eval = self.evaluation(model, self.test_env)
        self.board("Eval", eval)
        self.board("Train",train)
        self.time_recoder.recode(eval)
        model.save(self.save)
        del model

        self.iter += 1


    def tensorboard(self, dir):
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir, exist_ok=True)
        return SummaryWriter(dir)

    def board(self, prefix, dict):
        for key, val in dict.items():
            self.writer.add_scalar(prefix + "/" + key, val, self.iter)

    def evaluation(self, model, env = None):
        if not env: env = self.test_env

        obs = env.reset()
        done = False
        rewards = []
        while not done:
            obs = self.env.normalize_obs(obs)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            done = done
            rewards.append(reward)

        info = info

        rslt = {
            "1_Reward": sum(rewards),
            "2_Profit": info['profit'],
            "3_Risk": info['risk'],
            "4_Trade": info['cnt'],
            "5_Perform": info['profit']/info['cnt'],
        }

        return rslt



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

