import os
import shutil
import time

import gym
from torch.utils.tensorboard import SummaryWriter

from runner.callbacks import LearnEndCallback
from sim.env import StackedEnv
from sim.env.Dict_envtest import DictEnvTest
from sim.env.Dict_envtest2 import DictEnvTest2
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
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



ENV =StackedEnv
class IterRun:

    unit = 2050
    boost_step = 3* unit
    boosted = False

    gradient_steps = 2
    def __init__(self, MODEL, env_name ='Stacked-v0', arc=[256, 128, 32], nproc=1, init_boost=False, retrain=False, batch_size=128):
        self.env_name = env_name
        self.model_cls = MODEL
        self.name = MODEL.__name__
        self.test_env = DummyVecEnv([lambda: ENV(title=self.name, plot_dir="./sFig/{}".format(self.name))])
        self.env = self.make_env(nproc)
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
        else:
            if init_boost: self.init_boost()
            else:
                model = self._create()
                print(model.policy)
                model.save(self.save)
                del model

    def make_env(self, nproc=1):
        if nproc <2:
            env =  DummyVecEnv([lambda: ENV()])
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
            model = self._create(env=env)
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

    def _create(self, env=None, learning_starts = 100 ):
        policy_kwargs = dict(net_arch=self.arch)
        noise_std = 0.3
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=noise_std * np.ones(1)
        )
        if env is None:env = self.env
        self.seed = 12020948 #np.random.randint (1e8)
        model = self.model_cls("MlpPolicy", env, verbose=1, action_noise=noise, seed = self.seed,
                               gradient_steps= self.gradient_steps *self.nproc,
                               batch_size = self.batch_size, policy_kwargs=policy_kwargs,
                               learning_starts=learning_starts)
        return model



    def train_eval(self, steps =None):
        self.time_recoder.start()
        if steps is None: steps = self.unit*1
        self.seed = np.random.randint(1e8)
        model = self.model_cls.load(self.save, env=self.env)
        model.set_random_seed (self.seed)
        print(self.seed)


        if self.buffer: model.replay_buffer = self.buffer
        print("LOADED", self.save, self.iter, "SEED ", model.seed)
        print("BUFFER REUSE:", model.replay_buffer.size() * self.nproc)


        CB = LearnEndCallback()
        model.learn(total_timesteps=steps, tb_log_name=self.name, callback=CB)
        self.buffer = model.replay_buffer
        print(type(self.buffer))

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

    def board(self, prefix, dict):
        for key, val in dict.items():
            self.writer.add_scalar(prefix + "/" + key, val, self.iter)

    def evaluation(self, model, env = None):
        if not env: env = self.test_env
        train_evn = model.get_env()
        obs = env.reset()
        obs = train_evn.normalize_obs(obs)

        # print(train_evn.obs_rms.mean)
        # print(train_evn.obs_rms.var)
        # print(obs)

        # print(train_evn.obs_rms['obs'].mean)
        # print(train_evn.obs_rms['obs'].var)
        # print(obs)

        done = False
        rewards = []
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            obs = train_evn.normalize_obs (obs)
            done = done[0]
            rewards.append(reward[0])

        info = info[0]

        rslt = {
            "1_Reward": sum(rewards),
            "2_Profit": info['profit'],
            "3_Risk": info['risk'],
            "4_Neg": info['neg'],
            "5_Trade": info['cnt']
        }

        return rslt
