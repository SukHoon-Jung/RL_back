import gym
from gym import register
import numpy as np

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env(env_id, nproc=2):
    def _f(env_name, seed):
        env = gym.make(env_name)
        env.seed(seed)
        return env

    envs = [_f(env_id,seed) for seed in range(nproc)]
    return SubprocVecEnv(envs)

env_name ='StarTrader-v0'
def run(model_name):

    env = gym.make(env_name, title=model_name, plot_dir="./LOG_{}/figs".format(model_name))
    # env = make_env(env_name)
    # env= make_vec_env(model_name, n_envs=2, seed=0, vec_env_cls=SubprocVecEnv)

    policy_kwargs = [256, 256]
    policy_kwargs = dict(net_arch=policy_kwargs)
    noise_std = 0.3
    noise = NormalActionNoise(
        mean=np.zeros(1), sigma=noise_std * np.ones(1)
    )
    tensorboard_log="./summary/"

    if model_name =="DDPG":
        model = DDPG("MlpPolicy", env, verbose=1, action_noise=noise ,gradient_steps=2,
                     policy_kwargs = policy_kwargs, tensorboard_log=tensorboard_log)
    else:
        model = SAC("MlpPolicy", env, verbose=1, action_noise=noise ,
                     policy_kwargs = policy_kwargs, tensorboard_log=tensorboard_log)

    model.learn(total_timesteps=1e8, tb_log_name = model_name)





if __name__ == '__main__':
    register(
        id=env_name,
        entry_point='sim.env:StarTradingEnv',
    )
    run("DDPG")
