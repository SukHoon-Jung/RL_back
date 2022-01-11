import gym
from gym.wrappers import FlattenObservation

from runner.callbacks import LearnEndCallback
from sim.env.Dict_envtest import DictEnvTest
import numpy as np
from stable_baselines3 import DDPG, SAC,TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def consistency(model_class=TD3):
    """
    Make sure that dict obs with vector only vs using flatten obs is equivalent.
    This ensures notable that the network architectures are the same.
    """
    dict_env = DictEnvTest()
    env = FlattenObservation(DictEnvTest())

    dict_env =VecNormalize(DummyVecEnv([lambda : DictEnvTest()]), norm_obs_keys=["obs"])
    env =VecNormalize(DummyVecEnv([lambda :FlattenObservation(DictEnvTest())]))

    dict_env.seed(10)
    n_steps = 5
    kwargs = dict(
        buffer_size=20000,
        gradient_steps=2,
        batch_size =128
    )
    CB1 = LearnEndCallback()
    CB2 = LearnEndCallback()
    dict_model = model_class("MultiInputPolicy", dict_env, gamma=0.99, seed=1,  **kwargs)
    dict_model.learn(total_timesteps=n_steps, callback = CB1,)



    normal_model = model_class("MlpPolicy", env, gamma=0.99, seed=1, **kwargs)
    normal_model.learn(total_timesteps=n_steps, callback = CB2,)

    print(CB1.last_aloss, CB2.last_aloss)
    print(CB1.last_closs, CB2.last_closs)



    obs = dict_env.reset()
    for i in range(100):
        action_1, _ = dict_model.predict(obs)
        action_2, _ = normal_model.predict(obs["obs"])
        print(action_1, "\t", action_2)
        obs, reward, done, info  = dict_env.step(action_1)



if __name__ == '__main__':

    consistency()
