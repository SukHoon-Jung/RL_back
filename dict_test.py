import gym
from gym.wrappers import FlattenObservation

from runner.callbacks import LearnEndCallback
from sim.env.Dict_env import DictEnv
from sim.env.Dict_envtest import DictEnvTest
import numpy as np
from stable_baselines3 import DDPG, SAC,TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def consistency(model_class=TD3):
    """
    Make sure that dict obs with vector only vs using flatten obs is equivalent.
    This ensures notable that the network architectures are the same.
    """

    dict_env =VecNormalize(DummyVecEnv([lambda : DictEnv()]), norm_obs_keys=["stat", "obs"])
    env = VecNormalize (DummyVecEnv ([lambda: FlattenObservation (DictEnv ())]))
    env222 = VecNormalize(DummyVecEnv([lambda :FlattenObservation(DictEnvTest())]))
    dict_env.seed(10)
    env.seed (10)
    env222.seed (10)


    n_steps = 2
    kwargs = dict(
        buffer_size=20000,
        gradient_steps=2,
        batch_size =1
    )
    CB1 = LearnEndCallback()
    CB2 = LearnEndCallback()
    CB3 = LearnEndCallback ()


    dict_model = model_class("MultiInputPolicy", dict_env, gamma=0.99, seed=1,  **kwargs)
    # dict_model.learn(total_timesteps=n_steps, callback = CB1,)

    normal_model = model_class("MlpPolicy", env, gamma=0.99, seed=1, **kwargs)
    normal_model.learn(total_timesteps=n_steps, callback = CB2,)

    normal_model2 = model_class ("MlpPolicy", env222, gamma=0.99, seed=1, **kwargs)
    normal_model2.learn(total_timesteps=n_steps, callback=CB3, )




    print(CB1.last_aloss, CB1.last_closs)
    print (CB2.last_aloss, CB2.last_closs)
    print (CB3.last_aloss, CB3.last_closs)

    print (env.obs_rms.mean-env222.obs_rms.mean)

    obs = dict_env.reset()
    o2 = env.reset ()
    o3 = env222.reset ()
    # #
    # print(obs)
    # print (o2)
    # print (o3)
    # print("==========")
    # print(dict_env.unnormalize_obs(obs))
    # print (env.unnormalize_obs (o2))
    # print (env222.unnormalize_obs (o3))
    # print("==========")
    #

    for i in range(5):
        action_1, _ = dict_model.predict(obs)
        obs2 = obs["obs"].tolist()[0] + obs['stat'].tolist()[0]
        action_2, _ = normal_model.predict(obs2)
        action_22, _ = normal_model2.predict (obs2)
        print(action_1, "\t", action_2, "\t", action_22)
        obs, reward, done, info  = dict_env.step(action_1)



if __name__ == '__main__':

    consistency()
