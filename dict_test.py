import gym

from sim.env.Dict_env import DictEnv
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def run():

    env = DummyVecEnv([lambda :DictEnv()])
    env = VecNormalize(env, norm_obs_keys=["obs"])

    kwargs = dict(
        policy_kwargs=dict(
            net_arch=[32],
            features_extractor_kwargs=dict(cnn_output_dim=32),
        ),
        gradient_steps=2,
    )


    model = TD3("MultiInputPolicy"  , env, seed=1, **kwargs)

    model.learn(total_timesteps=7000)

if __name__ == '__main__':
    run()