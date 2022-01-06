import os
import shutil

import gym
from gym import register
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def evaluation(model, env):
    env.training=False
    env.norm_reward =False
    obs = env.reset()
    done = False
    rewards=[]
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        done = done[0]
        rewards.append(reward[0])

    info=info[0]
    env.training=True
    env.norm_reward=True
    return  sum(rewards), info['profit'], info['risk'], info['neg'], info['cnt']



def train_eval(MODEL, env, ckpt, buffer, writer, idx):
    name = MODEL.__name__
    env.reset()
    try:
        model = MODEL.load(ckpt, env = env)
        if buffer: model.replay_buffer = buffer
        print("LOADED", ckpt)

    except:
        policy_kwargs = [256, 128, 32]
        policy_kwargs = dict(net_arch=policy_kwargs)
        noise_std = 0.3
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=noise_std * np.ones(1)
        )
        tensorboard_log = "./summary/"
        model = MODEL("MlpPolicy", env, verbose=1, action_noise=noise, gradient_steps=2,
                      policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log)
        print(model.policy)



    model.learn(total_timesteps=7500, tb_log_name = name)

    al = model.last_log["train/actor_loss"]
    cl = model.last_log["train/critic_loss"]
    t_rw = model.last_log["rollout/ep_rew_mean"]
    fps = model.last_log["time/fps"]

    rwd, prf, risk, neg, cnt = evaluation(model, env)
    ckpt = "ckpt_"+name
    model.save(ckpt)
    buffer = model.replay_buffer

    del model


    writer.add_scalar("1/1reward", rwd, idx)
    writer.add_scalar("1/3profit", prf, idx)
    writer.add_scalar("1/2risk", risk, idx)
    writer.add_scalar("1/4neg", neg, idx)
    writer.add_scalar("1/5count", cnt, idx)

    writer.add_scalar("2/actor_loss", al, idx)
    writer.add_scalar("2/critic_loss", cl, idx)
    writer.add_scalar("2/t_reward", t_rw, idx)
    writer.add_scalar("2/fps", fps, idx)


    return None, None, rwd, prf, ckpt, buffer

def tensorboard(dir):

    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)
    return SummaryWriter(dir)


def compair_run(iter):
    sac = "SAC"
    ddpg = "DDPG"
    sac_env = DummyVecEnv([lambda :gym.make(env_name, title=sac, plot_dir="./LOG_{}/figs".format(sac))])
    dd_env = DummyVecEnv([lambda :gym.make(env_name, title=ddpg, plot_dir="./LOG_{}/figs".format(ddpg))])
    sac_env = VecNormalize(sac_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    dd_env = VecNormalize (dd_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print(DDPG.__name__)
    sum_sac = tensorboard("./summary_all/SAC/")
    sum_ddpg = tensorboard("./summary_all/DDPG/")

    ckpt_ddpg = "ckpt_DDPG"
    ckpt_sac = "dkpt_SAC"
    buffer_ddpg = None
    buffer_sac = None
    for idx in range(iter):
        al, cl, rwd, profit, ckpt_ddpg, buffer_ddpg = train_eval(DDPG, dd_env, ckpt_ddpg, buffer_ddpg, sum_ddpg, idx)
        al, cl, rwd, profit, ckpt_sac, buffer_sac = train_eval(SAC, sac_env, ckpt_sac, buffer_sac, sum_sac, idx)

# env_name ='StarTrader-v0'
# entry = 'sim.env:StarTradingEnv'
env_name ='Stacked-v0'
entry = 'sim.env:StackedEnv'


if __name__ == '__main__':
    register(
        id=env_name,
        entry_point=entry,
    )

    compair_run(1000)

