import os
import shutil

import gym
from gym import register
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise

env_name ='StarTrader-v0'



def evaluation(model, env):
    obs = env.reset()
    done = False
    rewards=[]
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

    return  sum(rewards), info['profit'], info['risk'], info['neg']



def train_eval(MODEL, env, ckpt, buffer, writer, idx):
    name = MODEL.__name__
    env.reset()
    if not ckpt:
        policy_kwargs = [256, 256]
        policy_kwargs = dict(net_arch=policy_kwargs)
        noise_std = 0.3
        noise = NormalActionNoise(
            mean=np.zeros(1), sigma=noise_std * np.ones(1)
        )
        tensorboard_log="./summary/"
        model = MODEL("MlpPolicy", env, verbose=1, action_noise=noise,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log)

    if ckpt:
        model = MODEL.load(ckpt, env = env)
        if buffer: model.replay_buffer = buffer
        print("LOADED", ckpt)


    model.learn(total_timesteps=7500, tb_log_name = name)

    al = model.last_log["train/actor_loss"]
    cl = model.last_log["train/critic_loss"]
    t_rw = model.last_log["rollout/ep_rew_mean"]
    fps = model.last_log["time/fps"]

    rwd, prf, risk, neg = evaluation(model, env)
    ckpt = "ckpt_"+name
    model.save(ckpt)
    buffer = model.replay_buffer

    del model


    writer.add_scalar("1/1reward", rwd, idx)
    writer.add_scalar("1/3profit", prf, idx)
    writer.add_scalar("1/2risk", risk, idx)
    writer.add_scalar("1/4neg", neg, idx)
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
    sac_env = gym.make(env_name, title=sac, plot_dir="./LOG_{}/figs".format(sac))
    dd_env = gym.make(env_name, title=ddpg, plot_dir="./LOG_{}/figs".format(ddpg))

    print(DDPG.__name__)
    sum_sac = tensorboard("./summary_all/SAC/")
    sum_ddpg = tensorboard("./summary_all/DDPG/")

    ckpt_ddpg = None
    ckpt_sac = None
    buffer_ddpg = None
    buffer_sac = None
    for idx in range(iter):
        al, cl, rwd, profit, ckpt_ddpg, buffer_ddpg = train_eval(DDPG, dd_env, ckpt_ddpg, buffer_ddpg, sum_ddpg, idx)
        al, cl, rwd, profit, ckpt_sac, buffer_sac = train_eval(SAC, sac_env, ckpt_sac, buffer_sac, sum_sac, idx)



if __name__ == '__main__':
    register(
        id=env_name,
        entry_point='sim.env:StarTradingEnv',
    )

    compair_run(1000)

