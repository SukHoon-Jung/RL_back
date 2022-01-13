
import gym
from gym import register

from runner.runner_dict import IterRun
from stable_baselines3 import DDPG, SAC,TD3
import numpy as np

# https://github.com/notadamking/RLTrader/issues/10



def compair_run(iter):
    noise_set = np.linspace(0.05, 0.4,5)
    nis_start = 1

    targets =[IterRun(TD3)]#, IterRun(DDPG), IterRun(SAC)]

    print("======================================")
    print ("======================================")
    print ("======================================")
    for i in range(1, iter):
        noise = None if i <nis_start else noise_set[(i-nis_start) % len(noise_set)]
        for iter_run in targets:
            iter_run.train_eval(noise=noise)



if __name__ == '__main__':

    compair_run(1000)

