
import gym
from gym import register

from runner.runner_dict import IterRun
from stable_baselines3 import DDPG, SAC,TD3

# https://github.com/notadamking/RLTrader/issues/10



def compair_run(iter):
    targets =[IterRun(DDPG), IterRun(SAC), IterRun(TD3)]


    print("======================================")
    print ("======================================")
    print ("======================================")
    print ("======================================")
    print ("======================================")

    for i in range(iter):
        for iter_run in targets:
            iter_run.train_eval()



if __name__ == '__main__':

    compair_run(1000)

