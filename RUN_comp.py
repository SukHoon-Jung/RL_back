
import gym
from gym import register

from runner.runner import IterRun
from stable_baselines3 import DDPG, SAC,TD3

# https://github.com/notadamking/RLTrader/issues/10



def compair_run(iter):
    targets =[IterRun(TD3)]


    print("======================================")
    print ("======================================")
    print ("======================================")
    print ("======================================")
    print ("======================================")

    for i in range(iter):
        for iter_run in targets:
            iter_run.train_eval()

# env_name ='StarTrader-v0'
# entry = 'sim.env:StarTradingEnv'
env_name ='Stacked-v0'
# entry = 'sim.env:StackedEnv'

if __name__ == '__main__':
    # env_nameee = 'Stacked-v0'
    # entryee = 'sim.env:StackedEnv'
    # register (
    #     id=env_nameee,
    #     entry_point=entryee,
    # )

    compair_run(1000)

