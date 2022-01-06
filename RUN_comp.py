
from gym import register
from runner import IterRun
from stable_baselines3 import DDPG, SAC,TD3

# https://github.com/notadamking/RLTrader/issues/10



def compair_run(iter):
    targets =[IterRun(TD3), IterRun(DDPG), IterRun(SAC)]

    for iter_run in targets:
        iter_run.init_boost()

    for i in range(iter):
        for iter_run in targets:
            iter_run.train_eval()

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

