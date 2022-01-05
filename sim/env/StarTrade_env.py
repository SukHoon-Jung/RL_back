# --------------------------- IMPORT LIBRARIES -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gym.utils import seeding
import gym
from gym import spaces
import sim.data_preprocessing as dp
import math

# ------------------------- GLOBAL PARAMETERS -------------------------
# Start and end period of historical data in question
from sim.epi_plot import EpisodePlot

START_TRAIN = datetime(2008, 12, 31)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)

STARTING_ACC_BALANCE = 0
MAX_TRADE = 2


DJI = dp.DJI
DJI_N = dp.DJI_N
CONTEXT_DATA = dp.CONTEXT_DATA
CONTEXT_DATA_N = dp.CONTEXT_DATA_N


NUMBER_OF_STOCKS = len(DJI)

# # Pools of stocks to trade
# DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
#        'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT']
#
# DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney'
#          ,'ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
#          'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble',
#          'United Technologies','UnitedHealth Group','Verizon Communications','Wal Mart']
#
# #Market and macroeconomic data to be used as context data
# CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']




# # # DJIA component stocks
# DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS',
#           'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX',
#           'UNH', 'VZ', 'WMT']
# DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney'
#          ,'ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
#          'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble',
#          'United Technologies','UnitedHealth Group','Verizon Communications','Wal Mart']
#
# CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']
#
# CONTEXT_DATA_N = ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ Composite', 'Russell 2000', 'SPDR S&P 500 ETF',
#  'Invesco QQQ Trust', 'CBOE Volatility Index', 'SPDR Gold Shares', 'Treasury Yield 30 Years',
#  'CBOE Interest Rate 10 Year T Note', 'iShares 1-3 Year Treasury Bond ETF', 'iShares Short Treasury Bond ETF']




# # DJIA component stocks




# ------------------------------ PREPROCESSING ---------------------------------
print ("\n")
print ("############################## Welcome to the playground of Star Trader!!   ###################################")
print ("\n")
print ("Hello, I am Star, I am learning to trade like a human. In this playground, I trade stocks and optimize my portfolio.")
print ("\n")

print ("Starting to pre-process data for trading environment construction ... ")
# Data Preprocessing


PRICE_FILE = './data/ddpg_WORLD.csv'
INPUT_FILE = './data/ddpg_input_states.csv'


try:
    input_states = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
    WORLD =  pd.read_csv(PRICE_FILE, index_col='Date', parse_dates=True)
    print("LOAD PRE_PROCESSED DATA")
    print(WORLD.head(10))
except :
    print ("LOAD FAIL.  PRE_PROCESSing DATA")
    dataset = dp.DataRetrieval()
    input_states = dataset.get_feature_dataframe (DJI)

    if len(CONTEXT_DATA):
        context_df = dataset.get_feature_dataframe (CONTEXT_DATA)
        input_states = pd.concat([context_df, input_states], axis=1)
    input_states = input_states.dropna()
    input_states.to_csv(INPUT_FILE)
    WORLD = dataset.components_df_o[DJI]
    WORLD.to_csv(PRICE_FILE)


# Without context data
#input_states = feature_df
feature_length = len(input_states.columns)
data_length = len(input_states)

COMMITION = 0.2
SLIPPAGE = 1#1  # 상방 하방
COST = SLIPPAGE+COMMITION


print("Pre-processing and stock selection complete, trading starts now ...")
print("_______________________________________________________________________________________________________________")


# ------------------------------ CLASSES ---------------------------------
obs_range=(-5., 5.)
class StarTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day = START_TRAIN, title="Star", plot_dir=None):
        self.plot_fig = None if not plot_dir else EpisodePlot(title, plot_dir)
        if NUMBER_OF_STOCKS !=1:
            raise Exception("NEED SINGLE TARGET")
        """
        Initializing the trading environment, trading parameters starting values are defined.
        """
        self.iteration = 0
        self.day = day

        # defined using Gym's Box action space function
        self.action_space = spaces.Box(low = -1.0, high = 1.0,shape = (NUMBER_OF_STOCKS,),dtype=np.float16)

        # [account balance]+[unrealized profit/loss] +[number of features, 36]+[portfolio stock of 5 stocks holdings]
        self.full_feature_length = 2 + feature_length
        self.share_idx = self.full_feature_length
        print("full length", self.full_feature_length)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.full_feature_length + NUMBER_OF_STOCKS,))
        self.reset()
    def reset(self):
        """
        Reset the environment once an episode end.

        """

        self.done = False
        self.up_cnt = 0
        self.down_cnt =0
        self.total_neg = []
        self.total_pos = []
        self.total_commition = 0
        self.unit_log =[0]
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance
        self.reward_log = [0]
        self.position = 0
        self.position_log = [self.position ]


        unrealized_pnl = 0.0
        self.unrealized_asset = [unrealized_pnl]

        self.buy_price = 0
        self.day = START_TRAIN

        self.day, self.data = self.skip_day(input_states, True)
        self.timeline = [self.day]
        self.state = self.acc_balance + [unrealized_pnl] + self.data.values.tolist() + [0]

        self.iteration += 1
        self.reward = 0


        return self.state


    def skip_day(self, input_st, first=False):
        # Sliding the timeline window day-by-day, skipping the non-trading day as data is not available
        if not first : self.day += timedelta (days=1)
        wrong_day = True
        add_day = 0
        while wrong_day:
            try:
                temp_date = self.day + timedelta(days=add_day)
                self_data = input_st.loc[temp_date]
                self_day = temp_date
                wrong_day = False
            except:
                add_day += 1
        return self_day, self_data


    def get_trade_num(self, normed_action, max_trade):
        action = normed_action + 1
        action = action * ( float(2*max_trade +1)/ 2.0) - max_trade
        action = math.floor(action)
        return min(action, max_trade)

    def _clean(self, cur_share, new_share):

        shift = new_share - cur_share
        direction = -1 if shift < 0 else 1 if shift > 0 else 0
        clean_all = False

        if cur_share == 0 or shift == 0:
            return direction, 0, shift, clean_all
        if (cur_share < 0 and shift < 0) or (cur_share > 0 and shift > 0):
            return direction, 0, shift, clean_all
        clean_all = (abs (cur_share) <= abs (shift))

        cleaned = -cur_share if clean_all else shift
        left = shift - cleaned

        return direction, cleaned, left, clean_all

    def getprice(self, date=None):
        if date is None: date = self.day
        return WORLD.loc[date][0]


    def __trade(self, pre_price, cur_share, new_share, date=None):
        # print("P ", self.getprice(date))
        buy_direction, cleaned, left_buy, cleaned_all = self._clean (cur_share, new_share)
        if buy_direction == 0:
            return 0, 0, 0, 0, pre_price

        assert cleaned + left_buy == (new_share - cur_share)
        cost = abs (cleaned + left_buy) * COMMITION

        transacted_price = self.getprice(date) + (SLIPPAGE * buy_direction)  # 살땐 비싸게, 팔땐 싸게

        if cleaned == 0:  # clean은 하지 않았으므로, 같은 방향의 변동
            assert cur_share + left_buy == new_share
            buy_price = (abs (cur_share) * pre_price + abs (left_buy) * transacted_price) / abs (new_share)
            realized = 0

        else:
            realized = -cleaned * (transacted_price - pre_price)
            if not cleaned_all:
                buy_price = pre_price # self.buy_price[idx] 안바뀜, 일부청산 이기 때문
            else:  # 모두 청산하여 예전 가격 필요없음. 더 거래시 현재 가격
                buy_price = transacted_price
        # print("T ", cleaned, realized, cost, left_buy, buy_price)
        return cleaned, realized, cost, left_buy, buy_price


    def _trade(self, action):

        cur_share = self.state[self.share_idx]
        new_share = self.get_trade_num(action ,MAX_TRADE)

        cleaned, profit, cost, _, buy_price = self.__trade(self.buy_price, cur_share, new_share)

        # print(">>>>>>>>>>>>>>",cur_share, new_share, profit-cost )

        cleaned = abs(cleaned)
        thresold = abs (cleaned) * COMMITION
        if profit > thresold:
            self.up_cnt += abs (cleaned)
        elif profit < thresold:
            self.down_cnt += abs (cleaned)
        else:
            pass

        self.buy_price = buy_price
        self.state[0] += profit - cost

        self.state[self.share_idx] = new_share
        return new_share

    def log_trade (self):
        trading_book = pd.DataFrame (index=self.timeline, columns=["Cash balance", "Unrealized value", "Total asset", "Rewards", "CumReward", "Position"])
        trading_book["Cash balance"] = self.acc_balance
        trading_book["Unrealized value"] = self.unrealized_asset
        trading_book["Total asset"] = self.total_asset
        trading_book["Rewards"] = self.reward_log
        trading_book["CumReward"] = trading_book["Rewards"].cumsum().fillna(0)
        trading_book["Position"]  = self.position_log
        trading_book["Unit"] = self.unit_log

        trading_book.to_csv ('./train_result/trading_book_train_{}.csv'.format (self.iteration - 1))

    def step_done(self, actions):
        self.step_normal(0) #clear_all

        print ("@@@@@@@@@@@@@@@@@")
        print ("Iteration", self.iteration - 1)

        total_neg = np.sum(self.total_neg)
        print("UP: {}, DOWN: {}, Commition: {}".format(self.up_cnt, self.down_cnt, self.total_commition))
        print("Acc: {}, Rwd: {}, Neg: {}".format(self.total_asset[-1], sum(self.reward_log),total_neg))

        risk_log = -1 * total_neg/ np.sum(self.total_pos)
        self.render()
        return self.state, self.reward, self.done, {'profit': self.total_asset[-1], 'risk':risk_log, 'neg': total_neg}

    def clear_all(self):
        self._trade(0)

    def step(self, actions):
        self.done = self.day >= END_TRAIN
        if self.done:
            return self.step_done(actions[0])
        else:
            return self.step_normal(actions[0])


    def _unrealized_profit(self, cur_buy_stat, buy_price, at=None):
        transaction_size = np.sum(abs(cur_buy_stat))
        if transaction_size ==0 : return 0
        now = (self.getprice(at) - buy_price) * cur_buy_stat
        now = now - (SLIPPAGE+COMMITION) * transaction_size
        return now


    def step_normal(self, action):

        pre_price = self.buy_price
        pre_date = self.day
        # Total asset is account balance + unrealized_pnl
        pre_unrealized_pnl = self.state[1]
        total_asset_starting = self.state[0] + pre_unrealized_pnl

        try:
            position = self._trade (action)
        except Exception as e:
            print(action)
            print (self.state)
            raise e

        self.position_log = np.append (self.position_log, position)
        #NEXT DAY
        self.day, self.data = self.skip_day (input_states)

        cur_buy_stat = self.state[self.share_idx]
        unrealized_pnl = self._unrealized_profit(cur_buy_stat, self.buy_price)

        self.state = [self.state[0]] + [unrealized_pnl] + self.data.values.tolist () + [cur_buy_stat]
        total_asset_ending = self.state[0] + unrealized_pnl
        step_profit = total_asset_ending - total_asset_starting

        # print(step_profit, unrealized_pnl)


        if step_profit <0: self.total_neg = np.append(self.total_neg, step_profit)
        else: self.total_pos = np.append(self.total_pos, step_profit)

        self.unit_log = np.append(self.unit_log, step_profit)
        self.acc_balance = np.append (self.acc_balance, self.state[0])
        self.unrealized_asset = np.append (self.unrealized_asset, unrealized_pnl)

        self.total_asset = np.append (self.total_asset, total_asset_ending)
        self.timeline = np.append (self.timeline, self.day)


        # self.reward = self.cal_reward(total_asset_starting, total_asset_ending, cur_buy_stat)
        self.reward = self.cal_opt_reward (pre_date, step_profit, pre_unrealized_pnl, pre_price, self.buy_price)
        # self.reward = self.cal_simple_reward(total_asset_starting, total_asset_ending)


        self.reward_log = np.append (self.reward_log, self.reward)
        return self.state, self.reward, self.done, {}

    def remain_risk(self, action_power):
        return 0.01 * (pow(action_power + 1, 3) -1)

    def get_optimal(self, base_date, base_share, base_unreal, base_price, next_price):
        check_trade = [-MAX_TRADE, 0, MAX_TRADE]
        optimal = self._unrealized_profit (base_share, base_price)

        for target in check_trade:
            if base_share == target: continue
            cleaned, profit, cost, _, buy_price = self.__trade(base_price, base_share, target, base_date)
            profit_sum = profit - cost
            unreal = self._unrealized_profit (target, next_price)
            profit_sum += unreal

            # print ("           >", base_share, target, profit_sum)
            # print(unreal - base_unreal )

            optimal = max(profit_sum, optimal)



        return optimal - base_unreal

    def cal_opt_reward (self, pre_date, profit, pre_unreal, pre_price, next_price):
        opt = self.get_optimal(pre_date, self.position_log[-2],pre_unreal, pre_price, next_price)
        reward = (profit - opt)  + 1
        # if (profit-0.001 > opt):
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(profit, "    ", opt, " >>>>", reward)
        return reward

    def cal_simple_reward(self, total_asset_starting, total_asset_ending):

        profit = (total_asset_ending - total_asset_starting)/ MAX_TRADE
        return profit


    def cal_reward(self, total_asset_starting, total_asset_ending, cur_buy_stat):

        action_power = np.mean(abs(cur_buy_stat/ MAX_TRADE))
        profit = (total_asset_ending - total_asset_starting)/ MAX_TRADE
        risk = self.remain_risk(action_power)
        profit = (profit - risk)

        if profit<0:
            profit = min(profit, -1 * pow(abs(profit), 2))
        else:
            profit = max(profit, pow(profit, 1.2))
        return profit+0.4

        return reward


    def render(self, mode='human'):
        if not self.plot_fig: return self.state

        self.plot_fig.update(iteration=self.iteration-1, idx=range(len(self.position_log)), pos=self.total_pos, neg= -self.total_neg,
                             cash=self.acc_balance, unreal=self.unrealized_asset, asset=self.total_asset,
                             reward=self.reward_log.cumsum(), position=self.position_log, unit=self.unit_log)
        return self.state

    def _seed(self, seed=None):
        """
        Seed the iteration.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
