#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: meihuaren
"""

import numpy as np
import pandas as pd

#-- import data
orderbook_cols = ['{}_{}_{}'.format(s,t,l) for l in range(1,6) for s in ['ask','bid'] for t in ['price', 'vol'] ]
orderbook_cols.insert(0,'Time')
orderbook_cols.append('label')
orderbook_ori = pd.read_csv('INTC_orderbook_1s_horizon.csv', \
                            header = None, names = orderbook_cols)
orderbook_ori.drop(columns = 'Time', inplace = True)
orderbook_ori.drop(index = 0, inplace = True)
orderbook_ori.reset_index(inplace = True)
orderbook_ori.drop(columns = 'index', inplace = True)

message_cols = ['time', 'type', 'order ID', 'size', 'price', 'direction']
message = pd.read_csv('INTC_2012-06-21_34200000_57600000_message_5.csv', \
                          header = None, names = message_cols)

total_data = pd.concat([message, orderbook_ori], axis = 1)
total_data['mid_price'] = (total_data.loc[:,'ask_price_1'] + total_data.loc[:,'bid_price_1']) / 2

all_features = pd.concat([message.time, orderbook_ori.iloc[:,:-1]], axis = 1)

#-- new feature 1: order flow (could try different lagged periods)
'''
order flow = the ratio of the volume of market buy(sell) orders arriving in the prior 50 observations 
             to the resting volume of ask(bid) limit orders at the top of book
This feature is constructed according to the paper.
# Since we do not have "number of order" data, here just use "volume of order" instead.
Intuition: an increase in this ratio will more likely deplete the best ask level and the mid-price will up-tick,
           and vice-versa for a down-tick.
'''
total_data['type_direction'] = total_data['type'] * total_data['direction']
total_data['buy_vol'] = 0
buy_order_index1 = total_data[total_data['type_direction'] == -4].index
total_data.loc[buy_order_index1, 'buy_vol'] = total_data['size']
# Not include type=5(Execution of a hidden limit order), since its update does not change the limit order book.
#buy_order_index2 = total_data[total_data['type_direction'] == -5].index
#total_data.loc[buy_order_index2, 'buy_vol'] = total_data['size']
total_data['sell_vol'] = 0
sell_order_index1 = total_data[total_data['type_direction'] == 4].index
total_data.loc[sell_order_index1, 'sell_vol'] = total_data['size']
#sell_order_index2 = total_data[total_data['type_direction'] == 5].index
#total_data.loc[sell_order_index2, 'sell_vol'] = total_data['size']

total_data['order_flow_buy'] = total_data['buy_vol'].rolling(50, min_periods = 1).sum() / total_data['ask_vol_1']
total_data['order_flow_sell'] = total_data['sell_vol'].rolling(50, min_periods = 1).sum() / total_data['bid_vol_1']

total_data['order_flow_buy'].fillna(method='ffill', inplace=True)
total_data['order_flow_sell'].fillna(method='ffill', inplace=True)
all_features['order_flow_buy'] = total_data['order_flow_buy']
all_features['order_flow_sell'] = total_data['order_flow_sell']

#-- new feature 2: liquidity imbalance
'''
liquidity imbalance at level i = ask_vol_i / (ask_vol_i + bid_vol_i)
This feature is constructed according to the ppt.
'''
for i in range(1,6):
    total_data['liq_imb_'+str(i)] = total_data['ask_vol_'+str(i)] \
                                  / (total_data['ask_vol_'+str(i)] + total_data['bid_vol_'+str(i)])
    
    total_data['liq_imb_'+str(i)].fillna(method='ffill', inplace=True)
    all_features['liq_imb_'+str(i)] = total_data['liq_imb_'+str(i)]

#-- new feature 3: actual spread
'''
actual spread = ask_price_1 - bid_price_1
This feature is constructed according to:
1. Michael Kearns..._Machine Learning for Market Microstructure...P7
2. Irene Aldridge_High-frequency trading...(2013) P190:
   First suggested by Bagehot (1971) and later developed by numerous researchers, the bid-ask spread 
   reflects the expectations of market movements by the market maker using asymmetric information.
'''
total_data['actual_spread'] = total_data['ask_price_1'] - total_data['bid_price_1']

total_data['actual_spread'].fillna(method='ffill', inplace=True)
all_features['actual_spread'] = total_data['actual_spread']

#-- new feature 4: actual market imbalance (could try different lagged periods)
'''
actual market imbalance = the volume of market buy orders arriving in the prior 50 observations
                        - the volume of market sell orders arriving in the prior 50 observations
This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
'''
total_data['actual_mkt_imb'] = total_data['buy_vol'].rolling(50, min_periods = 1).sum()\
                             - total_data['sell_vol'].rolling(50, min_periods = 1).sum()

total_data['actual_mkt_imb'].fillna(method='ffill', inplace=True)
all_features['actual_mkt_imb'] = total_data['actual_mkt_imb']

#-- new feature 5: relative market imbalance (could try different lagged periods)
'''
relative market imbalance = actual market imbalance / actual spread
This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
Intuition: a small actual spread combined with a strongly positive actual market imbalance
           would indicate buying pressure.
'''
total_data['relative_mkt_imb'] = total_data['actual_mkt_imb'] / total_data['actual_spread']

total_data['relative_mkt_imb'].fillna(method='ffill', inplace=True)
all_features['relative_mkt_imb'] = total_data['relative_mkt_imb']

#-- new feature 6: relative_mid_price_trend
'''
First, construct a variation on mid-price where the average of the bid and ask prices is weighted 
according to their inverse volume. Then, divide this variation by common mid price.
This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P10
Intuition: a larger relative_mid_price_trend would more likely lead to a up-tick.
'''

total_data['mid_price_inv_vol_weighted'] = (total_data['ask_price_1'] / total_data['ask_vol_1'] \
                                         + total_data['bid_price_1'] / total_data['bid_vol_1'])\
                                         / (1 / total_data['ask_vol_1'] + 1 / total_data['bid_vol_1'])
total_data['relative_mid_price_trend'] = total_data['mid_price_inv_vol_weighted'] / total_data['mid_price']

total_data['relative_mid_price_trend'].fillna(method='ffill', inplace=True)
all_features['relative_mid_price_trend'] = total_data['relative_mid_price_trend']

#-- new feature 7: relative spread
'''
relative spread =  (actual spread / mid price) * 10000
This feature is derived from paper: Angelo Ranaldo..._Order aggressiveness in limit order book markets...P4
'''
total_data['relative_spread'] = (total_data['actual_spread'] / total_data['mid_price']) * 1000

total_data['relative_spread'].fillna(method='ffill', inplace=True)
all_features['relative_spread'] = total_data['relative_spread']

#-- new feature 8: volatility (could try different lagged periods)
'''
The volatility is the standard deviation of the last 50 midquote returns then divided by 100
This feature is derived from paper: Angelo Ranaldo..._Order aggressiveness in limit order book markets...P4
'''
total_data['mid_price_return'] = total_data['mid_price'].shift(-1) - total_data['mid_price']
total_data['volatility_look_ahead'] = (total_data['mid_price_return'].rolling(50, min_periods = 1).std()) / 100
total_data['volatility'] = total_data['volatility_look_ahead'].shift(1)

total_data['volatility'].fillna(method='ffill', inplace=True)
all_features['volatility'] = total_data['volatility']

#-- new feature 9: limit order aggressiveness (could try different lagged periods)
'''
bid(ask) limit order aggressiveness = the ratio of bid(ask) limit orders submitted at no lower(higher) than 
                                                   the best bid(ask) prices in the prior 50 observations
                                                to total bid(ask) limit orders submitted in prior 50 observations
This feature is derived from book: Irene Aldridge_High-frequency trading...(2013) P186
Intuition: The higher the ratio, the more aggressive is the trader in his bid(ask) to capture the best 
           available price and the more likely the trader is to believe that the price is about to 
           move away from the mid price.
'''
# ask limit order aggressiveness
if_ask_sbmt_agr_mid1 = (total_data['type_direction'] == -1)
if_ask_sbmt_agr_mid2 = (total_data['price'] <= total_data['ask_price_1'].shift(1))
if_ask_sbmt_agr = (if_ask_sbmt_agr_mid1 & if_ask_sbmt_agr_mid2)

total_data['if_ask_sbmt_agr'] = if_ask_sbmt_agr
if_ask_sbmt_agr_index = total_data[total_data['if_ask_sbmt_agr'] == True].index
total_data['ask_vol_sbmt_agr'] = 0
total_data.loc[if_ask_sbmt_agr_index, 'ask_vol_sbmt_agr'] = total_data['size']

if_ask_sbmt_index = total_data[total_data['type_direction'] == -1].index
total_data['ask_vol_sbmt'] = 0
total_data.loc[if_ask_sbmt_index, 'ask_vol_sbmt'] = total_data['size']

total_data['lo_agr_ask'] = total_data['ask_vol_sbmt_agr'].rolling(50, min_periods = 1).sum()\
                         / total_data['ask_vol_sbmt'].rolling(50, min_periods = 1).sum()

total_data['lo_agr_ask'].fillna(method='ffill', inplace=True)
all_features['lo_agr_ask'] = total_data['lo_agr_ask']

# bid limit order aggressiveness
if_bid_sbmt_agr_mid1 = (total_data['type_direction'] == 1)
if_bid_sbmt_agr_mid2 = (total_data['price'] >= total_data['bid_price_1'].shift(1))
if_bid_sbmt_agr = (if_bid_sbmt_agr_mid1 & if_bid_sbmt_agr_mid2)

total_data['if_bid_sbmt_agr'] = if_bid_sbmt_agr
if_bid_sbmt_agr_index = total_data[total_data['if_bid_sbmt_agr'] == True].index
total_data['bid_vol_sbmt_agr'] = 0
total_data.loc[if_bid_sbmt_agr_index, 'bid_vol_sbmt_agr'] = total_data['size']

if_bid_sbmt_index = total_data[total_data['type_direction'] == 1].index
total_data['bid_vol_sbmt'] = 0
total_data.loc[if_bid_sbmt_index, 'bid_vol_sbmt'] = total_data['size']

total_data['lo_agr_bid'] = total_data['bid_vol_sbmt_agr'].rolling(50, min_periods = 1).sum()\
                         / total_data['bid_vol_sbmt'].rolling(50, min_periods = 1).sum()

total_data['lo_agr_bid'].fillna(method='ffill', inplace=True)
all_features['lo_agr_bid'] = total_data['lo_agr_bid']

#-- new feature 10: effective spread
'''
The effective spread is computed as difference between the latest trade price and midprice 
                                    divided by midprice, then times 1000.
This feature is derived from book: Irene Aldridge_High-frequency trading...(2013) P191
Intuition: The effective spread measures how far, in percentage terms, the latest realized price 
           fell away from the simple mid price.
'''
if_lastest_trade_index = total_data[total_data['type'] == 4].index
if_not_lastest_trade_index = total_data[total_data['type'] != 4].index
total_data['lastest_trade_price'] = 0
total_data.loc[if_lastest_trade_index,'lastest_trade_price'] = total_data['price']
total_data.loc[if_not_lastest_trade_index,'lastest_trade_price'] = np.nan
total_data['lastest_trade_price'].fillna(method='ffill',inplace = True)

total_data['effective_spread'] = (total_data['lastest_trade_price'] / total_data['mid_price'] - 1) * 1000

total_data['effective_spread'].fillna(method='ffill', inplace=True)
all_features['effective_spread'] = total_data['effective_spread']

#-- new feature 11: ILLIQ
"""
The illiquidity is computed as the ratio of absolute stock return to its dollar volume.

This feature is derived from Amihud (2002)

"""

total_data['mid_price_ret'] = np.log(total_data['mid_price']) - np.log(total_data['mid_price'].shift(1))  
total_data['ret_over_volume'] = abs(total_data['mid_price_ret']) / (total_data['ask_vol_1'] + total_data['bid_vol_1'])
total_data['ILLIQ'] = total_data['ret_over_volume'].rolling(50, min_periods = 1).sum()

all_features['ILLIQ'] = total_data['ILLIQ']

#-- new feature 12: relative volume
"""
Relative volume is computed as the ratio of current volume to the historical average volume
"""

for i in range(1,6):
    
    total_data['rel_ask_vol_'+str(i)] = total_data['ask_vol_'+str(i)] / total_data['ask_vol_'+str(i)].rolling(50, min_periods = 1).mean()
    total_data['rel_bid_vol_'+str(i)] = total_data['bid_vol_'+str(i)] / total_data['bid_vol_'+str(i)].rolling(50, min_periods = 1).mean()
 
    all_features['rel_bid_vol_'+str(i)] = total_data['rel_bid_vol_'+str(i)]
    all_features['rel_ask_vol_'+str(i)] = total_data['rel_ask_vol_'+str(i)]

#-- new feature 13: volume depth
    
"""
Volume depth is computed as the ratio of best volume to the sum of all depth volume
"""
total_data['depth_ask_vol'] = total_data['ask_vol_1'] / (total_data['ask_vol_1'] + total_data['ask_vol_2'] + total_data['ask_vol_3']\
          + total_data['ask_vol_4'] + total_data['ask_vol_5'])
total_data['depth_bid_vol'] = total_data['bid_vol_1'] / (total_data['bid_vol_1'] + total_data['bid_vol_2'] + total_data['bid_vol_3']\
          + total_data['bid_vol_4'] + total_data['bid_vol_5'])
 
all_features['depth_ask_vol'] = total_data['depth_ask_vol']
all_features['depth_bid_vol'] = total_data['depth_bid_vol']


#-- new feature 14: volume rank
"""
volume rank is computed as the rank of current volume with respect to the previous 50days volume
"""

rollrank = lambda x: (x.argsort().argsort()[-1]+1.0)/len(x)

for i in range(1,6):
    
    total_data['rank_ask_vol_'+str(i)] = total_data['ask_vol_'+str(i)].rolling(50, min_periods = 1).apply(rollrank)
    total_data['rank_bid_vol_'+str(i)] = total_data['bid_vol_'+str(i)].rolling(50, min_periods = 1).apply(rollrank)
    
    total_data['rank_ask_vol_'+str(i)] = total_data['rank_ask_vol_'+str(i)].fillna(method='ffill',axis=0)
    total_data['rank_bid_vol_'+str(i)] = total_data['rank_bid_vol_'+str(i)].fillna(method='ffill',axis=0)
    total_data['rank_ask_vol_'+str(i)] = np.clip(total_data['rank_ask_vol_'+str(i)], 0, 1)
    total_data['rank_bid_vol_'+str(i)] = np.clip(total_data['rank_bid_vol_'+str(i)], 0, 1)    
 
    all_features['rank_bid_vol_'+str(i)] = total_data['rank_bid_vol_'+str(i)]
    all_features['rank_ask_vol_'+str(i)] = total_data['rank_ask_vol_'+str(i)]

#-- new feature 15: ask bid volume correlation
"""
ask bid volume correlation is comupted as 50 days time series correlation between ask and bid volume for each level
"""

for i in range(1,6):  
    total_data['corr_vol_'+str(i)] = total_data['ask_vol_'+str(i)].rolling(50, min_periods = 1).corr(total_data['bid_vol_'+str(i)])
    
    total_data['corr_vol_'+str(i)] = total_data['corr_vol_'+str(i)].fillna(method='ffill',axis=0)
    total_data['corr_vol_'+str(i)] = np.clip(total_data['corr_vol_'+str(i)], -1, 1)
    
    all_features['corr_vol_'+str(i)] = total_data['corr_vol_'+str(i)]

#--
all_features['label'] = total_data['label']

all_features = all_features.dropna()

all_features['label'] = all_features['label'].astype(int)

all_features.set_index('time', inplace = True)

#-- export data
new_features_resultpath = '/Users/meihuaren/personal/OR_2018fall/Courses/E4720 Deep Learning/project_coding/Team E_code/'
#filename1 = new_features_resultpath + 'total_data.csv'
#total_data.to_csv(filename1)

filename2 = new_features_resultpath + 'all_features_new_intc_1s_fillna_time.csv'
all_features.to_csv(filename2,index=True)
