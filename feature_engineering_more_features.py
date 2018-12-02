#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: meihuaren
"""

import numpy as np
import pandas as pd

#-- import data
orderbook_cols = ['{}_{}_{}'.format(s,t,l) for l in range(1,6) for s in ['ask','bid'] for t in ['price', 'vol'] ]
orderbook_ori = pd.read_csv('INTC_2012-06-21_34200000_57600000_orderbook_5.csv', \
                            header = None, names = orderbook_cols)
orderbook = orderbook_ori.copy()
orderbook['mid_price'] = (orderbook.iloc[:,0] + orderbook.iloc[:,2]) / 2
orderbook['mid_price_mov'] = np.sign(orderbook['mid_price'].shift(-1)-orderbook['mid_price']) # the last one is nan

message_cols = ['time', 'type', 'order ID', 'size', 'price', 'direction']
message = pd.read_csv('INTC_2012-06-21_34200000_57600000_message_5.csv', \
                          header = None, names = message_cols)

total_data = pd.concat([message, orderbook], axis = 1)

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

#-- new feature 2: liquidity imbalance
'''
liquidity imbalance at level i = ask_vol_i / (ask_vol_i + bid_vol_i)
This feature is constructed according to the ppt.
'''
for i in range(1,6):
    total_data['liq_imb_'+str(i)] = total_data['ask_vol_'+str(i)] \
                                  / (total_data['ask_vol_'+str(i)] + total_data['bid_vol_'+str(i)])

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

#-- new feature 4: actual market imbalance (could try different lagged periods)
'''
actual market imbalance = the volume of market buy orders arriving in the prior 50 observations
                        - the volume of market sell orders arriving in the prior 50 observations
This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
'''
total_data['actual_mkt_imb'] = total_data['buy_vol'].rolling(50, min_periods = 1).sum()\
                             - total_data['sell_vol'].rolling(50, min_periods = 1).sum()

#-- new feature 5: relative market imbalance (could try different lagged periods)
'''
relative market imbalance = actual market imbalance / actual spread
This feature is derived from paper: Michael Kearns..._Machine Learning for Market Microstructure...P8
Intuition: a small actual spread combined with a strongly positive actual market imbalance
           would indicate buying pressure.
'''
total_data['relative_mkt_imb'] = total_data['actual_mkt_imb'] / total_data['actual_spread']

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

#-- new feature 7: relative spread
'''
relative spread =  (actual spread / mid price) * 10000
This feature is derived from paper: Angelo Ranaldo..._Order aggressiveness in limit order book markets...P4
'''
total_data['relative_spread'] = (total_data['actual_spread'] / total_data['mid_price']) * 1000

"""
#-- new feature 8: volatility (could try different lagged periods)
'''
The volatility is the standard deviation of the last 50 midquote returns then divided by 100
This feature is derived from paper: Angelo Ranaldo..._Order aggressiveness in limit order book markets...P4
'''
total_data['mid_price_return'] = total_data['mid_price'].shift(-1) - total_data['mid_price']
total_data['volatility'] = (total_data['mid_price_return'].rolling(50, min_periods = 1).std()) / 100
"""

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


#-- export total_data
#new_features_resultpath = '/Users/meihuaren/personal/OR_2018fall/Courses/E4720 Deep Learning/project_coding/Team E_code/'
new_features_resultpath = 'F:/Columbia OR/IEORE4720 Deep Learning/Course Project/Meihua Ren/'
filename = new_features_resultpath + 'total_data.csv'
total_data.to_csv(filename)

# select features

data = total_data.loc[:,['mid_price_mov','ask_price_1','ask_vol_1','bid_price_1','bid_vol_1','liq_imb_1','liq_imb_2'\
                         ,'mid_price_inv_vol_weighted','relative_mid_price_trend','relative_spread','lastest_trade_price','effective_spread']]
    
data = data.dropna()

data.rename(columns = {'mid_price_mov':'label'}, inplace = True)

data_0_1 = (data - np.min(data,axis=0)) / (np.max(data,axis=0) - np.min(data,axis=0))

data_0_1.loc[:,'label'] = data.loc[:,'label'].astype(int)

data_0_1.to_csv('F:/Columbia OR/IEORE4720 Deep Learning/Course Project/Data/INTC_0_1_Megan.csv',index=False)
