# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import os
import math
import OrderUtil as ou

class TradingStrategyBacktester: 
    '''Class to backtest our trading strategy'''
    def __init__(self, book, strategy, predictions, midprice_df, numupdates=None, timeunit=None):
        '''
        Initialize backtester, strategy is a dict of the form: {y_hat: ([bids_list], [asks_list])}
        for each yhat value. 
        '''
        self._pnl_ser = pd.Series()
        self._orderbook = book
        self._midprice_df = midprice_df
        self._midprice_df.loc[:, 'y_predict'] = predictions
        self._strategy = strategy
        self._tstart = self._midprice_df.index.min()
        self._tend = self._midprice_df.index.max()
        
        if(numupdates is None):
            self._timeupdates = True
            self._unit = timeunit
        else:
            self._timeupdates = False
            self._unit = numupdates
            
        self._y1_ind = self._midprice_df.columns.get_loc('y_1')
        self._bp1_ind = self._midprice_df.columns.get_loc('bp1')
        self._ap1_ind = self._midprice_df.columns.get_loc('ap1')
        self._ypred_ind = self._midprice_df.columns.get_loc('y_predict')
    
    def _place_and_eval_orders(self, y_pred, ind):
        if not self._strategy[y_pred]:
            return 0.0
        current_time = self._midprice_df.index[ind]
        active_strategy = self._strategy[y_pred]
        bids_list = active_strategy[0]
        asks_list = active_strategy[1]
        bid_orders = {}
        ask_orders = {}
        for bid in bids_list:
            if self._timeupdates:
                bid_orders[bid] = ou.TimeOrder(self._orderbook, timestamp=current_time, level=bid,
                                               is_buy=True, delta_t=self._unit)
                bid_orders[bid].update()
            else:
                bid_orders[bid] = ou.BookUpdatesOrder(self._orderbook, timestamp=current_time, level=bid,
                                                      is_buy=True, numupdates=self._unit)
                bid_orders[bid].update()
        for ask in asks_list:
            if self._timeupdates:
                ask_orders[ask] = ou.TimeOrder(self._orderbook, timestamp=current_time, level=ask,
                                               is_buy=False, delta_t=self._unit)
                ask_orders[ask].update()
            else:
                ask_orders[ask] = ou.BookUpdatesOrder(self._orderbook, timestamp=current_time, level=ask,
                                                      is_buy=False, numupdates=self._unit)
                ask_orders[ask].update()
                
        mkt_bid_ask = (self._midprice_df.values[ind + 1, self._bp1_ind],
                       self._midprice_df.values[ind + 1, self._ap1_ind])
        long_position = 0.0
        short_position = 0.0
        cash_flow = 0.0
        #now we see which orders are filled and adjust for profit
        for bid in bids_list:
            if bid_orders[bid].order_type("executed"):
                price = bid_orders[bid].get_order_price()
                cash_flow -= price
                long_position += 1
        for ask in asks_list:
            if ask_orders[ask].order_type("executed"):
                price = ask_orders[ask].get_order_price()
                cash_flow += price
                short_position += 1
                
        #net long
        if long_position > short_position:
            cash_flow += mkt_bid_ask[0]*(long_position - short_position)
            
        #net short 
        elif short_position > long_position:
            cash_flow -= mkt_bid_ask[1]*(short_position - long_position)
            
        return cash_flow
    
    def run_strategy(self, tstart=None, tend=None):
        if tstart is None:
            tstart = self._tstart
        if tend is None:
            tend = self._tend
            
        start_ind = self._midprice_df.index.get_loc(tstart)
        if isinstance(start_ind, slice):
            start_ind = start_ind.stop - 1
            
        end_ind = self._midprice_df.index.get_loc(tend)
        if isinstance(end_ind, slice):
            end_ind = end_ind.stop - 1
        
        self._pnl_ser = pd.Series()
        self._pnl_ser.index.name = 'timestamp'
        self._pnl_ser.loc[tstart] = 0.0
        for i in range(start_ind, end_ind):
            y_pred = self._midprice_df.values[i, self._ypred_ind]
            pnl = self._place_and_eval_orders(y_pred, i)
            time = self._midprice_df.index[i + 1]
            if (i - start_ind) % 1000 == 0:
                print('up to update number: ' + str(i - start_ind))
            self._pnl_ser.loc[self._midprice_df.index[i + 1]] = pnl
        
    
    def get_pnl_series(self):
        return self._pnl_ser
    
    def get_cumulative_pnl(self):
        return self._pnl_ser.cumsum()

