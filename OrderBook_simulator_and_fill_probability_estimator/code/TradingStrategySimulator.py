# coding: utf-8

# In[2]:
#Mosie Schrem
import pandas as pd
import numpy as np
import os
import math
import sys

import OrderUtil as ou

# coding: utf-8
class TradingStrategyBacktester: 
    '''Class to backtest our trading strategy'''
    def __init__(self, book, strategy, predictions, midprice_df, mkt_order_on_move_predictions=False,
                 set_edge_queue=None, relative_queue=None, tick_size=None, queue_tol=0.8,
                 max_exposure_dict={0:0, 1:0, -1:0}):
        '''
        Initialize backtester, strategy is a dict of the form: {y_hat: ([bids_list], [asks_list])}
        for each yhat value. 
        '''
        self._active = 0
        self._tick_size = tick_size
        self._relative_queue=relative_queue
        self._bid_rel_queue =1.0
        self._ask_rel_queue = 1.0
        self._set_edge_queue = set_edge_queue
        self._pnl_ser = pd.Series()
        self._orderbook = book
        self._midprice_df = midprice_df.copy()
        self._midprice_df.loc[:, 'y_predict'] = predictions.values
        for col in self._midprice_df.columns:
            self._midprice_df[col] = self._midprice_df[col].astype(int)
        self._strategy = strategy
        self._tstart = self._midprice_df.index.min()
        self._tend = self._midprice_df.index.max()
        self._bid_orders = {}
        self._ask_orders = {}
        self._mov_ind = self._midprice_df.columns.get_loc('movement')
        self._bp1_ind = self._midprice_df.columns.get_loc('bp1')
        self._ap1_ind = self._midprice_df.columns.get_loc('ap1')
        self._bq1_ind = self._midprice_df.columns.get_loc('bq1')
        self._aq1_ind = self._midprice_df.columns.get_loc('aq1')
        self._predict_ind = self._midprice_df.columns.get_loc('y_predict')
        self._index_ind = self._midprice_df.columns.get_loc('index_position')
        self._mid_ind = self._midprice_df.columns.get_loc('midprice')
        self._y0_ind = self._midprice_df.columns.get_loc('y_0')
        #set exposure levels
        self._max_exposure = max_exposure_dict
        self._long_position = 0.0
        self._short_position = 0.0
        self._send_mkt_order = mkt_order_on_move_predictions
        print(sum(self._midprice_df['y_predict'] == self._midprice_df['y_1'])/len(self._midprice_df))
        self._ask_spread_signal = 0
        self._bid_spread_signal = 0
        self._queue_tol = queue_tol
        
    def _place_orders(self, ind, y_pred):
        
        current_time = self._midprice_df.index[ind]
        index_reference = self._midprice_df.values[ind, self._index_ind]
        active_strategy = self._strategy[y_pred[0]]
        
        bids_to_add = active_strategy[0]
        asks_to_add = active_strategy[1]
        
        #check if there exist a previously placed order (from last update) on levels we want to place orders!
        #as these orders will be ahead of new orders in queue
        
        if not y_pred[-1] is None:
            keep_bids = {}
            keep_asks = {}
            for bid in bids_to_add:
                keys = list(self._bid_orders.keys())
                for key in reversed(keys):
                    if self._bid_orders[key].order_type("open") and self._bid_orders[key].get_current_level() == bid:
                        keep_bids[bid] = self._bid_orders[key]
                        
            for ask in asks_to_add:
                keys = list(self._ask_orders.keys())
                for key in reversed(keys):
                    if self._ask_orders[key].order_type("open") and self._ask_orders[key].get_current_level() == ask:
                        keep_asks[ask] = self._ask_orders[key]
                        
            self._bid_orders = keep_bids
            self._ask_orders = keep_asks
                        
                  
        for bid in bids_to_add:
            if bid in self._bid_orders:
                continue
            self._bid_orders[bid] = ou.IndexTrackedOrder(orderbook=self._orderbook, level=bid,
                                                         index_ser=self._midprice_df['index_position'],
                                                         ind_start=ind,
                                                         is_buy=True, relative_queue=self._relative_queue)
            if bid == self._orderbook.num_levels() and not self._set_edge_queue is None:
                self._bid_orders[bid].set_queue_position(round(self._set_edge_queue*
                                                               self._bid_orders[bid].get_queue_position()))
                
        for ask in asks_to_add:
            if ask in self._ask_orders:
                continue
            self._ask_orders[ask] = ou.IndexTrackedOrder(orderbook=self._orderbook, level=ask,
                                                         index_ser=self._midprice_df['index_position'],
                                                         ind_start=ind,
                                                         is_buy=False, relative_queue=self._relative_queue)
            if ask == self._orderbook.num_levels() and not self._set_edge_queue is None:
                self._ask_orders[ask].set_queue_position(round(self._set_edge_queue*
                                                               self._ask_orders[ask].get_queue_position()))
                
                
        self._set_relative_queues(2, 2)
                                                         
    def _update_orders(self):
        #now we update our orders to next step
        for bid in self._bid_orders:
            self._bid_orders[bid].update()
        for ask in self._ask_orders:
            self._ask_orders[ask].update()
            
    def _set_relative_queues(self, level_ask, level_bid):
        #check queue positions of level 2s
        if level_bid in self._bid_orders and self._send_mkt_order:
            self._bid_rel_queue = self._bid_orders[level_bid].get_relative_queue()
        if level_ask in self._ask_orders and self._send_mkt_order:
            self._ask_rel_queue = self._ask_orders[level_ask].get_relative_queue()
        
            
    def _handle_market_orders(self, ind, y_pred, y_actual):
        if not self._send_mkt_order:
            return 0.0
        
        mkt_prev_bid_ask = (self._midprice_df.values[ind, self._bp1_ind],
                             self._midprice_df.values[ind, self._ap1_ind])
        spread = mkt_prev_bid_ask[1] - mkt_prev_bid_ask[0]
        self._mkt_spread = False
        
        if spread <= self._tick_size*1.01:
            self._mkt_spread = True
            
        cash_flow = 0.0
        
        #when midprice movement occurs, we close out our position
        if self._active == 1 and self._midprice_df.values[ind + 1, self._mov_ind] != 0:
            self._active = 0
            self._close = True
            self._bid_spread_signal = 0
            self._ask_spread_signal = 0
                
        if self._active == 0 and self._mkt_spread and not self._close:
            
            if self._bid_rel_queue <= self._queue_tol*self._relative_queue:
                self._bid_spread_signal += y_pred[0]
            else:
                self._bid_spread_signal = 0
            if self._ask_rel_queue <= self._queue_tol*self._relative_queue:
                self._ask_spread_signal += y_pred[0]
            else:
                self._ask_spread_signal = 0
                
            if self._short_position + self._long_position <= self._max_exposure[y_pred[0]]:
                if self._ask_spread_signal > 0 and self._ask_rel_queue <= self._queue_tol*self._relative_queue:
                    self._active = 1
                    self._long_position += 1
                    ref_long_price = mkt_prev_bid_ask[1]
                    print("market buy at " + str(ref_long_price))
                    cash_flow -= ref_long_price
                    self._bid_spread_signal = 0
                    
                if self._bid_spread_signal < 0 and self._bid_rel_queue <= self._queue_tol*self._relative_queue:

                    self._active = 1
                    self._short_position += 1
                    ref_short_price = mkt_prev_bid_ask[0]
                    
                    print("market sell at " + str(ref_short_price))
                    
                    self._ask_spread_signal = 0
                    cash_flow += ref_short_price

        return cash_flow
        
    def _execute_orders(self, ind, y_pred, y_actual):
        cash_flow = 0.0
        #now we see which orders are filled and adjust for profit
        for bid in self._bid_orders:
            if self._bid_orders[bid].order_type("executed"):
                price = self._bid_orders[bid].get_order_price()
                cash_flow -= price
                print('limit bid executed at ' + str(price))
               
                self._long_position += 1
        
        for ask in self._ask_orders:
            if self._ask_orders[ask].order_type("executed"):
                price = self._ask_orders[ask].get_order_price()
                cash_flow += price
                print('limit ask executed at ' + str(price))
             
                self._short_position += 1
                
        return cash_flow
        
    def _close_positions(self, ind, y_pred, y_actual):
        cash_flow = 0.0
        mkt_bid_ask = (self._midprice_df.values[ind + 1, self._bp1_ind],
                   self._midprice_df.values[ind + 1, self._ap1_ind])
                    
        #net long
        if self._long_position > self._short_position:
            #close out if we hit our max market exposure level
            if self._long_position > self._max_exposure[y_pred[1]] + self._active or self._close:
                cash_flow += mkt_bid_ask[0]*(self._long_position - self._short_position)
                
                self._long_position = 0.0
                self._short_position = 0.0
                self._active = 0
                self._close = False
                self._bid_spread_signal = 0
                self._ask_spread_signal = 0
            
        #net short 
        elif self._short_position > self._long_position:
            #close out if we hit our max market exposure level
            if self._short_position > self._max_exposure[y_pred[1]] + self._active or self._close:
                
                cash_flow -= mkt_bid_ask[1]*(self._short_position - self._long_position)
                self._long_position = 0.0
                self._short_position = 0.0
                self._active = 0
                self._close = False
                self._bid_spread_signal = 0
                self._ask_spread_signal = 0
                
        #close as many positions as possible
        while self._long_position >= 1 and self._short_position >=1:
            self._long_position -= 1
            self._short_position -= 1
            self._active = 0
            self._close = False
            self._bid_spread_signal = 0
            self._ask_spread_signal = 0
            
        return cash_flow
    
    
    def _eval_orders(self, ind, y_pred, y_actual):
        cash_flow = 0.0
        cash_flow += self._handle_market_orders(ind, y_pred, y_actual)
        cash_flow += self._execute_orders(ind, y_pred, y_actual)
        cash_flow += self._close_positions(ind, y_pred, y_actual)
        
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
        y_actual = {}
        y_actual[-1] = None
        y_actual[0] = self._midprice_df.values[start_ind, self._y0_ind]
        y_pred = {}
        y_pred[-1] = None
        y_pred[0] = self._midprice_df.values[start_ind, self._predict_ind]
        self._close = False
        cum_pnl = 0
        for i in range(start_ind, end_ind - 1):
            if i == end_ind - 2:
                self._close = True
            y_pred[1] = self._midprice_df.values[i + 1, self._predict_ind]
            y_actual[1] = self._midprice_df.values[i + 1, self._y0_ind]
            
            #place update and evaluate orders
            self._place_orders(i, y_pred)
            self._update_orders()
            pnl = self._eval_orders(i, y_pred, y_actual)
            
            cum_pnl += pnl
            time = self._midprice_df.index[i + 1]
            
            midprice = self._midprice_df.values[i + 1, self._mid_ind]
            adjust = self._long_position*midprice - self._short_position*midprice
            self._pnl_ser.loc[time] = cum_pnl + adjust
            
            y_pred[-1] = y_pred[0]
            y_actual[-1] = y_actual[0]
            y_pred[0] = y_pred[1]
            y_actual[0] = y_actual[1]
            if (i - start_ind) % 1000 == 0:
                print('Current time:        ' + str(time))
                print('Current pnl: ' + str(self._pnl_ser.loc[time]))
                     
        self._pnl_ser = self._pnl_ser - self._pnl_ser.shift(1)
        self._pnl_ser.fillna(0.0, inplace=True)
            
    def get_cumulative_pnl_series(self):
        return self._pnl_ser.cumsum()
    
    def get_pnl_series(self):
        return self._pnl_ser

class StrategyLossEstimation():
    '''given a strategy we compute the loss matrix to be fed into our RNN'''
    def __init__(self, book, strategy, midprice_df, mkt_order_on_move_predictions=False,
                 set_edge_queue=False, relative_queue=None,tick_size=None,
                 max_exposure_dict={0:0, 1:0, -1:0}, accuracy_rate=None, queue_tol=0.8):
        #randomize our predictions
        simulator_help = {}
        for key in strategy:
            other_keys = [k for k in strategy if k != key]
            simulator_help[key] = other_keys
        
        self._midprice_movement = midprice_df['y_1'].copy()
        N = round(1000000/accuracy_rate)
        M = len(strategy.keys()) - 1
        random_generator = [np.random.randint(0, N) for i in range(len(midprice_df))]
        bernoulli_generator = [np.random.randint(0, M) for i in range(len(midprice_df))]
        self._rand_predictions = pd.Series([self._midprice_movement.values[i] if random_generator[i] <= 1000000 
                                            else simulator_help[self._midprice_movement.values[i]][bernoulli_generator[i]]
                                            for i in range(len(midprice_df))], index=midprice_df.index)
        
                                           
        #build backtester based on random predictions
        self._strategy_simulator = TradingStrategyBacktester(book=book, strategy=strategy,
                                                             predictions=self._rand_predictions,
                                                             midprice_df=midprice_df,
                                                             mkt_order_on_move_predictions=mkt_order_on_move_predictions,
                                                             set_edge_queue=set_edge_queue,
                                                             relative_queue=relative_queue,
                                                             max_exposure_dict=max_exposure_dict,
                                                             tick_size=tick_size, queue_tol=queue_tol)
        #rand predict and midprice movement have same index
        self._midprice_movement = self._strategy_simulator._midprice_df['y_1'].copy()
        self._rand_predictions = self._strategy_simulator._midprice_df['y_predict'].copy()
        
        self._loss_matrix = pd.DataFrame()
        
        N = len(self._midprice_movement) - 2
        self._actual_indices = {}
        for key in strategy:
            self._actual_indices[key] = [ind for ind in range(N)
                                         if self._midprice_movement.values[ind] == key]
        
        self._num_per_sample = {key: len(self._actual_indices[key]) for key in self._actual_indices}
        
        for key in self._num_per_sample:
            if self._num_per_sample[key] <= 0:
                raise Exception("No movements of type: " + str(key))
        self._pnls = pd.Series()
        self._strategy_simulator.run_strategy()
        self._strategy = strategy
        
    def get_pnl_series(self):
        return self._strategy_simulator.get_pnl_series()